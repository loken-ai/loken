// Attention over a quantised KV cache: the score pass and the output pass, specialised by
// head dimension and by how many queries share a launch.

// Both passes end the same way. A warp has finished a partial for one lane's slice - a column
// of the score row, or one component of the output vector - and the block's warps each hold a
// different partial for that same lane. The totals therefore run DOWN a column of a
// [warp][lane] staging area, not along a warp, so a shuffle cannot reach them and the sum goes
// through shared memory instead.
//
// Warp zero does the adding: every other warp would compute the same total and only one of
// them may store it. The two barriers are what make the staging area safe to reuse on the next
// component, so every thread of the block has to arrive at both - the caller's store must be
// guarded, never the barrier around it.

/// Add down one column of a `[warp][lane]` staging area of `n_warps` rows.
///
/// The thread's partial goes to its own slot; the value returned to warp zero is the total for
/// its lane, and every other warp gets a number it must not store. The caller supplies the
/// trailing barrier once it has finished reading the totals, so the area can be reused.
///
/// Thirty-two rows is the ceiling a 1024-thread block can reach, so the walk is written
/// against that bound and skips the rows the launch did not fill. Where the caller's warp
/// count is a constant the bound folds away and only the real rows are left.
static __device__ __forceinline__ float sum_down_warp_column(
    float * stage, int warp_id, int lane, int n_warps, float partial) {
    stage[warp_id * 32 + lane] = partial;
    __syncthreads();
    float total = 0.0f;
    if (warp_id == 0) {
        #pragma unroll
        for (int w = 0; w < 32; ++w) {
            if (w < n_warps) total += stage[w * 32 + lane];
        }
    }
    return total;
}

/// The same sum over a staging area declared with its warp count, which is then the bound.
/// The rows are contiguous, so it is the one above reading them flat.
template <int NWARPS>
static __device__ __forceinline__ float sum_down_warp_column(
    float (&stage)[NWARPS][32], int warp_id, int lane, float partial) {
    return sum_down_warp_column(&stage[0][0], warp_id, lane, NWARPS, partial);
}

// Gemma4 Global layers use HD=512. The generic fallback at the end of
// the macro handles arbitrary HD that's a multiple of 32.

// GQA variant: processes all kv-heads in a single launch (gridDim.y =
// n_kv_heads). Each block picks its kv-head from blockIdx.y and computes
// dot(Q[kv*n_q_per_kv + qh], K[n, kv]) for qh in [0, n_q_per_kv).
// Payoff: eliminates n_kv_heads-1 kernel-launch latencies per decode
// step - for GQA 8:1 with 8 kv-heads x ~10µs launch overhead each,
// that's ~70µs saved per attention forward.
#define ATTN_SCORE_GQA_KERNEL(NAME, HD)                                      \
extern "C" __global__ void NAME(                                             \
    const void * __restrict__ K_blob,                                        \
    const void * __restrict__ Q_blob,                                        \
    float * __restrict__ dst,                                                \
    int n_kv,                                                                \
    int n_kv_stride_blocks,      /* full token stride = n_kv_heads*HD/32 */  \
    int kv_head_stride_blocks,   /* per-kv-head stride within token = HD/32*/\
    int q_stride_blocks,         /* per-query-row stride in Q (padded) */    \
    int n_q_per_kv,                                                          \
    float scale                  /* pre-softmax scale (1/sqrt(head_dim)) */  \
) {                                                                          \
    const int kv = blockIdx.y;                                               \
    const int lane    = threadIdx.x;                                         \
    const int warp_id = threadIdx.y;                                         \
    const int n = blockIdx.x * blockDim.y + warp_id;                         \
    if (n >= n_kv) return;                                                   \
    const block_q8_0 * K_row =                                               \
        reinterpret_cast<const block_q8_0 *>(K_blob)                         \
        + n * n_kv_stride_blocks                                             \
        + kv * kv_head_stride_blocks;                                        \
    const block_q8_1 * Q_base =                                              \
        reinterpret_cast<const block_q8_1 *>(Q_blob)                         \
        + kv * n_q_per_kv * q_stride_blocks;                                 \
    if constexpr (HD == 128) {                                               \
        const int bi  = lane >> 3;                                           \
        const int iqs = lane & 7;                                            \
        const int k_val = four_bytes_unaligned(K_row[bi].qs, iqs);              \
        const float k_scale = __half2float(K_row[bi].d);                     \
        for (int qh = 0; qh < n_q_per_kv; ++qh) {                            \
            const block_q8_1 * Q_row = Q_base + qh * q_stride_blocks;        \
            const int q_val = four_bytes(Q_row[bi].qs, iqs);  \
            int sumi = __dp4a(k_val, q_val, 0);                              \
            sumi += __shfl_xor_sync(0xffffffff, sumi, 4);                    \
            sumi += __shfl_xor_sync(0xffffffff, sumi, 2);                    \
            sumi += __shfl_xor_sync(0xffffffff, sumi, 1);                    \
            float contrib = 0.0f;                                            \
            if (iqs == 0) {                                                  \
                const float q_scale = __half2float(__low2half(Q_row[bi].ds));\
                contrib = sumi * k_scale * q_scale;                          \
            }                                                                \
            contrib += __shfl_xor_sync(0xffffffff, contrib, 16);             \
            contrib += __shfl_xor_sync(0xffffffff, contrib, 8);              \
            if (lane == 0) {                                                 \
                const int q_row = kv * n_q_per_kv + qh;                      \
                dst[q_row * n_kv + n] = contrib * scale;                     \
            }                                                                \
        }                                                                    \
        return;                                                              \
    }                                                                        \
    if constexpr (HD == 64) {                                                \
        const int bi  = lane >> 3;                                           \
        const int iqs = lane & 7;                                            \
        const bool active = (lane < 16);                                     \
        int k_val = 0;                                                       \
        float k_scale = 0.0f;                                                \
        if (active) {                                                        \
            k_val = four_bytes_unaligned(K_row[bi].qs, iqs);                    \
            k_scale = __half2float(K_row[bi].d);                             \
        }                                                                    \
        for (int qh = 0; qh < n_q_per_kv; ++qh) {                            \
            const block_q8_1 * Q_row = Q_base + qh * q_stride_blocks;        \
            int sumi = 0;                                                    \
            if (active) {                                                    \
                const int q_val = four_bytes(Q_row[bi].qs, iqs);\
                sumi = __dp4a(k_val, q_val, 0);                              \
            }                                                                \
            sumi += __shfl_xor_sync(0xffffffff, sumi, 4);                    \
            sumi += __shfl_xor_sync(0xffffffff, sumi, 2);                    \
            sumi += __shfl_xor_sync(0xffffffff, sumi, 1);                    \
            float contrib = 0.0f;                                            \
            if (active && iqs == 0) {                                        \
                const float q_scale = __half2float(__low2half(Q_row[bi].ds));\
                contrib = sumi * k_scale * q_scale;                          \
            }                                                                \
            contrib += __shfl_xor_sync(0xffffffff, contrib, 8);              \
            if (lane == 0) {                                                 \
                const int q_row = kv * n_q_per_kv + qh;                      \
                dst[q_row * n_kv + n] = contrib * scale;                     \
            }                                                                \
        }                                                                    \
        return;                                                              \
    }                                                                        \
    /* Generic fallback for HD=256 etc.: iterate blocks, 8 lanes per block */\
    for (int qh = 0; qh < n_q_per_kv; ++qh) {                                \
        const block_q8_1 * Q_row = Q_base + qh * q_stride_blocks;            \
        float total = attn_score_row_q8_0_q8_1<HD>(K_row, Q_row, lane);      \
        if (lane == 0) {                                                     \
            const int q_row = kv * n_q_per_kv + qh;                          \
            dst[q_row * n_kv + n] = total * scale;                           \
        }                                                                    \
    }                                                                        \
}

ATTN_SCORE_GQA_KERNEL(attn_score_q8_0_q8_1_gqa_hd64,  64)
ATTN_SCORE_GQA_KERNEL(attn_score_q8_0_q8_1_gqa_hd128, 128)
ATTN_SCORE_GQA_KERNEL(attn_score_q8_0_q8_1_gqa_hd256, 256)
// Gemma4 Global layers use HD=512 -> falls into the generic loop at the
// end of the macro.
ATTN_SCORE_GQA_KERNEL(attn_score_q8_0_q8_1_gqa_hd512, 512)

// Inside-the-kernel GQA fan-out: each kernel block handles ALL `n_q_per_kv`
// query heads that share one kv-head. This is the same trick the Q8 V
// output kernel uses - K loads are reused across the qh axis, cutting
// global memory traffic by a factor of n_q_per_kv (8 for qwen3-coder,
// 4 for qwen3 base). Without it the kernel re-reads every K block
// n_q_per_kv times, which is what made the naive version bandwidth-bound.
//
// The LAST block (block_idx == full_blocks, when seq_kv % 32 != 0) is the
// partial residual - positions [full_blocks*32, seq_kv) live in the F16
// residual buffer, not the Q4 blob. The kernel reads those from
// `K_residual` instead of the Q4 blob, so scores for the whole valid
// seq_kv range (full blocks + residual) come from one kernel launch.
template <int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_score_q4_inner(
    const void * __restrict__ K_blocks,   // [seq_blocks, n_kv, head_dim] q4_0 cells
    const half * __restrict__ K_residual, // [n_kv, head_dim, 32] f16 (partial block)
    const float * __restrict__ Q,         // [n_q_heads, head_dim] f32
    float * __restrict__ scores,          // [n_q_heads, seq_kv] f32
    int seq_kv,
    int full_blocks,
    int n_kv,
    int n_q_per_kv,
    int sb,                               // seq-block index
    int h_kv,                             // kv head
    int lane,
    int warp_id,
    int n_warps
) {
    const int nib_byte = lane & 15;
    const bool is_high = lane >= 16;
    const int out_token = sb * 32 + lane;
    const bool is_partial = (sb == full_blocks);

    // Out-of-range tokens in the last (partial) block skip everything.
    if (out_token >= seq_kv) {
        // Still participate in __syncthreads below so the warp reduction
        // doesn't deadlock. We compute with k_val=0, which contributes 0.
    }

    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    for (int c = warp_id; c < HD; c += n_warps) {
        float k_val;
        if (is_partial) {
            // Residual F16 read. Layout is [n_kv, head_dim, 32]: the
            // `lane` axis directly indexes the slot within the partial
            // block.
            const size_t r_idx =
                (size_t)h_kv * (size_t)HD * 32
                + (size_t)c * 32
                + (size_t)lane;
            k_val = (out_token < seq_kv) ? __half2float(K_residual[r_idx]) : 0.0f;
        } else {
            const size_t block_idx =
                (size_t)sb * ((size_t)n_kv * (size_t)HD)
                + (size_t)h_kv * (size_t)HD
                + (size_t)c;
            const block_q4_0 * k_block =
                reinterpret_cast<const block_q4_0 *>(K_blocks) + block_idx;
            const uint8_t byte = k_block->qs[nib_byte];
            const int nib = is_high ? (byte >> 4) : (byte & 0xF);
            k_val = (nib - 8) * __half2float(k_block->d);
        }

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h_q = h_kv * n_q_per_kv + q;
                const float q_val = Q[(size_t)h_q * HD + c];
                acc[q] = fmaf(q_val, k_val, acc[q]);
            }
        }
    }

    extern __shared__ float shmem_buf[]; // [n_warps][32]
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total =
                sum_down_warp_column(shmem_buf, warp_id, lane, n_warps, acc[q]);
            if (warp_id == 0 && out_token < seq_kv) {
                const int h_q = h_kv * n_q_per_kv + q;
                scores[(size_t)h_q * seq_kv + out_token] = total;
            }
            __syncthreads();
        }
    }
}

#define ATTN_SCORE_Q4_KERNEL(NAME, HD, NQ)                                    \
extern "C" __global__ void NAME(                                              \
    const void * __restrict__ K_blocks,                                       \
    const half * __restrict__ K_residual,                                     \
    const float * __restrict__ Q,                                             \
    float * __restrict__ scores,                                              \
    int seq_kv,                                                               \
    int full_blocks,                                                          \
    int n_kv,                                                                 \
    int /* n_q_per_kv */                                                      \
) {                                                                           \
    const int sb   = blockIdx.x;                                              \
    const int h_kv = blockIdx.y;                                              \
    const int lane = threadIdx.x;                                             \
    const int warp = threadIdx.y;                                             \
    const int n_warps = blockDim.y;                                           \
    attn_score_q4_inner<HD, NQ>(                                              \
        K_blocks, K_residual, Q, scores, seq_kv, full_blocks, n_kv, NQ,       \
        sb, h_kv, lane, warp, n_warps                                         \
    );                                                                        \
}

ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd64_nq1,   64, 1)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd64_nq2,   64, 2)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd64_nq4,   64, 4)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd64_nq5,   64, 5)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd64_nq8,   64, 8)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd128_nq1, 128, 1)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd128_nq2, 128, 2)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd128_nq4, 128, 4)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd128_nq5, 128, 5)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd128_nq8, 128, 8)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd256_nq1, 256, 1)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd256_nq2, 256, 2)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd256_nq4, 256, 4)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd256_nq5, 256, 5)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd256_nq8, 256, 8)
// HD=512 specializations - for architectures with head_dim 512 global
// attention layers. The templated kernel scales cleanly with HD (just
// more channels-per-warp
// iterations), so this is a pure template instantiation.
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd512_nq1, 512, 1)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd512_nq2, 512, 2)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd512_nq4, 512, 4)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd512_nq5, 512, 5)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd512_nq8, 512, 8)
ATTN_SCORE_Q4_KERNEL(attn_score_q4_0_f32_kivi_hd512_nq16, 512, 16)

// ------------------------------------------------------------
// Device-position variant of attn_score_q4_*. Reads `seq_kv` from a device
// tensor instead of a host int, and writes scores into a pre-allocated
// `[n_q_heads, max_seq_padded]` buffer at the SAME stride every launch.
// Positions [seq_kv, max_seq_padded) are written as -INFINITY so the
// downstream softmax masks them out (matches the F-dtype graph-capture
// path which uses `padded_mask` for the same purpose).
//
// Required for CUDA graph capture of Q4-KV decode - the host updates
// `seq_kv_dev` outside the captured region each token, while the captured
// kernel uses the same dst pointer + max_seq_padded stride on every replay.
// ------------------------------------------------------------
template<int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_score_q4_inner_dev_pos(
    const void  * __restrict__ K_blocks,
    const half  * __restrict__ K_residual,
    const float * __restrict__ Q,
    float       * __restrict__ scores,        // [n_q_heads * max_seq_padded] f32
    const int32_t * __restrict__ seq_kv_dev,  // device ptr to current seq len (i32)
    int max_seq_padded,                       // stride of the scores buffer
    int n_kv,
    int n_q_per_kv,
    int sb, int h_kv, int lane, int warp_id, int n_warps
) {
    // Host stores `current_seq_len BEFORE the append` in seq_kv_dev (so
    // the append/flush kernels can derive slot/block from `pos & 31` and
    // `pos >> 5`). Attention sees the cache AFTER the append, so we
    // bump by 1 here.
    const int seq_kv = seq_kv_dev[0] + 1;
    const int full_blocks = seq_kv / 32;

    // Fast no-op for blocks entirely beyond `seq_kv`. Under graph capture
    // the grid is sized to `max_seq_padded / 32` blocks, which is the
    // model's full context. At small seq_kv the vast majority of those
    // blocks have no valid positions - skip the warp shuffles + global
    // reads and just write -INF directly. ATTN_SCORE_WARPS=16 normally;
    // we only need warp 0 + lane < max_seq_padded for the write.
    if (sb > full_blocks) {
        if (warp_id == 0) {
            const int out_token = sb * 32 + lane;
            if (out_token < max_seq_padded) {
                #pragma unroll
                for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
                    if (q < n_q_per_kv) {
                        const int h_q = h_kv * n_q_per_kv + q;
                        scores[(size_t)h_q * (size_t)max_seq_padded + (size_t)out_token]
                            = -INFINITY;
                    }
                }
            }
        }
        return;
    }

    const int nib_byte = lane & 15;
    const bool is_high = lane >= 16;
    const int out_token = sb * 32 + lane;
    const bool is_partial = (sb == full_blocks);
    const bool is_in_range = (out_token < seq_kv);

    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    for (int c = warp_id; c < HD; c += n_warps) {
        float k_val;
        if (is_partial) {
            const size_t r_idx =
                (size_t)h_kv * (size_t)HD * 32 + (size_t)c * 32 + (size_t)lane;
            k_val = is_in_range ? __half2float(K_residual[r_idx]) : 0.0f;
        } else {
            const size_t block_idx =
                (size_t)sb * ((size_t)n_kv * (size_t)HD)
                + (size_t)h_kv * (size_t)HD
                + (size_t)c;
            const block_q4_0 * k_block =
                reinterpret_cast<const block_q4_0 *>(K_blocks) + block_idx;
            const uint8_t byte = k_block->qs[nib_byte];
            const int nib = is_high ? (byte >> 4) : (byte & 0xF);
            k_val = (nib - 8) * __half2float(k_block->d);
        }
        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h_q = h_kv * n_q_per_kv + q;
                const float q_val = Q[(size_t)h_q * HD + c];
                acc[q] = fmaf(q_val, k_val, acc[q]);
            }
        }
    }

    extern __shared__ float shmem_buf[];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total =
                sum_down_warp_column(shmem_buf, warp_id, lane, n_warps, acc[q]);
            if (warp_id == 0) {
                const int h_q = h_kv * n_q_per_kv + q;
                // Always write to a STABLE offset using max_seq_padded as
                // the stride. Positions >= seq_kv get -INFINITY so softmax
                // ignores them. Valid positions get the real score.
                const size_t dst_off = (size_t)h_q * (size_t)max_seq_padded + (size_t)out_token;
                if (out_token < max_seq_padded) {
                    scores[dst_off] = is_in_range ? total : -INFINITY;
                }
            }
            __syncthreads();
        }
    }
}

#define ATTN_SCORE_Q4_DEV_POS_KERNEL(NAME, HD, NQ)                            \
extern "C" __global__ void NAME(                                              \
    const void  * __restrict__ K_blocks,                                      \
    const half  * __restrict__ K_residual,                                    \
    const float * __restrict__ Q,                                             \
    float       * __restrict__ scores,                                        \
    const int32_t * __restrict__ seq_kv_dev,                                  \
    int max_seq_padded,                                                       \
    int n_kv,                                                                 \
    int /* n_q_per_kv */                                                      \
) {                                                                           \
    const int sb   = blockIdx.x;                                              \
    const int h_kv = blockIdx.y;                                              \
    const int lane = threadIdx.x;                                             \
    const int warp = threadIdx.y;                                             \
    const int n_warps = blockDim.y;                                           \
    attn_score_q4_inner_dev_pos<HD, NQ>(                                      \
        K_blocks, K_residual, Q, scores, seq_kv_dev,                          \
        max_seq_padded, n_kv, NQ,                                             \
        sb, h_kv, lane, warp, n_warps                                         \
    );                                                                        \
}

ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd64_nq1,    64, 1)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd64_nq2,    64, 2)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd64_nq4,    64, 4)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd64_nq5,    64, 5)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd64_nq8,    64, 8)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd128_nq1, 128, 1)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd128_nq2, 128, 2)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd128_nq4, 128, 4)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd128_nq5, 128, 5)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd128_nq8, 128, 8)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd256_nq1, 256, 1)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd256_nq2, 256, 2)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd256_nq4, 256, 4)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd256_nq5, 256, 5)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd256_nq8, 256, 8)
// HD=512 dev-pos variants - for head_dim 512 global attention in graph mode.
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd512_nq1, 512, 1)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd512_nq2, 512, 2)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd512_nq4, 512, 4)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd512_nq5, 512, 5)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd512_nq8, 512, 8)
ATTN_SCORE_Q4_DEV_POS_KERNEL(attn_score_q4_0_f32_kivi_dev_pos_hd512_nq16, 512, 16)

// ------------------------------------------------------------
// Device-position variant of attn_score_q8_*. Same purpose as the Q4 dev_pos
// kernel above but for the Q8KvCache layout - K stored as block_q8_0 with no
// per-token residual (Q8 quantises per-token across the head_dim, so every
// token's K row is fully Q8 from append time).
//
// Differences vs the non-dev_pos Q8 attn_score kernels (line 3616+):
//   - Q is consumed as F32 directly. The non-dev_pos path pre-quantises Q to
//     Q8_1 outside the kernel via a host-side dev.alloc + quantize_q8_1
//     dispatch. The host-side alloc isn't graph-safe; F32 input is.
//   - `seq_kv` is read from a device i32 pointer.
//   - Scores are written into a fixed `[n_q_heads, max_seq_padded]` buffer.
//     Positions >= seq_kv get -INFINITY so the downstream softmax masks them.
//
// Launch contract:
//   blockDim = (32, n_warps_per_block, 1)
//   gridDim  = (ceil(max_seq_padded / n_warps_per_block), n_kv_heads, 1)
//   Each warp computes one (token, h_kv) and emits scores for all
//   `n_q_per_kv` query heads in that kv-group.
// ------------------------------------------------------------
template<int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_score_q8_inner_dev_pos(
    const void  * __restrict__ K_blocks,
    const float * __restrict__ Q,
    float       * __restrict__ scores,
    const int32_t * __restrict__ seq_kv_dev,
    int max_seq_padded,
    int n_kv,
    int n_q_per_kv,
    int token,
    int h_kv,
    int lane,
    int window
) {
    // Host stores `current_seq_len BEFORE the append` in seq_kv_dev so the
    // append kernels can derive the destination block index. Attention sees
    // the cache AFTER the append, so add 1.
    const int seq_kv = seq_kv_dev[0] + 1;

    if (token >= max_seq_padded) return;

    // Sliding-window attention (gemma4 SWA layers, window>0): tokens older
    // than `window` keys back are masked to -INF AND skip the Q.K dot - so the
    // scan cost is bounded at `window` instead of growing with the full KV
    // (the 2.5K decode collapse). window<=0 ⟹ full attention (bit-identical).
    const int win_start = (window > 0 && seq_kv > window) ? (seq_kv - window) : 0;

    // Out-of-range OR out-of-window: write -INFINITY and skip the dot.
    if (token >= seq_kv || token < win_start) {
        if (lane == 0) {
            #pragma unroll
            for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
                if (q < n_q_per_kv) {
                    const int h_q = h_kv * n_q_per_kv + q;
                    scores[(size_t)h_q * (size_t)max_seq_padded + (size_t)token]
                        = -INFINITY;
                }
            }
        }
        return;
    }

    constexpr int HD_BLOCKS = HD / 32;
    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    // Each lane holds one of 32 head_dim elements within each channel block.
    // Iterate channel blocks serially; with HD_BLOCKS <= 16 (HD <= 512) this
    // unrolls cleanly.
    #pragma unroll
    for (int b = 0; b < HD_BLOCKS; ++b) {
        const size_t block_off =
            ((size_t)token * (size_t)n_kv + (size_t)h_kv) * (size_t)HD_BLOCKS
            + (size_t)b;
        const block_q8_0 * k_block =
            reinterpret_cast<const block_q8_0 *>(K_blocks) + block_off;
        const float k_scale = __half2float(k_block->d);
        const int8_t k_i8   = k_block->qs[lane];
        const float k_val   = (float)k_i8 * k_scale;

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h_q = h_kv * n_q_per_kv + q;
                const float q_val = Q[(size_t)h_q * HD + b * 32 + lane];
                acc[q] = fmaf(q_val, k_val, acc[q]);
            }
        }
    }

    // Warp-reduce: 32 lanes each held HD/32 partial products; after the
    // butterfly, lane 0 holds the full dot product.
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            float v = acc[q];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off);
            }
            if (lane == 0) {
                const int h_q = h_kv * n_q_per_kv + q;
                scores[(size_t)h_q * (size_t)max_seq_padded + (size_t)token] = v;
            }
        }
    }
}

#define ATTN_SCORE_Q8_DEV_POS_KERNEL(NAME, HD, NQ)                            \
extern "C" __global__ void NAME(                                              \
    const void  * __restrict__ K_blocks,                                      \
    const float * __restrict__ Q,                                             \
    float       * __restrict__ scores,                                        \
    const int32_t * __restrict__ seq_kv_dev,                                  \
    int max_seq_padded,                                                       \
    int n_kv,                                                                 \
    int /* n_q_per_kv */,                                                     \
    int window                                                                \
) {                                                                           \
    const int token = blockIdx.x * blockDim.y + threadIdx.y;                  \
    const int h_kv  = blockIdx.y;                                             \
    const int lane  = threadIdx.x;                                            \
    attn_score_q8_inner_dev_pos<HD, NQ>(                                      \
        K_blocks, Q, scores, seq_kv_dev,                                      \
        max_seq_padded, n_kv, NQ,                                             \
        token, h_kv, lane, window                                            \
    );                                                                        \
}

ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd64_nq1,    64, 1)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd64_nq2,    64, 2)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd64_nq4,    64, 4)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd64_nq5,    64, 5)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd64_nq8,    64, 8)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd128_nq1, 128, 1)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd128_nq2, 128, 2)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd128_nq4, 128, 4)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd128_nq5, 128, 5)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd128_nq8, 128, 8)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd256_nq1, 256, 1)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd256_nq2, 256, 2)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd256_nq4, 256, 4)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd256_nq5, 256, 5)
ATTN_SCORE_Q8_DEV_POS_KERNEL(attn_score_q8_0_f32_dev_pos_hd256_nq8, 256, 8)

// V-path attention output: `out = probs @ V` where:
//   probs : [n_q_heads, seq_kv]   (f32, softmax output)
//   V     : [seq_kv, n_kv_heads, head_dim] Q8_0 blocks along head_dim
//   out   : [n_q_heads, head_dim] (f32)
//
// Reduction is along seq_kv (NOT head_dim), but Q8_0 blocks pack along
// head_dim - so one block spans 32 head_dim elements for a fixed (seq, kv).
// Each warp owns a (kv, block_idx, qh) triple: 32 lanes cover 32 head_dim
// outputs, one per lane, and each lane loops over seq_kv scanning its
// d-column. The block's shared scale `d` loads once per (s, kv, block_idx)
// iteration and is broadcast via the half-to-float conversion; there's no
// cross-lane reduction since each lane has an independent output.
//
// Launch: gridDim = (head_dim / 32, n_kv_heads, n_q_per_kv),
//         blockDim = (32, 1, 1). No shared memory needed.
//
// Compute bandwidth per output element: seq_kv x (1 i8 + 1/32 f16 scale +
// 1 f32 probs). For typical qwen3 (n_q_heads=32, head_dim=128, seq_kv=2K):
// ~33 KB V bytes per block tile x 32 tiles = 1 MB per call - memory-bound
// and well within <100µs on any modern GPU.
#define ATTN_OUTPUT_WARPS 32  /* warps/block splitting the seq_kv reduction */

// Per-warp loop that dequantises one V row element + multiplies by each
// of n_q_per_kv probs values, accumulating into register-resident per-qh
// accumulators. Pulls V from global memory exactly once per (s, kv,
// block_idx) - no re-reads across the qh dimension.
template <int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_output_inner(
    const void * __restrict__ V_blob,
    const float * __restrict__ probs,
    float * __restrict__ out,
    int seq_kv,
    int n_kv_stride_blocks,
    int kv_head_stride_blocks,
    int n_q_per_kv,
    int block_idx,
    int kv,
    int lane,
    int warp_id
) {
    const int HD_BLOCKS = HD / 32;
    (void)HD_BLOCKS;

    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    for (int s = warp_id; s < seq_kv; s += ATTN_OUTPUT_WARPS) {
        const block_q8_0 * v_block =
            reinterpret_cast<const block_q8_0 *>(V_blob)
            + (size_t)s * n_kv_stride_blocks
            + kv * kv_head_stride_blocks
            + block_idx;
        const int8_t v_i8 = v_block->qs[lane];
        const float v_f = static_cast<float>(v_i8) * __half2float(v_block->d);

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h = kv * n_q_per_kv + q;
                const float p = probs[(size_t)h * seq_kv + s];
                acc[q] = fmaf(p, v_f, acc[q]);
            }
        }
    }

    __shared__ float shmem[ATTN_OUTPUT_WARPS][32];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total = sum_down_warp_column(shmem, warp_id, lane, acc[q]);
            if (warp_id == 0) {
                const int h = kv * n_q_per_kv + q;
                const int d = block_idx * 32 + lane;
                out[(size_t)h * HD + d] = total;
            }
            __syncthreads();
        }
    }
}

// Specialise on exact n_q_per_kv so the `acc[N]` array stays in registers.
// 1 = MHA / MQA; 4 = qwen3 base GQA 8:1; 8 = qwen3-coder / Llama-3 70B;
// fallback to a slow dynamic path for anything else.
#define ATTN_OUTPUT_KERNEL(NAME, HD, NQ)                                     \
/* Launched at ATTN_OUTPUT_WARPS x 32 threads. Without the bound the compiler  \
   is free to spend more than 64 registers per thread on an arch it targets     \
   natively, and a 1024-thread block then fails to LAUNCH - out of resources,   \
   at run time, on exactly the cards compiled for most precisely. */            \
extern "C" __global__ void __launch_bounds__(ATTN_OUTPUT_WARPS * 32) NAME(                                             \
    const void * __restrict__ V_blob,                                        \
    const float * __restrict__ probs,                                        \
    float * __restrict__ out,                                                \
    int seq_kv,                                                              \
    int n_kv_stride_blocks,                                                  \
    int kv_head_stride_blocks,                                               \
    int /* n_q_per_kv */                                                     \
) {                                                                          \
    const int block_idx = blockIdx.x;                                        \
    const int kv        = blockIdx.y;                                        \
    const int lane      = threadIdx.x;                                       \
    const int warp_id   = threadIdx.y;                                       \
    if (block_idx >= (HD / 32)) return;                                      \
    attn_output_inner<HD, NQ>(                                               \
        V_blob, probs, out, seq_kv,                                          \
        n_kv_stride_blocks, kv_head_stride_blocks, NQ,                       \
        block_idx, kv, lane, warp_id                                         \
    );                                                                       \
}

ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd64_nq1,   64, 1)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd64_nq2,   64, 2)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd64_nq4,   64, 4)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd64_nq5,   64, 5)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd64_nq8,   64, 8)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd128_nq1, 128, 1)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd128_nq2, 128, 2)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd128_nq4, 128, 4)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd128_nq5, 128, 5)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd128_nq8, 128, 8)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd256_nq1, 256, 1)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd256_nq2, 256, 2)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd256_nq4, 256, 4)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd256_nq5, 256, 5)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd256_nq8, 256, 8)
// Gemma4 Global layers (HD=512). Same kernel template; HD/32=16 blocks.
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd512_nq1, 512, 1)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd512_nq2, 512, 2)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd512_nq4, 512, 4)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd512_nq5, 512, 5)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd512_nq8, 512, 8)
ATTN_OUTPUT_KERNEL(attn_output_q8_0_f32_hd512_nq16, 512, 16)

// ------------------------------------------------------------
// Device-position variant of attn_output_q8_*. Same template as the
// non-dev_pos kernel but:
//   - `seq_kv` is read from a device i32 pointer (host updates it outside
//     the captured region each token).
//   - `probs` is laid out at the fixed `max_seq_padded` stride. probs[h, s]
//     for s >= seq_kv MUST be zero (the upstream softmax produces this
//     because attn_score_q8_*_dev_pos writes -INFINITY at those positions).
// Required for CUDA graph capture of the Q8 decode attention output step.
// ------------------------------------------------------------
// 32 warps x 32 threads (1024 threads/block) exceeds register budget on
// some compute caps for the dev_pos variant - the extra `seq_kv_dev[0]`
// load + max_seq_padded indexing tip the per-thread reg count above the
// 64 reg limit that 1024 threads can fit in a 65 KB register file.
// Halve to 16 warps; the seq_kv reduction still parallelises 16-way.
#define ATTN_OUTPUT_DEV_POS_WARPS 16

template <int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_output_q8_inner_dev_pos(
    const void * __restrict__ V_blob,
    const float * __restrict__ probs,
    float * __restrict__ out,
    const int32_t * __restrict__ seq_kv_dev,
    int max_seq_padded,
    int n_kv_stride_blocks,
    int kv_head_stride_blocks,
    int n_q_per_kv,
    int block_idx,
    int kv,
    int lane,
    int warp_id
) {
    const int seq_kv = seq_kv_dev[0] + 1;

    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    for (int s = warp_id; s < seq_kv; s += ATTN_OUTPUT_DEV_POS_WARPS) {
        const block_q8_0 * v_block =
            reinterpret_cast<const block_q8_0 *>(V_blob)
            + (size_t)s * n_kv_stride_blocks
            + kv * kv_head_stride_blocks
            + block_idx;
        const int8_t v_i8 = v_block->qs[lane];
        const float v_f = static_cast<float>(v_i8) * __half2float(v_block->d);

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h = kv * n_q_per_kv + q;
                // probs stride is max_seq_padded (vs seq_kv in non-dev_pos)
                const float p = probs[(size_t)h * max_seq_padded + (size_t)s];
                acc[q] = fmaf(p, v_f, acc[q]);
            }
        }
    }

    __shared__ float shmem[ATTN_OUTPUT_DEV_POS_WARPS][32];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total = sum_down_warp_column(shmem, warp_id, lane, acc[q]);
            if (warp_id == 0) {
                const int h = kv * n_q_per_kv + q;
                const int d = block_idx * 32 + lane;
                out[(size_t)h * HD + d] = total;
            }
            __syncthreads();
        }
    }
}

#define ATTN_OUTPUT_Q8_DEV_POS_KERNEL(NAME, HD, NQ)                          \
extern "C" __global__ void NAME(                                             \
    const void * __restrict__ V_blob,                                        \
    const float * __restrict__ probs,                                        \
    float * __restrict__ out,                                                \
    const int32_t * __restrict__ seq_kv_dev,                                 \
    int max_seq_padded,                                                      \
    int n_kv_stride_blocks,                                                  \
    int kv_head_stride_blocks                                                \
) {                                                                          \
    const int block_idx = blockIdx.x;                                        \
    const int kv        = blockIdx.y;                                        \
    const int lane      = threadIdx.x;                                       \
    const int warp_id   = threadIdx.y;                                       \
    if (block_idx >= (HD / 32)) return;                                      \
    attn_output_q8_inner_dev_pos<HD, NQ>(                                    \
        V_blob, probs, out, seq_kv_dev,                                      \
        max_seq_padded,                                                      \
        n_kv_stride_blocks, kv_head_stride_blocks, NQ,                       \
        block_idx, kv, lane, warp_id                                         \
    );                                                                       \
}

ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd64_nq1,    64, 1)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd64_nq2,    64, 2)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd64_nq4,    64, 4)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd64_nq5,    64, 5)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd64_nq8,    64, 8)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd128_nq1, 128, 1)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd128_nq2, 128, 2)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd128_nq4, 128, 4)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd128_nq5, 128, 5)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd128_nq8, 128, 8)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd256_nq1, 256, 1)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd256_nq2, 256, 2)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd256_nq4, 256, 4)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd256_nq5, 256, 5)
ATTN_OUTPUT_Q8_DEV_POS_KERNEL(attn_output_q8_0_f32_dev_pos_hd256_nq8, 256, 8)

// --- Q4_0 V-path attention output ----------------------------------------
//
// Mirrors the Q8_0 path above with Q4_0 blocks (18 bytes / 32 elements).
// V layout is [seq_kv, n_kv, head_dim] of block_q4_0 - same as the Q8 V
// packing, just half the bytes per block. The kernel structure is
// identical; only the per-element dequantise changes.
template <int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_output_q4_inner(
    const void * __restrict__ V_blob,
    const float * __restrict__ probs,
    float * __restrict__ out,
    int seq_kv,
    int n_kv_stride_blocks,
    int kv_head_stride_blocks,
    int n_q_per_kv,
    int block_idx,
    int kv,
    int lane,
    int warp_id
) {
    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    // lane in 0..32 addresses one of the 32 dequantised head_dim values of
    // this block. Q4_0 packs element j into qs[j % 16]: low nibble when
    // j < 16, high nibble when j >= 16.
    const int nib_byte = lane & 15;
    const bool is_high = lane >= 16;

    for (int s = warp_id; s < seq_kv; s += ATTN_OUTPUT_WARPS) {
        const block_q4_0 * v_block =
            reinterpret_cast<const block_q4_0 *>(V_blob)
            + (size_t)s * n_kv_stride_blocks
            + kv * kv_head_stride_blocks
            + block_idx;
        const uint8_t byte = v_block->qs[nib_byte];
        const int nib = is_high ? (byte >> 4) : (byte & 0xF);
        const float v_f = (nib - 8) * __half2float(v_block->d);

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h = kv * n_q_per_kv + q;
                const float p = probs[(size_t)h * seq_kv + s];
                acc[q] = fmaf(p, v_f, acc[q]);
            }
        }
    }

    __shared__ float shmem[ATTN_OUTPUT_WARPS][32];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total = sum_down_warp_column(shmem, warp_id, lane, acc[q]);
            if (warp_id == 0) {
                const int h = kv * n_q_per_kv + q;
                const int d = block_idx * 32 + lane;
                out[(size_t)h * HD + d] = total;
            }
            __syncthreads();
        }
    }
}

#define ATTN_OUTPUT_Q4_KERNEL(NAME, HD, NQ)                                  \
extern "C" __global__ void NAME(                                             \
    const void * __restrict__ V_blob,                                        \
    const float * __restrict__ probs,                                        \
    float * __restrict__ out,                                                \
    int seq_kv,                                                              \
    int n_kv_stride_blocks,                                                  \
    int kv_head_stride_blocks,                                               \
    int /* n_q_per_kv */                                                     \
) {                                                                          \
    const int block_idx = blockIdx.x;                                        \
    const int kv        = blockIdx.y;                                        \
    const int lane      = threadIdx.x;                                       \
    const int warp_id   = threadIdx.y;                                       \
    if (block_idx >= (HD / 32)) return;                                      \
    attn_output_q4_inner<HD, NQ>(                                            \
        V_blob, probs, out, seq_kv,                                          \
        n_kv_stride_blocks, kv_head_stride_blocks, NQ,                       \
        block_idx, kv, lane, warp_id                                         \
    );                                                                       \
}

// --- Q4_0 V-path attention output (split-K variant) ----------------------
//
// Same dot product (probs @ V) but with the seq_kv reduction split across
// many blocks via grid_z. Each block handles a SEQ_PER_SPLIT slice of
// seq_kv for one (HD chunk, kv head) and atomicAdd's its partial into the
// pre-zeroed output. Targets the high-ctx regime (>~12k seq_kv) where the
// monolithic 32-warp kernel under-saturates SMs (~6 warps/SM); the
// split-K form scales the block count linearly with seq_kv.
//
// Kernel parameters:
//   ATTN_OUTPUT_SPLITK_WARPS  - warps per block (default 8 -> 256 threads)
//   seq_per_split              - runtime, controlled by host (default 256)
//
// Grid: (HD/32, n_kv, S) where S = ceil(seq_kv / seq_per_split).
//
// Caller MUST pre-zero `out` (cudaMemsetAsync) before launch because the
// kernel atomicAdd's its partial.
#define ATTN_OUTPUT_SPLITK_WARPS 8

template <int HD, int MAX_NQ_PER_KV, int WARPS_PER_BLOCK>
static __device__ __forceinline__ void attn_output_q4_splitk_inner(
    const void * __restrict__ V_blob,
    const float * __restrict__ probs,
    float * __restrict__ out,
    int seq_kv,
    int n_kv_stride_blocks,
    int kv_head_stride_blocks,
    int n_q_per_kv,
    int block_idx,
    int kv,
    int s_start,
    int s_end,
    int lane,
    int warp_id
) {
    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    const int nib_byte = lane & 15;
    const bool is_high = lane >= 16;

    // Each warp strides through the [s_start, s_end) range.
    for (int s = s_start + warp_id; s < s_end; s += WARPS_PER_BLOCK) {
        const block_q4_0 * v_block =
            reinterpret_cast<const block_q4_0 *>(V_blob)
            + (size_t)s * n_kv_stride_blocks
            + kv * kv_head_stride_blocks
            + block_idx;
        const uint8_t byte = v_block->qs[nib_byte];
        const int nib = is_high ? (byte >> 4) : (byte & 0xF);
        const float v_f = (nib - 8) * __half2float(v_block->d);

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h = kv * n_q_per_kv + q;
                const float p = probs[(size_t)h * seq_kv + s];
                acc[q] = fmaf(p, v_f, acc[q]);
            }
        }
    }

    __shared__ float shmem[WARPS_PER_BLOCK][32];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total = sum_down_warp_column(shmem, warp_id, lane, acc[q]);
            if (warp_id == 0) {
                const int h = kv * n_q_per_kv + q;
                const int d = block_idx * 32 + lane;
                atomicAdd(&out[(size_t)h * HD + d], total);
            }
            __syncthreads();
        }
    }
}

#define ATTN_OUTPUT_Q4_SPLITK_KERNEL(NAME, HD, NQ)                              \
extern "C" __global__ void NAME(                                                \
    const void * __restrict__ V_blob,                                           \
    const float * __restrict__ probs,                                           \
    float * __restrict__ out,                                                   \
    int seq_kv,                                                                 \
    int n_kv_stride_blocks,                                                     \
    int kv_head_stride_blocks,                                                  \
    int seq_per_split                                                           \
) {                                                                             \
    const int block_idx = blockIdx.x;                                           \
    const int kv        = blockIdx.y;                                           \
    const int sb        = blockIdx.z;                                           \
    const int lane      = threadIdx.x;                                          \
    const int warp_id   = threadIdx.y;                                          \
    if (block_idx >= (HD / 32)) return;                                         \
    const int s_start = sb * seq_per_split;                                     \
    const int s_end   = min(s_start + seq_per_split, seq_kv);                   \
    if (s_start >= s_end) return;                                               \
    attn_output_q4_splitk_inner<HD, NQ, ATTN_OUTPUT_SPLITK_WARPS>(              \
        V_blob, probs, out, seq_kv,                                             \
        n_kv_stride_blocks, kv_head_stride_blocks, NQ,                          \
        block_idx, kv, s_start, s_end, lane, warp_id                            \
    );                                                                          \
}

ATTN_OUTPUT_Q4_SPLITK_KERNEL(attn_output_q4_0_f32_splitk_hd64_nq1,    64, 1)
ATTN_OUTPUT_Q4_SPLITK_KERNEL(attn_output_q4_0_f32_splitk_hd64_nq4,    64, 4)
ATTN_OUTPUT_Q4_SPLITK_KERNEL(attn_output_q4_0_f32_splitk_hd64_nq8,    64, 8)
ATTN_OUTPUT_Q4_SPLITK_KERNEL(attn_output_q4_0_f32_splitk_hd128_nq1, 128, 1)
ATTN_OUTPUT_Q4_SPLITK_KERNEL(attn_output_q4_0_f32_splitk_hd128_nq4, 128, 4)
ATTN_OUTPUT_Q4_SPLITK_KERNEL(attn_output_q4_0_f32_splitk_hd128_nq8, 128, 8)
ATTN_OUTPUT_Q4_SPLITK_KERNEL(attn_output_q4_0_f32_splitk_hd256_nq1, 256, 1)
ATTN_OUTPUT_Q4_SPLITK_KERNEL(attn_output_q4_0_f32_splitk_hd256_nq4, 256, 4)
ATTN_OUTPUT_Q4_SPLITK_KERNEL(attn_output_q4_0_f32_splitk_hd256_nq8, 256, 8)

// --- Fused softmax + Q4 attn output -------------------------------------
// Eliminates the explicit `softmax_last_dim(scores)` launch + the probs
// roundtrip. Reads raw `scores` (post-Q@K^T, post-scale), computes per-head
// softmax stats inline, then folds the multiply with V into the same block
// pass. Saves ~48 launches/token at decode (one softmax per layer) plus the
// probs buffer write/read.
template <int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_output_softmax_q4_inner(
    const void * __restrict__ V_blob,
    const float * __restrict__ scores,
    float * __restrict__ out,
    int seq_kv,
    int n_kv_stride_blocks,
    int kv_head_stride_blocks,
    int n_q_per_kv,
    int block_idx,
    int kv,
    int lane,
    int warp_id
) {
    // Pass 1: per-h_q softmax max + 1/denom. Each (block_idx, kv) block
    // redundantly computes for its kv-group's q heads (minor: avoids
    // cross-block sync). Warp 0 only; thread `lane` handles h_q=lane.
    __shared__ float s_max[MAX_NQ_PER_KV];
    __shared__ float s_inv_denom[MAX_NQ_PER_KV];
    if (warp_id == 0 && lane < n_q_per_kv) {
        const int h_q = kv * n_q_per_kv + lane;
        const float * row = scores + (size_t)h_q * (size_t)seq_kv;
        float m = -INFINITY;
        for (int t = 0; t < seq_kv; ++t) m = fmaxf(m, row[t]);
        float d = 0.0f;
        // __expf: hardware intrinsic (~16x faster, ~2-ULP precision delta  - 
        // safe for softmax since input is pre-max-subtracted).
        for (int t = 0; t < seq_kv; ++t) d += __expf(row[t] - m);
        s_max[lane] = m;
        s_inv_denom[lane] = 1.0f / d;
    }
    __syncthreads();

    // Pass 2: stream V, fold-in softmax weights.
    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    const int nib_byte = lane & 15;
    const bool is_high = lane >= 16;

    for (int s = warp_id; s < seq_kv; s += ATTN_OUTPUT_WARPS) {
        const block_q4_0 * v_block =
            reinterpret_cast<const block_q4_0 *>(V_blob)
            + (size_t)s * n_kv_stride_blocks
            + kv * kv_head_stride_blocks
            + block_idx;
        const uint8_t byte = v_block->qs[nib_byte];
        const int nib = is_high ? (byte >> 4) : (byte & 0xF);
        const float v_f = (nib - 8) * __half2float(v_block->d);

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h = kv * n_q_per_kv + q;
                const float raw = scores[(size_t)h * (size_t)seq_kv + s];
                const float p = __expf(raw - s_max[q]) * s_inv_denom[q];
                acc[q] = fmaf(p, v_f, acc[q]);
            }
        }
    }

    // Cross-warp reduction (same as attn_output_q4_inner).
    __shared__ float shmem[ATTN_OUTPUT_WARPS][32];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total = sum_down_warp_column(shmem, warp_id, lane, acc[q]);
            if (warp_id == 0) {
                const int h = kv * n_q_per_kv + q;
                const int d = block_idx * 32 + lane;
                out[(size_t)h * HD + d] = total;
            }
            __syncthreads();
        }
    }
}

#define ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(NAME, HD, NQ)                          \
/* Launched at ATTN_OUTPUT_WARPS x 32 threads. Without the bound the compiler  \
   is free to spend more than 64 registers per thread on an arch it targets     \
   natively, and a 1024-thread block then fails to LAUNCH - out of resources,   \
   at run time, on exactly the cards compiled for most precisely. */            \
extern "C" __global__ void __launch_bounds__(ATTN_OUTPUT_WARPS * 32) NAME(                                             \
    const void * __restrict__ V_blob,                                        \
    const float * __restrict__ scores,                                       \
    float * __restrict__ out,                                                \
    int seq_kv,                                                              \
    int n_kv_stride_blocks,                                                  \
    int kv_head_stride_blocks,                                               \
    int /* n_q_per_kv */                                                     \
) {                                                                          \
    const int block_idx = blockIdx.x;                                        \
    const int kv        = blockIdx.y;                                        \
    const int lane      = threadIdx.x;                                       \
    const int warp_id   = threadIdx.y;                                       \
    if (block_idx >= (HD / 32)) return;                                      \
    attn_output_softmax_q4_inner<HD, NQ>(                                    \
        V_blob, scores, out, seq_kv,                                         \
        n_kv_stride_blocks, kv_head_stride_blocks, NQ,                       \
        block_idx, kv, lane, warp_id                                         \
    );                                                                       \
}

// --- Fused softmax + Q8 attn output -------------------------------------
// Q8 sibling of attn_output_softmax_q4. block_q8_0 has 32 int8 values
// per block (vs Q4's 32 nibbles in 16 bytes), so each lane reads one
// byte directly - no nibble split needed. Same softmax math.
//
// With a Q8 KV cache this replaces 4 launches (scores + scale + softmax
// + output) with 1, saving 3 launches per layer.
template <int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_output_softmax_q8_inner(
    const void * __restrict__ V_blob,
    const float * __restrict__ scores,
    float * __restrict__ out,
    int seq_kv,
    int n_kv_stride_blocks,
    int kv_head_stride_blocks,
    int n_q_per_kv,
    int block_idx,
    int kv,
    int lane,
    int warp_id
) {
    __shared__ float s_max[MAX_NQ_PER_KV];
    __shared__ float s_inv_denom[MAX_NQ_PER_KV];
    // Parallel reduction: warp 0 cooperates across 32 lanes for each
    // h_q row. For a long context (e.g. seq_kv=3000), this drops the row
    // reduction from ~6000 serial iters to ~94 iters + 5 warp-shuffle
    // log2(32) steps (the prior single-threaded reduction was the
    // bottleneck).
    if (warp_id == 0) {
        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h_q = kv * n_q_per_kv + q;
                const float * row = scores + (size_t)h_q * (size_t)seq_kv;
                // Pass 1: per-lane max over interleaved chunks.
                float m_lane = -INFINITY;
                for (int t = lane; t < seq_kv; t += 32) {
                    m_lane = fmaxf(m_lane, row[t]);
                }
                // Warp reduction for max.
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    m_lane = fmaxf(m_lane, __shfl_xor_sync(0xFFFFFFFFu, m_lane, off));
                }
                const float m = m_lane;
                // Pass 2: per-lane partial denom with broadcast max.
                // __expf: hardware intrinsic (~16x faster, ~2-ULP precision
                // delta - safe for softmax since inputs are pre-max-subtracted).
                float d_lane = 0.0f;
                for (int t = lane; t < seq_kv; t += 32) {
                    d_lane += __expf(row[t] - m);
                }
                // Warp reduction for sum.
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    d_lane += __shfl_xor_sync(0xFFFFFFFFu, d_lane, off);
                }
                if (lane == 0) {
                    s_max[q] = m;
                    s_inv_denom[q] = 1.0f / d_lane;
                }
            }
        }
    }
    __syncthreads();

    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    for (int s = warp_id; s < seq_kv; s += ATTN_OUTPUT_WARPS) {
        const block_q8_0 * v_block =
            reinterpret_cast<const block_q8_0 *>(V_blob)
            + (size_t)s * n_kv_stride_blocks
            + kv * kv_head_stride_blocks
            + block_idx;
        // Q8: one int8 per thread. lane in [0,32) directly indexes qs.
        const float v_f = (float)v_block->qs[lane] * __half2float(v_block->d);

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h = kv * n_q_per_kv + q;
                const float raw = scores[(size_t)h * (size_t)seq_kv + s];
                const float p = __expf(raw - s_max[q]) * s_inv_denom[q];
                acc[q] = fmaf(p, v_f, acc[q]);
            }
        }
    }

    __shared__ float shmem[ATTN_OUTPUT_WARPS][32];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total = sum_down_warp_column(shmem, warp_id, lane, acc[q]);
            if (warp_id == 0) {
                const int h = kv * n_q_per_kv + q;
                const int d = block_idx * 32 + lane;
                out[(size_t)h * HD + d] = total;
            }
            __syncthreads();
        }
    }
}

#define ATTN_OUTPUT_SOFTMAX_Q8_KERNEL(NAME, HD, NQ)                          \
/* Launched at ATTN_OUTPUT_WARPS x 32 threads. Without the bound the compiler  \
   is free to spend more than 64 registers per thread on an arch it targets     \
   natively, and a 1024-thread block then fails to LAUNCH - out of resources,   \
   at run time, on exactly the cards compiled for most precisely. */            \
extern "C" __global__ void __launch_bounds__(ATTN_OUTPUT_WARPS * 32) NAME(                                             \
    const void * __restrict__ V_blob,                                        \
    const float * __restrict__ scores,                                       \
    float * __restrict__ out,                                                \
    int seq_kv,                                                              \
    int n_kv_stride_blocks,                                                  \
    int kv_head_stride_blocks,                                               \
    int /* n_q_per_kv */                                                     \
) {                                                                          \
    const int block_idx = blockIdx.x;                                        \
    const int kv        = blockIdx.y;                                        \
    const int lane      = threadIdx.x;                                       \
    const int warp_id   = threadIdx.y;                                       \
    if (block_idx >= (HD / 32)) return;                                      \
    attn_output_softmax_q8_inner<HD, NQ>(                                    \
        V_blob, scores, out, seq_kv,                                         \
        n_kv_stride_blocks, kv_head_stride_blocks, NQ,                       \
        block_idx, kv, lane, warp_id                                         \
    );                                                                       \
}

// e.g. qwen2 with n_q=40, n_kv=8 -> nq_per_kv=5. Bucket to 8 (next power).
ATTN_OUTPUT_SOFTMAX_Q8_KERNEL(attn_softmax_output_q8_0_f32_hd128_nq8, 128, 8)
ATTN_OUTPUT_SOFTMAX_Q8_KERNEL(attn_softmax_output_q8_0_f32_hd128_nq4, 128, 4)
ATTN_OUTPUT_SOFTMAX_Q8_KERNEL(attn_softmax_output_q8_0_f32_hd128_nq1, 128, 1)
ATTN_OUTPUT_SOFTMAX_Q8_KERNEL(attn_softmax_output_q8_0_f32_hd64_nq8,   64, 8)
ATTN_OUTPUT_SOFTMAX_Q8_KERNEL(attn_softmax_output_q8_0_f32_hd64_nq4,   64, 4)
ATTN_OUTPUT_SOFTMAX_Q8_KERNEL(attn_softmax_output_q8_0_f32_hd64_nq1,   64, 1)

ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(attn_softmax_output_q4_0_f32_hd128_nq8, 128, 8)
ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(attn_softmax_output_q4_0_f32_hd128_nq4, 128, 4)
ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(attn_softmax_output_q4_0_f32_hd128_nq1, 128, 1)
ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(attn_softmax_output_q4_0_f32_hd64_nq8,   64, 8)
ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(attn_softmax_output_q4_0_f32_hd64_nq4,   64, 4)
ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(attn_softmax_output_q4_0_f32_hd64_nq1,   64, 1)
ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(attn_softmax_output_q4_0_f32_hd256_nq8, 256, 8)
ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(attn_softmax_output_q4_0_f32_hd256_nq4, 256, 4)
ATTN_OUTPUT_SOFTMAX_Q4_KERNEL(attn_softmax_output_q4_0_f32_hd256_nq1, 256, 1)

// ------------------------------------------------------------
// Q8 fused scale+softmax+output with device-side seq_kv (dev_pos variant).
// Combines the 3 separate launches (affine scale, softmax_last_dim, then
// attn_output_q8_0_f32_dev_pos) into ONE kernel - saves 2 launches per
// layer per token plus the intermediate probs write/read.
//
// Inputs:
//   - scores: RAW Q.K^T at fixed max_seq_padded stride. Positions
//             >= seq_kv have -INFINITY from attn_score_q8_0_f32_dev_pos.
//   - V_blob: Q8_0-packed V at [max_seq_padded, n_kv, head_dim] layout.
//   - scale:  1/√head_dim, applied in-kernel before softmax.
//   - seq_kv_dev: device i32 holding current_seq_len BEFORE this token's
//                 append + 1 (read as seq_kv_dev[0]+1, matching the
//                 score kernel's convention).
//
// Output: [n_q_heads, head_dim] f32.
//
// Block shape: 16 warps x 32 threads = 512 threads/block. Matches the
// dev_pos non-fused output kernel - fits phi2's hd=64 register budget
// where the 32-warp non-dev_pos kernel hits LAUNCH_OUT_OF_RESOURCES.
// ------------------------------------------------------------
#define ATTN_SOFTMAX_OUTPUT_DEV_POS_WARPS 16

template <int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_softmax_output_q8_inner_dev_pos(
    const void * __restrict__ V_blob,
    const float * __restrict__ scores,
    float * __restrict__ out,
    const int32_t * __restrict__ seq_kv_dev,
    int max_seq_padded,
    int n_kv_stride_blocks,
    int kv_head_stride_blocks,
    float scale,
    int n_q_per_kv,
    int block_idx,
    int kv,
    int lane,
    int warp_id,
    int window
) {
    const int seq_kv = seq_kv_dev[0] + 1;
    // Sliding-window (gemma4 SWA): only the last `window` keys contribute. The
    // score kernel already -INF'd the rest, so starting the reductions at
    // win_start is bit-identical AND skips their score/V loads (the perf win).
    const int win_start = (window > 0 && seq_kv > window) ? (seq_kv - window) : 0;

    __shared__ float s_max[MAX_NQ_PER_KV];
    __shared__ float s_inv_denom[MAX_NQ_PER_KV];

    // Warp 0 computes per-q row max + 1/sum(exp(...)) over the valid
    // [0, seq_kv) range. Padding past seq_kv is -INF in `scores` and
    // contributes 0 to both max and denom anyway - we still skip those
    // iterations explicitly to save the loads.
    if (warp_id == 0) {
        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h_q = kv * n_q_per_kv + q;
                const float * row = scores + (size_t)h_q * (size_t)max_seq_padded;
                // Pass 1: per-lane max.
                float m_lane = -INFINITY;
                for (int t = win_start + lane; t < seq_kv; t += 32) {
                    m_lane = fmaxf(m_lane, row[t] * scale);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    m_lane = fmaxf(m_lane, __shfl_xor_sync(0xFFFFFFFFu, m_lane, off));
                }
                const float m = m_lane;
                // Pass 2: per-lane partial denom with broadcast max.
                // __expf: hardware intrinsic, ~16x faster than expf.
                // Safe here because raw values are pre-subtracted by max so
                // input <= 0, output in (0, 1]. The ~2-ULP precision delta
                // is well within softmax's natural noise budget.
                float d_lane = 0.0f;
                for (int t = win_start + lane; t < seq_kv; t += 32) {
                    d_lane += __expf(row[t] * scale - m);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    d_lane += __shfl_xor_sync(0xFFFFFFFFu, d_lane, off);
                }
                if (lane == 0) {
                    s_max[q] = m;
                    s_inv_denom[q] = 1.0f / d_lane;
                }
            }
        }
    }
    __syncthreads();

    // All warps cooperate on the V multiply: each warp owns a stride of
    // seq positions (warp_id, warp_id + 16, ...). Inside each iteration
    // we re-read the row's score and apply scale + softmax inline (saves
    // the global probs buffer write/read from the unfused path).
    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    for (int s = win_start + warp_id; s < seq_kv; s += ATTN_SOFTMAX_OUTPUT_DEV_POS_WARPS) {
        const block_q8_0 * v_block =
            reinterpret_cast<const block_q8_0 *>(V_blob)
            + (size_t)s * n_kv_stride_blocks
            + kv * kv_head_stride_blocks
            + block_idx;
        const float v_f = (float)v_block->qs[lane] * __half2float(v_block->d);

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h = kv * n_q_per_kv + q;
                const float raw = scores[(size_t)h * max_seq_padded + s] * scale;
                const float p = __expf(raw - s_max[q]) * s_inv_denom[q];
                acc[q] = fmaf(p, v_f, acc[q]);
            }
        }
    }

    __shared__ float shmem[ATTN_SOFTMAX_OUTPUT_DEV_POS_WARPS][32];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total = sum_down_warp_column(shmem, warp_id, lane, acc[q]);
            if (warp_id == 0) {
                const int h = kv * n_q_per_kv + q;
                const int d = block_idx * 32 + lane;
                out[(size_t)h * HD + d] = total;
            }
            __syncthreads();
        }
    }
}

#define ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(NAME, HD, NQ)                  \
extern "C" __global__ void NAME(                                             \
    const void * __restrict__ V_blob,                                        \
    const float * __restrict__ scores,                                       \
    float * __restrict__ out,                                                \
    const int32_t * __restrict__ seq_kv_dev,                                 \
    int max_seq_padded,                                                      \
    int n_kv_stride_blocks,                                                  \
    int kv_head_stride_blocks,                                               \
    float scale,                                                             \
    int window                                                               \
) {                                                                          \
    const int block_idx = blockIdx.x;                                        \
    const int kv        = blockIdx.y;                                        \
    const int lane      = threadIdx.x;                                       \
    const int warp_id   = threadIdx.y;                                       \
    if (block_idx >= (HD / 32)) return;                                      \
    attn_softmax_output_q8_inner_dev_pos<HD, NQ>(                            \
        V_blob, scores, out, seq_kv_dev,                                     \
        max_seq_padded,                                                      \
        n_kv_stride_blocks, kv_head_stride_blocks,                           \
        scale, NQ,                                                           \
        block_idx, kv, lane, warp_id, window                                  \
    );                                                                       \
}

ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd64_nq1,    64, 1)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd64_nq2,    64, 2)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd64_nq4,    64, 4)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd64_nq5,    64, 5)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd64_nq8,    64, 8)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd128_nq1, 128, 1)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd128_nq2, 128, 2)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd128_nq4, 128, 4)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd128_nq5, 128, 5)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd128_nq8, 128, 8)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd256_nq1, 256, 1)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd256_nq2, 256, 2)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd256_nq4, 256, 4)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd256_nq5, 256, 5)
ATTN_SOFTMAX_OUTPUT_Q8_DEV_POS_KERNEL(attn_softmax_output_q8_0_f32_dev_pos_hd256_nq8, 256, 8)

// ------------------------------------------------------------
// Fully fused single-query attention for Q8 KV cache (phi2-class arches).
//
// Combines Q.K^T + scale + online-softmax + V.attn into ONE kernel launch
// per layer - replaces the existing 2-launch chain
// (attn_score_q8_0_f32_dev_pos + attn_softmax_output_q8_0_f32_dev_pos).
//
// Why: the 2-launch chain materialises a [n_q_heads x max_seq_padded] F32
// scores tensor (256 KB for head_dim 64 at 4 K ctx). The intermediate
// allocation + write + read + free shows up in the per-layer time budget
// as ~5-10 µs. Fusing keeps scores in registers (per-thread) and
// uses FlashAttention's online-softmax recurrence to fold V.attn inline.
//
// Launch contract:
//   blockDim  = (32, ATTN_FUSED_WARPS, 1)
//   gridDim   = (1, n_q_heads, 1)   -- one block per query head
//   shared mem = HD x sizeof(float) (for broadcasting per-token scores)
//
// One block handles all `seq_kv` positions for ONE query head. The block's
// HD threads (1 per head_dim element) each:
//   - hold Q[h_q][my_d] in a register
//   - accumulate VKQ[my_d] over seq_kv iterations
//   - track running max + running sum for softmax normalisation
//
// At seq_kv end, each thread writes out[h_q][my_d] = VKQ[my_d] / sum.
//
// Restricted to n_q_per_kv=1 (no GQA broadcast). Phi2's head layout
// satisfies this (n_head == n_kv_head == 32).
// ------------------------------------------------------------
#define ATTN_FUSED_WARPS 2   // 2 warps x 32 lanes = 64 threads = HD for phi2

template <int HD>
static __device__ __forceinline__ void attn_fused_q8_decode_inner_dev_pos(
    const void  * __restrict__ K_blocks,
    const void  * __restrict__ V_blocks,
    const float * __restrict__ Q,
    float       * __restrict__ out,
    const int32_t * __restrict__ seq_kv_dev,
    int max_seq_padded,
    int n_kv,
    float scale,
    int h_q,
    int tid,
    int lane,
    int warp_id
) {
    static_assert(HD == 64 || HD == 128, "HD must be 64 or 128 for fused kernel");

    const int seq_kv = seq_kv_dev[0] + 1;
    if (seq_kv <= 0) return;
    if (tid >= HD) return;

    // n_q_per_kv = 1: head_q == head_kv.
    const int h_kv = h_q;
    constexpr int HD_BLOCKS = HD / 32;

    // Load Q[h_q][tid] into register (single F32 per thread).
    const float q_my = Q[(size_t)h_q * HD + tid];

    // Online softmax state and VKQ accumulator (per-thread, per-d element).
    float vkq = 0.0f;
    float m   = -INFINITY;
    float s   =  0.0f;

    // Shared buffer to broadcast score from lane 0 -> all lanes in block.
    __shared__ float s_score;

    for (int t = 0; t < seq_kv; ++t) {
        // -- Q . K[t] --------------------------------------------------
        // Each lane reads its own (block-of-32, lane) element of K[h_kv][t].
        // Reduce 32 lanes within warp via __shfl_xor, then reduce across
        // warps via shared mem.
        const int b = tid / 32;   // HD_BLOCK index
        const int l = tid & 31;   // lane within block
        const size_t k_block_off =
            ((size_t)t * (size_t)n_kv + (size_t)h_kv) * (size_t)HD_BLOCKS
            + (size_t)b;
        const block_q8_0 * k_blk = reinterpret_cast<const block_q8_0 *>(K_blocks) + k_block_off;
        const float k_scale = __half2float(k_blk->d);
        const float k_val   = (float)k_blk->qs[l] * k_scale;
        float partial = q_my * k_val;

        // Warp-reduce across 32 lanes.
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            partial += __shfl_xor_sync(0xffffffff, partial, off);
        }
        // Lane 0 of each warp now holds the per-warp partial sum.
        // Reduce across warps via shared mem.
        __shared__ float warp_partials[ATTN_FUSED_WARPS];
        if (lane == 0) warp_partials[warp_id] = partial;
        __syncthreads();
        float score = 0.0f;
        if (warp_id == 0 && lane == 0) {
            #pragma unroll
            for (int w = 0; w < ATTN_FUSED_WARPS; ++w) {
                score += warp_partials[w];
            }
            s_score = score * scale;
        }
        __syncthreads();
        score = s_score;

        // -- Online softmax + V accumulation ----------------------------
        // __expf intrinsic: m_new = max(m,score) ensures both inputs to expf
        // are <= 0, so output is in (0, 1] - bounded range, safe ~2-ULP delta.
        const float m_new = fmaxf(m, score);
        const float scale_old = __expf(m - m_new);
        const float scale_new = __expf(score - m_new);
        s = s * scale_old + scale_new;

        // Load V[h_kv][t][tid] (F16 cache element).
        const size_t v_block_off =
            ((size_t)t * (size_t)n_kv + (size_t)h_kv) * (size_t)HD_BLOCKS
            + (size_t)b;
        const block_q8_0 * v_blk = reinterpret_cast<const block_q8_0 *>(V_blocks) + v_block_off;
        const float v_scale = __half2float(v_blk->d);
        const float v_val   = (float)v_blk->qs[l] * v_scale;

        vkq = vkq * scale_old + scale_new * v_val;
        m = m_new;
    }

    // Final normalisation. s > 0 since at least one score was valid.
    const float inv_s = (s > 0.0f) ? (1.0f / s) : 0.0f;
    out[(size_t)h_q * HD + tid] = vkq * inv_s;
}