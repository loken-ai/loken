// What the engine reaches directly beyond the passes above.

// ------------------------------------------------------------
// F-dtype (F16 KV) dev_pos analogues of the Q8 attention kernels above.
//
// Purpose: same launch contract as the Q8 dev_pos pair (attn_score +
// attn_softmax_output) but reads K/V directly as F16 instead of Q8 blocks.
// Targets head_dim 64, n_q_per_kv 1 models where kv_quant=Off keeps the
// cache in F-dtype - the standard_attention 4-launch path
// (matmul + affine + softmax + matmul) dominates per-layer time. This pair
// cuts that to 2 launches with the same online-softmax math.
//
// K/V layout in SpecKvCache: [b=1, n_kv, max_seq, HD] F16 contiguous.
//   K[h_kv][token][d] index = ((size_t)h_kv * max_seq + token) * HD + d
// The host call passes `max_seq_padded` = the cache's max_seq_len (full
// pre-allocated stride). `seq_kv_dev` follows the same +1 convention as
// the Q8 variant (host stores current_seq_len BEFORE append -> +1 here).
// ------------------------------------------------------------

template<int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_score_f16_inner_dev_pos(
    const __half * __restrict__ K,
    const float  * __restrict__ Q,
    float        * __restrict__ scores,
    const int32_t * __restrict__ seq_kv_dev,
    int max_seq_padded,
    int n_kv,
    int n_q_per_kv,
    int token,
    int h_kv,
    int lane
) {
    const int seq_kv = seq_kv_dev[0] + 1;

    if (token >= max_seq_padded) return;

    if (token >= seq_kv) {
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

    #pragma unroll
    for (int b = 0; b < HD_BLOCKS; ++b) {
        const size_t k_off =
            ((size_t)h_kv * (size_t)max_seq_padded + (size_t)token) * (size_t)HD
            + (size_t)b * 32 + (size_t)lane;
        const float k_val = __half2float(K[k_off]);

        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h_q = h_kv * n_q_per_kv + q;
                const float q_val = Q[(size_t)h_q * HD + b * 32 + lane];
                acc[q] = fmaf(q_val, k_val, acc[q]);
            }
        }
    }

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

#define ATTN_SCORE_F16_DEV_POS_KERNEL(NAME, HD, NQ)                           \
extern "C" __global__ void NAME(                                              \
    const __half * __restrict__ K,                                            \
    const float  * __restrict__ Q,                                            \
    float        * __restrict__ scores,                                       \
    const int32_t * __restrict__ seq_kv_dev,                                  \
    int max_seq_padded,                                                       \
    int n_kv,                                                                 \
    int /* n_q_per_kv */                                                      \
) {                                                                           \
    const int token = blockIdx.x * blockDim.y + threadIdx.y;                  \
    const int h_kv  = blockIdx.y;                                             \
    const int lane  = threadIdx.x;                                            \
    attn_score_f16_inner_dev_pos<HD, NQ>(                                     \
        K, Q, scores, seq_kv_dev,                                             \
        max_seq_padded, n_kv, NQ,                                             \
        token, h_kv, lane                                                     \
    );                                                                        \
}

ATTN_SCORE_F16_DEV_POS_KERNEL(attn_score_f16_f32_dev_pos_hd64_nq1,    64, 1)
ATTN_SCORE_F16_DEV_POS_KERNEL(attn_score_f16_f32_dev_pos_hd64_nq2,    64, 2)
ATTN_SCORE_F16_DEV_POS_KERNEL(attn_score_f16_f32_dev_pos_hd64_nq4,    64, 4)
ATTN_SCORE_F16_DEV_POS_KERNEL(attn_score_f16_f32_dev_pos_hd64_nq8,    64, 8)
ATTN_SCORE_F16_DEV_POS_KERNEL(attn_score_f16_f32_dev_pos_hd128_nq1, 128, 1)
ATTN_SCORE_F16_DEV_POS_KERNEL(attn_score_f16_f32_dev_pos_hd128_nq2, 128, 2)
ATTN_SCORE_F16_DEV_POS_KERNEL(attn_score_f16_f32_dev_pos_hd128_nq4, 128, 4)
ATTN_SCORE_F16_DEV_POS_KERNEL(attn_score_f16_f32_dev_pos_hd128_nq8, 128, 8)
ATTN_SCORE_F16_DEV_POS_KERNEL(attn_score_f16_f32_dev_pos_hd256_nq1, 256, 1)

/// Add up one partial per warp and hand every warp the total for its lane's position.
///
/// The warps of a block split the key positions between them, so each ends up holding part of
/// the same output element and the only way they can meet is a shared tile. Warp zero already
/// has its own partial in a register and keeps it there; the others post theirs, which is why
/// row zero of the tile is written by nobody and read by nobody. The sum then runs from row
/// one in warp order, so the additions land in the order the tile is laid out in.
///
/// Both barriers are load-bearing. The first orders the posting against the reading; the
/// second holds the readers until everyone is done, because the caller walks several output
/// elements through the same tile and the next one overwrites it.
template <int WARPS>
static __device__ __forceinline__ float sum_across_warps(
    float (&tile)[WARPS][32],
    float mine,
    int warp_id,
    int lane
) {
    if (warp_id != 0) {
        tile[warp_id][lane] = mine;
    }
    __syncthreads();

    float sum = mine;
    #pragma unroll
    for (int w = 1; w < WARPS; ++w) {
        sum += tile[w][lane];
    }
    __syncthreads();
    return sum;
}

template <int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_softmax_output_f16_inner_dev_pos(
    const __half * __restrict__ V,
    const float  * __restrict__ scores,
    float        * __restrict__ out,
    const int32_t * __restrict__ seq_kv_dev,
    int max_seq_padded,
    int n_kv,
    float scale,
    int n_q_per_kv,
    int block_idx,
    int kv,
    int lane,
    int warp_id
) {
    const int seq_kv = seq_kv_dev[0] + 1;

    __shared__ float s_max[MAX_NQ_PER_KV];
    __shared__ float s_inv_denom[MAX_NQ_PER_KV];

    if (warp_id == 0) {
        #pragma unroll
        for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
            if (q < n_q_per_kv) {
                const int h_q = kv * n_q_per_kv + q;
                const float * row = scores + (size_t)h_q * (size_t)max_seq_padded;
                float m_lane = -INFINITY;
                for (int t = lane; t < seq_kv; t += 32) {
                    m_lane = fmaxf(m_lane, row[t] * scale);
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    m_lane = fmaxf(m_lane, __shfl_xor_sync(0xFFFFFFFFu, m_lane, off));
                }
                const float m = m_lane;
                // __expf intrinsic - safe: input pre-subtracted by max so
                // <= 0, output bounded in (0, 1].
                float d_lane = 0.0f;
                for (int t = lane; t < seq_kv; t += 32) {
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

    float acc[MAX_NQ_PER_KV];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) acc[q] = 0.0f;

    for (int s = warp_id; s < seq_kv; s += ATTN_SOFTMAX_OUTPUT_DEV_POS_WARPS) {
        // V[kv][s][block_idx * 32 + lane]
        const size_t v_off =
            ((size_t)kv * (size_t)max_seq_padded + (size_t)s) * (size_t)HD
            + (size_t)block_idx * 32 + (size_t)lane;
        const float v_f = __half2float(V[v_off]);

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
            const float total = sum_across_warps(shmem, acc[q], warp_id, lane);
            if (warp_id == 0) {
                const int h = kv * n_q_per_kv + q;
                const int d = block_idx * 32 + lane;
                out[(size_t)h * HD + d] = total;
            }
        }
    }
}

#define ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(NAME, HD, NQ)                 \
extern "C" __global__ void NAME(                                             \
    const __half * __restrict__ V,                                           \
    const float  * __restrict__ scores,                                      \
    float        * __restrict__ out,                                         \
    const int32_t * __restrict__ seq_kv_dev,                                 \
    int max_seq_padded,                                                      \
    int n_kv,                                                                \
    float scale                                                              \
) {                                                                          \
    const int block_idx = blockIdx.x;                                        \
    const int kv        = blockIdx.y;                                        \
    const int lane      = threadIdx.x;                                       \
    const int warp_id   = threadIdx.y;                                       \
    if (block_idx >= (HD / 32)) return;                                      \
    attn_softmax_output_f16_inner_dev_pos<HD, NQ>(                           \
        V, scores, out, seq_kv_dev,                                          \
        max_seq_padded, n_kv,                                                \
        scale, NQ,                                                           \
        block_idx, kv, lane, warp_id                                         \
    );                                                                       \
}

ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(attn_softmax_output_f16_f32_dev_pos_hd64_nq1,    64, 1)
ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(attn_softmax_output_f16_f32_dev_pos_hd64_nq2,    64, 2)
ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(attn_softmax_output_f16_f32_dev_pos_hd64_nq4,    64, 4)
ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(attn_softmax_output_f16_f32_dev_pos_hd64_nq8,    64, 8)
ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(attn_softmax_output_f16_f32_dev_pos_hd128_nq1, 128, 1)
ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(attn_softmax_output_f16_f32_dev_pos_hd128_nq2, 128, 2)
ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(attn_softmax_output_f16_f32_dev_pos_hd128_nq4, 128, 4)
ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(attn_softmax_output_f16_f32_dev_pos_hd128_nq8, 128, 8)
ATTN_SOFTMAX_OUTPUT_F16_DEV_POS_KERNEL(attn_softmax_output_f16_f32_dev_pos_hd256_nq1, 256, 1)

ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd64_nq1,   64, 1)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd64_nq2,   64, 2)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd64_nq4,   64, 4)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd64_nq5,   64, 5)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd64_nq8,   64, 8)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd128_nq1, 128, 1)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd128_nq2, 128, 2)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd128_nq4, 128, 4)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd128_nq5, 128, 5)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd128_nq8, 128, 8)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd256_nq1, 256, 1)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd256_nq2, 256, 2)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd256_nq4, 256, 4)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd256_nq5, 256, 5)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd256_nq8, 256, 8)
// HD=512 - matching the attn_score_q4 HD=512 set added above.
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd512_nq1, 512, 1)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd512_nq2, 512, 2)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd512_nq4, 512, 4)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd512_nq5, 512, 5)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd512_nq8, 512, 8)
ATTN_OUTPUT_Q4_KERNEL(attn_output_q4_0_f32_hd512_nq16, 512, 16)

// ------------------------------------------------------------
// Device-position variant of attn_output_q4_*. Reads `seq_kv` from a
// device tensor instead of taking it as a host int, and reads `probs`
// at a fixed `max_seq_padded` stride (matches the
// `attn_score_q4_0_f32_kivi_dev_pos_*` scores output, which writes
// -INFINITY at positions >= seq_kv so softmax produces 0 there). Output
// `out` shape is [n_q_heads, HD] regardless of seq_kv.
// ------------------------------------------------------------
template<int HD, int MAX_NQ_PER_KV>
static __device__ __forceinline__ void attn_output_q4_inner_dev_pos(
    const void  * __restrict__ V_blob,
    const float * __restrict__ probs,
    float       * __restrict__ out,
    const int32_t * __restrict__ seq_kv_dev,
    int max_seq_padded,
    int n_kv_stride_blocks,
    int kv_head_stride_blocks,
    int n_q_per_kv,
    int block_idx, int kv, int lane, int warp_id
) {
    // See attn_score_q4_inner_dev_pos: host stores `current_seq_len BEFORE
    // the append`; attention sees the cache AFTER the append.
    const int seq_kv = seq_kv_dev[0] + 1;
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
                // probs uses max_seq_padded stride, not seq_kv
                const float p = probs[(size_t)h * (size_t)max_seq_padded + (size_t)s];
                acc[q] = fmaf(p, v_f, acc[q]);
            }
        }
    }

    __shared__ float shmem[ATTN_OUTPUT_WARPS][32];
    #pragma unroll
    for (int q = 0; q < MAX_NQ_PER_KV; ++q) {
        if (q < n_q_per_kv) {
            const float total = sum_across_warps(shmem, acc[q], warp_id, lane);
            if (warp_id == 0) {
                const int h = kv * n_q_per_kv + q;
                const int d = block_idx * 32 + lane;
                out[(size_t)h * HD + d] = total;
            }
        }
    }
}

#define ATTN_OUTPUT_Q4_DEV_POS_KERNEL(NAME, HD, NQ)                            \
extern "C" __global__ void NAME(                                               \
    const void  * __restrict__ V_blob,                                         \
    const float * __restrict__ probs,                                          \
    float       * __restrict__ out,                                            \
    const int32_t * __restrict__ seq_kv_dev,                                   \
    int max_seq_padded,                                                        \
    int n_kv_stride_blocks,                                                    \
    int kv_head_stride_blocks,                                                 \
    int /* n_q_per_kv */                                                       \
) {                                                                            \
    const int block_idx = blockIdx.x;                                          \
    const int kv        = blockIdx.y;                                          \
    const int lane      = threadIdx.x;                                         \
    const int warp_id   = threadIdx.y;                                         \
    if (block_idx >= (HD / 32)) return;                                        \
    attn_output_q4_inner_dev_pos<HD, NQ>(                                      \
        V_blob, probs, out, seq_kv_dev, max_seq_padded,                        \
        n_kv_stride_blocks, kv_head_stride_blocks, NQ,                         \
        block_idx, kv, lane, warp_id                                           \
    );                                                                         \
}

ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd64_nq1,    64, 1)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd64_nq2,    64, 2)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd64_nq4,    64, 4)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd64_nq5,    64, 5)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd64_nq8,    64, 8)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd128_nq1, 128, 1)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd128_nq2, 128, 2)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd128_nq4, 128, 4)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd128_nq5, 128, 5)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd128_nq8, 128, 8)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd256_nq1, 256, 1)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd256_nq2, 256, 2)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd256_nq4, 256, 4)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd256_nq5, 256, 5)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd256_nq8, 256, 8)
// HD=512 dev-pos output variants - graph-mode partner for the score kernel set.
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd512_nq1, 512, 1)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd512_nq2, 512, 2)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd512_nq4, 512, 4)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd512_nq5, 512, 5)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd512_nq8, 512, 8)
ATTN_OUTPUT_Q4_DEV_POS_KERNEL(attn_output_q4_0_f32_dev_pos_hd512_nq16, 512, 16)

// Half-precision input variant. Saves a promote+copy when the source is
// already in f16 (common for K/V after attention projection in f16 models).
extern "C" __global__ void quantize_q8_0_f16(const half * __restrict__ x, void * __restrict__ vy, const int kx, const int kx_padded) {
    const int ix = blockDim.x*blockIdx.x + threadIdx.x;

    if (ix >= kx_padded) {
        return;
    }

    const int iy = blockDim.y*blockIdx.y + threadIdx.y;

    const int i_padded = iy*kx_padded + ix;

    block_q8_0 * y = (block_q8_0 *) vy;

    const int ib = i_padded / QK8_0;
    const int iqs = i_padded % QK8_0;

    const float xi = ix < kx ? __half2float(x[iy*kx + ix]) : 0.0f;
    float amax = fabsf(xi);

    amax = warp_max(amax);

    const float d = amax / 127;
    const float id = amax == 0.0f ? 0.0f : 1.0f / d;
    const int8_t q = roundf(xi * id);

    y[ib].qs[iqs] = q;

    if (iqs == 0) {
        y[ib].d = d;
    }
}
