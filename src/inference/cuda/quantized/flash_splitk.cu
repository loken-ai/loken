// Flash attention with k split across blocks, and the pass that folds the partials back.

#define ATTN_FUSED_Q8_DEV_POS_KERNEL(NAME, HD)                                \
extern "C" __global__ void NAME(                                              \
    const void  * __restrict__ K_blocks,                                      \
    const void  * __restrict__ V_blocks,                                      \
    const float * __restrict__ Q,                                             \
    float       * __restrict__ out,                                           \
    const int32_t * __restrict__ seq_kv_dev,                                  \
    int max_seq_padded,                                                       \
    int n_kv,                                                                 \
    float scale                                                               \
) {                                                                           \
    const int h_q     = blockIdx.y;                                           \
    const int lane    = threadIdx.x;                                          \
    const int warp_id = threadIdx.y;                                          \
    const int tid     = warp_id * 32 + lane;                                  \
    attn_fused_q8_decode_inner_dev_pos<HD>(                                   \
        K_blocks, V_blocks, Q, out, seq_kv_dev,                               \
        max_seq_padded, n_kv, scale,                                          \
        h_q, tid, lane, warp_id                                               \
    );                                                                        \
}

ATTN_FUSED_Q8_DEV_POS_KERNEL(attn_fused_q8_decode_dev_pos_hd64,    64)
ATTN_FUSED_Q8_DEV_POS_KERNEL(attn_fused_q8_decode_dev_pos_hd128, 128)

// ------------------------------------------------------------
// SPLIT-K flash-decode for Q8_0 KV (dev_pos). The v3 fused kernel above is
// numerically correct but loses to the 2-kernel score+softmax_output chain
// (-2%) because it runs only n_heads (=32) blocks, each looping the FULL
// ~750-token KV serially with a per-token block reduction + 2 __syncthreads.
//
// Split-K fixes the parallelism: grid = (NSPLIT, n_heads) -> NSPLITx more
// blocks (e.g. 16x32 = 512), each a SINGLE WARP doing online-softmax over a
// short ~ceil(seq_kv/NSPLIT)-token chunk with pure warp-shuffle reductions
// (NO per-token __syncthreads). A second tiny combine kernel flash-merges
// the NSPLIT partials per head. Parallelism of the 2-kernel chain AND the
// fusion (no HBM scores) of v3.
//
// Partial layout: partials[h*NSPLIT + split] = (HD+2) f32 = [vkq[0..HD], m, l]
// (vkq exp-weighted by split-local max m, l = split-local denom). Combine:
// gm = max_split m_i; out = (Σ vkq_i.exp(m_i-gm)) / (Σ l_i.exp(m_i-gm)).
// ------------------------------------------------------------
#define FLASH_SPLITK_NSPLIT 16

template <int HD>
static __device__ __forceinline__ void flash_splitk_q8_partial_inner(
    const void  * __restrict__ K_blocks,
    const void  * __restrict__ V_blocks,
    const float * __restrict__ Q,
    float       * __restrict__ partials,
    const int32_t * __restrict__ seq_kv_dev,
    int n_kv,
    float scale,
    int h_q,
    int split,
    int lane
) {
    const int seq_kv = seq_kv_dev[0] + 1;
    constexpr int HD_BLOCKS = HD / 32;
    const int h_kv = h_q;                        // n_q_per_kv == 1
    const int chunk = (seq_kv + FLASH_SPLITK_NSPLIT - 1) / FLASH_SPLITK_NSPLIT;
    const int t0 = split * chunk;
    const int t1 = min(t0 + chunk, seq_kv);

    float * p = partials + ((size_t)h_q * FLASH_SPLITK_NSPLIT + split) * (HD + 2);

    if (t0 >= t1) {
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) p[b * 32 + lane] = 0.0f;
        if (lane == 0) { p[HD] = -INFINITY; p[HD + 1] = 0.0f; }
        return;
    }

    float q_reg[HD_BLOCKS];
    #pragma unroll
    for (int b = 0; b < HD_BLOCKS; ++b) q_reg[b] = Q[(size_t)h_q * HD + b * 32 + lane];

    float m = -INFINITY, l = 0.0f;
    float vkq[HD_BLOCKS];
    #pragma unroll
    for (int b = 0; b < HD_BLOCKS; ++b) vkq[b] = 0.0f;

    for (int t = t0; t < t1; ++t) {
        const block_q8_0 * kb = reinterpret_cast<const block_q8_0 *>(K_blocks)
            + ((size_t)t * n_kv + h_kv) * HD_BLOCKS;
        float dot = 0.0f;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) {
            const block_q8_0 * kbb = kb + b;
            dot += q_reg[b] * ((float)kbb->qs[lane] * __half2float(kbb->d));
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        const float score = dot * scale;

        const float m_new = fmaxf(m, score);
        const float so = __expf(m - m_new);
        const float sn = __expf(score - m_new);
        l = l * so + sn;

        const block_q8_0 * vb = reinterpret_cast<const block_q8_0 *>(V_blocks)
            + ((size_t)t * n_kv + h_kv) * HD_BLOCKS;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) {
            const block_q8_0 * vbb = vb + b;
            const float vv = (float)vbb->qs[lane] * __half2float(vbb->d);
            vkq[b] = vkq[b] * so + sn * vv;
        }
        m = m_new;
    }

    #pragma unroll
    for (int b = 0; b < HD_BLOCKS; ++b) p[b * 32 + lane] = vkq[b];
    if (lane == 0) { p[HD] = m; p[HD + 1] = l; }
}

template <int HD>
static __device__ __forceinline__ void flash_splitk_q8_combine_inner(
    const float * __restrict__ partials,
    float       * __restrict__ out,
    int h_q,
    int lane
) {
    constexpr int HD_BLOCKS = HD / 32;
    const float * base = partials + (size_t)h_q * FLASH_SPLITK_NSPLIT * (HD + 2);

    float gm = -INFINITY;
    #pragma unroll
    for (int sp = 0; sp < FLASH_SPLITK_NSPLIT; ++sp) {
        gm = fmaxf(gm, base[(size_t)sp * (HD + 2) + HD]);
    }

    float gl = 0.0f;
    float acc[HD_BLOCKS];
    #pragma unroll
    for (int b = 0; b < HD_BLOCKS; ++b) acc[b] = 0.0f;

    for (int sp = 0; sp < FLASH_SPLITK_NSPLIT; ++sp) {
        const float * p = base + (size_t)sp * (HD + 2);
        const float li = p[HD + 1];
        if (li <= 0.0f) continue;
        const float w = __expf(p[HD] - gm);
        gl += li * w;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) acc[b] += p[b * 32 + lane] * w;
    }

    const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    #pragma unroll
    for (int b = 0; b < HD_BLOCKS; ++b) out[(size_t)h_q * HD + b * 32 + lane] = acc[b] * inv;
}

#define FLASH_SPLITK_PARTIAL_KERNEL(NAME, HD)                                 \
extern "C" __global__ void NAME(                                              \
    const void  * __restrict__ K_blocks,                                      \
    const void  * __restrict__ V_blocks,                                      \
    const float * __restrict__ Q,                                             \
    float       * __restrict__ partials,                                      \
    const int32_t * __restrict__ seq_kv_dev,                                  \
    int n_kv,                                                                 \
    float scale                                                               \
) {                                                                           \
    flash_splitk_q8_partial_inner<HD>(                                        \
        K_blocks, V_blocks, Q, partials, seq_kv_dev,                          \
        n_kv, scale, blockIdx.y, blockIdx.x, threadIdx.x                      \
    );                                                                        \
}

#define FLASH_SPLITK_COMBINE_KERNEL(NAME, HD)                                 \
extern "C" __global__ void NAME(                                              \
    const float * __restrict__ partials,                                      \
    float       * __restrict__ out                                            \
) {                                                                           \
    flash_splitk_q8_combine_inner<HD>(partials, out, blockIdx.x, threadIdx.x);\
}

FLASH_SPLITK_PARTIAL_KERNEL(flash_splitk_q8_partial_hd64,  64)
FLASH_SPLITK_PARTIAL_KERNEL(flash_splitk_q8_partial_hd128, 128)
FLASH_SPLITK_COMBINE_KERNEL(flash_splitk_q8_combine_hd64,  64)
FLASH_SPLITK_COMBINE_KERNEL(flash_splitk_q8_combine_hd128, 128)

// ------------------------------------------------------------
// GQA split-K partial with RUNTIME-ADAPTIVE nsplit. The MHA partial above
// re-reads K/V once PER QUERY HEAD (h_kv == h_q); for GQA (n_q_per_kv query
// heads share one KV head) that wastes n_q_per_kvx of the KV bandwidth. Here
// one warp per (split, h_KV) reads its KV head's chunk ONCE and feeds all
// n_q_per_kv query heads from registers (KV read n_kv_headsx, the minimum).
//
// nsplit is a RUNTIME arg, not a compile constant: GQA has few KV heads
// (4-8), so a fixed nsplit=16 gives only 16xn_kv_heads = 64-128 single-warp
// blocks -> underfills the GPU (occupancy-limited, the reason the fixed version
// lost -9%). The launcher picks nsplit to target ~256-512 blocks regardless of
// n_kv_heads. Partials are laid out [h_q*nsplit + split]; the matching combine
// loops nsplit. MAX_NQ bounds per-head registers; runtime n_q_per_kv <= MAX_NQ.
// Grid: (nsplit, n_kv_heads), block (32,1).
template <int HD, int MAX_NQ>
static __device__ __forceinline__ void flash_splitk_q8_gqa_partial_inner(
    const void  * __restrict__ K_blocks,
    const void  * __restrict__ V_blocks,
    const float * __restrict__ Q,
    float       * __restrict__ partials,
    const int32_t * __restrict__ seq_kv_dev,
    int n_kv,
    int n_q_per_kv,
    int nsplit,
    float scale,
    int h_kv,
    int split,
    int lane
) {
    const int seq_kv = seq_kv_dev[0] + 1;
    constexpr int HD_BLOCKS = HD / 32;
    const int chunk = (seq_kv + nsplit - 1) / nsplit;
    const int t0 = split * chunk;
    const int t1 = min(t0 + chunk, seq_kv);

    if (t0 >= t1) {
        for (int q = 0; q < MAX_NQ; ++q) {
            if (q >= n_q_per_kv) break;
            const int h_q = h_kv * n_q_per_kv + q;
            float * p = partials + ((size_t)h_q * nsplit + split) * (HD + 2);
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) p[b * 32 + lane] = 0.0f;
            if (lane == 0) { p[HD] = -INFINITY; p[HD + 1] = 0.0f; }
        }
        return;
    }

    float q_reg[MAX_NQ][HD_BLOCKS];
    #pragma unroll
    for (int q = 0; q < MAX_NQ; ++q) {
        if (q >= n_q_per_kv) break;
        const int h_q = h_kv * n_q_per_kv + q;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b)
            q_reg[q][b] = Q[(size_t)h_q * HD + b * 32 + lane];
    }

    float m[MAX_NQ], l[MAX_NQ], vkq[MAX_NQ][HD_BLOCKS];
    #pragma unroll
    for (int q = 0; q < MAX_NQ; ++q) {
        m[q] = -INFINITY; l[q] = 0.0f;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) vkq[q][b] = 0.0f;
    }

    for (int t = t0; t < t1; ++t) {
        const block_q8_0 * kb = reinterpret_cast<const block_q8_0 *>(K_blocks)
            + ((size_t)t * n_kv + h_kv) * HD_BLOCKS;
        const block_q8_0 * vb = reinterpret_cast<const block_q8_0 *>(V_blocks)
            + ((size_t)t * n_kv + h_kv) * HD_BLOCKS;
        float kval[HD_BLOCKS], vval[HD_BLOCKS];
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) {
            const block_q8_0 * kbb = kb + b;
            kval[b] = (float)kbb->qs[lane] * __half2float(kbb->d);
            const block_q8_0 * vbb = vb + b;
            vval[b] = (float)vbb->qs[lane] * __half2float(vbb->d);
        }
        #pragma unroll
        for (int q = 0; q < MAX_NQ; ++q) {
            if (q >= n_q_per_kv) break;
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) dot += q_reg[q][b] * kval[b];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
            const float score = dot * scale;
            const float m_new = fmaxf(m[q], score);
            const float so = __expf(m[q] - m_new);
            const float sn = __expf(score - m_new);
            l[q] = l[q] * so + sn;
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) vkq[q][b] = vkq[q][b] * so + sn * vval[b];
            m[q] = m_new;
        }
    }

    #pragma unroll
    for (int q = 0; q < MAX_NQ; ++q) {
        if (q >= n_q_per_kv) break;
        const int h_q = h_kv * n_q_per_kv + q;
        float * p = partials + ((size_t)h_q * nsplit + split) * (HD + 2);
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) p[b * 32 + lane] = vkq[q][b];
        if (lane == 0) { p[HD] = m[q]; p[HD + 1] = l[q]; }
    }
}

// Combine with runtime nsplit (per query head merges its nsplit partials).
// CW warps cooperate: each warp online-merges a STRIDED subset of the nsplit
// partials into a local (m,l,vkq), then warp 0 merges the CW locals via shared
// memory. With the adaptive launcher pushing nsplit toward 256 for long context,
// a single-warp serial loop over nsplit became the bottleneck (~0.1 µs/split);
// CW=8 cuts that ~8x so the higher nsplit (which makes the partial kernel faster
// via occupancy) is a net win. Merge math is the associative log-sum-exp combine,
// so the result is unchanged vs the serial version (FP reassociated).
#define FLASH_SPLITK_COMBINE_WARPS 8
template <int HD>
static __device__ __forceinline__ void flash_splitk_gqa_combine_inner(
    const float * __restrict__ partials,
    float       * __restrict__ out,
    int nsplit,
    int h_q,
    int lane,
    int warp
) {
    constexpr int HD_BLOCKS = HD / 32;
    constexpr int CW = FLASH_SPLITK_COMBINE_WARPS;
    const float * base = partials + (size_t)h_q * nsplit * (HD + 2);

    // Per-warp online merge over a strided subset of splits.
    float lm = -INFINITY, ll = 0.0f;
    float lacc[HD_BLOCKS];
    #pragma unroll
    for (int b = 0; b < HD_BLOCKS; ++b) lacc[b] = 0.0f;
    for (int sp = warp; sp < nsplit; sp += CW) {
        const float * p = base + (size_t)sp * (HD + 2);
        const float li = p[HD + 1];
        if (li <= 0.0f) continue;
        const float mi = p[HD];
        const float m_new = fmaxf(lm, mi);
        const float so = __expf(lm - m_new);
        const float sn = __expf(mi - m_new);
        ll = ll * so + sn * li;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) lacc[b] = lacc[b] * so + sn * p[b * 32 + lane];
        lm = m_new;
    }

    __shared__ float smax[CW];
    __shared__ float sl[CW];
    __shared__ float sacc[CW * HD];
    if (lane == 0) { smax[warp] = lm; sl[warp] = ll; }
    #pragma unroll
    for (int b = 0; b < HD_BLOCKS; ++b) sacc[warp * HD + b * 32 + lane] = lacc[b];
    __syncthreads();

    // Warp 0 merges the CW per-warp locals and writes the output.
    if (warp == 0) {
        float gm = -INFINITY;
        #pragma unroll
        for (int w = 0; w < CW; ++w) gm = fmaxf(gm, smax[w]);
        float gl = 0.0f;
        float acc[HD_BLOCKS];
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) acc[b] = 0.0f;
        #pragma unroll
        for (int w = 0; w < CW; ++w) {
            const float weight = (smax[w] == -INFINITY) ? 0.0f : __expf(smax[w] - gm);
            gl += sl[w] * weight;
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) acc[b] += sacc[w * HD + b * 32 + lane] * weight;
        }
        const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) out[(size_t)h_q * HD + b * 32 + lane] = acc[b] * inv;
    }
}

// -- Register software-pipelined twin of flash_splitk_q8_gqa_partial ----------
// The kernel is COMPUTE-bound (69%, per-q warp-shuffle reductions) with a fast,
// overlappable 31% KV-load (measured, load-only twin). cp.async can't
// cleanly stage block_q8_0 (34 B stride -> not 4/8/16-B aligned), so overlap the
// load the simplest way: issue token t+1's global loads into registers BEFORE
// computing token t, so their memory latency hides behind the t-compute shuffles.
// Costs +HD_BLOCKSx2 registers (knext/vnext) - may hit the occupancy cliff; the
// microbench A/B decides. Numerics identical to the base kernel (same math order).
template <int HD, int MAX_NQ>
static __device__ __forceinline__ void flash_splitk_q8_gqa_partial_pf_inner(
    const void  * __restrict__ K_blocks,
    const void  * __restrict__ V_blocks,
    const float * __restrict__ Q,
    float       * __restrict__ partials,
    const int32_t * __restrict__ seq_kv_dev,
    int n_kv, int n_q_per_kv, int nsplit, float scale,
    int h_kv, int split, int lane
) {
    const int seq_kv = seq_kv_dev[0] + 1;
    constexpr int HD_BLOCKS = HD / 32;
    const int chunk = (seq_kv + nsplit - 1) / nsplit;
    const int t0 = split * chunk;
    const int t1 = min(t0 + chunk, seq_kv);
    if (t0 >= t1) {
        for (int q = 0; q < MAX_NQ; ++q) {
            if (q >= n_q_per_kv) break;
            const int h_q = h_kv * n_q_per_kv + q;
            float * p = partials + ((size_t)h_q * nsplit + split) * (HD + 2);
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) p[b * 32 + lane] = 0.0f;
            if (lane == 0) { p[HD] = -INFINITY; p[HD + 1] = 0.0f; }
        }
        return;
    }
    float q_reg[MAX_NQ][HD_BLOCKS];
    #pragma unroll
    for (int q = 0; q < MAX_NQ; ++q) {
        if (q >= n_q_per_kv) break;
        const int h_q = h_kv * n_q_per_kv + q;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) q_reg[q][b] = Q[(size_t)h_q * HD + b * 32 + lane];
    }
    float m[MAX_NQ], l[MAX_NQ], vkq[MAX_NQ][HD_BLOCKS];
    #pragma unroll
    for (int q = 0; q < MAX_NQ; ++q) {
        m[q] = -INFINITY; l[q] = 0.0f;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) vkq[q][b] = 0.0f;
    }
    // Prologue: load token t0.
    float kcur[HD_BLOCKS], vcur[HD_BLOCKS];
    {
        const block_q8_0 * kb = reinterpret_cast<const block_q8_0 *>(K_blocks) + ((size_t)t0 * n_kv + h_kv) * HD_BLOCKS;
        const block_q8_0 * vb = reinterpret_cast<const block_q8_0 *>(V_blocks) + ((size_t)t0 * n_kv + h_kv) * HD_BLOCKS;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) {
            kcur[b] = (float)kb[b].qs[lane] * __half2float(kb[b].d);
            vcur[b] = (float)vb[b].qs[lane] * __half2float(vb[b].d);
        }
    }
    for (int t = t0; t < t1; ++t) {
        float knext[HD_BLOCKS], vnext[HD_BLOCKS];
        const bool has_next = (t + 1 < t1);
        if (has_next) { // issue next token's loads EARLY (overlap latency w/ compute)
            const block_q8_0 * kb = reinterpret_cast<const block_q8_0 *>(K_blocks) + ((size_t)(t + 1) * n_kv + h_kv) * HD_BLOCKS;
            const block_q8_0 * vb = reinterpret_cast<const block_q8_0 *>(V_blocks) + ((size_t)(t + 1) * n_kv + h_kv) * HD_BLOCKS;
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) {
                knext[b] = (float)kb[b].qs[lane] * __half2float(kb[b].d);
                vnext[b] = (float)vb[b].qs[lane] * __half2float(vb[b].d);
            }
        }
        #pragma unroll
        for (int q = 0; q < MAX_NQ; ++q) {
            if (q >= n_q_per_kv) break;
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) dot += q_reg[q][b] * kcur[b];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
            const float score = dot * scale;
            const float m_new = fmaxf(m[q], score);
            const float so = __expf(m[q] - m_new);
            const float sn = __expf(score - m_new);
            l[q] = l[q] * so + sn;
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) vkq[q][b] = vkq[q][b] * so + sn * vcur[b];
            m[q] = m_new;
        }
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) { kcur[b] = knext[b]; vcur[b] = vnext[b]; }
    }
    #pragma unroll
    for (int q = 0; q < MAX_NQ; ++q) {
        if (q >= n_q_per_kv) break;
        const int h_q = h_kv * n_q_per_kv + q;
        float * p = partials + ((size_t)h_q * nsplit + split) * (HD + 2);
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) p[b * 32 + lane] = vkq[q][b];
        if (lane == 0) { p[HD] = m[q]; p[HD + 1] = l[q]; }
    }
}

#define FLASH_SPLITK_GQA_PARTIAL_KERNEL(NAME, HD, MAXNQ)                       \
extern "C" __global__ void NAME(                                              \
    const void  * __restrict__ K_blocks,                                      \
    const void  * __restrict__ V_blocks,                                      \
    const float * __restrict__ Q,                                             \
    float       * __restrict__ partials,                                      \
    const int32_t * __restrict__ seq_kv_dev,                                  \
    int n_kv,                                                                 \
    int n_q_per_kv,                                                           \
    int nsplit,                                                               \
    float scale                                                               \
) {                                                                           \
    flash_splitk_q8_gqa_partial_inner<HD, MAXNQ>(                             \
        K_blocks, V_blocks, Q, partials, seq_kv_dev,                          \
        n_kv, n_q_per_kv, nsplit, scale, blockIdx.y, blockIdx.x, threadIdx.x  \
    );                                                                        \
}

#define FLASH_SPLITK_GQA_COMBINE_KERNEL(NAME, HD)                             \
extern "C" __global__ void NAME(                                             \
    const float * __restrict__ partials,                                     \
    float       * __restrict__ out,                                          \
    int nsplit                                                               \
) {                                                                          \
    flash_splitk_gqa_combine_inner<HD>(partials, out, nsplit, blockIdx.x, threadIdx.x, threadIdx.y); \
}

// -- Diagnostic: LOAD-ONLY twin of flash_splitk_q8_gqa_partial ----------------
// Reads EXACTLY the same K+V block_q8_0 bytes with the SAME grid/lane access
// pattern as the partial kernel, but does NO dot / warp-shuffle / online-softmax  - 
// just dequant-accumulates into a register and writes one float (DCE guard). Its
// wall-clock = the pure KV-load half of the real kernel. Comparing it to the full
// kernel decomposes load vs compute WITHOUT ncu: if load-only ≈ ½ of full, the
// load and compute are serialized and a cp.async load↔compute overlap could ~halve
// the kernel (justifies the rewrite); if load-only ≈ full, it is bandwidth-bound
// (cp.async cannot help -> accept the floor). Resolves the ncu gate reboot-free.
template <int HD>
static __device__ __forceinline__ void flash_splitk_q8_loadonly_inner(
    const void  * __restrict__ K_blocks,
    const void  * __restrict__ V_blocks,
    float       * __restrict__ partials,
    const int32_t * __restrict__ seq_kv_dev,
    int n_kv,
    int nsplit,
    int h_kv,
    int split,
    int lane
) {
    const int seq_kv = seq_kv_dev[0] + 1;
    constexpr int HD_BLOCKS = HD / 32;
    const int chunk = (seq_kv + nsplit - 1) / nsplit;
    const int t0 = split * chunk;
    const int t1 = min(t0 + chunk, seq_kv);
    float acc = 0.0f;
    for (int t = t0; t < t1; ++t) {
        const block_q8_0 * kb = reinterpret_cast<const block_q8_0 *>(K_blocks)
            + ((size_t)t * n_kv + h_kv) * HD_BLOCKS;
        const block_q8_0 * vb = reinterpret_cast<const block_q8_0 *>(V_blocks)
            + ((size_t)t * n_kv + h_kv) * HD_BLOCKS;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) {
            acc += (float)kb[b].qs[lane] * __half2float(kb[b].d);
            acc += (float)vb[b].qs[lane] * __half2float(vb[b].d);
        }
    }
    partials[((size_t)h_kv * nsplit + split) * 32 + lane] = acc;
}

#define FLASH_SPLITK_LOADONLY_KERNEL(NAME, HD)                                 \
extern "C" __global__ void NAME(                                              \
    const void  * __restrict__ K_blocks,                                      \
    const void  * __restrict__ V_blocks,                                      \
    float       * __restrict__ partials,                                      \
    const int32_t * __restrict__ seq_kv_dev,                                  \
    int n_kv,                                                                 \
    int nsplit                                                                \
) {                                                                           \
    flash_splitk_q8_loadonly_inner<HD>(                                       \
        K_blocks, V_blocks, partials, seq_kv_dev,                             \
        n_kv, nsplit, blockIdx.y, blockIdx.x, threadIdx.x                     \
    );                                                                        \
}
FLASH_SPLITK_LOADONLY_KERNEL(flash_splitk_q8_loadonly_hd64,  64)
FLASH_SPLITK_LOADONLY_KERNEL(flash_splitk_q8_loadonly_hd128, 128)

#define FLASH_SPLITK_GQA_PARTIAL_PF_KERNEL(NAME, HD, MAXNQ)                    \
extern "C" __global__ void NAME(                                              \
    const void  * __restrict__ K_blocks,                                      \
    const void  * __restrict__ V_blocks,                                      \
    const float * __restrict__ Q,                                             \
    float       * __restrict__ partials,                                      \
    const int32_t * __restrict__ seq_kv_dev,                                  \
    int n_kv, int n_q_per_kv, int nsplit, float scale                         \
) {                                                                           \
    flash_splitk_q8_gqa_partial_pf_inner<HD, MAXNQ>(                          \
        K_blocks, V_blocks, Q, partials, seq_kv_dev,                          \
        n_kv, n_q_per_kv, nsplit, scale, blockIdx.y, blockIdx.x, threadIdx.x  \
    );                                                                        \
}

FLASH_SPLITK_GQA_PARTIAL_KERNEL(flash_splitk_q8_gqa_partial_hd64,  64,  8)
FLASH_SPLITK_GQA_PARTIAL_KERNEL(flash_splitk_q8_gqa_partial_hd128, 128, 8)
FLASH_SPLITK_GQA_PARTIAL_PF_KERNEL(flash_splitk_q8_gqa_partial_pf_hd64,  64,  8)
FLASH_SPLITK_GQA_PARTIAL_PF_KERNEL(flash_splitk_q8_gqa_partial_pf_hd128, 128, 8)
FLASH_SPLITK_GQA_COMBINE_KERNEL(flash_splitk_gqa_combine_hd64,  64)
FLASH_SPLITK_GQA_COMBINE_KERNEL(flash_splitk_gqa_combine_hd128, 128)

// -- WIDE combine: one block per (query head, 32-dim slice) -------------------
// The single-block-per-head combine above serially re-reads all HD dims of each
// partial per warp; at the long-context nsplit (~140-256) its 16-block grid
// (n_q_heads) leaves the GPU idle and it costs more than the partial saves
// (nsys: 5.8 µs vs 2.5 µs at nsplit 48). Splitting the grid over
// HD/32 slices gives HD_BLOCKSx the blocks; each warp's stride loop only
// touches its 32-dim slice (plus the 2 (m,l) floats, re-read per slice  - 
// negligible). Same associative log-sum-exp merge, same result.
template <int HD>
static __device__ __forceinline__ void flash_splitk_gqa_combine_wide_inner(
    const float * __restrict__ partials,
    float       * __restrict__ out,
    int nsplit,
    int h_q,
    int hb,     // which 32-dim slice of the head
    int lane,
    int warp
) {
    constexpr int CW = FLASH_SPLITK_COMBINE_WARPS;
    const float * base = partials + (size_t)h_q * nsplit * (HD + 2);

    // Per-warp online merge over a strided subset of splits, 32-dim slice only.
    float lm = -INFINITY, ll = 0.0f, lacc = 0.0f;
    for (int sp = warp; sp < nsplit; sp += CW) {
        const float * p = base + (size_t)sp * (HD + 2);
        const float li = p[HD + 1];
        if (li <= 0.0f) continue;
        const float mi = p[HD];
        const float m_new = fmaxf(lm, mi);
        const float so = __expf(lm - m_new);
        const float sn = __expf(mi - m_new);
        ll = ll * so + sn * li;
        lacc = lacc * so + sn * p[hb * 32 + lane];
        lm = m_new;
    }

    __shared__ float smax[CW];
    __shared__ float sl[CW];
    __shared__ float sacc[CW * 32];
    if (lane == 0) { smax[warp] = lm; sl[warp] = ll; }
    sacc[warp * 32 + lane] = lacc;
    __syncthreads();

    if (warp == 0) {
        float gm = -INFINITY;
        #pragma unroll
        for (int w = 0; w < CW; ++w) gm = fmaxf(gm, smax[w]);
        float gl = 0.0f, acc = 0.0f;
        #pragma unroll
        for (int w = 0; w < CW; ++w) {
            const float weight = (smax[w] == -INFINITY) ? 0.0f : __expf(smax[w] - gm);
            gl += sl[w] * weight;
            acc += sacc[w * 32 + lane] * weight;
        }
        const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
        out[(size_t)h_q * HD + hb * 32 + lane] = acc * inv;
    }
}

#define FLASH_SPLITK_GQA_COMBINE_WIDE_KERNEL(NAME, HD)                        \
extern "C" __global__ void NAME(                                              \
    const float * __restrict__ partials,                                      \
    float       * __restrict__ out,                                           \
    int nsplit                                                                \
) {                                                                           \
    flash_splitk_gqa_combine_wide_inner<HD>(partials, out, nsplit,            \
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);                    \
}
FLASH_SPLITK_GQA_COMBINE_WIDE_KERNEL(flash_splitk_gqa_combine_wide_hd64,  64)
FLASH_SPLITK_GQA_COMBINE_WIDE_KERNEL(flash_splitk_gqa_combine_wide_hd128, 128)

// ------------------------------------------------------------
// Q4_0 KIVI GQA split-K partial - fused flash-decode for the qwen3moe Q4 KV
// cache. Same online-softmax + adaptive-nsplit structure as the Q8 gqa partial
// (lane = hd-dim, loop tokens, warp-reduce dot, accumulate vkq) and the SAME
// runtime-nsplit combine - only the K/V reads differ because the Q4 cache uses
// a KIVI layout: K is PER-CHANNEL (K_blocks[seq_block][h_kv][channel], each a
// block_q4_0 of 32 tokens; the partial last block lives in F16 K_residual
// [h_kv][channel][32]); V is per-token Q4_0 (V_blob[token][h_kv][hd_block]).
// This replaces the 2-kernel attn_score(->HBM scores)+attn_softmax_output chain
// with one fused pass (no HBM scores round-trip). seq_kv is a host int (the Q4
// path is not graph-captured). Grid: (nsplit, n_kv_heads), block (32,1).
template <int HD, int MAX_NQ>
static __device__ __forceinline__ void flash_splitk_q4_gqa_partial_inner(
    const void  * __restrict__ K_blocks,   // [seq_block, n_kv, HD] q4_0 (per-channel)
    const half  * __restrict__ K_residual, // [n_kv, HD, 32] f16 (partial last block)
    const void  * __restrict__ V_blob,     // [seq, n_kv, HD/32] q4_0 (per-token)
    const float * __restrict__ Q,          // [n_q, HD] f32
    float       * __restrict__ partials,
    int seq_kv,
    int full_blocks,
    int n_kv,
    int n_q_per_kv,
    int nsplit,
    float scale,
    int h_kv,
    int split,
    int lane
) {
    constexpr int HD_BLOCKS = HD / 32;
    const int v_n_kv_stride   = n_kv * HD_BLOCKS; // V per-token block stride
    const int v_kv_head_stride = HD_BLOCKS;
    const int chunk = (seq_kv + nsplit - 1) / nsplit;
    const int t0 = split * chunk;
    const int t1 = min(t0 + chunk, seq_kv);

    if (t0 >= t1) {
        for (int q = 0; q < MAX_NQ; ++q) {
            if (q >= n_q_per_kv) break;
            const int h_q = h_kv * n_q_per_kv + q;
            float * p = partials + ((size_t)h_q * nsplit + split) * (HD + 2);
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) p[b * 32 + lane] = 0.0f;
            if (lane == 0) { p[HD] = -INFINITY; p[HD + 1] = 0.0f; }
        }
        return;
    }

    float q_reg[MAX_NQ][HD_BLOCKS];
    #pragma unroll
    for (int q = 0; q < MAX_NQ; ++q) {
        if (q >= n_q_per_kv) break;
        const int h_q = h_kv * n_q_per_kv + q;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b)
            q_reg[q][b] = Q[(size_t)h_q * HD + b * 32 + lane];
    }

    float m[MAX_NQ], l[MAX_NQ], vkq[MAX_NQ][HD_BLOCKS];
    #pragma unroll
    for (int q = 0; q < MAX_NQ; ++q) {
        m[q] = -INFINITY; l[q] = 0.0f;
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) vkq[q][b] = 0.0f;
    }

    for (int t = t0; t < t1; ++t) {
        const int sb  = t >> 5;
        const int tib = t & 31;
        const int knb_byte = tib & 15;
        const bool k_high = tib >= 16;
        // K[t][h_kv][c] for the HD_BLOCKS channels c = b*32+lane (KIVI).
        float kval[HD_BLOCKS];
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) {
            const int c = b * 32 + lane;
            float kv;
            if (sb < full_blocks) {
                const size_t bidx = (size_t)sb * ((size_t)n_kv * HD)
                                  + (size_t)h_kv * HD + (size_t)c;
                const block_q4_0 * kb = reinterpret_cast<const block_q4_0 *>(K_blocks) + bidx;
                const uint8_t byte = kb->qs[knb_byte];
                const int nib = k_high ? (byte >> 4) : (byte & 0xF);
                kv = (nib - 8) * __half2float(kb->d);
            } else {
                const size_t r_idx = (size_t)h_kv * HD * 32 + (size_t)c * 32 + (size_t)tib;
                kv = __half2float(K_residual[r_idx]);
            }
            kval[b] = kv;
        }
        // V[t][h_kv][d] for d = b*32+lane (per-token Q4_0).
        const int vnb_byte = lane & 15;
        const bool v_high = lane >= 16;
        float vval[HD_BLOCKS];
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) {
            const block_q4_0 * vb = reinterpret_cast<const block_q4_0 *>(V_blob)
                + (size_t)t * v_n_kv_stride + (size_t)h_kv * v_kv_head_stride + b;
            const uint8_t byte = vb->qs[vnb_byte];
            const int nib = v_high ? (byte >> 4) : (byte & 0xF);
            vval[b] = (nib - 8) * __half2float(vb->d);
        }
        #pragma unroll
        for (int q = 0; q < MAX_NQ; ++q) {
            if (q >= n_q_per_kv) break;
            float dot = 0.0f;
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) dot += q_reg[q][b] * kval[b];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
            const float score = dot * scale;
            const float m_new = fmaxf(m[q], score);
            const float so = __expf(m[q] - m_new);
            const float sn = __expf(score - m_new);
            l[q] = l[q] * so + sn;
            #pragma unroll
            for (int b = 0; b < HD_BLOCKS; ++b) vkq[q][b] = vkq[q][b] * so + sn * vval[b];
            m[q] = m_new;
        }
    }

    #pragma unroll
    for (int q = 0; q < MAX_NQ; ++q) {
        if (q >= n_q_per_kv) break;
        const int h_q = h_kv * n_q_per_kv + q;
        float * p = partials + ((size_t)h_q * nsplit + split) * (HD + 2);
        #pragma unroll
        for (int b = 0; b < HD_BLOCKS; ++b) p[b * 32 + lane] = vkq[q][b];
        if (lane == 0) { p[HD] = m[q]; p[HD + 1] = l[q]; }
    }
}

#define FLASH_SPLITK_Q4_GQA_PARTIAL_KERNEL(NAME, HD, MAXNQ)                   \
extern "C" __global__ void NAME(                                             \
    const void  * __restrict__ K_blocks,                                     \
    const half  * __restrict__ K_residual,                                   \
    const void  * __restrict__ V_blob,                                       \
    const float * __restrict__ Q,                                            \
    float       * __restrict__ partials,                                     \
    int seq_kv,                                                              \
    int full_blocks,                                                         \
    int n_kv,                                                                \
    int n_q_per_kv,                                                          \
    int nsplit,                                                              \
    float scale                                                              \
) {                                                                          \
    flash_splitk_q4_gqa_partial_inner<HD, MAXNQ>(                            \
        K_blocks, K_residual, V_blob, Q, partials,                          \
        seq_kv, full_blocks, n_kv, n_q_per_kv, nsplit, scale,               \
        blockIdx.y, blockIdx.x, threadIdx.x                                 \
    );                                                                       \
}

FLASH_SPLITK_Q4_GQA_PARTIAL_KERNEL(flash_splitk_q4_gqa_partial_hd64,  64,  8)
FLASH_SPLITK_Q4_GQA_PARTIAL_KERNEL(flash_splitk_q4_gqa_partial_hd128, 128, 8)

// Device-position sibling of the Q4 GQA partial: reads seq_kv from a device
// int (cur_pos_dev[0]+1) so it is graph-capturable, mirroring the Q8 dev-pos
// partial. Same inner kernel + the shared runtime-nsplit combine. Lets the
// qwen3 Q4 decode use the fused flash split-K (no HBM scores round-trip)
// instead of the 2-kernel attn_score+attn_softmax_output chain.
#define FLASH_SPLITK_Q4_GQA_PARTIAL_DEVPOS_KERNEL(NAME, HD, MAXNQ)            \
extern "C" __global__ void NAME(                                             \
    const void  * __restrict__ K_blocks,                                     \
    const half  * __restrict__ K_residual,                                   \
    const void  * __restrict__ V_blob,                                       \
    const float * __restrict__ Q,                                            \
    float       * __restrict__ partials,                                     \
    const int32_t * __restrict__ seq_kv_dev,                                 \
    int n_kv,                                                                \
    int n_q_per_kv,                                                          \
    int nsplit,                                                              \
    float scale                                                              \
) {                                                                          \
    const int seq_kv = seq_kv_dev[0] + 1;                                    \
    const int full_blocks = seq_kv >> 5;                                     \
    flash_splitk_q4_gqa_partial_inner<HD, MAXNQ>(                            \
        K_blocks, K_residual, V_blob, Q, partials,                          \
        seq_kv, full_blocks, n_kv, n_q_per_kv, nsplit, scale,               \
        blockIdx.y, blockIdx.x, threadIdx.x                                 \
    );                                                                       \
}

FLASH_SPLITK_Q4_GQA_PARTIAL_DEVPOS_KERNEL(flash_splitk_q4_gqa_partial_devpos_hd64,  64,  8)
FLASH_SPLITK_Q4_GQA_PARTIAL_DEVPOS_KERNEL(flash_splitk_q4_gqa_partial_devpos_hd128, 128, 8)