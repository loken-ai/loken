// Causal GQA flash attention for multi-token prefill over an F16 KV history.
//
// One CTA owns BM = NW * 16 query rows of ONE kv head. The rows of the heads that share
// that kv head are interleaved token-major (row r -> token r / G, head kvh * G + r % G),
// so every K/V tile staged in shared memory serves all G query heads of the group.
// Scores, the online softmax and the P.V product stay in registers: S = Q.K^T and O += P.V
// are mma.sync m16n8k16 (F16 in, F32 accumulate), and the S accumulator layout is reused
// directly as the P operand. K and V tiles are double-buffered through cp.async so the
// load of tile i+1 overlaps the math of tile i.
//
// Layouts (element strides, b = 1):
//   Q  F32 [n_head, seq_q, HD] contiguous
//   K  F16, token t / kv head h at  K + t * kv_tok_stride + h * kv_head_stride  (HD contiguous)
//   V  F16, same strides as K
//   O  F32 [seq_q, n_head, HD] contiguous (token-major, ready for the output projection)
//
// Mask: query token i sits at absolute position i + (seq_kv - seq_q); it attends key j iff
//   j <= pos  and  (window <= 0 or j + window >= pos).
// Requires sm_80+ (m16n8k16 HMMA, cp.async); the launcher reports failure below that.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int BN   = 64;  // keys per tile
constexpr int KPAD = 8;   // halfs of padding per smem row (bank-conflict-free fragment loads)

__device__ __forceinline__ void mma_16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t r[4], const half* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(a));
#endif
}

__device__ __forceinline__ void cp_async_16(half* dst, const half* src, bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    const int n = valid ? 16 : 0;  // 0 source bytes -> the 16 destination bytes are zeroed
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(src), "r"(n));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait_1() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

// Stage keys [t0, t0 + BN) of one kv head into a [BN][HD + KPAD] shared tile; keys past
// seq_kv are zero-filled so no NaN from stale memory reaches the P.V product.
template <int HD, int NT>
__device__ __forceinline__ void load_tile(half* sm, const half* __restrict__ base, int t0,
                                          int seq_kv, long long tok_stride, int tid) {
    constexpr int CHUNKS_PER_ROW = HD / 8;  // 16-byte chunks
    constexpr int TOTAL = BN * CHUNKS_PER_ROW;
    #pragma unroll
    for (int i = tid; i < TOTAL; i += NT) {
        const int r = i / CHUNKS_PER_ROW;
        const int c = (i % CHUNKS_PER_ROW) * 8;
        const int t = t0 + r;
        const bool ok = t < seq_kv;
        const half* src = base + (ok ? (long long)t * tok_stride : 0) + c;
        cp_async_16(sm + r * (HD + KPAD) + c, src, ok);
    }
}

template <int HD, int NW>
__global__ void __launch_bounds__(NW * 32)
flash_prefill_gqa_kernel(const float* __restrict__ Q, const half* __restrict__ K,
                         const half* __restrict__ V, float* __restrict__ O,
                         int n_head, int n_kv, int seq_q, int seq_kv,
                         long long kv_tok_stride, long long kv_head_stride,
                         float scale_log2, int window, int pos_off,
                         float* __restrict__ ML) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    constexpr int NT = NW * 32;
    constexpr int BM = NW * 16;
    constexpr int KS = HD / 16;   // k-steps of Q.K^T
    constexpr int SN = BN / 8;    // n-tiles of S
    constexpr int ON = HD / 8;    // n-tiles of O
    constexpr int PS = BN / 16;   // k-steps of P.V
    constexpr int LD = HD + KPAD;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    half* Ks = reinterpret_cast<half*>(smem_raw);
    half* Vs = Ks + BN * LD;

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int g    = lane >> 2;   // fragment row group
    const int tq   = lane & 3;    // thread in group

    const int G       = n_head / n_kv;
    const int kvh     = blockIdx.y;
    const int rows    = seq_q * G;
    const int row0    = blockIdx.x * BM;

    // This thread's two fragment rows (g and g + 8 of the warp's 16).
    int  r_tok[2], r_head[2], r_pos[2];
    bool r_ok[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int r = row0 + warp * 16 + g + h * 8;
        r_ok[h]   = r < rows;
        r_tok[h]  = r_ok[h] ? r / G : 0;
        r_head[h] = kvh * G + (r_ok[h] ? r % G : 0);
        r_pos[h]  = r_tok[h] + pos_off;
    }

    // Q fragments (A operand, row-major 16 x HD), pre-scaled so the softmax runs in base 2.
    uint32_t qa[KS][4];
    #pragma unroll
    for (int kk = 0; kk < KS; ++kk) {
        #pragma unroll
        for (int hi = 0; hi < 2; ++hi) {        // column half: +0 / +8
            #pragma unroll
            for (int h = 0; h < 2; ++h) {       // row half: g / g + 8
                float2 v = make_float2(0.f, 0.f);
                if (r_ok[h]) {
                    const float* qp = Q + ((long long)r_head[h] * seq_q + r_tok[h]) * HD
                                    + kk * 16 + hi * 8 + 2 * tq;
                    v = *reinterpret_cast<const float2*>(qp);
                }
                qa[kk][hi * 2 + h] = pack_half2(v.x * scale_log2, v.y * scale_log2);
            }
        }
    }

    // Key range this CTA needs: causal end of its last row, window start of its first.
    const int last_row  = min(rows, row0 + BM) - 1;
    const int first_pos = (row0 / G) + pos_off;
    const int last_pos  = (last_row / G) + pos_off;
    const int kv_end    = min(seq_kv, last_pos + 1);
    int kv_begin = 0;
    if (window > 0) kv_begin = max(0, first_pos - window) / BN * BN;

    // The warp's own row span, to skip tiles entirely outside it.
    const int w_first_row = row0 + warp * 16;
    const int w_last_row  = min(rows, w_first_row + 16) - 1;
    const bool warp_live  = w_first_row < rows;
    const int w_first_pos = w_first_row / G + pos_off;
    const int w_last_pos  = (warp_live ? w_last_row / G : 0) + pos_off;

    float o[ON][4];
    #pragma unroll
    for (int n = 0; n < ON; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    float m_run[2] = {-INFINITY, -INFINITY};
    float l_run[2] = {0.f, 0.f};

    const half* Kb = K + (long long)kvh * kv_head_stride;
    const half* Vb = V + (long long)kvh * kv_head_stride;

    load_tile<HD, NT>(Ks, Kb, kv_begin, seq_kv, kv_tok_stride, tid);
    cp_async_commit();
    load_tile<HD, NT>(Vs, Vb, kv_begin, seq_kv, kv_tok_stride, tid);
    cp_async_commit();

    for (int t0 = kv_begin; t0 < kv_end; t0 += BN) {
        const bool more = t0 + BN < kv_end;
        // Tile useful to this warp: some key <= its last position and inside the window
        // of its first position.
        const bool warp_tile = warp_live && t0 <= w_last_pos
                            && (window <= 0 || t0 + BN - 1 + window >= w_first_pos);

        cp_async_wait_1();       // K(t0) landed; V(t0) may still be in flight
        __syncthreads();

        float s[SN][4];
        if (warp_tile) {
            #pragma unroll
            for (int n = 0; n < SN; ++n) {
                s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
                const half* kr = Ks + (n * 8 + g) * LD + 2 * tq;
                #pragma unroll
                for (int kk = 0; kk < KS; ++kk) {
                    const uint32_t b0 = *reinterpret_cast<const uint32_t*>(kr + kk * 16);
                    const uint32_t b1 = *reinterpret_cast<const uint32_t*>(kr + kk * 16 + 8);
                    mma_16816(s[n], qa[kk], b0, b1);
                }
            }
        }
        __syncthreads();         // every warp is done reading K(t0)
        if (more) load_tile<HD, NT>(Ks, Kb, t0 + BN, seq_kv, kv_tok_stride, tid);
        cp_async_commit();

        if (warp_tile) {
            // Mask only where the tile crosses the causal diagonal, the window edge or
            // the end of the history.
            const bool need_mask = (t0 + BN - 1 > w_first_pos) || (t0 + BN > seq_kv)
                                || (window > 0 && t0 + window < w_last_pos);
            if (need_mask) {
                #pragma unroll
                for (int n = 0; n < SN; ++n) {
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int j = t0 + n * 8 + 2 * tq + (e & 1);
                        const int p = r_pos[e >> 1];
                        const bool keep = r_ok[e >> 1] && j < seq_kv && j <= p
                                       && (window <= 0 || j + window >= p);
                        if (!keep) s[n][e] = -INFINITY;
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                float mx = -INFINITY;
                #pragma unroll
                for (int n = 0; n < SN; ++n) mx = fmaxf(mx, fmaxf(s[n][2 * h], s[n][2 * h + 1]));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
                const float m_new = fmaxf(m_run[h], mx);
                // A row with no visible key yet keeps a zero reference so no inf - inf.
                const float m_ref = (m_new == -INFINITY) ? 0.f : m_new;
                const float alpha = exp2f(m_run[h] - m_ref);
                float sum = 0.f;
                #pragma unroll
                for (int n = 0; n < SN; ++n) {
                    const float p0 = exp2f(s[n][2 * h] - m_ref);
                    const float p1 = exp2f(s[n][2 * h + 1] - m_ref);
                    s[n][2 * h] = p0;
                    s[n][2 * h + 1] = p1;
                    sum += p0 + p1;
                }
                l_run[h] = l_run[h] * alpha + sum;
                m_run[h] = m_new;
                #pragma unroll
                for (int n = 0; n < ON; ++n) {
                    o[n][2 * h] *= alpha;
                    o[n][2 * h + 1] *= alpha;
                }
            }
        }

        cp_async_wait_1();       // V(t0) landed; K(t0 + BN) may still be in flight
        __syncthreads();

        if (warp_tile) {
            #pragma unroll
            for (int ks = 0; ks < PS; ++ks) {
                uint32_t pa[4];
                pa[0] = pack_half2(s[2 * ks][0], s[2 * ks][1]);
                pa[1] = pack_half2(s[2 * ks][2], s[2 * ks][3]);
                pa[2] = pack_half2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
                pa[3] = pack_half2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
                const int vrow = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
                const half* vp = Vs + vrow * LD + (lane >> 4) * 8;
                #pragma unroll
                for (int np = 0; np < HD / 16; ++np) {
                    uint32_t vb[4];
                    ldmatrix_x4_trans(vb, vp + np * 16);
                    mma_16816(o[2 * np], pa, vb[0], vb[1]);
                    mma_16816(o[2 * np + 1], pa, vb[2], vb[3]);
                }
            }
        }
        __syncthreads();         // every warp is done reading V(t0)
        if (more) load_tile<HD, NT>(Vs, Vb, t0 + BN, seq_kv, kv_tok_stride, tid);
        cp_async_commit();
    }

    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        float l = l_run[h];
        l += __shfl_xor_sync(0xffffffff, l, 1);
        l += __shfl_xor_sync(0xffffffff, l, 2);
        if (!r_ok[h]) continue;
        const float inv = l > 0.f ? 1.f / l : 0.f;
        // The row's maximum score, back from base 2, and its sum, for a caller merging ranges.
        if (ML != nullptr && tq == 0) {
            float* ml = ML + ((long long)r_tok[h] * n_head + r_head[h]) * 2;
            ml[0] = m_run[h] * 0.6931471805599453f;
            ml[1] = l;
        }
        float* orow = O + ((long long)r_tok[h] * n_head + r_head[h]) * HD + 2 * tq;
        #pragma unroll
        for (int n = 0; n < ON; ++n) {
            *reinterpret_cast<float2*>(orow + n * 8) =
                make_float2(o[n][2 * h] * inv, o[n][2 * h + 1] * inv);
        }
    }
#endif
}

template <int HD, int NW>
int launch(const float* Q, const half* K, const half* V, float* O, int n_head, int n_kv,
           int seq_q, int seq_kv, long long tok_stride, long long head_stride, float scale,
           int window, int pos_off, float* ML, cudaStream_t stream) {
    constexpr int BM = NW * 16;
    const size_t smem = 2 * (size_t)BN * (HD + KPAD) * sizeof(half);
    auto kern = flash_prefill_gqa_kernel<HD, NW>;
    if (cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)
        != cudaSuccess) {
        return 3;
    }
    const int G = n_head / n_kv;
    dim3 grid((seq_q * G + BM - 1) / BM, n_kv, 1);
    const float log2e = 1.4426950408889634f;
    // The check below reads this launch's error, not one an earlier call left on the thread.
    (void) cudaGetLastError();
    kern<<<grid, NW * 32, smem, stream>>>(Q, K, V, O, n_head, n_kv, seq_q, seq_kv, tok_stride,
                                          head_stride, scale * log2e, window, pos_off, ML);
    return cudaGetLastError() == cudaSuccess ? 0 : 4;
}

}  // namespace

namespace {
int dispatch(const void* Q, const void* K, const void* V, void* O, float* ML, int n_head,
             int n_kv, int seq_q, int seq_kv, int head_dim, long long kv_tok_stride,
             long long kv_head_stride, float scale, int window, int pos_off,
             long long stream_i64) {
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 8) return 2;
    cudaStream_t stream = (cudaStream_t)stream_i64;
    const float* q = (const float*)Q;
    const half* k = (const half*)K;
    const half* v = (const half*)V;
    float* o = (float*)O;
    switch (head_dim) {
        case 64:
            return launch<64, 4>(q, k, v, o, n_head, n_kv, seq_q, seq_kv, kv_tok_stride,
                                 kv_head_stride, scale, window, pos_off, ML, stream);
        case 128:
            return launch<128, 4>(q, k, v, o, n_head, n_kv, seq_q, seq_kv, kv_tok_stride,
                                  kv_head_stride, scale, window, pos_off, ML, stream);
        case 256:
            // Half the warps of the narrower sizes: the O accumulator and the Q fragments of a
            // 256-wide head fill the register file at four warps' worth of rows.
            return launch<256, 2>(q, k, v, o, n_head, n_kv, seq_q, seq_kv, kv_tok_stride,
                                  kv_head_stride, scale, window, pos_off, ML, stream);
        default:
            return 1;
    }
}
}  // namespace

// Returns 0 on launch, 1 for an unsupported head size, 2 for a device below sm_80,
// 3/4 for a CUDA launch error.
extern "C" int loken_flash_prefill_gqa_f16kv(
    const void* Q, const void* K, const void* V, void* O, int n_head, int n_kv, int seq_q,
    int seq_kv, int head_dim, long long kv_tok_stride, long long kv_head_stride, float scale,
    int window, long long stream_i64) {
    return dispatch(Q, K, V, O, nullptr, n_head, n_kv, seq_q, seq_kv, head_dim, kv_tok_stride,
                    kv_head_stride, scale, window, seq_kv - seq_q, stream_i64);
}

// The same attention over keys whose positions start `pos_off` before the first query
// token's (token i attends key j iff j <= i + pos_off), each row's maximum score and sum of
// exponentials written to `ML` `[seq_q, n_head, 2]` beside its normalized output.
extern "C" int loken_flash_prefill_gqa_parts_f16kv(
    const void* Q, const void* K, const void* V, void* O, float* ML, int n_head, int n_kv,
    int seq_q, int seq_kv, int head_dim, long long kv_tok_stride, long long kv_head_stride,
    float scale, int pos_off, long long stream_i64) {
    return dispatch(Q, K, V, O, ML, n_head, n_kv, seq_q, seq_kv, head_dim, kv_tok_stride,
                    kv_head_stride, scale, 0, pos_off, stream_i64);
}
