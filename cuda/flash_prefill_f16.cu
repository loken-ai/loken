// Fused FlashAttention-2-style PREFILL for GQA over F16 K/V (seq_q > 1, causal).
//
// The generic multi-token attention path materializes the full [n_head, seq_q,
// seq_kv] cuBLAS scores buffer (q.kᵀ -> softmax -> att.v; ~736 MB/layer at
// seq=2398) plus repeat_kv / v.contiguous copies - HBM-traffic bound, dropping
// O(seq²) at long context while ollama's flash-attention stays flat. This kernel
// tiles QxK over the sequence and keeps the [seq_q x seq_kv] scores ENTIRELY in
// registers/SRAM (online-softmax), never touching HBM, using tensor-core
// mma.sync (nvcuda::wmma m16n16k16 HMMA on sm_80+/sm_120) for BOTH q.kᵀ and p.v.
//
// One warp per (query-block of 16 rows, query head, batch). GQA is handled
// in-kernel (kv head = head / (n_head/n_kv)) - no repeat_kv expansion. The causal
// (+ optional sliding-window) mask is computed ANALYTICALLY from the token
// indices, reproducing generic_transformer.rs::make_mask exactly:
//   query row i (absolute pos abs_i = i + (seq_kv - seq_q)) attends key j iff
//     j <= abs_i  AND  (window <= 0  OR  j + window >= abs_i)
// so no [seq_q x seq_kv] mask tensor is read from HBM either. Causal upper-
// triangular KV tiles are skipped (the tb loop stops at the last query's diagonal).
//
// F16 in / F16 out; the online softmax + accumulator run in F32 (matches the
// batched softmax reference closely enough for greedy-argmax parity - validated
// via the coherence gate at the call site). Layouts (all contiguous, row-major):
//   Q [b, n_head,  seq_q,  HD]      O [b, n_head, seq_q, HD]
//   K [b, n_kv,    seq_kv, HD]      V [b, n_kv,   seq_kv, HD]
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

template <int HD>
static __device__ __forceinline__ void flash_prefill_f16_inner(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ O,
    int n_head, int n_kv, int seq_q, int seq_kv, float scale, int window,
    int qblk, int head, int bi, int lane)
{
    constexpr int KSTEPS = HD / 16;   // QKᵀ contraction chunks
    constexpr int HCHUNK = HD / 16;   // P.V output head-dim chunks
    const int groups  = n_head / n_kv;
    const int kvh     = head / groups;
    const int q_base  = qblk * 16;
    const int Mrows   = min(16, seq_q - q_base);
    const int pos_off = seq_kv - seq_q;          // == index_pos (make_mask)

    const half* Qh = Q + ((size_t)(bi * n_head + head) * seq_q + q_base) * HD;
    const half* Kh = K + (size_t)(bi * n_kv + kvh) * seq_kv * HD;
    const half* Vh = V + (size_t)(bi * n_kv + kvh) * seq_kv * HD;
    half*       Oh = O + ((size_t)(bi * n_head + head) * seq_q + q_base) * HD;

    __shared__ half  Qsh[16 * HD];
    __shared__ half  Kt [16 * HD];
    __shared__ half  Vt [16 * HD];
    __shared__ half  Psh[16 * 16];
    __shared__ float Ssh[16 * 16];
    __shared__ float Osh[16 * 16];
    __shared__ float Oacc[16 * HD];
    __shared__ float m_row[16];
    __shared__ float l_row[16];
    __shared__ float alpha_sh[16];

    // Stage this block's Q rows into shared; zero the pad rows.
    for (int i = lane; i < 16 * HD; i += 32) {
        const int r = i / HD, d = i % HD;
        Qsh[i] = (r < Mrows) ? Qh[(size_t)r * HD + d] : __float2half(0.0f);
    }
    for (int i = lane; i < 16 * HD; i += 32) Oacc[i] = 0.0f;
    if (lane < 16) { m_row[lane] = -INFINITY; l_row[lane] = 0.0f; }
    __syncwarp();

    // Causal bound: the last query in this block (abs pos q_base+Mrows-1+pos_off)
    // attends keys up to itself, so no KV tile beyond that diagonal is needed.
    const int kv_end = min(seq_kv, q_base + Mrows + pos_off);

    for (int tb = 0; tb < kv_end; tb += 16) {
        const int ntok = min(16, seq_kv - tb);
        // Load the [16 x HD] K/V tile from F16 global (zero the pad tokens).
        for (int i = lane; i < 16 * HD; i += 32) {
            const int r = i / HD, d = i % HD;
            if (r < ntok) {
                Kt[i] = Kh[(size_t)(tb + r) * HD + d];
                Vt[i] = Vh[(size_t)(tb + r) * HD + d];
            } else {
                Kt[i] = __float2half(0.0f);
                Vt[i] = __float2half(0.0f);
            }
        }
        __syncwarp();

        // S[16(M) x 16(token)] = Q . Kᵀ  (contract HD in 16-wide chunks).
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cS;
        wmma::fill_fragment(cS, 0.0f);
        #pragma unroll
        for (int kk = 0; kk < KSTEPS; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bF;
            wmma::load_matrix_sync(aF, Qsh + kk * 16, HD);
            wmma::load_matrix_sync(bF, Kt  + kk * 16, HD);
            wmma::mma_sync(cS, aF, bF, cS);
        }
        wmma::store_matrix_sync(Ssh, cS, 16, wmma::mem_row_major);
        __syncwarp();

        // Online softmax per packed row (lanes 0..15 own rows). Analytic causal
        // (+ sliding-window) mask; write rescale alpha into shared so the Oacc
        // rescale below parallelizes across all 32 lanes.
        for (int r = lane; r < 16; r += 32) {
            const int abs_i = q_base + r + pos_off;
            float smax = -INFINITY;
            #pragma unroll
            for (int n = 0; n < 16; ++n) {
                const int j = tb + n;
                const bool keep = (n < ntok) && (r < Mrows)
                    && (j <= abs_i) && (window <= 0 || j + window >= abs_i);
                const float s = keep ? Ssh[r * 16 + n] * scale : -INFINITY;
                Ssh[r * 16 + n] = s;
                smax = fmaxf(smax, s);
            }
            const float m_new = fmaxf(m_row[r], smax);
            const float alpha = (m_row[r] == -INFINITY) ? 0.0f : __expf(m_row[r] - m_new);
            float l_add = 0.0f;
            #pragma unroll
            for (int n = 0; n < 16; ++n) {
                const float p = (Ssh[r * 16 + n] == -INFINITY) ? 0.0f
                                                               : __expf(Ssh[r * 16 + n] - m_new);
                Psh[r * 16 + n] = __float2half(p);
                l_add += p;
            }
            l_row[r] = l_row[r] * alpha + l_add;
            m_row[r] = m_new;
            alpha_sh[r] = alpha;
        }
        __syncwarp();
        for (int idx = lane; idx < 16 * HD; idx += 32) Oacc[idx] *= alpha_sh[idx / HD];
        __syncwarp();

        // O[16(M) x HD] += P . V  (one HMMA per 16-wide hd chunk).
        #pragma unroll
        for (int hc = 0; hc < HCHUNK; ++hc) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cO;
            wmma::fill_fragment(cO, 0.0f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vF;
            wmma::load_matrix_sync(pF, Psh, 16);
            wmma::load_matrix_sync(vF, Vt + hc * 16, HD);
            wmma::mma_sync(cO, pF, vF, cO);
            wmma::store_matrix_sync(Osh, cO, 16, wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 16 * 16; idx += 32) {
                const int r = idx >> 4;
                const int c = idx & 15;
                Oacc[r * HD + hc * 16 + c] += Osh[idx];
            }
            __syncwarp();
        }
    }

    // Normalize + write out.
    for (int r = 0; r < Mrows; ++r) {
        const float inv = (l_row[r] > 0.0f) ? (1.0f / l_row[r]) : 0.0f;
        for (int d = lane; d < HD; d += 32)
            Oh[(size_t)r * HD + d] = __float2half(Oacc[r * HD + d] * inv);
    }
}

#define FLASH_PREFILL_KERNEL(NAME, HD)                                          \
extern "C" __global__ void NAME(                                               \
    const half* __restrict__ Q, const half* __restrict__ K,                    \
    const half* __restrict__ V, half* __restrict__ O,                          \
    int n_head, int n_kv, int seq_q, int seq_kv, float scale, int window)      \
{                                                                              \
    flash_prefill_f16_inner<HD>(                                               \
        Q, K, V, O, n_head, n_kv, seq_q, seq_kv, scale, window,                \
        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x);                      \
}

FLASH_PREFILL_KERNEL(flash_prefill_f16_hd64_kernel, 64)
FLASH_PREFILL_KERNEL(flash_prefill_f16_hd128_kernel, 128)

// -- SPLIT-K over the KV axis ------------------------------------------------
// The one-warp-per-(q-block,head) kernel above underfills the GPU at the LATE
// prefill chunks: seq_q is a fixed 512-token chunk but seq_kv grows to the full
// prompt (e.g. 8672), so each of the ~n_qblk*n_head warps serially scans ~540 KV
// tiles - far less parallelism than cuBLAS's [512x8672] GEMM. Splitting the KV
// scan into `nsplit` chunks (grid z) gives nsplitx more blocks (online-softmax
// partials per split, merged by the combine kernel), matching the decode split-K
// pattern. Causal: a split whose KV range is entirely beyond every query row's
// diagonal emits a neutral partial (m=-inf) and the combine skips it.
template <int HD>
static __device__ __forceinline__ void flash_prefill_split_inner(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, float* __restrict__ partials,
    int n_head, int n_kv, int seq_q, int seq_kv, int nsplit, float scale, int window,
    int qblk, int head, int split, int bi, int lane)
{
    constexpr int KSTEPS = HD / 16;
    constexpr int HCHUNK = HD / 16;
    const int groups  = n_head / n_kv;
    const int kvh     = head / groups;
    const int q_base  = qblk * 16;
    const int Mrows   = min(16, seq_q - q_base);
    const int pos_off = seq_kv - seq_q;

    // This split's KV token range, clamped to the block's causal bound.
    const int kv_causal = min(seq_kv, q_base + Mrows + pos_off);
    const int chunk = (seq_kv + nsplit - 1) / nsplit;
    const int t0 = split * chunk;
    const int t1 = min(t0 + chunk, kv_causal);

    // Global row base: row (bi, head, q_base+r) in the flat [b*n_head*seq_q] order.
    const size_t grow = ((size_t)(bi * n_head + head) * seq_q + q_base);

    if (t0 >= t1) {   // empty (or fully-causal-masked) split -> neutral partials
        for (int r = 0; r < Mrows; ++r) {
            float* p = partials + ((grow + r) * nsplit + split) * (HD + 2);
            for (int d = lane; d < HD; d += 32) p[d] = 0.0f;
            if (lane == 0) { p[HD] = -INFINITY; p[HD + 1] = 0.0f; }
        }
        return;
    }

    const half* Qh = Q + (grow) * HD;
    const half* Kh = K + (size_t)(bi * n_kv + kvh) * seq_kv * HD;
    const half* Vh = V + (size_t)(bi * n_kv + kvh) * seq_kv * HD;

    __shared__ half  Qsh[16 * HD];
    __shared__ half  Kt [16 * HD];
    __shared__ half  Vt [16 * HD];
    __shared__ half  Psh[16 * 16];
    __shared__ float Ssh[16 * 16];
    __shared__ float Osh[16 * 16];
    __shared__ float Oacc[16 * HD];
    __shared__ float m_row[16];
    __shared__ float l_row[16];
    __shared__ float alpha_sh[16];

    for (int i = lane; i < 16 * HD; i += 32) {
        const int r = i / HD, d = i % HD;
        Qsh[i] = (r < Mrows) ? Qh[(size_t)r * HD + d] : __float2half(0.0f);
    }
    for (int i = lane; i < 16 * HD; i += 32) Oacc[i] = 0.0f;
    if (lane < 16) { m_row[lane] = -INFINITY; l_row[lane] = 0.0f; }
    __syncwarp();

    for (int tb = t0; tb < t1; tb += 16) {
        const int ntok = min(16, seq_kv - tb);
        for (int i = lane; i < 16 * HD; i += 32) {
            const int r = i / HD, d = i % HD;
            if (r < ntok) {
                Kt[i] = Kh[(size_t)(tb + r) * HD + d];
                Vt[i] = Vh[(size_t)(tb + r) * HD + d];
            } else { Kt[i] = __float2half(0.0f); Vt[i] = __float2half(0.0f); }
        }
        __syncwarp();

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> cS;
        wmma::fill_fragment(cS, 0.0f);
        #pragma unroll
        for (int kk = 0; kk < KSTEPS; ++kk) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> aF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bF;
            wmma::load_matrix_sync(aF, Qsh + kk * 16, HD);
            wmma::load_matrix_sync(bF, Kt  + kk * 16, HD);
            wmma::mma_sync(cS, aF, bF, cS);
        }
        wmma::store_matrix_sync(Ssh, cS, 16, wmma::mem_row_major);
        __syncwarp();

        for (int r = lane; r < 16; r += 32) {
            const int abs_i = q_base + r + pos_off;
            float smax = -INFINITY;
            #pragma unroll
            for (int n = 0; n < 16; ++n) {
                const int j = tb + n;
                const bool keep = (n < ntok) && (r < Mrows)
                    && (j <= abs_i) && (window <= 0 || j + window >= abs_i);
                const float s = keep ? Ssh[r * 16 + n] * scale : -INFINITY;
                Ssh[r * 16 + n] = s;
                smax = fmaxf(smax, s);
            }
            const float m_new = fmaxf(m_row[r], smax);
            const float alpha = (m_row[r] == -INFINITY) ? 0.0f : __expf(m_row[r] - m_new);
            float l_add = 0.0f;
            #pragma unroll
            for (int n = 0; n < 16; ++n) {
                const float p = (Ssh[r * 16 + n] == -INFINITY) ? 0.0f
                                                               : __expf(Ssh[r * 16 + n] - m_new);
                Psh[r * 16 + n] = __float2half(p);
                l_add += p;
            }
            l_row[r] = l_row[r] * alpha + l_add;
            m_row[r] = m_new;
            alpha_sh[r] = alpha;
        }
        __syncwarp();
        for (int idx = lane; idx < 16 * HD; idx += 32) Oacc[idx] *= alpha_sh[idx / HD];
        __syncwarp();

        #pragma unroll
        for (int hc = 0; hc < HCHUNK; ++hc) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> cO;
            wmma::fill_fragment(cO, 0.0f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pF;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vF;
            wmma::load_matrix_sync(pF, Psh, 16);
            wmma::load_matrix_sync(vF, Vt + hc * 16, HD);
            wmma::mma_sync(cO, pF, vF, cO);
            wmma::store_matrix_sync(Osh, cO, 16, wmma::mem_row_major);
            __syncwarp();
            for (int idx = lane; idx < 16 * 16; idx += 32) {
                const int r = idx >> 4;
                const int c = idx & 15;
                Oacc[r * HD + hc * 16 + c] += Osh[idx];
            }
            __syncwarp();
        }
    }

    // Emit this split's partial (unnormalized Oacc + m,l) per row.
    for (int r = 0; r < Mrows; ++r) {
        float* p = partials + ((grow + r) * nsplit + split) * (HD + 2);
        for (int d = lane; d < HD; d += 32) p[d] = Oacc[r * HD + d];
        if (lane == 0) { p[HD] = m_row[r]; p[HD + 1] = l_row[r]; }
    }
}

#define FLASH_PREFILL_SPLIT_KERNEL(NAME, HD)                                    \
extern "C" __global__ void NAME(                                               \
    const half* __restrict__ Q, const half* __restrict__ K,                    \
    const half* __restrict__ V, float* __restrict__ partials,                  \
    int n_head, int n_kv, int seq_q, int seq_kv, int nsplit, float scale, int window) \
{                                                                              \
    flash_prefill_split_inner<HD>(                                             \
        Q, K, V, partials, n_head, n_kv, seq_q, seq_kv, nsplit, scale, window, \
        blockIdx.x, blockIdx.y, blockIdx.z, /*bi=*/0, threadIdx.x);            \
}

FLASH_PREFILL_SPLIT_KERNEL(flash_prefill_split_hd64_kernel, 64)
FLASH_PREFILL_SPLIT_KERNEL(flash_prefill_split_hd128_kernel, 128)

// Combine: per row, log-sum-exp merge of its nsplit partials -> O[row, HD] (f16).
// One warp per row; lane d owns dims d, d+32, ... Serial nsplit merge.
template <int HD>
static __device__ __forceinline__ void flash_prefill_combine_inner(
    const float* __restrict__ partials, half* __restrict__ O,
    int nsplit, int row, int lane)
{
    constexpr int DPL = HD / 32;
    const float* base = partials + (size_t)row * nsplit * (HD + 2);
    float gm = -INFINITY, gl = 0.0f;
    float acc[DPL];
    #pragma unroll
    for (int b = 0; b < DPL; ++b) acc[b] = 0.0f;
    for (int sp = 0; sp < nsplit; ++sp) {
        const float* p = base + (size_t)sp * (HD + 2);
        const float mi = p[HD];
        if (mi == -INFINITY) continue;           // neutral (empty/causal-masked) split
        const float li = p[HD + 1];
        const float m_new = fmaxf(gm, mi);
        const float so = (gm == -INFINITY) ? 0.0f : __expf(gm - m_new);
        const float sn = __expf(mi - m_new);
        gl = gl * so + sn * li;
        #pragma unroll
        for (int b = 0; b < DPL; ++b) acc[b] = acc[b] * so + sn * p[b * 32 + lane];
        gm = m_new;
    }
    const float inv = (gl > 0.0f) ? (1.0f / gl) : 0.0f;
    #pragma unroll
    for (int b = 0; b < DPL; ++b) O[(size_t)row * HD + b * 32 + lane] = __float2half(acc[b] * inv);
}

#define FLASH_PREFILL_COMBINE_KERNEL(NAME, HD)                                  \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ partials, half* __restrict__ O, int nsplit)      \
{                                                                              \
    flash_prefill_combine_inner<HD>(partials, O, nsplit, blockIdx.x, threadIdx.x); \
}

FLASH_PREFILL_COMBINE_KERNEL(flash_prefill_combine_hd64_kernel, 64)
FLASH_PREFILL_COMBINE_KERNEL(flash_prefill_combine_hd128_kernel, 128)

extern "C" void flash_prefill_split_f16(
    const void* Q, const void* K, const void* V, float* partials, void* O,
    int batch, int n_head, int n_kv, int seq_q, int seq_kv, int head_dim, int nsplit,
    float scale, int window, long long stream_i64)
{
    (void)batch;   // batch handled by caller looping bi into the row base (bi=0 here)
    cudaStream_t stream = (cudaStream_t)stream_i64;
    const int nqblk = (seq_q + 15) / 16;
    dim3 g1(nqblk, n_head, nsplit), b1(32, 1, 1);
    const int total_rows = n_head * seq_q;   // batch==1 fast path
    dim3 g2(total_rows, 1, 1), b2(32, 1, 1);
    if (head_dim == 128) {
        flash_prefill_split_hd128_kernel<<<g1, b1, 0, stream>>>(
            (const half*)Q, (const half*)K, (const half*)V, partials,
            n_head, n_kv, seq_q, seq_kv, nsplit, scale, window);
        flash_prefill_combine_hd128_kernel<<<g2, b2, 0, stream>>>(partials, (half*)O, nsplit);
    } else if (head_dim == 64) {
        flash_prefill_split_hd64_kernel<<<g1, b1, 0, stream>>>(
            (const half*)Q, (const half*)K, (const half*)V, partials,
            n_head, n_kv, seq_q, seq_kv, nsplit, scale, window);
        flash_prefill_combine_hd64_kernel<<<g2, b2, 0, stream>>>(partials, (half*)O, nsplit);
    }
}

// Host launcher (extern "C", driven from inference::fused_kernels). All pointers
// device; Q/K/V/O contiguous F16 as documented above. head_dim must be 64 or 128.
extern "C" void flash_prefill_f16(
    const void* Q, const void* K, const void* V, void* O,
    int batch, int n_head, int n_kv, int seq_q, int seq_kv, int head_dim,
    float scale, int window, long long stream_i64)
{
    cudaStream_t stream = (cudaStream_t)stream_i64;
    const int nqblk = (seq_q + 15) / 16;
    dim3 grid(nqblk, n_head, batch);
    dim3 block(32, 1, 1);
    if (head_dim == 128) {
        flash_prefill_f16_hd128_kernel<<<grid, block, 0, stream>>>(
            (const half*)Q, (const half*)K, (const half*)V, (half*)O,
            n_head, n_kv, seq_q, seq_kv, scale, window);
    } else if (head_dim == 64) {
        flash_prefill_f16_hd64_kernel<<<grid, block, 0, stream>>>(
            (const half*)Q, (const half*)K, (const half*)V, (half*)O,
            n_head, n_kv, seq_q, seq_kv, scale, window);
    }
}
