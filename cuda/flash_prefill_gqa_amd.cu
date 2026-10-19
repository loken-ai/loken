// Causal GQA flash attention for multi-token prefill over an F16 KV history, for AMD cards: the
// entry points of `flash_prefill_gqa.cu`, with the same arguments, layouts, mask and results,
// on AMD's 16-bit matrix instructions (`cuda/hip/amd_mma16.cuh`).
//
//   Q  F32 [n_head, seq_q, HD] contiguous
//   K  F16, token t / kv head h at  K + t * kv_tok_stride + h * kv_head_stride  (HD contiguous)
//   V  F16, same strides as K
//   O  F32 [seq_q, n_head, HD] contiguous
//
// Query token i sits at position i + pos_off; it attends key j iff j <= pos and
// (window <= 0 or j + window >= pos).
//
// A block owns NW * 16 query rows of one kv head, the rows of the heads sharing it interleaved
// token-major (row r -> token r / G, head kvh * G + r % G), so every K/V tile staged in shared
// memory serves the whole group. A wave owns 16 rows and keeps its scores and its output in
// registers, transposed: `S^T = K Q^T` puts one query in each accumulator column, which is the
// layout `O^T = V^T P^T` takes P in, so the probabilities never leave the lane and a query's
// running maximum, sum and rescale are all the lane's own. Q is pre-scaled into base 2 and held
// in F16 in shared memory, as are the K and V tiles.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "amd_mma16.cuh"

namespace {

using loken_amd_mma16::Core;
namespace mma16 = loken_amd_mma16;

constexpr int PAD = 8;  // halfs of padding per shared row

/// Keys per tile for a head size: a narrower tile keeps a 256-wide head's tiles in 64 KB.
constexpr int tile_keys(int hd) { return hd >= 256 ? 32 : 64; }

template <int HD, int NW>
__global__ void __launch_bounds__(NW * Core::wave)
    flash_prefill_gqa_amd(const float* __restrict__ Q, const _Float16* __restrict__ K,
                          const _Float16* __restrict__ V, float* __restrict__ O, int n_head,
                          int n_kv, int seq_q, int seq_kv, long long kv_tok_stride,
                          long long kv_head_stride, float scale_log2, int window, int pos_off,
                          float* __restrict__ ML) {
    constexpr int BN = tile_keys(HD);
    constexpr int NT = NW * Core::wave;
    constexpr int BM = NW * 16;
    constexpr int KS = HD / 16;  // k steps of K Q^T, and d tiles of the output
    constexpr int SM = BN / 16;  // key tiles of S^T
    constexpr int LD = HD + PAD;

    __shared__ __attribute__((aligned(16))) _Float16 Ks[BN * LD];
    __shared__ __attribute__((aligned(16))) _Float16 Vs[BN * LD];
    __shared__ __attribute__((aligned(16))) _Float16 Qs[NW][16 * LD];

    const int tid = threadIdx.x;
    const int wave = tid / Core::wave;
    const int lane = tid % Core::wave;
    const int qi = lane & 15;

    const int G = n_head / n_kv;
    const int kvh = blockIdx.y;
    const int rows = seq_q * G;
    const int row0 = blockIdx.x * BM;

    // The lane's query: column `qi` of every accumulator.
    const int r = row0 + wave * 16 + qi;
    const bool r_ok = r < rows;
    const int r_tok = r_ok ? r / G : 0;
    const int r_head = kvh * G + (r_ok ? r % G : 0);
    const int r_pos = r_tok + pos_off;

    // The wave's 16 query rows, pre-scaled, in F16; a row past the end is zero.
    {
        _Float16* qs = Qs[wave];
        for (int i = lane; i < 16 * HD / 4; i += Core::wave) {
            const int row = i / (HD / 4), c = (i % (HD / 4)) * 4;
            const int rr = row0 + wave * 16 + row;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (rr < rows) {
                const int tok = rr / G, head = kvh * G + rr % G;
                v = *reinterpret_cast<const float4*>(Q + ((long long) head * seq_q + tok) * HD + c);
            }
            _Float16* d = qs + row * LD + c;
            d[0] = (_Float16) (v.x * scale_log2);
            d[1] = (_Float16) (v.y * scale_log2);
            d[2] = (_Float16) (v.z * scale_log2);
            d[3] = (_Float16) (v.w * scale_log2);
        }
    }

    // Key range the block needs: causal end of its last row, window start of its first.
    const int last_row = min(rows, row0 + BM) - 1;
    const int first_pos = (row0 / G) + pos_off;
    const int last_pos = (last_row / G) + pos_off;
    const int kv_end = min(seq_kv, last_pos + 1);
    int kv_begin = 0;
    if (window > 0) kv_begin = max(0, first_pos - window) / BN * BN;

    // The wave's own row span, to skip tiles entirely outside it.
    const int w_first_row = row0 + wave * 16;
    const int w_last_row = min(rows, w_first_row + 16) - 1;
    const bool wave_live = w_first_row < rows;
    const int w_first_pos = w_first_row / G + pos_off;
    const int w_last_pos = (wave_live ? w_last_row / G : 0) + pos_off;

    Core::accum o[KS];
#pragma unroll
    for (int n = 0; n < KS; ++n) o[n] = Core::accum{};
    float m_run = -INFINITY;
    float l_run = 0.f;  // this lane's share of the query's sum

    const _Float16* Kb = K + (long long) kvh * kv_head_stride;
    const _Float16* Vb = V + (long long) kvh * kv_head_stride;
    const _Float16* q_row = Qs[wave] + qi * LD;

    for (int t0 = kv_begin; t0 < kv_end; t0 += BN) {
        // Every wave is done reading the previous tile.
        __syncthreads();
        constexpr int CHUNKS = HD / 8;
        for (int i = tid; i < BN * CHUNKS; i += NT) {
            const int kr = i / CHUNKS, c = (i % CHUNKS) * 8;
            const int t = t0 + kr;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (t < seq_kv) {
                kv = *reinterpret_cast<const uint4*>(Kb + (long long) t * kv_tok_stride + c);
                vv = *reinterpret_cast<const uint4*>(Vb + (long long) t * kv_tok_stride + c);
            }
            *reinterpret_cast<uint4*>(Ks + kr * LD + c) = kv;
            *reinterpret_cast<uint4*>(Vs + kr * LD + c) = vv;
        }
        __syncthreads();

        const bool wave_tile = wave_live && t0 <= w_last_pos &&
                               (window <= 0 || t0 + BN - 1 + window >= w_first_pos);
        if (!wave_tile) continue;

        Core::accum s[SM];
#pragma unroll
        for (int m = 0; m < SM; ++m) s[m] = Core::accum{};
#pragma unroll
        for (int kk = 0; kk < KS; ++kk) {
            const Core::frag qb = mma16::load_row(q_row + kk * 16, lane);
#pragma unroll
            for (int m = 0; m < SM; ++m) {
                const Core::frag ka = mma16::load_row(Ks + (m * 16 + qi) * LD + kk * 16, lane);
                s[m] = mma16::mma<__half>(ka, qb, s[m]);
            }
        }

        // Mask only where the tile crosses the causal diagonal, the window edge or the end of
        // the history.
        const bool need_mask = (t0 + BN - 1 > w_first_pos) || (t0 + BN > seq_kv) ||
                               (window > 0 && t0 + window < w_last_pos);
        if (need_mask) {
#pragma unroll
            for (int m = 0; m < SM; ++m)
#pragma unroll
                for (int e = 0; e < Core::acc; ++e) {
                    const int j = t0 + m * 16 + Core::row(lane, e);
                    const bool keep = r_ok && j < seq_kv && j <= r_pos &&
                                      (window <= 0 || j + window >= r_pos);
                    if (!keep) s[m][e] = -INFINITY;
                }
        }
        float mx = -INFINITY;
#pragma unroll
        for (int m = 0; m < SM; ++m)
#pragma unroll
            for (int e = 0; e < Core::acc; ++e) mx = fmaxf(mx, s[m][e]);
        mx = mma16::column_max(mx);
        const float m_new = fmaxf(m_run, mx);
        // A row with no visible key yet keeps a zero reference so no inf - inf.
        const float m_ref = (m_new == -INFINITY) ? 0.f : m_new;
        const float alpha = exp2f(m_run - m_ref);
        float sum = 0.f;
#pragma unroll
        for (int m = 0; m < SM; ++m)
#pragma unroll
            for (int e = 0; e < Core::acc; ++e) {
                const float p = exp2f(s[m][e] - m_ref);
                s[m][e] = p;
                sum += p;
            }
        l_run = l_run * alpha + sum;
        m_run = m_new;
#pragma unroll
        for (int n = 0; n < KS; ++n) o[n] *= alpha;

#pragma unroll
        for (int m = 0; m < SM; ++m) {
            const Core::frag pb = mma16::accum_as_operand<__half>(s[m], lane);
#pragma unroll
            for (int n = 0; n < KS; ++n) {
                const Core::frag va = mma16::load_column(Vs + m * 16 * LD + n * 16 + qi, LD, lane);
                o[n] = mma16::mma<__half>(va, pb, o[n]);
            }
        }
    }

    const float l = mma16::column_sum(l_run);
    if (!r_ok) return;
    const float inv = l > 0.f ? 1.f / l : 0.f;
    // The row's maximum score, back from base 2, and its sum, for a caller merging ranges.
    if (ML != nullptr && lane < 16) {
        float* ml = ML + ((long long) r_tok * n_head + r_head) * 2;
        ml[0] = m_run * 0.6931471805599453f;
        ml[1] = l;
    }
    float* orow = O + ((long long) r_tok * n_head + r_head) * HD;
#pragma unroll
    for (int n = 0; n < KS; ++n)
#pragma unroll
        for (int e = 0; e < Core::acc; ++e) orow[n * 16 + Core::row(lane, e)] = o[n][e] * inv;
}

template <int HD, int NW>
int launch(const float* Q, const _Float16* K, const _Float16* V, float* O, int n_head, int n_kv,
           int seq_q, int seq_kv, long long tok_stride, long long head_stride, float scale,
           int window, int pos_off, float* ML, hipStream_t stream) {
    constexpr int BM = NW * 16;
    const int G = n_head / n_kv;
    const dim3 grid((seq_q * G + BM - 1) / BM, n_kv, 1);
    const float log2e = 1.4426950408889634f;
    flash_prefill_gqa_amd<HD, NW><<<grid, NW * Core::wave, 0, stream>>>(
        Q, K, V, O, n_head, n_kv, seq_q, seq_kv, tok_stride, head_stride, scale * log2e, window,
        pos_off, ML);
    return hipGetLastError() == hipSuccess ? 0 : 4;
}

int dispatch(const void* Q, const void* K, const void* V, void* O, float* ML, int n_head,
             int n_kv, int seq_q, int seq_kv, int head_dim, long long kv_tok_stride,
             long long kv_head_stride, float scale, int window, int pos_off,
             long long stream_i64) {
    hipStream_t stream = (hipStream_t) stream_i64;
    const float* q = (const float*) Q;
    const _Float16* k = (const _Float16*) K;
    const _Float16* v = (const _Float16*) V;
    float* o = (float*) O;
    switch (head_dim) {
        case 64:
            return launch<64, 4>(q, k, v, o, n_head, n_kv, seq_q, seq_kv, kv_tok_stride,
                                 kv_head_stride, scale, window, pos_off, ML, stream);
        case 128:
            return launch<128, 4>(q, k, v, o, n_head, n_kv, seq_q, seq_kv, kv_tok_stride,
                                  kv_head_stride, scale, window, pos_off, ML, stream);
        case 256:
            return launch<256, 2>(q, k, v, o, n_head, n_kv, seq_q, seq_kv, kv_tok_stride,
                                  kv_head_stride, scale, window, pos_off, ML, stream);
        default:
            return 1;
    }
}

}  // namespace

// Returns 0 on launch, 1 for an unsupported head size, 4 for a launch error.
extern "C" int loken_flash_prefill_gqa_f16kv(const void* Q, const void* K, const void* V,
                                             void* O, int n_head, int n_kv, int seq_q,
                                             int seq_kv, int head_dim, long long kv_tok_stride,
                                             long long kv_head_stride, float scale, int window,
                                             long long stream_i64) {
    return dispatch(Q, K, V, O, nullptr, n_head, n_kv, seq_q, seq_kv, head_dim, kv_tok_stride,
                    kv_head_stride, scale, window, seq_kv - seq_q, stream_i64);
}

extern "C" int loken_flash_prefill_gqa_parts_f16kv(const void* Q, const void* K, const void* V,
                                                   void* O, float* ML, int n_head, int n_kv,
                                                   int seq_q, int seq_kv, int head_dim,
                                                   long long kv_tok_stride,
                                                   long long kv_head_stride, float scale,
                                                   int pos_off, long long stream_i64) {
    return dispatch(Q, K, V, O, ML, n_head, n_kv, seq_q, seq_kv, head_dim, kv_tok_stride,
                    kv_head_stride, scale, 0, pos_off, stream_i64);
}
