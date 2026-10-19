// Non-causal flash attention for diffusion transformers, BF16, for AMD cards: the entry points
// of `flash_dit_bf16.cu`, with the same arguments, layouts, radial mask and results, on AMD's
// 16-bit matrix instructions (`cuda/hip/amd_mma16.cuh`).
//
//   Q [b, n_head, seq_q, HD]   K/V [b, n_head, seq_kv, HD]   O [b, n_head, seq_q, HD]
//
// all contiguous BF16, no GQA. The online softmax and the accumulator run in F32; the two
// matrix products and the probabilities are BF16.
//
// The mask is decided per key tile of `BLOCK_N` keys for spans of `BLOCK_M` and of `WARP_M`
// query rows, aligned to their size, as the NVIDIA kernel decides it for its block and for
// each of its warps: a query row sees a tile when both spans holding it keep the tile.
//
// A block owns `WAVES * 16` query rows of one head and stages each K/V tile once for all of
// them. A wave owns 16 rows and keeps its scores and its output in registers, transposed:
// `S^T = K Q^T` puts one query in each accumulator column, which is the layout `O^T = V^T P^T`
// takes P in, so a query's running maximum, sum and rescale are all the lane's own.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

#include "amd_mma16.cuh"

namespace {

using loken_amd_mma16::Core;
namespace mma16 = loken_amd_mma16;

#define BLOCK_N 32
#define WARP_M 32
#define BLOCK_M 256
#define WAVES 8
#define NATIVE_FRAME_PATCHES 1560
#define GLOBAL_TILE_STRIDE 8

constexpr int PAD = 8;  // bf16 of padding per shared row

// Whether a span of query rows `[q_lo, q_hi]` sees the key tile `[k_lo, k_hi]` under the radial
// mask (arXiv 2506.19852): full attention between neighbouring frames, a window halving with
// the temporal band beyond, measured in two dimensions over a frame `grid_w` patches wide and
// bounded at band 0 by `NATIVE_FRAME_PATCHES`, one tile in `GLOBAL_TILE_STRIDE` always kept.
// `frame_tokens == 0` keeps every tile.
__device__ __forceinline__ bool radial_tile_kept(
    int q_lo, int q_hi, int k_lo, int k_hi, int frame_tokens, int grid_w, int tile_index)
{
    if (frame_tokens <= 0 || grid_w <= 0) return true;
    // A coarse view of the whole frame, whatever the distance.
    if ((tile_index % GLOBAL_TILE_STRIDE) == 0) return true;

    const int fq_lo = q_lo / frame_tokens, fq_hi = q_hi / frame_tokens;
    const int fk_lo = k_lo / frame_tokens, fk_hi = k_hi / frame_tokens;
    int d = 0;
    if (fk_lo > fq_hi)      d = fk_lo - fq_hi;
    else if (fq_lo > fk_hi) d = fq_lo - fk_hi;
    int band = 0;
    for (int t = d; t > 1; t >>= 1) ++band;        // floor(log2(d)), 0 for d <= 1

    // Radius in PATCHES, halving with the temporal band exactly as the density does, and
    // bounded at band 0 by the frame the model was trained on.
    const int grid_h = (frame_tokens + grid_w - 1) / grid_w;
    int r = (grid_w >> band) / 2;
    if (band == 0) {
        // Largest odd box holding at most NATIVE_FRAME_PATCHES tokens.
        int side = 1;
        while ((side + 2) * (side + 2) <= NATIVE_FRAME_PATCHES) side += 2;
        const int r_native = side / 2;
        if (r > r_native) r = r_native;
    }
    if (r < 1) r = 1;
    if (r >= grid_w && r >= grid_h) return true;   // the window covers the frame

    // A span that crosses a frame boundary touches every position in one of them.
    if (fq_hi > fq_lo || fk_hi > fk_lo) return true;
    const int sq_lo = q_lo % frame_tokens, sq_hi = q_hi % frame_tokens;
    const int sk_lo = k_lo % frame_tokens, sk_hi = k_hi % frame_tokens;
    const int qr0 = sq_lo / grid_w, qr1 = sq_hi / grid_w;
    const int kr0 = sk_lo / grid_w, kr1 = sk_hi / grid_w;
    int dr = 0;
    if (kr0 > qr1)      dr = kr0 - qr1;
    else if (qr0 > kr1) dr = qr0 - kr1;
    if (dr > r) return false;
    // A span of more than one row covers every column, so only a single-row span can be
    // declined on its columns.
    if (qr1 > qr0 || kr1 > kr0) return true;
    const int qc0 = sq_lo % grid_w, qc1 = sq_hi % grid_w;
    const int kc0 = sk_lo % grid_w, kc1 = sk_hi % grid_w;
    int dc = 0;
    if (kc0 > qc1)      dc = kc0 - qc1;
    else if (qc0 > kc1) dc = qc0 - kc1;
    return dc <= r;
}

template <int HD>
__global__ void __launch_bounds__(WAVES * Core::wave)
    flash_dit_bf16_amd(const uint16_t* __restrict__ Q, const uint16_t* __restrict__ K,
                       const uint16_t* __restrict__ V, __hip_bfloat16* __restrict__ O,
                       int n_head, int seq_q, int seq_kv, float scale, int frame_tokens,
                       int grid_w) {
    constexpr int KS = HD / 16;       // k steps of K Q^T, and d tiles of the output
    constexpr int SM = BLOCK_N / 16;  // key tiles of S^T
    constexpr int LD = HD + PAD;
    constexpr int NT = WAVES * Core::wave;

    __shared__ __attribute__((aligned(16))) uint16_t Ks[BLOCK_N * LD];
    __shared__ __attribute__((aligned(16))) uint16_t Vs[BLOCK_N * LD];

    const int tid = threadIdx.x;
    const int wave = tid / Core::wave;
    const int lane = tid % Core::wave;
    const int qi = lane & 15;
    const int head = blockIdx.y;
    const int bi = blockIdx.z;

    const int q_wave = blockIdx.x * WAVES * 16 + wave * 16;
    const int q = q_wave + qi;
    const bool q_ok = q < seq_q;
    const size_t bh = (size_t) bi * n_head + head;
    const uint16_t* Kh = K + bh * seq_kv * HD;
    const uint16_t* Vh = V + bh * seq_kv * HD;

    // The spans the mask is decided for.
    const int blk_lo = blockIdx.x * WAVES * 16 / BLOCK_M * BLOCK_M;
    const int blk_hi = min(blk_lo + BLOCK_M, seq_q) - 1;
    const int warp_lo = q_wave / WARP_M * WARP_M;
    const int warp_hi = min(warp_lo + WARP_M, seq_q) - 1;
    const bool wave_live = q_wave < seq_q;

    // The lane's query row as the B operand of every k step; a row past the end is zero.
    Core::frag qb[KS];
    {
        const uint16_t* qrow = Q + (bh * seq_q + (q_ok ? q : 0)) * HD;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk)
            qb[kk] = q_ok ? mma16::load_row(qrow + kk * 16, lane) : Core::frag{};
    }

    Core::accum o[KS];
#pragma unroll
    for (int n = 0; n < KS; ++n) o[n] = Core::accum{};
    float m_run = -INFINITY;
    float l_run = 0.f;  // this lane's share of the query's sum

    for (int t0 = 0; t0 < seq_kv; t0 += BLOCK_N) {
        const int ntok = min(BLOCK_N, seq_kv - t0);
        const int tile_index = t0 / BLOCK_N;
        if (!radial_tile_kept(blk_lo, blk_hi, t0, t0 + ntok - 1, frame_tokens, grid_w,
                              tile_index))
            continue;
        // Every wave is done reading the previous tile.
        __syncthreads();
        {
            constexpr int CHUNKS = HD / 8;
            const int live = ntok * CHUNKS;
            for (int i = tid; i < BLOCK_N * CHUNKS; i += NT) {
                const int kr = i / CHUNKS, c = (i % CHUNKS) * 8;
                uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
                if (i < live) {
                    kv = *reinterpret_cast<const uint4*>(Kh + (size_t) (t0 + kr) * HD + c);
                    vv = *reinterpret_cast<const uint4*>(Vh + (size_t) (t0 + kr) * HD + c);
                }
                *reinterpret_cast<uint4*>(Ks + kr * LD + c) = kv;
                *reinterpret_cast<uint4*>(Vs + kr * LD + c) = vv;
            }
        }
        __syncthreads();
        if (!wave_live || !radial_tile_kept(warp_lo, warp_hi, t0, t0 + ntok - 1, frame_tokens,
                                            grid_w, tile_index))
            continue;

        Core::accum s[SM];
#pragma unroll
        for (int m = 0; m < SM; ++m) s[m] = Core::accum{};
#pragma unroll
        for (int kk = 0; kk < KS; ++kk)
#pragma unroll
            for (int m = 0; m < SM; ++m) {
                const Core::frag ka = mma16::load_row(Ks + (m * 16 + qi) * LD + kk * 16, lane);
                s[m] = mma16::mma<__hip_bfloat16>(ka, qb[kk], s[m]);
            }

        float mx = -INFINITY;
#pragma unroll
        for (int m = 0; m < SM; ++m)
#pragma unroll
            for (int e = 0; e < Core::acc; ++e) {
                const int j = m * 16 + Core::row(lane, e);
                const float v = (j < ntok && q_ok) ? s[m][e] * scale : -INFINITY;
                s[m][e] = v;
                mx = fmaxf(mx, v);
            }
        mx = mma16::column_max(mx);
        const float m_new = fmaxf(m_run, mx);
        float sum = 0.f;
#pragma unroll
        for (int m = 0; m < SM; ++m)
#pragma unroll
            for (int e = 0; e < Core::acc; ++e) {
                const float p = (s[m][e] == -INFINITY) ? 0.f : __expf(s[m][e] - m_new);
                s[m][e] = p;
                sum += p;
            }
        const float alpha = (m_run == -INFINITY) ? 0.f : __expf(m_run - m_new);
        m_run = m_new;
        l_run = l_run * alpha + sum;
#pragma unroll
        for (int n = 0; n < KS; ++n) o[n] *= alpha;

#pragma unroll
        for (int m = 0; m < SM; ++m) {
            const Core::frag pb = mma16::accum_as_operand<__hip_bfloat16>(s[m], lane);
#pragma unroll
            for (int n = 0; n < KS; ++n) {
                const Core::frag va = mma16::load_column(Vs + m * 16 * LD + n * 16 + qi, LD, lane);
                o[n] = mma16::mma<__hip_bfloat16>(va, pb, o[n]);
            }
        }
    }

    const float l = mma16::column_sum(l_run);
    if (!q_ok) return;
    __hip_bfloat16* orow = O + (bh * seq_q + q) * HD;
#pragma unroll
    for (int n = 0; n < KS; ++n)
#pragma unroll
        for (int e = 0; e < Core::acc; ++e)
            orow[n * 16 + Core::row(lane, e)] = __float2bfloat16(l > 0.f ? o[n][e] / l : 0.f);
}

}  // namespace

#define LAUNCH(HD)                                                                          \
    extern "C" void flash_dit_bf16_hd##HD(const void* Q, const void* K, const void* V,      \
                                          void* O, int b, int n_head, int seq_q, int seq_kv, \
                                          float scale, int frame_tokens, int grid_w,         \
                                          long long stream_i64) {                            \
        const dim3 grid((seq_q + WAVES * 16 - 1) / (WAVES * 16), n_head, b);                \
        flash_dit_bf16_amd<HD><<<grid, WAVES * Core::wave, 0, (hipStream_t) stream_i64>>>(  \
            (const uint16_t*) Q, (const uint16_t*) K, (const uint16_t*) V,                  \
            (__hip_bfloat16*) O, n_head, seq_q, seq_kv, scale, frame_tokens, grid_w);       \
    }

LAUNCH(64)
LAUNCH(128)
