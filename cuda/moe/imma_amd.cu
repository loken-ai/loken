// The integer matrix products for AMD cards: the entry points of `q4k_mmvq_imma.cu`,
// `dense_q4k_imma_m8.cu`, `dense_q5k_imma_m8.cu`, `moe_q4k_imma_m8.cu` and
// `moe_q4k_imma_m8_down.cu`, with the same arguments and the same results, on AMD's integer
// matrix instructions.
//
//   weights     Q4_K or Q5_K rows [N, K], or an expert stack of them
//   activation  Q8_1 rows [tokens, K/32], one per token or per routed pair
//   output      F32 [tokens, N]
//
// A wave owns a tile of 16 weight rows by `16 MT` tokens. The matrix product takes the
// activation as its first operand and the weights as its second, so a lane's accumulator
// column is one weight row (`lane % 16`) and its accumulator elements are tokens. Every k-quant
// sub-block of 32 values carries one scale per weight row and every q8_1 block one scale per
// token, so the integer products are summed over exactly one sub-block, then scaled into f32:
//
//   out[t][w] += d_w sc_w(s) d_x(t) Σ q_w q_x  -  dmin_w m_w(s) (d_x Σ q_x)(t)
//
// Within one sub-block both operands take the same k order, which leaves the dot product
// unchanged; each lane reads `span` contiguous bytes of the 32 at `offset(lane)`.
//
//   family  wave  span  instructions per sub-block       accumulator row of element r
//   RDNA4    32    16   two 16x16x16 (8 bytes a lane)     8 (lane/16) + r
//   RDNA3    32    32   two 16x16x16 (16 bytes a lane)    2 r + lane/16
//   CDNA3    64     8   one 16x16x32                      4 (lane/16) + r
//   CDNA     64     8   two 16x16x16 (4 bytes a lane)     4 (lane/16) + r
//
// A token's q8_1 scale and block sum are read by the lane that holds its activation row and
// handed through shared memory to the lanes whose accumulators hold that token; `slot` places
// the tokens so that those of one lane's accumulator are contiguous.

#include "gguf.cuh"
#include <hip/hip_runtime.h>
#include <cstdint>

namespace loken_imma_amd {

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

/// The integer matrix instruction of the target being compiled, and its fragment geometry.
struct Core {
#if defined(RDNA4)
    static constexpr int wave = 32, acc = 8, span = 16;
    static __device__ __forceinline__ int row(int lane, int r) { return 8 * (lane >> 4) + r; }
    static __device__ __forceinline__ int slot(int t) { return t; }
#elif defined(RDNA3)
    static constexpr int wave = 32, acc = 8, span = 32;
    static __device__ __forceinline__ int row(int lane, int r) { return 2 * r + (lane >> 4); }
    static __device__ __forceinline__ int slot(int t) { return 8 * (t & 1) + (t >> 1); }
#elif defined(CDNA)
    static constexpr int wave = 64, acc = 4, span = 8;
    static __device__ __forceinline__ int row(int lane, int r) { return 4 * (lane >> 4) + r; }
    static __device__ __forceinline__ int slot(int t) { return t; }
#else
    // No integer matrix instruction: the kernel body stops the wave, and the host side never
    // selects these kernels for such a card.
    static constexpr int wave = 32, acc = 8, span = 16;
    static __device__ __forceinline__ int row(int lane, int r) { return 8 * (lane >> 4) + r; }
    static __device__ __forceinline__ int slot(int t) { return t; }
#endif
    /// 32-bit words a lane holds of one operand for one sub-block.
    static constexpr int words = span / 4;
    /// Lane groups of sixteen that hold different parts of the sub-block.
    static constexpr int groups = 32 / span;
    typedef int accum __attribute__((ext_vector_type(acc)));

    /// The byte of the sub-block a lane's share starts at.
    static __device__ __forceinline__ int offset(int lane) {
        return span * ((lane >> 4) % groups);
    }

    /// `Σ a.b` over one sub-block: the activation words `a` and the weight words `b`.
    static __device__ __forceinline__ accum dot(const int (&a)[words], const int (&b)[words]) {
        accum c = {};
#if defined(RDNA4)
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, v2i{a[0], a[1]}, true,
                                                             v2i{b[0], b[1]}, c, false);
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, v2i{a[2], a[3]}, true,
                                                             v2i{b[2], b[3]}, c, false);
#elif defined(RDNA3)
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, v4i{a[0], a[1], a[2], a[3]}, true,
                                                       v4i{b[0], b[1], b[2], b[3]}, c, false);
        c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, v4i{a[4], a[5], a[6], a[7]}, true,
                                                       v4i{b[4], b[5], b[6], b[7]}, c, false);
#elif defined(CDNA3)
        const long a64 = (long) (uint32_t) a[0] | ((long) a[1] << 32);
        const long b64 = (long) (uint32_t) b[0] | ((long) b[1] << 32);
        c = __builtin_amdgcn_mfma_i32_16x16x32_i8(a64, b64, c, 0, 0, 0);
#elif defined(CDNA)
        c = __builtin_amdgcn_mfma_i32_16x16x16i8(a[0], b[0], c, 0, 0, 0);
        c = __builtin_amdgcn_mfma_i32_16x16x16i8(a[1], b[1], c, 0, 0, 0);
#else
        (void) a;
        (void) b;
        __builtin_trap();
#endif
        return c;
    }
};

/// Weight rows per tile, and tokens per accumulator tile.
constexpr int tile_rows = 16;
constexpr int tile_tokens = 16;
/// Sub-blocks of 32 in a k-quant superblock.
constexpr int sub_blocks = QK_K / 32;

/// `words` 32-bit words from `p`, which is aligned to the width it is read at.
template <int words>
__device__ __forceinline__ void load_aligned(const uint8_t* __restrict__ p, uint32_t (&w)[words]) {
    if constexpr (words % 4 == 0) {
#pragma unroll
        for (int i = 0; i < words / 4; ++i) {
            const uint4 v = reinterpret_cast<const uint4*>(p)[i];
            w[4 * i] = v.x;
            w[4 * i + 1] = v.y;
            w[4 * i + 2] = v.z;
            w[4 * i + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < words / 2; ++i) {
            const uint2 v = reinterpret_cast<const uint2*>(p)[i];
            w[2 * i] = v.x;
            w[2 * i + 1] = v.y;
        }
    }
}

/// The two k-quant formats these kernels read: Q4_K, and Q5_K with its fifth-bit plane.
template <bool Fifth>
struct Format;

template <>
struct Format<false> {
    typedef block_q4_K block;
    /// The fifth-bit words of a lane's share: none.
    static __device__ __forceinline__ void high(const block*, int, uint32_t (&h)[Core::words]) {
#pragma unroll
        for (int i = 0; i < Core::words; ++i) h[i] = 0;
    }
    /// The codes of sub-block `s` from the nibble words `q` its byte chunk holds.
    static __device__ __forceinline__ int code(uint32_t q, uint32_t, int s) {
        return (int) ((q >> (4 * (s & 1))) & 0x0F0F0F0Fu);
    }
};

template <>
struct Format<true> {
    typedef block_q5_K block;
    /// The fifth-bit plane bytes at the lane's offset: byte `l` holds bit `s` of value `l` of
    /// every sub-block `s`.
    static __device__ __forceinline__ void high(const block* b, int off,
                                                uint32_t (&h)[Core::words]) {
        load_aligned<Core::words>(b->qh + off, h);
    }
    static __device__ __forceinline__ int code(uint32_t q, uint32_t h, int s) {
        return (int) (((q >> (4 * (s & 1))) & 0x0F0F0F0Fu) | (((h >> s) << 4) & 0x10101010u));
    }
};

/// The two halves of a raw half2 word, as floats.
__device__ __forceinline__ float half_lo(uint32_t v) {
    return (float) __builtin_bit_cast(_Float16, (uint16_t) (v & 0xffffu));
}
__device__ __forceinline__ float half_hi(uint32_t v) {
    return (float) __builtin_bit_cast(_Float16, (uint16_t) (v >> 16));
}

/// The products of `NW` weight rows with `MT` tiles of token rows, accumulated over the whole
/// of k into `out[nw][mt][r]`: weight row `w[nw]` against the token held at accumulator element
/// `r` of tile `mt`. `x[mt]` is the q8_1 row of the lane's own token in tile `mt`; a tile whose
/// bit is clear in `tiles` is skipped. Every pointer names a row that exists: a lane past the
/// matrix or past the tokens reads a row in range, and its results land only in outputs nobody
/// writes, so no load waits on a branch.
///
/// The lanes of the first group put their tokens' q8_1 scale words of a superblock in shared
/// memory, at the positions that make the tokens of a lane's accumulator contiguous.
template <bool Fifth, int NW, int MT>
__device__ __forceinline__ void tile_dot(const typename Format<Fifth>::block* const (&w)[NW],
                                         const block_q8_1* const (&x)[MT], int num_super, int lane,
                                         float (&out)[NW][MT][Core::acc],
                                         unsigned tiles = ~0u) {
    using F = Format<Fifth>;
    constexpr int words = Core::words;
    __shared__ uint32_t scales[MT][sub_blocks][tile_tokens];
    const int off = Core::offset(lane);
    const int mine = Core::slot(lane & 15);
    const int first = Core::slot(Core::row(lane, 0));
#pragma unroll
    for (int nw = 0; nw < NW; ++nw)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int r = 0; r < Core::acc; ++r) out[nw][mt][r] = 0.f;

    for (int isb = 0; isb < num_super; ++isb) {
        __syncthreads();
        if (lane < tile_tokens) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int s = 0; s < sub_blocks; ++s)
                    scales[mt][s][mine] =
                        *reinterpret_cast<const uint32_t*>(&x[mt][isb * sub_blocks + s].ds);
        }

        float dsc[NW][sub_blocks], dmm[NW][sub_blocks];
        uint32_t qh[NW][words];
#pragma unroll
        for (int nw = 0; nw < NW; ++nw) {
            const typename F::block* b = w[nw] + isb;
            const uint32_t dm = *reinterpret_cast<const uint32_t*>(&b->dm);
            const float d = half_lo(dm), dmin = half_hi(dm);
#pragma unroll
            for (int s = 0; s < sub_blocks; ++s) {
                uint8_t sc, m;
                get_scale_min_k4(s, b->scales, sc, m);
                dsc[nw][s] = d * (float) sc;
                dmm[nw][s] = dmin * (float) m;
            }
            F::high(b, off, qh[nw]);
        }
        __syncthreads();

#pragma unroll
        for (int chunk = 0; chunk < sub_blocks / 2; ++chunk) {
            // One chunk of 32 bytes holds the codes of sub-blocks `2 chunk` (low nibbles) and
            // `2 chunk + 1` (high nibbles).
            uint32_t q[NW][words];
#pragma unroll
            for (int nw = 0; nw < NW; ++nw)
                load_aligned<words>(w[nw][isb].qs + 32 * chunk + off, q[nw]);
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int s = 2 * chunk + half;
                int b[NW][words];
#pragma unroll
                for (int nw = 0; nw < NW; ++nw)
#pragma unroll
                    for (int i = 0; i < words; ++i) b[nw][i] = F::code(q[nw][i], qh[nw][i], s);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (!((tiles >> mt) & 1)) continue;
                    int a[words];
                    const int* qs =
                        reinterpret_cast<const int*>(x[mt][isb * sub_blocks + s].qs + off);
#pragma unroll
                    for (int i = 0; i < words; ++i) a[i] = qs[i];
                    uint32_t ds[Core::acc];
#pragma unroll
                    for (int r = 0; r < Core::acc; r += 4) {
                        const uint4 v = *reinterpret_cast<const uint4*>(&scales[mt][s][first + r]);
                        ds[r] = v.x;
                        ds[r + 1] = v.y;
                        ds[r + 2] = v.z;
                        ds[r + 3] = v.w;
                    }
#pragma unroll
                    for (int nw = 0; nw < NW; ++nw) {
                        const typename Core::accum c = Core::dot(a, b[nw]);
#pragma unroll
                        for (int r = 0; r < Core::acc; ++r)
                            out[nw][mt][r] += dsc[nw][s] * half_lo(ds[r]) * (float) c[r] -
                                              dmm[nw][s] * half_hi(ds[r]);
                    }
                }
            }
        }
    }
}

/// The epilogues of the gate and up pair.
enum Act { identity = 0, silu = 1, gelu_tanh = 2 };

template <int A>
__device__ __forceinline__ float activate(float g) {
    if constexpr (A == silu) {
        return g / (1.f + __expf(-g));
    } else if constexpr (A == gelu_tanh) {
        // gelu(x) = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
        const float k0 = 0.7978845608028654f;
        const float k1 = 0.044715f;
        return 0.5f * g * (1.f + tanhf(k0 * (g + k1 * g * g * g)));
    } else {
        return g;
    }
}

/// `dst = Act(gate . x) * (up . x)`, or `dst = Act(gate . x)` without `up`, for dense rows.
template <bool Fifth, bool WithUp, int A, int MT>
__global__ void __launch_bounds__(Core::wave)
    dense_kernel(const void* __restrict__ gate_w, const void* __restrict__ up_w,
                 const void* __restrict__ inputs_q81, float* __restrict__ dst, int size_m, int N,
                 int K) {
    using block = typename Format<Fifth>::block;
    constexpr int NW = WithUp ? 2 : 1;
    const int lane = threadIdx.x;
    const int num_super = K / QK_K;
    const int row = blockIdx.x * tile_rows + (lane & 15);
    const int tok0 = blockIdx.y * tile_tokens * MT;
    const bool live = row < N;

    const size_t at = (size_t) (live ? row : N - 1) * num_super;
    const block* w[NW];
    w[0] = (const block*) gate_w + at;
    if constexpr (WithUp) w[NW - 1] = (const block*) up_w + at;
    const block_q8_1* x[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int t = min(tok0 + mt * tile_tokens + (lane & 15), size_m - 1);
        x[mt] = (const block_q8_1*) inputs_q81 + (size_t) t * num_super * sub_blocks;
    }

    float out[NW][MT][Core::acc];
    tile_dot<Fifth, NW, MT>(w, x, num_super, lane, out);

    if (!live) return;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int r = 0; r < Core::acc; ++r) {
            const int t = tok0 + mt * tile_tokens + Core::row(lane, r);
            if (t >= size_m) continue;
            const float g = activate<A>(out[0][mt][r]);
            dst[(size_t) t * N + row] = WithUp ? g * out[NW - 1][mt][r] : g;
        }
}

/// Whether pair `p` of a window is the first of the window to name expert `e`, so that one pass
/// per distinct expert serves every pair once.
__device__ __forceinline__ bool first_of_expert(const int32_t* __restrict__ expert_ids,
                                                int pair0, int p, int e) {
    for (int q = 0; q < p; ++q)
        if (expert_ids[pair0 + q] == e) return false;
    return true;
}

/// The MoE expert products over a window of `16 MT` routed pairs, one pass per distinct expert
/// in the window; a pass multiplies only the tiles holding a pair of its expert and writes only
/// those pairs, one output row per pair at its slot.
///
///   GateUp  `w` is `[experts, 2N, K]`, the gate rows first; the activation is one q8_1 row per
///           real token; the row is `gelu(gate) * up`
///   down    `w` is `[experts, N, K]`; the activation is one q8_1 row per routed pair, at its
///           slot; the row is scaled by the pair's top-k weight
template <bool GateUp, int MT>
__global__ void __launch_bounds__(Core::wave)
    moe_kernel(const void* __restrict__ weights, const void* __restrict__ inputs_q81,
               const int32_t* __restrict__ sorted_token_ids,
               const int32_t* __restrict__ expert_ids, const float* __restrict__ topk_weights,
               float* __restrict__ dst, int num_experts, int topk, int size_m, int N, int K) {
    constexpr int NW = GateUp ? 2 : 1;
    const int lane = threadIdx.x;
    const int num_super = K / QK_K;
    const int row = blockIdx.x * tile_rows + (lane & 15);
    const int pair0 = blockIdx.y * tile_tokens * MT;
    const int window = min(tile_tokens * MT, size_m - pair0);
    const bool live = row < N;

    int my_slot[MT], my_expert[MT];
    const block_q8_1* x[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int pair = pair0 + mt * tile_tokens + (lane & 15);
        const bool here = pair < size_m;
        my_slot[mt] = here ? sorted_token_ids[pair] : 0;
        my_expert[mt] = here ? expert_ids[pair] : -1;
        const int row_x = GateUp ? my_slot[mt] / topk : my_slot[mt];
        x[mt] = (const block_q8_1*) inputs_q81 + (size_t) row_x * num_super * sub_blocks;
    }

#pragma unroll 1
    for (int p = 0; p < window; ++p) {
        const int named = expert_ids[pair0 + p];
        if (p > 0 && named == expert_ids[pair0 + p - 1]) continue;
        // The same expert on every lane, which keeps its weight addresses scalar.
        const int e = __builtin_amdgcn_readfirstlane(named);
        if (e < 0 || e >= num_experts || !first_of_expert(expert_ids, pair0, p, e)) continue;

        unsigned tiles = 0;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
            if (__ballot(my_expert[mt] == e)) tiles |= 1u << mt;

        const size_t first_row = (size_t) e * NW * N + (live ? row : N - 1);
        const block_q4_K* w[NW];
#pragma unroll
        for (int nw = 0; nw < NW; ++nw)
            w[nw] = (const block_q4_K*) weights + (first_row + (size_t) nw * N) * num_super;
        float out[NW][MT][Core::acc];
        tile_dot<false, NW, MT>(w, x, num_super, lane, out, tiles);

#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (!((tiles >> mt) & 1)) continue;
#pragma unroll
            for (int r = 0; r < Core::acc; ++r) {
                const int q = Core::row(lane, r);
                const int slot = __shfl(my_slot[mt], q);
                const int expert = __shfl(my_expert[mt], q);
                if (!live || pair0 + mt * tile_tokens + q >= size_m || expert != e) continue;
                dst[(size_t) slot * N + row] =
                    GateUp ? activate<gelu_tanh>(out[0][mt][r]) * out[NW - 1][mt][r]
                           : out[0][mt][r] * topk_weights[slot];
            }
        }
    }
}

/// The token tiles a wave takes: more tiles share each weight load, fewer leave less of the
/// last tile empty.
inline int token_tiles(int size_m) {
    return size_m >= 4 * tile_tokens ? 4 : size_m > tile_tokens ? 2 : 1;
}

template <bool GateUp>
void launch_moe(const void* weights, const void* inputs_q81, const int32_t* sorted_token_ids,
                const int32_t* expert_ids, const float* topk_weights, float* dst, int num_experts,
                int topk, int size_m, int N, int K, hipStream_t stream) {
    const int mt = token_tiles(size_m);
    const dim3 grid((N + tile_rows - 1) / tile_rows,
                    (size_m + tile_tokens * mt - 1) / (tile_tokens * mt));
    const dim3 block(Core::wave);
    if (mt == 4) {
        moe_kernel<GateUp, 4><<<grid, block, 0, stream>>>(weights, inputs_q81, sorted_token_ids,
                                                          expert_ids, topk_weights, dst,
                                                          num_experts, topk, size_m, N, K);
    } else if (mt == 2) {
        moe_kernel<GateUp, 2><<<grid, block, 0, stream>>>(weights, inputs_q81, sorted_token_ids,
                                                          expert_ids, topk_weights, dst,
                                                          num_experts, topk, size_m, N, K);
    } else {
        moe_kernel<GateUp, 1><<<grid, block, 0, stream>>>(weights, inputs_q81, sorted_token_ids,
                                                          expert_ids, topk_weights, dst,
                                                          num_experts, topk, size_m, N, K);
    }
}

template <bool Fifth, bool WithUp, int A>
void launch_dense(const void* gate_w, const void* up_w, const void* inputs_q81, float* dst,
                  int size_m, int N, int K, hipStream_t stream) {
    const int mt = token_tiles(size_m);
    const dim3 grid((N + tile_rows - 1) / tile_rows,
                    (size_m + tile_tokens * mt - 1) / (tile_tokens * mt));
    const dim3 block(Core::wave);
    if (mt == 4) {
        dense_kernel<Fifth, WithUp, A, 4>
            <<<grid, block, 0, stream>>>(gate_w, up_w, inputs_q81, dst, size_m, N, K);
    } else if (mt == 2) {
        dense_kernel<Fifth, WithUp, A, 2>
            <<<grid, block, 0, stream>>>(gate_w, up_w, inputs_q81, dst, size_m, N, K);
    } else {
        dense_kernel<Fifth, WithUp, A, 1>
            <<<grid, block, 0, stream>>>(gate_w, up_w, inputs_q81, dst, size_m, N, K);
    }
}

template <bool Fifth>
void launch_actmul(const void* gate_w, const void* up_w, const void* inputs_q81, float* dst,
                   int size_m, int N, int K, int use_gelu, hipStream_t stream) {
    if (use_gelu) {
        launch_dense<Fifth, true, gelu_tanh>(gate_w, up_w, inputs_q81, dst, size_m, N, K, stream);
    } else {
        launch_dense<Fifth, true, silu>(gate_w, up_w, inputs_q81, dst, size_m, N, K, stream);
    }
}

}  // namespace loken_imma_amd

extern "C" void loken_moe_sum_topk(const float* pairs, float* outputs, int tokens, int topk,
                                   int cols, hipStream_t stream);

// The Q4_K mat-vec of one q8_1 row: the dense projection at one token.
extern "C" void q4k_mmvq_imma(const void* vx, const void* vy, void* dst, int ncols_x, int nrows_x,
                              hipStream_t stream) {
    if (ncols_x <= 0 || nrows_x <= 0) return;
    loken_imma_amd::launch_dense<false, false, loken_imma_amd::identity>(
        vx, nullptr, vy, static_cast<float*>(dst), 1, nrows_x, ncols_x, stream);
}

extern "C" void loken_dense_q4k_imma_m8_actmul(const void* gate_w, const void* up_w,
                                               const void* inputs_q81, float* dst_f32, int size_m,
                                               int N, int K, int use_gelu, hipStream_t stream) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    loken_imma_amd::launch_actmul<false>(gate_w, up_w, inputs_q81, dst_f32, size_m, N, K,
                                         use_gelu, stream);
}

extern "C" void loken_dense_q4k_imma_m8_plain(const void* w, const void* inputs_q81,
                                              float* dst_f32, int size_m, int N, int K,
                                              hipStream_t stream) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    loken_imma_amd::launch_dense<false, false, loken_imma_amd::identity>(
        w, nullptr, inputs_q81, dst_f32, size_m, N, K, stream);
}

extern "C" void loken_dense_q5k_imma_m8_plain(const void* w, const void* inputs_q81,
                                              float* dst_f32, int size_m, int N, int K,
                                              hipStream_t stream) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    loken_imma_amd::launch_dense<true, false, loken_imma_amd::identity>(
        w, nullptr, inputs_q81, dst_f32, size_m, N, K, stream);
}

extern "C" void loken_dense_q5k_imma_m8_actmul(const void* gate_w, const void* up_w,
                                               const void* inputs_q81, float* dst_f32, int size_m,
                                               int N, int K, int use_gelu, hipStream_t stream) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    loken_imma_amd::launch_actmul<true>(gate_w, up_w, inputs_q81, dst_f32, size_m, N, K,
                                        use_gelu, stream);
}

extern "C" void loken_moe_q4k_imma_m8_gate_up(const void* gate_up_w, const void* inputs_q81,
                                              const int32_t* sorted_token_ids,
                                              const int32_t* expert_ids, float* dst_f32,
                                              int num_experts, int topk, int size_m, int N, int K,
                                              hipStream_t stream) {
    if (size_m <= 0 || N <= 0 || K <= 0) return;
    loken_imma_amd::launch_moe<true>(gate_up_w, inputs_q81, sorted_token_ids, expert_ids, nullptr,
                                     dst_f32, num_experts, topk, size_m, N, K, stream);
}

// One row per routed pair, folded into the tokens in slot order afterwards, as the CUDA
// launcher does: the same sum on every run. The rows are a stream-ordered allocation.
extern "C" void loken_moe_q4k_imma_m8_down(const void* down_w, const void* inputs_q81,
                                           const int32_t* sorted_token_ids,
                                           const int32_t* expert_ids, const float* topk_weights,
                                           float* dst_f32, int num_experts, int topk, int size_m,
                                           int hidden, int K, hipStream_t stream) {
    using namespace loken_imma_amd;
    if (size_m <= 0 || hidden <= 0 || K <= 0) return;
    float* pairs = nullptr;
    const size_t pair_bytes = (size_t) size_m * hidden * sizeof(float);
    if (hipMallocAsync(reinterpret_cast<void**>(&pairs), pair_bytes, stream) != hipSuccess) return;
    (void) hipMemsetAsync(pairs, 0, pair_bytes, stream);
    launch_moe<false>(down_w, inputs_q81, sorted_token_ids, expert_ids, topk_weights, pairs,
                      num_experts, topk, size_m, hidden, K, stream);
    loken_moe_sum_topk(pairs, dst_f32, size_m / topk, topk, hidden, stream);
    (void) hipFreeAsync(pairs, stream);
}
