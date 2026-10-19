// The 16x16x16 matrix product in 16-bit floats with an f32 accumulator, on the AMD matrix
// instruction of the target being compiled. `D[i][j] = Σ_k A[i][k] B[k][j] + C[i][j]`.
//
// A lane holds row `i = lane % 16` of A and column `j = lane % 16` of B, each over the same
// `span` consecutive values of k starting at `k0(lane)`; its accumulator holds column
// `j = lane % 16` of D at rows `row(lane, r)`.
//
//   family  wave  span  k0             accumulator row of element r   instruction
//   RDNA4    32     8   8 (lane/16)    8 (lane/16) + r                wmma 16x16x16, gfx12
//   RDNA3    32    16   0              2 r + lane/16                  wmma 16x16x16, gfx11
//   CDNA     64     4   4 (lane/16)    4 (lane/16) + r                mfma 16x16x16
//
// On RDNA4 and CDNA the rows of an accumulator are the k range of an operand, so a product's
// accumulator is, element for element, the B operand of a product whose k runs over its rows;
// RDNA3 completes it with the other half held by the lane sixteen away.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>

namespace loken_amd_mma16 {

struct Core {
#if defined(RDNA4)
    static constexpr int wave = 32, span = 8, acc = 8;
    static __device__ __forceinline__ int k0(int lane) { return 8 * ((lane >> 4) & 1); }
    static __device__ __forceinline__ int row(int lane, int r) { return 8 * ((lane >> 4) & 1) + r; }
#elif defined(RDNA3)
    static constexpr int wave = 32, span = 16, acc = 8;
    static __device__ __forceinline__ int k0(int) { return 0; }
    static __device__ __forceinline__ int row(int lane, int r) { return 2 * r + ((lane >> 4) & 1); }
#elif defined(CDNA)
    static constexpr int wave = 64, span = 4, acc = 4;
    static __device__ __forceinline__ int k0(int lane) { return 4 * ((lane >> 4) & 3); }
    static __device__ __forceinline__ int row(int lane, int r) { return 4 * ((lane >> 4) & 3) + r; }
#else
    // No matrix instruction: a product stops the wave, and the host side never selects these
    // kernels for such a card.
    static constexpr int wave = 32, span = 8, acc = 8;
    static __device__ __forceinline__ int k0(int lane) { return 8 * ((lane >> 4) & 1); }
    static __device__ __forceinline__ int row(int lane, int r) { return 8 * ((lane >> 4) & 1) + r; }
#endif
    /// One operand: `span` 16-bit floats, as their bits.
    typedef short frag __attribute__((ext_vector_type(span)));
    typedef float accum __attribute__((ext_vector_type(acc)));
};

typedef _Float16 h4 __attribute__((ext_vector_type(4)));
typedef _Float16 h8 __attribute__((ext_vector_type(8)));
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef __bf16 b8 __attribute__((ext_vector_type(8)));
typedef __bf16 b16 __attribute__((ext_vector_type(16)));
typedef short s2 __attribute__((ext_vector_type(2)));
typedef short s4 __attribute__((ext_vector_type(4)));
typedef float f4 __attribute__((ext_vector_type(4)));

/// The carrier of the operands: `half` or `__hip_bfloat16`.
template <typename T>
struct Carrier;
template <>
struct Carrier<__half> {
    static constexpr bool bf16 = false;
    static __device__ __forceinline__ short bits(float v) {
        return __builtin_bit_cast(short, (_Float16) v);
    }
};
template <>
struct Carrier<__hip_bfloat16> {
    static constexpr bool bf16 = true;
    static __device__ __forceinline__ short bits(float v) {
        return __builtin_bit_cast(short, __float2bfloat16(v));
    }
};

/// `c + a b` in the carrier `T`.
template <typename T>
__device__ __forceinline__ Core::accum mma(const Core::frag& a, const Core::frag& b,
                                           const Core::accum& c) {
#if defined(RDNA4)
    if constexpr (Carrier<T>::bf16)
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(b8, a),
                                                                 __builtin_bit_cast(b8, b), c);
    else
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(h8, a),
                                                                __builtin_bit_cast(h8, b), c);
#elif defined(RDNA3)
    if constexpr (Carrier<T>::bf16)
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(b16, a),
                                                           __builtin_bit_cast(b16, b), c);
    else
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16, a),
                                                          __builtin_bit_cast(h16, b), c);
#elif defined(CDNA)
    if constexpr (Carrier<T>::bf16) {
#if defined(CDNA1)
        // Two products of eight: each takes two of the lane's four values of k.
        Core::accum d = __builtin_amdgcn_mfma_f32_16x16x8bf16(s2{a[0], a[1]}, s2{b[0], b[1]}, c,
                                                              0, 0, 0);
        return __builtin_amdgcn_mfma_f32_16x16x8bf16(s2{a[2], a[3]}, s2{b[2], b[3]}, d, 0, 0, 0);
#else
        return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
#endif
    } else {
        return __builtin_amdgcn_mfma_f32_16x16x16f16(__builtin_bit_cast(h4, a),
                                                     __builtin_bit_cast(h4, b), c, 0, 0, 0);
    }
#else
    (void) a;
    (void) b;
    __builtin_trap();
    return c;
#endif
}

/// The lane's operand share of a row whose k values are contiguous: `span` values from
/// `row + k0(lane)`, which is aligned to eight bytes.
__device__ __forceinline__ Core::frag load_row(const void* row, int lane) {
    const short* p = static_cast<const short*>(row) + Core::k0(lane);
    Core::frag f;
#pragma unroll
    for (int i = 0; i < Core::span; i += 4) {
        const s4 v = *reinterpret_cast<const s4*>(p + i);
        f[i] = v[0];
        f[i + 1] = v[1];
        f[i + 2] = v[2];
        f[i + 3] = v[3];
    }
    return f;
}

/// The lane's operand share of a column whose k values are `stride` elements apart.
__device__ __forceinline__ Core::frag load_column(const void* column, int stride, int lane) {
    const short* p = static_cast<const short*>(column) + (size_t) Core::k0(lane) * stride;
    Core::frag f;
#pragma unroll
    for (int i = 0; i < Core::span; ++i) f[i] = p[(size_t) i * stride];
    return f;
}

/// The accumulator `c` of a product, in the carrier `T`, as the B operand of a product whose k
/// runs over the rows of `c`: column `lane % 16` at k `k0(lane) + [0, span)`. Every lane of
/// the wave takes part.
template <typename T>
__device__ __forceinline__ Core::frag accum_as_operand(const Core::accum& c, int lane) {
    Core::frag f;
#if defined(RDNA3)
    // Lane `lane` holds rows `2 r + h`, the lane sixteen away rows `2 r + 1 - h`.
    const int h = (lane >> 4) & 1;
#pragma unroll
    for (int r = 0; r < Core::acc; ++r) {
        const short mine = Carrier<T>::bits(c[r]);
        const short other = (short) __shfl_xor((int) mine, 16, Core::wave);
        f[2 * r] = h ? other : mine;
        f[2 * r + 1] = h ? mine : other;
    }
#else
    (void) lane;
#pragma unroll
    for (int r = 0; r < Core::acc; ++r) f[r] = Carrier<T>::bits(c[r]);
#endif
    return f;
}

/// `v` reduced over the lanes that hold the same accumulator column.
__device__ __forceinline__ float column_max(float v) {
#pragma unroll
    for (int m = 16; m < Core::wave; m <<= 1) v = fmaxf(v, __shfl_xor(v, m, Core::wave));
    return v;
}
__device__ __forceinline__ float column_sum(float v) {
#pragma unroll
    for (int m = 16; m < Core::wave; m <<= 1) v += __shfl_xor(v, m, Core::wave);
    return v;
}

}  // namespace loken_amd_mma16
