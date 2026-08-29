#pragma once

#include "mmq_common.cuh"

#include <cstdint>

// The run length - how many consecutive ints one call walks - is stated per format in
// `gguf.cuh`, included above: `VDR_<format>_Q8_1_MMQ` for the tiled path, `_MMVQ` for the
// row-at-a-time one. Only the one value on which the two paths disagree is restated here, at
// the call it governs.

template <typename T, int vdr> static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const T & d8_0, const T & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dot4_i8(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * ((T) sumi);
}

// q2_K is the format whose two paths want different run lengths. The call below covers `QR2_K`
// times this many ints and reads the activation's stated sums two blocks at a time, so its run
// spans two q8_1 blocks; the row-at-a-time path reaches the same values one block per call and
// so asks for half. The shared value is undefined rather than shadowed, leaving one spelling in
// scope from here on.
#undef VDR_Q2_K_Q8_1_MMQ
#define VDR_Q2_K_Q8_1_MMQ 4

/// A tile's worth of a two-bit superblock, against a q8_1 activation.
///
/// A weight of this format is `scale * code - minimum`, where the scale and the minimum belong
/// to its sub-block of sixteen and both arrive already multiplied by the superblock's own two
/// factors. So a sub-block contributes
///
///     scale * sum(code * activation) - minimum * sum(activation)
///
/// and the second sum is one the activation may already carry: a q8_1 block states its own sum
/// alongside its scale. `ns8` says how many of them do - the last quarter of a row does not  - 
/// and where it does not, the sum is taken the only other way there is, a dot product against
/// a vector of ones.
///
/// The two sums are kept apart to the end because they are scaled differently: everything
/// weighted by the activation's own scale goes in one, and what the activation's stated sums
/// already account for goes in the other.
template <int ns8>
static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmq(
    const int * __restrict__ v, const int * __restrict__ u, const half2 * dm2, const float & d8,
    const half2 * s8) {

    // A sub-block is sixteen values, which is half of what a q8_1 block covers.
    constexpr int ints_per_subblock = QI8_1 / 2;

    float from_stated_sums = 0.0f;
    float to_scale_by_d8   = 0.0f;

#pragma unroll
    for (int i0 = 0; i0 < QR2_K*VDR_Q2_K_Q8_1_MMQ; i0 += QI8_1) {
        // The two sub-blocks this activation block covers.
        float2 scale_minimum[2];
        int weighted[2] = {0, 0};

#pragma unroll
        for (int half = 0; half < 2; ++half) {
            scale_minimum[half] = __half22float2(dm2[i0/ints_per_subblock + half]);
            const int first = i0 + half*ints_per_subblock;
#pragma unroll
            for (int i = first; i < first + ints_per_subblock; ++i) {
                weighted[half] = dot4_i8(v[i], u[i], weighted[half]);
            }
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            to_scale_by_d8 += scale_minimum[half].x * weighted[half];
        }

        if (i0/QI8_1 < ns8) {
            const float2 stated = __half22float2(s8[i0/QI8_1]);
            from_stated_sums -= scale_minimum[0].y * stated.x;
            from_stated_sums -= scale_minimum[1].y * stated.y;
        } else {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int first = i0 + half*ints_per_subblock;
                int ones = 0;
#pragma unroll
                for (int i = first; i < first + ints_per_subblock; ++i) {
                    ones = dot4_i8(0x01010101, u[i], ones);
                }
                to_scale_by_d8 -= scale_minimum[half].y * ones;
            }
        }
    }

    return from_stated_sums + d8*to_scale_by_d8;
}
