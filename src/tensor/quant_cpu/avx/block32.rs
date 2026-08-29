//! AVX2 dot products for the formats whose block is 32 values.
//!
//! One vector register holds a whole block's codes, so there is no inner loop:
//! unpack, multiply against the activation, accumulate. What separates the six
//! is only how a code becomes a byte and where the zero point goes.
//!
//! Two shapes, by what the block reconstructs to:
//!
//! - `y = q.d` with `q` signed (q4_0, q5_0, q8_0, mxfp4). The bias is folded
//!   into the byte before the product - `nibble - 8` for q4_0, a table lookup
//!   for mxfp4 - and the result feeds [`dot_signed`].
//! - `y = q.d + m` with `q` unsigned (q4_1, q5_1). Over a block against a
//!   q8_1 activation `a.da`,
//!
//!       Σ (q.d + m)(a.da) = d.da . Σ(q.a) + m . (da.Σa)
//!
//!   and `da.Σa` is exactly what q8_1 stores in `s` - the reason that format
//!   carries a second scale. The minimum costs one scalar multiply per block
//!   and no vector work at all, so these use [`dot_unsigned_signed`] directly.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use super::{dot_signed, dot_unsigned_signed, lanes_sum, mask_from_bits, split_nibbles};
use crate::tensor::quant_cpu::{
    e8m0_to_fp32_half, BlockMxFp4, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
    BlockQ8_1, KVALUES_MXFP4,
};
use half::f16;

// -- y = q.d, signed codes --

/// Four-bit codes biased by 8: subtracting it before the product is one
/// instruction and leaves a signed byte the dot takes directly.
#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let bias = _mm256_set1_epi8(8);
        for (x, y) in xs.iter().zip(ys) {
            let scale = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            let codes = _mm256_sub_epi8(split_nibbles(x.qs.as_ptr()), bias);
            let acts = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            acc = _mm256_fmadd_ps(scale, dot_signed(codes, acts), acc);
        }
        lanes_sum(acc)
    }
}

/// Five-bit codes, the fifth bit held apart in `qh`.
///
/// The scalar path extracts that bit per element and subtracts 16. As a signed
/// byte the same value `v = (nib | bit<<4) - 16` is just `nib | 0xF0` wherever
/// the bit is CLEAR - so the high bits become a mask, one OR, and the codes are
/// signed already. Pair magnitudes stay under `2.16.127 < 2^15`: no saturation.
#[inline(always)]
pub(crate) fn vec_dot_q5_0_q8_0(xs: &[BlockQ5_0], ys: &[BlockQ8_0]) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let sign_fill = _mm256_set1_epi8(0xF0u8 as i8);
        for (x, y) in xs.iter().zip(ys) {
            let scale = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            let fill = _mm256_andnot_si256(mask_from_bits(&x.qh), sign_fill);
            let codes = _mm256_or_si256(split_nibbles(x.qs.as_ptr()), fill);
            let acts = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            acc = _mm256_fmadd_ps(scale, dot_signed(codes, acts), acc);
        }
        lanes_sum(acc)
    }
}

/// Eight-bit codes: the block is already the vector.
#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0(xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys) {
            let scale = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            let codes = _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i);
            let acts = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            acc = _mm256_fmadd_ps(scale, dot_signed(codes, acts), acc);
        }
        lanes_sum(acc)
    }
}

/// Four-bit codes that index a table of sixteen signed values rather than
/// meaning a number, and a block scale that is a bare exponent.
///
/// The table fits a 128-bit lane, so broadcasting it to both lanes turns the
/// lookup into one in-lane byte shuffle. Largest product is `12.127`, so the
/// pair sums stay under `2^15`.
#[inline(always)]
pub(crate) fn vec_dot_mxfp4_q8_0(xs: &[BlockMxFp4], ys: &[BlockQ8_0]) -> f32 {
    unsafe {
        let table =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i));
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys) {
            let scale = _mm256_set1_ps(e8m0_to_fp32_half(x.e) * f16::to_f32(y.d));
            let codes = _mm256_shuffle_epi8(table, split_nibbles(x.qs.as_ptr()));
            let acts = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            acc = _mm256_fmadd_ps(scale, dot_signed(codes, acts), acc);
        }
        lanes_sum(acc)
    }
}

// -- y = q.d + m, unsigned codes --

/// Four-bit unsigned codes and a per-block minimum.
#[cfg(target_feature = "avx2")]
pub(crate) fn vec_dot_q4_1_q8_1(xs: &[BlockQ4_1], ys: &[BlockQ8_1]) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        // Exact in f32 and needing no vector work; a separate scalar also keeps
        // it out of the horizontal sum's rounding.
        let mut from_min = 0f32;
        for (x, y) in xs.iter().zip(ys) {
            let scale = _mm256_set1_ps(x.d.to_f32() * y.d.to_f32());
            let codes = split_nibbles(x.qs.as_ptr());
            let acts = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            acc = _mm256_fmadd_ps(scale, dot_unsigned_signed(codes, acts), acc);
            from_min += x.m.to_f32() * y.s.to_f32();
        }
        lanes_sum(acc) + from_min
    }
}

/// Five-bit unsigned codes and a per-block minimum. Unlike q5_0 there is no
/// bias to fold, so the fifth bit is worth exactly sixteen: mask it and add.
#[cfg(target_feature = "avx2")]
pub(crate) fn vec_dot_q5_1_q8_1(xs: &[BlockQ5_1], ys: &[BlockQ8_1]) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut from_min = 0f32;
        let sixteen = _mm256_set1_epi8(16);
        for (x, y) in xs.iter().zip(ys) {
            let scale = _mm256_set1_ps(x.d.to_f32() * y.d.to_f32());
            let high = _mm256_and_si256(mask_from_bits(&x.qh), sixteen);
            let codes = _mm256_add_epi8(split_nibbles(x.qs.as_ptr()), high);
            let acts = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            acc = _mm256_fmadd_ps(scale, dot_unsigned_signed(codes, acts), acc);
            from_min += x.m.to_f32() * y.s.to_f32();
        }
        lanes_sum(acc) + from_min
    }
}

/// Eight-bit codes on both sides, and the second scale `s` unused: it says what
/// a weight's minimum would be worth, and a q8_1 weight has none. Identical in
/// shape to [`vec_dot_q8_0_q8_0`] over a block that carries one extra f16.
#[inline(always)]
pub(crate) fn vec_dot_q8_1_q8_1(xs: &[BlockQ8_1], ys: &[BlockQ8_1]) -> f32 {
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys) {
            let scale = _mm256_set1_ps(x.d.to_f32() * y.d.to_f32());
            let codes = _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i);
            let acts = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            acc = _mm256_fmadd_ps(scale, dot_signed(codes, acts), acc);
        }
        lanes_sum(acc)
    }
}
