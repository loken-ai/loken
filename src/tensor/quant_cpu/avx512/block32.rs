//! AVX-512 dot products for the 32-weight formats (q4_0, q5_0, q8_0, mxfp4, q4_1, q5_1, q8_1).
//!
//! Two blocks go through one `vpdpbusd`: the codes of block `i` in the low 256 bits, those of
//! block `i + 1` in the high ones. The integer sums are exact and every i32 lane covers the same
//! four products as in the AVX2 kernel; the float accumulator then takes the two blocks one after
//! the other with the same fused multiply-add, so the result has the bits of the AVX2 kernel.

use core::arch::x86_64::*;

use super::require_avx512;
use crate::tensor::quant_cpu::avx::{
    dot_signed, dot_unsigned_signed, f16c_to_f32, lanes_sum, mask_from_bits, split_nibbles,
};
use crate::tensor::quant_cpu::{
    e8m0_to_fp32_half, BlockMxFp4, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0,
    BlockQ8_1, KVALUES_MXFP4,
};

#[inline(always)]
unsafe fn join(lo: __m256i, hi: __m256i) -> __m512i {
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

/// The sum of 32 byte-products per 256-bit half, both operands signed.
#[inline(always)]
unsafe fn dot512_signed(w: __m512i, a: __m512i) -> __m512i {
    let neg = _mm512_movepi8_mask(w);
    let a = _mm512_mask_sub_epi8(a, neg, _mm512_setzero_si512(), a);
    _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(w), a)
}

/// The same with unsigned codes on the left.
#[inline(always)]
unsafe fn dot512_unsigned(w: __m512i, a: __m512i) -> __m512i {
    _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, a)
}

#[inline(always)]
unsafe fn codes_q4_0(x: &BlockQ4_0) -> __m256i {
    _mm256_sub_epi8(split_nibbles(x.qs.as_ptr()), _mm256_set1_epi8(8))
}

#[inline(always)]
unsafe fn codes_q5_0(x: &BlockQ5_0) -> __m256i {
    let fill = _mm256_andnot_si256(mask_from_bits(&x.qh), _mm256_set1_epi8(0xF0u8 as i8));
    _mm256_or_si256(split_nibbles(x.qs.as_ptr()), fill)
}

#[inline(always)]
unsafe fn codes_q8_0(x: &BlockQ8_0) -> __m256i {
    _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i)
}

#[inline(always)]
unsafe fn codes_mxfp4(x: &BlockMxFp4) -> __m256i {
    let table =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i));
    _mm256_shuffle_epi8(table, split_nibbles(x.qs.as_ptr()))
}

#[inline(always)]
unsafe fn codes_q4_1(x: &BlockQ4_1) -> __m256i {
    split_nibbles(x.qs.as_ptr())
}

#[inline(always)]
unsafe fn codes_q5_1(x: &BlockQ5_1) -> __m256i {
    let high = _mm256_and_si256(mask_from_bits(&x.qh), _mm256_set1_epi8(16));
    _mm256_add_epi8(split_nibbles(x.qs.as_ptr()), high)
}

#[inline(always)]
unsafe fn codes_q8_1(x: &BlockQ8_1) -> __m256i {
    _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i)
}

macro_rules! dot32 {
    (
        $name:ident, $kernel:ident, $x:ty, $y:ty, $codes:ident, $dot512:ident, $dot256:ident,
        scale: |$sx:ident, $sy:ident| $scale:expr
        $(, min: |$mx:ident, $my:ident| $min:expr)?
    ) => {
        pub(crate) fn $name(xs: &[$x], ys: &[$y]) -> f32 {
            require_avx512();
            // SAFETY: `require_avx512` just confirmed the CPU has every feature the kernel enables.
            unsafe { $kernel(xs, ys) }
        }

        #[target_feature(
            enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"
        )]
        unsafe fn $kernel(xs: &[$x], ys: &[$y]) -> f32 {
            let n = xs.len().min(ys.len());
            let mut acc = _mm256_setzero_ps();
            $(let mut from_min = 0f32; let _ = stringify!($mx);)?
            let mut i = 0;
            while i + 2 <= n {
                let (xa, xb, ya, yb) = (&xs[i], &xs[i + 1], &ys[i], &ys[i + 1]);
                let acts = join(
                    _mm256_loadu_si256(ya.qs.as_ptr() as *const __m256i),
                    _mm256_loadu_si256(yb.qs.as_ptr() as *const __m256i),
                );
                let dots = $dot512(join($codes(xa), $codes(xb)), acts);
                let (sa, sb) = {
                    let ($sx, $sy) = (xa, ya);
                    let a: f32 = $scale;
                    let ($sx, $sy) = (xb, yb);
                    let b: f32 = $scale;
                    (a, b)
                };
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(sa),
                    _mm256_cvtepi32_ps(_mm512_castsi512_si256(dots)),
                    acc,
                );
                acc = _mm256_fmadd_ps(
                    _mm256_set1_ps(sb),
                    _mm256_cvtepi32_ps(_mm512_extracti64x4_epi64::<1>(dots)),
                    acc,
                );
                $(
                    {
                        let ($mx, $my) = (xa, ya);
                        from_min += $min;
                        let ($mx, $my) = (xb, yb);
                        from_min += $min;
                    }
                )?
                i += 2;
            }
            if i < n {
                let (x, y) = (&xs[i], &ys[i]);
                let acts = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
                let scale = {
                    let ($sx, $sy) = (x, y);
                    let s: f32 = $scale;
                    s
                };
                acc = _mm256_fmadd_ps(_mm256_set1_ps(scale), $dot256($codes(x), acts), acc);
                $(
                    {
                        let ($mx, $my) = (x, y);
                        from_min += $min;
                    }
                )?
            }
            lanes_sum(acc) $(+ { let _ = stringify!($mx); from_min })?
        }
    };
}

dot32!(
    vec_dot_q4_0_q8_0, q4_0_kernel, BlockQ4_0, BlockQ8_0, codes_q4_0, dot512_signed, dot_signed,
    scale: |x, y| f16c_to_f32(x.d) * f16c_to_f32(y.d)
);
dot32!(
    vec_dot_q5_0_q8_0, q5_0_kernel, BlockQ5_0, BlockQ8_0, codes_q5_0, dot512_signed, dot_signed,
    scale: |x, y| f16c_to_f32(x.d) * f16c_to_f32(y.d)
);
dot32!(
    vec_dot_q8_0_q8_0, q8_0_kernel, BlockQ8_0, BlockQ8_0, codes_q8_0, dot512_signed, dot_signed,
    scale: |x, y| f16c_to_f32(x.d) * f16c_to_f32(y.d)
);
dot32!(
    vec_dot_mxfp4_q8_0, mxfp4_kernel, BlockMxFp4, BlockQ8_0, codes_mxfp4, dot512_signed, dot_signed,
    scale: |x, y| e8m0_to_fp32_half(x.e) * f16c_to_f32(y.d)
);
dot32!(
    vec_dot_q8_1_q8_1, q8_1_kernel, BlockQ8_1, BlockQ8_1, codes_q8_1, dot512_signed, dot_signed,
    scale: |x, y| f16c_to_f32(x.d) * f16c_to_f32(y.d)
);
dot32!(
    vec_dot_q4_1_q8_1, q4_1_kernel, BlockQ4_1, BlockQ8_1, codes_q4_1, dot512_unsigned,
    dot_unsigned_signed,
    scale: |x, y| f16c_to_f32(x.d) * f16c_to_f32(y.d),
    min: |x, y| f16c_to_f32(x.m) * f16c_to_f32(y.s)
);
dot32!(
    vec_dot_q5_1_q8_1, q5_1_kernel, BlockQ5_1, BlockQ8_1, codes_q5_1, dot512_unsigned,
    dot_unsigned_signed,
    scale: |x, y| f16c_to_f32(x.d) * f16c_to_f32(y.d),
    min: |x, y| f16c_to_f32(x.m) * f16c_to_f32(y.s)
);
