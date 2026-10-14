//! The unquantised carriers (`f32`, `f16`, `bf16`) on 512-bit registers.
//!
//! The AVX2 kernels in [`super::super::cpu`] hold four accumulator chains of eight lanes and fold
//! them in a pinned order, and the order is part of the answer. Here chains 0 and 1 share one zmm
//! (one in each 256-bit half) and chains 2 and 3 another, so every lane performs the same
//! multiply, then the same add, as in the AVX2 kernel, and the fold below reproduces each of the
//! two pinned orders from the halves. The results have the bits of the AVX2 kernels.

use core::arch::x86_64::*;
use half::{bf16, f16};

/// f32 values per iteration, as in the AVX2 kernels: four chains of eight.
const STRIDE: usize = 32;

#[inline(always)]
unsafe fn across_lanes(v: __m256) -> f32 {
    let halves = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    let pairs = _mm_hadd_ps(halves, halves);
    _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs))
}

/// `[a0 | a1]` and `[a2 | a3]` folded as `(a0 + a1) + (a2 + a3)`.
#[inline(always)]
unsafe fn fold_adjacent(lo: __m512, hi: __m512) -> f32 {
    let a01 = _mm256_add_ps(_mm512_castps512_ps256(lo), _mm512_extractf32x8_ps::<1>(lo));
    let a23 = _mm256_add_ps(_mm512_castps512_ps256(hi), _mm512_extractf32x8_ps::<1>(hi));
    across_lanes(_mm256_add_ps(a01, a23))
}

/// `[a0 | a1]` and `[a2 | a3]` folded as `(a0 + a2) + (a1 + a3)`.
#[inline(always)]
unsafe fn fold_halving(lo: __m512, hi: __m512) -> f32 {
    let s = _mm512_add_ps(lo, hi);
    across_lanes(_mm256_add_ps(
        _mm512_castps512_ps256(s),
        _mm512_extractf32x8_ps::<1>(s),
    ))
}

#[inline(always)]
unsafe fn load_f32(p: *const f32) -> __m512 {
    _mm512_loadu_ps(p)
}

#[inline(always)]
unsafe fn load_f16(p: *const f16) -> __m512 {
    _mm512_cvtph_ps(_mm256_loadu_si256(p as *const __m256i))
}

#[inline(always)]
unsafe fn load_bf16(p: *const bf16) -> __m512 {
    let raw = _mm256_loadu_si256(p as *const __m256i);
    _mm512_castsi512_ps(_mm512_slli_epi32::<16>(_mm512_cvtepu16_epi32(raw)))
}

macro_rules! carrier {
    ($name:ident, $ty:ty, $load:ident, $fold:ident) => {
        /// Same contract as the AVX2 carrier: whole strides vectorised, the tail scalar, and the
        /// result written to `c`.
        #[target_feature(
            enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"
        )]
        pub(crate) unsafe fn $name(a: *const $ty, b: *const $ty, c: *mut f32, k: usize) {
            let whole = k & !(STRIDE - 1);
            let (mut lo, mut hi) = (_mm512_setzero_ps(), _mm512_setzero_ps());
            for i in (0..whole).step_by(STRIDE) {
                let (x0, y0) = ($load(a.add(i)), $load(b.add(i)));
                let (x1, y1) = ($load(a.add(i + 16)), $load(b.add(i + 16)));
                lo = _mm512_add_ps(_mm512_mul_ps(x0, y0), lo);
                hi = _mm512_add_ps(_mm512_mul_ps(x1, y1), hi);
            }
            let mut sum = $fold(lo, hi);
            for i in whole..k {
                sum += f32::from(*a.add(i)) * f32::from(*b.add(i));
            }
            *c = sum;
        }
    };
}

carrier!(vec_dot_f32, f32, load_f32, fold_adjacent);
carrier!(vec_dot_f16, f16, load_f16, fold_halving);
carrier!(vec_dot_bf16, bf16, load_bf16, fold_halving);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu::cpu;

    fn values(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 8.0
            })
            .collect()
    }

    /// Every length around the stride, so the vector body, the tail and their join are all met.
    #[test]
    fn carriers_have_the_bits_of_the_avx2_kernels() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        for k in [0usize, 1, 7, 8, 31, 32, 33, 63, 64, 65, 96, 257, 1000, 4096] {
            let (a, b) = (values(7, k), values(11, k));
            let (ah, bh): (Vec<f16>, Vec<f16>) = (
                a.iter().map(|&v| f16::from_f32(v)).collect(),
                b.iter().map(|&v| f16::from_f32(v)).collect(),
            );
            let (ab, bb): (Vec<bf16>, Vec<bf16>) = (
                a.iter().map(|&v| bf16::from_f32(v)).collect(),
                b.iter().map(|&v| bf16::from_f32(v)).collect(),
            );
            let (mut want, mut got) = (0f32, 0f32);
            unsafe {
                cpu::vectorised::vec_dot_f32(a.as_ptr(), b.as_ptr(), &mut want, k);
                vec_dot_f32(a.as_ptr(), b.as_ptr(), &mut got, k);
                assert_eq!(got.to_bits(), want.to_bits(), "f32, k={k}");
                cpu::vectorised::vec_dot_f16(ah.as_ptr(), bh.as_ptr(), &mut want, k);
                vec_dot_f16(ah.as_ptr(), bh.as_ptr(), &mut got, k);
                assert_eq!(got.to_bits(), want.to_bits(), "f16, k={k}");
                cpu::vectorised::vec_dot_bf16(ab.as_ptr(), bb.as_ptr(), &mut want, k);
                vec_dot_bf16(ab.as_ptr(), bb.as_ptr(), &mut got, k);
                assert_eq!(got.to_bits(), want.to_bits(), "bf16, k={k}");
            }
        }
    }
}
