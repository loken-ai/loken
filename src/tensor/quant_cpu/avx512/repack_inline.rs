//! AVX-512 GEMM over the block-major `[n/8][nb][8]` layouts of Q4_0, Q5_0 and Q8_0.
//!
//! As in the AVX2 kernels a column's block is decoded once and meets every row of the tile. Here
//! two rows share a zmm (the block in both halves, one row in each), so a single `vpdpbusd`
//! serves two rows, and the float tail runs on a `[row r | row r + 1]` accumulator. Every lane
//! sees the same operations in the same order as in the AVX2 kernel, so the results are
//! bit-identical.

use core::arch::x86_64::*;

use crate::tensor::quant_cpu::avx::{mask_from_bits, split_nibbles};
use crate::tensor::quant_cpu::{BlockQ4_0, BlockQ5_0, BlockQ8_0};

/// `dot(w, a)` per i32 lane for signed `w` and `a`, both zmm (`|w|` against `a` signed like `w`).
#[inline(always)]
unsafe fn dot_signed(w: __m512i, a: __m512i) -> __m512i {
    let neg = _mm512_movepi8_mask(w);
    let a = _mm512_mask_sub_epi8(a, neg, _mm512_setzero_si512(), a);
    _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(w), a)
}

/// Both rows of a pair in one zmm; an odd last row is paired with itself and its copy ignored.
#[inline(always)]
unsafe fn rows(a0: &BlockQ8_0, a1: &BlockQ8_0) -> __m512i {
    _mm512_inserti64x4::<1>(
        _mm512_castsi256_si512(_mm256_loadu_si256(a0.qs.as_ptr() as *const __m256i)),
        _mm256_loadu_si256(a1.qs.as_ptr() as *const __m256i),
    )
}

macro_rules! gemm {
    ($name:ident, $blk:ty, $decode:expr) => {
        /// Row-tiled GEMM: one 8-column weight group by `acts.len()` (at most 8) rows. Writes
        /// `out[r*8 .. r*8+8]`.
        #[target_feature(
            enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"
        )]
        pub(crate) unsafe fn $name(b: &[$blk], acts: &[&[BlockQ8_0]], nb: usize, out: &mut [f32]) {
            let mt = acts.len();
            let pairs = mt.div_ceil(2);
            let decode = $decode;
            for j in 0..8 {
                let mut acc = [_mm512_setzero_ps(); 4];
                for l in 0..nb {
                    let blk = &b[l * 8 + j];
                    let w = _mm512_broadcast_i64x4(decode(blk));
                    let dj = crate::tensor::quant_cpu::avx::f16c_to_f32(blk.d);
                    for p in 0..pairs {
                        let (a0, a1) = (&acts[2 * p][l], &acts[(2 * p + 1).min(mt - 1)][l]);
                        let d = _mm512_mask_blend_ps(
                            0xFF00,
                            _mm512_set1_ps(dj * crate::tensor::quant_cpu::avx::f16c_to_f32(a0.d)),
                            _mm512_set1_ps(dj * crate::tensor::quant_cpu::avx::f16c_to_f32(a1.d)),
                        );
                        let dot = _mm512_cvtepi32_ps(dot_signed(w, rows(a0, a1)));
                        acc[p] = _mm512_fmadd_ps(d, dot, acc[p]);
                    }
                }
                for p in 0..pairs {
                    let lo = _mm512_castps512_ps256(acc[p]);
                    out[2 * p * 8 + j] = crate::tensor::quant_cpu::avx::lanes_sum(lo);
                    if 2 * p + 1 < mt {
                        let hi = _mm512_extractf32x8_ps::<1>(acc[p]);
                        out[(2 * p + 1) * 8 + j] = crate::tensor::quant_cpu::avx::lanes_sum(hi);
                    }
                }
            }
        }
    };
}

gemm!(gemm_group_q4_0, BlockQ4_0, |blk: &BlockQ4_0| {
    _mm256_sub_epi8(split_nibbles(blk.qs.as_ptr()), _mm256_set1_epi8(8))
});

gemm!(gemm_group_q5_0, BlockQ5_0, |blk: &BlockQ5_0| {
    // (nibble | bit << 4) - 16, as a signed byte, is the nibble OR 0xF0 where the high bit is clear.
    _mm256_or_si256(
        split_nibbles(blk.qs.as_ptr()),
        _mm256_andnot_si256(mask_from_bits(&blk.qh), _mm256_set1_epi8(0xF0u8 as i8)),
    )
});

gemm!(gemm_group_q8_0, BlockQ8_0, |blk: &BlockQ8_0| {
    _mm256_loadu_si256(blk.qs.as_ptr() as *const __m256i)
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu::{
        repack_q4_0, repack_q5_0, repack_q8_0, BlockFormat, BlockQ5_0, BlockQ8_0,
    };

    type Gemm<B> = unsafe fn(&[B], &[&[BlockQ8_0]], usize, &mut [f32]);

    fn check<B: BlockFormat + Clone>(
        name: &str,
        repack: fn(&[B], usize, usize) -> Vec<B>,
        avx2: Gemm<B>,
        avx512: Gemm<B>,
    ) {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let (k, n, m) = (320usize, 16usize, 8usize);
        let nb = k / 32;
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let wf: Vec<f32> = (0..n * k).map(|i| next() * (1 << (i % 3)) as f32).collect();
        let mut wq = vec![B::zeros(); n * nb];
        for row in 0..n {
            B::quantize(
                &wf[row * k..(row + 1) * k],
                &mut wq[row * nb..(row + 1) * nb],
            );
        }
        let mut aq = vec![BlockQ8_0::zeros(); m * nb];
        for r in 0..m {
            let af: Vec<f32> = (0..k).map(|_| next() * 3.0).collect();
            BlockQ8_0::quantize(&af, &mut aq[r * nb..(r + 1) * nb]);
        }
        let x8 = repack(&wq, n, nb);
        for g in 0..n / 8 {
            let bgrp = &x8[g * nb * 8..(g + 1) * nb * 8];
            for mt in 1..=m {
                let refs: Vec<&[BlockQ8_0]> = (0..mt).map(|t| &aq[t * nb..(t + 1) * nb]).collect();
                let (mut want, mut got) = ([0f32; 64], [0f32; 64]);
                unsafe {
                    avx2(bgrp, &refs, nb, &mut want);
                    avx512(bgrp, &refs, nb, &mut got);
                }
                for i in 0..mt * 8 {
                    assert_eq!(
                        got[i].to_bits(),
                        want[i].to_bits(),
                        "{name} mt={mt} lane {i}: {} vs the AVX2 {}",
                        got[i],
                        want[i]
                    );
                }
            }
        }
    }

    #[test]
    fn q4_0_has_the_bits_of_the_avx2_kernel() {
        check::<BlockQ4_0>(
            "q4_0",
            repack_q4_0::repack,
            repack_q4_0::gemm_group_avx2,
            gemm_group_q4_0,
        );
    }

    #[test]
    fn q5_0_has_the_bits_of_the_avx2_kernel() {
        check::<BlockQ5_0>(
            "q5_0",
            repack_q5_0::repack,
            repack_q5_0::gemm_group_avx2,
            gemm_group_q5_0,
        );
    }

    #[test]
    fn q8_0_has_the_bits_of_the_avx2_kernel() {
        check::<BlockQ8_0>(
            "q8_0",
            repack_q8_0::repack,
            repack_q8_0::gemm_group_avx2,
            gemm_group_q8_0,
        );
    }
}
