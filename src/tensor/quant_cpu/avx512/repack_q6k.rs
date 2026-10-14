//! AVX-512 kernel over the x8 Q6_K layout of [`crate::tensor::quant_cpu::repack_q6k`].
//!
//! One column at a time, as in the AVX2 kernel: its 256 six-bit weights are decoded once into four
//! zmm (two 32-weight chunks each) and meet every activation row of the tile. The products go
//! through `vpdpbusd`, the group scales multiply the i32 partials, and the integer total is exact,
//! so the output has the bits of the AVX2 kernel.

use core::arch::x86_64::*;

use crate::tensor::quant_cpu::repack_q6k::BlockQ6Kx8;
use crate::tensor::quant_cpu::{BlockQ8K, QK_K};

/// Eight lanes of an i32 vector down to one value, exactly.
#[inline(always)]
unsafe fn hsum_i32_8(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0x4E>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0xB1>(s));
    _mm_cvtsi128_si32(s)
}

/// Row-tiled GEMM: one 8-column weight group by `acts.len()` rows. Adds into
/// `out[r*8 .. r*8+8]`, which the caller zeroes.
#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
pub(crate) unsafe fn gemm_group_avx512(
    b: &[BlockQ6Kx8],
    acts: &[&[BlockQ8K]],
    nb: usize,
    out: &mut [f32],
) {
    let mt = acts.len();
    let m4 = _mm512_set1_epi8(0xF);
    let m2 = _mm512_set1_epi8(3);
    // The 2-bit field of each 32-weight chunk of a pair: chunks (0,1) read fields at bit 0 and 2
    // of the same 32 high-plane bytes, chunks (2,3) at bit 4 and 6.
    let shift_01 = _mm512_inserti64x4::<1>(
        _mm512_castsi256_si512(_mm256_set1_epi16(0)),
        _mm256_set1_epi16(2),
    );
    let shift_23 = _mm512_inserti64x4::<1>(
        _mm512_castsi256_si512(_mm256_set1_epi16(4)),
        _mm256_set1_epi16(6),
    );
    // Byte i of a four-scale word becomes four lanes: one scale over each 4-lane group.
    let spread = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    for l in 0..nb {
        let blk = &b[l];
        for j in 0..8 {
            let qlp = blk.ql.as_ptr().add(j * (QK_K / 2));
            let qhp = blk.qh.as_ptr().add(j * (QK_K / 4));
            let scp = blk.scales.as_ptr().add(j * (QK_K / 16));

            // 256 unsigned weights (0..63) as four zmm: pair p holds chunks 2p and 2p + 1.
            let mut wdec = [_mm512_setzero_si512(); 4];
            for step in 0..2 {
                let ql = _mm512_loadu_si512(qlp.add(64 * step) as *const _);
                let qh32 = _mm256_loadu_si256(qhp.add(32 * step) as *const __m256i);
                let qh = _mm512_broadcast_i64x4(qh32);
                let h_01 =
                    _mm512_slli_epi16::<4>(_mm512_and_si512(_mm512_srlv_epi16(qh, shift_01), m2));
                let h_23 =
                    _mm512_slli_epi16::<4>(_mm512_and_si512(_mm512_srlv_epi16(qh, shift_23), m2));
                wdec[2 * step] = _mm512_or_si512(_mm512_and_si512(ql, m4), h_01);
                wdec[2 * step + 1] =
                    _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16::<4>(ql), m4), h_23);
            }
            // Pair p spans groups 4p..4p + 3 of 16 weights: one scale per four i32 lanes.
            let mut sv = [_mm512_setzero_si512(); 4];
            for (p, svp) in sv.iter_mut().enumerate() {
                let word = (scp.add(4 * p) as *const i32).read_unaligned();
                *svp = _mm512_cvtepi8_epi32(_mm_shuffle_epi8(_mm_cvtsi32_si128(word), spread));
            }
            let scales16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(scp as *const __m128i));
            let dj = crate::tensor::quant_cpu::avx::f16c_to_f32(blk.d[j]);
            for (r, act) in acts.iter().enumerate().take(mt) {
                let a = &act[l];
                let q8 = a.qs.as_ptr();
                let mut sumi = _mm512_setzero_si512();
                for p in 0..4 {
                    let q = _mm512_loadu_si512(q8.add(64 * p) as *const _);
                    let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wdec[p], q);
                    sumi = _mm512_add_epi32(sumi, _mm512_mullo_epi32(dot, sv[p]));
                }
                // The fixed zero point of 32, folded once from the activation group sums.
                let bsums = _mm256_loadu_si256(a.bsums.as_ptr() as *const __m256i);
                let corr = _mm256_slli_epi32::<5>(_mm256_madd_epi16(bsums, scales16));
                let s = _mm512_reduce_add_epi32(sumi) - hsum_i32_8(corr);
                out[r * 8 + j] += dj * a.d * s as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu::{repack_q6k, BlockFormat, BlockQ6K};

    #[test]
    fn gemm_has_the_bits_of_the_avx2_kernel() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let (k, n, m) = (768usize, 16usize, 8usize);
        let nb = k / QK_K;
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let wf: Vec<f32> = (0..n * k)
            .map(|i| next() * (1 << (i % 4)) as f32 * 0.2)
            .collect();
        let mut wq = vec![BlockQ6K::zeros(); n * nb];
        for row in 0..n {
            BlockQ6K::quantize(
                &wf[row * k..(row + 1) * k],
                &mut wq[row * nb..(row + 1) * nb],
            );
        }
        let mut aq = vec![BlockQ8K::zeros(); m * nb];
        for r in 0..m {
            let af: Vec<f32> = (0..k).map(|_| next() * 3.0).collect();
            BlockQ8K::quantize(&af, &mut aq[r * nb..(r + 1) * nb]);
        }
        let x8 = repack_q6k::repack(&wq, n, nb);
        for g in 0..n / 8 {
            let bgrp = &x8[g * nb..(g + 1) * nb];
            for mt in 1..=m {
                let refs: Vec<&[BlockQ8K]> = (0..mt).map(|t| &aq[t * nb..(t + 1) * nb]).collect();
                let (mut avx2, mut got) = ([0f32; 64], [0f32; 64]);
                unsafe {
                    repack_q6k::gemm_group_avx2(bgrp, &refs, nb, &mut avx2);
                    gemm_group_avx512(bgrp, &refs, nb, &mut got);
                }
                for i in 0..mt * 8 {
                    assert_eq!(
                        got[i].to_bits(),
                        avx2[i].to_bits(),
                        "mt={mt} lane {i}: {} vs the AVX2 {}",
                        got[i],
                        avx2[i]
                    );
                }
            }
        }
    }
}
