//! AVX-512 kernel over the x8 Q5_K layout of [`crate::tensor::quant_cpu::repack_q5k`].
//!
//! One column at a time, as in the AVX2 kernel. A zmm holds two 32-weight groups (the low and the
//! high nibble of the same 32 bytes, each with its fifth bit), so `vpdpbusd` covers two groups and
//! the two 256-bit halves of the scaled result are the two terms the AVX2 kernel adds one after the
//! other. Folding them in i32 before the first float operation gives the AVX2 kernel's bits.

use core::arch::x86_64::*;

use crate::tensor::quant_cpu::repack_q4k::sub_scales_and_mins_bytes;
use crate::tensor::quant_cpu::repack_q5k::BlockQ5Kx8;
use crate::tensor::quant_cpu::{BlockQ8K, QK_K};

/// Eight lanes of an i32 vector down to one value, exactly.
#[inline(always)]
unsafe fn hsum_i32_8(v: __m256i) -> i32 {
    let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0x4E>(s));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32::<0xB1>(s));
    _mm_cvtsi128_si32(s)
}

/// Row-tiled GEMM: one 8-column weight group by `acts.len()` (at most 8) rows. Adds into
/// `out[r*8 .. r*8+8]`, which the caller zeroes.
#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
pub(crate) unsafe fn gemm_group_avx512(
    b: &[BlockQ5Kx8],
    acts: &[&[BlockQ8K]],
    nb: usize,
    out: &mut [f32],
) {
    let mt = acts.len();
    let m4 = _mm512_set1_epi8(0x0F);
    let one = _mm512_set1_epi8(1);
    // Low half keeps the low nibble, high half takes the high one.
    let nib_shift = _mm512_inserti64x4::<1>(
        _mm512_castsi256_si512(_mm256_set1_epi16(0)),
        _mm256_set1_epi16(4),
    );
    for j in 0..8 {
        let mut accm = [_mm256_setzero_ps(); 8];
        let mut summs = [0f32; 8];
        for l in 0..nb {
            let blk = &b[l];
            let qsp = blk.qs.as_ptr().add(j * (QK_K / 2));
            let qhp = blk.qh.as_ptr().add(j * (QK_K / 8));
            let sm = sub_scales_and_mins_bytes(&blk.scales[j * 12..j * 12 + 12]);
            // Pair `it` is the groups 2it (low nibbles) and 2it + 1 (high nibbles) of 32 bytes.
            let qh = _mm512_broadcast_i64x4(_mm256_loadu_si256(qhp as *const __m256i));
            let mut wdec = [_mm512_setzero_si512(); 4];
            let mut sv = [_mm512_setzero_si512(); 4];
            for it in 0..(QK_K / 64) {
                let raw =
                    _mm512_broadcast_i64x4(_mm256_loadu_si256(qsp.add(it * 32) as *const __m256i));
                let base = _mm512_and_si512(_mm512_srlv_epi16(raw, nib_shift), m4);
                let plane = _mm512_inserti64x4::<1>(
                    _mm512_castsi256_si512(_mm256_set1_epi16(2 * it as i16)),
                    _mm256_set1_epi16((2 * it + 1) as i16),
                );
                let hb =
                    _mm512_slli_epi16::<4>(_mm512_and_si512(_mm512_srlv_epi16(qh, plane), one));
                wdec[it] = _mm512_or_si512(base, hb);
                sv[it] = _mm512_inserti64x4::<1>(
                    _mm512_castsi256_si512(_mm256_set1_epi32(sm[2 * it] as i32)),
                    _mm256_set1_epi32(sm[2 * it + 1] as i32),
                );
            }
            let dj = crate::tensor::quant_cpu::avx::f16c_to_f32(blk.d[j]);
            let dminj = crate::tensor::quant_cpu::avx::f16c_to_f32(blk.dmin[j]);
            let minv = _mm256_setr_epi16(
                sm[8] as i16,
                sm[8] as i16,
                sm[9] as i16,
                sm[9] as i16,
                sm[10] as i16,
                sm[10] as i16,
                sm[11] as i16,
                sm[11] as i16,
                sm[12] as i16,
                sm[12] as i16,
                sm[13] as i16,
                sm[13] as i16,
                sm[14] as i16,
                sm[14] as i16,
                sm[15] as i16,
                sm[15] as i16,
            );
            for (r, act) in acts.iter().enumerate().take(mt) {
                let a = &act[l];
                let q8 = a.qs.as_ptr();
                let mut acc = _mm512_setzero_si512();
                for p in 0..4 {
                    let q = _mm512_loadu_si512(q8.add(64 * p) as *const _);
                    let dot = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wdec[p], q);
                    acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(dot, sv[p]));
                }
                let sumi = _mm256_add_epi32(
                    _mm512_castsi512_si256(acc),
                    _mm512_extracti64x4_epi64::<1>(acc),
                );
                let d = a.d * dj;
                accm[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accm[r]);
                let bsums = _mm256_loadu_si256(a.bsums.as_ptr() as *const __m256i);
                let mint = hsum_i32_8(_mm256_madd_epi16(bsums, minv));
                let dmin = -a.d * dminj;
                summs[r] += dmin * mint as f32;
            }
        }
        for (r, s) in summs.iter().enumerate().take(mt) {
            out[r * 8 + j] += crate::tensor::quant_cpu::avx::lanes_sum(accm[r]) + s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu::{repack_q5k, BlockFormat, BlockQ5K};

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
        let mut wq = vec![BlockQ5K::zeros(); n * nb];
        for row in 0..n {
            BlockQ5K::quantize(
                &wf[row * k..(row + 1) * k],
                &mut wq[row * nb..(row + 1) * nb],
            );
        }
        let mut aq = vec![BlockQ8K::zeros(); m * nb];
        for r in 0..m {
            let af: Vec<f32> = (0..k).map(|_| next() * 3.0).collect();
            BlockQ8K::quantize(&af, &mut aq[r * nb..(r + 1) * nb]);
        }
        let x8 = repack_q5k::repack(&wq, n, nb);
        for g in 0..n / 8 {
            let bgrp = &x8[g * nb..(g + 1) * nb];
            for mt in 1..=m {
                let refs: Vec<&[BlockQ8K]> = (0..mt).map(|t| &aq[t * nb..(t + 1) * nb]).collect();
                let (mut avx2, mut got) = ([0f32; 64], [0f32; 64]);
                unsafe {
                    repack_q5k::gemm_group_avx2(bgrp, &refs, nb, &mut avx2);
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
