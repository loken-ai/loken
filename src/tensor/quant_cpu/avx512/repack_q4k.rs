//! AVX-512 kernels over the x8-interleaved Q4_K layout of [`crate::tensor::quant_cpu::repack_q4k`].
//!
//! The stored layout is untouched: one 64-byte load is the 8 columns' 8 bytes of one k-step, so a
//! zmm holds the whole weight group. `vpdpbusd` multiplies each column's 8 weight bytes by the
//! broadcast 8 activation bytes into two i32 partials per column, summed once per sub-block after
//! the scale. The integer sums are exact and the float part is the AVX2 kernel's, statement for
//! statement, so the output has the AVX2 kernel's bits.

use core::arch::x86_64::*;

use crate::tensor::quant_cpu::repack_q4k::{sub_scales_and_mins_bytes, BlockQ4Kx8};
use crate::tensor::quant_cpu::{BlockQ8K, QK_K};

/// Each column's scale twice: the two i32 partials a column leaves in a zmm.
#[inline(always)]
unsafe fn scale_lanes(sm: &[[u8; 16]; 8], idx: usize) -> __m512i {
    let s = |c: usize| sm[c][idx] as i32;
    _mm512_setr_epi32(
        s(0),
        s(0),
        s(1),
        s(1),
        s(2),
        s(2),
        s(3),
        s(3),
        s(4),
        s(4),
        s(5),
        s(5),
        s(6),
        s(6),
        s(7),
        s(7),
    )
}

/// Eight activation bytes in every qword, so each column's 8 weight bytes meet the same 8.
#[inline(always)]
unsafe fn broadcast8(p: *const i8) -> __m512i {
    _mm512_set1_epi64((p as *const i64).read_unaligned())
}

/// One sub-block pair (`sbk`) of one super-block for `N` activation rows: decode the 64-byte weight
/// loads once per k-step, `vpdpbusd` them against each row, scale, add into `isum`.
#[inline(always)]
unsafe fn strip<const N: usize>(
    qs: *const u8,
    acts: &[&[BlockQ8K]],
    l: usize,
    sbk: usize,
    scale_lo: __m512i,
    scale_hi: __m512i,
    m4: __m512i,
    isum: &mut [__m512i],
) {
    let mut p_lo = [_mm512_setzero_si512(); N];
    let mut p_hi = [_mm512_setzero_si512(); N];
    for kk in 0..4 {
        let k = sbk * 4 + kk;
        let q = _mm512_loadu_si512(qs.add(k * 64) as *const _);
        let q_lo = _mm512_and_si512(q, m4);
        let q_hi = _mm512_and_si512(_mm512_srli_epi16::<4>(q), m4);
        let off = sbk * 64 + kk * 8;
        for j in 0..N {
            let abase = acts[j][l].qs.as_ptr().add(off);
            p_lo[j] = _mm512_dpbusd_epi32(p_lo[j], q_lo, broadcast8(abase));
            p_hi[j] = _mm512_dpbusd_epi32(p_hi[j], q_hi, broadcast8(abase.add(32)));
        }
    }
    for j in 0..N {
        isum[j] = _mm512_add_epi32(
            isum[j],
            _mm512_add_epi32(
                _mm512_mullo_epi32(p_lo[j], scale_lo),
                _mm512_mullo_epi32(p_hi[j], scale_hi),
            ),
        );
    }
}

/// Row-tiled GEMM: one 8-column weight group by `acts.len() <= 8` activation rows, processed in
/// strips of `RT` rows so the strip's partials stay in registers. Writes `out[r*8 .. r*8+8]`.
#[target_feature(enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni")]
pub(crate) unsafe fn gemm_group_avx512_rt<const RT: usize>(
    b: &[BlockQ4Kx8],
    acts: &[&[BlockQ8K]],
    nb: usize,
    out: &mut [f32],
) {
    let mt = acts.len();
    debug_assert!(mt <= 8);
    let m4 = _mm512_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); 8];
    let mut acc_min = [_mm256_setzero_ps(); 8];
    for l in 0..nb {
        let blk = &b[l];
        let mut sm = [[0u8; 16]; 8];
        for (sb, smv) in sm.iter_mut().enumerate() {
            *smv = sub_scales_and_mins_bytes(&blk.scales[sb * 12..sb * 12 + 12]);
        }
        let mut isum = [_mm512_setzero_si512(); 8];
        let qs = blk.qs.as_ptr();
        for sbk in 0..(QK_K / 64) {
            let scale_lo = scale_lanes(&sm, sbk * 2);
            let scale_hi = scale_lanes(&sm, sbk * 2 + 1);
            let mut rt0 = 0;
            while rt0 < mt {
                let rn = RT.min(mt - rt0);
                // The strip's width is a constant inside `strip`, so its partials are registers.
                macro_rules! run {
                    ($n:literal) => {
                        strip::<$n>(
                            qs,
                            &acts[rt0..rt0 + $n],
                            l,
                            sbk,
                            scale_lo,
                            scale_hi,
                            m4,
                            &mut isum[rt0..rt0 + $n],
                        )
                    };
                }
                match rn {
                    1 => run!(1),
                    2 => run!(2),
                    3 => run!(3),
                    4 => run!(4),
                    5 => run!(5),
                    6 => run!(6),
                    7 => run!(7),
                    _ => run!(8),
                }
                rt0 += rn;
            }
        }
        let f32_of = |h: half::f16| crate::tensor::quant_cpu::avx::f16c_to_f32(h);
        let dvec = _mm256_setr_ps(
            f32_of(blk.d[0]),
            f32_of(blk.d[1]),
            f32_of(blk.d[2]),
            f32_of(blk.d[3]),
            f32_of(blk.d[4]),
            f32_of(blk.d[5]),
            f32_of(blk.d[6]),
            f32_of(blk.d[7]),
        );
        let dminv = _mm256_setr_ps(
            f32_of(blk.dmin[0]),
            f32_of(blk.dmin[1]),
            f32_of(blk.dmin[2]),
            f32_of(blk.dmin[3]),
            f32_of(blk.dmin[4]),
            f32_of(blk.dmin[5]),
            f32_of(blk.dmin[6]),
            f32_of(blk.dmin[7]),
        );
        let mut minf = [_mm256_setzero_ps(); 8];
        for (sb, mv) in minf.iter_mut().enumerate() {
            let m = 8 + sb;
            *mv = _mm256_cvtepi32_ps(_mm256_setr_epi32(
                sm[0][m] as i32,
                sm[1][m] as i32,
                sm[2][m] as i32,
                sm[3][m] as i32,
                sm[4][m] as i32,
                sm[5][m] as i32,
                sm[6][m] as i32,
                sm[7][m] as i32,
            ));
        }
        for r in 0..mt {
            let act = &acts[r][l];
            // Lane 2c + 1 added into lane 2c, then the even dwords gathered: the 8 column sums in
            // column order.
            let pairs = _mm512_add_epi32(isum[r], _mm512_srli_epi64::<32>(isum[r]));
            let columns = _mm512_cvtepi64_epi32(pairs);
            let dall = _mm256_mul_ps(dvec, _mm256_set1_ps(act.d));
            acc[r] = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(columns), acc[r]);
            let mut imin = _mm256_setzero_si256();
            for sb in 0..8 {
                let bsum = (act.bsums[sb * 2] as i32 + act.bsums[sb * 2 + 1] as i32) as f32;
                imin = _mm256_castps_si256(_mm256_fmadd_ps(
                    minf[sb],
                    _mm256_set1_ps(bsum),
                    _mm256_castsi256_ps(imin),
                ));
            }
            let dminall = _mm256_mul_ps(dminv, _mm256_set1_ps(act.d));
            acc_min[r] = _mm256_fmadd_ps(dminall, _mm256_castsi256_ps(imin), acc_min[r]);
        }
    }
    for r in 0..mt {
        let res = _mm256_sub_ps(acc[r], acc_min[r]);
        _mm256_storeu_ps(out.as_mut_ptr().add(r * 8), res);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu::{repack_q4k, BlockFormat, BlockQ4K};

    fn prng() -> impl FnMut() -> f32 {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    /// Every tile height and strip width: the bits of the AVX2 kernel, and the scalar oracle to
    /// float tolerance.
    #[test]
    fn gemm_has_the_bits_of_the_avx2_kernel_and_matches_the_scalar_oracle() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let (k, n, m) = (768usize, 16usize, 8usize);
        let nb = k / QK_K;
        let mut next = prng();
        let wf: Vec<f32> = (0..n * k)
            .map(|i| next() * (1 << (i % 4)) as f32 * 0.1)
            .collect();
        let mut wq = vec![BlockQ4K::zeros(); n * nb];
        for row in 0..n {
            BlockQ4K::quantize(
                &wf[row * k..(row + 1) * k],
                &mut wq[row * nb..(row + 1) * nb],
            );
        }
        let mut aq = vec![BlockQ8K::zeros(); m * nb];
        for r in 0..m {
            let af: Vec<f32> = (0..k).map(|_| next() * 3.0).collect();
            BlockQ8K::quantize(&af, &mut aq[r * nb..(r + 1) * nb]);
        }
        let x8 = repack_q4k::repack(&wq, n, nb);
        for g in 0..n / 8 {
            let bgrp = &x8[g * nb..(g + 1) * nb];
            for mt in 1..=m {
                let refs: Vec<&[BlockQ8K]> = (0..mt).map(|t| &aq[t * nb..(t + 1) * nb]).collect();
                let mut avx2 = [0f32; 64];
                let mut scalar = vec![0f32; mt * 8];
                unsafe { repack_q4k::gemm_group_avx2_rt::<1>(bgrp, &refs, nb, &mut avx2) };
                for (t, a) in refs.iter().enumerate() {
                    repack_q4k::gemv_group_scalar(bgrp, a, nb, &mut scalar[t * 8..t * 8 + 8]);
                }
                macro_rules! strip {
                    ($rt:literal) => {{
                        let mut got = [0f32; 64];
                        unsafe { gemm_group_avx512_rt::<$rt>(bgrp, &refs, nb, &mut got) };
                        for i in 0..mt * 8 {
                            assert_eq!(
                                got[i].to_bits(),
                                avx2[i].to_bits(),
                                "RT={} mt={mt} lane {i}: {} vs the AVX2 {}",
                                $rt,
                                got[i],
                                avx2[i]
                            );
                            let rel = (got[i] - scalar[i]).abs() / scalar[i].abs().max(1e-3);
                            assert!(
                                rel < 1e-4,
                                "RT={} mt={mt} lane {i}: {} vs {}",
                                $rt,
                                got[i],
                                scalar[i]
                            );
                        }
                    }};
                }
                strip!(1);
                strip!(2);
                strip!(4);
                strip!(8);
            }
        }
    }
}
