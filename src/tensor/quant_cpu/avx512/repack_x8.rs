//! AVX-512 GEMM over the column-interleaved x8 layouts of the 32-weight formats (Q8_0 and MXFP4).
//!
//! Both store a block as four element groups of 64 bytes: eight qwords, one per column, holding
//! that column's 8 consecutive weights. A zmm load is therefore a whole group, and one activation
//! row meets all eight columns with a single broadcast qword and `vpdpbusd`. The AVX2 kernels reach
//! the same integers through llama.cpp's shuffle sequence; here the sums are exact in i32 (the
//! activation is biased to unsigned and the bias removed through the column sums), so only the
//! float tail has to agree, and it does the same operations in the same order.

use core::arch::x86_64::*;

use crate::tensor::quant_cpu::repack_mxfp4_x8::{BlockMxFp4x8, BlockQ8_0x4};
use crate::tensor::quant_cpu::repack_q8_0_x8::BlockQ8_0x8;
use crate::tensor::quant_cpu::{e8m0_to_fp32_half, KVALUES_MXFP4};

/// The pairs of adjacent i32 lanes (a column's two halves) summed, two accumulators at a time:
/// `[cols of a | cols of b]`.
#[inline(always)]
unsafe fn pair_sum(a: __m512i, b: __m512i) -> __m512i {
    let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    let odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    _mm512_add_epi32(
        _mm512_permutex2var_epi32(a, even, b),
        _mm512_permutex2var_epi32(a, odd, b),
    )
}

/// One weight block against `G` activation groups of four rows. `w[eg]` is element group `eg`
/// of the block, `cs` the eight column scales. `acc[2g + q]` holds rows `2q` and `2q + 1` of
/// group `g` as `[row 2q cols | row 2q + 1 cols]`.
#[inline(always)]
unsafe fn block<const G: usize>(
    w: &[__m512i; 4],
    cs: __m256,
    a: [&BlockQ8_0x4; G],
    acc: &mut [__m512],
) {
    let bias = _mm512_set1_epi8(i8::MIN);
    let ones = _mm512_set1_epi8(1);
    // Activations enter as a + 128 (unsigned); 128 * (the column's weight sum) comes back out.
    let mut wsum = _mm512_setzero_si512();
    for g in w {
        wsum = _mm512_dpbusd_epi32(wsum, ones, *g);
    }
    let corr = _mm512_slli_epi32::<7>(pair_sum(wsum, wsum));
    let cs2 = _mm512_broadcast_f32x8(cs);
    for (g, blk) in a.iter().enumerate() {
        let aq = blk.qs.as_ptr();
        let d = blk.d.as_ptr();
        for q in 0..2 {
            let mut rows = [_mm512_setzero_si512(); 2];
            for (r, row) in rows.iter_mut().enumerate() {
                for (eg, wg) in w.iter().enumerate() {
                    let bytes = (aq.add(eg * 32 + (2 * q + r) * 8) as *const i64).read_unaligned();
                    let x = _mm512_xor_si512(_mm512_set1_epi64(bytes), bias);
                    *row = _mm512_dpbusd_epi32(*row, x, *wg);
                }
            }
            let sums = _mm512_sub_epi32(pair_sum(rows[0], rows[1]), corr);
            let rs = _mm512_mask_blend_ps(
                0xFF00,
                _mm512_set1_ps((*d.add(2 * q)).to_f32()),
                _mm512_set1_ps((*d.add(2 * q + 1)).to_f32()),
            );
            let i = 2 * g + q;
            acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(sums), _mm512_mul_ps(cs2, rs), acc[i]);
        }
    }
}

#[inline(always)]
unsafe fn q8_weights(bb: &BlockQ8_0x8) -> ([__m512i; 4], __m256) {
    let bq = bb.qs.as_ptr();
    let mut w = [_mm512_setzero_si512(); 4];
    for (eg, wg) in w.iter_mut().enumerate() {
        *wg = _mm512_loadu_si512(bq.add(eg * 64) as *const _);
    }
    (
        w,
        _mm256_cvtph_ps(_mm_loadu_si128(bb.d.as_ptr() as *const __m128i)),
    )
}

#[inline(always)]
unsafe fn mxfp4_weights(bb: &BlockMxFp4x8, lut: __m512i) -> ([__m512i; 4], __m256) {
    let m4 = _mm512_set1_epi8(0x0F);
    let bq = bb.qs.as_ptr();
    let mut w = [_mm512_setzero_si512(); 4];
    // Half `h` of every column's 16 bytes: low nibbles are element group `h`, high ones `h + 2`.
    for h in 0..2 {
        let v = _mm512_loadu_si512(bq.add(h * 64) as *const _);
        w[h] = _mm512_shuffle_epi8(lut, _mm512_and_si512(v, m4));
        w[h + 2] = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16::<4>(v), m4));
    }
    let e = &bb.e;
    let cs = _mm256_set_ps(
        e8m0_to_fp32_half(e[7]),
        e8m0_to_fp32_half(e[6]),
        e8m0_to_fp32_half(e[5]),
        e8m0_to_fp32_half(e[4]),
        e8m0_to_fp32_half(e[3]),
        e8m0_to_fp32_half(e[2]),
        e8m0_to_fp32_half(e[1]),
        e8m0_to_fp32_half(e[0]),
    );
    (w, cs)
}

macro_rules! kernels {
    ($one:ident, $four:ident, $blk:ty, $weights:expr, $prep:expr) => {
        /// One tile of four activation rows against eight columns. `out[m*8 + j]`.
        #[target_feature(
            enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"
        )]
        pub(crate) unsafe fn $one(a: &[BlockQ8_0x4], b: &[$blk], nb: usize, out: &mut [f32; 32]) {
            let prep = $prep;
            let mut acc = [_mm512_setzero_ps(); 2];
            for l in 0..nb {
                let (w, cs) = $weights(&b[l], prep);
                block::<1>(&w, cs, [&a[l]], &mut acc);
            }
            _mm512_storeu_ps(out.as_mut_ptr(), acc[0]);
            _mm512_storeu_ps(out.as_mut_ptr().add(16), acc[1]);
        }

        /// Sixteen activation rows (four groups): each weight block decodes once.
        /// `out[rp*32 + m*8 + j]`.
        #[target_feature(
            enable = "avx2,fma,f16c,bmi2,avx512f,avx512bw,avx512vl,avx512dq,avx512vnni"
        )]
        pub(crate) unsafe fn $four(
            a0: &[BlockQ8_0x4],
            a1: &[BlockQ8_0x4],
            a2: &[BlockQ8_0x4],
            a3: &[BlockQ8_0x4],
            b: &[$blk],
            nb: usize,
            out: &mut [f32; 128],
        ) {
            let prep = $prep;
            let mut acc = [_mm512_setzero_ps(); 8];
            for l in 0..nb {
                let (w, cs) = $weights(&b[l], prep);
                block::<4>(&w, cs, [&a0[l], &a1[l], &a2[l], &a3[l]], &mut acc);
            }
            for (i, v) in acc.iter().enumerate() {
                _mm512_storeu_ps(out.as_mut_ptr().add(16 * i), *v);
            }
        }
    };
}

kernels!(
    gemm_group_q8_0,
    gemm_group16_q8_0,
    BlockQ8_0x8,
    |bb: &BlockQ8_0x8, _: ()| q8_weights(bb),
    ()
);

kernels!(
    gemm_group_mxfp4,
    gemm_group16_mxfp4,
    BlockMxFp4x8,
    |bb: &BlockMxFp4x8, lut: __m512i| mxfp4_weights(bb, lut),
    _mm512_broadcast_i32x4(_mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i))
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quant_cpu::{repack_mxfp4_x8, repack_q8_0_x8, BlockMxFp4, BlockQ8_0};

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    /// Activation groups as `quantize_act` makes them: every code of -127..=127, extremes included
    /// (-128 never occurs, and the AVX2 sign trick is not exact on it).
    fn activations(groups: usize, nb: usize, next: &mut impl FnMut() -> u64) -> Vec<BlockQ8_0x4> {
        (0..groups * nb)
            .map(|i| {
                let mut b = BlockQ8_0x4 {
                    d: [half::f16::ZERO; 4],
                    qs: [0; 128],
                };
                for m in 0..4 {
                    b.d[m] = half::f16::from_f32(0.001 + (next() % 1000) as f32 * 1e-4);
                }
                for q in b.qs.iter_mut() {
                    *q = match next() % 9 {
                        0 => 127,
                        1 => -127,
                        _ => (next() % 255) as i64 as i8 - 127,
                    };
                }
                if i == 0 {
                    b.qs[..32].fill(127);
                }
                b
            })
            .collect()
    }

    fn same(what: &str, got: &[f32], want: &[f32]) {
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{what} lane {i}: {g} vs the AVX2 {w}"
            );
        }
    }

    #[test]
    fn q8_0_has_the_bits_of_the_avx2_kernel() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let (nb, n) = (5usize, 8usize);
        let mut next = rng(0x9E37_79B9_7F4A_7C15);
        let raw: Vec<BlockQ8_0> = (0..n * nb)
            .map(|_| {
                let mut qs = [0i8; 32];
                for q in qs.iter_mut() {
                    *q = (next() % 255) as i64 as i8 - 127;
                }
                BlockQ8_0 {
                    d: half::f16::from_f32(0.01 + (next() % 100) as f32 * 1e-3),
                    qs,
                }
            })
            .collect();
        let w = repack_q8_0_x8::repack(&raw, n, nb);
        let acts = activations(4, nb, &mut next);
        let (mut want, mut got) = ([0f32; 32], [0f32; 32]);
        unsafe {
            repack_q8_0_x8::gemm_group_avx2(&acts[..nb], &w, nb, &mut want);
            gemm_group_q8_0(&acts[..nb], &w, nb, &mut got);
        }
        same("q8_0 4 rows", &got, &want);
        let (mut want, mut got) = ([0f32; 128], [0f32; 128]);
        let c = |i: usize| &acts[i * nb..(i + 1) * nb];
        unsafe {
            repack_q8_0_x8::gemm_group16_avx2(c(0), c(1), c(2), c(3), &w, nb, &mut want);
            gemm_group16_q8_0(c(0), c(1), c(2), c(3), &w, nb, &mut got);
        }
        same("q8_0 16 rows", &got, &want);
    }

    /// A weight of -128 against a negative activation overflows the AVX2 sign trick (the
    /// quantizer never emits it); the 512-bit kernel stays exact and answers the scalar oracle.
    #[test]
    fn q8_0_is_exact_on_the_most_negative_weight() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let (nb, n) = (3usize, 8usize);
        let mut next = rng(0xD1B5_4A32_D192_ED03);
        let raw: Vec<BlockQ8_0> = (0..n * nb)
            .map(|_| BlockQ8_0 {
                d: half::f16::from_f32(0.02),
                qs: [i8::MIN; 32],
            })
            .collect();
        let w = repack_q8_0_x8::repack(&raw, n, nb);
        let acts = activations(1, nb, &mut next);
        let (mut want, mut got) = ([0f32; 32], [0f32; 32]);
        repack_q8_0_x8::gemm_scalar(&acts[..nb], &w, nb, &mut want);
        unsafe { gemm_group_q8_0(&acts[..nb], &w, nb, &mut got) };
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 1e-4 * w.abs().max(1.0),
                "{g} vs the oracle {w}"
            );
        }
    }

    #[test]
    fn mxfp4_has_the_bits_of_the_avx2_kernel() {
        if !crate::cpu::hardware_has_avx512() {
            return;
        }
        let (nb, n) = (5usize, 8usize);
        let mut next = rng(0x2545_F491_4F6C_DD1D);
        let raw: Vec<BlockMxFp4> = (0..n * nb)
            .map(|_| {
                let mut qs = [0u8; 16];
                for q in qs.iter_mut() {
                    *q = next() as u8;
                }
                BlockMxFp4 {
                    e: 110 + (next() % 20) as u8,
                    qs,
                }
            })
            .collect();
        let w = repack_mxfp4_x8::repack(&raw, n, nb);
        let acts = activations(4, nb, &mut next);
        let (mut want, mut got) = ([0f32; 32], [0f32; 32]);
        unsafe {
            repack_mxfp4_x8::gemm_group_avx2(&acts[..nb], &w, nb, &mut want);
            gemm_group_mxfp4(&acts[..nb], &w, nb, &mut got);
        }
        same("mxfp4 4 rows", &got, &want);
        let (mut want, mut got) = ([0f32; 128], [0f32; 128]);
        let c = |i: usize| &acts[i * nb..(i + 1) * nb];
        unsafe {
            repack_mxfp4_x8::gemm_group16_avx2(c(0), c(1), c(2), c(3), &w, nb, &mut want);
            gemm_group16_mxfp4(c(0), c(1), c(2), c(3), &w, nb, &mut got);
        }
        same("mxfp4 16 rows", &got, &want);
    }
}
