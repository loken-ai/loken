// Transcribed kernels: the index arithmetic IS the layout, the argument lists are the
// reference's, and the `unsafe fn`s wrap intrinsics whose contract is the intrinsic's.
// Named rather than `clippy::all` so anything else here still gets reported.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::missing_safety_doc,
    clippy::type_complexity,
    clippy::redundant_closure
)]

use super::{BlockQ6K, BlockQ8K, QK_K};
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use half::f16;

/// 8 interleaved `BlockQ6K` columns (one super-block each), stored in the
/// native packed layout: the 4-bit low plane (`ql`), the 2-bit high plane
/// (`qh`) and the signed group scales are kept unexpanded, so the group is
/// only as large as eight raw blocks rather than a dequantized byte plane.
/// Each column `j` occupies the contiguous slices `ql[j*128..]`,
/// `qh[j*64..]`, `scales[j*16..]`, `d[j]`.
#[derive(Clone)]
#[repr(C)]
pub struct BlockQ6Kx8 {
    pub d: [f16; 8],
    pub scales: [i8; 8 * (QK_K / 16)],
    pub ql: [u8; 8 * (QK_K / 2)],
    pub qh: [u8; 8 * (QK_K / 4)],
}

impl BlockQ6Kx8 {
    fn zeros() -> Self {
        BlockQ6Kx8 {
            d: [f16::ZERO; 8],
            scales: [0i8; 8 * (QK_K / 16)],
            ql: [0u8; 8 * (QK_K / 2)],
            qh: [0u8; 8 * (QK_K / 4)],
        }
    }
}

/// Repack `[n, k]` row-major `BlockQ6K` weights into `[n/8][k/256]`
/// `BlockQ6Kx8`. `n` must be a multiple of 8; callers fall back to the
/// per-column path otherwise. `nb = k / QK_K` superblocks per row. The
/// packed planes are copied verbatim per column - this is a gather, so the
/// dequantized values are unchanged.
pub fn repack(rhs: &[BlockQ6K], n: usize, nb: usize) -> Vec<BlockQ6Kx8> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(rhs.len(), n * nb);
    let groups = n / 8;
    let mut out = vec![BlockQ6Kx8::zeros(); groups * nb];
    for g in 0..groups {
        for sb in 0..nb {
            let dst = &mut out[g * nb + sb];
            for j in 0..8 {
                let src = &rhs[(g * 8 + j) * nb + sb];
                dst.d[j] = src.d;
                dst.scales[j * (QK_K / 16)..(j + 1) * (QK_K / 16)].copy_from_slice(&src.scales);
                dst.ql[j * (QK_K / 2)..(j + 1) * (QK_K / 2)].copy_from_slice(&src.ql);
                dst.qh[j * (QK_K / 4)..(j + 1) * (QK_K / 4)].copy_from_slice(&src.qh);
            }
        }
    }
    out
}

/// Dequantize one column's super-block into signed 6-bit weights, in the
/// element order the Q8_K activation is laid out. Mirrors the reference
/// dequant exactly: low plane nibble, high plane 2 bits shifted into place,
/// minus the fixed 32 zero-point.
#[inline]
fn dequant_col(ql: &[u8], qh: &[u8], aux8: &mut [i8; QK_K]) {
    let mut jj = 0;
    while jj < QK_K {
        let q4 = &ql[jj / 2..];
        let qhh = &qh[jj / 4..];
        for l in 0..32 {
            aux8[jj + l] = (((q4[l] & 0xF) | ((qhh[l] & 3) << 4)) as i32 - 32) as i8;
            aux8[jj + l + 32] =
                (((q4[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
            aux8[jj + l + 64] = (((q4[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
            aux8[jj + l + 96] =
                (((q4[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
        }
        jj += 128;
    }
}

/// Scalar reference GEMV: one 8-column group x one activation row. THE
/// ORACLE - the integer accumulation is exact (order-independent), so the
/// tiled AVX2 kernel is validated against this and it must not be changed.
/// Writes `out[0..8]`, one dot per column.
pub fn gemv_group_scalar(b: &[BlockQ6Kx8], a: &[BlockQ8K], nb: usize, out: &mut [f32]) {
    let mut sumf = [0f32; 8];
    let mut aux8 = [0i8; QK_K];
    for l in 0..nb {
        let blk = &b[l];
        let act = &a[l];
        for j in 0..8 {
            let ql = &blk.ql[j * (QK_K / 2)..];
            let qh = &blk.qh[j * (QK_K / 4)..];
            let sc = &blk.scales[j * (QK_K / 16)..];
            dequant_col(ql, qh, &mut aux8);
            // Integer dot, grouped by 16 weights sharing a signed scale.
            let mut sumi = 0i32;
            for (g, &scale) in sc.iter().enumerate().take(QK_K / 16) {
                let mut gs = 0i32;
                for e in 0..16 {
                    gs += aux8[g * 16 + e] as i32 * act.qs[g * 16 + e] as i32;
                }
                sumi += scale as i32 * gs;
            }
            sumf[j] += blk.d[j].to_f32() * act.d * sumi as f32;
        }
    }
    out[..8].copy_from_slice(&sumf);
}

/// Reduce an 8-lane i32 register to a scalar sum.
#[cfg(target_feature = "avx2")]
#[inline]
unsafe fn hsum_i32_8(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    _mm_cvtsi128_si32(s)
}

/// Row-tiled GEMM: one 8-column weight group x `mt` activation rows. Each
/// column's weights are decoded from the packed planes ONCE and then meet
/// every row of the tile, so the decode (which combines the two bit planes
/// and applies the zero-point) is paid once per tile instead of once per
/// row - the reuse that makes prefill compute-bound. Writes
/// `out[r*8 .. r*8+8]` for r in 0..mt. Integer accumulation is exact, so
/// this matches the scalar oracle within the float reduction tolerance.
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group_avx2(b: &[BlockQ6Kx8], acts: &[&[BlockQ8K]], nb: usize, out: &mut [f32]) {
    let mt = acts.len();
    let m4 = _mm256_set1_epi8(0xF);
    let m2 = _mm256_set1_epi8(3);
    for l in 0..nb {
        let blk = &b[l];
        for j in 0..8 {
            let qlp = blk.ql.as_ptr().add(j * (QK_K / 2));
            let qhp = blk.qh.as_ptr().add(j * (QK_K / 4));
            let scp = blk.scales.as_ptr().add(j * (QK_K / 16));
            // Decode this column's 256 UNSIGNED 6-bit weights (0..63) into
            // eight 32-wide registers (chunk c holds elements c*32..c*32+32),
            // once for the whole tile. The fixed 32 zero-point is applied
            // once per row below from the activation group-sums, rather than
            // subtracted from every weight. Reusing this decode across the
            // rows is what makes the tiled path faster than one decode per
            // (row, column).
            let mut wdec = [_mm256_setzero_si256(); 8];
            let mut jj = 0usize;
            let mut wi = 0usize;
            while jj < QK_K {
                let ql1 = _mm256_loadu_si256(qlp.add(jj / 2) as *const __m256i);
                let ql2 = _mm256_loadu_si256(qlp.add(jj / 2 + 32) as *const __m256i);
                let qh = _mm256_loadu_si256(qhp.add(jj / 4) as *const __m256i);
                let h0 = _mm256_slli_epi16(_mm256_and_si256(qh, m2), 4);
                let h1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m2), 4);
                let h2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), m2), 4);
                let h3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), m2), 4);
                wdec[wi] = _mm256_or_si256(_mm256_and_si256(ql1, m4), h0);
                wdec[wi + 1] = _mm256_or_si256(_mm256_and_si256(ql2, m4), h1);
                wdec[wi + 2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), h2);
                wdec[wi + 3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql2, 4), m4), h3);
                wi += 4;
                jj += 128;
            }
            // Per 32-weight chunk c, the two 16-groups it spans share scales
            // scp[2c] (low 16 elements) and scp[2c+1] (high 16). Build one
            // i16 vector per chunk with those two scales in the right halves.
            let mut sv = [_mm256_setzero_si256(); 8];
            for (c, svc) in sv.iter_mut().enumerate() {
                let s0 = *scp.add(2 * c) as i16;
                let s1 = *scp.add(2 * c + 1) as i16;
                *svc = _mm256_insertf128_si256(
                    _mm256_castsi128_si256(_mm_set1_epi16(s0)),
                    _mm_set1_epi16(s1),
                    1,
                );
            }
            // The 16 signed scales, natural order, for the zero-point term.
            let scales16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(scp as *const __m128i));
            let dj = blk.d[j].to_f32();
            for (r, act) in acts.iter().enumerate().take(mt) {
                let a = &act[l];
                let q8 = a.qs.as_ptr();
                let mut sumi = _mm256_setzero_si256();
                for (c, &w) in wdec.iter().enumerate() {
                    let q = _mm256_loadu_si256(q8.add(c * 32) as *const __m256i);
                    // Unsigned weight . signed activation; pairs land in i16,
                    // the per-group scale folds in and widens to i32.
                    let prod = _mm256_maddubs_epi16(w, q);
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(prod, sv[c]));
                }
                // Fold the 32 zero-point once: 32.Σ scale[g].bsum[g], where
                // bsum[g] is this row's sum of activations in group g.
                let bsums = _mm256_loadu_si256(a.bsums.as_ptr() as *const __m256i);
                let corr = _mm256_slli_epi32(_mm256_madd_epi16(bsums, scales16), 5);
                let s = hsum_i32_8(_mm256_sub_epi32(sumi, corr));
                out[r * 8 + j] += dj * a.d * s as f32;
            }
        }
    }
}
