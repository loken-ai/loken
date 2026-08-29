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

// q5_K's twelve scale bytes are packed exactly as q4_K's - the fifth bit lives
// in a separate plane and never touches them - so the split is read through
// q4_K's statement of it rather than restated here.
use super::repack_q4k::sub_scales_and_mins_bytes;
use super::{BlockQ5K, BlockQ8K, QK_K};
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use half::f16;

/// 8 interleaved `BlockQ5K` columns, stored in the native packed layout:
/// super scale + min, the 12 packed sub-scale/min bytes, the 4-bit base
/// plane and the 1-bit high plane, all unexpanded. Column `j` occupies the
/// contiguous slices `scales[j*12..]`, `qs[j*128..]`, `qh[j*32..]`, `d[j]`,
/// `dmin[j]`.
#[derive(Clone)]
#[repr(C)]
pub struct BlockQ5Kx8 {
    pub d: [f16; 8],
    pub dmin: [f16; 8],
    pub scales: [u8; 8 * 12],
    pub qs: [u8; 8 * (QK_K / 2)],
    pub qh: [u8; 8 * (QK_K / 8)],
}

impl BlockQ5Kx8 {
    fn zeros() -> Self {
        BlockQ5Kx8 {
            d: [f16::ZERO; 8],
            dmin: [f16::ZERO; 8],
            scales: [0u8; 8 * 12],
            qs: [0u8; 8 * (QK_K / 2)],
            qh: [0u8; 8 * (QK_K / 8)],
        }
    }
}

/// Repack `[n, k]` row-major `BlockQ5K` into `[n/8][k/256]` `BlockQ5Kx8`.
/// `n % 8 == 0`; a gather, so the values are unchanged.
pub fn repack(rhs: &[BlockQ5K], n: usize, nb: usize) -> Vec<BlockQ5Kx8> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(rhs.len(), n * nb);
    let groups = n / 8;
    let mut out = vec![BlockQ5Kx8::zeros(); groups * nb];
    for g in 0..groups {
        for sb in 0..nb {
            let dst = &mut out[g * nb + sb];
            for j in 0..8 {
                let src = &rhs[(g * 8 + j) * nb + sb];
                dst.d[j] = src.d;
                dst.dmin[j] = src.dmin;
                dst.scales[j * 12..j * 12 + 12].copy_from_slice(&src.scales);
                dst.qs[j * (QK_K / 2)..(j + 1) * (QK_K / 2)].copy_from_slice(&src.qs);
                dst.qh[j * (QK_K / 8)..(j + 1) * (QK_K / 8)].copy_from_slice(&src.qh);
            }
        }
    }
    out
}

/// Dequantize one column's super-block into the 5-bit weights (0..31), in
/// the element order the Q8_K activation is laid out. Mirrors the reference:
/// low then high nibble per 32-block, with the high bit taken from the plane
/// at the block's bit position.
#[inline]
fn dequant_col(qs: &[u8], qh: &[u8], aux8: &mut [u8; QK_K]) {
    for it in 0..(QK_K / 64) {
        let q5 = &qs[it * 32..];
        let lo_bit = 1u8 << (2 * it);
        let hi_bit = 1u8 << (2 * it + 1);
        for l in 0..32 {
            let hl = qh[l];
            aux8[it * 64 + l] = (q5[l] & 0x0F) + if hl & lo_bit != 0 { 16 } else { 0 };
            aux8[it * 64 + 32 + l] = (q5[l] >> 4) + if hl & hi_bit != 0 { 16 } else { 0 };
        }
    }
}

/// Scalar reference GEMV: one 8-column group x one activation row. THE
/// ORACLE. Integer accumulation is exact, so the tiled kernel is validated
/// against this and it must not change. Writes `out[0..8]`.
pub fn gemv_group_scalar(b: &[BlockQ5Kx8], a: &[BlockQ8K], nb: usize, out: &mut [f32]) {
    let mut sumf = [0f32; 8];
    let mut aux8 = [0u8; QK_K];
    for l in 0..nb {
        let blk = &b[l];
        let act = &a[l];
        for j in 0..8 {
            let qs = &blk.qs[j * (QK_K / 2)..];
            let qh = &blk.qh[j * (QK_K / 8)..];
            dequant_col(qs, qh, &mut aux8);
            let sm = sub_scales_and_mins_bytes(&blk.scales[j * 12..j * 12 + 12]);
            // Scaled integer dot: each 32-weight group shares a sub-scale.
            let mut sumi = 0i32;
            for (sc, &scale) in sm.iter().enumerate().take(8) {
                let mut gsum = 0i32;
                for e in 0..32 {
                    gsum += aux8[sc * 32 + e] as i32 * act.qs[sc * 32 + e] as i32;
                }
                sumi += scale as i32 * gsum;
            }
            // Min term: 16 sub-groups, each min shared by two groups of 16.
            let mut mint = 0i32;
            for grp in 0..16 {
                mint += act.bsums[grp] as i32 * sm[8 + grp / 2] as i32;
            }
            sumf[j] += blk.d[j].to_f32() * act.d * sumi as f32
                - blk.dmin[j].to_f32() * act.d * mint as f32;
        }
    }
    out[..8].copy_from_slice(&sumf);
}

/// Row-tiled GEMM: one 8-column group x `mt` activation rows. Each column's
/// weights decode once and meet every row of the tile - the reuse that makes
/// prefill compute-bound. Integer accumulation is exact, so this matches the
/// scalar oracle within the float reduction tolerance. `mt <= 8`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group_avx2(b: &[BlockQ5Kx8], acts: &[&[BlockQ8K]], nb: usize, out: &mut [f32]) {
    let mt = acts.len();
    let m4 = _mm256_set1_epi8(0x0F);
    // Column-outer so each row's accumulation mirrors the per-column
    // `vec_dot_q5k_q8k` bit-for-bit: the scaled integer dot lands in an f32
    // vector accumulator (`accm`) via one fmadd per super-block, and the min
    // term accumulates in a separate scalar (`summs`); the two combine only
    // at the end. Matching that reduction order keeps greedy tokens
    // identical (the K-quant dot is integer-exact, so only float summation
    // order can differ). `mt <= 8`.
    for j in 0..8 {
        let mut accm = [_mm256_setzero_ps(); 8];
        let mut summs = [0f32; 8];
        for l in 0..nb {
            let blk = &b[l];
            let qsp = blk.qs.as_ptr().add(j * (QK_K / 2));
            let qhp = blk.qh.as_ptr().add(j * (QK_K / 8));
            let sm = sub_scales_and_mins_bytes(&blk.scales[j * 12..j * 12 + 12]);
            // Decode this column's 256 weights (0..31) into eight 32-wide
            // registers, once for the whole tile. The high bit is OR'd into
            // bit 4 of each nibble.
            let mut wdec = [_mm256_setzero_si256(); 8];
            let qh = _mm256_loadu_si256(qhp as *const __m256i); // 32 high-bit bytes
            let mut wi = 0usize;
            for it in 0..(QK_K / 64) {
                let raw = _mm256_loadu_si256(qsp.add(it * 32) as *const __m256i);
                let lo = _mm256_and_si256(raw, m4);
                let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), m4);
                // high bit at plane position (2*it) for lo, (2*it+1) for hi:
                // isolate the bit and shift it to bit 4 (value 16). The plane
                // position is a runtime value, so use the variable shift.
                let one = _mm256_set1_epi8(1);
                let sh_lo = _mm_cvtsi32_si128(2 * it as i32);
                let sh_hi = _mm_cvtsi32_si128((2 * it + 1) as i32);
                let hb_lo =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, sh_lo), one), 4);
                let hb_hi =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, sh_hi), one), 4);
                wdec[wi] = _mm256_or_si256(lo, hb_lo);
                wdec[wi + 1] = _mm256_or_si256(hi, hb_hi);
                wi += 2;
            }
            // Per 32-weight group c, one sub-scale sm[c] broadcast to i16.
            let mut sv = [_mm256_setzero_si256(); 8];
            for (c, svc) in sv.iter_mut().enumerate() {
                *svc = _mm256_set1_epi16(sm[c] as i16);
            }
            let dj = blk.d[j].to_f32();
            let dminj = blk.dmin[j].to_f32();
            // Min contribution per column depends only on the weight (the
            // mins) and the row's bsums; build the per-column mins vector.
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
                let mut sumi = _mm256_setzero_si256();
                for (c, &w) in wdec.iter().enumerate() {
                    let q = _mm256_loadu_si256(q8.add(c * 32) as *const __m256i);
                    // Unsigned weight (0..31) . signed activation; pairs to
                    // i16, scaled by the group sub-scale and widened to i32.
                    let prod = _mm256_maddubs_epi16(w, q);
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(prod, sv[c]));
                }
                // Main term: same `y.d * x.d` scale and fmadd order as the
                // per-column reference (the scaled dot is exact in i32).
                let d = a.d * dj;
                accm[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accm[r]);
                // Min term from the row's group sums, scalar-accumulated with
                // the reference's `-y.d * x.dmin` sign convention.
                let bsums = _mm256_loadu_si256(a.bsums.as_ptr() as *const __m256i);
                let mint = hsum_i32(_mm256_madd_epi16(bsums, minv));
                let dmin = -a.d * dminj;
                summs[r] += dmin * mint as f32;
            }
        }
        for (r, s) in summs.iter().enumerate().take(mt) {
            out[r * 8 + j] += super::avx::lanes_sum(accm[r]) + s;
        }
    }
}

/// Reduce an 8-lane i32 register to a scalar sum.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[inline]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
    _mm_cvtsi128_si32(s)
}
