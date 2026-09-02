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

use super::{BlockQ4K, BlockQ8K, QK_K};
#[cfg(all(target_feature = "avx2", target_arch = "x86"))]
use core::arch::x86::*;
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use half::f16;

/// 8 interleaved `BlockQ4K` rows (one superblock each). Field layout matches
/// llama.cpp `block_q4_Kx8`: `d[8]`, `dmin[8]`, `scales[96]` (12 packed bytes
/// per row), `qs[1024]` (128 nibble bytes per row, interleaved 8 bytes at a
/// time across the 8 rows).
#[derive(Clone)]
#[repr(C)]
pub struct BlockQ4Kx8 {
    pub d: [f16; 8],
    pub dmin: [f16; 8],
    pub scales: [u8; 96],
    pub qs: [u8; 1024],
}

impl BlockQ4Kx8 {
    fn zeros() -> Self {
        BlockQ4Kx8 {
            d: [f16::ZERO; 8],
            dmin: [f16::ZERO; 8],
            scales: [0u8; 96],
            qs: [0u8; 1024],
        }
    }
}

/// Repack `[n, k]` row-major `BlockQ4K` weights into `[n/8][k/256]`
/// `BlockQ4Kx8`. `n` must be a multiple of 8; callers fall back to the
/// per-column path otherwise. `nb = k / QK_K` superblocks per row.
pub fn repack(rhs: &[BlockQ4K], n: usize, nb: usize) -> Vec<BlockQ4Kx8> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(rhs.len(), n * nb);
    let groups = n / 8;
    let mut out = vec![BlockQ4Kx8::zeros(); groups * nb];
    for g in 0..groups {
        for sb in 0..nb {
            let dst = &mut out[g * nb + sb];
            for j in 0..8 {
                let src = &rhs[(g * 8 + j) * nb + sb];
                dst.d[j] = src.d;
                dst.dmin[j] = src.dmin;
                // 12 packed scale/min bytes per row, contiguous per row.
                dst.scales[j * 12..j * 12 + 12].copy_from_slice(&src.scales);
                // Interleave the 128 nibble bytes: source byte p of row j
                // lands at (p/8)*64 + j*8 + (p%8) so the GEMV streams one
                // 8-byte lane per row across all 8 columns.
                for p in 0..128 {
                    dst.qs[(p / 8) * 64 + j * 8 + (p % 8)] = src.qs[p];
                }
            }
        }
    }
    out
}

/// The eight sub-block scales and eight sub-block minimums a q4_K or q5_K
/// super-block packs into twelve bytes, six bits each. THE ONE STATEMENT of
/// that layout on this side; a second one drifts silently, because a wrong
/// unpack yields a plausible scale rather than an error.
///
/// Bytes 0..4 hold scales 0..4 and bytes 4..8 minimums 0..4, in their low six
/// bits. The two bits left spare above each of those eight are the high half of
/// the four scales and four minimums that remain; the low halves are the
/// nibbles of bytes 8..12 - a scale's low, a minimum's high.
///
/// Returned as four words - scales 0..4, scales 4..8, minimums 0..4, minimums
/// 4..8 - the form a kernel loads straight into a register.
/// [`sub_scales_and_mins_bytes`] is the same sixteen values indexable one at a
/// time; every reader of the packing goes through one of the two.
#[inline(always)]
pub(crate) fn sub_scales_and_mins(packed: &[u8]) -> [u32; 4] {
    // A field stored whole: six bits, in every byte of a word.
    const WHOLE: u32 = 0x3f3f_3f3f;
    // The low half of an assembled field: four bits, in every byte.
    const LOW_HALF: u32 = 0x0f0f_0f0f;
    // Its high half, once the two spare bits sit at the bottom of the byte.
    const HIGH_HALF: u32 = 0x0303_0303;

    let word =
        |i: usize| u32::from_le_bytes([packed[i], packed[i + 1], packed[i + 2], packed[i + 3]]);
    let (scales, mins, halves) = (word(0), word(4), word(8));
    // An assembled field is its nibble plus the two bits its whole counterpart
    // left above bit 5, moved up to bits 4 and 5.
    let assemble = |low_half: u32, whole: u32| low_half | (((whole >> 6) & HIGH_HALF) << 4);
    [
        scales & WHOLE,
        assemble(halves & LOW_HALF, scales),
        mins & WHOLE,
        assemble((halves >> 4) & LOW_HALF, mins),
    ]
}

/// The same sixteen values as [`sub_scales_and_mins`], one byte each: scales
/// `0..8` then minimums `0..8`, for the readers that index one sub-block rather
/// than feed a register.
#[inline(always)]
pub(crate) fn sub_scales_and_mins_bytes(packed: &[u8]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for (quarter, word) in sub_scales_and_mins(packed).iter().enumerate() {
        out[quarter * 4..quarter * 4 + 4].copy_from_slice(&word.to_le_bytes());
    }
    out
}

/// Like `repack`, but the `scales[96]` are re-packed six-bit as well, in the
/// sub-block-interleaved order `gemv_group_avx2_v2` and the `gemm_*` kernels
/// read: four consecutive rows' scales, then their minimums, then the byte
/// pairing each row's spare high halves. d/dmin/qs are identical to `repack`,
/// and the values are the same six-bit fields - bit-identical results to dot.
pub fn repack_v2(rhs: &[BlockQ4K], n: usize, nb: usize) -> Vec<BlockQ4Kx8> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(rhs.len(), n * nb);
    let groups = n / 8;
    let mut out = vec![BlockQ4Kx8::zeros(); groups * nb];
    for g in 0..groups {
        for sb in 0..nb {
            let dst = &mut out[g * nb + sb];
            let inr = |j: usize| &rhs[(g * 8 + j) * nb + sb];
            for j in 0..8 {
                dst.d[j] = inr(j).d;
                dst.dmin[j] = inr(j).dmin;
                for p in 0..128 {
                    dst.qs[(p / 8) * 64 + j * 8 + (p % 8)] = inr(j).qs[p];
                }
            }
            // The eight rows' sixteen six-bit fields, read once through the one
            // statement of the packing, then re-packed twelve bytes at a time.
            let row: [[u8; 16]; 8] =
                core::array::from_fn(|j| sub_scales_and_mins_bytes(&inr(j).scales));
            // A group of twelve output bytes holds the same sub-block field of
            // all eight rows: rows 0..4 whole (their top two bits carrying rows
            // 4..8's high halves), then rows 4..8's low nibbles, scale and
            // minimum sharing a byte.
            let mut pack12 = |at: usize, s: &[u8; 8], m: &[u8; 8]| {
                for t in 0..4 {
                    dst.scales[at + t] = (s[t] & 63) + ((s[t + 4] & 48) << 2);
                    dst.scales[at + 4 + t] = (m[t] & 63) + ((m[t + 4] & 48) << 2);
                    dst.scales[at + 8 + t] = (s[t + 4] & 15) + ((m[t + 4] & 15) << 4);
                }
            };
            // Sub-blocks 0..4 fill the first half of `scales`, 4..8 the second.
            for i in 0..4 {
                let s: [u8; 8] = core::array::from_fn(|j| row[j][i]);
                let m: [u8; 8] = core::array::from_fn(|j| row[j][8 + i]);
                pack12(i * 12, &s, &m);
            }
            for i in 0..4 {
                let s: [u8; 8] = core::array::from_fn(|j| row[j][4 + i]);
                let m: [u8; 8] = core::array::from_fn(|j| row[j][12 + i]);
                pack12(i * 12 + 48, &s, &m);
            }
        }
    }
    out
}

/// Scalar reference (parity oracle): one column group of 8 outputs from the
/// interleaved weights and a single activation row. Mirrors llama.cpp's
/// `ggml_gemv_q4_K_8x8_q8_K_generic`.
pub fn gemv_group_scalar(b: &[BlockQ4Kx8], a: &[BlockQ8K], nb: usize, out: &mut [f32]) {
    let blocklen = 8usize;
    let mut sumf = [0f32; 8];
    let mut sum_minf = [0f32; 8];
    for l in 0..nb {
        let blk = &b[l];
        let act = &a[l];
        // Unpack the 8 rows' scales/mins into a [8][16] table.
        let mut sm = [[0u8; 16]; 8];
        for (sb, smv) in sm.iter_mut().enumerate() {
            *smv = sub_scales_and_mins_bytes(&blk.scales[sb * 12..sb * 12 + 12]);
        }
        for k in 0..(QK_K / (2 * blocklen)) {
            // sub-block pair index; low nibble uses sub-block 2.sbk, high
            // uses 2.sbk+1. `sm[col]` holds column `col`'s 8 scales + 8 mins.
            let sbk = k / 4;
            for j in 0..8 {
                let scale_lo = sm[j][sbk * 2] as i32;
                let scale_hi = sm[j][sbk * 2 + 1] as i32;
                // Plain Q8_K activation order: weight byte k*8+i of column j
                // dequantizes to elements e_lo/e_hi (low/high nibble).
                let ebase = (k / 4) * 64 + (k % 4) * 8;
                let mut sumi = 0i32;
                for i in 0..blocklen {
                    let q = blk.qs[k * 64 + j * 8 + i];
                    let v0 = (q & 0x0F) as i32;
                    let v1 = (q >> 4) as i32;
                    let a0 = act.qs[ebase + i] as i32;
                    let a1 = act.qs[ebase + 32 + i] as i32;
                    sumi += v0 * a0 * scale_lo + v1 * a1 * scale_hi;
                }
                sumf[j] += sumi as f32 * blk.d[j].to_f32() * act.d;
            }
        }
        for sb in 0..8 {
            let bsum = (act.bsums[sb * 2] as i32 + act.bsums[sb * 2 + 1] as i32) as f32;
            for j in 0..8 {
                let min = sm[j][8 + sb] as f32;
                sum_minf[j] += min * bsum * blk.dmin[j].to_f32() * act.d;
            }
        }
    }
    for j in 0..8 {
        out[j] = sumf[j] - sum_minf[j];
    }
}

/// Row-tiled GEMM: one 8-column weight group x `mt` activation rows
/// (`acts[r]` = row r's `nb` Q8_K blocks). Each loaded weight nibble feeds
/// all `mt` rows before the next is read - the weight-reuse that makes
/// prefill compute-bound instead of re-streaming the matrix per row. Writes
/// `out[r*8 .. r*8+8]` for r in 0..mt. `mt <= 4`.
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group_avx2(b: &[BlockQ4Kx8], acts: &[&[BlockQ8K]], nb: usize, out: &mut [f32]) {
    let mt = acts.len();
    debug_assert!(mt <= 8);
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); 8];
    let mut acc_min = [_mm256_setzero_ps(); 8];
    // Straightening permutation, applied once per (super-block, row): the
    // pairwise horizontal add of the two half-width column accumulators
    // yields the interleaved order [c0 c1 c4 c5 c2 c3 c6 c7]; this puts the
    // columns back into lane order c0..c7.
    let straighten = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for l in 0..nb {
        let blk = &b[l];
        let mut sm = [[0u8; 16]; 8];
        for (sb, smv) in sm.iter_mut().enumerate() {
            *smv = sub_scales_and_mins_bytes(&blk.scales[sb * 12..sb * 12 + 12]);
        }
        // Per-column dot accumulators kept in the NATIVE maddubs lane order:
        // two i32 partials per column, columns 0-3 in `isum_a` and 4-7 in
        // `isum_b`. Nothing is shuffled across lanes here - the cross-lane
        // straighten is deferred to once per super-block per row, because
        // the float scale d[j].act.d is the first thing that actually needs
        // one value per column.
        let mut isum_a = [_mm256_setzero_si256(); 8];
        let mut isum_b = [_mm256_setzero_si256(); 8];
        let qs = blk.qs.as_ptr();
        // All 4 k-steps of a sub-block `sbk = k/4` share the same pair of
        // 6-bit scales. Rather than reduce+scale every k-step, accumulate
        // the raw maddubs partials in i16 across the whole sub-block and
        // apply the scale ONCE with a single madd - which simultaneously
        // multiplies by the scale, widens i16->i32 and folds pairs. This is
        // the same integer sum by distributivity, just factored.
        for sbk in 0..(QK_K / 64) {
            let s0 = sbk * 2;
            let s1 = sbk * 2 + 1;
            // Each column's 6-bit scale, broadcast over that column's four
            // i16 partial lanes.
            let scale_lo_a = scale_lanes_i16(&sm, 0, s0);
            let scale_lo_b = scale_lanes_i16(&sm, 4, s0);
            let scale_hi_a = scale_lanes_i16(&sm, 0, s1);
            let scale_hi_b = scale_lanes_i16(&sm, 4, s1);
            // i16 sub-block partials, per tile row. Headroom: one maddubs
            // lane is 2 products of a 4-bit unsigned weight by a signed
            // 8-bit activation, and only the 4 k-steps of one sub-block are
            // summed before the widening madd - bounded well inside i16.
            let mut p_lo_a = [_mm256_setzero_si256(); 8];
            let mut p_lo_b = [_mm256_setzero_si256(); 8];
            let mut p_hi_a = [_mm256_setzero_si256(); 8];
            let mut p_hi_b = [_mm256_setzero_si256(); 8];
            for kk in 0..4 {
                let k = sbk * 4 + kk;
                // Decode the weight nibbles ONCE for this k...
                let q = _mm256_loadu_si256(qs.add(k * 64) as *const __m256i);
                let q2 = _mm256_loadu_si256(qs.add(k * 64 + 32) as *const __m256i);
                let qlo0 = _mm256_and_si256(q, m4);
                let qhi0 = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                let qlo1 = _mm256_and_si256(q2, m4);
                let qhi1 = _mm256_and_si256(_mm256_srli_epi16(q2, 4), m4);
                let off = sbk * 64 + kk * 8;
                // ...and apply to every activation row in the tile. The
                // inner step is now nothing but maddubs + i16 add.
                for r in 0..mt {
                    let act = &acts[r][l];
                    let abase = act.qs.as_ptr().add(off);
                    let a_lo = load_a8_broadcast(abase);
                    let a_hi = load_a8_broadcast(abase.add(32));
                    p_lo_a[r] = _mm256_add_epi16(p_lo_a[r], _mm256_maddubs_epi16(qlo0, a_lo));
                    p_lo_b[r] = _mm256_add_epi16(p_lo_b[r], _mm256_maddubs_epi16(qlo1, a_lo));
                    p_hi_a[r] = _mm256_add_epi16(p_hi_a[r], _mm256_maddubs_epi16(qhi0, a_hi));
                    p_hi_b[r] = _mm256_add_epi16(p_hi_b[r], _mm256_maddubs_epi16(qhi1, a_hi));
                }
            }
            // One scale+widen+reduce per sub-block per row, in-lane.
            for r in 0..mt {
                isum_a[r] = _mm256_add_epi32(
                    isum_a[r],
                    _mm256_add_epi32(
                        _mm256_madd_epi16(p_lo_a[r], scale_lo_a),
                        _mm256_madd_epi16(p_hi_a[r], scale_hi_a),
                    ),
                );
                isum_b[r] = _mm256_add_epi32(
                    isum_b[r],
                    _mm256_add_epi32(
                        _mm256_madd_epi16(p_lo_b[r], scale_lo_b),
                        _mm256_madd_epi16(p_hi_b[r], scale_hi_b),
                    ),
                );
            }
        }
        let dvec = _mm256_setr_ps(
            blk.d[0].to_f32(),
            blk.d[1].to_f32(),
            blk.d[2].to_f32(),
            blk.d[3].to_f32(),
            blk.d[4].to_f32(),
            blk.d[5].to_f32(),
            blk.d[6].to_f32(),
            blk.d[7].to_f32(),
        );
        let dminv = _mm256_setr_ps(
            blk.dmin[0].to_f32(),
            blk.dmin[1].to_f32(),
            blk.dmin[2].to_f32(),
            blk.dmin[3].to_f32(),
            blk.dmin[4].to_f32(),
            blk.dmin[5].to_f32(),
            blk.dmin[6].to_f32(),
            blk.dmin[7].to_f32(),
        );
        // The 8-lane min gathers depend only on the weight super-block, not
        // on the activation row, yet sat inside the row loop and were
        // rebuilt identically for every row of the tile. Build them once per
        // super-block. Same values, just hoisted.
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
            // Fold the two i32 partials per column and reorder to c0..c7  -
            // the one cross-lane step of the whole super-block, needed only
            // because the float scale below is per column.
            let paired = _mm256_hadd_epi32(isum_a[r], isum_b[r]);
            let isum = _mm256_permutevar8x32_epi32(paired, straighten);
            let dall = _mm256_mul_ps(dvec, _mm256_set1_ps(act.d));
            acc[r] = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(isum), acc[r]);
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

/// Same result as `gemm_group_avx2`, but the sub-block accumulation is
/// blocked into row strips of `RT`. The wide-tile form keeps four i16
/// partials live per row across the whole k-step loop; past a handful of
/// rows that exceeds the architectural vector register file, so every
/// k-step turns into load-modify-store against spill slots. Narrowing the
/// strip keeps the partials resident and pays instead by re-decoding the
/// weight nibbles once per strip - a decode is a few ops, a spilled
/// accumulator is touched on every k-step.
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group_avx2_rt<const RT: usize>(
    b: &[BlockQ4Kx8],
    acts: &[&[BlockQ8K]],
    nb: usize,
    out: &mut [f32],
) {
    let mt = acts.len();
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); 8];
    let mut acc_min = [_mm256_setzero_ps(); 8];
    let straighten = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for l in 0..nb {
        let blk = &b[l];
        let mut sm = [[0u8; 16]; 8];
        for (sb, smv) in sm.iter_mut().enumerate() {
            *smv = sub_scales_and_mins_bytes(&blk.scales[sb * 12..sb * 12 + 12]);
        }
        let mut isum_a = [_mm256_setzero_si256(); 8];
        let mut isum_b = [_mm256_setzero_si256(); 8];
        let qs = blk.qs.as_ptr();
        for sbk in 0..(QK_K / 64) {
            let s0 = sbk * 2;
            let s1 = sbk * 2 + 1;
            let scale_lo_a = scale_lanes_i16(&sm, 0, s0);
            let scale_lo_b = scale_lanes_i16(&sm, 4, s0);
            let scale_hi_a = scale_lanes_i16(&sm, 0, s1);
            let scale_hi_b = scale_lanes_i16(&sm, 4, s1);
            let mut rt0 = 0;
            while rt0 < mt {
                let rn = RT.min(mt - rt0);
                let mut p_lo_a = [_mm256_setzero_si256(); RT];
                let mut p_lo_b = [_mm256_setzero_si256(); RT];
                let mut p_hi_a = [_mm256_setzero_si256(); RT];
                let mut p_hi_b = [_mm256_setzero_si256(); RT];
                for kk in 0..4 {
                    let k = sbk * 4 + kk;
                    let q = _mm256_loadu_si256(qs.add(k * 64) as *const __m256i);
                    let q2 = _mm256_loadu_si256(qs.add(k * 64 + 32) as *const __m256i);
                    let qlo0 = _mm256_and_si256(q, m4);
                    let qhi0 = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                    let qlo1 = _mm256_and_si256(q2, m4);
                    let qhi1 = _mm256_and_si256(_mm256_srli_epi16(q2, 4), m4);
                    let off = sbk * 64 + kk * 8;
                    for j in 0..rn {
                        let act = &acts[rt0 + j][l];
                        let abase = act.qs.as_ptr().add(off);
                        let a_lo = load_a8_broadcast(abase);
                        let a_hi = load_a8_broadcast(abase.add(32));
                        p_lo_a[j] = _mm256_add_epi16(p_lo_a[j], _mm256_maddubs_epi16(qlo0, a_lo));
                        p_lo_b[j] = _mm256_add_epi16(p_lo_b[j], _mm256_maddubs_epi16(qlo1, a_lo));
                        p_hi_a[j] = _mm256_add_epi16(p_hi_a[j], _mm256_maddubs_epi16(qhi0, a_hi));
                        p_hi_b[j] = _mm256_add_epi16(p_hi_b[j], _mm256_maddubs_epi16(qhi1, a_hi));
                    }
                }
                for j in 0..rn {
                    let r = rt0 + j;
                    isum_a[r] = _mm256_add_epi32(
                        isum_a[r],
                        _mm256_add_epi32(
                            _mm256_madd_epi16(p_lo_a[j], scale_lo_a),
                            _mm256_madd_epi16(p_hi_a[j], scale_hi_a),
                        ),
                    );
                    isum_b[r] = _mm256_add_epi32(
                        isum_b[r],
                        _mm256_add_epi32(
                            _mm256_madd_epi16(p_lo_b[j], scale_lo_b),
                            _mm256_madd_epi16(p_hi_b[j], scale_hi_b),
                        ),
                    );
                }
                rt0 += rn;
            }
        }
        let dvec = _mm256_setr_ps(
            blk.d[0].to_f32(),
            blk.d[1].to_f32(),
            blk.d[2].to_f32(),
            blk.d[3].to_f32(),
            blk.d[4].to_f32(),
            blk.d[5].to_f32(),
            blk.d[6].to_f32(),
            blk.d[7].to_f32(),
        );
        let dminv = _mm256_setr_ps(
            blk.dmin[0].to_f32(),
            blk.dmin[1].to_f32(),
            blk.dmin[2].to_f32(),
            blk.dmin[3].to_f32(),
            blk.dmin[4].to_f32(),
            blk.dmin[5].to_f32(),
            blk.dmin[6].to_f32(),
            blk.dmin[7].to_f32(),
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
            let paired = _mm256_hadd_epi32(isum_a[r], isum_b[r]);
            let isum = _mm256_permutevar8x32_epi32(paired, straighten);
            let dall = _mm256_mul_ps(dvec, _mm256_set1_ps(act.d));
            acc[r] = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(isum), acc[r]);
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

/// Per-super-block scale vectors, which depend only on the weight group.
/// The packed 6-bit scales must be widened into one lane per column before
/// they can meet the integer partials; that widening is a chain of scalar
/// inserts, so doing it inside the row loop repeats it once per row tile
/// even though the values never change. Building it once per weight group
/// hoists that work out of the hot path.
#[cfg(target_feature = "avx2")]
#[derive(Clone, Copy)]
pub struct GroupScales {
    pub(super) lo_a: [__m256i; 4],
    pub(super) lo_b: [__m256i; 4],
    pub(super) hi_a: [__m256i; 4],
    pub(super) hi_b: [__m256i; 4],
    pub(super) minf: [__m256; 8],
    pub(super) dvec: __m256,
    pub(super) dminv: __m256,
}

/// Precompute `GroupScales` for every super-block of one weight group.
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn prep_group_scales(b: &[BlockQ4Kx8], nb: usize) -> Vec<GroupScales> {
    let mut out = Vec::with_capacity(nb);
    for l in 0..nb {
        let blk = &b[l];
        let mut sm = [[0u8; 16]; 8];
        for (sb, smv) in sm.iter_mut().enumerate() {
            *smv = sub_scales_and_mins_bytes(&blk.scales[sb * 12..sb * 12 + 12]);
        }
        let mut lo_a = [_mm256_setzero_si256(); 4];
        let mut lo_b = [_mm256_setzero_si256(); 4];
        let mut hi_a = [_mm256_setzero_si256(); 4];
        let mut hi_b = [_mm256_setzero_si256(); 4];
        for sbk in 0..4 {
            lo_a[sbk] = scale_lanes_i16(&sm, 0, sbk * 2);
            lo_b[sbk] = scale_lanes_i16(&sm, 4, sbk * 2);
            hi_a[sbk] = scale_lanes_i16(&sm, 0, sbk * 2 + 1);
            hi_b[sbk] = scale_lanes_i16(&sm, 4, sbk * 2 + 1);
        }
        let mut minf = [_mm256_setzero_ps(); 8];
        for (sb, mv) in minf.iter_mut().enumerate() {
            let mi = 8 + sb;
            *mv = _mm256_cvtepi32_ps(_mm256_setr_epi32(
                sm[0][mi] as i32,
                sm[1][mi] as i32,
                sm[2][mi] as i32,
                sm[3][mi] as i32,
                sm[4][mi] as i32,
                sm[5][mi] as i32,
                sm[6][mi] as i32,
                sm[7][mi] as i32,
            ));
        }
        let dvec = _mm256_setr_ps(
            blk.d[0].to_f32(),
            blk.d[1].to_f32(),
            blk.d[2].to_f32(),
            blk.d[3].to_f32(),
            blk.d[4].to_f32(),
            blk.d[5].to_f32(),
            blk.d[6].to_f32(),
            blk.d[7].to_f32(),
        );
        let dminv = _mm256_setr_ps(
            blk.dmin[0].to_f32(),
            blk.dmin[1].to_f32(),
            blk.dmin[2].to_f32(),
            blk.dmin[3].to_f32(),
            blk.dmin[4].to_f32(),
            blk.dmin[5].to_f32(),
            blk.dmin[6].to_f32(),
            blk.dmin[7].to_f32(),
        );
        out.push(GroupScales {
            lo_a,
            lo_b,
            hi_a,
            hi_b,
            minf,
            dvec,
            dminv,
        });
    }
    out
}

/// Hoisted scale vectors combined with row-strip blocking: removes the
/// per-row-tile scale rebuild and keeps the hot integer partials inside the
/// vector register file at the same time.
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group_avx2_pre_rt<const RT: usize>(
    b: &[BlockQ4Kx8],
    acts: &[&[BlockQ8K]],
    nb: usize,
    gs: &[GroupScales],
    out: &mut [f32],
) {
    let mt = acts.len();
    let m4 = _mm256_set1_epi8(0x0F);
    let mut acc = [_mm256_setzero_ps(); 8];
    let mut acc_min = [_mm256_setzero_ps(); 8];
    let straighten = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
    for l in 0..nb {
        let blk = &b[l];
        let g = &gs[l];
        let mut isum_a = [_mm256_setzero_si256(); 8];
        let mut isum_b = [_mm256_setzero_si256(); 8];
        let qs = blk.qs.as_ptr();
        for sbk in 0..(QK_K / 64) {
            let scale_lo_a = g.lo_a[sbk];
            let scale_lo_b = g.lo_b[sbk];
            let scale_hi_a = g.hi_a[sbk];
            let scale_hi_b = g.hi_b[sbk];
            let mut rt0 = 0;
            while rt0 < mt {
                let rn = RT.min(mt - rt0);
                let mut p_lo_a = [_mm256_setzero_si256(); RT];
                let mut p_lo_b = [_mm256_setzero_si256(); RT];
                let mut p_hi_a = [_mm256_setzero_si256(); RT];
                let mut p_hi_b = [_mm256_setzero_si256(); RT];
                for kk in 0..4 {
                    let k = sbk * 4 + kk;
                    let q = _mm256_loadu_si256(qs.add(k * 64) as *const __m256i);
                    let q2 = _mm256_loadu_si256(qs.add(k * 64 + 32) as *const __m256i);
                    let qlo0 = _mm256_and_si256(q, m4);
                    let qhi0 = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                    let qlo1 = _mm256_and_si256(q2, m4);
                    let qhi1 = _mm256_and_si256(_mm256_srli_epi16(q2, 4), m4);
                    let off = sbk * 64 + kk * 8;
                    for j in 0..rn {
                        let act = &acts[rt0 + j][l];
                        let abase = act.qs.as_ptr().add(off);
                        let a_lo = load_a8_broadcast(abase);
                        let a_hi = load_a8_broadcast(abase.add(32));
                        p_lo_a[j] = _mm256_add_epi16(p_lo_a[j], _mm256_maddubs_epi16(qlo0, a_lo));
                        p_lo_b[j] = _mm256_add_epi16(p_lo_b[j], _mm256_maddubs_epi16(qlo1, a_lo));
                        p_hi_a[j] = _mm256_add_epi16(p_hi_a[j], _mm256_maddubs_epi16(qhi0, a_hi));
                        p_hi_b[j] = _mm256_add_epi16(p_hi_b[j], _mm256_maddubs_epi16(qhi1, a_hi));
                    }
                }
                for j in 0..rn {
                    let r = rt0 + j;
                    isum_a[r] = _mm256_add_epi32(
                        isum_a[r],
                        _mm256_add_epi32(
                            _mm256_madd_epi16(p_lo_a[j], scale_lo_a),
                            _mm256_madd_epi16(p_hi_a[j], scale_hi_a),
                        ),
                    );
                    isum_b[r] = _mm256_add_epi32(
                        isum_b[r],
                        _mm256_add_epi32(
                            _mm256_madd_epi16(p_lo_b[j], scale_lo_b),
                            _mm256_madd_epi16(p_hi_b[j], scale_hi_b),
                        ),
                    );
                }
                rt0 += rn;
            }
        }
        for r in 0..mt {
            let act = &acts[r][l];
            let paired = _mm256_hadd_epi32(isum_a[r], isum_b[r]);
            let isum = _mm256_permutevar8x32_epi32(paired, straighten);
            let dall = _mm256_mul_ps(g.dvec, _mm256_set1_ps(act.d));
            acc[r] = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(isum), acc[r]);
            let mut imin = _mm256_setzero_ps();
            for sb in 0..8 {
                let bsum = (act.bsums[sb * 2] as i32 + act.bsums[sb * 2 + 1] as i32) as f32;
                imin = _mm256_fmadd_ps(g.minf[sb], _mm256_set1_ps(bsum), imin);
            }
            let dminall = _mm256_mul_ps(g.dminv, _mm256_set1_ps(act.d));
            acc_min[r] = _mm256_fmadd_ps(dminall, imin, acc_min[r]);
        }
    }
    for r in 0..mt {
        let res = _mm256_sub_ps(acc[r], acc_min[r]);
        _mm256_storeu_ps(out.as_mut_ptr().add(r * 8), res);
    }
}

#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_group_avx2(b: &[BlockQ4Kx8], a: &[BlockQ8K], nb: usize, out: &mut [f32]) {
    let m4 = _mm256_set1_epi8(0x0F);
    // 8 column accumulators in one __m256 (one f32 lane per column).
    let mut acc = _mm256_setzero_ps();
    let mut acc_min = _mm256_setzero_ps();
    for l in 0..nb {
        // SW-prefetch the (large) next Q4_K super-block group - shared primitive, same
        // BW-saturation win as Q4_0 (4 lines since BlockQ4Kx8 is large).
        const PF: usize = 3;
        if l + PF < nb {
            super::gemv_prefetch_t0::<4>(b.as_ptr().add(l + PF) as *const i8);
        }
        let blk = &b[l];
        let act = &a[l];
        let mut sm = [[0u8; 16]; 8];
        for (sb, smv) in sm.iter_mut().enumerate() {
            *smv = sub_scales_and_mins_bytes(&blk.scales[sb * 12..sb * 12 + 12]);
        }
        // Per-column integer accumulator (32-bit, in i32 lanes across cols).
        let mut isum = _mm256_setzero_si256();
        let qs = blk.qs.as_ptr();
        for k in 0..(QK_K / 16) {
            let sbk = k / 4;
            // scale_lo lane j = column j's scale for sub-block 2.sbk.
            let s0 = sbk * 2;
            let s1 = sbk * 2 + 1;
            let scale_lo = _mm256_setr_epi32(
                sm[0][s0] as i32,
                sm[1][s0] as i32,
                sm[2][s0] as i32,
                sm[3][s0] as i32,
                sm[4][s0] as i32,
                sm[5][s0] as i32,
                sm[6][s0] as i32,
                sm[7][s0] as i32,
            );
            let scale_hi = _mm256_setr_epi32(
                sm[0][s1] as i32,
                sm[1][s1] as i32,
                sm[2][s1] as i32,
                sm[3][s1] as i32,
                sm[4][s1] as i32,
                sm[5][s1] as i32,
                sm[6][s1] as i32,
                sm[7][s1] as i32,
            );
            // 64 interleaved nibble bytes for this k (8 cols x 8 bytes).
            let q = _mm256_loadu_si256(qs.add(k * 64) as *const __m256i);
            let q2 = _mm256_loadu_si256(qs.add(k * 64 + 32) as *const __m256i);
            let qlo0 = _mm256_and_si256(q, m4);
            let qhi0 = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            let qlo1 = _mm256_and_si256(q2, m4);
            let qhi1 = _mm256_and_si256(_mm256_srli_epi16(q2, 4), m4);
            // Plain Q8_K activation: the 8-byte lane the low/high nibbles
            // of this k pair against, broadcast to all 8 column lanes.
            let abase = act.qs.as_ptr().add((k / 4) * 64 + (k % 4) * 8);
            let a_lo = load_a8_broadcast(abase);
            let a_hi = load_a8_broadcast(abase.add(32));
            // maddubs over interleaved 8-byte lanes: each i32 column lane
            // accumulates the 8-element low- and high-nibble dot products.
            isum = madd_cols(isum, qlo0, qlo1, a_lo, scale_lo);
            isum = madd_cols(isum, qhi0, qhi1, a_hi, scale_hi);
        }
        // d[j] * act.d per column.
        let dvec = _mm256_setr_ps(
            blk.d[0].to_f32(),
            blk.d[1].to_f32(),
            blk.d[2].to_f32(),
            blk.d[3].to_f32(),
            blk.d[4].to_f32(),
            blk.d[5].to_f32(),
            blk.d[6].to_f32(),
            blk.d[7].to_f32(),
        );
        let dall = _mm256_mul_ps(dvec, _mm256_set1_ps(act.d));
        acc = _mm256_fmadd_ps(dall, _mm256_cvtepi32_ps(isum), acc);
        // mins contribution per column.
        let mut imin = _mm256_setzero_si256();
        for sb in 0..8 {
            let bsum = (act.bsums[sb * 2] as i32 + act.bsums[sb * 2 + 1] as i32) as f32;
            let m = 8 + sb; // column j's min for sub-block sb = sm[j][8+sb]
            let minv = _mm256_setr_epi32(
                sm[0][m] as i32,
                sm[1][m] as i32,
                sm[2][m] as i32,
                sm[3][m] as i32,
                sm[4][m] as i32,
                sm[5][m] as i32,
                sm[6][m] as i32,
                sm[7][m] as i32,
            );
            let bs = _mm256_cvtepi32_ps(minv);
            imin = _mm256_castps_si256(_mm256_fmadd_ps(
                bs,
                _mm256_set1_ps(bsum),
                _mm256_castsi256_ps(imin),
            ));
        }
        let dminv = _mm256_setr_ps(
            blk.dmin[0].to_f32(),
            blk.dmin[1].to_f32(),
            blk.dmin[2].to_f32(),
            blk.dmin[3].to_f32(),
            blk.dmin[4].to_f32(),
            blk.dmin[5].to_f32(),
            blk.dmin[6].to_f32(),
            blk.dmin[7].to_f32(),
        );
        let dminall = _mm256_mul_ps(dminv, _mm256_set1_ps(act.d));
        acc_min = _mm256_fmadd_ps(dminall, _mm256_castsi256_ps(imin), acc_min);
    }
    let res = _mm256_sub_ps(acc, acc_min);
    _mm256_storeu_ps(out.as_mut_ptr(), res);
}

/// Build the i16 scale operand for four consecutive columns starting at
/// `col0`, taking each column's 6-bit scale/min entry `idx` from the
/// unpacked table. A maddubs result holds four i16 partials per column, all
/// sharing that column's scale, so the scale is replicated across those four
/// lanes: multiplying by this operand with a pairwise-accumulating madd
/// scales, widens and reduces in one step, entirely within each lane.
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn scale_lanes_i16(sm: &[[u8; 16]; 8], col0: usize, idx: usize) -> __m256i {
    let s0 = sm[col0][idx] as i16;
    let s1 = sm[col0 + 1][idx] as i16;
    let s2 = sm[col0 + 2][idx] as i16;
    let s3 = sm[col0 + 3][idx] as i16;
    _mm256_setr_epi16(
        s0, s0, s0, s0, s1, s1, s1, s1, s2, s2, s2, s2, s3, s3, s3, s3,
    )
}

/// Load 8 activation bytes and broadcast the 8-element pattern to all 8
/// column lanes as i16 (for maddubs). Returns two __m256i (low/high halves
/// not needed - handled by caller via `madd_cols`).
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn load_a8_broadcast(p: *const i8) -> __m256i {
    // 8 signed activation bytes, replicated across the 32-byte register so
    // each column's 8-byte lane multiplies the same activation 8-vector.
    let lo = _mm_loadl_epi64(p as *const __m128i); // 8 bytes in low 64 bits
    let b64 = _mm_unpacklo_epi64(lo, lo); // duplicate -> 16 bytes
                                          // -> 32 bytes (4 copies of the 8-byte lane)
    _mm256_inserti128_si256(_mm256_castsi128_si256(b64), b64, 1)
}

/// Column-wise madd: weights are unsigned nibbles (0..15), activations are
/// signed int8. Compute per-8-byte-lane dot, scale per column, accumulate
/// into the i32 column accumulator. Each 32-byte register holds 4 columns
/// (8 bytes each); two registers (lo01, lo23-pair) cover 8 columns.
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn madd_cols(
    acc: __m256i,
    q_c0123: __m256i,
    q_c4567: __m256i,
    a8: __m256i,
    scale: __m256i,
) -> __m256i {
    // maddubs: u8(weight) * i8(act) -> i16 pairs, horizontally added in 2s.
    let p0 = _mm256_maddubs_epi16(q_c0123, a8); // cols 0-3, 4 i16 partials each
    let p1 = _mm256_maddubs_epi16(q_c4567, a8); // cols 4-7
                                                // Sum the 4 i16 partials per 8-byte lane down to one i32 per column.
    let ones = _mm256_set1_epi16(1);
    let s0 = _mm256_madd_epi16(p0, ones); // -> 2 i32 per 64-bit lane (cols 0-3 across 128-bit halves)
    let s1 = _mm256_madd_epi16(p1, ones);
    // s0 lanes: [c0a c0b c1a c1b | c2a c2b c3a c3b]; horizontally pair-add
    // to get one i32 per column: hadd within adjacent i32s.
    let c0123 = _mm256_hadd_epi32(s0, s0); // [c0 c1 c0 c1 | c2 c3 c2 c3]
    let c4567 = _mm256_hadd_epi32(s1, s1);
    // Gather columns into lane order [c0 c1 c2 c3 c4 c5 c6 c7].
    let lo = _mm256_castsi256_si128(c0123); // c0 c1 c0 c1
    let hi = _mm256_extracti128_si256(c0123, 1); // c2 c3 c2 c3
    let c0_3 = _mm_unpacklo_epi64(lo, hi); // c0 c1 c2 c3
    let lo2 = _mm256_castsi256_si128(c4567);
    let hi2 = _mm256_extracti128_si256(c4567, 1);
    let c4_7 = _mm_unpacklo_epi64(lo2, hi2); // c4 c5 c6 c7
    let cols = _mm256_inserti128_si256(_mm256_castsi128_si256(c0_3), c4_7, 1); // [c0..c3 | c4..c7]
    let scaled = _mm256_mullo_epi32(cols, scale);
    _mm256_add_epi32(acc, scaled)
}

/// Ported verbatim from llama.cpp `ggml_gemv_q4_K_8x8_q8_K` (AVX2,
/// ggml-cpu/arch/x86/repack.cpp): the M=1 decode kernel that dots ONE Q8_K
/// activation row against an 8-column interleaved `BlockQ4Kx8` group in a
/// single pass, batching TWO Q4_K sub-blocks per iteration (vs the per-16-elem
/// scale reload of `gemv_group_avx2`). llama.cpp REPACK runs this for MoE
/// decode (overrides MUL_MAT_ID) and it beats plain dot by ~40% at M=1.
/// Writes 8 column outputs. Bit-identical to `dot` (validated in the bench).
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,f16c")]
/// [`gemv_group_avx2_v2`] reading the PLAIN per-row scales layout written by
/// [`repack`] (each row's canonical 12-byte Q4_K packing) instead of the
/// llama-interleaved 6-bit packing of [`repack_v2`]. One storage layout can then
/// serve BOTH this M=1 GEMV and the tiled prefill GEMM - the enabler for
/// repacking weights IN PLACE at load instead of caching a second copy.
/// The dot math is byte-for-byte the packed variant's; only the scale/min
/// register assembly differs (scalar canonical extraction per row).
pub unsafe fn gemv_group_avx2_plain(b: &[BlockQ4Kx8], a: &[BlockQ8K], nb: usize, out: &mut [f32]) {
    let m4b = _mm256_set1_epi8(0x0F);
    let deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    let scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    let finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    let mut acc_row = _mm256_setzero_ps();
    let mut acc_min_rows = _mm256_setzero_ps();

    for bi in 0..nb {
        let blk = &b[bi];
        let act = &a[bi];
        let row_scale_f32 = _mm256_set1_ps(act.d);
        let col_scale_f32 = _mm256_cvtph_ps(_mm_shuffle_epi8(
            _mm_loadu_si128(blk.d.as_ptr() as *const __m128i),
            deltamask,
        ));
        let col_dmin_f32 = _mm256_cvtph_ps(_mm_loadu_si128(blk.dmin.as_ptr() as *const __m128i));

        let mut iacc_b = _mm256_setzero_si256();
        let mut iacc_min_b = _mm256_setzero_si256();

        // Per-block canonical unpack, ONCE: all 8 sub-blocks' (scale, min) for the
        // 8 rows into sb-indexed 16-byte tables (the per-call scalar extraction
        // inside the sub-block loop measured 1.3x slower than the packed path).
        let mut ms_all = [[0u8; 16]; 8];
        for j in 0..8 {
            let q = &blk.scales[j * 12..j * 12 + 12];
            for i in 0..4 {
                ms_all[i][j] = q[i] & 63;
                ms_all[i][8 + j] = q[i + 4] & 63;
            }
            for i in 4..8 {
                ms_all[i][j] = (q[i + 4] & 0x0F) | ((q[i - 4] >> 6) << 4);
                ms_all[i][8 + j] = (q[i + 4] >> 4) | ((q[i] >> 6) << 4);
            }
        }

        let q8sums = _mm256_loadu_si256(act.bsums.as_ptr() as *const __m256i);
        let mut q8s = _mm256_castsi128_si256(_mm_hadd_epi16(
            _mm256_castsi256_si128(q8sums),
            _mm256_extracti128_si256::<1>(q8sums),
        ));
        q8s = _mm256_permute2f128_si256::<0>(q8s, q8s);

        for sb in 0..(QK_K / 64) {
            let qs = blk.qs.as_ptr();
            let o = sb * 256;
            let raw0123_0 = _mm256_loadu_si256(qs.add(o) as *const __m256i);
            let raw4567_0 = _mm256_loadu_si256(qs.add(o + 32) as *const __m256i);
            let raw0123_1 = _mm256_loadu_si256(qs.add(o + 64) as *const __m256i);
            let raw4567_1 = _mm256_loadu_si256(qs.add(o + 96) as *const __m256i);
            let raw0123_2 = _mm256_loadu_si256(qs.add(o + 128) as *const __m256i);
            let raw4567_2 = _mm256_loadu_si256(qs.add(o + 160) as *const __m256i);
            let raw0123_3 = _mm256_loadu_si256(qs.add(o + 192) as *const __m256i);
            let raw4567_3 = _mm256_loadu_si256(qs.add(o + 224) as *const __m256i);

            let v0123_00 = _mm256_and_si256(raw0123_0, m4b);
            let v4567_00 = _mm256_and_si256(raw4567_0, m4b);
            let v0123_01 = _mm256_and_si256(raw0123_1, m4b);
            let v4567_01 = _mm256_and_si256(raw4567_1, m4b);
            let v0123_02 = _mm256_and_si256(raw0123_2, m4b);
            let v4567_02 = _mm256_and_si256(raw4567_2, m4b);
            let v0123_03 = _mm256_and_si256(raw0123_3, m4b);
            let v4567_03 = _mm256_and_si256(raw4567_3, m4b);

            let v0123_10 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw0123_0), m4b);
            let v4567_10 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw4567_0), m4b);
            let v0123_11 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw0123_1), m4b);
            let v4567_11 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw4567_1), m4b);
            let v0123_12 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw0123_2), m4b);
            let v4567_12 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw4567_2), m4b);
            let v0123_13 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw0123_3), m4b);
            let v4567_13 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw4567_3), m4b);

            // PLAIN-scales loader: read the block's hoisted per-sub-block tables.
            let mins_and_scales_0 = _mm_loadu_si128(ms_all[2 * sb].as_ptr() as *const __m128i);
            let scales_0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(mins_and_scales_0, scalemask));
            let mins_and_scales_1 = _mm_loadu_si128(ms_all[2 * sb + 1].as_ptr() as *const __m128i);
            let scales_1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(mins_and_scales_1, scalemask));
            let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                _mm_shuffle_epi32::<78>(mins_and_scales_0),
                _mm_shuffle_epi32::<78>(mins_and_scales_1),
            ));

            let lhs = act.qs.as_ptr();
            let mut l00 =
                _mm256_castsi128_si256(_mm_loadu_si128(lhs.add(sb * 64) as *const __m128i));
            let mut l01 =
                _mm256_castsi128_si256(_mm_loadu_si128(lhs.add(16 + sb * 64) as *const __m128i));
            let mut l10 =
                _mm256_castsi128_si256(_mm_loadu_si128(lhs.add(32 + sb * 64) as *const __m128i));
            let mut l11 =
                _mm256_castsi128_si256(_mm_loadu_si128(lhs.add(48 + sb * 64) as *const __m128i));
            l00 = _mm256_permute2f128_si256::<0>(l00, l00);
            l01 = _mm256_permute2f128_si256::<0>(l01, l01);
            l10 = _mm256_permute2f128_si256::<0>(l10, l10);
            l11 = _mm256_permute2f128_si256::<0>(l11, l11);

            let mut iacc_0 = _mm256_setzero_si256();
            let mut iacc_1 = _mm256_setzero_si256();

            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_00, _mm256_shuffle_epi32::<177>(v4567_00)),
                    _mm256_shuffle_epi32::<0>(l00),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_00), v4567_00),
                    _mm256_shuffle_epi32::<85>(l00),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_01, _mm256_shuffle_epi32::<177>(v4567_01)),
                    _mm256_shuffle_epi32::<170>(l00),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_01), v4567_01),
                    _mm256_shuffle_epi32::<255>(l00),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_02, _mm256_shuffle_epi32::<177>(v4567_02)),
                    _mm256_shuffle_epi32::<0>(l01),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_02), v4567_02),
                    _mm256_shuffle_epi32::<85>(l01),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_03, _mm256_shuffle_epi32::<177>(v4567_03)),
                    _mm256_shuffle_epi32::<170>(l01),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_03), v4567_03),
                    _mm256_shuffle_epi32::<255>(l01),
                ),
            );
            iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_10, _mm256_shuffle_epi32::<177>(v4567_10)),
                    _mm256_shuffle_epi32::<0>(l10),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_10), v4567_10),
                    _mm256_shuffle_epi32::<85>(l10),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_11, _mm256_shuffle_epi32::<177>(v4567_11)),
                    _mm256_shuffle_epi32::<170>(l10),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_11), v4567_11),
                    _mm256_shuffle_epi32::<255>(l10),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_12, _mm256_shuffle_epi32::<177>(v4567_12)),
                    _mm256_shuffle_epi32::<0>(l11),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_12), v4567_12),
                    _mm256_shuffle_epi32::<85>(l11),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_13, _mm256_shuffle_epi32::<177>(v4567_13)),
                    _mm256_shuffle_epi32::<170>(l11),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_13), v4567_13),
                    _mm256_shuffle_epi32::<255>(l11),
                ),
            );
            iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

            let iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);
            let q8s_sb = _mm256_shuffle_epi32::<0>(q8s);
            let iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
            q8s = _mm256_bsrli_epi128::<4>(q8s);
            iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
            iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
        }
        acc_row = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_b),
            _mm256_mul_ps(col_scale_f32, row_scale_f32),
            acc_row,
        );
        acc_min_rows = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_min_b),
            _mm256_mul_ps(col_dmin_f32, row_scale_f32),
            acc_min_rows,
        );
    }
    acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
    _mm256_storeu_ps(out.as_mut_ptr(), _mm256_sub_ps(acc_row, acc_min_rows));
}

pub unsafe fn gemv_group_avx2_v2(b: &[BlockQ4Kx8], a: &[BlockQ8K], nb: usize, out: &mut [f32]) {
    let m4b = _mm256_set1_epi8(0x0F);
    let deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    let scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    let finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    let mut acc_row = _mm256_setzero_ps();
    let mut acc_min_rows = _mm256_setzero_ps();

    for bi in 0..nb {
        let blk = &b[bi];
        let act = &a[bi];
        let row_scale_f32 = _mm256_set1_ps(act.d);
        let col_scale_f32 = _mm256_cvtph_ps(_mm_shuffle_epi8(
            _mm_loadu_si128(blk.d.as_ptr() as *const __m128i),
            deltamask,
        ));
        let col_dmin_f32 = _mm256_cvtph_ps(_mm_loadu_si128(blk.dmin.as_ptr() as *const __m128i));

        let mut iacc_b = _mm256_setzero_si256();
        let mut iacc_min_b = _mm256_setzero_si256();

        let q8sums = _mm256_loadu_si256(act.bsums.as_ptr() as *const __m256i);
        let mut q8s = _mm256_castsi128_si256(_mm_hadd_epi16(
            _mm256_castsi256_si128(q8sums),
            _mm256_extracti128_si256::<1>(q8sums),
        ));
        q8s = _mm256_permute2f128_si256::<0>(q8s, q8s);

        for sb in 0..(QK_K / 64) {
            let qs = blk.qs.as_ptr();
            let o = sb * 256;
            let raw0123_0 = _mm256_loadu_si256(qs.add(o) as *const __m256i);
            let raw4567_0 = _mm256_loadu_si256(qs.add(o + 32) as *const __m256i);
            let raw0123_1 = _mm256_loadu_si256(qs.add(o + 64) as *const __m256i);
            let raw4567_1 = _mm256_loadu_si256(qs.add(o + 96) as *const __m256i);
            let raw0123_2 = _mm256_loadu_si256(qs.add(o + 128) as *const __m256i);
            let raw4567_2 = _mm256_loadu_si256(qs.add(o + 160) as *const __m256i);
            let raw0123_3 = _mm256_loadu_si256(qs.add(o + 192) as *const __m256i);
            let raw4567_3 = _mm256_loadu_si256(qs.add(o + 224) as *const __m256i);

            let v0123_00 = _mm256_and_si256(raw0123_0, m4b);
            let v4567_00 = _mm256_and_si256(raw4567_0, m4b);
            let v0123_01 = _mm256_and_si256(raw0123_1, m4b);
            let v4567_01 = _mm256_and_si256(raw4567_1, m4b);
            let v0123_02 = _mm256_and_si256(raw0123_2, m4b);
            let v4567_02 = _mm256_and_si256(raw4567_2, m4b);
            let v0123_03 = _mm256_and_si256(raw0123_3, m4b);
            let v4567_03 = _mm256_and_si256(raw4567_3, m4b);

            let v0123_10 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw0123_0), m4b);
            let v4567_10 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw4567_0), m4b);
            let v0123_11 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw0123_1), m4b);
            let v4567_11 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw4567_1), m4b);
            let v0123_12 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw0123_2), m4b);
            let v4567_12 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw4567_2), m4b);
            let v0123_13 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw0123_3), m4b);
            let v4567_13 = _mm256_and_si256(_mm256_srli_epi16::<4>(raw4567_3), m4b);

            // Sub-blocks `2*sb` and `2*sb+1`, one twelve-byte group each. The
            // re-packed group carries the same six-bit split, over the eight
            // rows instead of the eight sub-blocks: rows 0..4 whole, rows 4..8
            // assembled. So the words come out rows 0..4 scales, rows 4..8
            // scales, then the two halves of the minimums - the order
            // `_mm_set_epi32` reads back below.
            let utmp_0 = sub_scales_and_mins(&blk.scales[24 * sb..24 * sb + 12]);
            let utmp_1 = sub_scales_and_mins(&blk.scales[24 * sb + 12..24 * sb + 24]);

            let mins_and_scales_0 = _mm_set_epi32(
                utmp_0[3] as i32,
                utmp_0[2] as i32,
                utmp_0[1] as i32,
                utmp_0[0] as i32,
            );
            let scales_0 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(mins_and_scales_0, scalemask));
            let mins_and_scales_1 = _mm_set_epi32(
                utmp_1[3] as i32,
                utmp_1[2] as i32,
                utmp_1[1] as i32,
                utmp_1[0] as i32,
            );
            let scales_1 = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(mins_and_scales_1, scalemask));
            let mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(
                _mm_shuffle_epi32::<78>(mins_and_scales_0),
                _mm_shuffle_epi32::<78>(mins_and_scales_1),
            ));

            let lhs = act.qs.as_ptr();
            let mut l00 =
                _mm256_castsi128_si256(_mm_loadu_si128(lhs.add(sb * 64) as *const __m128i));
            let mut l01 =
                _mm256_castsi128_si256(_mm_loadu_si128(lhs.add(16 + sb * 64) as *const __m128i));
            let mut l10 =
                _mm256_castsi128_si256(_mm_loadu_si128(lhs.add(32 + sb * 64) as *const __m128i));
            let mut l11 =
                _mm256_castsi128_si256(_mm_loadu_si128(lhs.add(48 + sb * 64) as *const __m128i));
            l00 = _mm256_permute2f128_si256::<0>(l00, l00);
            l01 = _mm256_permute2f128_si256::<0>(l01, l01);
            l10 = _mm256_permute2f128_si256::<0>(l10, l10);
            l11 = _mm256_permute2f128_si256::<0>(l11, l11);

            let mut iacc_0 = _mm256_setzero_si256();
            let mut iacc_1 = _mm256_setzero_si256();

            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_00, _mm256_shuffle_epi32::<177>(v4567_00)),
                    _mm256_shuffle_epi32::<0>(l00),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_00), v4567_00),
                    _mm256_shuffle_epi32::<85>(l00),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_01, _mm256_shuffle_epi32::<177>(v4567_01)),
                    _mm256_shuffle_epi32::<170>(l00),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_01), v4567_01),
                    _mm256_shuffle_epi32::<255>(l00),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_02, _mm256_shuffle_epi32::<177>(v4567_02)),
                    _mm256_shuffle_epi32::<0>(l01),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_02), v4567_02),
                    _mm256_shuffle_epi32::<85>(l01),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_03, _mm256_shuffle_epi32::<177>(v4567_03)),
                    _mm256_shuffle_epi32::<170>(l01),
                ),
            );
            iacc_0 = _mm256_add_epi16(
                iacc_0,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_03), v4567_03),
                    _mm256_shuffle_epi32::<255>(l01),
                ),
            );
            iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_10, _mm256_shuffle_epi32::<177>(v4567_10)),
                    _mm256_shuffle_epi32::<0>(l10),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_10), v4567_10),
                    _mm256_shuffle_epi32::<85>(l10),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_11, _mm256_shuffle_epi32::<177>(v4567_11)),
                    _mm256_shuffle_epi32::<170>(l10),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_11), v4567_11),
                    _mm256_shuffle_epi32::<255>(l10),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_12, _mm256_shuffle_epi32::<177>(v4567_12)),
                    _mm256_shuffle_epi32::<0>(l11),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_12), v4567_12),
                    _mm256_shuffle_epi32::<85>(l11),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(v0123_13, _mm256_shuffle_epi32::<177>(v4567_13)),
                    _mm256_shuffle_epi32::<170>(l11),
                ),
            );
            iacc_1 = _mm256_add_epi16(
                iacc_1,
                _mm256_maddubs_epi16(
                    _mm256_blend_epi32::<170>(_mm256_shuffle_epi32::<177>(v0123_13), v4567_13),
                    _mm256_shuffle_epi32::<255>(l11),
                ),
            );
            iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

            let iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);
            let q8s_sb = _mm256_shuffle_epi32::<0>(q8s);
            let iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
            q8s = _mm256_bsrli_epi128::<4>(q8s);
            iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
            iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
        }
        acc_row = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_b),
            _mm256_mul_ps(col_scale_f32, row_scale_f32),
            acc_row,
        );
        acc_min_rows = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(iacc_min_b),
            _mm256_mul_ps(col_dmin_f32, row_scale_f32),
            acc_min_rows,
        );
    }
    acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
    _mm256_storeu_ps(out.as_mut_ptr(), _mm256_sub_ps(acc_row, acc_min_rows));
}
