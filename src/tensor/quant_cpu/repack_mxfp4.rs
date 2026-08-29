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

use super::{BlockMxFp4, BlockQ8_0, KVALUES_MXFP4};
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::*;

/// Transpose `[n, nb]` row-major `BlockMxFp4` into `[n/8][nb][8]` (group,
/// block, row). `n % 8 == 0`; callers fall back otherwise.
pub fn repack(rhs: &[BlockMxFp4], n: usize, nb: usize) -> Vec<BlockMxFp4> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(rhs.len(), n * nb);
    let groups = n / 8;
    let zero = BlockMxFp4 {
        e: 0,
        qs: [0u8; super::QK_MXFP4 / 2],
    };
    let mut out = vec![zero; groups * nb * 8];
    for g in 0..groups {
        for l in 0..nb {
            for j in 0..8 {
                out[(g * nb + l) * 8 + j] = rhs[(g * 8 + j) * nb + l].clone();
            }
        }
    }
    out
}

/// Scalar reference: one 8-column group's `nb*8` blocks (block-major, then
/// row) x the row's `nb` Q8_0 activation blocks. Writes `out[0..8]`. Mirrors
/// `vec_dot_mxfp4_q8_0`: integer code.activation summed per 32-block, scaled
/// by `e8m0_half(e).d_act`.
pub fn gemv_group_scalar(b: &[BlockMxFp4], a: &[BlockQ8_0], nb: usize, out: &mut [f32]) {
    let mut acc = [0f32; 8];
    for l in 0..nb {
        let act = &a[l];
        let dact = act.d.to_f32();
        for j in 0..8 {
            let blk = &b[l * 8 + j];
            let mut sumi = 0i32;
            for i in 0..16 {
                let q = blk.qs[i];
                let v0 = KVALUES_MXFP4[(q & 0x0F) as usize] as i32;
                let v1 = KVALUES_MXFP4[(q >> 4) as usize] as i32;
                sumi += v0 * act.qs[i] as i32 + v1 * act.qs[i + 16] as i32;
            }
            acc[j] += sumi as f32 * super::e8m0_to_fp32_half(blk.e) * dact;
        }
    }
    out[..8].copy_from_slice(&acc);
}

/// AVX2 GEMV over one 8-column group: hoists the activation load out of the
/// 8-row inner loop, 8 independent accumulator chains. Same per-block ops as
/// `vec_dot_mxfp4_q8_0`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_group_avx2(b: &[BlockMxFp4], a: &[BlockQ8_0], nb: usize, out: &mut [f32]) {
    let lut =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i));
    let mut acc = [_mm256_setzero_ps(); 8];
    for l in 0..nb {
        const PF: usize = 4;
        if l + PF < nb {
            super::gemv_prefetch_t0::<2>(b.as_ptr().add((l + PF) * 8) as *const i8);
        }
        let act = &a[l];
        let dact = act.d.to_f32();
        let by = _mm256_loadu_si256(act.qs.as_ptr() as *const __m256i);
        for j in 0..8 {
            let blk = &b[l * 8 + j];
            let d = _mm256_set1_ps(super::e8m0_to_fp32_half(blk.e) * dact);
            let bx = _mm256_shuffle_epi8(lut, super::avx::split_nibbles(blk.qs.as_ptr()));
            let q = super::avx::dot_signed(bx, by);
            acc[j] = _mm256_fmadd_ps(d, q, acc[j]);
        }
    }
    for j in 0..8 {
        out[j] = super::avx::lanes_sum(acc[j]);
    }
}

/// Row-tiled GEMM: one 8-column group x `mt` activation rows (`mt <= 8`).
/// Columns are processed one at a time so each column's `nb` weight blocks
/// decode once and meet every row of the tile. Mirrors `vec_dot_mxfp4_q8_0`'s
/// reduction order exactly (code->LUT shuffle -> pairwise int8 dot -> one fmadd
/// of `e8m0_half(e).d_act` per 32-block into an f32 vector -> `hsum` at the
/// end), so it is bit-identical to the per-column path. Writes
/// `out[r*8 .. r*8+8]` for r in 0..mt.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group_avx2(b: &[BlockMxFp4], acts: &[&[BlockQ8_0]], nb: usize, out: &mut [f32]) {
    let mt = acts.len();
    let lut =
        _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_MXFP4.as_ptr() as *const __m128i));
    for j in 0..8 {
        let mut acc = [_mm256_setzero_ps(); 8];
        for l in 0..nb {
            let blk = &b[l * 8 + j];
            let bx = _mm256_shuffle_epi8(lut, super::avx::split_nibbles(blk.qs.as_ptr()));
            let dj = super::e8m0_to_fp32_half(blk.e);
            for (r, act) in acts.iter().enumerate().take(mt) {
                let a = &act[l];
                let by = _mm256_loadu_si256(a.qs.as_ptr() as *const __m256i);
                let d = _mm256_set1_ps(dj * a.d.to_f32());
                acc[r] = _mm256_fmadd_ps(d, super::avx::dot_signed(bx, by), acc[r]);
            }
        }
        for (r, a) in acc.iter().enumerate().take(mt) {
            out[r * 8 + j] = super::avx::lanes_sum(*a);
        }
    }
}
