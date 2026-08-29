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

use super::{BlockQ4_0, BlockQ8_0};
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
use core::arch::x86_64::*;
use half::f16;

/// Transpose `[n, nb]` row-major `BlockQ4_0` into `[n/8][nb][8]` (group,
/// block, row). `n % 8 == 0`; callers fall back otherwise.
pub fn repack(rhs: &[BlockQ4_0], n: usize, nb: usize) -> Vec<BlockQ4_0> {
    debug_assert_eq!(n % 8, 0);
    debug_assert_eq!(rhs.len(), n * nb);
    let groups = n / 8;
    let zero = BlockQ4_0 {
        d: f16::ZERO,
        qs: [0u8; 16],
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
/// row) x the row's `nb` Q8_0 activation blocks. Writes `out[0..8]`.
pub fn gemv_group_scalar(b: &[BlockQ4_0], a: &[BlockQ8_0], nb: usize, out: &mut [f32]) {
    let mut acc = [0f32; 8];
    for l in 0..nb {
        let act = &a[l];
        let dact = act.d.to_f32();
        for j in 0..8 {
            let blk = &b[l * 8 + j];
            let mut sumi = 0i32;
            for i in 0..16 {
                let q = blk.qs[i];
                let v0 = (q & 0x0F) as i32 - 8;
                let v1 = (q >> 4) as i32 - 8;
                sumi += v0 * act.qs[i] as i32 + v1 * act.qs[i + 16] as i32;
            }
            acc[j] += sumi as f32 * blk.d.to_f32() * dact;
        }
    }
    out[..8].copy_from_slice(&acc);
}

/// AVX2: identical dot to `vec_dot_q4_0_q8_0` but hoists the activation load
/// out of the 8-row inner loop and keeps 8 independent accumulator chains.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_group_avx2(b: &[BlockQ4_0], a: &[BlockQ8_0], nb: usize, out: &mut [f32]) {
    let off = _mm256_set1_epi8(8);
    let mut acc = [_mm256_setzero_ps(); 8];
    for l in 0..nb {
        // SW-prefetch the next Q4_0 weight group ahead of the cold RAM stream (shared
        // primitive; without it decode sat at 93% of peak BW vs ollama's 100%).
        const PF: usize = 4;
        if l + PF < nb {
            super::gemv_prefetch_t0::<2>(b.as_ptr().add((l + PF) * 8) as *const i8);
        }
        let act = &a[l];
        let dact = act.d.to_f32();
        let by = _mm256_loadu_si256(act.qs.as_ptr() as *const __m256i);
        for j in 0..8 {
            let blk = &b[l * 8 + j];
            let d = _mm256_set1_ps(blk.d.to_f32() * dact);
            let bx = _mm256_sub_epi8(super::avx::split_nibbles(blk.qs.as_ptr()), off);
            let q = super::avx::dot_signed(bx, by);
            acc[j] = _mm256_fmadd_ps(d, q, acc[j]);
        }
    }
    for j in 0..8 {
        out[j] = super::avx::lanes_sum(acc[j]);
    }
}

/// Row-tiled GEMM: one 8-column group x `mt` activation rows. Columns are
/// processed one at a time so each column's `nb` weight blocks decode once
/// and then meet every row of the tile - the reuse that makes prefill
/// compute-bound. Mirrors `vec_dot_q4_0_q8_0`'s reduction order exactly (the
/// 4-bit weight is the nibble minus the fixed 8 zero point, fed to the
/// pairwise integer dot, then one fmadd of `x.d*y.d` per 32-block into an f32
/// vector, `hsum` at the end), so the result is bit-identical to the
/// per-column path. Writes `out[r*8 .. r*8+8]` for r in 0..mt. `mt <= 8`.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemm_group_avx2(b: &[BlockQ4_0], acts: &[&[BlockQ8_0]], nb: usize, out: &mut [f32]) {
    let mt = acts.len();
    let off = _mm256_set1_epi8(8);
    for j in 0..8 {
        let mut acc = [_mm256_setzero_ps(); 8];
        for l in 0..nb {
            let blk = &b[l * 8 + j];
            // v = nibble - 8 (Q4_0 zero point), no high plane.
            let bx = _mm256_sub_epi8(super::avx::split_nibbles(blk.qs.as_ptr()), off);
            let dj = blk.d.to_f32();
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
