//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

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

use super::*;

/// Prefill-tiled Q5_0 GEMM support. Q5_0 is a legacy 32-weight block (4-bit
/// `qs` nibbles + a 1-bit high plane in `qh` + one scale, zero point -16). The
/// per-column path re-decodes and re-reads every weight block once per prompt
/// row; grouping 8 columns and reusing each decoded block across a tile of rows
/// is the weight reuse that makes prefill compute-bound. The packed planes are
/// kept unexpanded - the group is only as large as eight raw blocks.
pub mod repack_q5_0 {
    use super::{BlockQ5_0, BlockQ8_0};
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    use core::arch::x86_64::*;
    use half::f16;

    /// Transpose `[n, nb]` row-major `BlockQ5_0` into `[n/8][nb][8]` (group,
    /// block, row). `n % 8 == 0`; callers fall back otherwise. A gather - the
    /// packed blocks are copied verbatim, so the values are unchanged.
    pub fn repack(rhs: &[BlockQ5_0], n: usize, nb: usize) -> Vec<BlockQ5_0> {
        debug_assert_eq!(n % 8, 0);
        debug_assert_eq!(rhs.len(), n * nb);
        let groups = n / 8;
        let zero = BlockQ5_0 {
            d: f16::ZERO,
            qh: [0u8; 4],
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

    /// Scalar reference GEMV: one 8-column group (`nb*8` blocks, block-major then
    /// row) x one row's `nb` Q8_0 blocks. THE ORACLE - integer accumulation is
    /// exact, so the tiled AVX2 kernel is validated against it and it must not be
    /// changed. Writes `out[0..8]`. The 5-bit weight is the nibble OR the high
    /// bit shifted into place, minus the fixed 16 zero point.
    pub fn gemv_group_scalar(b: &[BlockQ5_0], a: &[BlockQ8_0], nb: usize, out: &mut [f32]) {
        let mut acc = [0f32; 8];
        for l in 0..nb {
            let act = &a[l];
            let dact = act.d.to_f32();
            for j in 0..8 {
                let blk = &b[l * 8 + j];
                let qh = u32::from_le_bytes(blk.qh);
                let mut sumi = 0i32;
                for i in 0..16 {
                    let xh0 = (((qh >> i) & 1) << 4) as i32;
                    let xh1 = (((qh >> (i + 16)) & 1) << 4) as i32;
                    let v0 = ((blk.qs[i] & 0x0F) as i32 | xh0) - 16;
                    let v1 = ((blk.qs[i] >> 4) as i32 | xh1) - 16;
                    sumi += v0 * act.qs[i] as i32 + v1 * act.qs[i + 16] as i32;
                }
                acc[j] += sumi as f32 * blk.d.to_f32() * dact;
            }
        }
        out[..8].copy_from_slice(&acc);
    }

    /// Row-tiled GEMM: one 8-column group x `mt` activation rows. Columns are
    /// processed one at a time so each column's `nb` weight blocks decode once
    /// and then meet every row of the tile - the reuse that makes prefill
    /// compute-bound. The decode is the dot's: the nibbles, plus the high
    /// plane folded in as the sign of the byte, fed straight to the pairwise
    /// integer dot. Writes `out[r*8 .. r*8+8]` for r in 0..mt. `mt <= 8`.
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemm_group_avx2(
        b: &[BlockQ5_0],
        acts: &[&[BlockQ8_0]],
        nb: usize,
        out: &mut [f32],
    ) {
        let mt = acts.len();
        let himask = _mm256_set1_epi8(0xF0u8 as i8);
        for j in 0..8 {
            let mut acc = [_mm256_setzero_ps(); 8];
            for l in 0..nb {
                let blk = &b[l * 8 + j];
                // v = (nibble | bit<<4) - 16 equals, as a signed byte, the
                // nibble OR 0xF0 wherever the high bit is CLEAR.
                let bx = _mm256_or_si256(
                    super::avx::split_nibbles(blk.qs.as_ptr()),
                    _mm256_andnot_si256(super::avx::mask_from_bits(&blk.qh), himask),
                );
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
}

/// Prefill-tiled Q8_0 GEMM support. Q8_0 is a 32-weight block: 32 signed `i8`
/// weights plus one f16 scale `d`, no zero point and no packing. It has no
/// tiled repack otherwise, so the per-column path re-reads every weight block
/// once per prompt row; grouping 8 columns and reusing each block across a tile
/// of rows is the weight reuse that makes prefill compute-bound. The blocks are
/// kept verbatim - no dequant, the `i8` weights feed the pairwise integer dot
/// directly.
pub mod repack_q8_0 {
    use super::{BlockQ8_0, QK8_0};
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    use core::arch::x86_64::*;
    use half::f16;

    /// Transpose `[n, nb]` row-major `BlockQ8_0` into `[n/8][nb][8]` (group,
    /// block, row). `n % 8 == 0`; callers fall back otherwise. A gather - the
    /// blocks are copied verbatim, so the values are unchanged.
    pub fn repack(rhs: &[BlockQ8_0], n: usize, nb: usize) -> Vec<BlockQ8_0> {
        debug_assert_eq!(n % 8, 0);
        debug_assert_eq!(rhs.len(), n * nb);
        let groups = n / 8;
        let zero = BlockQ8_0 {
            d: f16::ZERO,
            qs: [0i8; QK8_0],
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

    /// Scalar reference GEMV: one 8-column group (`nb*8` blocks, block-major then
    /// row) x one row's `nb` Q8_0 blocks. THE ORACLE - integer accumulation is
    /// exact, so the tiled AVX2 kernel is validated against it and it must not be
    /// changed. Writes `out[0..8]`. The weight is the raw signed `i8`.
    pub fn gemv_group_scalar(b: &[BlockQ8_0], a: &[BlockQ8_0], nb: usize, out: &mut [f32]) {
        let mut acc = [0f32; 8];
        for l in 0..nb {
            let act = &a[l];
            let dact = act.d.to_f32();
            for j in 0..8 {
                let blk = &b[l * 8 + j];
                let mut sumi = 0i32;
                for i in 0..QK8_0 {
                    sumi += blk.qs[i] as i32 * act.qs[i] as i32;
                }
                acc[j] += sumi as f32 * blk.d.to_f32() * dact;
            }
        }
        out[..8].copy_from_slice(&acc);
    }

    /// Row-tiled GEMM: one 8-column group x `mt` activation rows. Columns are
    /// processed one at a time so each column's `nb` weight blocks are read once
    /// and then meet every row of the tile - the reuse that makes prefill
    /// compute-bound. Mirrors `vec_dot_q8_0_q8_0`'s reduction order exactly (the
    /// integer i8.i8 dot lands in an f32 vector via one fmadd of `x.d*y.d` per
    /// 32-block, `hsum` at the end), so the result is bit-identical to the
    /// per-column path. Writes `out[r*8 .. r*8+8]` for r in 0..mt. `mt <= 8`.
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemm_group_avx2(
        b: &[BlockQ8_0],
        acts: &[&[BlockQ8_0]],
        nb: usize,
        out: &mut [f32],
    ) {
        let mt = acts.len();
        for j in 0..8 {
            let mut acc = [_mm256_setzero_ps(); 8];
            for l in 0..nb {
                let blk = &b[l * 8 + j];
                let bx = _mm256_loadu_si256(blk.qs.as_ptr() as *const __m256i);
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
}

#[cfg(test)]
#[test]
pub(super) fn repacked_q4_0_gemv_matches_vec_dot() {
    // Deterministic pseudo-random Q4_0 weights [n=16, nb=4] + Q8_0 activation;
    // the repacked group GEMV must match the per-row dot within fp noise.
    let (n, nb) = (16usize, 4usize);
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut rows = Vec::with_capacity(n * nb);
    for _ in 0..n * nb {
        let mut qs = [0u8; 16];
        for q in qs.iter_mut() {
            *q = (rng() & 0xFF) as u8;
        }
        rows.push(BlockQ4_0 {
            d: f16::from_f32((rng() % 1000) as f32 / 500.0 - 1.0),
            qs,
        });
    }
    let mut act = Vec::with_capacity(nb);
    for _ in 0..nb {
        let mut qs = [0i8; 32];
        for q in qs.iter_mut() {
            *q = ((rng() % 255) as i32 - 127) as i8;
        }
        act.push(BlockQ8_0 {
            d: f16::from_f32((rng() % 1000) as f32 / 500.0),
            qs,
        });
    }
    // reference: per-row dot
    let mut want = vec![0f32; n];
    for (r, w) in want.iter_mut().enumerate() {
        *w = BlockQ4_0::dot(&rows[r * nb..r * nb + nb], &act);
    }
    // repacked group GEMV; check the AVX2 kernel AND the scalar reference.
    let packed = repack_q4_0::repack(&rows, n, nb);
    let mut got = vec![0f32; n];
    let mut got_scalar = vec![0f32; n];
    for g in 0..n / 8 {
        let bgrp = &packed[g * nb * 8..(g + 1) * nb * 8];
        let mut local = [0f32; 8];
        repack_q4_0::gemv_group_scalar(bgrp, &act, nb, &mut local);
        got_scalar[g * 8..g * 8 + 8].copy_from_slice(&local);
        #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
        unsafe {
            repack_q4_0::gemv_group_avx2(bgrp, &act, nb, &mut local);
        }
        got[g * 8..g * 8 + 8].copy_from_slice(&local);
    }
    for r in 0..n {
        let tol = 1e-2 + 1e-3 * want[r].abs();
        assert!(
            (want[r] - got_scalar[r]).abs() <= tol,
            "scalar row {r}: want {} got {}",
            want[r],
            got_scalar[r]
        );
        assert!(
            (want[r] - got[r]).abs() <= tol,
            "row {r}: want {} got {}",
            want[r],
            got[r]
        );
    }
}

#[cfg(test)]
#[cfg(target_feature = "avx2")]
#[test]
pub(super) fn repacked_q5_0_matches_vec_dot_and_oracle() {
    let (n, nb) = (16usize, 4usize);
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut rows = Vec::with_capacity(n * nb);
    for _ in 0..n * nb {
        let mut qs = [0u8; 16];
        for q in qs.iter_mut() {
            *q = (rng() & 0xFF) as u8;
        }
        let mut qh = [0u8; 4];
        for h in qh.iter_mut() {
            *h = (rng() & 0xFF) as u8;
        }
        rows.push(BlockQ5_0 {
            d: f16::from_f32((rng() % 1000) as f32 / 500.0 - 1.0),
            qh,
            qs,
        });
    }
    let m = 8usize; // exercise the full tile height
    let mut acts = Vec::with_capacity(m * nb);
    for _ in 0..m * nb {
        let mut qs = [0i8; 32];
        for q in qs.iter_mut() {
            *q = ((rng() % 255) as i32 - 127) as i8;
        }
        acts.push(BlockQ8_0 {
            d: f16::from_f32((rng() % 1000) as f32 / 500.0),
            qs,
        });
    }
    let packed = repack_q5_0::repack(&rows, n, nb);
    // Oracle vs production per-column dot (row 0).
    let a0 = &acts[0..nb];
    let mut want = vec![0f32; n];
    for (c, w) in want.iter_mut().enumerate() {
        *w = avx::vec_dot_q5_0_q8_0(&rows[c * nb..c * nb + nb], a0);
    }
    for g in 0..n / 8 {
        let bgrp = &packed[g * nb * 8..(g + 1) * nb * 8];
        let mut got = [0f32; 8];
        repack_q5_0::gemv_group_scalar(bgrp, a0, nb, &mut got);
        for c in 0..8 {
            let r = want[g * 8 + c];
            let rel = (r - got[c]).abs() / r.abs().max(1e-3);
            assert!(
                rel < 1e-4,
                "oracle col {}: dot {r} vs {} rel {rel}",
                g * 8 + c,
                got[c]
            );
        }
    }
    // Tiled AVX2 gemm vs oracle for every tile height.
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    for g in 0..n / 8 {
        let bgrp = &packed[g * nb * 8..(g + 1) * nb * 8];
        for mt in 1..=m {
            let refs: Vec<&[BlockQ8_0]> = (0..mt).map(|t| &acts[t * nb..(t + 1) * nb]).collect();
            let mut want_t = vec![0f32; mt * 8];
            for (t, a) in refs.iter().enumerate() {
                repack_q5_0::gemv_group_scalar(bgrp, a, nb, &mut want_t[t * 8..t * 8 + 8]);
            }
            let mut got = [0f32; 64];
            unsafe {
                repack_q5_0::gemm_group_avx2(bgrp, &refs, nb, &mut got);
            }
            for (i, w) in want_t.iter().enumerate() {
                let rel = (got[i] - w).abs() / w.abs().max(1e-3);
                assert!(rel < 1e-4, "gemm mt={mt} lane {i}: got {} want {w}", got[i]);
            }
        }
    }
}

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[test]
pub(super) fn q5_0_full_matmul_matches_reference() {
    // Full driver (matmul_q5_0_repacked_tiled) vs the per-column reference
    // (matmul_bytes -> vec_dot_q5_0_q8_0) at a realistic projection shape. The
    // tiled kernel mirrors the reference's reduction order (per 32-block
    // fmadd of `x.d*y.d` into an f32 vector, hsum at the end), so the result
    // is bit-identical and greedy tokens stay unchanged.
    let (m, k, n) = (13usize, 2048usize, 512usize);
    let nb = k / 32;
    let mut seed = 0x243F6A8885A308D3u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let wf: Vec<f32> = (0..n * k).map(|_| rng() * 4.0).collect();
    let mut wq = vec![
        BlockQ5_0 {
            d: f16::ZERO,
            qh: [0u8; 4],
            qs: [0u8; 16]
        };
        n * nb
    ];
    for row in 0..n {
        BlockQ5_0::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let wbytes: &[u8] = unsafe {
        std::slice::from_raw_parts(wq.as_ptr() as *const u8, std::mem::size_of_val(&wq[..]))
    };
    let lhs: Vec<f32> = (0..m * k).map(|_| rng() * 3.0).collect();

    let mut want = vec![0f32; m * n];
    matmul_bytes(GgmlDType::Q5_0, (m, k, n), &lhs, wbytes, &mut want).unwrap();

    let g8 = repack_q5_0::repack(&wq, n, nb);
    let mut got = vec![0f32; m * n];
    matmul_q5_0_repacked_tiled((m, k, n), &lhs, &g8, &mut got).unwrap();

    let mut maxabs = 0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    assert_eq!(
        maxabs, 0.0,
        "full q5_0 matmul must match reference bit-for-bit"
    );
}

#[cfg(test)]
#[cfg(target_feature = "avx2")]
#[test]
pub(super) fn repacked_q8_0_matches_vec_dot_and_oracle() {
    let (n, nb) = (16usize, 4usize);
    let mut seed = 0x14057B7EF767814Fu64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut rows = Vec::with_capacity(n * nb);
    for _ in 0..n * nb {
        let mut qs = [0i8; 32];
        for q in qs.iter_mut() {
            *q = ((rng() % 255) as i32 - 127) as i8;
        }
        rows.push(BlockQ8_0 {
            d: f16::from_f32((rng() % 1000) as f32 / 500.0 - 1.0),
            qs,
        });
    }
    let m = 8usize; // exercise the full tile height
    let mut acts = Vec::with_capacity(m * nb);
    for _ in 0..m * nb {
        let mut qs = [0i8; 32];
        for q in qs.iter_mut() {
            *q = ((rng() % 255) as i32 - 127) as i8;
        }
        acts.push(BlockQ8_0 {
            d: f16::from_f32((rng() % 1000) as f32 / 500.0),
            qs,
        });
    }
    let packed = repack_q8_0::repack(&rows, n, nb);
    // Oracle vs production per-column dot (row 0).
    let a0 = &acts[0..nb];
    let mut want = vec![0f32; n];
    for (c, w) in want.iter_mut().enumerate() {
        *w = avx::vec_dot_q8_0_q8_0(&rows[c * nb..c * nb + nb], a0);
    }
    for g in 0..n / 8 {
        let bgrp = &packed[g * nb * 8..(g + 1) * nb * 8];
        let mut got = [0f32; 8];
        repack_q8_0::gemv_group_scalar(bgrp, a0, nb, &mut got);
        for c in 0..8 {
            let r = want[g * 8 + c];
            let rel = (r - got[c]).abs() / r.abs().max(1e-3);
            assert!(
                rel < 1e-4,
                "oracle col {}: dot {r} vs {} rel {rel}",
                g * 8 + c,
                got[c]
            );
        }
    }
    // Tiled AVX2 gemm vs oracle for every tile height.
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    for g in 0..n / 8 {
        let bgrp = &packed[g * nb * 8..(g + 1) * nb * 8];
        for mt in 1..=m {
            let refs: Vec<&[BlockQ8_0]> = (0..mt).map(|t| &acts[t * nb..(t + 1) * nb]).collect();
            let mut want_t = vec![0f32; mt * 8];
            for (t, a) in refs.iter().enumerate() {
                repack_q8_0::gemv_group_scalar(bgrp, a, nb, &mut want_t[t * 8..t * 8 + 8]);
            }
            let mut got = [0f32; 64];
            unsafe {
                repack_q8_0::gemm_group_avx2(bgrp, &refs, nb, &mut got);
            }
            for (i, w) in want_t.iter().enumerate() {
                let rel = (got[i] - w).abs() / w.abs().max(1e-3);
                assert!(rel < 1e-4, "gemm mt={mt} lane {i}: got {} want {w}", got[i]);
            }
        }
    }
}

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[test]
pub(super) fn q8_0_full_matmul_matches_reference() {
    // Full driver (matmul_q8_0_repacked_tiled) vs the per-column reference
    // (matmul_bytes -> vec_dot_q8_0_q8_0) at a realistic projection shape. The
    // tiled kernel mirrors the reference's reduction order (per 32-block fmadd
    // of `x.d*y.d` into an f32 vector, hsum at the end), so the result is
    // bit-identical and greedy tokens stay unchanged.
    let (m, k, n) = (13usize, 2048usize, 512usize);
    let nb = k / 32;
    let mut seed = 0xB5026F5AA96619E9u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let wf: Vec<f32> = (0..n * k).map(|_| rng() * 4.0).collect();
    let mut wq = vec![BlockQ8_0::zeros(); n * nb];
    for row in 0..n {
        BlockQ8_0::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let wbytes: &[u8] = unsafe {
        std::slice::from_raw_parts(wq.as_ptr() as *const u8, std::mem::size_of_val(&wq[..]))
    };
    let lhs: Vec<f32> = (0..m * k).map(|_| rng() * 3.0).collect();

    let mut want = vec![0f32; m * n];
    matmul_bytes(GgmlDType::Q8_0, (m, k, n), &lhs, wbytes, &mut want).unwrap();

    let g8 = repack_q8_0::repack(&wq, n, nb);
    let mut got = vec![0f32; m * n];
    matmul_q8_0_repacked_tiled((m, k, n), &lhs, &g8, &mut got).unwrap();

    let mut maxabs = 0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    assert_eq!(
        maxabs, 0.0,
        "full q8_0 matmul must match reference bit-for-bit"
    );
}
