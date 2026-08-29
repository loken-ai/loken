//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[test]
pub(super) fn repacked_q4_0_gemm_matches_vec_dot_and_oracle() {
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
    let packed = repack_q4_0::repack(&rows, n, nb);
    // Oracle vs production per-column dot (row 0).
    let a0 = &acts[0..nb];
    let mut want = vec![0f32; n];
    for (c, w) in want.iter_mut().enumerate() {
        *w = avx::vec_dot_q4_0_q8_0(&rows[c * nb..c * nb + nb], a0);
    }
    for g in 0..n / 8 {
        let bgrp = &packed[g * nb * 8..(g + 1) * nb * 8];
        let mut got = [0f32; 8];
        repack_q4_0::gemv_group_scalar(bgrp, a0, nb, &mut got);
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
    for g in 0..n / 8 {
        let bgrp = &packed[g * nb * 8..(g + 1) * nb * 8];
        for mt in 1..=m {
            let refs: Vec<&[BlockQ8_0]> = (0..mt).map(|t| &acts[t * nb..(t + 1) * nb]).collect();
            let mut want_t = vec![0f32; mt * 8];
            for (t, a) in refs.iter().enumerate() {
                repack_q4_0::gemv_group_scalar(bgrp, a, nb, &mut want_t[t * 8..t * 8 + 8]);
            }
            let mut got = [0f32; 64];
            unsafe {
                repack_q4_0::gemm_group_avx2(bgrp, &refs, nb, &mut got);
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
pub(super) fn q4_0_full_matmul_matches_reference() {
    // Full driver (matmul_q4_0_repacked_tiled) vs the per-column reference
    // (matmul_bytes -> vec_dot_q4_0_q8_0) at a realistic projection shape. The
    // tiled kernel mirrors the reference's reduction order (per 32-block fmadd
    // of `x.d*y.d` into an f32 vector, hsum at the end), so the result is
    // bit-identical and greedy tokens stay unchanged.
    let (m, k, n) = (13usize, 2048usize, 512usize);
    let nb = k / 32;
    let mut seed = 0x8A5CD789635D2DFFu64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let wf: Vec<f32> = (0..n * k).map(|_| rng() * 4.0).collect();
    let mut wq = vec![
        BlockQ4_0 {
            d: f16::ZERO,
            qs: [0u8; 16]
        };
        n * nb
    ];
    for row in 0..n {
        BlockQ4_0::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let wbytes: &[u8] = unsafe {
        std::slice::from_raw_parts(wq.as_ptr() as *const u8, std::mem::size_of_val(&wq[..]))
    };
    let lhs: Vec<f32> = (0..m * k).map(|_| rng() * 3.0).collect();

    let mut want = vec![0f32; m * n];
    matmul_bytes(GgmlDType::Q4_0, (m, k, n), &lhs, wbytes, &mut want).unwrap();

    let g8 = repack_q4_0::repack(&wq, n, nb);
    let mut got = vec![0f32; m * n];
    matmul_q4_0_repacked_tiled((m, k, n), &lhs, &g8, &mut got).unwrap();

    let mut maxabs = 0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    assert_eq!(
        maxabs, 0.0,
        "full q4_0 matmul must match reference bit-for-bit"
    );
}

#[cfg(test)]
#[test]
pub(super) fn mxfp4_full_matmul_matches_reference() {
    // Full driver (matmul_mxfp4_repacked_tiled) vs the per-column reference
    // (matmul_bytes -> vec_dot_mxfp4_q8_0) at a realistic projection shape. The
    // tiled kernel mirrors the reference's reduction order (code->LUT shuffle,
    // pairwise int8 dot, one fmadd of `e8m0_half(e)*d_act` per 32-block into an
    // f32 vector, hsum at the end), so it is bit-identical and greedy tokens
    // stay unchanged.
    let (m, k, n) = (13usize, 2048usize, 512usize);
    let nb = k / QK_MXFP4;
    let mut seed = 0x51D3F00DABCDEF01u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let wf: Vec<f32> = (0..n * k).map(|_| rng() * 4.0).collect();
    let mut wq = vec![
        BlockMxFp4 {
            e: 0,
            qs: [0u8; QK_MXFP4 / 2]
        };
        n * nb
    ];
    for row in 0..n {
        BlockMxFp4::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let wbytes: &[u8] = unsafe {
        std::slice::from_raw_parts(wq.as_ptr() as *const u8, std::mem::size_of_val(&wq[..]))
    };
    let lhs: Vec<f32> = (0..m * k).map(|_| rng() * 3.0).collect();

    let mut want = vec![0f32; m * n];
    matmul_bytes(GgmlDType::MxFp4, (m, k, n), &lhs, wbytes, &mut want).unwrap();

    let g8 = repack_mxfp4::repack(&wq, n, nb);
    let mut got = vec![0f32; m * n];
    matmul_mxfp4_repacked_tiled((m, k, n), &lhs, &g8, &mut got).unwrap();

    let mut maxabs = 0f32;
    for (a, b) in got.iter().zip(want.iter()) {
        maxabs = maxabs.max((a - b).abs());
    }
    assert_eq!(
        maxabs, 0.0,
        "full mxfp4 matmul must match reference bit-for-bit"
    );
}

#[cfg(test)]
#[test]
pub(super) fn mxfp4_x8_scalar_matches_reference() {
    // The column-interleaved scalar oracle (repack_mxfp4_x8::matmul_scalar) vs the
    // per-column reference (matmul_bytes -> vec_dot_mxfp4_q8_0). NOT bit-identical
    // (different reduction order + independent Q8_0 activation quantization), so
    // the bar is a small relative error - this validates the interleaved LAYOUT
    // (repack + activation packing + the 8x8 dot indexing). m not a multiple of 4
    // exercises the row padding.
    let (m, k, n) = (13usize, 2048usize, 512usize);
    let nb = k / QK_MXFP4;
    let mut seed = 0x1234_5678_9ABC_DEF1u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let wf: Vec<f32> = (0..n * k).map(|_| rng() * 4.0).collect();
    let mut wq = vec![
        BlockMxFp4 {
            e: 0,
            qs: [0u8; QK_MXFP4 / 2]
        };
        n * nb
    ];
    for row in 0..n {
        BlockMxFp4::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let wbytes: &[u8] = unsafe {
        std::slice::from_raw_parts(wq.as_ptr() as *const u8, std::mem::size_of_val(&wq[..]))
    };
    let lhs: Vec<f32> = (0..m * k).map(|_| rng() * 3.0).collect();

    let mut want = vec![0f32; m * n];
    matmul_bytes(GgmlDType::MxFp4, (m, k, n), &lhs, wbytes, &mut want).unwrap();

    let mut got = vec![0f32; m * n];
    repack_mxfp4_x8::matmul_scalar((m, k, n), &lhs, &wq, &mut got);

    let mut num = 0f64;
    let mut den = 0f64;
    for (a, b) in got.iter().zip(want.iter()) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    let rel = (num / den).sqrt();
    assert!(
        rel < 1e-2,
        "interleaved scalar oracle rel error {rel} too high (layout bug?)"
    );
}

#[cfg(test)]
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[test]
pub(super) fn mxfp4_x8_avx2_matches_scalar() {
    // The AVX2 column-interleaved kernel must be BIT-IDENTICAL to the scalar
    // oracle (same reduction order by construction). One 4-row x 8-col tile.
    use repack_mxfp4_x8::{BlockMxFp4x8, BlockQ8_0x4};
    let nb = 6usize;
    let mut seed = 0xDEAD_BEEF_1357_9BDFu64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 33) as u32
    };
    let mut b = vec![
        BlockMxFp4x8 {
            e: [0u8; 8],
            qs: [0u8; 128]
        };
        nb
    ];
    let mut a = vec![
        BlockQ8_0x4 {
            d: [f16::ZERO; 4],
            qs: [0i8; 128]
        };
        nb
    ];
    for blk in 0..nb {
        for c in 0..8 {
            b[blk].e[c] = (120 + (rng() % 16)) as u8;
        }
        for x in b[blk].qs.iter_mut() {
            *x = (rng() & 0xFF) as u8;
        }
        for m in 0..4 {
            a[blk].d[m] = f16::from_f32(0.01 + (rng() % 100) as f32 * 0.001);
        }
        for x in a[blk].qs.iter_mut() {
            *x = ((rng() % 255) as i32 - 127) as i8;
        }
    }
    let mut want = [0f32; 32];
    repack_mxfp4_x8::gemm_scalar(&a, &b, nb, &mut want);
    let mut got = [0f32; 32];
    unsafe { repack_mxfp4_x8::gemm_group_avx2(&a, &b, nb, &mut got) };
    // Not bit-identical to the scalar (the AVX2 folds the scale as
    // `iacc.(col.row)` via fmadd vs the scalar's `(sumi.col).row` with separate
    // adds) - the bar is small relative error, same as the ggml generic-vs-AVX2
    // relationship. A lane/shuffle bug would blow this up.
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    let rel = (num / den).sqrt();
    assert!(
        rel < 1e-4,
        "AVX2 kernel rel error {rel} vs scalar oracle too high (shuffle bug?)"
    );
}

#[cfg(test)]
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[test]
pub(super) fn mxfp4_x8_gemm16_matches_scalar() {
    // The 16-row kernel (decode-once, 4 activation groups) must equal the 4-row
    // kernel applied to each group - same math, so bit-identical to gemm_group_avx2
    // (which is rel<1e-4 vs the scalar oracle). Check vs the scalar oracle per group.
    use repack_mxfp4_x8::{BlockMxFp4x8, BlockQ8_0x4};
    let nb = 5usize;
    let mut seed = 0x0F1E_2D3C_4B5A_6978u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 33) as u32
    };
    let mut b = vec![
        BlockMxFp4x8 {
            e: [0u8; 8],
            qs: [0u8; 128]
        };
        nb
    ];
    for blk in b.iter_mut() {
        for c in 0..8 {
            blk.e[c] = (120 + (rng() % 16)) as u8;
        }
        for x in blk.qs.iter_mut() {
            *x = (rng() & 0xFF) as u8;
        }
    }
    let mut mkgrp = || {
        let mut g = vec![
            BlockQ8_0x4 {
                d: [f16::ZERO; 4],
                qs: [0i8; 128]
            };
            nb
        ];
        for blk in g.iter_mut() {
            for m in 0..4 {
                blk.d[m] = f16::from_f32(0.01 + (rng() % 100) as f32 * 0.001);
            }
            for x in blk.qs.iter_mut() {
                *x = ((rng() % 255) as i32 - 127) as i8;
            }
        }
        g
    };
    let (a0, a1, a2, a3) = (mkgrp(), mkgrp(), mkgrp(), mkgrp());
    let mut got = [0f32; 128];
    unsafe { repack_mxfp4_x8::gemm_group16_avx2(&a0, &a1, &a2, &a3, &b, nb, &mut got) };
    for (rp, ag) in [&a0, &a1, &a2, &a3].iter().enumerate() {
        let mut want = [0f32; 32];
        repack_mxfp4_x8::gemm_scalar(ag, &b, nb, &mut want);
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..32 {
            num += ((got[rp * 32 + i] - want[i]) as f64).powi(2);
            den += (want[i] as f64).powi(2);
        }
        assert!((num / den).sqrt() < 1e-4, "gemm16 row-group {rp} mismatch");
    }
}

#[cfg(test)]
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[test]
pub(super) fn mxfp4_x8_tiled_driver_matches_reference() {
    // Full driver (matmul_mxfp4_x8_tiled) vs matmul_bytes. m=20 exercises one
    // 16-row chunk + one 4-row remainder.
    let (m, k, n) = (20usize, 1024usize, 256usize);
    let nb = k / QK_MXFP4;
    let mut seed = 0xABCD_1234_5678_9EF0u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let wf: Vec<f32> = (0..n * k).map(|_| rng() * 4.0).collect();
    let mut wq = vec![
        BlockMxFp4 {
            e: 0,
            qs: [0u8; QK_MXFP4 / 2]
        };
        n * nb
    ];
    for row in 0..n {
        BlockMxFp4::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let wbytes: &[u8] = unsafe {
        std::slice::from_raw_parts(wq.as_ptr() as *const u8, std::mem::size_of_val(&wq[..]))
    };
    let lhs: Vec<f32> = (0..m * k).map(|_| rng() * 3.0).collect();
    let mut want = vec![0f32; m * n];
    matmul_bytes(GgmlDType::MxFp4, (m, k, n), &lhs, wbytes, &mut want).unwrap();
    let x8 = repack_mxfp4_x8::repack(&wq, n, nb);
    let mut got = vec![0f32; m * n];
    matmul_mxfp4_x8_tiled((m, k, n), &lhs, &x8, &mut got).unwrap();
    let (mut num, mut den) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want.iter()) {
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    assert!(
        (num / den).sqrt() < 1e-2,
        "x8 tiled driver rel error too high"
    );
}

#[cfg(test)]
#[test]
pub(super) fn q8_0_x8_scalar_matches_reference() {
    // Column-interleaved Q8_0 scalar oracle vs the per-column reference
    // (matmul_bytes -> vec_dot_q8_0_q8_0). Validates the BlockQ8_0x8 layout +
    // activation packing. rel-close (independent activation quantization).
    let (m, k, n) = (13usize, 1024usize, 256usize);
    let nb = k / 32;
    let mut seed = 0x7788_99AA_BBCC_DDEEu64;
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
    let mut got = vec![0f32; m * n];
    repack_q8_0_x8::matmul_scalar((m, k, n), &lhs, &wq, &mut got);
    let (mut num, mut den) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want.iter()) {
        num += ((g - w) as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    let rel = (num / den).sqrt();
    assert!(
        rel < 1e-2,
        "q8_0 x8 scalar oracle rel error {rel} too high (layout bug?)"
    );
}

#[cfg(test)]
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[test]
pub(super) fn q8_0_x8_avx2_matches_scalar() {
    // Q8_0 AVX2 kernel (4-row and 16-row) vs the scalar oracle - rel<1e-4 (float
    // scale rounding only; the integer dot is order-independent).
    use repack_q8_0_x8::{BlockQ8_0x4, BlockQ8_0x8};
    let nb = 5usize;
    let mut seed = 0x2468_ACE0_1357_9BDFu64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 33) as u32
    };
    let mut b = vec![
        BlockQ8_0x8 {
            d: [f16::ZERO; 8],
            qs: [0i8; 256]
        };
        nb
    ];
    for blk in b.iter_mut() {
        for c in 0..8 {
            blk.d[c] = f16::from_f32(0.02 + (rng() % 50) as f32 * 0.002);
        }
        for x in blk.qs.iter_mut() {
            *x = ((rng() % 255) as i32 - 127) as i8;
        }
    }
    let mut mkgrp = || {
        let mut g = vec![
            BlockQ8_0x4 {
                d: [f16::ZERO; 4],
                qs: [0i8; 128]
            };
            nb
        ];
        for blk in g.iter_mut() {
            for m in 0..4 {
                blk.d[m] = f16::from_f32(0.01 + (rng() % 100) as f32 * 0.001);
            }
            for x in blk.qs.iter_mut() {
                *x = ((rng() % 255) as i32 - 127) as i8;
            }
        }
        g
    };
    let (a0, a1, a2, a3) = (mkgrp(), mkgrp(), mkgrp(), mkgrp());
    // 4-row tile
    for ag in [&a0, &a1, &a2, &a3] {
        let mut want = [0f32; 32];
        repack_q8_0_x8::gemm_scalar(ag, &b, nb, &mut want);
        let mut got = [0f32; 32];
        unsafe { repack_q8_0_x8::gemm_group_avx2(ag, &b, nb, &mut got) };
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..32 {
            num += ((got[i] - want[i]) as f64).powi(2);
            den += (want[i] as f64).powi(2);
        }
        assert!((num / den).sqrt() < 1e-4, "q8_0 4-row AVX2 mismatch");
    }
    // 16-row tile
    let mut got = [0f32; 128];
    unsafe { repack_q8_0_x8::gemm_group16_avx2(&a0, &a1, &a2, &a3, &b, nb, &mut got) };
    for (rp, ag) in [&a0, &a1, &a2, &a3].iter().enumerate() {
        let mut want = [0f32; 32];
        repack_q8_0_x8::gemm_scalar(ag, &b, nb, &mut want);
        let (mut num, mut den) = (0f64, 0f64);
        for i in 0..32 {
            num += ((got[rp * 32 + i] - want[i]) as f64).powi(2);
            den += (want[i] as f64).powi(2);
        }
        assert!(
            (num / den).sqrt() < 1e-4,
            "q8_0 16-row AVX2 row-group {rp} mismatch"
        );
    }
}

#[cfg(test)]
#[test]
pub(super) fn repacked_q5k_matches_vec_dot_and_oracle() {
    let (n, nb) = (16usize, 4usize);
    let mut seed = 0xC2B2AE3D27D4EB4Fu64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let k = nb * QK_K;
    let wf: Vec<f32> = (0..n * k).map(|_| rng() * 4.0).collect();
    let mut wq = vec![BlockQ5K::zeros(); n * nb];
    for row in 0..n {
        BlockQ5K::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let m = 8usize;
    let mut aq = vec![BlockQ8K::zeros(); m * nb];
    for r in 0..m {
        let af: Vec<f32> = (0..k).map(|_| rng() * 3.0).collect();
        BlockQ8K::quantize(&af, &mut aq[r * nb..(r + 1) * nb]);
    }
    let x8 = repack_q5k::repack(&wq, n, nb);
    // Oracle vs production per-column dot (row 0).
    let a0 = &aq[0..nb];
    let mut want = vec![0f32; n];
    for (c, w) in want.iter_mut().enumerate() {
        *w = avx::vec_dot_q5k_q8k(&wq[c * nb..c * nb + nb], a0);
    }
    for g in 0..n / 8 {
        let bgrp = &x8[g * nb..(g + 1) * nb];
        let mut got = [0f32; 8];
        repack_q5k::gemv_group_scalar(bgrp, a0, nb, &mut got);
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
    // Tiled AVX2 gemm vs the scalar oracle for every tile height. The gemm
    // targets the per-column `dot` reduction order (main term in an f32
    // vector, min term in a separate scalar) - validated bit-for-bit by
    // `q5k_full_matmul_matches_reference` - whereas the oracle folds both into
    // one scalar per super-block, so the two differ only by float summation
    // order; a loose bound still catches any structural (scale/plane/min) bug.
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    for g in 0..n / 8 {
        let bgrp = &x8[g * nb..(g + 1) * nb];
        for mt in 1..=m {
            let refs: Vec<&[BlockQ8K]> = (0..mt).map(|t| &aq[t * nb..(t + 1) * nb]).collect();
            let mut want_t = vec![0f32; mt * 8];
            for (t, a) in refs.iter().enumerate() {
                repack_q5k::gemv_group_scalar(bgrp, a, nb, &mut want_t[t * 8..t * 8 + 8]);
            }
            let mut got = [0f32; 64];
            unsafe {
                repack_q5k::gemm_group_avx2(bgrp, &refs, nb, &mut got);
            }
            for (i, w) in want_t.iter().enumerate() {
                let rel = (got[i] - w).abs() / w.abs().max(1e-3);
                assert!(rel < 1e-3, "gemm mt={mt} lane {i}: got {} want {w}", got[i]);
            }
        }
    }
}

#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
#[test]
pub(super) fn q5k_full_matmul_matches_reference() {
    // Full driver (matmul_q5k_repacked_tiled) vs the per-column reference
    // (matmul_bytes -> vec_dot_q5k_q8k) at a realistic projection shape.
    let (m, k, n) = (13usize, 2048usize, 512usize);
    let nb = k / QK_K;
    let mut seed = 0x9E3779B97F4A7C15u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    };
    let wf: Vec<f32> = (0..n * k).map(|_| rng() * 4.0).collect();
    let mut wq = vec![BlockQ5K::zeros(); n * nb];
    for row in 0..n {
        BlockQ5K::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    // Raw byte view of the weights, as the mmap'd path sees them.
    let wbytes: &[u8] = unsafe {
        std::slice::from_raw_parts(wq.as_ptr() as *const u8, std::mem::size_of_val(&wq[..]))
    };
    let lhs: Vec<f32> = (0..m * k).map(|_| rng() * 3.0).collect();

    let mut want = vec![0f32; m * n];
    matmul_bytes(GgmlDType::Q5K, (m, k, n), &lhs, wbytes, &mut want).unwrap();

    let x8 = repack_q5k::repack(&wq, n, nb);
    let mut got = vec![0f32; m * n];
    matmul_q5k_repacked_tiled((m, k, n), &lhs, &x8, &mut got).unwrap();

    // The tiled kernel mirrors the per-column reference's reduction order, so
    // the result is bit-identical (the K-quant dot is integer-exact and the
    // float summation order matches) - greedy tokens stay unchanged.
    let mut maxabs = 0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        maxabs = maxabs.max((g - w).abs());
    }
    assert_eq!(
        maxabs, 0.0,
        "full q5k matmul must match reference bit-for-bit"
    );
}

/// Column-chunk grid for an `[m, n]` output sharded across the pool: chunk
/// size targets several chunks per thread (cheap atomic dispatch makes small
/// chunks affordable, and they absorb thread-speed jitter).
pub(in crate::tensor) fn gemv_grid(m: usize, n: usize, threads: usize) -> (usize, usize) {
    // Several chunks per thread: the region's tail is one chunk long (the
    // last grabbed chunk gates the publisher), so finer chunks shrink it;
    // the atomic grab keeps per-chunk dispatch cheap. 8/thread is the measured
    // sweet spot: 32/thread is a wash (+0.4%, inside noise - the ~6% of cycles
    // `perf` finds in the worker spin loop is per-region dispatch/sync latency,
    // NOT tail imbalance, so finer chunks don't reclaim it), while one
    // contiguous chunk per thread is the slowest (-1.9%).
    let chunk_cols = (n / (threads * 8)).clamp(16, 256).min(n.max(1));
    let chunks_per_row = n.div_ceil(chunk_cols);
    (chunk_cols, chunks_per_row * m)
}

// https://github.com/ggml-org/llama.cpp/blob/aa3ee0eb0b80efca126cedf9bcb4fb5864b46ce3/ggml/src/ggml-cpu/ggml-cpu.c#L1205
pub fn matmul<T: BlockFormat>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()>
where
    T::ActivationBlock: 'static,
{
    debug_assert_eq!(
        T::BLOCK_LEN,
        T::ActivationBlock::BLOCK_LEN,
        "Mismatched block sizes"
    );
    // A REAL check, not a debug one. This invariant was asserted with `debug_assert_eq!`,
    // which is compiled out of the binary that ships - so when a caller disagreed with the
    // shape it declared, nothing caught it here and the mismatch surfaced further down as a
    // `copy_from_slice` panic inside the activation staging, killing the render task with a
    // message naming two slice lengths and no operation. Returning an error costs one
    // comparison per matmul and turns that into something a caller can report or recover
    // from.
    if m * k != lhs.len() {
        return Err(Error(format!(
            "quant matmul: activation has {} elements but the shape says {m}x{k} = {}              (n={n}) - the caller's shape and its data disagree",
            lhs.len(),
            m * k
        )));
    }
    let k_in_blocks = k.div_ceil(T::BLOCK_LEN);
    let pool = gemv_pool::pool();

    // Reused per-thread scratch (see `qscratch_take`) instead of a fresh
    // allocation per matmul - returned at the end so its buffer is recycled.
    let mut lhs_b = qscratch_take::<T::ActivationBlock>(m * k_in_blocks);
    // f32, f16, and bf16 support direct copy
    if T::COPIES_VERBATIM {
        // The scratch is RECYCLED and only ever grows, so it is routinely LONGER than this
        // call needs - handing over the whole buffer made the copy's two lengths disagree
        // and panic, killing the request. It only shows up once a bigger matmul has run on
        // the same thread first, which is why a path can work for months and then fail the
        // day something larger precedes it.
        T::ActivationBlock::copy_verbatim(lhs, &mut lhs_b[..m * k_in_blocks]);
    } else if m >= 4 {
        // Prefill: activation quantization is row-parallel (it was a serial
        // pass over every prompt token before the columns even start).
        let lhs_b_ptr = SendMutPtr(lhs_b.as_mut_ptr());
        pool.run(m, &|row_idx| {
            let lhs_b_ptr = &lhs_b_ptr;
            // SAFETY: rows are disjoint `k_in_blocks` slices of `lhs_b`.
            let lhs_b_mut = unsafe {
                std::slice::from_raw_parts_mut(lhs_b_ptr.0.add(row_idx * k_in_blocks), k_in_blocks)
            };
            let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
            T::ActivationBlock::quantize(lhs, lhs_b_mut)
        });
    } else {
        // Decode (m<4): the activation quantize ran single-threaded over all K
        // while the GEMV pool idled - a per-GEMV serial tail that dilutes the
        // wide-FFN GEMVs below the streaming ceiling. Parallelise it over
        // block-chunks when K is block-aligned and there is enough work to
        // amortise the dispatch; otherwise keep the serial pass.
        let bs = T::ActivationBlock::BLOCK_LEN;
        if k % bs == 0 && k_in_blocks >= 2 {
            // Mirror ggml's `quantize` partition: split the row's K-blocks
            // EVENLY across all pool workers (one contiguous chunk per thread),
            // not fixed 8-block chunks gated on a 32-block minimum. For the
            // common K=5120 weights (20 blocks) the old path fell to the serial
            // branch (20 < 32) so the quantize ran single-threaded while every
            // other worker idled; an even K/nth split engages all of them.
            let nth = pool.threads.min(k_in_blocks);
            let lhs_b_ptr = SendMutPtr(lhs_b.as_mut_ptr());
            pool.run(m * nth, &|chunk| {
                let lhs_b_ptr = &lhs_b_ptr;
                let row_idx = chunk / nth;
                let t = chunk % nth;
                let b0 = t * k_in_blocks / nth;
                let b1 = (t + 1) * k_in_blocks / nth;
                if b1 <= b0 {
                    return;
                }
                let xs = &lhs[row_idx * k + b0 * bs..row_idx * k + b1 * bs];
                // SAFETY: each chunk owns a disjoint block range [b0,b1) of row.
                let ys = unsafe {
                    std::slice::from_raw_parts_mut(
                        lhs_b_ptr.0.add(row_idx * k_in_blocks + b0),
                        b1 - b0,
                    )
                };
                T::ActivationBlock::quantize(xs, ys);
            });
        } else {
            for row_idx in 0..m {
                let lhs_b_mut = &mut lhs_b[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
                let lhs = &lhs[row_idx * k..(row_idx + 1) * k];
                T::ActivationBlock::quantize(lhs, lhs_b_mut)
            }
        }
    }
    let lhs_b_slice = lhs_b.as_slice();
    let dst_ptr = SendMutPtr(dst.as_mut_ptr());

    if m >= 4 {
        // Prefill: weight-stationary row tiling. Parallelise over column chunks;
        // within a chunk, hold a tile of R activation rows resident (L2) and reuse
        // each loaded weight column across all R rows before moving on. The weight
        // block then streams from DRAM ~m/R times instead of once per row - the
        // per-column path (decode branch below) re-reads the whole weight matrix
        // for every prompt row, which dominates prefill for K-quants that have no
        // specialised repacked GEMM (e.g. Q6_K down/V). Bit-identical: the same
        // `dot` calls, only reordered. Q4_K is diverted to the repacked tiled
        // GEMM before reaching here; this lifts every other K-quant to the same
        // weight-reuse regime.
        const R: usize = 4;
        let chunk_cols = (n / (pool.threads * 8)).clamp(8, 256).min(n.max(1));
        let n_col_chunks = n.div_ceil(chunk_cols);
        pool.run(n_col_chunks, &|cc| {
            let dst_ptr = &dst_ptr;
            let col0 = cc * chunk_cols;
            let col1 = (col0 + chunk_cols).min(n);
            let mut r0 = 0;
            while r0 < m {
                let r1 = (r0 + R).min(m);
                for col in col0..col1 {
                    let rhs_col = &rhs_t[col * k_in_blocks..(col + 1) * k_in_blocks];
                    for row in r0..r1 {
                        let lhs_row = &lhs_b_slice[row * k_in_blocks..(row + 1) * k_in_blocks];
                        let v = T::dot(rhs_col, lhs_row);
                        // SAFETY: each (row, col) is written exactly once - column
                        // chunks are disjoint across threads, rows disjoint within.
                        unsafe { *dst_ptr.0.add(row * n + col) = v };
                    }
                }
                r0 = r1;
            }
        });
    } else {
        // Decode (m<4): per-column path. No row reuse is possible at m=1, and the
        // GEMV is bandwidth-bound - streaming each weight column once is optimal.
        let (chunk_cols, n_chunks) = gemv_grid(m, n, pool.threads);
        let chunks_per_row = n.div_ceil(chunk_cols);
        pool.run(n_chunks, &|chunk| {
            let dst_ptr = &dst_ptr;
            let row_idx = chunk / chunks_per_row;
            let col0 = (chunk % chunks_per_row) * chunk_cols;
            let col1 = (col0 + chunk_cols).min(n);
            let lhs_row = &lhs_b_slice[row_idx * k_in_blocks..(row_idx + 1) * k_in_blocks];
            // SAFETY: chunks address disjoint [row, col0..col1] ranges of dst.
            let dst_row = unsafe {
                std::slice::from_raw_parts_mut(dst_ptr.0.add(row_idx * n + col0), col1 - col0)
            };
            for (i, d) in dst_row.iter_mut().enumerate() {
                let col_idx = col0 + i;
                let rhs_col = &rhs_t[col_idx * k_in_blocks..(col_idx + 1) * k_in_blocks];
                *d = T::dot(rhs_col, lhs_row);
            }
        });
    }
    qscratch_return(lhs_b);
    Ok(())
}

/// One driver for EVERY M over the unified plain-scales Q4_K repack - the storage-level
/// entry for in-place repacked weights. M=1 runs the interleaved decode GEMV
/// (`gemv_group_avx2_plain`, llama.cpp-parity numerics); M>=2 runs the row-tiled
/// prefill GEMM (weight-reuse; covered by the m=1,3,4,7,9 parity suite).
pub fn matmul_q4k_plain(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_x8: &[repack_q4k::BlockQ4Kx8],
    dst: &mut [f32],
) -> Result<()> {
    #[cfg(not(all(target_feature = "avx2", target_arch = "x86_64")))]
    {
        let _ = (m, k, n, lhs, rhs_x8, dst);
        Err(Error("matmul_q4k_plain: avx2 build required".into()))
    }
    #[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
    {
        if m >= 2 {
            return matmul_q4k_repacked_tiled((m, k, n), lhs, rhs_x8, dst);
        }
        let nb = k / QK_K;
        let mut aq = qscratch_take::<BlockQ8K>(nb);
        BlockQ8K::quantize(lhs, &mut aq[..nb]);
        let r = matmul_q4k_plain_pre((k, n), &aq[..nb], rhs_x8, dst);
        qscratch_return(aq);
        r
    }
}

/// Quantise one activation row to Q8_K, for callers that then drive SEVERAL
/// weights from it via [`matmul_q4k_plain_pre`]. Same routine the GEMV uses
/// internally, so sharing the result is bit-identical to quantising per call.
#[cfg(all(target_feature = "avx2", target_arch = "x86_64"))]
pub fn quantize_activation_q8k(x: &[f32], nb: usize) -> Vec<BlockQ8K> {
    let mut aq = vec![BlockQ8K::zeros(); nb];
    BlockQ8K::quantize(x, &mut aq);
    aq
}
