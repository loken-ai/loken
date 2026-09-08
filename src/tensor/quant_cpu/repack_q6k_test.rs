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

fn prng() -> impl FnMut() -> f32 {
    let mut s = 0x0FED_CBA9_8765_4321u64;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

// The repacked Q6_K GEMV oracle must match the production per-column
// vec_dot_q6k_q8k: same activation, same dequant, only the layout differs.
#[cfg(target_feature = "avx2")]
#[test]
fn repacked_q6k_gemv_matches_vec_dot() {
    let k = 512usize; // 2 superblocks
    let n = 16usize; // 2 column groups of 8
    let nb = k / QK_K;
    let mut next = prng();
    let wf: Vec<f32> = (0..n * k).map(|_| next() * 4.0).collect();
    let mut wq = vec![BlockQ6K::zeros(); n * nb];
    for row in 0..n {
        BlockQ6K::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let af: Vec<f32> = (0..k).map(|_| next() * 3.0).collect();
    let mut aq = vec![BlockQ8K::zeros(); nb];
    BlockQ8K::quantize(&af, &mut aq);

    // Reference: per-column dot (the production path for Q6_K).
    let mut reference = vec![0f32; n];
    for (col, r) in reference.iter_mut().enumerate() {
        *r = avx::vec_dot_q6k_q8k(&wq[col * nb..(col + 1) * nb], &aq);
    }

    let x8 = repack_q6k::repack(&wq, n, nb);
    let mut out_scalar = vec![0f32; n];
    for g in 0..n / 8 {
        let bgrp = &x8[g * nb..(g + 1) * nb];
        repack_q6k::gemv_group_scalar(bgrp, &aq, nb, &mut out_scalar[g * 8..g * 8 + 8]);
    }
    for col in 0..n {
        let r = reference[col];
        let rel = (r - out_scalar[col]).abs() / r.abs().max(1e-3);
        assert!(
            rel < 1e-4,
            "col {col}: scalar {} vs ref {r} rel {rel}",
            out_scalar[col]
        );
    }
}

// The tiled AVX2 GEMM must agree with the scalar oracle for every tile
// height, since production picks the tile size by row count.
#[cfg(target_feature = "avx2")]
#[test]
fn repacked_q6k_gemm_matches_scalar_oracle() {
    let k = 512usize;
    let n = 16usize;
    let nb = k / QK_K;
    let mut next = prng();
    let wf: Vec<f32> = (0..n * k).map(|_| next() * 4.0).collect();
    let mut wq = vec![BlockQ6K::zeros(); n * nb];
    for row in 0..n {
        BlockQ6K::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let m = 8usize;
    let mut aq = vec![BlockQ8K::zeros(); m * nb];
    for r in 0..m {
        let af: Vec<f32> = (0..k).map(|_| next() * 3.0).collect();
        BlockQ8K::quantize(&af, &mut aq[r * nb..(r + 1) * nb]);
    }
    let x8 = repack_q6k::repack(&wq, n, nb);
    for g in 0..n / 8 {
        let bgrp = &x8[g * nb..(g + 1) * nb];
        for mt in 1..=m {
            let act_refs: Vec<&[BlockQ8K]> = (0..mt).map(|t| &aq[t * nb..(t + 1) * nb]).collect();
            let mut want = vec![0f32; mt * 8];
            for (t, a) in act_refs.iter().enumerate() {
                repack_q6k::gemv_group_scalar(bgrp, a, nb, &mut want[t * 8..t * 8 + 8]);
            }
            let mut got = [0f32; 64];
            unsafe {
                repack_q6k::gemm_group_avx2(bgrp, &act_refs, nb, &mut got);
            }
            for (i, w) in want.iter().enumerate() {
                let rel = (got[i] - w).abs() / w.abs().max(1e-3);
                assert!(rel < 1e-4, "mt={mt} lane {i}: got {} want {w}", got[i]);
            }
        }
    }
}
