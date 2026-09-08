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
    let mut s = 0x1234_5678_9ABC_DEF0u64;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

// The prefill GEMM kernels (row-strip blocked, with and without the
// hoisted per-group scale vectors) must agree with the scalar oracle for
// every tile height, since production picks the variant by row count.
/// Isolated timing of the decode GEMV at the shapes a small dense model
/// actually runs, so a kernel change can be judged without a whole model in
/// the way. The campaign puts ~81% of CPU decode time in the FFN, and these
/// are its two shapes: [k=2048, n=8192] for gate and up, [k=8192, n=2048]
/// for down. Reports us per call and the implied MAC rate, which is what a
/// comparison against another implementation needs.
///
/// Ignored by default - it is a measurement, not a check. Run with:
///   cargo test --release -p loken gemv_q4k_decode_shapes -- --ignored --nocapture
#[test]
#[ignore = "benchmark: run explicitly with --ignored --nocapture"]
fn gemv_q4k_decode_shapes() {
    for (k, n) in [(2048usize, 8192usize), (8192usize, 2048usize)] {
        let nb = k / QK_K;
        let mut next = prng();
        let wf: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
        let mut wq = vec![BlockQ4K::zeros(); n * nb];
        for row in 0..n {
            BlockQ4K::quantize(
                &wf[row * k..(row + 1) * k],
                &mut wq[row * nb..(row + 1) * nb],
            );
        }
        let x8 = repack_q4k::repack(&wq, n, nb);
        let x: Vec<f32> = (0..k).map(|_| next()).collect();
        let mut out = vec![0f32; n];

        for _ in 0..20 {
            matmul_q4k_plain((1, k, n), &x, &x8, &mut out).unwrap();
        }
        // Enough repetitions that the timer's own cost disappears, and the
        // median of several rounds so one scheduling hiccup does not decide.
        let mut rounds = Vec::new();
        for _ in 0..7 {
            const IT: usize = 100;
            let t = std::time::Instant::now();
            for _ in 0..IT {
                matmul_q4k_plain((1, k, n), &x, &x8, &mut out).unwrap();
            }
            rounds.push(t.elapsed().as_secs_f64() * 1e6 / IT as f64);
        }
        rounds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let us = rounds[rounds.len() / 2];
        let gmac = (k * n) as f64 / (us * 1e3);
        println!(
            "GEMV q4k k={k} n={n}: {us:.1} us/call  {gmac:.2} GMAC/s  (best {:.1}, worst {:.1})",
            rounds[0],
            rounds[rounds.len() - 1]
        );

        // Same shape through the GENERIC byte path - what a weight whose dtype
        // has no repacked decode GEMV actually takes. Q4_K_M stores ffn_down as
        // Q6_K, and the decode slice path only tries the Q4_K repack, so this is
        // the path that weight really runs.
        let mut wq6 = vec![BlockQ6K::zeros(); n * nb];
        for row in 0..n {
            BlockQ6K::quantize(
                &wf[row * k..(row + 1) * k],
                &mut wq6[row * nb..(row + 1) * nb],
            );
        }
        let bytes6: &[u8] = unsafe {
            std::slice::from_raw_parts(wq6.as_ptr() as *const u8, std::mem::size_of_val(&wq6[..]))
        };
        for _ in 0..5 {
            matmul_bytes(GgmlDType::Q6K, (1, k, n), &x, bytes6, &mut out).unwrap();
        }
        let mut r6 = Vec::new();
        for _ in 0..5 {
            const IT6: usize = 30;
            let t = std::time::Instant::now();
            for _ in 0..IT6 {
                matmul_bytes(GgmlDType::Q6K, (1, k, n), &x, bytes6, &mut out).unwrap();
            }
            r6.push(t.elapsed().as_secs_f64() * 1e6 / IT6 as f64);
        }
        // Same kernel, but with the workers allowed to PARK between calls -
        // which is what happens in a model, where every matmul is separated by
        // norms, activations and attention. A tight loop keeps them hot and
        // hides whatever a wake costs.
        let mut spaced = Vec::new();
        for _ in 0..40 {
            let gap = std::time::Instant::now();
            while gap.elapsed().as_micros() < 300 {
                std::hint::spin_loop();
            }
            let t = std::time::Instant::now();
            matmul_q4k_plain((1, k, n), &x, &x8, &mut out).unwrap();
            spaced.push(t.elapsed().as_secs_f64() * 1e6);
        }
        spaced.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  spaced (workers parked between calls): {:.1} us median, {:.1} best, {:.1} worst  -> {:.1}x the hot loop",
            spaced[spaced.len() / 2], spaced[0], spaced[spaced.len() - 1],
            spaced[spaced.len() / 2] / us
        );

        r6.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let us6 = r6[r6.len() / 2];
        println!(
            "GEMV q6k-bytes k={k} n={n}: {us6:.1} us/call  {:.2} GMAC/s  -> {:.1}x the repacked q4k",
            (k * n) as f64 / (us6 * 1e3),
            us6 / us
        );
    }
}

#[test]
fn repacked_q4k_gemm_matches_scalar_oracle() {
    let k = 512usize;
    let n = 16usize;
    let nb = k / QK_K;
    let mut next = prng();
    let wf: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
    let mut wq = vec![BlockQ4K::zeros(); n * nb];
    for row in 0..n {
        BlockQ4K::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    let m = 8usize;
    let mut aq = vec![BlockQ8K::zeros(); m * nb];
    for r in 0..m {
        let af: Vec<f32> = (0..k).map(|_| next()).collect();
        BlockQ8K::quantize(&af, &mut aq[r * nb..(r + 1) * nb]);
    }
    let x8 = repack_q4k::repack(&wq, n, nb);
    for g in 0..n / 8 {
        let bgrp = &x8[g * nb..(g + 1) * nb];
        for mt in 1..=m {
            let act_refs: Vec<&[BlockQ8K]> = (0..mt).map(|t| &aq[t * nb..(t + 1) * nb]).collect();
            // Oracle: one scalar group GEMV per row.
            let mut want = vec![0f32; mt * 8];
            for (t, a) in act_refs.iter().enumerate() {
                repack_q4k::gemv_group_scalar(bgrp, a, nb, &mut want[t * 8..t * 8 + 8]);
            }
            #[cfg(target_feature = "avx2")]
            for variant in 0..2 {
                let mut got = [0f32; 64];
                unsafe {
                    if variant == 0 {
                        repack_q4k::gemm_group_avx2_rt::<1>(bgrp, &act_refs, nb, &mut got);
                    } else {
                        let gs = repack_q4k::prep_group_scales(bgrp, nb);
                        repack_q4k::gemm_group_avx2_pre_rt::<1>(bgrp, &act_refs, nb, &gs, &mut got);
                    }
                }
                for (i, w) in want.iter().enumerate() {
                    let rel = (got[i] - w).abs() / w.abs().max(1e-3);
                    assert!(
                        rel < 1e-4,
                        "variant {variant} mt={mt} lane {i}: got {} want {w}",
                        got[i]
                    );
                }
            }
        }
    }
}

// The repacked Q4_K GEMV (scalar + AVX2) must match the production
// per-column vec_dot_q4k_q8k within k_quants tolerance (same Q8_K
// activation, same dequant math, only the block layout differs).
#[cfg(target_feature = "avx2")]
#[test]
fn repacked_q4k_gemv_matches_vec_dot() {
    let k = 512usize; // 2 superblocks
    let n = 16usize; // 2 column groups of 8
    let nb = k / QK_K;
    let mut next = prng();
    // Weight [n, k] f32 -> Q4_K row-major.
    let wf: Vec<f32> = (0..n * k).map(|_| next() * 0.1).collect();
    let mut wq = vec![BlockQ4K::zeros(); n * nb];
    for row in 0..n {
        BlockQ4K::quantize(
            &wf[row * k..(row + 1) * k],
            &mut wq[row * nb..(row + 1) * nb],
        );
    }
    // Activation row [1, k] -> Q8_K.
    let af: Vec<f32> = (0..k).map(|_| next()).collect();
    let mut aq = vec![BlockQ8K::zeros(); nb];
    BlockQ8K::quantize(&af, &mut aq);

    // Reference: per-column dot.
    let mut reference = vec![0f32; n];
    for (col, r) in reference.iter_mut().enumerate() {
        *r = avx::vec_dot_q4k_q8k(&wq[col * nb..(col + 1) * nb], &aq);
    }

    // Repacked scalar + avx2.
    let x8 = repack_q4k::repack(&wq, n, nb);
    let groups = n / 8;
    let mut out_scalar = vec![0f32; n];
    let mut out_avx = vec![0f32; n];
    for g in 0..groups {
        let bgrp = &x8[g * nb..(g + 1) * nb];
        repack_q4k::gemv_group_scalar(bgrp, &aq, nb, &mut out_scalar[g * 8..g * 8 + 8]);
        #[cfg(target_feature = "avx2")]
        unsafe {
            repack_q4k::gemv_group_avx2(bgrp, &aq, nb, &mut out_avx[g * 8..g * 8 + 8]);
        }
    }

    for col in 0..n {
        let r = reference[col];
        let scale = r.abs().max(1e-3);
        let ds = (r - out_scalar[col]).abs() / scale;
        assert!(
            ds < 1e-4,
            "col {col}: scalar {} vs ref {r} rel {ds}",
            out_scalar[col]
        );
        #[cfg(target_feature = "avx2")]
        {
            let da = (r - out_avx[col]).abs() / scale;
            assert!(
                da < 1e-4,
                "col {col}: avx {} vs ref {r} rel {da}",
                out_avx[col]
            );
        }
    }

    // Unified-layout gate for the in-place repack: the PLAIN-scales GEMV
    // (reading `repack`'s layout) must be BIT-IDENTICAL to the packed v2 GEMV
    // (reading `repack_v2`'s) - same dot math, only the scale/min register
    // assembly differs, so any deviation is a lane-order bug.
    #[cfg(target_feature = "avx2")]
    {
        let x8_v2 = repack_q4k::repack_v2(&wq, n, nb);
        for g in 0..groups {
            let mut out_v2 = [0f32; 8];
            let mut out_plain = [0f32; 8];
            unsafe {
                repack_q4k::gemv_group_avx2_v2(&x8_v2[g * nb..(g + 1) * nb], &aq, nb, &mut out_v2);
                repack_q4k::gemv_group_avx2_plain(
                    &x8[g * nb..(g + 1) * nb],
                    &aq,
                    nb,
                    &mut out_plain,
                );
            }
            for c in 0..8 {
                assert!(
                    out_plain[c].to_bits() == out_v2[c].to_bits(),
                    "group {g} col {c}: plain {} vs v2 {} (not bit-identical)",
                    out_plain[c],
                    out_v2[c]
                );
            }
        }
    }

    // Full matmul driver bit-parity: decode GEMV (m=1) + prefill tiled GEMM
    // across tile boundaries (m=4,7,9 exercise mt=4 + remainder tiles).
    for m in [1usize, 3, 4, 7, 9] {
        let lhs: Vec<f32> = {
            let mut g = prng();
            (0..m * k).map(|_| g()).collect()
        };
        let mut dst_ref = vec![0f32; m * n];
        matmul::<BlockQ4K>((m, k, n), &lhs, &wq, &mut dst_ref).unwrap();
        let mut dst_x8 = vec![0f32; m * n];
        matmul_q4k_repacked((m, k, n), &lhs, &x8, &mut dst_x8).unwrap();
        let mut dst_tiled = vec![0f32; m * n];
        matmul_q4k_repacked_tiled((m, k, n), &lhs, &x8, &mut dst_tiled).unwrap();
        for i in 0..m * n {
            let r = dst_ref[i];
            let rel = (r - dst_x8[i]).abs() / r.abs().max(1e-3);
            assert!(
                rel < 1e-4,
                "m={m} idx {i}: x8 {} vs ref {r} rel {rel}",
                dst_x8[i]
            );
            let relt = (r - dst_tiled[i]).abs() / r.abs().max(1e-3);
            assert!(
                relt < 1e-4,
                "m={m} idx {i}: tiled {} vs ref {r} rel {relt}",
                dst_tiled[i]
            );
        }
    }
}

/// Times the cooperative FFN the decode path actually runs against the three
/// repacked GEMVs the same work would cost on the fast path, at a small dense
/// model's shapes. The profile puts ~81% of CPU decode in this stage, so the
/// ratio between these two numbers is the size of the prize.
///
///   cargo test --release -p loken ffn_coop_vs_repacked -- --ignored --nocapture
#[test]
#[ignore = "benchmark: run explicitly with --ignored --nocapture"]
fn ffn_coop_vs_repacked() {
    let (hidden, inter) = (2048usize, 8192usize);
    let (nbh, nbi) = (hidden / QK_K, inter / QK_K);
    let mut next = prng();
    let mut mk = |rows: usize, k: usize, nb: usize| {
        let wf: Vec<f32> = (0..rows * k).map(|_| next() * 0.1).collect();
        let mut q = vec![BlockQ4K::zeros(); rows * nb];
        for r in 0..rows {
            BlockQ4K::quantize(&wf[r * k..(r + 1) * k], &mut q[r * nb..(r + 1) * nb]);
        }
        q
    };
    let gq = mk(inter, hidden, nbh);
    let uq = mk(inter, hidden, nbh);
    let dq = mk(hidden, inter, nbi);
    let raw = |v: &[BlockQ4K]| -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    };
    let mut x: Vec<f32> = (0..hidden).map(|_| next()).collect();
    let mut nb_buf = vec![0f32; hidden];
    let mut gu_buf = vec![0f32; 2 * inter];
    let mut act_buf = vec![0f32; inter];
    let norm_fn = |src: &[f32], dst: &mut [f32]| dst.copy_from_slice(src);
    let act_fn = |g: &[f32], u: &[f32], o: &mut [f32]| {
        for i in 0..o.len() {
            o[i] = g[i] * u[i];
        }
    };
    let run = |x: &mut Vec<f32>, nb_buf: &mut Vec<f32>, gu: &mut Vec<f32>, ab: &mut Vec<f32>| {
        super::ffn_swiglu_coop(
            x,
            Some((GgmlDType::Q4K, hidden, inter, raw(&gq))),
            (GgmlDType::Q4K, hidden, inter, raw(&uq)),
            (GgmlDType::Q4K, inter, hidden, raw(&dq)),
            None,
            &norm_fn,
            &act_fn,
            nb_buf,
            gu,
            ab,
        )
    };
    match run(&mut x, &mut nb_buf, &mut gu_buf, &mut act_buf) {
        Ok(true) => {}
        other => {
            println!("ffn_swiglu_coop declined ({other:?}); nothing to compare");
            return;
        }
    }
    for _ in 0..3 {
        let _ = run(&mut x, &mut nb_buf, &mut gu_buf, &mut act_buf);
    }
    let mut rounds = Vec::new();
    for _ in 0..5 {
        const IT: usize = 10;
        let t = std::time::Instant::now();
        for _ in 0..IT {
            let _ = run(&mut x, &mut nb_buf, &mut gu_buf, &mut act_buf);
        }
        rounds.push(t.elapsed().as_secs_f64() * 1e6 / IT as f64);
    }
    rounds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let coop = rounds[rounds.len() / 2];

    let gx8 = repack_q4k::repack(&gq, inter, nbh);
    let ux8 = repack_q4k::repack(&uq, inter, nbh);
    let dx8 = repack_q4k::repack(&dq, hidden, nbi);
    let xin: Vec<f32> = (0..hidden).map(|_| 0.01).collect();
    let mid: Vec<f32> = (0..inter).map(|_| 0.01).collect();
    let (mut o1, mut o2, mut o3) = (vec![0f32; inter], vec![0f32; inter], vec![0f32; hidden]);
    let fast = |o1: &mut Vec<f32>, o2: &mut Vec<f32>, o3: &mut Vec<f32>| {
        super::matmul_q4k_plain((1, hidden, inter), &xin, &gx8, o1).unwrap();
        super::matmul_q4k_plain((1, hidden, inter), &xin, &ux8, o2).unwrap();
        super::matmul_q4k_plain((1, inter, hidden), &mid, &dx8, o3).unwrap();
    };
    for _ in 0..5 {
        fast(&mut o1, &mut o2, &mut o3);
    }
    let mut fr = Vec::new();
    for _ in 0..5 {
        const IT: usize = 20;
        let t = std::time::Instant::now();
        for _ in 0..IT {
            fast(&mut o1, &mut o2, &mut o3);
        }
        fr.push(t.elapsed().as_secs_f64() * 1e6 / IT as f64);
    }
    fr.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let rep = fr[fr.len() / 2];
    println!("FFN hidden={hidden} inter={inter}");
    println!("  cooperative (what decode runs): {coop:.1} us");
    println!("  three repacked GEMVs          : {rep:.1} us");
    println!("  ratio                         : {:.1}x", coop / rep);
}
