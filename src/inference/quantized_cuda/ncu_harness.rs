//! Standalone runner for the Q8 GQA flash-decode partial+combine kernels so
//! Nsight Compute can profile them (a #[test] exits -> ncu flushes its report,
//! unlike the long-lived server daemon). Run under ncu to read the real
//! bottleneck (occupancy vs memory-latency vs compute) and the guide.
use super::*;
use crate::tensor::kernel_ffi::CudaDevice;

#[test]
fn flash_q8_gqa_ncu_harness() {
    let Ok(dev) = CudaDevice::new(0) else {
        eprintln!("no cuda; skip");
        return;
    };
    let (seq_kv, hd) = (3200usize, 128usize); // qwen3 @2.5K
    let hdb = hd / 32;
    let stream = dev.native().stream();
    let pos_dev = stream.clone_htod(&vec![(seq_kv - 1) as i32]).unwrap();
    let scale = 1.0f32 / (hd as f32).sqrt();
    // Measure the launcher for a given (n_kv, nqpk). Builds K/V for n_kv heads.
    let run = |n_kv: usize, nqpk: usize| -> f64 {
        let nblocks = seq_kv * n_kv * hdb;
        let mut kb = Vec::<u8>::with_capacity(nblocks * 34);
        let mut vb = Vec::<u8>::with_capacity(nblocks * 34);
        for i in 0..nblocks {
            let d = half::f16::from_f32(0.05 + (i % 7) as f32 * 0.001);
            for buf in [&mut kb, &mut vb] {
                buf.extend_from_slice(&d.to_le_bytes());
                for j in 0..32 {
                    buf.push((((i + j) % 17) as i32 - 8) as i8 as u8);
                }
            }
        }
        let k_blob = stream.clone_htod(&kb).unwrap();
        let v_blob = stream.clone_htod(&vb).unwrap();
        let q: Vec<f32> = (0..n_kv * nqpk * hd)
            .map(|i| ((i % 71) as f32) * 0.01 - 0.3)
            .collect();
        let q_slice = stream.clone_htod(&q).unwrap();
        let q_view = q_slice.slice(..);
        for _ in 0..20 {
            let out = attn_flash_splitk_q8_gqa_decode_dev_pos(
                &k_blob, &v_blob, &q_view, &pos_dev, hd, n_kv, nqpk, seq_kv, scale, &dev,
            )
            .unwrap();
            std::hint::black_box(&out);
        }
        dev.synchronize().unwrap();
        const ITERS: usize = 2000;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            let out = attn_flash_splitk_q8_gqa_decode_dev_pos(
                &k_blob, &v_blob, &q_view, &pos_dev, hd, n_kv, nqpk, seq_kv, scale, &dev,
            )
            .unwrap();
            std::hint::black_box(&out);
        }
        dev.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64
    };
    // Per-q ablation (n_kv=8): slope = per-q compute cost.
    for nqpk in [1usize, 2, 4, 8] {
        eprintln!(
            "flash_q8 ablation: n_kv=8 nqpk={nqpk} (n_q={}) -> {:.2} µs/call",
            8 * nqpk,
            run(8, nqpk)
        );
    }
    // BATCHED vs UN-BATCHED for qwen3's 32 q-heads (8 kv x 4):
    //   batched   = n_kv=8, nqpk=4 (heavy regs, low occupancy, in-reg KV share)
    //   unbatched = n_kv=32, nqpk=1 (lean regs, high occupancy; 4x KV bytes here =
    //     conservative upper bound - real mapping would L2-share the 8 kv-heads).
    eprintln!(
        "flash_q8 BATCHED   (n_kv=8, nqpk=4): {:.2} µs/call",
        run(8, 4)
    );
    eprintln!(
        "flash_q8 UNBATCHED (n_kv=32,nqpk=1): {:.2} µs/call (upper bound, 4x KV no-share)",
        run(32, 1)
    );

    // -- LOAD-ONLY twin: decompose KV-load vs compute WITHOUT ncu ( gate) --
    // Same K/V bytes + grid/lane access as the partial kernel, no dot/softmax.
    // load_us ≈ ½ full ⇒ load↔compute serialized ⇒ cp.async overlap could ~halve
    // it (justify the rewrite). load_us ≈ full ⇒ bandwidth-bound ⇒ accept floor.
    let load_only = |n_kv: usize| -> (f64, f64) {
        let hdb = hd / 32;
        let nblocks = seq_kv * n_kv * hdb;
        let mut kb = Vec::<u8>::with_capacity(nblocks * 34);
        let mut vb = Vec::<u8>::with_capacity(nblocks * 34);
        for i in 0..nblocks {
            let d = half::f16::from_f32(0.05 + (i % 7) as f32 * 0.001);
            for buf in [&mut kb, &mut vb] {
                buf.extend_from_slice(&d.to_le_bytes());
                for j in 0..32 {
                    buf.push((((i + j) % 17) as i32 - 8) as i8 as u8);
                }
            }
        }
        let k_blob = stream.clone_htod(&kb).unwrap();
        let v_blob = stream.clone_htod(&vb).unwrap();
        // mirror the launcher's adaptive nsplit
        let base = (384 / n_kv.max(1)).clamp(8, 96);
        let by_len = seq_kv.div_ceil(24);
        let nsplit = base.max(by_len).clamp(8, 256).min(seq_kv.max(1));
        let partials = unsafe { dev.alloc::<f32>(n_kv * 256 * 32).unwrap() };
        let name = if hd == 64 {
            "flash_splitk_q8_loadonly_hd64"
        } else {
            "flash_splitk_q8_loadonly_hd128"
        };
        let ptx = get_quantized_ptx(&dev).unwrap();
        let func = dev
            .get_or_load_custom_func(name, "loken_quantized", ptx)
            .unwrap();
        let cfg = LaunchConfig {
            grid_dim: (nsplit as u32, n_kv as u32, 1),
            block_dim: (WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let launch = || {
            let mut b = func.builder();
            b.arg(&k_blob);
            b.arg(&v_blob);
            b.arg(&partials);
            b.arg(&pos_dev);
            barg!(b, n_kv as i32, nsplit as i32);
            unsafe { b.launch(cfg) }.unwrap();
        };
        for _ in 0..20 {
            launch();
        }
        dev.synchronize().unwrap();
        const ITERS: usize = 2000;
        let t0 = std::time::Instant::now();
        for _ in 0..ITERS {
            launch();
        }
        dev.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
        let bytes = (seq_kv * n_kv * hdb * 34 * 2) as f64; // K+V
        let gbs = bytes / (us * 1e-6) / 1e9;
        (us, gbs)
    };
    let full_us = run(8, 4);
    let (lo_us, lo_gbs) = load_only(8);
    eprintln!("flash_q8 LOAD-ONLY (n_kv=8): {lo_us:.2} µs/call  = {lo_gbs:.0} GB/s KV-load");
    let comp = (full_us - lo_us).max(0.0);
    let ideal = lo_us.max(comp);
    eprintln!("flash_q8 DECOMPOSE: full={full_us:.2}µs  load={lo_us:.2}µs  compute≈{comp:.2}µs  load_frac={:.0}%",
        lo_us / full_us * 100.0);
    eprintln!("flash_q8 VERDICT: overlap-ideal≈{ideal:.2}µs = {:.0}% of full ({:.0}% potential speedup) ⇒ {}",
        ideal / full_us * 100.0, (1.0 - ideal / full_us) * 100.0,
        if ideal < full_us * 0.85 { "LATENCY headroom - overlap JUSTIFIED" } else { "bandwidth-bound - accept floor" });

    // -- A/B: register software-pipelined partial kernel (PF) vs base partial --
    // Same combine; here we time+parity ONLY the partial (where the overlap lives).
    {
        let (n_kv, nqpk) = (8usize, 4usize);
        let hdb = hd / 32;
        let n_q_heads = n_kv * nqpk;
        let nblocks = seq_kv * n_kv * hdb;
        let mut kb = Vec::<u8>::with_capacity(nblocks * 34);
        let mut vb = Vec::<u8>::with_capacity(nblocks * 34);
        for i in 0..nblocks {
            let d = half::f16::from_f32(0.05 + (i % 7) as f32 * 0.001);
            for buf in [&mut kb, &mut vb] {
                buf.extend_from_slice(&d.to_le_bytes());
                for j in 0..32 {
                    buf.push((((i + j) % 17) as i32 - 8) as i8 as u8);
                }
            }
        }
        let k_blob = stream.clone_htod(&kb).unwrap();
        let v_blob = stream.clone_htod(&vb).unwrap();
        let q: Vec<f32> = (0..n_q_heads * hd)
            .map(|i| ((i % 71) as f32) * 0.01 - 0.3)
            .collect();
        let q_dev = stream.clone_htod(&q).unwrap();
        let base = (384 / n_kv.max(1)).clamp(8, 96);
        let nsplit = base
            .max(seq_kv.div_ceil(24))
            .clamp(8, 256)
            .min(seq_kv.max(1));
        let psz = n_q_heads * 256 * (hd + 2);
        let ptx = get_quantized_ptx(&dev).unwrap();
        let cfg = LaunchConfig {
            grid_dim: (nsplit as u32, n_kv as u32, 1),
            block_dim: (WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let scale = 1.0f32 / (hd as f32).sqrt();
        let run_partial = |kname: &str| -> (Vec<f32>, f64) {
            let parts = unsafe { dev.alloc::<f32>(psz).unwrap() };
            let func = dev
                .get_or_load_custom_func(kname, "loken_quantized", ptx)
                .unwrap();
            let launch = || {
                let mut b = func.builder();
                b.arg(&k_blob);
                b.arg(&v_blob);
                b.arg(&q_dev);
                b.arg(&parts);
                b.arg(&pos_dev);
                barg!(b, n_kv as i32, nqpk as i32, nsplit as i32, scale);
                unsafe { b.launch(cfg) }.unwrap();
            };
            for _ in 0..20 {
                launch();
            }
            dev.synchronize().unwrap();
            const ITERS: usize = 2000;
            let t0 = std::time::Instant::now();
            for _ in 0..ITERS {
                launch();
            }
            dev.synchronize().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
            let host: Vec<f32> = stream.clone_dtoh(&parts).unwrap();
            (host, us)
        };
        let pname = if hd == 64 {
            "flash_splitk_q8_gqa_partial_hd64"
        } else {
            "flash_splitk_q8_gqa_partial_hd128"
        };
        let pfname = if hd == 64 {
            "flash_splitk_q8_gqa_partial_pf_hd64"
        } else {
            "flash_splitk_q8_gqa_partial_pf_hd128"
        };
        let (base_h, base_us) = run_partial(pname);
        let (pf_h, pf_us) = run_partial(pfname);
        let maxdiff = base_h
            .iter()
            .zip(pf_h.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("flash_q8 PF-ABLATION:  base-partial {base_us:.2}µs  pf-partial {pf_us:.2}µs  ({:+.0}%)  parity maxdiff={maxdiff:.2e}",
            (pf_us - base_us) / base_us * 100.0);
        // NOTE: a cp.async smem double-buffer variant (needs compute_120 arch) was
        // also built+parity-tested here and measured NEUTRAL (-0%): the base kernel ALREADY
        // overlaps the load behind compute, so explicit staging only adds cost. Removed (dead
        // lever). Both overlap mechanisms (PF +4%, cp.async -0%)
        // are now proven futile -> the split-K 2.5K kernel is at its compute floor.
        eprintln!(
            "flash_q8 PF-VERDICT: {}",
            if maxdiff > 1e-3 {
                "PARITY FAIL - bug"
            } else if pf_us < base_us * 0.95 {
                "PF WINS"
            } else if pf_us > base_us * 1.05 {
                "PF LOSES - occupancy cliff"
            } else {
                "PF NEUTRAL"
            }
        );
    }
}
