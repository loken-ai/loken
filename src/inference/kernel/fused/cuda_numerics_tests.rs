use super::*;
use crate::tensor::{Device, Tensor};

/// Numerical equivalence of `fused_bias_gelu_new` with the unfused
/// `x.broadcast_add(bias)? .gelu()?` reference. Runs on the first
/// CUDA device that can be opened. If none are usable (CI without
/// GPU, contended VRAM), the test is silently skipped - it must
/// never fail spuriously when CUDA is simply unavailable.
fn try_open_cuda() -> Option<Device> {
    for idx in 0..4 {
        if let Ok(dev) = Device::new_cuda(idx) {
            // Touch the device with a 4-byte allocation to confirm
            // it's actually usable (not just enumerable).
            if Tensor::zeros_on((1,), DType::F32, &dev).is_ok() {
                return Some(dev);
            }
        }
    }
    None
}

#[test]
fn fused_bias_gelu_new_matches_unfused_reference() {
    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };

    // Shape mirrors a typical phi2 decode step: (B=1, T=1, D=2048).
    // D chosen to exercise both the block-shrink path and the modulo
    // bias indexing across multiple rows in larger batches.
    const D: usize = 2048;
    const B: usize = 1;
    const T: usize = 4; // multi-row to verify broadcast across rows

    let x_data: Vec<f32> = (0..B * T * D)
        .map(|i| ((i as f32) * 0.0137).sin() * 3.0)
        .collect();
    let bias_data: Vec<f32> = (0..D).map(|i| ((i as f32) * 0.0241).cos() * 0.5).collect();

    let x = Tensor::from_slice(&x_data, (B, T, D), &dev).unwrap();
    let bias = Tensor::from_slice(&bias_data, (D,), &dev).unwrap();

    // Reference
    let ref_out = x.broadcast_add(&bias).unwrap().gelu().unwrap();
    let ref_v: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();

    // Fused
    let fused_out = fused_bias_gelu_new(&x, &bias).unwrap();
    let fused_v: Vec<f32> = fused_out.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(ref_v.len(), fused_v.len(), "length mismatch");

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (r, f) in ref_v.iter().zip(fused_v.iter()) {
        let abs = (r - f).abs();
        max_abs = max_abs.max(abs);
        let denom = r.abs().max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    // GELU's tanh-approx isn't bit-exact between the reference CPU/GPU
    // implementation and our raw kernel - tanhf vs. high-precision
    // tanh differ in the last 2-3 ULPs. 1e-5 absolute is comfortable.
    assert!(
        max_abs < 1e-5,
        "fused vs unfused diverged: max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
    );
}

/// Fallback (CPU / non-F32) path must not panic and must produce the
/// same result as `x.broadcast_add(bias)?.gelu()?` - i.e., the
/// wrapper must yield to the unfused chain rather than misroute.
#[test]
fn fused_bias_gelu_new_falls_back_on_cpu() {
    let dev = Device::Cpu;
    let x = Tensor::from_slice(&[1.0f32, 2.0, -1.0, 0.5], (1, 1, 4), &dev).unwrap();
    let bias = Tensor::from_slice(&[0.1f32, -0.1, 0.0, 0.2], (4,), &dev).unwrap();
    let out = fused_bias_gelu_new(&x, &bias).unwrap();
    let want = x.broadcast_add(&bias).unwrap().gelu().unwrap();
    let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    let want_v: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
    for (a, b) in out_v.iter().zip(want_v.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }
}

/// fused rms_norm GPU parity at cols=896 (qwen2.5:0.5b's hidden,
/// = 128x7, not a power-of-2 / not 256-multiple). Covers BOTH the wide path
/// (rows<=64 -> block_dim = 896.next_power_of_two() = 1024) and the narrow
/// path (rows>64 -> block 256), against an f64 reference.
#[test]
fn fused_rmsnorm_f32_parity_cols896() {
    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };
    let cols = 896usize;
    let eps = 1e-6f32;
    let weight: Vec<f32> = (0..cols).map(|i| 0.5 + ((i % 13) as f32) * 0.05).collect();
    for rows in [1usize, 100usize] {
        let x: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 97) as f32) * 0.021 - 1.0)
            .collect();
        let xt = Tensor::from_slice(&x, (rows, cols), &dev).unwrap();
        let wt = Tensor::from_slice(&weight, (cols,), &dev).unwrap();
        let got: Vec<f32> = fused_rmsnorm_f32(&xt, &wt, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let mut max_abs = 0f64;
        for r in 0..rows {
            let mut ss = 0f64;
            for j in 0..cols {
                let v = x[r * cols + j] as f64;
                ss += v * v;
            }
            let rms = 1.0 / ((ss / cols as f64) + eps as f64).sqrt();
            for j in 0..cols {
                let want = x[r * cols + j] as f64 * rms * weight[j] as f64;
                let g = got[r * cols + j] as f64;
                let d = (g - want).abs();
                if d > max_abs {
                    max_abs = d;
                }
                assert!(
                    d <= 1e-4 + 1e-3 * want.abs(),
                    "rmsnorm cols=896 rows={rows} [{r},{j}]: {g} vs {want}"
                );
            }
        }
        eprintln!(
            "rmsnorm cols=896 rows={rows} ({}) parity OK max_abs={max_abs:.2e}",
            if rows <= 64 {
                "wide/block1024"
            } else {
                "narrow/block256"
            }
        );
    }
}

#[test]
fn fused_phi2_residual_merge_matches_unfused_reference() {
    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };

    const D: usize = 2048;
    const B: usize = 1;
    const T: usize = 4;

    let residual_data: Vec<f32> = (0..B * T * D)
        .map(|i| ((i as f32) * 0.0173).sin() * 0.5)
        .collect();
    let attn_data: Vec<f32> = (0..B * T * D)
        .map(|i| ((i as f32) * 0.0211).cos() * 0.3)
        .collect();
    let attn_bias_data: Vec<f32> = (0..D).map(|i| ((i as f32) * 0.0123).sin() * 0.1).collect();
    let ffn_data: Vec<f32> = (0..B * T * D)
        .map(|i| ((i as f32) * 0.0287).sin() * 0.4)
        .collect();
    let ffn_bias_data: Vec<f32> = (0..D).map(|i| ((i as f32) * 0.0341).cos() * 0.2).collect();

    let residual = Tensor::from_slice(&residual_data, (B, T, D), &dev).unwrap();
    let attn_out = Tensor::from_slice(&attn_data, (B, T, D), &dev).unwrap();
    let attn_bias = Tensor::from_slice(&attn_bias_data, (D,), &dev).unwrap();
    let ffn_out = Tensor::from_slice(&ffn_data, (B, T, D), &dev).unwrap();
    let ffn_bias = Tensor::from_slice(&ffn_bias_data, (D,), &dev).unwrap();

    // Reference: same op order the wiring will replace.
    let attn_biased = attn_out.broadcast_add(&attn_bias).unwrap();
    let ffn_biased = ffn_out.broadcast_add(&ffn_bias).unwrap();
    let ref_out = ((&residual + &attn_biased).unwrap() + &ffn_biased).unwrap();
    let ref_v: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();

    // Fused
    let fused_out =
        fused_phi2_residual_merge(&residual, &attn_out, &attn_bias, &ffn_out, &ffn_bias).unwrap();
    let fused_v: Vec<f32> = fused_out.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(ref_v.len(), fused_v.len(), "length mismatch");
    let mut max_abs = 0.0f32;
    for (r, f) in ref_v.iter().zip(fused_v.iter()) {
        max_abs = max_abs.max((r - f).abs());
    }
    // Floating-point addition is associative for the same exact
    // value sequence, so this should be bit-identical. Allow 1
    // ULP wiggle for compiler reordering: kernel does
    // `r + a + ab + f + fb`, reference does
    // `(r + (a + ab)) + (f + fb)`. The associativity break can
    // surface in the last bit of mantissa for extreme values.
    assert!(
        max_abs < 1e-5,
        "fused vs unfused diverged: max_abs={max_abs:.3e}",
    );
}

/// `fused_rmsnorm_then_add` must match `rms_norm(x, w, eps) + residual`
/// bit-for-bit (well, ~1 ULP - reduction order can differ slightly).
/// Gemma4 post_ffn_norm replacement depends on this. A divergence
/// here means generated tokens may drift from the reference path.
#[test]
fn fused_rmsnorm_then_add_matches_unfused_reference() {
    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };

    // Gemma4:latest hidden=2560 - exercises the typical decode shape
    // (B=1, T=1, D=2560).
    const D: usize = 2560;
    const B: usize = 1;
    const T: usize = 1;

    let x_data: Vec<f32> = (0..B * T * D)
        .map(|i| ((i as f32) * 0.0119).sin() * 2.0)
        .collect();
    let w_data: Vec<f32> = (0..D)
        .map(|i| 0.8 + ((i as f32) * 0.011).cos() * 0.15)
        .collect();
    let res_data: Vec<f32> = (0..B * T * D)
        .map(|i| ((i as f32) * 0.0173).cos() * 0.5)
        .collect();

    let x = Tensor::from_slice(&x_data, (B, T, D), &dev).unwrap();
    let w = Tensor::from_slice(&w_data, (D,), &dev).unwrap();
    let residual = Tensor::from_slice(&res_data, (B, T, D), &dev).unwrap();

    let eps = 1e-6_f32;
    let ref_out = (crate::tensor::ops::rms_norm(&x, &w, eps).unwrap() + &residual).unwrap();
    let ref_v: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();

    let fused_out = fused_rmsnorm_then_add(&x, &w, &residual, eps).unwrap();
    let fused_v: Vec<f32> = fused_out.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(ref_v.len(), fused_v.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (r, f) in ref_v.iter().zip(fused_v.iter()) {
        let abs = (r - f).abs();
        max_abs = max_abs.max(abs);
        let denom = r.abs().max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    // RMS reduction over 2560 elements can differ from the substrate's
    // (which may use a different tree shape) by up to a few ULPs in
    // the final divide. 5e-5 absolute / 1e-4 relative is comfortable.
    assert!(
        max_abs < 5e-5 && max_rel < 1e-4,
        "fused vs unfused diverged: max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
    );
}

/// `fused_rmsnorm_add_scale` must match the 3-launch chain
/// `(rms_norm(x, w, eps) + residual).broadcast_mul(scale)` within ~1 ULP.
/// This is the kernel used at gemma4's PLE block tail; a divergence
/// here would produce silent token drift across all gemma4 layers.
#[test]
fn fused_rmsnorm_add_scale_matches_unfused_reference() {
    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };

    const D: usize = 2560;
    const B: usize = 1;
    const T: usize = 1;

    let x_data: Vec<f32> = (0..B * T * D)
        .map(|i| ((i as f32) * 0.0091).sin() * 1.7)
        .collect();
    let w_data: Vec<f32> = (0..D)
        .map(|i| 0.75 + ((i as f32) * 0.0087).cos() * 0.18)
        .collect();
    let res_data: Vec<f32> = (0..B * T * D)
        .map(|i| ((i as f32) * 0.0145).cos() * 0.4)
        .collect();
    // Layer scale values in gemma4 are typically 0.02-0.80; sample
    // a range that exercises the multiplicative reduction.
    let scale_data: Vec<f32> = (0..D)
        .map(|i| 0.05 + ((i as f32) * 0.013).sin().abs() * 0.7)
        .collect();

    let x = Tensor::from_slice(&x_data, (B, T, D), &dev).unwrap();
    let w = Tensor::from_slice(&w_data, (D,), &dev).unwrap();
    let residual = Tensor::from_slice(&res_data, (B, T, D), &dev).unwrap();
    let scale = Tensor::from_slice(&scale_data, (D,), &dev).unwrap();

    let eps = 1e-6_f32;
    let ref_out = (crate::tensor::ops::rms_norm(&x, &w, eps).unwrap() + &residual)
        .unwrap()
        .broadcast_mul(&scale)
        .unwrap();
    let ref_v: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();

    let fused_out = fused_rmsnorm_add_scale(&x, &w, &residual, &scale, eps).unwrap();
    let fused_v: Vec<f32> = fused_out.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(ref_v.len(), fused_v.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (r, f) in ref_v.iter().zip(fused_v.iter()) {
        let abs = (r - f).abs();
        max_abs = max_abs.max(abs);
        let denom = r.abs().max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    assert!(
        max_abs < 5e-5 && max_rel < 1e-4,
        "fused_rmsnorm_add_scale vs unfused diverged: max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
    );
}

/// `fused_split_gelu_mul` operates on a packed [B, T, 2N] tensor (the
/// gate||up concatenation that fused_ffn_gate_up models produce) and
/// emits [B, T, N] = gelu_tanh(gate) * up - no intermediate narrow +
/// contiguous copies. Must match the unfused
/// `gate.gelu()?.mul(&up)?` reference within ~1 ULP.
#[test]
fn fused_split_gelu_mul_matches_unfused_reference() {
    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };

    // Gemma4:latest intermediate=10240 - exercises decode shape.
    const N: usize = 10_240;
    const B: usize = 1;
    const T: usize = 1;

    // Packed [B, T, 2N] = gate first, then up.
    let mut data = Vec::with_capacity(B * T * 2 * N);
    for i in 0..(B * T * N) {
        data.push(((i as f32) * 0.0027).sin() * 1.4); // gate half
    }
    for i in 0..(B * T * N) {
        data.push(((i as f32) * 0.0019).cos() * 1.1); // up half
    }
    let gu = Tensor::from_slice(&data, (B, T, 2 * N), &dev).unwrap();

    // Reference: narrow + narrow + gelu(tanh) * mul.
    let gate_ref = gu.narrow(crate::tensor::D::Minus1, 0, N).unwrap();
    let up_ref = gu.narrow(crate::tensor::D::Minus1, N, N).unwrap();
    let ref_out = (gate_ref.gelu().unwrap() * &up_ref).unwrap();
    let ref_v: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();

    let fused_out = fused_split_gelu_mul(&gu).unwrap();
    let fused_v: Vec<f32> = fused_out.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(fused_v.len(), ref_v.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (r, f) in ref_v.iter().zip(fused_v.iter()) {
        let abs = (r - f).abs();
        max_abs = max_abs.max(abs);
        let denom = r.abs().max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    assert!(
        max_abs < 5e-5 && max_rel < 1e-4,
        "fused_split_gelu_mul vs unfused diverged: max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
    );
}

/// SiLU sister test - `fused_split_silu_mul` must match
/// `silu(narrow_gate) * narrow_up` within ~1 ULP. Same packing
/// invariant as the GELU variant.
#[test]
fn fused_split_silu_mul_matches_unfused_reference() {
    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };

    // qwen2-ish intermediate=18944, but use a smaller test shape.
    const N: usize = 8_192;
    const B: usize = 1;
    const T: usize = 1;

    let mut data = Vec::with_capacity(B * T * 2 * N);
    for i in 0..(B * T * N) {
        data.push(((i as f32) * 0.0031).sin() * 1.6);
    }
    for i in 0..(B * T * N) {
        data.push(((i as f32) * 0.0023).cos() * 1.2);
    }
    let gu = Tensor::from_slice(&data, (B, T, 2 * N), &dev).unwrap();

    let gate_ref = gu.narrow(crate::tensor::D::Minus1, 0, N).unwrap();
    let up_ref = gu.narrow(crate::tensor::D::Minus1, N, N).unwrap();
    let ref_out = (&up_ref * crate::tensor::ops::silu(&gate_ref).unwrap()).unwrap();
    let ref_v: Vec<f32> = ref_out.flatten_all().unwrap().to_vec1().unwrap();

    let fused_out = fused_split_silu_mul(&gu).unwrap();
    let fused_v: Vec<f32> = fused_out.flatten_all().unwrap().to_vec1().unwrap();

    assert_eq!(fused_v.len(), ref_v.len());
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (r, f) in ref_v.iter().zip(fused_v.iter()) {
        let abs = (r - f).abs();
        max_abs = max_abs.max(abs);
        let denom = r.abs().max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    assert!(
        max_abs < 5e-5 && max_rel < 1e-4,
        "fused_split_silu_mul vs unfused diverged: max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
    );
}

/// End-to-end Path B contract: the u32-output kernel result, wrapped
/// via `CudaStorage::wrap_cuda_slice` into a U32 Tensor, must drive
/// `Embedding::forward` to the same row vector that a host
/// `Tensor::new(&[next_token], dev)` lookup produces. This is the
/// chain Path B step 3 will rely on to skip the
/// per-token memcpy_dtoh sync.
#[test]
fn fused_penalty_argmax_u32_drives_embedding_forward() {
    use crate::tensor::DType;

    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };

    const VOCAB: usize = 1024;
    const HIDDEN: usize = 64;
    let logits_data: Vec<f32> = (0..VOCAB)
        .map(|i| ((i as f32) * 0.013).sin() * 4.0 + (i as f32) * 0.001)
        .collect();
    let logits = Tensor::from_slice(&logits_data, (1, 1, VOCAB), &dev).unwrap();
    let emb_table_data: Vec<f32> = (0..VOCAB * HIDDEN)
        .map(|i| ((i as f32) * 0.0017).cos())
        .collect();
    let emb_weight = Tensor::from_slice(&emb_table_data, (VOCAB, HIDDEN), &dev).unwrap();
    let embedding = crate::tensor::layer::Embedding::new(emb_weight);

    let (host_tok, dev_slice) = fused_penalty_argmax_u32_with_device(&logits, &[], 1.0).unwrap();

    // Reference path: host token -> Tensor::new on device -> forward.
    let host_idx = Tensor::new(&[host_tok], &dev).unwrap();
    let host_row = embedding.forward(&host_idx).unwrap();

    // Device path: wrap kernel's CudaSlice<u32> directly as a Tensor.
    let cuda_dev = dev.as_cuda_device().unwrap().clone();
    let dev_storage = crate::tensor::cuda_ext::CudaStorage::wrap_cuda_slice(dev_slice, cuda_dev);
    let dev_tensor = crate::tensor::cuda_ext::tensor_from_cuda_storage(dev_storage, (1,)).unwrap();
    assert_eq!(dev_tensor.dtype(), DType::U32);
    let dev_row = embedding.forward(&dev_tensor).unwrap();

    let host_v: Vec<f32> = host_row.flatten_all().unwrap().to_vec1().unwrap();
    let dev_v: Vec<f32> = dev_row.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(host_v.len(), dev_v.len(), "row length mismatch");
    for (h, d) in host_v.iter().zip(dev_v.iter()) {
        assert_eq!(
            h, d,
            "embedding row diverged between host and device lookup"
        );
    }
}

/// `fused_penalty_argmax_u32_with_device` must return the same token
/// as the i32-output variant. The penalty
/// application and reduction are bit-identical; only the output type
/// differs. A divergence here would mean the kernel macro hash was
/// off (cached PTX served the wrong fn).
#[test]
fn fused_penalty_argmax_u32_matches_i32_variant() {
    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };

    const VOCAB: usize = 32_000;
    let logits_data: Vec<f32> = (0..VOCAB)
        .map(|i| ((i as f32) * 0.00073).sin() * 5.0 + ((i % 137) as f32) * 0.01)
        .collect();
    let logits = Tensor::from_slice(&logits_data, (1, 1, VOCAB), &dev).unwrap();
    let penalties: Vec<u32> = vec![100, 101, 102, 7777, 12345];

    let (tok_i32, _) = fused_penalty_argmax_with_device(&logits, &penalties, 1.1).unwrap();
    let (tok_u32, _) = fused_penalty_argmax_u32_with_device(&logits, &penalties, 1.1).unwrap();
    assert_eq!(
        tok_i32, tok_u32,
        "u32 variant returned different token than i32 variant"
    );
}

#[test]
fn fused_phi2_residual_merge_falls_back_on_cpu() {
    let dev = Device::Cpu;
    let r = Tensor::from_slice(&[1.0f32, 2.0, -1.0, 0.5], (1, 1, 4), &dev).unwrap();
    let a = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], (1, 1, 4), &dev).unwrap();
    let ab = Tensor::from_slice(&[0.01f32, -0.01, 0.0, 0.02], (4,), &dev).unwrap();
    let f = Tensor::from_slice(&[-0.5f32, 0.5, 1.5, -1.5], (1, 1, 4), &dev).unwrap();
    let fb = Tensor::from_slice(&[0.0f32, 0.0, 0.1, -0.1], (4,), &dev).unwrap();
    let out = fused_phi2_residual_merge(&r, &a, &ab, &f, &fb).unwrap();
    let want = {
        let s1 = a.broadcast_add(&ab).unwrap();
        let s2 = f.broadcast_add(&fb).unwrap();
        ((&r + &s1).unwrap() + &s2).unwrap()
    };
    let o: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
    for (oo, ww) in o.iter().zip(w.iter()) {
        assert!((oo - ww).abs() < 1e-6, "{oo} vs {ww}");
    }
}
