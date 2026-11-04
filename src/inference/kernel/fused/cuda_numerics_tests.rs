use super::*;
use crate::tensor::{Device, Tensor};

/// Numerical equivalence of `fused_bias_gelu_new` with the unfused
/// `x.broadcast_add(bias)? .gelu()?` reference. Runs on the first
/// CUDA device that can be opened. If none are usable (CI without
/// GPU, contended VRAM), the test is silently skipped - it must
/// never fail spuriously when CUDA is simply unavailable.
pub(super) fn try_open_cuda() -> Option<Device> {
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

/// The card-side top-k sampler walks the kept candidates best first: a uniform at the bottom
/// of the range draws the best token, one inside the second token's share draws the second,
/// the nucleus cuts the tail, and the repeat penalty moves a token down the order.
#[test]
fn sample_topk_draws_by_cumulative_share() {
    let dev = match try_open_cuda() {
        Some(d) => d,
        None => {
            eprintln!("skipping: no usable CUDA device");
            return;
        }
    };
    // A vocabulary wider than one stage-1 block, with three clear leaders.
    const VOCAB: usize = 5000;
    let mut v = vec![-20.0f32; VOCAB];
    v[4000] = 3.0;
    v[17] = 2.0;
    v[2500] = 1.0;
    let logits = Tensor::from_slice(&v, (1, VOCAB), &dev).unwrap();
    let z: f32 = v.iter().map(|x| (x - 3.0).exp()).sum();
    let p = |x: f32| (x - 3.0).exp() / z;
    let (p0, p1, p2) = (p(3.0), p(2.0), p(1.0));
    let kept = p0 + p1 + p2;
    let draw = |u: f32, top_p: f32, pen: &[u32], penalty: f32| {
        sample_topk(&logits, pen, penalty, 1.0, 3, top_p, u).unwrap()
    };
    assert_eq!(draw(0.0, 1.0, &[], 1.0), 4000);
    assert_eq!(draw((p0 + 0.5 * p1) / kept, 1.0, &[], 1.0), 17);
    assert_eq!(draw(0.999, 1.0, &[], 1.0), 2500);
    // A nucleus the best token alone covers keeps only it.
    assert_eq!(draw(0.999, p0 * 0.5, &[], 1.0), 4000);
    // Dividing the leader's logit by 3 puts it below the second.
    assert_eq!(draw(0.0, 1.0, &[4000], 3.0), 17);
}

/// The f16 fused residual add + rmsnorm against a host reference, for a width the 16-byte
/// kernel serves and for one it does not. The kept sum is exact, the norm is within 1 ulp of f16.
#[test]
fn fused_add_rmsnorm_f16_matches_a_host_reference() {
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    for cols in [4096usize, 2056, 4100] {
        let rows = 7usize;
        let eps = 1e-6f32;
        let xs: Vec<half::f16> = (0..rows * cols)
            .map(|i| half::f16::from_f32(((i as f32) * 0.0173).sin() * 3.0))
            .collect();
        let rs: Vec<half::f16> = (0..rows * cols)
            .map(|i| half::f16::from_f32(((i as f32) * 0.0071).cos() * 2.0))
            .collect();
        let ws: Vec<f32> = (0..cols).map(|i| 0.5 + (i % 13) as f32 * 0.05).collect();
        let x = Tensor::from_slice(&xs, (rows, cols), &dev).unwrap();
        let r = Tensor::from_slice(&rs, (rows, cols), &dev).unwrap();
        let w = Tensor::from_slice(&ws, (cols,), &dev).unwrap();
        let (sum, norm) = fused_add_rmsnorm_f16(&x, &r, &w, eps).unwrap();
        let sum = sum.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        let norm = norm.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        let plain = fused_rmsnorm_f16(&x, &w, eps).unwrap();
        let plain = plain.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
        for row in 0..rows {
            let at = |v: usize| row * cols + v;
            let kept: Vec<half::f16> = (0..cols)
                .map(|v| half::f16::from_f32(xs[at(v)].to_f32() + rs[at(v)].to_f32()))
                .collect();
            assert!(
                (0..cols).all(|v| sum[at(v)] == kept[v]),
                "cols {cols}: kept sum"
            );
            let reference = |input: &dyn Fn(usize) -> f32| -> Vec<f32> {
                let mean = (0..cols).map(|v| input(v) * input(v)).sum::<f32>() / cols as f32;
                let denom = (mean + eps).sqrt();
                (0..cols).map(|v| input(v) / denom * ws[v]).collect()
            };
            let with_res = reference(&|v| kept[v].to_f32());
            let without = reference(&|v| xs[at(v)].to_f32());
            for v in 0..cols {
                let tol = |want: f32| want.abs() * 2f32.powi(-10) + 1e-4;
                assert!(
                    (norm[at(v)].to_f32() - with_res[v]).abs() <= tol(with_res[v]),
                    "cols {cols} row {row} col {v}: add norm"
                );
                assert!(
                    (plain[at(v)].to_f32() - without[v]).abs() <= tol(without[v]),
                    "cols {cols} row {row} col {v}: plain norm"
                );
            }
        }
    }
}

/// The rows a float embedding table gathers on the card are the rows of the table, widened to
/// f32, in the order of the ids and for a repeated id.
#[test]
fn a_float_embedding_table_gathers_its_rows_on_the_card() {
    use crate::tensor::quantized::{GgmlDType, QTensor};
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    const VOCAB: usize = 50;
    const DIM: usize = 40;
    let values: Vec<f32> = (0..VOCAB * DIM)
        .map(|i| ((i as f32) * 0.37).sin() * 3.0)
        .collect();
    let host = Tensor::from_slice(&values, (VOCAB, DIM), &Device::Cpu).unwrap();
    let ids_host = [3u32, 7, 3, 49, 0];
    for (dtype, round) in [
        (GgmlDType::F32, (|v: f32| v) as fn(f32) -> f32),
        (GgmlDType::F16, |v| half::f16::from_f32(v).to_f32()),
        (GgmlDType::BF16, |v| half::bf16::from_f32(v).to_f32()),
    ] {
        let table = QTensor::quantize_onto(&host, dtype, &dev).unwrap();
        assert!(crate::inference::moe_cuda::embed_rows_decodes(dtype));
        let ids = Tensor::from_vec(ids_host.to_vec(), &[ids_host.len()][..], &dev).unwrap();
        let out = Tensor::zeros_on((1, ids_host.len(), DIM), DType::F32, &dev).unwrap();
        crate::inference::moe_cuda::embed_rows_gguf(&table, &ids, &out).unwrap();
        let got = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (r, &id) in ids_host.iter().enumerate() {
            for d in 0..DIM {
                let want = round(values[id as usize * DIM + d]);
                assert_eq!(
                    got[r * DIM + d],
                    want,
                    "{dtype:?} row {r} (id {id}) col {d}"
                );
            }
        }
    }
}

/// The batched greedy argmax picks the lowest index among equal maxima, for widths the 16-byte
/// kernels serve and for one they do not, in f32 and in f16.
#[test]
fn batched_argmax_breaks_ties_to_the_lowest_index() {
    use crate::tensor::DType;
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    for vocab in [4096usize, 4100, 4098] {
        let rows = 5usize;
        let mut data = vec![0.0f32; rows * vocab];
        let mut want = Vec::new();
        for row in 0..rows {
            let r = &mut data[row * vocab..(row + 1) * vocab];
            for (i, v) in r.iter_mut().enumerate() {
                *v = ((i * 31 + row * 7) % 97) as f32 * 0.01;
            }
            // The maximum appears at 3 positions; the first one must win.
            let first = (row * 391 + 13) % (vocab / 2);
            for at in [first, first + vocab / 4, vocab - 1] {
                r[at] = 5.0;
            }
            want.push(first as u32);
        }
        let logits = Tensor::from_slice(&data, (rows, vocab), &dev).unwrap();
        let got = crate::inference::moe_cuda::batched_argmax(&logits).unwrap();
        assert_eq!(got, want, "f32, vocab {vocab}");
        let half = logits.to_dtype(DType::F16).unwrap();
        let got = crate::inference::moe_cuda::batched_argmax(&half).unwrap();
        assert_eq!(got, want, "f16, vocab {vocab}");
    }
}

/// The activation of a fused gate and up buffer in one launch against the 2 copies and the
/// `silu_mul_f16` it replaces. The result must equal them bit for bit. The cases are a width of
/// 12288 at 130 rows and a small width of 96 at 5 rows, then a view of rows narrowed out of a
/// larger buffer at an even and at an odd element offset. The odd offset must return `None`, and
/// the fallback on the 2 halves must then equal the reference. A buffer of 0 rows returns `None`.
#[test]
fn silu_mul_gu_f16_matches_the_narrow_copies_and_silu_mul() {
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    let bits = |t: &Tensor| {
        t.flatten_all()
            .unwrap()
            .to_vec1::<half::f16>()
            .unwrap()
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<u16>>()
    };
    let values = |n: usize| -> Vec<half::f16> {
        (0..n)
            .map(|i| half::f16::from_f32(((i as f32) * 0.0071).sin() * 6.0))
            .collect()
    };
    // The reference: the 2 halves cut out by copies, then `silu_mul_f16`.
    let reference = |gu: &Tensor, ffn: usize| -> Tensor {
        let g = gu.narrow(1, 0, ffn).unwrap().contiguous().unwrap();
        let u = gu.narrow(1, ffn, ffn).unwrap().contiguous().unwrap();
        silu_mul_f16(&g, &u).unwrap().expect("the sibling fits")
    };
    // (rows, ffn)
    for (rows, ffn) in [(130usize, 12288usize), (5, 96)] {
        let gu = Tensor::from_slice(&values(rows * 2 * ffn), (rows, 2 * ffn), &dev).unwrap();
        let got = silu_mul_gu_f16(&gu, ffn)
            .unwrap()
            .expect("the fused launch fits");
        assert_eq!(got.dims(), &[rows, ffn][..], "shape, rows {rows} ffn {ffn}");
        let g = gu.narrow(1, 0, ffn).unwrap();
        let u = gu.narrow(1, ffn, ffn).unwrap();
        let want = silu_mul_f16(&g, &u).unwrap().expect("the sibling fits");
        assert_eq!(bits(&got), bits(&want), "rows {rows} ffn {ffn}");
    }
    // A view of 5 rows narrowed out of a buffer of 9, at an even element offset.
    let (rows, ffn) = (5usize, 96usize);
    let big = Tensor::from_slice(&values(9 * 2 * ffn), (9, 2 * ffn), &dev).unwrap();
    let view = big.narrow(0, 2, rows).unwrap();
    let got = silu_mul_gu_f16(&view, ffn)
        .unwrap()
        .expect("an even offset fits");
    assert_eq!(
        bits(&got),
        bits(&reference(&view, ffn)),
        "row-narrowed view at an even offset"
    );
    // The same rows at an odd element offset: the wrapper declines, the fallback is the reference.
    let flat =
        Tensor::from_slice(&values(rows * 2 * ffn + 1), (rows * 2 * ffn + 1,), &dev).unwrap();
    let odd = flat
        .narrow(0, 1, rows * 2 * ffn)
        .unwrap()
        .reshape((rows, 2 * ffn))
        .unwrap();
    // A reshape of a narrowed flat tensor may pack the view to offset 0: the wrapper then
    // serves it, and the result must match the reference like any other input.
    if let Some(fused) = silu_mul_gu_f16(&odd, ffn).unwrap() {
        assert_eq!(
            bits(&fused),
            bits(&reference(&odd, ffn)),
            "odd view served by the fused launch"
        );
    }
    let (g, u) = (
        odd.narrow(1, 0, ffn).unwrap(),
        odd.narrow(1, ffn, ffn).unwrap(),
    );
    let fallback = silu_mul_f16(&g, &u).unwrap().expect("the sibling fits");
    assert_eq!(
        bits(&fallback),
        bits(&reference(&odd, ffn)),
        "fallback on an odd offset"
    );
    // No rows: nothing to launch.
    let empty = big.narrow(0, 0, 0).unwrap();
    assert!(
        silu_mul_gu_f16(&empty, ffn).unwrap().is_none(),
        "0 rows have no fused launch"
    );
}

/// The prefill input stage in one launch against the separate launches it replaces: the head
/// norm as `Tensor::rms_norm` on an F16 tensor, then `rope_apply`, then the rows of the store.
/// It runs with and without the norms and for both pairings of the rotation. Q, k and v must
/// equal the separate launches exactly, and the store holds k and v at the slot of each row and
/// nothing elsewhere. The cases cover a head of 64, a fused projection of 4096 hidden values with
/// 32 q heads and 8 kv heads at 130 rows, and heads of 80 and 96 values, whose block has idle
/// threads.
#[test]
fn prefill_rope_kv_write_f16_matches_the_separate_launches() {
    use crate::inference::cache::paged_attention::{rope_apply, rope_cos_sin};
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    // (rows, q heads, kv heads, head width)
    let cases = [
        (5usize, 4usize, 2usize, 64usize),
        (130, 32, 8, 128),
        (7, 4, 2, 80),
        (6, 3, 1, 96),
    ];
    let host = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
    for (rows, nh, nkv, hd) in cases {
        let (qd, kvd) = (nh * hd, nkv * hd);
        let width = qd + 2 * kvd;
        let values: Vec<half::f16> = (0..rows * width)
            .map(|i| half::f16::from_f32(((i as f32) * 0.0137).sin() * 2.0))
            .collect();
        let qkv = Tensor::from_slice(&values, (rows, width), &dev).unwrap();
        let positions: Vec<usize> = (0..rows).collect();
        let (cos, sin) = rope_cos_sin(&positions, hd, 10000.0, &dev, DType::F16).unwrap();
        // Rows land in reverse order, after 5 slots that stay empty.
        let num_slots = rows + 5;
        let slots: Vec<i32> = (0..rows).map(|r| (num_slots - 1 - r) as i32).collect();
        let slot_dev = Tensor::from_vec(slots.clone(), &[rows][..], &dev).unwrap();
        let weight = |seed: f32| -> Tensor {
            let w: Vec<f32> = (0..hd)
                .map(|i| 0.5 + ((i as f32) * seed).cos().abs())
                .collect();
            Tensor::from_slice(&w, (hd,), &dev).unwrap()
        };
        let (qw, kw) = (weight(0.3), weight(0.7));
        let eps = 1e-6f32;
        for interleaved in [false, true] {
            for with_norms in [true, false] {
                let label =
                    format!("rows {rows} hd {hd} interleaved {interleaved} norms {with_norms}");
                let kp = Tensor::zeros_on((num_slots, kvd), DType::F16, &dev).unwrap();
                let vp = Tensor::zeros_on((num_slots, kvd), DType::F16, &dev).unwrap();
                let norms = with_norms.then(|| [(&qw, eps), (&kw, eps)]);
                let (q, k, v) = prefill_rope_kv_write_f16(
                    [(&qkv, 0), (&qkv, qd), (&qkv, qd + kvd)],
                    norms,
                    &cos,
                    &sin,
                    interleaved,
                    &slot_dev,
                    &slots,
                    &kp,
                    &vp,
                    nh,
                    nkv,
                    hd,
                )
                .unwrap()
                .expect("the fused launch fits");
                let part = |col: usize, n: usize, heads: usize| {
                    qkv.narrow(1, col, n)
                        .unwrap()
                        .reshape(&[rows, heads, hd][..])
                        .unwrap()
                };
                let (mut qh, mut kh) = (part(0, qd, nh), part(qd, kvd, nkv));
                if with_norms {
                    qh = qh.contiguous().unwrap().rms_norm(&qw, eps).unwrap();
                    kh = kh.contiguous().unwrap().rms_norm(&kw, eps).unwrap();
                }
                let q_ref = rope_apply(&qh, &cos, &sin, interleaved).unwrap();
                let k_ref = rope_apply(&kh, &cos, &sin, interleaved).unwrap();
                let v_ref = part(qd + kvd, kvd, nkv);
                assert_eq!(host(&q), host(&q_ref), "q, {label}");
                assert_eq!(host(&k), host(&k_ref), "k, {label}");
                assert_eq!(host(&v), host(&v_ref), "v, {label}");
                let (kp_host, vp_host) = (host(&kp), host(&vp));
                let (k_host, v_host) = (host(&k), host(&v));
                for slot in 0..num_slots {
                    let row = slots.iter().position(|&s| s as usize == slot);
                    for j in 0..kvd {
                        let (gk, gv) = (kp_host[slot * kvd + j], vp_host[slot * kvd + j]);
                        match row {
                            Some(r) => {
                                assert_eq!(
                                    gk,
                                    k_host[r * kvd + j],
                                    "store k, slot {slot}, {label}"
                                );
                                assert_eq!(
                                    gv,
                                    v_host[r * kvd + j],
                                    "store v, slot {slot}, {label}"
                                );
                            }
                            None => assert!(
                                gk.to_f32() == 0.0 && gv.to_f32() == 0.0,
                                "slot {slot} was not written, {label}"
                            ),
                        }
                    }
                }
            }
        }
    }
}

/// The input stage of a decode batch in one launch, against the loop it replaces: the head norms
/// of q and k by `fused_rmsnorm_f16`, their rotation by `rope_apply` and the write of k and v by
/// `paged_kv_write`.
///
/// The batch is 64 rows, each at its own position and in its own slot, with 32 q heads and 8 kv
/// heads, for both pairings of the rotation, with and without head norms. The head is 128 values,
/// 64 values (the vector norm with a single step) and 96 values (the vector norm with threads left
/// idle). Q and every row of the store must equal the separate launches bit for bit, and the slots
/// that no row owns stay zero.
#[test]
fn decode_rope_kv_write_f16_matches_rope_apply_and_paged_kv_write_bitwise() {
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    for hd in [128usize, 64, 96] {
        for with_norms in [false, true] {
            decode_stage_case(&dev, hd, with_norms);
        }
    }
}

fn decode_stage_case(dev: &Device, hd: usize, with_norms: bool) {
    use crate::inference::cache::paged_attention::{rope_apply, rope_cos_sin};
    let (rows, nh, nkv) = (64usize, 32usize, 8usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let width = qd + 2 * kvd;
    let values: Vec<half::f16> = (0..rows * width)
        .map(|i| half::f16::from_f32(((i as f32) * 0.0137).sin() * 2.0))
        .collect();
    let qkv = Tensor::from_slice(&values, (rows, width), dev).unwrap();
    // Distinct positions, and slots in an order that is not the row order.
    let positions: Vec<usize> = (0..rows).map(|r| 3 + 7 * r).collect();
    let (cos, sin) = rope_cos_sin(&positions, hd, 10000.0, dev, DType::F16).unwrap();
    let num_slots = 2 * rows;
    let slots: Vec<i32> = (0..rows)
        .map(|r| (2 * ((r * 5) % rows) + 1) as i32)
        .collect();
    let slot_dev = Tensor::from_vec(slots.clone(), &[rows][..], dev).unwrap();
    let weight = |seed: f32| -> Tensor {
        let w: Vec<f32> = (0..hd)
            .map(|i| 0.5 + ((i as f32) * seed).cos().abs())
            .collect();
        Tensor::from_slice(&w, (hd,), dev).unwrap()
    };
    let (qw, kw) = (weight(0.3), weight(0.7));
    let eps = 1e-6f32;
    let bits = |t: &Tensor| -> Vec<u16> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<half::f16>()
            .unwrap()
            .iter()
            .map(|h| h.to_bits())
            .collect()
    };
    for interleaved in [false, true] {
        let label = format!("head {hd}, norms {with_norms}, interleaved {interleaved}");
        let store = || Tensor::zeros_on((num_slots, kvd), DType::F16, dev).unwrap();
        let (kp_new, vp_new, kp_old, vp_old) = (store(), store(), store(), store());
        let norms = with_norms.then_some([(&qw, eps), (&kw, eps)]);
        let (q_new, _, _) = decode_rope_kv_write_f16(
            [(&qkv, 0), (&qkv, qd), (&qkv, qd + kvd)],
            norms,
            &cos,
            &sin,
            interleaved,
            &slot_dev,
            &kp_new,
            &vp_new,
            nh,
            nkv,
            hd,
        )
        .unwrap()
        .expect("the fused launch fits");
        let (q, k, v) = (
            qkv.narrow(1, 0, qd).unwrap(),
            qkv.narrow(1, qd, kvd).unwrap(),
            qkv.narrow(1, qd + kvd, kvd).unwrap(),
        );
        let mut qh = q.reshape(&[rows, nh, hd][..]).unwrap();
        let mut kh = k.reshape(&[rows, nkv, hd][..]).unwrap();
        if with_norms {
            qh = fused_rmsnorm_f16(&qh.contiguous().unwrap(), &qw, eps).unwrap();
            kh = fused_rmsnorm_f16(&kh.contiguous().unwrap(), &kw, eps).unwrap();
        }
        let q_old = rope_apply(&qh, &cos, &sin, interleaved).unwrap();
        let k_old = rope_apply(&kh, &cos, &sin, interleaved)
            .unwrap()
            .reshape(&[rows, kvd][..])
            .unwrap();
        paged_kv_write(&k_old, &v, &kp_old, &vp_old, &slot_dev, rows, kvd).unwrap();
        dev.synchronize().unwrap();
        assert_eq!(bits(&q_new), bits(&q_old), "q, {label}");
        assert_eq!(bits(&kp_new), bits(&kp_old), "store k, {label}");
        assert_eq!(bits(&vp_new), bits(&vp_old), "store v, {label}");
        let written = bits(&kp_new);
        for slot in (0..num_slots).filter(|s| !slots.contains(&(*s as i32))) {
            assert!(
                written[slot * kvd..(slot + 1) * kvd]
                    .iter()
                    .all(|&b| b & 0x7fff == 0),
                "slot {slot} was written, {label}"
            );
        }
    }
}

/// The attention of a prefill layer fed by the fused stage in the flash kernel's layout, against
/// the path it replaces: the q cast to F32, the transpose to heads first, the flash kernel with
/// an F32 output, and the cast of that output to F16. The outputs must equal bit for bit. The
/// cases are 1 sequence of 130 rows and 2 sequences of 70 and 60 rows, with 32 q heads, 8 kv
/// heads and a head of 64, 128 and 256 values. On a card that has the flash kernel, the
/// head-major launch must take the shapes.
#[test]
fn head_major_attention_matches_the_cast_transpose_and_cast_path() {
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    for hd in [64usize, 128, 256] {
        head_major_attention_case(&dev, hd);
    }
}

fn head_major_attention_case(dev: &Device, hd: usize) {
    use crate::inference::cache::paged_attention::{
        paged_prefill_attn_seqs, paged_prefill_attn_seqs_head_major, rope_cos_sin,
    };
    let (rows, nh, nkv) = (130usize, 32usize, 8usize);
    let (qd, kvd) = (nh * hd, nkv * hd);
    let width = qd + 2 * kvd;
    let values: Vec<half::f16> = (0..rows * width)
        .map(|i| half::f16::from_f32(((i as f32) * 0.0137).sin() * 2.0))
        .collect();
    let qkv = Tensor::from_slice(&values, (rows, width), dev).unwrap();
    let positions: Vec<usize> = (0..rows).collect();
    let (cos, sin) = rope_cos_sin(&positions, hd, 10000.0, dev, DType::F16).unwrap();
    let slots: Vec<i32> = (0..rows as i32).collect();
    let slot_dev = Tensor::from_vec(slots.clone(), &[rows][..], dev).unwrap();
    let weight = |seed: f32| -> Tensor {
        let w: Vec<f32> = (0..hd)
            .map(|i| 0.5 + ((i as f32) * seed).cos().abs())
            .collect();
        Tensor::from_slice(&w, (hd,), dev).unwrap()
    };
    let (qw, kw) = (weight(0.3), weight(0.7));
    let stage = |head_major: bool| {
        let kp = Tensor::zeros_on((rows, kvd), DType::F16, dev).unwrap();
        let vp = Tensor::zeros_on((rows, kvd), DType::F16, dev).unwrap();
        let srcs = [(&qkv, 0), (&qkv, qd), (&qkv, qd + kvd)];
        let norms = Some([(&qw, 1e-6f32), (&kw, 1e-6f32)]);
        let out = if head_major {
            prefill_rope_kv_write_head_major_f16(
                srcs, norms, &cos, &sin, false, &slot_dev, &slots, &kp, &vp, nh, nkv, hd,
            )
        } else {
            prefill_rope_kv_write_f16(
                srcs, norms, &cos, &sin, false, &slot_dev, &slots, &kp, &vp, nh, nkv, hd,
            )
        };
        out.unwrap().expect("the fused launch fits")
    };
    let (q_dense, k, v) = stage(false);
    let (q_major, k2, v2) = stage(true);
    assert_eq!(
        q_major.dims(),
        &[nh, rows, hd][..],
        "head-major shape, head {hd}"
    );
    assert_eq!(q_major.dtype(), DType::F32, "head-major dtype, head {hd}");
    let host16 = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
    assert_eq!(
        host16(&k),
        host16(&k2),
        "k is the same in both modes, head {hd}"
    );
    assert_eq!(
        host16(&v),
        host16(&v2),
        "v is the same in both modes, head {hd}"
    );
    // The head-major q is the dense q widened and transposed.
    let widened: Vec<f32> = q_dense
        .to_dtype(DType::F32)
        .unwrap()
        .transpose(0, 1)
        .unwrap()
        .contiguous()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let got_q: Vec<f32> = q_major.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(got_q, widened, "head-major q, head {hd}");
    let scale = 1.0 / (hd as f32).sqrt();
    if flash_prefill_gqa_card(dev).unwrap() {
        let (k3, v3) = (
            k.reshape((rows, nkv, hd)).unwrap(),
            v.reshape((rows, nkv, hd)).unwrap(),
        );
        assert!(
            flash_prefill_gqa_varlen_head_major(&q_major, &k3, &v3, &[rows], scale)
                .unwrap()
                .is_some(),
            "the head-major launch takes head {hd}"
        );
    }
    for lens in [vec![rows], vec![70usize, 60]] {
        let want = paged_prefill_attn_seqs(&q_dense, &k, &v, &lens, nh, nkv, hd, scale).unwrap();
        let got = paged_prefill_attn_seqs_head_major(&q_major, &k, &v, &lens, nh, nkv, hd, scale)
            .unwrap();
        assert_eq!(
            got.dtype(),
            DType::F16,
            "output dtype, head {hd}, lens {lens:?}"
        );
        assert_eq!(
            got.dims(),
            want.dims(),
            "output shape, head {hd}, lens {lens:?}"
        );
        assert_eq!(
            host16(&got),
            host16(&want),
            "attention, head {hd}, lens {lens:?}"
        );
    }
}
