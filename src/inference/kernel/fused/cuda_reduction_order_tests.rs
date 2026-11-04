//! Kernels that reduce in another order than the kernels they sit beside. Each is compared with
//! its sibling within a few ulps of f16, and bit for bit where it declines the shape.
use super::cuda_numerics_tests::try_open_cuda;
use super::*;
use crate::tensor::{DType, Tensor};

/// The largest gap between 2 f16 results in ulps of the larger magnitude, and whether it stays
/// within `ulps`. A magnitude under 1 counts as 1, since a rotation subtracts operands of that
/// scale and a small result keeps their error.
fn within_ulps(a: &[half::f16], b: &[half::f16], ulps: f32) -> (bool, f32) {
    let mut worst = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (x.to_f32(), y.to_f32());
        let ulp = x.abs().max(y.abs()).max(1.0) * 2f32.powi(-10);
        worst = worst.max((x - y).abs() / ulp);
    }
    (worst <= ulps, worst)
}

fn host(t: &Tensor) -> Vec<half::f16> {
    t.flatten_all().unwrap().to_vec1::<half::f16>().unwrap()
}

/// The register kernel of the residual add and norm against the shared memory kernel. The kept
/// sum is equal and the norm differs by at most 2 ulps. A width the register kernel does not
/// serve (2056 is 257 vectors) runs the other kernel and must equal it bit for bit.
#[test]
fn fused_add_rmsnorm_f16_registers_match_the_shared_memory_kernel() {
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    for (cols, same_bits) in [
        (4096usize, false),
        (2560, false),
        (8192, false),
        (2056, true),
    ] {
        let rows = 9usize;
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
        let (sum_old, norm_old) = fused_add_rmsnorm_f16_with(&x, &r, &w, eps, false).unwrap();
        let (sum_new, norm_new) = fused_add_rmsnorm_f16_with(&x, &r, &w, eps, true).unwrap();
        assert_eq!(host(&sum_old), host(&sum_new), "cols {cols}: kept sum");
        let (ok, worst) = within_ulps(&host(&norm_old), &host(&norm_new), 2.0);
        assert!(ok, "cols {cols}: norm differs by {worst} ulps");
        if same_bits {
            assert_eq!(worst, 0.0, "cols {cols}: the fallback differs");
        }
        eprintln!("registers, cols {cols}: norm within {worst} ulps");
    }
}

/// The wide activation against the exact one: at most 2 ulps apart for a width the wide kernel
/// serves (12288 and 96), and bit for bit for one it does not (100).
#[test]
fn silu_mul_gu_f16_wide_matches_the_exact_kernel() {
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    for (rows, ffn, same_bits) in [(64usize, 12288usize, false), (5, 96, false), (5, 100, true)] {
        let values: Vec<half::f16> = (0..rows * 2 * ffn)
            .map(|i| half::f16::from_f32(((i as f32) * 0.0071).sin() * 8.0))
            .collect();
        let gu = Tensor::from_slice(&values, (rows, 2 * ffn), &dev).unwrap();
        let exact = host(&silu_mul_gu_f16_with(&gu, ffn, false).unwrap().unwrap());
        let wide = host(&silu_mul_gu_f16_with(&gu, ffn, true).unwrap().unwrap());
        let (ok, worst) = within_ulps(&exact, &wide, 2.0);
        assert!(ok, "ffn {ffn}: differs by {worst} ulps");
        if same_bits {
            assert_eq!(worst, 0.0, "ffn {ffn}: the fallback differs");
        }
        eprintln!("wide activation, ffn {ffn}: within {worst} ulps");
    }
}

/// The warp-per-head decode input stage against the block-per-head one, for heads of 64, 128 and
/// 256 values (the warp kernel) and 96 (the fallback, bit for bit), with and without head norms,
/// for both pairings of the rotation. V in the store is equal. Q and k are at most 8 ulps apart.
#[test]
fn decode_q_rope_kv_write_f16_matches_the_block_per_head_stage() {
    use crate::inference::cache::paged_attention::rope_cos_sin;
    let Some(dev) = try_open_cuda() else {
        eprintln!("skipping: no usable CUDA device");
        return;
    };
    for hd in [128usize, 64, 256, 96] {
        for with_norms in [false, true] {
            for interleaved in [false, true] {
                let (rows, nh, nkv) = (64usize, 32usize, 8usize);
                let (qd, kvd) = (nh * hd, nkv * hd);
                let width = qd + 2 * kvd;
                let values: Vec<half::f16> = (0..rows * width)
                    .map(|i| half::f16::from_f32(((i as f32) * 0.0137).sin() * 2.0))
                    .collect();
                let qkv = Tensor::from_slice(&values, (rows, width), &dev).unwrap();
                let positions: Vec<usize> = (0..rows).map(|r| 3 + 7 * r).collect();
                let (cos, sin) = rope_cos_sin(&positions, hd, 10000.0, &dev, DType::F16).unwrap();
                let num_slots = 2 * rows;
                let slots: Vec<i32> = (0..rows)
                    .map(|r| (2 * ((r * 5) % rows) + 1) as i32)
                    .collect();
                let slot_dev = Tensor::from_vec(slots, &[rows][..], &dev).unwrap();
                let weight = |seed: f32| -> Tensor {
                    let w: Vec<f32> = (0..hd)
                        .map(|i| 0.5 + ((i as f32) * seed).cos().abs())
                        .collect();
                    Tensor::from_slice(&w, (hd,), &dev).unwrap()
                };
                let (qw, kw) = (weight(0.3), weight(0.7));
                let eps = 1e-6f32;
                let norms = with_norms.then_some([(&qw, eps), (&kw, eps)]);
                let store = || Tensor::zeros_on((num_slots, kvd), DType::F16, &dev).unwrap();
                let (kp_old, vp_old, kp_new, vp_new) = (store(), store(), store(), store());
                let srcs = [(&qkv, 0), (&qkv, qd), (&qkv, qd + kvd)];
                let (q_old, _, _) = decode_rope_kv_write_f16(
                    srcs,
                    norms,
                    &cos,
                    &sin,
                    interleaved,
                    &slot_dev,
                    &kp_old,
                    &vp_old,
                    nh,
                    nkv,
                    hd,
                )
                .unwrap()
                .expect("the block kernel fits");
                let q_new = decode_q_rope_kv_write_f16(
                    srcs,
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
                .expect("the warp kernel fits");
                dev.synchronize().unwrap();
                let label = format!("head {hd}, norms {with_norms}, interleaved {interleaved}");
                let ulps = if hd == 96 { 0.0 } else { 8.0 };
                assert_eq!(host(&vp_new), host(&vp_old), "store v, {label}");
                let (ok, worst_q) = within_ulps(&host(&q_old), &host(&q_new), ulps);
                assert!(ok, "q differs by {worst_q} ulps, {label}");
                let (ok, worst_k) = within_ulps(&host(&kp_old), &host(&kp_new), ulps);
                assert!(ok, "store k differs by {worst_k} ulps, {label}");
                eprintln!("warp stage, {label}: q within {worst_q}, k within {worst_k} ulps");
            }
        }
    }
}
