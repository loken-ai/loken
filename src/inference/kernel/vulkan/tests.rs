//! The mat-vec kernels against the host's dequantised weights, on every Vulkan device listed.
//! A machine without one passes them without running anything.

use super::stack::{matvec_probe, RawTensor};
use crate::tensor::quantized::{GgmlDType, QTensor};
use crate::tensor::{Device, Tensor};

/// A deterministic spread of values in `[-1, 1)`.
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn check(dtype: GgmlDType, tokens: usize) {
    let (rows, cols) = (70, 512);
    let w = Tensor::from_vec(values(rows * cols, 7), (rows, cols), &Device::Cpu).unwrap();
    let q = QTensor::quantize(&w, dtype).unwrap();
    let dense = q
        .dequantize(&Device::Cpu)
        .unwrap()
        .to_dtype(crate::tensor::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let x = values(tokens * cols, 11);
    // The kernels read int8 blocks of 32 for every format but F16, as the quantiser makes them.
    let seen = if dtype == GgmlDType::F16 {
        x.clone()
    } else {
        x.chunks(if cols % 128 == 0 { 128 } else { 32 })
            .flat_map(|b| {
                let mx = b.iter().fold(0f32, |m, v| m.max(v.abs()));
                let (d, inv) = (mx / 127.0, if mx > 0.0 { 127.0 / mx } else { 0.0 });
                b.iter()
                    .map(move |v| (v * inv).round_ties_even().clamp(-127.0, 127.0) * d)
            })
            .collect()
    };
    let want: Vec<f32> = (0..tokens)
        .flat_map(|t| {
            let (seen, dense) = (&seen, &dense);
            (0..rows).map(move |r| {
                (0..cols)
                    .map(|c| dense[r * cols + c] as f64 * seen[t * cols + c] as f64)
                    .sum::<f64>() as f32
            })
        })
        .collect();
    let raw = RawTensor {
        dtype,
        rows,
        cols,
        bytes: q.data().unwrap(),
    };
    for d in super::devices() {
        let got = matvec_probe(d.index, &raw, &x, tokens).unwrap();
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        println!(
            "{} on {}: max error {err:.2e} of {scale:.2e}",
            dtype.gguf_name(),
            d.name
        );
        assert!(
            err <= scale * 1e-4,
            "{} on {}: error {err} of {scale}",
            dtype.gguf_name(),
            d.name
        );
    }
}

#[test]
fn matvec_q4k() {
    check(GgmlDType::Q4K, 3);
}

#[test]
fn matvec_q6k() {
    check(GgmlDType::Q6K, 3);
}

#[test]
fn matvec_q8_0() {
    check(GgmlDType::Q8_0, 3);
}

#[test]
fn matvec_f16() {
    check(GgmlDType::F16, 3);
}

#[test]
fn matmat_q4k() {
    check(GgmlDType::Q4K, 40);
}

#[test]
fn matmat_q6k() {
    check(GgmlDType::Q6K, 40);
}

#[test]
fn matmat_q8_0() {
    check(GgmlDType::Q8_0, 40);
}
