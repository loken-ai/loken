//! The mat-vec kernels against the host's dequantised weights, on every Vulkan device listed.
//! A machine without one passes them without running anything.

use super::stack::{grouped_probe, matvec_probe, RawTensor};
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

/// Runs `dtype` on `tokens` tokens through the integer kernels (`coop` false) or, where a device
/// has them, the cooperative-matrix ones, whose f16 tiles round the weights and activations.
fn check(dtype: GgmlDType, tokens: usize, coop: bool) {
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
    let raw = RawTensor {
        dtype,
        rows,
        cols,
        bytes: q.data().unwrap(),
    };
    for d in super::devices() {
        let (got, shared) = matvec_probe(d.index, &raw, &x, tokens, coop).unwrap();
        // The kernels read int8 blocks of 32 for every format but the floats, as the quantiser
        // makes them, `shared` values to a scale.
        let seen = if matches!(dtype, GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::F32) {
            x.clone()
        } else {
            x.chunks(shared)
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
            err <= scale * if coop { 2e-3 } else { 1e-4 },
            "{} on {}: error {err} of {scale}",
            dtype.gguf_name(),
            d.name
        );
    }
}

#[test]
fn matvec_q4k() {
    check(GgmlDType::Q4K, 3, false);
}

#[test]
fn matvec_q6k() {
    check(GgmlDType::Q6K, 3, false);
}

#[test]
fn matvec_q8_0() {
    check(GgmlDType::Q8_0, 3, false);
}

#[test]
fn matvec_q5_0() {
    check(GgmlDType::Q5_0, 3, false);
}

#[test]
fn matvec_mxfp4() {
    check(GgmlDType::MxFp4, 3, false);
}

#[test]
fn many_tokens_mxfp4() {
    check(GgmlDType::MxFp4, 40, false);
}

#[test]
fn matvec_f16() {
    check(GgmlDType::F16, 3, false);
}

#[test]
fn matvec_bf16() {
    check(GgmlDType::BF16, 3, false);
}

#[test]
fn many_tokens_bf16() {
    check(GgmlDType::BF16, 40, false);
}

#[test]
fn matvec_f32() {
    check(GgmlDType::F32, 3, false);
}

#[test]
fn many_tokens_f32() {
    check(GgmlDType::F32, 40, false);
}

#[test]
fn matmat_q4k() {
    check(GgmlDType::Q4K, 40, false);
}

#[test]
fn matmat_q6k() {
    check(GgmlDType::Q6K, 40, false);
}

#[test]
fn matmat_q8_0() {
    check(GgmlDType::Q8_0, 40, false);
}

#[test]
fn matmat_q5_0() {
    check(GgmlDType::Q5_0, 40, false);
}

#[test]
fn coop_q4k() {
    check(GgmlDType::Q4K, 100, true);
}

#[test]
fn coop_q6k() {
    check(GgmlDType::Q6K, 100, true);
}

#[test]
fn coop_q8_0() {
    check(GgmlDType::Q8_0, 100, true);
}

#[test]
fn coop_q5_0() {
    check(GgmlDType::Q5_0, 100, true);
}

#[test]
fn coop_f16() {
    check(GgmlDType::F16, 100, true);
}

#[test]
fn matmat_f16() {
    check(GgmlDType::F16, 40, false);
}

/// The grouped expert kernel of `dtype` against the per-pair kernel on the same routes, with
/// the activations of the tokens, or with `by_pair` of the pairs.
///
/// Expert 0 has more pairs than a tile holds.
/// Expert 2 has a single pair.
/// Expert 3 has none.
fn grouped(dtype: GgmlDType, by_pair: bool) {
    let (experts, rows, cols, used, tokens) = (4, 70, 512, 2, 40);
    let w = Tensor::from_vec(
        values(experts * rows * cols, 7),
        (experts * rows, cols),
        &Device::Cpu,
    )
    .unwrap();
    let q = QTensor::quantize(&w, dtype).unwrap();
    let raw = RawTensor {
        dtype,
        rows: experts * rows,
        cols,
        bytes: q.data().unwrap(),
    };
    let chosen: Vec<u32> = (0..tokens)
        .flat_map(|t| [0, if t == 7 { 2 } else { 1 }])
        .collect();
    let xrows = if by_pair { chosen.len() } else { tokens };
    let x = values(xrows * cols, 11);
    for d in super::devices() {
        let (got, want, shared) =
            grouped_probe(d.index, &raw, experts, &x, &chosen, used, by_pair).unwrap();
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()));
        println!(
            "{} grouped on {}: max error {err:.2e} of {scale:.2e}, {shared} values a scale",
            dtype.gguf_name(),
            d.name
        );
        assert!(got.iter().all(|v| v.is_finite()), "{}", d.name);
        assert!(
            err <= scale * 1e-4,
            "{} on {}: error {err} of {scale}",
            dtype.gguf_name(),
            d.name
        );
    }
}

#[test]
fn grouped_q4k() {
    grouped(GgmlDType::Q4K, false);
}

#[test]
fn grouped_q6k() {
    grouped(GgmlDType::Q6K, true);
}

#[test]
fn grouped_q8_0() {
    grouped(GgmlDType::Q8_0, false);
}

#[test]
fn grouped_q5_0() {
    grouped(GgmlDType::Q5_0, true);
}

/// The cache store keeps each key and value at the f16 nearest to it, ties to even.
#[test]
fn cache_store_rounds_to_nearest() {
    let (heads, hd) = (4, 128);
    let k = values(heads * hd, 3);
    let v = values(heads * hd, 5);
    for d in super::devices() {
        let (kc, vc) = super::stack::cache_store_probe(d.index, &k, &v, heads, hd).unwrap();
        let off = |got: &[u16], want: &[f32]| {
            got.iter()
                .zip(want)
                .filter(|(g, w)| **g != half::f16::from_f32(**w).to_bits())
                .count()
        };
        let (bk, bv) = (off(&kc, &k), off(&vc, &v));
        println!(
            "cache store on {}: {bk} keys and {bv} values of {} off the nearest f16",
            d.name,
            k.len()
        );
        assert_eq!((bk, bv), (0, 0), "{}", d.name);
    }
}
