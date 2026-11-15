//! The 32-value nibble formats through a card's mat-vec, against the float product of their
//! dequantised weights and against the host's integer dot.
//!
//! Q4_0 and Q5_0 store their codes with a bias, Q4_1 and Q5_1 with a per-block minimum.
//! A decoded row reads these formats through the mat-vec kernels.

use super::{GgmlDType, QMatMul, QTensor};
use crate::tensor::{Device, Tensor};

/// Values in `[-scale, scale)` from a fixed seed.
fn values(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

const FORMATS: [GgmlDType; 4] = [
    GgmlDType::Q4_0,
    GgmlDType::Q5_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_1,
];

/// The largest difference of `got` from `want`, relative to the largest magnitude of `want`.
fn relative(got: &[f32], want: &[f64]) -> f64 {
    let peak = want.iter().fold(0f64, |m, v| m.max(v.abs()));
    got.iter()
        .zip(want)
        .map(|(a, b)| (f64::from(*a) - b).abs())
        .fold(0f64, f64::max)
        / peak
}

/// One weight `[n, k]` in `dtype`, the product of an activation row against it on the card, on
/// the host, and in f64 over the dequantised weight.
fn products(dtype: GgmlDType, n: usize, k: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f64>) {
    let card = Device::new_cuda(0).expect("a card for the card test");
    let w = Tensor::from_vec(values(n * k, seed, 0.05), (n, k), &Device::Cpu).unwrap();
    let x = values(k, seed + 1, 1.0);
    let on_host = QMatMul::from_qtensor(QTensor::quantize(&w, dtype).unwrap()).unwrap();
    let on_card = QMatMul::from_qtensor(QTensor::quantize_onto(&w, dtype, &card).unwrap()).unwrap();
    let row = |dev: &Device| Tensor::from_vec(x.clone(), (1, k), dev).unwrap();
    let flat = |t: Tensor| {
        t.to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    };
    let host = flat(on_host.forward(&row(&Device::Cpu)).unwrap());
    let got = flat(on_card.forward(&row(&card)).unwrap());
    let dq = QTensor::quantize(&w, dtype)
        .unwrap()
        .dequantize(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let exact = (0..n)
        .map(|r| {
            (0..k)
                .map(|c| f64::from(dq[r * k + c]) * f64::from(x[c]))
                .sum::<f64>()
        })
        .collect();
    (got, host, exact)
}

/// A card's mat-vec of a nibble format stays as close to the float product as the host's
/// integer dot does, at the widths of a served layer.
#[test]
#[ignore = "needs an idle CUDA card"]
fn a_nibble_format_matvec_on_a_card_matches_the_float_product() {
    let mut failures = Vec::new();
    for dtype in FORMATS {
        for (n, k) in [(4096usize, 4096usize), (2560, 640), (640, 2560)] {
            let (got, host, exact) = products(dtype, n, k, 7);
            let card = relative(&got, &exact);
            let host_err = relative(&host, &exact);
            let card_host = relative(
                &got,
                &host.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
            );
            eprintln!(
                "{dtype:?} {n}x{k}: card {card:.5}, host {host_err:.5}, card against host \
                 {card_host:.5} of the float product's peak"
            );
            if card > 2.0 * host_err.max(1e-3) {
                failures.push(format!(
                    "{dtype:?} {n}x{k}: the card is {card} away, the host {host_err}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The time of a card's mat-vec of a nibble format, a row at a time, for a comparison between
/// builds.
#[test]
#[ignore = "needs an idle CUDA card"]
fn time_a_nibble_format_matvec_on_a_card() {
    let card = Device::new_cuda(0).expect("a card for the card test");
    for dtype in [GgmlDType::Q4_0, GgmlDType::Q5_0] {
        for (n, k) in [(4096usize, 4096usize), (2560, 640)] {
            let w = Tensor::from_vec(values(n * k, 3, 0.05), (n, k), &Device::Cpu).unwrap();
            let m =
                QMatMul::from_qtensor(QTensor::quantize_onto(&w, dtype, &card).unwrap()).unwrap();
            let x = Tensor::from_vec(values(k, 4, 1.0), (1, k), &card).unwrap();
            for _ in 0..20 {
                m.forward(&x).unwrap();
            }
            card.synchronize().unwrap();
            let rounds = 200;
            let mut best = f64::MAX;
            for _ in 0..5 {
                let t = std::time::Instant::now();
                for _ in 0..rounds {
                    m.forward(&x).unwrap();
                }
                card.synchronize().unwrap();
                best = best.min(t.elapsed().as_secs_f64() * 1e6 / rounds as f64);
            }
            eprintln!("{dtype:?} {n}x{k}: {best:.2} us a call, best of 5 runs of {rounds}");
        }
    }
}
