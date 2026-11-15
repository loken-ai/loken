//! A mixer of the wide residual whose projections are stored quantised, on a card and on the
//! host.
//!
//! The down and write-back projections are Q4_K and the up projection is Q5_0 over the low rank,
//! at the widths a released file stores them at.
//! A card runs such a mixer through the quantised products, not the fused launches.

use super::hyper::HyperMixer;
use super::{DType, Device, Tensor};
use crate::tensor::quantized::{GgmlDType, QTensor};

const STREAMS: usize = 4;
const DIM: usize = 2560;
const RANK: usize = 320;

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

fn quantised(n: usize, k: usize, seed: u64, dtype: GgmlDType) -> QTensor {
    let t = Tensor::from_vec(values(n * k, seed, 0.05), (n, k), &Device::Cpu).unwrap();
    QTensor::quantize(&t, dtype).unwrap()
}

/// The mixer with its gamma on `dev`, its weights read from the same seeds every time.
fn mixer(dev: &Device) -> HyperMixer {
    let gamma: Vec<f32> = values(STREAMS * DIM, 1, 0.3)
        .into_iter()
        .map(|v| 1.0 + v)
        .collect();
    let gamma = Tensor::from_vec(gamma, STREAMS * DIM, &Device::Cpu)
        .unwrap()
        .to_device(dev)
        .unwrap();
    HyperMixer::new(
        gamma,
        quantised(RANK, STREAMS * DIM, 2, GgmlDType::Q4K),
        quantised(STREAMS * DIM, RANK, 3, GgmlDType::Q5_0),
        Some(quantised(STREAMS, STREAMS * DIM, 4, GgmlDType::Q4K)),
        STREAMS,
        DIM,
        1e-6,
    )
    .unwrap()
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// The largest difference of `got` from `want`, relative to the largest magnitude of `want`.
fn relative(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let d = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    d / scale
}

/// The rounding of the activation each side quantises before a product, through three
/// products: a few percent of the output's scale.
const QUANTISED_MIX_SLACK: f32 = 0.05;

/// A card's mixer gives the host's sublayer input and write-back weights for one decoded row,
/// a short batch and prompt chunks, and gives the same values when run again.
#[test]
#[ignore = "needs an idle CUDA card"]
fn a_quantised_mixer_on_a_card_matches_the_host() {
    let card = Device::new_cuda(0).expect("a card for the card test");
    let host = mixer(&Device::Cpu);
    let on_card = mixer(&card);
    let mut failures = Vec::new();
    for rows in [1, 3, 5, 6, 8, 9, 64, 109] {
        let r = Tensor::from_vec(
            values(rows * STREAMS * DIM, 10 + rows as u64, 1.0),
            (1, rows, STREAMS * DIM),
            &Device::Cpu,
        )
        .unwrap();
        let (want, want_inj) = host.mix(&r).unwrap();
        let r_card = r.to_device(&card).unwrap();
        let run = || {
            let xn = on_card.normed_on_card(&r_card).unwrap();
            let (mixed, inj) = on_card.mix_on_card(&xn, false).unwrap();
            let weights = (crate::tensor::ops::sigmoid(&(inj.unwrap() / STREAMS as f64).unwrap())
                .unwrap()
                * 2.0)
                .unwrap();
            (flat(&mixed), flat(&weights))
        };
        let (got, got_inj) = run();
        let (again, again_inj) = run();
        let input = relative(&got, &flat(&want));
        let write = relative(&got_inj, &flat(&want_inj.unwrap()));
        eprintln!("{rows} rows: input {input}, write-back weights {write}");
        if input > QUANTISED_MIX_SLACK {
            failures.push(format!("{rows} rows: the sublayer input is off by {input}"));
        }
        if write > QUANTISED_MIX_SLACK {
            failures.push(format!(
                "{rows} rows: the write-back weights are off by {write}"
            ));
        }
        if got != again || got_inj != again_inj {
            failures.push(format!("{rows} rows: a second run gives other values"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
