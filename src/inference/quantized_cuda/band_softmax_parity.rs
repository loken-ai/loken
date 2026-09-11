//! The card's band softmax, judged against the operations it replaces.
//!
//! The kernel folds six passes into one: widen, mask, reduce to a row maximum, shift,
//! exponentiate and narrow. Everything it can get wrong is silent - a row that reads its
//! neighbour's scores, a mask off by one at the causal edge, a maximum taken over the
//! masked columns as well - and comes out as a plausible distribution rather than an
//! error. So it is compared against the same arithmetic written as separate operations,
//! on a band that crosses the diagonal, with a running maximum already in place.

use crate::tensor::{DType, Device, Tensor, D};

#[test]
fn the_card_band_softmax_answers_what_the_operations_answer() {
    let Some(device) = crate::tensor::cuda::CudaDevice::new(0)
        .ok()
        .map(Device::Cuda)
    else {
        eprintln!("no CUDA device; the band softmax kernel NOT covered by this run");
        return;
    };
    // A band that starts before the diagonal and ends after it, so rows differ in how much
    // of it they may see and some see none of it at all.
    let (heads, seq, cols, past, c0) = (3usize, 24usize, 100usize, 40usize, 30usize);
    let rows = heads * seq;

    let scores = Tensor::from_vec(
        (0..rows * cols)
            .map(|i| (i as f32 * 0.37).sin() * 4.0)
            .collect::<Vec<f32>>(),
        (1, heads, seq, cols),
        &device,
    )
    .unwrap()
    .to_dtype(DType::F16)
    .unwrap();
    // A running maximum that some rows exceed in this band and others do not, which is
    // what decides whether the shift comes from here or from before.
    let running = Tensor::from_vec(
        (0..rows)
            .map(|i| if i % 3 == 0 { 10.0 } else { -10.0 })
            .collect::<Vec<f32>>(),
        rows,
        &device,
    )
    .unwrap();

    let (weights, top, sum) =
        super::band_softmax_f16(&scores, &running, rows, cols, seq, past, c0).expect("kernel ran");

    // The same thing as separate operations.
    let mask = Tensor::from_vec(
        (0..seq)
            .flat_map(|p| {
                (0..cols).map(move |j| {
                    if c0 + j > past + p {
                        f32::NEG_INFINITY
                    } else {
                        0f32
                    }
                })
            })
            .collect::<Vec<f32>>(),
        (seq, cols),
        &device,
    )
    .unwrap();
    let wide = scores.to_dtype(DType::F32).unwrap();
    let masked = wide.broadcast_add(&mask).unwrap();
    let chain_top = masked
        .max_keepdim(D::Minus1)
        .unwrap()
        .maximum(&running.reshape((1, heads, seq, 1)).unwrap())
        .unwrap();
    let chain_weights = masked
        .broadcast_sub(&chain_top)
        .unwrap()
        .exp()
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let chain_sum = chain_weights
        .to_dtype(DType::F32)
        .unwrap()
        .sum_keepdim(D::Minus1)
        .unwrap();

    let pull = |t: &Tensor| -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap()
    };
    // A fully masked row softmaxes to nothing in both, and the chain writes exp(-inf)
    // where the kernel writes a zero: the same number, arrived at differently.
    let (a, b) = (pull(&weights), pull(&chain_weights));
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < 1e-3, "weight {i}: {x} vs {y}");
    }
    for (name, x, y) in [
        ("max", pull(&top), pull(&chain_top)),
        ("sum", pull(&sum), pull(&chain_sum)),
    ] {
        assert_eq!(x.len(), y.len(), "{name} length");
        for (i, (p, q)) in x.iter().zip(y.iter()).enumerate() {
            // A row that sees nothing has no maximum; both say so the same way.
            if !p.is_finite() && !q.is_finite() {
                continue;
            }
            assert!((p - q).abs() < 1e-2, "{name} {i}: {p} vs {q}");
        }
    }
}
