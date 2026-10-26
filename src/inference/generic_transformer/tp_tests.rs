//! A model whose layers split their heads and feed-forward, or their experts, across two cards
//! computes what the model whole on one card computes: a prompt, decoded tokens, a verify batch
//! and a cut back.

use super::tiny_model;
use super::GenericHeteroTransformer;
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::{Device, Tensor};

const LAYERS: usize = 3;

fn load(path: &std::path::Path, cards: &[usize]) -> GenericHeteroTransformer {
    let mut plan = HeteroPlan::forced_gpu(LAYERS, LAYERS, cards[0]);
    plan.tp_cards = (cards.len() > 1).then(|| {
        cards
            .iter()
            .map(|&index| crate::inference::place::layer_executor::TpCard { index, weight: 1.0 })
            .collect()
    });
    let devices: std::collections::HashMap<usize, Device> = cards
        .iter()
        .map(|&c| (c, Device::new_cuda(c).unwrap()))
        .collect();
    let g = crate::tensor::quantized::gguf_file::open_mapped(path).unwrap();
    let bytes = g.mmap.clone();
    GenericHeteroTransformer::from_gguf(g.content, &bytes[..], &devices, &plan).unwrap()
}

fn rows(logits: &Tensor) -> Vec<f32> {
    logits
        .to_dtype(crate::tensor::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

fn ids(toks: &[u32]) -> Tensor {
    Tensor::new(toks, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
}

fn close(got: &[f32], want: &[f32], what: &str) {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert_eq!(got.len(), want.len(), "{what}");
    assert!(worst <= 1e-4 * scale, "{what}: off by {worst} of {scale}");
}

#[test]
#[ignore = "needs two idle CUDA cards"]
fn a_layer_split_across_two_cards_computes_the_whole_layer() {
    if Device::new_cuda(1).is_err() {
        eprintln!("skipping: two usable CUDA devices are needed");
        return;
    }
    let path = tiny_model::write("tp", LAYERS);
    let mut whole = load(&path, &[0]);
    let mut split = load(&path, &[0, 1]);
    assert!(
        split.layers.iter().all(|l| l.tp_shards.len() == 1),
        "every layer split"
    );
    // Each share reads its weights on its own card.
    let card = |i: usize| Device::new_cuda(i).unwrap();
    for l in &split.layers {
        assert!(
            l.ffn_down.device().same_device(&card(0)),
            "rank 0 on card 0"
        );
        assert!(
            l.tp_shards[0].ffn_down.device().same_device(&card(1)),
            "rank 1 on card 1"
        );
        assert!(
            l.tp_shards[0].attn_output.device().same_device(&card(1)),
            "rank 1 on card 1"
        );
    }
    let mut both = |what: &str, f: &dyn Fn(&mut GenericHeteroTransformer) -> Tensor| {
        close(&rows(&f(&mut split)), &rows(&f(&mut whole)), what);
    };
    both("prompt", &|m| {
        m.forward(&ids(&[1, 4, 2, 9, 3, 5]), 0).unwrap()
    });
    both("token 6", &|m| m.forward(&ids(&[8]), 6).unwrap());
    both("token 7", &|m| m.forward(&ids(&[2]), 7).unwrap());
    both("verify batch", &|m| {
        m.forward_all(&ids(&[5, 1]), 8).unwrap()
    });
    both("after a cut", &|m| {
        m.trim_kv(9);
        m.forward(&ids(&[3]), 9).unwrap()
    });
    // A snapshot of the sequence, another sequence over it, the snapshot restored and continued:
    // the shares' caches go aside and come back with the layer's.
    both("after a snapshot is restored", &|m| {
        m.snapshot_kv(vec![1, 4, 2, 9, 3, 5, 8, 2, 5, 1, 3], 10)
            .unwrap();
        m.forward(&ids(&[7, 7, 6]), 0).unwrap();
        let (i, _) = m
            .best_kv_snapshot(&[1, 4, 2, 9, 3, 5, 8, 2, 5, 1, 3, 4])
            .unwrap();
        m.restore_kv_snapshot(i).unwrap();
        m.forward(&ids(&[4]), 10).unwrap()
    });
    // A context shift drops positions 2 to 4 and rotates the keys behind them, in every share.
    both("after a context shift", &|m| {
        m.shift_kv_tail(5, 3).unwrap();
        m.forward(&ids(&[6]), 8).unwrap()
    });
    tiny_model::remove(&path);
}

/// A mixture whose cards each hold a range of the experts computes what it computes whole on one
/// card: a prompt, decoded tokens, a verify batch and a cut back.
#[test]
#[ignore = "needs two idle CUDA cards"]
fn a_mixture_split_across_two_cards_computes_the_whole_layer() {
    if Device::new_cuda(1).is_err() {
        eprintln!("skipping: two usable CUDA devices are needed");
        return;
    }
    let path = tiny_model::write_moe("tp-moe", LAYERS);
    let mut whole = load(&path, &[0]);
    let mut split = load(&path, &[0, 1]);
    assert!(
        split.layers.iter().all(|l| l.tp_shards.len() == 1),
        "every layer split"
    );
    for l in &split.layers {
        let ranges: Vec<_> = std::iter::once(l)
            .chain(&l.tp_shards)
            .map(|p| {
                p.moe
                    .as_ref()
                    .map(|m| (m.local.map(|x| x.first), m.n_experts))
            })
            .collect();
        assert_eq!(
            ranges,
            vec![Some((Some(0), 4)), Some((Some(4), 4))],
            "experts by range"
        );
    }
    let mut both = |what: &str, f: &dyn Fn(&mut GenericHeteroTransformer) -> Tensor| {
        close(&rows(&f(&mut split)), &rows(&f(&mut whole)), what);
    };
    both("prompt", &|m| {
        m.forward(&ids(&[1, 4, 2, 9, 3, 5]), 0).unwrap()
    });
    both("token 6", &|m| m.forward(&ids(&[8]), 6).unwrap());
    both("token 7", &|m| m.forward(&ids(&[2]), 7).unwrap());
    both("verify batch", &|m| {
        m.forward_all(&ids(&[5, 1]), 8).unwrap()
    });
    both("after a cut", &|m| {
        m.trim_kv(9);
        m.forward(&ids(&[3]), 9).unwrap()
    });
    tiny_model::remove(&path);
}

/// A group of one card sums a block onto itself through the same communicators the split layers
/// use, so a library that cannot sum is found here and not at a layer.
#[cfg(feature = "gpu")]
#[test]
fn a_group_of_one_card_sums_through_nccl() {
    use crate::tensor::{DType, Device, Tensor};
    let Ok(device) = Device::new_cuda(0) else {
        return;
    };
    let Some(group) = super::TpComms::new(std::slice::from_ref(&device)) else {
        return;
    };
    let part = Tensor::ones((4096,), DType::F32, &device).unwrap();
    let summed = group.all_reduce(&[part]).expect("the group sums");
    device.synchronize().unwrap();
    let got = summed[0].to_vec1::<f32>().unwrap();
    assert!(got.iter().all(|&v| v == 1.0));
}

/// Two cards sum a block through the communicators the split layers use.
#[cfg(feature = "gpu")]
#[test]
fn two_cards_sum_through_nccl() {
    use crate::tensor::{DType, Device, Tensor};
    let (Ok(first), Ok(second)) = (Device::new_cuda(0), Device::new_cuda(1)) else {
        return;
    };
    let devices = [first, second];
    // The planner measures the link between the two cards before it weighs a sum across them.
    let _ = crate::inference::place::device_probe::card_pair_link(0, 1);
    let Some(group) = super::TpComms::new(&devices) else {
        return;
    };
    let parts: Vec<Tensor> = devices
        .iter()
        .map(|d| Tensor::ones((4096,), DType::F32, d).unwrap())
        .collect();
    let summed = group.all_reduce(&parts).expect("the group sums");
    for d in &devices {
        d.synchronize().unwrap();
    }
    for s in &summed {
        assert!(s.to_vec1::<f32>().unwrap().iter().all(|&v| v == 2.0));
    }
}

/// The measurements the planner takes before it weighs a split, in its order, then the sum.
#[cfg(feature = "gpu")]
#[test]
fn the_planner_measurements_then_a_sum_across_two_cards() {
    use crate::inference::place::device_probe as probe;
    if crate::tensor::Device::new_cuda(1).is_err() {
        return;
    }
    let steps: Vec<(&str, Box<dyn Fn() -> bool>)> = vec![
        ("probe", Box::new(|| !probe::probe_cuda_gpus().is_empty())),
        ("link", Box::new(|| probe::card_pair_link(0, 1).is_some())),
        (
            "read rates",
            Box::new(|| !crate::inference::place::expert_parallel::card_read_rates().is_empty()),
        ),
        (
            "matmul 0",
            Box::new(|| probe::card_matmul_rate(0).is_some()),
        ),
        (
            "matmul 1",
            Box::new(|| probe::card_matmul_rate(1).is_some()),
        ),
    ];
    for (what, step) in &steps {
        eprintln!("step {what}: {}", step());
    }
    // Two cards NCCL does not join, of two makers or without NCCL, never sum across.
    let devices: Vec<crate::tensor::Device> = (0..2)
        .map(|i| crate::tensor::Device::new_cuda(i).unwrap())
        .collect();
    if super::TpComms::new(&devices).is_none() {
        return;
    }
    assert!(
        probe::group_sum(&[0, 1]).is_some(),
        "the sum across the two cards failed"
    );
}
