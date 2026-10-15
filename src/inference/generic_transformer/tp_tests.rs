//! A model whose layers split their heads and feed-forward across two cards computes what the
//! model whole on one card computes: a prompt, decoded tokens, a verify batch and a cut back.

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
