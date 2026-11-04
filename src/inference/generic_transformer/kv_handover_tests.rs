//! The KV a prompt leaves is handed over as rows of whole layers, whatever way the plan that read
//! it cut the stack: a model read under a plan of 2 segments gives rows that a model under a plan
//! of 1 segment imports, and the other way round, and both go on to the same logits.

use super::tiny_model;
use super::*;
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};

const LAYERS: usize = 4;
const PROMPT: [u32; 6] = [1, 4, 2, 9, 3, 5];
const NEXT: [u32; 3] = [8, 2, 6];

fn segment(start: usize, end: usize) -> HeteroSegment {
    HeteroSegment {
        kind: DeviceKind::Cpu,
        layer_start: start,
        layer_end: end,
        free_memory_bytes: u64::MAX,
    }
}

/// The stack cut into `cuts + 1` consecutive segments.
fn plan(cuts: usize) -> HeteroPlan {
    let step = LAYERS / (cuts + 1);
    HeteroPlan {
        segments: (0..=cuts)
            .map(|i| segment(i * step, if i == cuts { LAYERS } else { (i + 1) * step }))
            .collect(),
        tp_cards: None,
        total_layers: LAYERS,
        host_experts_from: None,
    }
}

fn load(path: &std::path::Path, plan: &HeteroPlan) -> GenericHeteroTransformer {
    let g = crate::tensor::quantized::gguf_file::open_mapped(path).unwrap();
    let bytes = g.mmap.clone();
    GenericHeteroTransformer::from_gguf(
        g.content,
        &bytes[..],
        &std::collections::HashMap::new(),
        plan,
    )
    .unwrap()
}

fn ids(toks: &[u32]) -> Tensor {
    Tensor::new(toks, &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
}

fn logits(m: &mut GenericHeteroTransformer, toks: &[u32], at: usize) -> Vec<f32> {
    m.forward(&ids(toks), at)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

/// The prompt's rows read by `from` and imported by `to`, then `NEXT` decoded by both.
fn hand_over(from: &mut GenericHeteroTransformer, to: &mut GenericHeteroTransformer) {
    logits(from, &PROMPT, 0);
    from.snapshot_kv(PROMPT.to_vec(), PROMPT.len()).unwrap();
    let mut asked = PROMPT.to_vec();
    asked.push(NEXT[0]);
    let (index, held) = from.best_kv_snapshot(&asked).unwrap();
    assert_eq!(held, PROMPT.len());
    let rows = from.export_kv_rows(index, 0, held).unwrap();
    assert_eq!(rows.len(), LAYERS);
    assert_eq!(
        from.kv_layout(),
        to.kv_layout(),
        "one layout for both plans"
    );
    let dtype = from.kv_snapshot_dtype(index).unwrap();
    to.import_kv_snapshot(PROMPT.to_vec(), held, rows.clone(), dtype)
        .unwrap();
    let (imported, held_here) = to.best_kv_snapshot(&asked).unwrap();
    assert_eq!(held_here, held);
    assert!(
        to.export_kv_rows(imported, 0, held).unwrap() == rows,
        "the rows come back as they went in"
    );
    to.restore_kv_snapshot(imported).unwrap();
    for (i, &t) in NEXT.iter().enumerate() {
        let want = logits(from, &[t], PROMPT.len() + i);
        let got = logits(to, &[t], PROMPT.len() + i);
        assert!(got == want, "token {i} after the hand-over differs");
    }
}

#[test]
fn rows_read_under_2_segments_are_imported_under_1_and_back() {
    let path = tiny_model::write("handover-segments", LAYERS);
    hand_over(&mut load(&path, &plan(1)), &mut load(&path, &plan(0)));
    hand_over(&mut load(&path, &plan(0)), &mut load(&path, &plan(1)));
    tiny_model::remove(&path);
}
