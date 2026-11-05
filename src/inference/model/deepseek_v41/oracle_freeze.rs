//! The logits of the V4.1 path, frozen.
//!
//! `tests/vectors/deepseek_v41_oracle/vectors.safetensors` holds what the V4.1 model computes on
//! the synthetic weights of the `model.rs` tests, for 2 band layouts:
//!
//! - `prefill`: the logits of every position of a prefill.
//! - `decode`: the logits of the same tokens decoded 1 at a time.
//! - `verify`: the logits of a block verified onto a prefilled prefix.
//! - `rollback_verify`: a rejected block rewound, then a shorter block verified.
//! - `rollback_decode`: the tokens after the shorter block, decoded 1 at a time.
//! - `greedy_ids` and `greedy_logits`: 64 greedy tokens after a short prompt, and the logits that
//!   chose each.
//!
//! A path that serves the same model must reproduce these within the tolerance below.
//! `write_the_frozen_oracle` rewrites the file from the current code, run on demand with
//! `--ignored`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::SafeTensors;

use super::model::oracle_gate::synthetic;
use super::model::DeepseekV41Model;
use crate::tensor::Tensor;

/// Largest relative L2 distance, per position, between a frozen row and a recomputed one.
const ROW_TOLERANCE: f32 = 1e-4;

/// Tokens of the greedy run.
const GREEDY_TOKENS: usize = 64;

/// Tokens of the prompt that starts the greedy run.
const GREEDY_PROMPT: usize = 8;

/// Tokens a rewound block keeps.
const KEPT: usize = 3;

/// Tokens of the verified block.
const BLOCK: usize = 6;

/// The band layouts: compress ratios, kv source layers, index source layers.
const LAYOUTS: [(&[u32], &[u32], &[u32]); 2] =
    [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];

/// One recorded array.
#[derive(Debug, Clone, PartialEq)]
enum Record {
    /// Row-major logits, `[rows, cols]`.
    Logits(usize, usize, Vec<f32>),
    /// Token ids.
    Ids(Vec<i32>),
}

fn vector_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/deepseek_v41_oracle/vectors.safetensors")
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn logits(t: &Tensor) -> Record {
    let (rows, cols) = t.dims2().unwrap();
    Record::Logits(rows, cols, flat(t))
}

fn argmax(row: &[f32]) -> i32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap()
}

/// Decode `tokens` 1 at a time onto `state`, and stack the logits.
fn decode_all(
    model: &DeepseekV41Model,
    tokens: &[u32],
    state: &mut super::cache::DecodeState,
) -> Record {
    let mut rows = Vec::new();
    let mut cols = 0;
    for &t in tokens {
        let row = flat(&model.forward_decode(t, state).unwrap());
        cols = row.len();
        rows.extend(row);
    }
    Record::Logits(tokens.len(), cols, rows)
}

/// What the current code computes for every layout, keyed by name.
fn record_current() -> BTreeMap<String, Record> {
    let mut out = BTreeMap::new();
    for (n, (ratios, kv_src, index_src)) in LAYOUTS.into_iter().enumerate() {
        let (model, tokens) = synthetic(ratios, kv_src, index_src, 0);
        let mut put = |name: &str, r: Record| {
            out.insert(format!("l{n}_{name}"), r);
        };

        put("prefill", logits(&model.forward_prefill(&tokens).unwrap()));

        let mut stepped = model.new_decode_state();
        put("decode", decode_all(&model, &tokens, &mut stepped));

        let split = tokens.len() - BLOCK;
        let mut state = model.new_decode_state();
        model.prefill_into(&tokens[..split], &mut state).unwrap();
        let mark = state.checkpoint();
        let block = &tokens[split..];
        put(
            "verify",
            logits(&model.forward_verify_batch(block, &mut state).unwrap()),
        );

        state.rewind(&mark);
        put(
            "rollback_verify",
            logits(
                &model
                    .forward_verify_batch(&block[..KEPT], &mut state)
                    .unwrap(),
            ),
        );
        put(
            "rollback_decode",
            decode_all(&model, &block[KEPT..], &mut state),
        );

        let mut state = model.new_decode_state();
        let mut last = flat(
            &model
                .prefill_into(&tokens[..GREEDY_PROMPT], &mut state)
                .unwrap(),
        );
        let (mut ids, mut rows) = (Vec::new(), Vec::new());
        for _ in 0..GREEDY_TOKENS {
            let id = argmax(&last);
            ids.push(id);
            rows.extend_from_slice(&last);
            last = flat(&model.forward_decode(id as u32, &mut state).unwrap());
        }
        let cols = last.len();
        put("greedy_ids", Record::Ids(ids));
        put("greedy_logits", Record::Logits(GREEDY_TOKENS, cols, rows));
    }
    out
}

fn to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Rewrite the frozen file from the current code.
#[test]
#[ignore = "rewrites tests/vectors/deepseek_v41_oracle from the current code"]
fn write_the_frozen_oracle() {
    let recorded = record_current();
    let bytes: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = recorded
        .into_iter()
        .map(|(name, r)| match r {
            Record::Logits(rows, cols, v) => (name, Dtype::F32, vec![rows, cols], to_bytes(&v)),
            Record::Ids(v) => (
                name,
                Dtype::I32,
                vec![v.len()],
                v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
        })
        .collect();
    let views: Vec<(String, TensorView)> = bytes
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    let path = vector_path();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    safetensors::serialize_to_file(views, None, &path).unwrap();
}

fn read_frozen() -> BTreeMap<String, Record> {
    let bytes = std::fs::read(vector_path()).unwrap();
    let file = SafeTensors::deserialize(&bytes).unwrap();
    file.tensors()
        .into_iter()
        .map(|(name, view)| {
            let words = view.data().chunks_exact(4);
            let r = match view.dtype() {
                Dtype::F32 => Record::Logits(
                    view.shape()[0],
                    view.shape()[1],
                    words
                        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                ),
                Dtype::I32 => Record::Ids(
                    words
                        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                        .collect(),
                ),
                other => panic!("{name}: unexpected dtype {other:?}"),
            };
            (name, r)
        })
        .collect()
}

/// The first difference between a frozen record and a recomputed one, or `None` when they agree
/// within the tolerance: ids exactly, logits per row by relative L2.
fn first_difference(name: &str, want: &Record, got: &Record) -> Option<String> {
    match (want, got) {
        (Record::Ids(w), Record::Ids(g)) => (w != g).then(|| format!("{name}: ids {g:?} != {w:?}")),
        (Record::Logits(wr, wc, w), Record::Logits(gr, gc, g)) => {
            if (wr, wc) != (gr, gc) {
                return Some(format!("{name}: shape [{gr}, {gc}] != [{wr}, {wc}]"));
            }
            for (row, (a, b)) in w.chunks_exact(*wc).zip(g.chunks_exact(*wc)).enumerate() {
                let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                let den: f32 = a.iter().map(|x| x * x).sum();
                let rel = (num / den).sqrt();
                if rel.is_nan() || rel > ROW_TOLERANCE {
                    return Some(format!("{name}: row {row} relative L2 {rel}"));
                }
            }
            None
        }
        _ => Some(format!("{name}: kind changed")),
    }
}

fn first_difference_all(
    want: &BTreeMap<String, Record>,
    got: &BTreeMap<String, Record>,
) -> Option<String> {
    if want.keys().ne(got.keys()) {
        return Some(format!(
            "records {:?} != {:?}",
            got.keys().collect::<Vec<_>>(),
            want.keys().collect::<Vec<_>>()
        ));
    }
    want.iter()
        .find_map(|(name, w)| first_difference(name, w, &got[name]))
}

/// The current V4.1 path reproduces the frozen prefill, decode, verify, rollback and greedy run.
#[test]
fn the_current_path_reproduces_the_frozen_logits() {
    let want = read_frozen();
    let got = record_current();
    if let Some(d) = first_difference_all(&want, &got) {
        panic!("the V4.1 path no longer reproduces its frozen logits: {d}");
    }
}

/// The frozen file is a usable oracle: every logit row is finite, and the comparison rejects a
/// change of the size it guards.
#[test]
fn the_comparison_rejects_a_changed_logit() {
    let want = read_frozen();
    assert!(!want.is_empty());
    for (name, r) in &want {
        if let Record::Logits(_, _, v) = r {
            assert!(v.iter().all(|x| x.is_finite()), "{name}: not finite");
        }
    }

    let mut nudged = want.clone();
    let Some(Record::Logits(_, _, v)) = nudged.get_mut("l0_decode") else {
        panic!("l0_decode missing");
    };
    let head = v.len().min(48);
    let norm = v[..head].iter().map(|x| x * x).sum::<f32>().sqrt();
    v[0] += 10.0 * ROW_TOLERANCE * norm;
    assert!(first_difference_all(&want, &nudged).is_some());

    let mut turned = want.clone();
    let Some(Record::Ids(ids)) = turned.get_mut("l1_greedy_ids") else {
        panic!("l1_greedy_ids missing");
    };
    ids[GREEDY_TOKENS - 1] += 1;
    assert!(first_difference_all(&want, &turned).is_some());
}
