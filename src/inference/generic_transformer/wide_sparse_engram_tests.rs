//! The engram layers of the Sinkhorn wide-residual model on the generic route, against the V4.1
//! path, on the synthetic model of the frozen V4.1 oracle with engram rows added.
//!
//! The file is the oracle's, written by the helpers the V4.1 tests write it with.
//! The keys and tensors of 2 engram layers are added to it, in the layout the V4.1 engram test
//! writes: 2 heads, 3-grams, tables of 48 rows of 8 values, tokens 3 and 4 sharing 1 id.
//! The V4.1 model and the generic model are both loaded from that file.

use std::collections::BTreeMap;

use super::ngram::engram::EngramMeta;
use super::wide_sparse_batch_tests::{batched, prompts, serial};
use super::wide_sparse_layer::TokenLog;
use super::wide_sparse_oracle_tests::{
    bit_difference, generic, record_generic, record_v41, LAYOUTS,
};
use super::*;
use crate::inference::cache::state_disk::StateStore;
use crate::inference::model::deepseek_v41::engram as v41;
use crate::inference::model::deepseek_v41::model::oracle_gate::synthetic;
use crate::inference::model::deepseek_v41::model::tests::{
    real_entries_for, real_md_layout, VOCAB,
};
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::{open_mapped, MappedGguf, Value};
use crate::tensor::quantized::GgmlDType;

/// The layers that carry an engram: the first, and the third, a window layer of the first
/// layout and a band layer of the second.
const ENGRAM_LAYERS: [usize; 2] = [0, 2];

/// Heads of each n-gram size.
const HEADS: usize = 2;

/// Longest n-gram hashed.
const NGRAM: usize = 3;

/// Values of a table row.
const ROW: usize = 8;

/// Rows of each layer's table.
const ROWS: usize = 48;

/// The token whose compressed id a slot before the sequence start takes.
const PAD: u32 = 2;

/// Per engram layer, 1 multiplier per look-back.
const MULTIPLIERS: [u64; 6] = [3, 5, 7, 11, 13, 17];

/// Per engram layer, the bucket modulus of each column: every range fits in [`ROWS`].
const PRIMES: [u64; 8] = [7, 11, 13, 17, 5, 7, 11, 13];

/// Per engram layer, the first row of each column's bucket range.
const OFFSETS: [u64; 8] = [0, 7, 18, 31, 0, 5, 12, 23];

/// Element `i` of every tensor the V4.1 model tests write, as their `f32e` makes it.
fn value(i: usize) -> f32 {
    i as f32 * 0.0007 - 0.05
}

fn f32_entry(name: String, dims: Vec<usize>) -> GgufEntry {
    let n: usize = dims.iter().product();
    GgufEntry {
        name,
        dims,
        dtype: GgmlDType::F32,
        data: (0..n).flat_map(|i| value(i).to_le_bytes()).collect(),
    }
}

/// The value of `key` in `md`, a number.
fn md_usize(md: &[(String, Value)], key: &str) -> usize {
    md.iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.to_u32().ok())
        .unwrap_or_else(|| panic!("the oracle's metadata has no {key}")) as usize
}

/// The oracle's file in a band layout with the engram keys and tensors of [`ENGRAM_LAYERS`],
/// mapped.
pub(super) fn engram_file(ratios: &[u32], kv_src: &[u32], index_src: &[u32]) -> MappedGguf {
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let mut md = real_md_layout(ratios, kv_src, index_src);
    let mut entries = real_entries_for(&bands);
    let arch = md
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| v.to_string().ok().cloned())
        .expect("the oracle's architecture");
    let dim = md_usize(&md, &format!("{arch}.embedding_length"));
    let streams = md_usize(&md, &format!("{arch}.hyper_connection.count"));
    let key = |k: &str| format!("{arch}.engram.{k}");
    let array = |xs: &[u64]| Value::Array(xs.iter().map(|&x| Value::U64(x)).collect());
    let token_map: Vec<u64> = (0..VOCAB as u64)
        .map(|t| if t == 4 { 3 } else { t })
        .collect();
    let layer_ids = ENGRAM_LAYERS
        .iter()
        .map(|&l| Value::I32(l as i32))
        .collect();
    md.extend([
        (key("layer_ids"), Value::Array(layer_ids)),
        (key("head_count"), Value::U32(HEADS as u32)),
        (key("key_length"), Value::U32(ROW as u32)),
        (key("max_ngram_size"), Value::U32(NGRAM as u32)),
        (key("pad_id"), Value::U32(PAD)),
        (key("multipliers"), array(&MULTIPLIERS)),
        (key("primes"), array(&PRIMES)),
        (key("offsets"), array(&OFFSETS)),
        (key("token_map"), array(&token_map)),
    ]);
    for l in ENGRAM_LAYERS {
        let p = format!("blk.{l}");
        entries.extend([
            f32_entry(format!("{p}.engram_embd"), vec![ROWS, ROW]),
            f32_entry(format!("{p}.engram_q"), vec![streams, dim]),
            f32_entry(format!("{p}.engram_k"), vec![streams, dim]),
            f32_entry(
                format!("{p}.engram_wkv"),
                vec![dim * (streams + 1), (NGRAM - 1) * HEADS * ROW],
            ),
        ]);
    }
    // Unique per call: cargo runs the tests in parallel.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "loken-wide-sparse-engram-{}-{seq}.gguf",
        std::process::id()
    ));
    write_gguf_with_metadata(&path, &md, &entries).unwrap();
    let g = open_mapped(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    g
}

/// The generic model of the engram file in a band layout, every layer on the host.
pub(super) fn generic_engram(
    ratios: &[u32],
    kv_src: &[u32],
    index_src: &[u32],
) -> GenericHeteroTransformer {
    let MappedGguf { content, mmap, .. } = engram_file(ratios, kv_src, index_src);
    let plan = HeteroPlan::calculate_with_kv(ratios.len(), 1 << 20, &[], &[], 1.0, 0);
    GenericHeteroTransformer::from_gguf(content, &mmap[..], &HashMap::new(), &plan).unwrap()
}

/// The V4.1 model of the engram file in a band layout.
pub(super) fn v41_engram(ratios: &[u32], kv_src: &[u32], index_src: &[u32]) -> DeepseekV41Model {
    let g = engram_file(ratios, kv_src, index_src);
    let cfg = DeepseekV41Config::from_gguf(&g.content).unwrap();
    assert_eq!(
        cfg.engram.as_ref().map(|e| e.layer_ids.clone()),
        Some(ENGRAM_LAYERS.to_vec()),
        "the V4.1 reader found no engram layout"
    );
    DeepseekV41Model::load(&g, &cfg).unwrap()
}

fn ids(tokens: &[u32]) -> Tensor {
    Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// For both layouts, the generic model of the engram file gives the bits of the V4.1 model of
/// the same file on every record of the oracle: the prefill, the decode, the verify, the rewound
/// verify and the decode after it, and the greedy run.
/// The engram layers change the V4.1 logits, so the comparison covers them.
/// Both count the same bytes on the path every token reads, the engram projections included.
#[test]
fn the_generic_route_gives_the_bits_of_the_v41_path_with_engram_layers() {
    let mut differ = Vec::new();
    for (n, (ratios, kv_src, index_src)) in LAYOUTS.into_iter().enumerate() {
        let (plain, tokens) = synthetic(ratios, kv_src, index_src, 0);
        let v41 = v41_engram(ratios, kv_src, index_src);
        assert_ne!(
            bits(&v41.forward_prefill(&tokens).unwrap()),
            bits(&plain.forward_prefill(&tokens).unwrap()),
            "layout {ratios:?}: the engram layers left the V4.1 logits unchanged"
        );
        let (mut want, mut got) = (BTreeMap::new(), BTreeMap::new());
        record_v41(&v41, &tokens, n, &mut want);
        let mut model = generic_engram(ratios, kv_src, index_src);
        assert_eq!(
            model.resident_path_bytes(),
            Some(v41.resident_path_bytes() as u64),
            "layout {ratios:?}: the always-read bytes differ"
        );
        record_generic(&mut model, &tokens, n, &mut got);
        for (name, w) in &want {
            match got.get(name) {
                Some(g) => {
                    if let Some(d) = bit_difference(w, g) {
                        differ.push(format!("{name}: {d}"));
                    }
                }
                None => differ.push(format!("{name}: missing")),
            }
        }
    }
    assert!(
        differ.is_empty(),
        "the generic route differs from the V4.1 path with engram layers:\n{}",
        differ.join("\n")
    );
}

/// The token log hashes the rows the V4.1 engram state hashes: a whole prompt, a forward that
/// continues it, and a forward from an earlier position that replaces the positions after it.
/// A forward past the end of the log, and rows past it, are refused.
#[test]
fn the_token_log_hashes_the_rows_of_the_v41_engram_state() {
    let meta = EngramMeta {
        layer_ids: vec![1, 4],
        heads: 2,
        head_dim: 4,
        max_ngram: 3,
        token_map: vec![0, 1, 2, 3, 3, 5],
        pad_id: 2,
        multipliers: vec![vec![3, 5, 7], vec![11, 13, 17]],
        primes: vec![vec![11, 13, 17, 19], vec![23, 29, 31, 37]],
        offsets: vec![vec![0, 11, 24, 41], vec![0, 23, 52, 83]],
    };
    let cfg = v41::EngramConfig {
        layer_ids: meta.layer_ids.clone(),
        n_heads: meta.heads,
        head_dim: meta.head_dim,
        max_ngram: meta.max_ngram,
        token_map: meta.token_map.clone(),
        pad_id: meta.pad_id,
        multipliers: meta.multipliers.clone(),
        primes: meta.primes.clone(),
        offsets: meta.offsets.clone(),
    };
    // Token 9 lies past the map and hashes as the pad.
    let tokens = [4u32, 0, 9, 5, 3, 1, 2, 2];
    let split = 5;
    let mut log = TokenLog::default();
    log.open(0, &tokens[..split]).unwrap();
    log.open(split, &tokens[split..]).unwrap();
    for which in 0..meta.layer_ids.len() {
        assert_eq!(
            log.engram_hashes(&meta, which, 0, tokens.len()).unwrap(),
            v41::prefill_hashes(&cfg, which, &tokens)
        );
        let mut state = v41::EngramState::default();
        for &t in &tokens[..split] {
            state.push(&cfg, t);
        }
        assert_eq!(
            log.engram_hashes(&meta, which, split, tokens.len() - split)
                .unwrap(),
            v41::prefill_hashes_from(&cfg, which, &tokens[split..], state)
        );
    }
    let rewound = [4u32, 0, 9, 1, 1];
    log.open(3, &rewound[3..]).unwrap();
    assert_eq!(
        log.engram_hashes(&meta, 0, 0, rewound.len()).unwrap(),
        v41::prefill_hashes(&cfg, 0, &rewound)
    );
    assert!(log.engram_hashes(&meta, 0, 4, 2).is_err());
    assert!(log.open(rewound.len() + 1, &[1]).is_err());
}

/// For both layouts, 4 sequences batched on the engram model, each over its own state and token
/// log, give the tokens and the logits bits of the same 4 sequences run serially.
#[test]
fn four_engram_sequences_batched_give_the_bits_of_their_serial_runs() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let prompts = prompts();
        let want = serial(&mut generic_engram(ratios, kv_src, index_src), &prompts);
        let got = batched(&mut generic_engram(ratios, kv_src, index_src), &prompts);
        for (s, (w, g)) in want.iter().zip(&got).enumerate() {
            assert_eq!(
                w.0, g.0,
                "layout {ratios:?}: sequence {s} chose other tokens"
            );
            assert!(
                w.1 == g.1,
                "layout {ratios:?}: sequence {s} left the serial bits"
            );
        }
    }
}

/// A prefix restored from the snapshot table after another prompt ran hashes the ids of its own
/// sequence: the next token gives the bits of the same token decoded right after the prefix.
#[test]
fn a_restored_prefix_hashes_the_ids_of_its_own_sequence() {
    let (ratios, kv_src, index_src) = LAYOUTS[0];
    let (_, tokens) = synthetic(ratios, kv_src, index_src, 0);
    let half = tokens.len() / 2;
    let (prefix, other) = (&tokens[..half], &tokens[half..]);
    let next = tokens[0];

    let mut alone = generic_engram(ratios, kv_src, index_src);
    alone.forward(&ids(prefix), 0).unwrap();
    let want = alone.forward(&ids(&[next]), prefix.len()).unwrap();

    let mut model = generic_engram(ratios, kv_src, index_src);
    let logits = model.forward(&ids(prefix), 0).unwrap();
    model.snapshot_prefix(prefix, &logits).unwrap();
    model.forward(&ids(other), 0).unwrap();
    let index = model
        .state_snapshot_covering(prefix)
        .expect("the prefix is kept");
    model.restore_kv_snapshot(index).unwrap();
    let got = model.forward(&ids(&[next]), prefix.len()).unwrap();
    assert_eq!(bits(&got), bits(&want));
}

/// The token ids of the engram layers have no stored form: the state cache is refused for the
/// engram model and taken by the same model without engram layers.
#[test]
fn a_model_with_engram_layers_keeps_no_state_cache() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let identity = [0u8; 32];
        let mut plain = generic(ratios, kv_src, index_src);
        let store = Arc::new(StateStore::in_memory(1 << 20, 0));
        assert!(plain.attach_state_cache(store, identity), "{ratios:?}");
        let mut hashed = generic_engram(ratios, kv_src, index_src);
        let store = Arc::new(StateStore::in_memory(1 << 20, 0));
        assert!(!hashed.attach_state_cache(store, identity), "{ratios:?}");
    }
}
