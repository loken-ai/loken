//! The generic route against the V4.1 path, on the synthetic model of the frozen V4.1 oracle.
//!
//! The model is the one the V4.1 oracle froze, in its 2 band layouts.
//! Its file is written by the toy writers of `wide_sparse_toy`.
//! The generic model is built from that file by the generic loader and runs on the host.
//!
//! The records are the oracle's: a prefill, the same tokens decoded 1 at a time, a verify onto a
//! prefilled prefix, a shorter verify after the first is rewound, the tokens after it decoded,
//! and 64 greedy tokens after a short prompt.
//! They are compared with the oracle's vectors within its tolerance, and with the bits the V4.1
//! path gave, recorded in `deepseek_v41_frozen`.

use std::collections::BTreeMap;

use safetensors::tensor::Dtype;
use safetensors::SafeTensors;

use super::mixer::{layer_mixer_kind, MixerKind};
use super::*;
use crate::inference::generic_transformer::wide_sparse_toy::prompt;
use crate::inference::generic_transformer::wide_sparse_toy::{real_entries_for, real_md_layout};
use crate::tensor::gguf_write::write_gguf_with_metadata;
use crate::tensor::quantized::gguf_file::{open_mapped, MappedGguf};

/// The oracle's band layouts: compress ratios, kv source layers, index source layers.
pub(super) const LAYOUTS: [(&[u32], &[u32], &[u32]); 2] =
    [(&[0, 2, 0], &[1], &[1]), (&[0, 2, 2, 0], &[1], &[1, 3])];

/// Tokens of the oracle's greedy run.
const GREEDY_TOKENS: usize = 64;

/// Tokens of the prompt that starts the oracle's greedy run.
const GREEDY_PROMPT: usize = 8;

/// Tokens the oracle's rewound block keeps.
const KEPT: usize = 3;

/// Tokens of the oracle's verified block.
const BLOCK: usize = 6;

/// The relative L2 distance per row the frozen oracle grants a path that serves the same model.
const FROZEN_ROW_TOLERANCE: f32 = 1e-4;

/// The records of a layout, in the order a forward builds on the one before.
pub(super) const RECORDS: [&str; 7] = [
    "prefill",
    "decode",
    "verify",
    "rollback_verify",
    "rollback_decode",
    "greedy_ids",
    "greedy_logits",
];

/// One recorded array.
#[derive(Debug, Clone)]
pub(super) enum Record {
    /// Row-major logits, `[rows, cols]`.
    Logits(usize, usize, Vec<f32>),
    /// Token ids.
    Ids(Vec<i32>),
}

fn flat(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// Logits of any rank as rows of the last dimension.
fn logits(t: &Tensor) -> Record {
    let cols = *t.dims().last().unwrap();
    let v = flat(t);
    Record::Logits(v.len() / cols, cols, v)
}

/// The oracle's greedy choice: the last of the largest.
fn argmax(row: &[f32]) -> i32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap()
}

fn ids(tokens: &[u32]) -> Tensor {
    Tensor::from_vec(tokens.to_vec(), (1, tokens.len()), &Device::Cpu).unwrap()
}

/// The synthetic model's file in a band layout, mapped.
pub(super) fn toy_file(ratios: &[u32], kv_src: &[u32], index_src: &[u32]) -> MappedGguf {
    let bands: Vec<bool> = (0..ratios.len())
        .map(|l| ratios[l] != 0 || kv_src.contains(&(l as u32)) || index_src.contains(&(l as u32)))
        .collect();
    let md = real_md_layout(ratios, kv_src, index_src);
    let entries = real_entries_for(&bands);
    // Unique per call: cargo runs the tests in parallel.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "loken-wide-sparse-{}-{seq}.gguf",
        std::process::id()
    ));
    write_gguf_with_metadata(&path, &md, &entries).unwrap();
    let g = open_mapped(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    g
}

/// The generic model of the synthetic file in a band layout, every layer on the host.
pub(super) fn generic(
    ratios: &[u32],
    kv_src: &[u32],
    index_src: &[u32],
) -> GenericHeteroTransformer {
    let MappedGguf { content, mmap, .. } = toy_file(ratios, kv_src, index_src);
    let plan = HeteroPlan::calculate_with_kv(ratios.len(), 1 << 20, &[], &[], 1.0, 0);
    GenericHeteroTransformer::from_gguf(content, &mmap[..], &HashMap::new(), &plan).unwrap()
}

/// `tokens` decoded 1 at a time by the generic model from position `pos`.
fn generic_decode(model: &mut GenericHeteroTransformer, tokens: &[u32], pos: usize) -> Record {
    let mut rows = Vec::new();
    let mut cols = 0;
    for (i, &t) in tokens.iter().enumerate() {
        let row = flat(&model.forward(&ids(&[t]), pos + i).unwrap());
        cols = row.len();
        rows.extend(row);
    }
    Record::Logits(tokens.len(), cols, rows)
}

/// The same records through the generic model.
///
/// A prefill and a verify return every position's logits, a prompt and a decode step the last
/// one's.
/// The rewind cuts the model back to the verified block's start.
pub(super) fn record_generic(
    model: &mut GenericHeteroTransformer,
    tokens: &[u32],
    n: usize,
    out: &mut BTreeMap<String, Record>,
) {
    let mut put = |name: &str, r: Record| {
        out.insert(format!("l{n}_{name}"), r);
    };
    put(
        "prefill",
        logits(&model.forward_all(&ids(tokens), 0).unwrap()),
    );
    put("decode", generic_decode(model, tokens, 0));

    let split = tokens.len() - BLOCK;
    model.forward(&ids(&tokens[..split]), 0).unwrap();
    let block = &tokens[split..];
    put(
        "verify",
        logits(&model.forward_all(&ids(block), split).unwrap()),
    );
    model.trim_kv(split);
    put(
        "rollback_verify",
        logits(&model.forward_all(&ids(&block[..KEPT]), split).unwrap()),
    );
    put(
        "rollback_decode",
        generic_decode(model, &block[KEPT..], split + KEPT),
    );

    let mut last = flat(&model.forward(&ids(&tokens[..GREEDY_PROMPT]), 0).unwrap());
    let (mut chosen, mut rows) = (Vec::new(), Vec::new());
    for i in 0..GREEDY_TOKENS {
        let id = argmax(&last);
        chosen.push(id);
        rows.extend_from_slice(&last);
        last = flat(
            &model
                .forward(&ids(&[id as u32]), GREEDY_PROMPT + i)
                .unwrap(),
        );
    }
    let cols = last.len();
    put("greedy_ids", Record::Ids(chosen));
    put("greedy_logits", Record::Logits(GREEDY_TOKENS, cols, rows));
}

/// A record's shape and the digest of its bits, as the frozen records hold it.
pub(super) fn record_digest(r: &Record) -> String {
    match r {
        Record::Logits(rows, cols, v) => {
            format!(
                "logits {rows}x{cols} {}",
                super::wide_sparse_frozen::digest_f32(v)
            )
        }
        Record::Ids(ids) => format!("ids {ids:?}"),
    }
}

/// Asserts every record of `got` has the bits recorded under `prefix` and its name in `file`.
pub(super) fn expect_records(file: &str, prefix: &str, got: &BTreeMap<String, Record>) {
    for (name, r) in got {
        super::wide_sparse_frozen::expect(file, &format!("{prefix}{name}"), &record_digest(r));
    }
}

/// The first row of `got` farther than the frozen oracle's tolerance from `want`'s.
fn frozen_difference(want: &Record, got: &Record) -> Option<String> {
    match (want, got) {
        (Record::Ids(w), Record::Ids(g)) => (w != g).then(|| format!("ids {g:?} != {w:?}")),
        (Record::Logits(wr, wc, w), Record::Logits(gr, gc, g)) => {
            if (wr, wc) != (gr, gc) {
                return Some(format!("shape [{gr}, {gc}] != [{wr}, {wc}]"));
            }
            for (row, (a, b)) in w.chunks_exact(*wc).zip(g.chunks_exact(*wc)).enumerate() {
                let num: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                let den: f32 = a.iter().map(|x| x * x).sum();
                let rel = (num / den).sqrt();
                if rel.is_nan() || rel > FROZEN_ROW_TOLERANCE {
                    return Some(format!("row {row} relative L2 {rel}"));
                }
            }
            None
        }
        _ => Some("kind changed".to_string()),
    }
}

/// Every record of `want` that `got` does not reproduce within the oracle's tolerance, in the
/// order the records build on each other.
fn differences(want: &BTreeMap<String, Record>, got: &BTreeMap<String, Record>) -> Vec<String> {
    let mut out = Vec::new();
    for n in 0..LAYOUTS.len() {
        for record in RECORDS {
            let name = format!("l{n}_{record}");
            match (want.get(&name), got.get(&name)) {
                (Some(w), Some(g)) => {
                    if let Some(d) = frozen_difference(w, g) {
                        out.push(format!("{name}: {d}"));
                    }
                }
                _ => out.push(format!("{name}: missing")),
            }
        }
    }
    out
}

/// The frozen records.
fn read_frozen() -> BTreeMap<String, Record> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/vectors/deepseek_v41_oracle/vectors.safetensors");
    let bytes = std::fs::read(path).unwrap();
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

/// Every block of the synthetic file is detected as latent attention over 1 key head, whatever
/// its role in the band.
#[test]
fn every_block_of_the_synthetic_file_is_latent_attention_over_one_key_head() {
    for (ratios, kv_src, index_src) in LAYOUTS {
        let g = toy_file(ratios, kv_src, index_src);
        for il in 0..ratios.len() {
            assert_eq!(layer_mixer_kind(&g.content, il), MixerKind::LatentMqa);
        }
    }
}

/// The generic route gives the recorded bits of the V4.1 path on every record of both layouts:
/// the prefill, the decode, the verify, the rewound verify and the decode after it, and the
/// greedy run.
#[test]
fn the_generic_route_gives_the_bits_of_the_v41_path() {
    let mut got = BTreeMap::new();
    for (n, (ratios, kv_src, index_src)) in LAYOUTS.into_iter().enumerate() {
        let mut model = generic(ratios, kv_src, index_src);
        record_generic(&mut model, &prompt(), n, &mut got);
    }
    assert_eq!(got.len(), LAYOUTS.len() * RECORDS.len());
    expect_records("oracle", "", &got);
}

/// The generic route reproduces the frozen V4.1 logits and greedy tokens, within the oracle's
/// tolerance.
#[test]
fn the_generic_route_reproduces_the_frozen_oracle() {
    let want = read_frozen();
    let mut got = BTreeMap::new();
    for (n, (ratios, kv_src, index_src)) in LAYOUTS.into_iter().enumerate() {
        let tokens = prompt();
        let mut model = generic(ratios, kv_src, index_src);
        record_generic(&mut model, &tokens, n, &mut got);
    }
    let differ = differences(&want, &got);
    assert!(
        differ.is_empty(),
        "the generic route does not reproduce the frozen oracle:\n{}",
        differ.join("\n")
    );
}
