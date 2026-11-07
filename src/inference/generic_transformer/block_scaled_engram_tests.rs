//! Engram layers of a block-scaled checkpoint on the generic route against the V4.1 path.
//!
//! The checkpoint is the one of `block_scaled_oracle_tests` with 2 engram layers.
//! It ships a word-level `tokenizer.json` in which some tokens normalise alike.
//! Each engram table is stored as e4m3 rows beside ue8m0 scales, as the release stores it.
//! Both paths open the same directory and build the token map from its tokenizer.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

use super::block_scaled_oracle_tests::{
    checkpoint_with, entries, entry, generic_model, reference_config, tokens, Entry, Format, DIM,
    STREAMS, VOCAB,
};
use super::ngram::engram_hash::primes_and_offsets;
use super::wide_sparse_oracle_tests::{
    bit_difference, record_generic, record_v41, Record, RECORDS,
};
use crate::inference::model::deepseek_v41::engram::EngramConfig;
use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
use crate::inference::model::deepseek_v41::safetensors_source::SafeTensorsSource;
use crate::inference::model::deepseek_v41::{token_map, DeepseekV41Config};

/// The engram layers: a window layer and the band layer of the checkpoint.
pub(super) const ENGRAM_LAYERS: [usize; 2] = [0, 1];
/// Heads of each n-gram size.
const ENGRAM_HEADS: usize = 2;
/// Width of 1 table row.
const ENGRAM_HEAD_DIM: usize = 32;
/// Longest n-gram hashed.
const MAX_NGRAM: usize = 3;
/// The engram vocabulary the bucket primes start from.
const ENGRAM_VOCAB: u64 = 10;
/// The token whose compressed id a slot before the sequence start takes.
const PAD: usize = 2;
/// Values of a table row that share 1 scale.
const TABLE_RUN: usize = 16;
/// Every eighth token is the upper-case spelling of the token before it.
const FOLD: usize = 8;

/// The tokenizer the checkpoint ships.
///
/// Token `i` is `t{i}`, except every `FOLD`th, which spells the token before it in upper case.
pub(crate) fn tokenizer() -> tokenizers::Tokenizer {
    let vocab: ahash::AHashMap<String, u32> = (0..VOCAB)
        .map(|i| {
            let word = if i % FOLD == FOLD - 1 {
                format!("T{}", i - 1)
            } else {
                format!("t{i}")
            };
            (word, i as u32)
        })
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("t0".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
}

/// Rows of every engram table: the largest sum of the bucket primes of 1 layer.
fn table_rows() -> usize {
    let (primes, _) =
        primes_and_offsets(ENGRAM_LAYERS.len(), MAX_NGRAM, ENGRAM_HEADS, ENGRAM_VOCAB);
    primes
        .iter()
        .map(|layer| layer.iter().sum::<u64>() as usize)
        .max()
        .unwrap()
}

/// The engram tensors of every engram layer.
///
/// The gate is the query and key weights, as the release stores it.
fn engram_entries() -> Vec<Entry> {
    let cols = (MAX_NGRAM - 1) * ENGRAM_HEADS;
    let mut out = Vec::new();
    for l in ENGRAM_LAYERS {
        let mut put = |reference: &str, loader: &str, format: Format, shape: &[usize]| {
            out.push(entry(
                format!("layers.{l}.{reference}"),
                format!("blk.{l}.{loader}"),
                format,
                shape,
            ));
        };
        put(
            "engram.embed.weight",
            "engram_embd",
            Format::Fp8Rows(TABLE_RUN),
            &[table_rows(), ENGRAM_HEAD_DIM],
        );
        put(
            "engram.wkv.weight",
            "engram_wkv",
            Format::Fp8,
            &[DIM * (STREAMS + 1), cols * ENGRAM_HEAD_DIM],
        );
        put("engram.q_weight", "engram_q", Format::F32, &[STREAMS, DIM]);
        put("engram.k_weight", "engram_k", Format::Bf16, &[STREAMS, DIM]);
    }
    out
}

/// Writes the checkpoint with engram layers into a new directory, its tokenizer files beside it.
pub(crate) fn engram_checkpoint(tag: &str) -> PathBuf {
    let tokenizer = tokenizer();
    let (_, compressed) = token_map::build(&tokenizer).unwrap();
    let mut reference = reference_config();
    let args = serde_json::json!({
        "engram_layer_ids": ENGRAM_LAYERS, "engram_n_heads": ENGRAM_HEADS,
        "engram_head_dim": ENGRAM_HEAD_DIM, "engram_max_ngram_size": MAX_NGRAM,
        "engram_vocab_size": ENGRAM_VOCAB, "engram_compressed_vocab_size": compressed,
        "engram_pad_id": PAD
    });
    let fields = reference.as_object_mut().unwrap();
    fields.extend(args.as_object().unwrap().clone());
    let mut all = entries();
    all.extend(engram_entries());
    let dir = checkpoint_with(tag, &all, &reference);
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
    let names = serde_json::json!({ "bos_token": "t1", "eos_token": "t2" });
    std::fs::write(dir.join("tokenizer_config.json"), names.to_string()).unwrap();
    dir
}

/// The V4.1 model of the checkpoint as its checkpoint loader builds it.
///
/// With `engram`, the hash layout comes from the shipped tokenizer, else the model runs without
/// its engram layers.
fn v41_model(dir: &Path, engram: bool) -> DeepseekV41Model {
    let src = SafeTensorsSource::open(dir).unwrap();
    let layout = engram.then(|| {
        let (map, _) = token_map::build_from_file(&dir.join("tokenizer.json")).unwrap();
        EngramConfig::derive(&src.inference_config, map).expect("a V4.1 engram layout")
    });
    let cfg = DeepseekV41Config::from_reference_config(&src.inference_config, layout).unwrap();
    DeepseekV41Model::load(&src, &cfg).unwrap()
}

fn prefill_bits(model: &DeepseekV41Model, tokens: &[u32]) -> Vec<u32> {
    let logits = model.forward_prefill(tokens).unwrap();
    let v = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    v.iter().map(|x| x.to_bits()).collect()
}

/// The generic model read from the checkpoint with engram layers gives the bits of the V4.1 path
/// read from the same directory: the prefill, the decode, the verify, the rewound verify and the
/// decode after it, and the greedy run.
/// The engram layers change the V4.1 prefill, so the records cover them.
#[test]
fn a_checkpoint_with_engram_layers_gives_the_bits_of_the_v41_path() {
    let dir = engram_checkpoint("engram");
    let v41 = v41_model(&dir, true);
    let plain = v41_model(&dir, false);
    let mut generic = generic_model(&dir);
    let tokens = tokens();
    assert_ne!(
        prefill_bits(&v41, &tokens),
        prefill_bits(&plain, &tokens),
        "the engram layers leave the V4.1 prefill unchanged"
    );
    let (mut want, mut got): (BTreeMap<String, Record>, BTreeMap<String, Record>) =
        (BTreeMap::new(), BTreeMap::new());
    record_v41(&v41, &tokens, 0, &mut want);
    record_generic(&mut generic, &tokens, 0, &mut got);
    let _ = std::fs::remove_dir_all(&dir);
    let mut differ = Vec::new();
    for record in RECORDS {
        let name = format!("l0_{record}");
        match (want.get(&name), got.get(&name)) {
            (Some(w), Some(g)) => {
                if let Some(d) = bit_difference(w, g) {
                    differ.push(format!("{name}: {d}"));
                }
            }
            _ => differ.push(format!("{name}: missing")),
        }
    }
    assert!(
        differ.is_empty(),
        "the generic model of the engram checkpoint differs from the V4.1 path:\n{}",
        differ.join("\n")
    );
}
