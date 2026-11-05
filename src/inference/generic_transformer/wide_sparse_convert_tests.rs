//! The generic converter of block-scaled checkpoints against the V4.1 converter.
//!
//! Both converters read the small checkpoint of the block-scaled reader tests, its head grown so
//! the always-read format applies.
//! Both take the same calibration records and the same formats.
//! The 2 files they write are compared byte for byte.
//!
//! Only a host build is compared: a card build of the V4.1 converter quantises on the cards.

use std::path::{Path, PathBuf};

use safetensors::tensor::TensorView;
use safetensors::Dtype;

use super::block_scaled_oracle_tests::{checkpoint_of, entries};
use super::wide_sparse_convert::{checkpoint_config, model_name, plan};
use super::BlockScaledCheckpoint;
use crate::inference::load::block_scaled_convert::{self, ConvertPlan, Formats, Produce};
use crate::inference::load::calibrated::{Recipe, Records};
use crate::inference::model::deepseek_v41::convert as v41;
use crate::inference::model::deepseek_v41::engram::EngramConfig;
use crate::inference::model::deepseek_v41::safetensors_source::SafeTensorsSource;
use crate::inference::model::deepseek_v41::DeepseekV41Config;
use crate::tensor::quantized::GgmlDType;

/// Rows of the head: with the checkpoint's width, enough values for the always-read format.
const HEAD_ROWS: usize = 4096;
/// Rows a record keeps when it is enough for the correction.
const KEPT: usize = 12;
/// Rows a record keeps when it is too few for the correction.
const FEW: usize = 2;
/// The recipe of a conversion request that names none.
const RECIPE: Recipe = Recipe {
    min_rows: 8,
    damping: 1.0,
};
/// The model name both files carry.
const NAME: &str = "DeepSeek-V4.1-Flash";
/// The engram layers of the engram checkpoint.
const ENGRAM_LAYERS: [usize; 2] = [1, 2];
/// Values of an engram table row.
const ENGRAM_DIM: usize = 32;
/// Rows of each engram layer's table.
const ENGRAM_ROWS: [usize; 2] = [40, 24];
/// Rows of an engram layer's query and key weights.
const ENGRAM_COPIES: usize = 4;

/// The mixes of formats a conversion request names.
fn mixes() -> [(&'static str, Formats); 4] {
    use GgmlDType::*;
    [
        ("default", Formats::default()),
        (
            "iq2-q6attn",
            Formats {
                gate_up: Iq2Xxs,
                down: Q2K,
                always_read: Q6K,
            },
        ),
        (
            "iq2-iq3",
            Formats {
                gate_up: Iq2Xxs,
                down: Iq3Xxs,
                always_read: Q8_0,
            },
        ),
        (
            "iq3",
            Formats {
                gate_up: Iq3Xxs,
                down: Iq3Xxs,
                always_read: Q8_0,
            },
        ),
    ]
}

/// The checkpoint of the block-scaled reader tests with a head of `HEAD_ROWS` rows.
fn fixture(tag: &str) -> PathBuf {
    let mut all = entries();
    for e in all.iter_mut().filter(|e| e.reference == "head.weight") {
        e.shape[0] = HEAD_ROWS;
    }
    checkpoint_of(tag, &all)
}

/// Adds to `dir` a shard of engram layers: an e4m3 table with row scales, a bf16 table, and
/// each layer's query, key and value weights in F32.
fn add_engram(dir: &Path, dim: usize) {
    let f32s = |n: usize, seed: usize| -> Vec<u8> {
        (0..n)
            .flat_map(|i| (((i * 31 + seed * 7) % 97) as f32 * 0.01 - 0.4).to_le_bytes())
            .collect()
    };
    let mut tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    for (k, (&layer, &rows)) in ENGRAM_LAYERS.iter().zip(&ENGRAM_ROWS).enumerate() {
        let stem = format!("layers.{layer}.engram");
        if k == 0 {
            // An e4m3 byte keeps an exponent field below 7, a ue8m0 byte below 127.
            let bytes = (0..rows * ENGRAM_DIM)
                .map(|i| {
                    let sign = if i.is_multiple_of(3) { 0x80 } else { 0 };
                    sign | ((i * 13 + 5) % 0x38) as u8
                })
                .collect();
            let scales = (0..rows).map(|r| 120 + (r % 6) as u8).collect();
            tensors.push((
                format!("{stem}.embed.weight"),
                Dtype::F8_E4M3,
                vec![rows, ENGRAM_DIM],
                bytes,
            ));
            tensors.push((
                format!("{stem}.embed.scale"),
                Dtype::F8_E8M0,
                vec![rows, 1],
                scales,
            ));
        } else {
            let bytes = (0..rows * ENGRAM_DIM)
                .flat_map(|i| {
                    let v = ((i * 17) % 89) as f32 * 0.02 - 0.9;
                    ((v.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect();
            tensors.push((
                format!("{stem}.embed.weight"),
                Dtype::BF16,
                vec![rows, ENGRAM_DIM],
                bytes,
            ));
        }
        for (i, part) in ["q_weight", "k_weight"].iter().enumerate() {
            tensors.push((
                format!("{stem}.{part}"),
                Dtype::F32,
                vec![ENGRAM_COPIES, dim],
                f32s(ENGRAM_COPIES * dim, 10 * k + i),
            ));
        }
        tensors.push((
            format!("{stem}.wkv.weight"),
            Dtype::F32,
            vec![2 * ENGRAM_DIM, dim],
            f32s(2 * ENGRAM_DIM * dim, 10 * k + 5),
        ));
    }
    let views: Vec<(String, TensorView)> = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect();
    safetensors::serialize_to_file(views, None, &dir.join("model-engram.safetensors")).unwrap();
}

/// The hash layout of the engram shard's layers.
fn engram_config(vocab: usize) -> EngramConfig {
    EngramConfig {
        layer_ids: ENGRAM_LAYERS.to_vec(),
        n_heads: 1,
        head_dim: ENGRAM_DIM,
        max_ngram: 2,
        token_map: (0..vocab).map(|t| (t % 5) as u32).collect(),
        pad_id: 0,
        multipliers: vec![vec![3, 5], vec![7, 11]],
        primes: vec![vec![13], vec![17]],
        offsets: vec![vec![0], vec![0]],
    }
}

/// A record of `n` rows of `cols` values.
///
/// Its values are multiples of 1/16, so the ridge's sum of squares is exact in any order.
fn record(cols: usize, n: usize, seed: usize) -> (Vec<f32>, Vec<f32>, usize) {
    let importance = (0..cols)
        .map(|j| 1.0 + ((j + seed) % 7) as f32 / 8.0)
        .collect();
    let rows = (0..n * cols)
        .map(|i| ((i * 7 + seed * 3) % 33) as f32 / 16.0 - 1.0)
        .collect();
    (importance, rows, n)
}

/// The records under the names the V4.1 calibration writes them.
///
/// Layer 0 has own rows for some experts, 1 expert with too few, the block's input and the pooled
/// down importance.
/// Layer 1 has the block's input and the pooled down importance alone.
/// Layer 2 has no record.
fn records(src: &BlockScaledCheckpoint) -> Records {
    let cols = |w: &str| {
        src.stored_shape(&format!("layers.0.ffn.experts.0.{w}.weight"))
            .unwrap()[1]
    };
    let (dim, inter) = (cols("w1"), cols("w2"));
    let mut r = Records::new();
    for e in 0..3 {
        r.insert(format!("blk.0.ffn_gate_up_exps.{e}"), record(dim, KEPT, e));
    }
    r.insert("blk.0.ffn_gate_up_exps.3".into(), record(dim, FEW, 3));
    for e in 0..2 {
        r.insert(
            format!("blk.0.ffn_down_exps.{e}"),
            record(inter, KEPT, 10 + e),
        );
    }
    for l in 0..2 {
        r.insert(format!("blk.{l}.ffn_inp"), record(dim, KEPT, 20 + l));
        r.insert(format!("blk.{l}.ffn_down_exps"), record(inter, 0, 30 + l));
    }
    r
}

/// The planned format of tensor `name`.
fn planned(plan: &ConvertPlan, name: &str) -> GgmlDType {
    plan.tensors
        .iter()
        .find(|(e, _)| e.name == name)
        .unwrap_or_else(|| panic!("no planned tensor {name}"))
        .0
        .dtype
}

/// Converts `dir` with both converters into `{tag}-v41.gguf` and `{tag}-generic.gguf`, and
/// asserts the 2 files and their digests are equal.
fn both_write_the_same_file(
    dir: &Path,
    cfg: &DeepseekV41Config,
    formats: Formats,
    layers: std::ops::Range<usize>,
    tag: &str,
) -> ConvertPlan {
    let old_src = SafeTensorsSource::open(dir).unwrap();
    let ours = BlockScaledCheckpoint::open(dir).unwrap();
    let records = records(&ours);
    let old_out = dir.join(format!("{tag}-v41.gguf"));
    let old_formats = v41::Formats {
        gate_up: formats.gate_up,
        down: formats.down,
        always_read: formats.always_read,
    };
    let old_digest = v41::convert(
        &old_src,
        cfg,
        &records,
        &old_out,
        old_formats,
        RECIPE,
        layers.clone(),
        None,
        0,
        &|_, _, _| {},
    )
    .unwrap();
    let new_out = dir.join(format!("{tag}-generic.gguf"));
    let p = plan(&ours, cfg, formats, layers, NAME, None).unwrap();
    let new_digest =
        block_scaled_convert::convert(&ours, &p, &records, RECIPE, &new_out, 0, &|_, _, _| {})
            .unwrap();
    let (a, b) = (
        std::fs::read(&old_out).unwrap(),
        std::fs::read(&new_out).unwrap(),
    );
    assert_eq!(a.len(), b.len(), "{tag}: file lengths");
    if let Some(at) = a.iter().zip(&b).position(|(x, y)| x != y) {
        panic!("{tag}: the files first differ at byte {at}");
    }
    assert_eq!(old_digest, new_digest, "{tag}: digests");
    p
}

/// For each mix, the generic converter writes the bytes of the V4.1 converter, every layer.
#[test]
fn every_mix_writes_the_bytes_of_the_v41_converter() {
    let dir = fixture("convert-mixes");
    let src = SafeTensorsSource::open(&dir).unwrap();
    let cfg = DeepseekV41Config::from_reference_config(&src.inference_config, None).unwrap();
    for (tag, formats) in mixes() {
        let p = both_write_the_same_file(&dir, &cfg, formats, 0..cfg.n_layers, tag);
        // The fixture reaches every format of the mix.
        assert_eq!(planned(&p, "output.weight"), formats.always_read, "{tag}");
        assert_eq!(planned(&p, "output_norm.weight"), GgmlDType::F32, "{tag}");
        for l in 0..cfg.n_layers {
            let stack = |s: &str| planned(&p, &format!("blk.{l}.{s}.weight"));
            assert_eq!(stack("ffn_gate_exps"), formats.gate_up, "{tag}");
            assert_eq!(stack("ffn_up_exps"), formats.gate_up, "{tag}");
            assert_eq!(stack("ffn_down_exps"), formats.down, "{tag}");
        }
        // Layer 0's gate and up stacks read the records the test wrote.
        let Some((_, Produce::Experts { records: names, .. })) = p
            .tensors
            .iter()
            .find(|(e, _)| e.name == "blk.0.ffn_gate_exps.weight")
        else {
            panic!("{tag}: layer 0's gate stack is not an expert stack");
        };
        let ours = BlockScaledCheckpoint::open(&dir).unwrap();
        let written = records(&ours);
        assert!(names.own[..3].iter().all(|n| written.contains_key(n)));
        assert!(written.contains_key(&names.shared));
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// With engram layers, the generic converter writes the bytes of the V4.1 converter, for the
/// whole file and for a file of 1 middle layer.
#[test]
fn engram_layers_and_a_partial_file_write_the_bytes_of_the_v41_converter() {
    let dir = fixture("convert-engram");
    let src = SafeTensorsSource::open(&dir).unwrap();
    let mut cfg = DeepseekV41Config::from_reference_config(&src.inference_config, None).unwrap();
    add_engram(&dir, cfg.d_model);
    cfg.engram = Some(engram_config(cfg.vocab_size));
    let layers = 0..cfg.n_layers;
    let whole = both_write_the_same_file(&dir, &cfg, Formats::default(), layers, "whole");
    // The whole file carries both engram layers' gates and 1 table of both tables' rows.
    for l in ENGRAM_LAYERS {
        assert_eq!(
            planned(&whole, &format!("blk.{l}.engram_gate.weight")),
            GgmlDType::F32
        );
    }
    let table = whole
        .tensors
        .iter()
        .find(|(_, p)| matches!(p, Produce::Table { .. }))
        .expect("an engram table");
    assert_eq!(table.0.dims, vec![ENGRAM_ROWS.iter().sum(), ENGRAM_DIM]);
    let part = both_write_the_same_file(&dir, &cfg, Formats::default(), 1..2, "middle");
    // The partial file holds layer 1 alone, its engram table among its tensors.
    assert!(part
        .tensors
        .iter()
        .all(|(e, _)| !e.name.starts_with("blk.0.") && !e.name.starts_with("blk.2.")));
    assert!(part
        .tensors
        .iter()
        .any(|(_, p)| matches!(p, Produce::Table { .. })));
    let _ = std::fs::remove_dir_all(&dir);
}

/// The hub id of the checkpoint a convert request names.
const CHECKPOINT: &str = "deepseek-ai/DeepSeek-V4.1-Flash";

/// Writes beside the checkpoint in `dir` a word-level `tokenizer.json` of `vocab` ids, its last 2
/// the sequence and end tokens, and the `tokenizer_config.json` that names them.
fn add_tokenizer(dir: &Path, vocab: usize) {
    use tokenizers::models::wordlevel::WordLevel;
    let words: ahash::AHashMap<String, u32> = (0..vocab as u32 - 2)
        .map(|i| (format!("t{i}"), i))
        .collect();
    let model = WordLevel::builder()
        .vocab(words)
        .unk_token("t0".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.add_special_tokens(&[
        tokenizers::AddedToken::from("<bos>", true),
        tokenizers::AddedToken::from("<eos>", true),
    ]);
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();
    let config = serde_json::json!({ "bos_token": "<bos>", "eos_token": "<eos>" });
    std::fs::write(dir.join("tokenizer_config.json"), config.to_string()).unwrap();
}

/// The model name is the last segment of the checkpoint id, the name the V4.1 converter writes.
#[test]
fn the_model_name_is_the_last_segment_of_the_checkpoint_id() {
    assert_eq!(model_name(CHECKPOINT), NAME);
    assert_eq!(model_name(NAME), NAME);
}

/// The steps a convert request runs on the generic route write the bytes of the V4.1 converter
/// run as the request ran it before: its own configuration, its model name, the tokenizer files.
#[test]
fn the_convert_request_writes_the_bytes_of_the_v41_converter() {
    let dir = fixture("convert-request");
    let ours = BlockScaledCheckpoint::open(&dir).unwrap();
    let vocab = DeepseekV41Config::from_reference_config(&ours.reference, None)
        .unwrap()
        .vocab_size;
    add_tokenizer(&dir, vocab);
    let old_src = SafeTensorsSource::open(&dir).unwrap();
    let old_cfg = v41::released_config(&old_src, &dir).unwrap();
    let cfg = checkpoint_config(&ours, &dir).unwrap();
    assert_eq!(
        format!("{cfg:?}"),
        format!("{old_cfg:?}"),
        "the configurations"
    );
    let tokenizer = v41::TokenizerFiles::read(&dir).unwrap();
    let records = records(&ours);
    let formats = Formats::default();
    let old_formats = v41::Formats {
        gate_up: formats.gate_up,
        down: formats.down,
        always_read: formats.always_read,
    };
    let old_out = dir.join("request-v41.gguf");
    let old_digest = v41::convert(
        &old_src,
        &old_cfg,
        &records,
        &old_out,
        old_formats,
        RECIPE,
        0..old_cfg.n_layers,
        Some(&tokenizer),
        0,
        &|_, _, _| {},
    )
    .unwrap();
    let new_out = dir.join("request-generic.gguf");
    let p = plan(
        &ours,
        &cfg,
        formats,
        0..cfg.n_layers,
        model_name(CHECKPOINT),
        Some(&tokenizer),
    )
    .unwrap();
    let new_digest =
        block_scaled_convert::convert(&ours, &p, &records, RECIPE, &new_out, 0, &|_, _, _| {})
            .unwrap();
    let (a, b) = (
        std::fs::read(&old_out).unwrap(),
        std::fs::read(&new_out).unwrap(),
    );
    assert_eq!(a.len(), b.len(), "file lengths");
    if let Some(at) = a.iter().zip(&b).position(|(x, y)| x != y) {
        panic!("the files first differ at byte {at}");
    }
    assert_eq!(old_digest, new_digest, "digests");
    let _ = std::fs::remove_dir_all(&dir);
}
