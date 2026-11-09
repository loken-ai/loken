//! The calibrate request on the generic route against the V4.1 calibration.
//!
//! The checkpoints are the ones of the block-scaled reader tests, without and with engram
//! layers, each with a word-level tokenizer.
//! The V4.1 side runs the steps the request ran before: its own configuration, its own model,
//! its own excerpts and row count.
//! The generic side runs the request's job with the model on the host.
//! The 2 record files are compared byte for byte.
//!
//! Only a host build is compared: a card build of the V4.1 calibration runs experts on the cards.

use std::path::{Path, PathBuf};

use super::block_scaled_engram_tests::{engram_checkpoint, tokenizer};
use super::block_scaled_oracle_tests::checkpoint;
use super::wide_sparse_calibrate::{calibrate_checkpoint, excerpts, rows_to_keep, Request};
use super::wide_sparse_config::WideSparseMeta;
use super::{BlockScaledCheckpoint, GenericHeteroTransformer};
use crate::inference::load::calibration;
use crate::inference::model::deepseek_v41::convert as v41;
use crate::inference::model::deepseek_v41::safetensors_source::SafeTensorsSource;
use crate::inference::model::deepseek_v41::source::WeightSource;
use crate::inference::place::layer_executor::HeteroPlan;

/// Tokens per excerpt.
const EXCERPT: usize = 8;
/// Tokens in all: 5 excerpts over 2 rounds of the texts.
const TOKENS: usize = 40;
/// Rows a record keeps when it samples.
const FEW_ROWS: usize = 3;

/// The texts a request carries: words of the tokenizer, of 3 lengths.
fn texts() -> Vec<String> {
    [30usize, 17, 9]
        .iter()
        .enumerate()
        .map(|(k, &n)| {
            (0..n)
                .map(|i| format!("t{}", (i * 5 + k * 11 + 1) % 47))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect()
}

/// The checkpoint of the block-scaled reader tests with the tokenizer of the engram tests.
fn plain_checkpoint(tag: &str) -> PathBuf {
    let dir = checkpoint(tag);
    tokenizer().save(dir.join("tokenizer.json"), false).unwrap();
    dir
}

/// The record the calibrate request wrote before, at `out`, by the V4.1 steps it ran.
fn v41_record(dir: &Path, request: &Request<'_>, out: &Path) {
    let src = SafeTensorsSource::open(dir).unwrap();
    let cfg = v41::released_config(&src, dir).unwrap();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).unwrap();
    let streams: Vec<Vec<u32>> = request
        .texts
        .iter()
        .map(|t| tok.encode(t.as_str(), false).unwrap().get_ids().to_vec())
        .collect();
    let chosen = v41::excerpts(&streams, request.excerpt, request.tokens);
    let inter = src.shape("layers.0.ffn.experts.0.w2.weight").unwrap()[1];
    let tokens: usize = chosen.iter().map(Vec::len).sum();
    let rows = v41::rows_to_keep(&cfg, inter, tokens, request.rows);
    let record = v41::calibrate(&src, &cfg, &chosen, rows, &|_, _| {}).unwrap();
    let note = format!(
        "{} excerpts of {} tokens from {} texts",
        chosen.len(),
        request.excerpt,
        request.texts.len()
    );
    record.write(out, &note).unwrap();
}

/// The generic model of the checkpoint, every layer on the host.
fn on_host(
    ckpt: &BlockScaledCheckpoint,
    meta: &WideSparseMeta,
    _context: usize,
) -> crate::tensor::Result<GenericHeteroTransformer> {
    let plan = HeteroPlan::calculate_with_kv(meta.n_layers, 1 << 20, &[], &[], 1.0, 0);
    GenericHeteroTransformer::from_block_scaled(
        ckpt,
        &plan,
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
    )
}

/// For each checkpoint and row count, the calibrate request's job writes the record file of the
/// V4.1 calibration byte for byte.
/// A record that samples keeps fewer rows than it saw, so the sampling is compared too.
#[test]
fn the_calibrate_request_writes_the_record_of_the_v41_calibration() {
    let texts = texts();
    for (tag, dir) in [
        ("plain", plain_checkpoint("calibrate-plain")),
        ("engram", engram_checkpoint("calibrate-engram")),
    ] {
        for rows in [Some(FEW_ROWS), None] {
            let request = Request {
                texts: &texts,
                excerpt: EXCERPT,
                tokens: TOKENS,
                rows,
            };
            let what = format!("{tag} {rows:?}");
            let old_out = dir.join("record-v41.gguf");
            let new_out = dir.join("record-generic.gguf");
            v41_record(&dir, &request, &old_out);
            let layers = std::sync::Mutex::new(Vec::new());
            calibrate_checkpoint(&dir, &request, &new_out, on_host, &|done, total| {
                layers.lock().unwrap().push((done, total));
            })
            .unwrap();
            let (a, b) = (
                std::fs::read(&old_out).unwrap(),
                std::fs::read(&new_out).unwrap(),
            );
            assert_eq!(a.len(), b.len(), "{what}: file lengths");
            if let Some(at) = a.iter().zip(&b).position(|(x, y)| x != y) {
                panic!("{what}: the files first differ at byte {at}");
            }
            // Every layer was reported once, in order.
            let reported = layers.into_inner().unwrap();
            let n = reported.len();
            assert!(n > 0, "{what}");
            assert!(
                reported.iter().enumerate().all(|(i, &p)| p == (i + 1, n)),
                "{what}"
            );

            // The record covers every layer's block input, and each expert's rows when kept.
            let read = calibration::read(&new_out).unwrap();
            for l in 0..n {
                let (importance, kept, _) = &read[&format!("blk.{l}.ffn_inp")];
                assert!(
                    !importance.is_empty() && !kept.is_empty(),
                    "{what}: layer {l}"
                );
            }
            let g = crate::tensor::quantized::gguf_file::open_mapped(&new_out).unwrap();
            let seen = g.content.metadata["calibration.rows_seen"]
                .to_string()
                .unwrap()
                .clone();
            let input_seen: usize = seen
                .split(' ')
                .find_map(|s| s.strip_prefix("blk.0.ffn_inp:"))
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(input_seen, TOKENS, "{what}");
            let input_kept = g.content.tensor_infos["blk.0.ffn_inp.rows"].shape.dims()[0];
            match rows {
                Some(cap) => assert_eq!(input_kept, cap, "{what}"),
                None => assert_eq!(input_kept, TOKENS, "{what}"),
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// The excerpts and the row count follow the V4.1 rules.
#[test]
fn excerpts_and_rows_are_the_v41_ones() {
    let streams: Vec<Vec<u32>> = [25usize, 3, 16, 0]
        .iter()
        .map(|&n| (0..n as u32).collect())
        .collect();
    for (length, budget) in [(4, 40), (8, 20), (5, 3), (30, 100), (1, 7)] {
        assert_eq!(
            excerpts(&streams, length, budget),
            v41::excerpts(&streams, length, budget),
            "{length} {budget}"
        );
    }
    let dir = plain_checkpoint("calibrate-rows");
    let ours = BlockScaledCheckpoint::open(&dir).unwrap();
    let src = SafeTensorsSource::open(&dir).unwrap();
    let meta = super::wide_sparse_convert::checkpoint_config(&ours, &dir).unwrap();
    let cfg = v41::released_config(&src, &dir).unwrap();
    for (tokens, asked) in [(40, None), (40, Some(5))] {
        assert_eq!(
            rows_to_keep(&meta, 256, tokens, asked),
            v41::rows_to_keep(&cfg, 256, tokens, asked),
            "{tokens} {asked:?}"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}
