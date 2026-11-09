//! The calibration record of a block-scaled checkpoint of a wide sparse model.
//!
//! The generic model of the checkpoint runs over excerpts of texts.
//! Every excerpt crosses a layer before any goes on to the next.
//! Each layer's routed experts are then read once for the whole run.
//! The record keeps, per layer, what its routed feed-forward took in.
//! It keeps, per routed expert, the rows routed to it and the rows its down projection took.
//! A conversion reads the record to fit each expert's low-bit format to what it saw.

use super::wide_sparse_config::WideSparseMeta;
use super::wide_sparse_convert::{
    checkpoint_config, down_pooled, down_seen, gate_up_seen, moe_input,
};
use super::{BlockScaledCheckpoint, GenericHeteroTransformer};
use crate::inference::load::calibration::Calibration;
use crate::tensor::{Error, Result};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Excerpts of `length` tokens, taken in turn from each text until `budget` tokens are chosen.
///
/// Round `r` takes from each text in order its tokens `r * length..(r + 1) * length`.
/// A text too short for the round gives nothing in it.
/// The same texts always give the same excerpts.
pub fn excerpts(texts: &[Vec<u32>], length: usize, budget: usize) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let (mut taken, mut round) = (0usize, 0usize);
    while taken + length <= budget {
        let mut any = false;
        for t in texts {
            let start = round * length;
            if start + length > t.len() {
                continue;
            }
            if taken + length > budget {
                break;
            }
            out.push(t[start..start + length].to_vec());
            taken += length;
            any = true;
        }
        if !any {
            break;
        }
        round += 1;
    }
    out
}

/// The most rows a record keeps, `asked` when given.
///
/// A run over `tokens` tokens keeps every row while all of them fit in half the memory
/// available now.
/// Per token and layer it sees the block's input once, and each chosen expert's input and down
/// input.
/// A longer run keeps a sample sized so the whole record fits, never below the floor the
/// correction needs.
pub fn rows_to_keep(
    meta: &WideSparseMeta,
    inter: usize,
    tokens: usize,
    asked: Option<usize>,
) -> usize {
    if let Some(rows) = asked {
        return rows;
    }
    let used = meta.mixture.used;
    let per_token = meta.n_layers * (meta.d_model + used * (meta.d_model + inter));
    let bytes_all = tokens
        .saturating_mul(per_token)
        .saturating_mul(std::mem::size_of::<f32>());
    let budget = {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.available_memory() as usize / 2
    };
    if bytes_all <= budget {
        return usize::MAX;
    }
    (tokens * budget / bytes_all.max(1)).max(used * 64)
}

/// The record of `model` over `prompts`, at most `rows_kept` rows per record.
///
/// The prompts run layer after layer, each from position 0.
/// `progress` is told the layers done and the layers in all.
pub fn record(
    model: &mut GenericHeteroTransformer,
    prompts: &[Vec<u32>],
    rows_kept: usize,
    progress: &dyn Fn(usize, usize),
) -> Result<Calibration> {
    let dim = model.config.embedding_length;
    let record = Arc::new(Mutex::new(Calibration::default()));
    let sink = record.clone();
    model.observe_experts(Some(Arc::new(
        move |layer: usize, expert: Option<usize>, rows: &[f32], down: &[f32]| {
            let mut c = sink.lock().unwrap_or_else(|e| e.into_inner());
            match expert {
                None => {
                    for row in rows.chunks(dim) {
                        c.observe(&moe_input(layer), row, rows_kept);
                    }
                }
                Some(e) => {
                    for row in rows.chunks(dim) {
                        c.observe(&gate_up_seen(layer, e), row, rows_kept);
                    }
                    let n = rows.len() / dim;
                    if n > 0 {
                        for row in down.chunks(down.len() / n) {
                            c.observe(&down_seen(layer, e), row, rows_kept);
                            // The pooled record lends its importance only: it keeps no rows.
                            c.observe(&down_pooled(layer), row, 0);
                        }
                    }
                }
            }
        },
    )));
    let run = model.prefill_by_layer(prompts, progress);
    model.observe_experts(None);
    run?;
    Arc::try_unwrap(record)
        .map_err(|_| Error::msg("calibration: the record is still shared"))
        .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()))
}

/// What a calibrate request asks for.
pub struct Request<'a> {
    /// The texts excerpts are taken from, in turn.
    pub texts: &'a [String],
    /// Tokens per excerpt.
    pub excerpt: usize,
    /// Tokens in all.
    pub tokens: usize,
    /// Rows a record keeps, by [`rows_to_keep`] when `None`.
    pub rows: Option<usize>,
}

/// Writes at `out` the calibration record of the checkpoint in `dir` over the excerpts `request`
/// asks for.
///
/// `load` gives the generic model of the checkpoint for a context of the given tokens.
/// `progress` is told the layers done and the layers in all.
pub fn calibrate_checkpoint(
    dir: &Path,
    request: &Request<'_>,
    out: &Path,
    load: impl FnOnce(
        &BlockScaledCheckpoint,
        &WideSparseMeta,
        usize,
    ) -> Result<GenericHeteroTransformer>,
    progress: &dyn Fn(usize, usize),
) -> Result<()> {
    let ckpt = BlockScaledCheckpoint::open(dir)?;
    let meta = checkpoint_config(&ckpt, dir)?;
    let tokenizer_path = dir.join("tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| Error::msg(format!("{}: {e}", tokenizer_path.display())))?;
    let streams = request
        .texts
        .iter()
        .map(|t| {
            tok.encode(t.as_str(), false)
                .map(|ids| ids.get_ids().to_vec())
        })
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Error::msg(e.to_string()))?;
    let chosen = excerpts(&streams, request.excerpt, request.tokens);
    if chosen.is_empty() {
        return Err(Error::msg(format!(
            "calibrate: no text holds an excerpt of {} tokens",
            request.excerpt
        )));
    }
    let inter = ckpt
        .stored_shape("layers.0.ffn.experts.0.w2.weight")
        .map(|s| s[1])
        .ok_or_else(|| Error::msg("calibrate: the checkpoint has no routed expert"))?;
    let tokens: usize = chosen.iter().map(Vec::len).sum();
    let rows = rows_to_keep(&meta, inter, tokens, request.rows);
    let longest = chosen.iter().map(Vec::len).max().unwrap_or(0);
    let mut model = load(&ckpt, &meta, longest)?;
    let record = record(&mut model, &chosen, rows, progress)?;
    drop(model);
    let note = format!(
        "{} excerpts of {} tokens from {} texts",
        chosen.len(),
        request.excerpt,
        request.texts.len()
    );
    record.write(out, &note)
}
