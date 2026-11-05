//! The plan of a GGUF in llama.cpp's layout from a block-scaled checkpoint of a wide sparse model.
//!
//! The plan lists the tensors the reference converter writes, in its order, in its formats.
//! The top tensors come first: embeddings, final norm, head.
//! Then per layer its tensors by sorted reference name, then its gate, up and down expert stacks.
//! Then the engram tables of every kept layer, stacked as 1 table.
//!
//! An engram layer's query and key weights are written as their product.
//! A routed expert stack names the calibration records of its experts and its layer.
//! The keys are the reference converter's, under the model name the caller gives.

use super::BlockScaledCheckpoint;
use crate::inference::load::block_scaled_convert::{
    bytes_per, written_format, ConvertPlan, Formats, Produce, StackRecords,
};
use crate::inference::model::deepseek_v41::convert::{tokenizer_metadata, TokenizerFiles};
use crate::inference::model::deepseek_v41::engram::EngramConfig;
use crate::inference::model::deepseek_v41::mainline;
use crate::inference::model::deepseek_v41::source::{llamacpp_name, ENGRAM_TABLE};
use crate::inference::model::deepseek_v41::{token_map, DeepseekV41Config};
use crate::tensor::gguf_write::PlannedEntry;
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use std::ops::Range;
use std::path::Path;

/// The record of what layer `layer`'s MoE block took in.
///
/// The record names are those `deepseek_v41/convert.rs` `calibrate` writes.
pub fn moe_input(layer: usize) -> String {
    format!("blk.{layer}.ffn_inp")
}

/// The record of the rows routed to expert `expert` of layer `layer`, as its gate and up see them.
pub fn gate_up_seen(layer: usize, expert: usize) -> String {
    format!("blk.{layer}.ffn_gate_up_exps.{expert}")
}

/// The record of the rows expert `expert` of layer `layer` hands its down projection.
pub fn down_seen(layer: usize, expert: usize) -> String {
    format!("blk.{layer}.ffn_down_exps.{expert}")
}

/// The record of every down projection of layer `layer`, importance only.
pub fn down_pooled(layer: usize) -> String {
    format!("blk.{layer}.ffn_down_exps")
}

/// The model name the file of the checkpoint whose hub id is `checkpoint` carries: the id's last
/// segment.
pub fn model_name(checkpoint: &str) -> &str {
    checkpoint.rsplit('/').next().unwrap_or(checkpoint)
}

/// The configuration of the checkpoint in `dir`, its reference arguments read by `src`.
///
/// The engram layers hash over the token map of the checkpoint's `tokenizer.json`.
pub fn checkpoint_config(src: &BlockScaledCheckpoint, dir: &Path) -> Result<DeepseekV41Config> {
    let (map, _) = token_map::build_from_file(&dir.join("tokenizer.json"))?;
    let engram = EngramConfig::derive(&src.reference, map);
    DeepseekV41Config::from_reference_config(&src.reference, engram)
}

/// The routed expert matrices of a layer in file order: gate, up, down, with their stack names.
const STACKS: [(&str, &str); 3] = [
    ("w1", "ffn_gate_exps"),
    ("w3", "ffn_up_exps"),
    ("w2", "ffn_down_exps"),
];

/// The file's keys: the configuration of its first `n_layers` layers, the context the checkpoint
/// declares, and the tokenizer as the checkpoint ships it and as llama.cpp's vocabulary arrays.
fn file_metadata(
    src: &BlockScaledCheckpoint,
    cfg: &DeepseekV41Config,
    n_layers: usize,
    engram_rows: &[usize],
    model_name: &str,
    tokenizer: Option<&TokenizerFiles>,
) -> Result<Vec<(String, Value)>> {
    let hf = src.config.get("text_config").unwrap_or(&src.config);
    let context_length = hf
        .get("max_position_embeddings")
        .and_then(|v| v.as_u64())
        .map_or_else(|| mainline::trained_context(cfg), |v| v as usize);
    let mut keys = Vec::new();
    if let Some(files) = tokenizer {
        keys.extend(tokenizer_metadata(files));
        keys.extend(mainline::vocab_metadata(&files.json, cfg.vocab_size)?);
    }
    let extras = mainline::Extras {
        name: model_name.to_string(),
        context_length,
        tokenizer: keys,
    };
    mainline::metadata(cfg, n_layers, engram_rows, &extras)
}

/// The plan of the file of layers `layers` of `src`, under the model name `model_name`.
///
/// The file holds the first `layers.end` layers' keys and the tensors of `layers` alone.
/// Routed experts take `formats.gate_up` and `formats.down`.
/// A large matrix takes `formats.always_read`, any other tensor F32.
pub fn plan(
    src: &BlockScaledCheckpoint,
    cfg: &DeepseekV41Config,
    formats: Formats,
    layers: Range<usize>,
    model_name: &str,
    tokenizer: Option<&TokenizerFiles>,
) -> Result<ConvertPlan> {
    let shape_of = |name: &str| {
        src.stored_shape(name)
            .ok_or_else(|| Error::msg(format!("no tensor {name}")))
    };
    let mut tensors: Vec<(PlannedEntry, Produce)> = Vec::new();
    let dense = |name: &str, shape: &[usize], tensors: &mut Vec<(PlannedEntry, Produce)>| {
        let Some(target) = llamacpp_name(name) else {
            return;
        };
        let elems: usize = shape.iter().product();
        let dtype = written_format(shape, formats.always_read);
        tensors.push((
            PlannedEntry {
                name: target,
                dims: shape.to_vec(),
                dtype,
                byte_len: bytes_per(dtype, elems),
            },
            Produce::Dense(name.to_string()),
        ));
    };

    for name in ["embed.weight", "norm.weight", "head.weight"] {
        dense(name, &shape_of(name)?, &mut tensors);
    }
    let mut names: Vec<String> = src.stored_names().into_iter().map(str::to_string).collect();
    names.sort();
    let n_layers = layers.end.min(cfg.n_layers);
    let below: Vec<usize> = mainline::engram_layers(cfg, n_layers)
        .into_iter()
        .filter_map(|i| cfg.engram.as_ref().map(|e| e.layer_ids[i]))
        .collect();
    // The offsets the keys declare count the rows of every engram layer below `n_layers`.
    let engram_rows = below
        .iter()
        .map(|l| shape_of(&format!("layers.{l}.engram.embed.weight")).map(|s| s[0]))
        .collect::<Result<Vec<usize>>>()?;
    let engram_layers: Vec<usize> = below.into_iter().filter(|l| layers.contains(l)).collect();
    let mut tables: Vec<String> = Vec::new();
    for layer in layers.start..n_layers {
        let prefix = format!("layers.{layer}.");
        for name in names.iter().filter(|n| n.starts_with(&prefix)) {
            if name.contains(".ffn.experts.") || name.ends_with(".scale") {
                continue;
            }
            // An engram layer's tables go into the 1 table, its query and key weights into its gate.
            if name.contains(".engram.") && !engram_layers.contains(&layer) {
                continue;
            }
            if name.ends_with("engram.embed.weight") {
                tables.push(name.clone());
                continue;
            }
            if name.ends_with("engram.q_weight") {
                let dims = shape_of(name)?;
                let elems: usize = dims.iter().product();
                let stem = format!("layers.{layer}.engram");
                let gate = format!("{stem}.gate.weight");
                // A stored product is taken as it is.
                let produce = if src.stored_shape(&gate).is_some() {
                    Produce::Dense(gate)
                } else {
                    Produce::Product(name.clone(), format!("{stem}.k_weight"))
                };
                tensors.push((
                    PlannedEntry {
                        name: format!("blk.{layer}.engram_gate.weight"),
                        dims,
                        dtype: GgmlDType::F32,
                        byte_len: bytes_per(GgmlDType::F32, elems),
                    },
                    produce,
                ));
                continue;
            }
            dense(name, &shape_of(name)?, &mut tensors);
        }
        let n = cfg.n_routed_experts;
        for (w, stack) in STACKS {
            let one = src
                .stored_shape(&format!("layers.{layer}.ffn.experts.0.{w}.weight"))
                .ok_or_else(|| Error::msg(format!("layer {layer}: no expert {w}")))?;
            let dims = vec![n, one[0], one[1]];
            let elems: usize = dims.iter().product();
            let dtype = if w == "w2" {
                formats.down
            } else {
                formats.gate_up
            };
            // Down sees its own rows, or the layer's pooled importance without rows.
            // Gate and up see their own rows, or the rows of the whole block's input.
            let records = if w == "w2" {
                StackRecords {
                    own: (0..n).map(|e| down_seen(layer, e)).collect(),
                    shared: down_pooled(layer),
                    shared_rows: false,
                }
            } else {
                StackRecords {
                    own: (0..n).map(|e| gate_up_seen(layer, e)).collect(),
                    shared: moe_input(layer),
                    shared_rows: true,
                }
            };
            tensors.push((
                PlannedEntry {
                    name: format!("blk.{layer}.{stack}.weight"),
                    dims,
                    dtype,
                    byte_len: bytes_per(dtype, elems),
                },
                Produce::Experts {
                    stored: (0..n)
                        .map(|e| format!("layers.{layer}.ffn.experts.{e}.{w}.weight"))
                        .collect(),
                    records,
                },
            ));
        }
    }

    if let (Some(e), false) = (cfg.engram.as_ref(), tables.is_empty()) {
        let mut rows = 0;
        for t in &tables {
            rows += shape_of(t)?[0];
        }
        tensors.push((
            PlannedEntry {
                name: ENGRAM_TABLE.into(),
                dims: vec![rows, e.head_dim],
                dtype: mainline::ENGRAM_TABLE_FORMAT,
                byte_len: mainline::engram_table_bytes(rows, e.head_dim)?,
            },
            Produce::Table {
                stored: tables,
                rows_per_piece: mainline::TABLE_ROWS_PER_PIECE,
            },
        ));
    }

    let metadata = file_metadata(src, cfg, n_layers, &engram_rows, model_name, tokenizer)?;
    Ok(ConvertPlan { metadata, tensors })
}
