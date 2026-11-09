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

use super::ngram::engram::EngramMeta;
use super::ngram::token_map;
use super::wide_sparse_config::{NgramHashFields, WideSparseMeta};
use super::wide_sparse_llamacpp as mainline;
use super::wide_sparse_names::{llamacpp_name, ENGRAM_TABLE};
use super::BlockScaledCheckpoint;
use crate::inference::engine::llm_engine::EMBEDDED_TOKENIZER_KEY;
use crate::inference::load::block_scaled_convert::{
    bytes_per, written_format, ConvertPlan, Formats, Produce, StackRecords,
};
use crate::tensor::gguf_write::PlannedEntry;
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use std::ops::Range;
use std::path::Path;

/// The record of what layer `layer`'s MoE block took in.
///
/// The record names are those a calibration of the checkpoint writes.
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
pub fn checkpoint_config(src: &BlockScaledCheckpoint, dir: &Path) -> Result<WideSparseMeta> {
    let (map, _) = token_map::build_from_file(&dir.join("tokenizer.json"))?;
    let engram = EngramMeta::derive(&src.reference, map);
    WideSparseMeta::from_reference(&src.reference, engram.as_ref().map(NgramHashFields::from))
}

/// The checkpoint's chat format, as its `encoding/encoding.py` renders a conversation.
///
/// The sequence token comes first, then a system turn when one is given.
/// Every turn follows its role token.
/// The assistant header is closed by `</think>` for a plain answer.
/// It is opened by `<think>` when `enable_thinking` is set.
/// The encoder's default reasoning effort then leads the system turn.
/// Tool turns are outside this template: the server flattens them.
pub const CHAT_TEMPLATE: &str = concat!(
    "{{ '<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>' }}",
    "{% set thinking = enable_thinking is defined and enable_thinking %}",
    "{% if thinking %}",
    "<\u{ff5c}System\u{ff5c}>Reasoning Effort: 75 (range 1-100, the higher the value, ",
    "the more thorough the reasoning)\n\n",
    "{% endif %}",
    "{% for m in messages %}",
    "{% if m.role == 'system' %}",
    "{% if not (thinking and loop.first) %}<\u{ff5c}System\u{ff5c}>{% endif %}",
    "{% if m.content %}{{ m.content }}{% endif %}",
    "{% elif m.role == 'user' %}",
    "<\u{ff5c}User\u{ff5c}>{% if m.content %}{{ m.content }}{% endif %}",
    "{% elif m.role == 'assistant' %}",
    "{% if m.content %}{{ m.content }}{% endif %}<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "{% endif %}",
    "{% if m.role != 'assistant' and ((loop.last and add_generation_prompt) ",
    "or (not loop.last and loop.nextitem.role == 'assistant')) %}",
    "<\u{ff5c}Assistant\u{ff5c}>",
    "{% if thinking and loop.last %}<think>{% else %}</think>{% endif %}",
    "{% endif %}",
    "{% endfor %}",
);

/// What the checkpoint says about its tokens: `tokenizer.json` whole, and the ids its
/// `tokenizer_config.json` names for the sequence and end tokens.
pub struct TokenizerFiles {
    pub json: String,
    pub bos: u32,
    pub eos: u32,
}

impl TokenizerFiles {
    pub fn read(dir: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(dir.join("tokenizer.json"))
            .map_err(|e| Error::msg(format!("tokenizer.json: {e}")))?;
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("tokenizer_config.json"))
                .map_err(|e| Error::msg(format!("tokenizer_config.json: {e}")))?,
        )
        .map_err(|e| Error::msg(format!("tokenizer_config.json: {e}")))?;
        let tok: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| Error::msg(format!("tokenizer.json: {e}")))?;
        let id = |key: &str| -> Result<u32> {
            let name = match cfg.get(key) {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v
                    .get("content")
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| {
                        Error::msg(format!("tokenizer_config.json: {key} has no content"))
                    })?
                    .to_string(),
                None => return Err(Error::msg(format!("tokenizer_config.json: no {key}"))),
            };
            tok.get("added_tokens")
                .and_then(|a| a.as_array())
                .and_then(|a| {
                    a.iter()
                        .find(|t| t.get("content").and_then(|c| c.as_str()) == Some(&name))
                })
                .and_then(|t| t.get("id"))
                .and_then(|i| i.as_u64())
                .map(|i| i as u32)
                .ok_or_else(|| Error::msg(format!("tokenizer.json: {name} is not an added token")))
        };
        Ok(Self {
            bos: id("bos_token")?,
            eos: id("eos_token")?,
            json,
        })
    }
}

/// The tokenizer's metadata: the file whole, the 2 token ids the loader stops and starts on,
/// and the chat format.
///
/// The sequence token is written by the template, not by the tokenizer, as the checkpoint's
/// configuration says.
pub fn tokenizer_metadata(files: &TokenizerFiles) -> Vec<(String, Value)> {
    vec![
        (
            EMBEDDED_TOKENIZER_KEY.to_string(),
            Value::String(files.json.clone()),
        ),
        (
            "tokenizer.ggml.bos_token_id".to_string(),
            Value::U32(files.bos),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            Value::U32(files.eos),
        ),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Value::Bool(false),
        ),
        (
            "tokenizer.chat_template".to_string(),
            Value::String(CHAT_TEMPLATE.to_string()),
        ),
    ]
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
    cfg: &WideSparseMeta,
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
        keys.extend(mainline::vocab_metadata(&files.json, cfg.vocab)?);
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
    cfg: &WideSparseMeta,
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
        .filter_map(|i| cfg.ngram.as_ref().map(|e| e.layer_ids[i]))
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
        let n = cfg.mixture.routed;
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

    if let (Some(e), false) = (cfg.ngram.as_ref(), tables.is_empty()) {
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
