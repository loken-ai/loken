//! llama.cpp's layout of this architecture, `deepseek41`: the tag, the keys its loaders read, the
//! names and shapes it reads the tensors under, and the engram tables as one table. A file in
//! this layout loads in llama.cpp, in the engines that vendor it, and here.
//!
//! [`metadata`] writes the keys from a configuration; [`tensor_names`] lists the tensors the
//! loader creates for it, no more and no fewer, since the loader refuses a file that carries a
//! tensor it does not create. [`relayout`] rewrites a file in loken's own layout into this one,
//! carrying every tensor's bytes.

use super::engram::qk_weight;
use super::source::{
    expert_unit_name, gguf_names, llamacpp_name, WeightSource, BLOCK_TENSORS, ENGRAM_TABLE,
    EXPERT_STACKS,
};
use crate::inference::generic_transformer::wide_sparse_config::WideSparseMeta;
pub use crate::inference::generic_transformer::wide_sparse_llamacpp::*;
use crate::inference::load::model_dirs::ensure_room;
use crate::tensor::gguf_write::{planned_len, GgufStreamWriter, Placement, PlannedEntry};
use crate::tensor::quantized::gguf_file::{MappedGguf, Value};
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use std::path::Path;

/// A tensor's bytes where the mapping holds them.
fn bytes_of<'a>(src: &'a MappedGguf, name: &str) -> Result<&'a [u8]> {
    let i = src
        .content
        .tensor_infos
        .get(name)
        .ok_or_else(|| Error::msg(format!("no tensor {name}")))?;
    let start = (src.content.tensor_data_offset + i.offset) as usize;
    Ok(&src.mmap[start..start + i.size_in_bytes()])
}

/// Where one tensor of a re-laid file comes from.
enum Part {
    /// A source tensor's bytes.
    Carried(String),
    /// Experts stored one tensor each, stacked in expert order.
    Stacked(Vec<String>),
    /// An engram layer's query and key weights multiplied, f32.
    Gate(usize),
    /// Every engram layer's table in [`ENGRAM_TABLE_FORMAT`], one after the other.
    Table(Vec<usize>),
}

/// Write `out` in llama.cpp's layout from `src`, a file in loken's own: every tensor llama.cpp
/// reads, its bytes carried; experts stored one tensor each stacked again; each engram layer's
/// query and key weights multiplied into its gate; the engram tables laid out as one table.
/// The draft, and any tensor llama.cpp has no place for, is left out. A disk that cannot hold the
/// file is refused before its first byte. Returns the SHA-256 of the file written.
pub fn relayout(
    src: &MappedGguf,
    out: &Path,
    progress: &dyn Fn(usize, usize, &str),
) -> Result<String> {
    let cfg = WideSparseMeta::from_file(&src.content)?;
    let info = |n: &str| src.content.tensor_infos.get(n);
    let present = |reference: &str| {
        gguf_names(reference)
            .into_iter()
            .find(|n| info(n).is_some())
    };
    let mut plan: Vec<(PlannedEntry, Part)> = Vec::new();
    let carry = |plan: &mut Vec<(PlannedEntry, Part)>, reference: &str| {
        let (Some(from), Some(to)) = (present(reference), llamacpp_name(reference)) else {
            return;
        };
        let i = &src.content.tensor_infos[&from];
        plan.push((
            PlannedEntry {
                name: to,
                dims: i.shape.dims().to_vec(),
                dtype: i.ggml_dtype,
                byte_len: i.size_in_bytes() as u64,
            },
            Part::Carried(from),
        ));
    };
    for reference in ["embed.weight", "norm.weight", "head.weight"] {
        carry(&mut plan, reference);
    }
    let n_layers = cfg.n_layers;
    let kept = engram_layers(&cfg, n_layers);
    for l in 0..n_layers {
        for rest in BLOCK_TENSORS {
            carry(&mut plan, &format!("layers.{l}.{rest}"));
        }
        let engram_layer = cfg.ngram.as_ref().is_some_and(|e| e.layer_ids.contains(&l));
        if engram_layer && present(&format!("layers.{l}.engram.gate.weight")).is_none() {
            let q = present(&format!("layers.{l}.engram.q_weight")).ok_or_else(|| {
                Error::msg(format!("layer {l}: an engram layer without its weights"))
            })?;
            let dims = src.content.tensor_infos[&q].shape.dims().to_vec();
            let elems: usize = dims.iter().product();
            plan.push((
                PlannedEntry {
                    name: format!("blk.{l}.engram_gate.weight"),
                    dims,
                    dtype: GgmlDType::F32,
                    byte_len: (elems * std::mem::size_of::<f32>()) as u64,
                },
                Part::Gate(l),
            ));
        }
        for (stack, w) in EXPERT_STACKS {
            let reference = format!("layers.{l}.ffn.experts.0.{w}.weight");
            let to = llamacpp_name(&reference)
                .ok_or_else(|| Error::msg(format!("no llama.cpp name for {reference}")))?;
            match present(&reference) {
                Some(from) if src.content.tensor_infos[&from].shape.dims().len() == 3 => {
                    let i = &src.content.tensor_infos[&from];
                    plan.push((
                        PlannedEntry {
                            name: to,
                            dims: i.shape.dims().to_vec(),
                            dtype: i.ggml_dtype,
                            byte_len: i.size_in_bytes() as u64,
                        },
                        Part::Carried(from),
                    ));
                }
                _ => {
                    let units: Vec<String> = (0..cfg.mixture.routed)
                        .map(|e| expert_unit_name(l, stack, e))
                        .collect();
                    let first = info(&units[0])
                        .ok_or_else(|| Error::msg(format!("layer {l}: no {stack}")))?;
                    let mut dims = vec![cfg.mixture.routed];
                    dims.extend(first.shape.dims());
                    let (dtype, one) = (first.ggml_dtype, first.size_in_bytes() as u64);
                    for u in &units {
                        let i = info(u).ok_or_else(|| Error::msg(format!("no tensor {u}")))?;
                        if i.ggml_dtype != dtype || i.size_in_bytes() as u64 != one {
                            return Err(Error::msg(format!("{u}: not stored as {}", units[0])));
                        }
                    }
                    plan.push((
                        PlannedEntry {
                            name: to,
                            dims,
                            dtype,
                            byte_len: one * cfg.mixture.routed as u64,
                        },
                        Part::Stacked(units),
                    ));
                }
            }
        }
    }
    let mut engram_rows = Vec::with_capacity(kept.len());
    if let Some(e) = cfg.ngram.as_ref().filter(|_| !kept.is_empty()) {
        for &i in &kept {
            let table =
                src.engram_table(&format!("layers.{}.engram.embed.weight", e.layer_ids[i]))?;
            engram_rows.push(table.n_rows());
        }
        let rows: usize = engram_rows.iter().sum();
        plan.push((
            PlannedEntry {
                name: ENGRAM_TABLE.into(),
                dims: vec![rows, e.head_dim],
                dtype: ENGRAM_TABLE_FORMAT,
                byte_len: engram_table_bytes(rows, e.head_dim)?,
            },
            Part::Table(kept.iter().map(|&i| e.layer_ids[i]).collect()),
        ));
    }

    let md = &src.content.metadata;
    let string = |k: &str| md.get(k).and_then(|v| v.to_string().ok().cloned());
    let arch = string("general.architecture").unwrap_or_default();
    let mut tokenizer: Vec<(String, Value)> = md
        .iter()
        .filter(|(k, _)| k.starts_with("tokenizer."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    tokenizer.sort_by(|a, b| a.0.cmp(&b.0));
    if !md.contains_key("tokenizer.ggml.tokens") {
        let json = string(super::convert::TOKENIZER_KEY).ok_or_else(|| {
            Error::msg("relayout: the file carries no tokenizer llama.cpp can read")
        })?;
        tokenizer.extend(vocab_metadata(&json, cfg.vocab)?);
    }
    let extras = Extras {
        name: string("general.name").unwrap_or_else(|| ARCH.to_string()),
        context_length: md
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or_else(|| trained_context(&cfg)),
        tokenizer,
    };
    let metadata = metadata(&cfg, n_layers, &engram_rows, &extras)?;

    let entries: Vec<PlannedEntry> = plan
        .iter()
        .map(|(e, _)| PlannedEntry {
            name: e.name.clone(),
            dims: e.dims.clone(),
            dtype: e.dtype,
            byte_len: e.byte_len,
        })
        .collect();
    let need = planned_len(out, &metadata, &entries, &Placement::packed())?;
    let room = out.parent().unwrap_or(Path::new("."));
    ensure_room("the re-laid file", room, need).map_err(Error::msg)?;
    let mut writer = GgufStreamWriter::create(out, &metadata, &entries)?;
    let total = plan.len();
    for (n, (entry, part)) in plan.iter().enumerate() {
        match part {
            Part::Carried(name) => writer.append(bytes_of(src, name)?)?,
            Part::Stacked(units) => {
                writer.append_parts(units.iter().map(|u| bytes_of(src, u).map(<[u8]>::to_vec)))?
            }
            Part::Gate(l) => {
                let gate = qk_weight(src, &format!("layers.{l}.engram"))?;
                writer.append(
                    &gate
                        .iter()
                        .flatten()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?
            }
            Part::Table(layers) => {
                let mut tables = Vec::with_capacity(layers.len());
                for l in layers {
                    let table = src.engram_table(&format!("layers.{l}.engram.embed.weight"))?;
                    table.advise_sequential();
                    tables.push(table);
                }
                writer.append_parts(tables.iter().flat_map(|t| {
                    let rows = t.n_rows();
                    (0..rows).step_by(TABLE_ROWS_PER_PIECE).map(move |start| {
                        t.encode_rows(
                            start..(start + TABLE_ROWS_PER_PIECE).min(rows),
                            ENGRAM_TABLE_FORMAT,
                        )
                    })
                }))?
            }
        }
        progress(n + 1, total, &entry.name);
    }
    writer.finish_with_digest()
}
