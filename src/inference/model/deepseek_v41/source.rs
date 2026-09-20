//! Where the weights come from.
//!
//! The loaders ask for tensors by the reference implementation's own names (`embed.weight`,
//! `layers.3.attn.wq_a.weight`, `layers.3.ffn.experts.17.w1.weight`, ...) - the names the released
//! checkpoint carries. A source answers with a dense f32 tensor, an in-place projection the dot
//! engine reads where the bytes lie, a per-layer expert loader, or an engram table; how a format
//! stores and names them is the source's business. A GGUF is one such source: its converter-given
//! names are translated here, and its stacked expert tensors are viewed one expert at a time.

use super::engram::EngramTable;
use super::load::load_f32;
use crate::inference::offload::projection::Projection;
use crate::inference::offload::store::{ExpertLoader, QuantExpertLoader};
use crate::tensor::quantized::gguf_source::GgufSource;
use crate::tensor::quantized::GgmlDType;
use crate::tensor::mapped::MappedBytes;
use crate::tensor::{Error, Result, Tensor};
use std::sync::Arc;

pub trait WeightSource {
    /// The logical shape of `name`, `None` when the source has no such tensor.
    fn shape(&self, name: &str) -> Option<Vec<usize>>;
    /// `name` as dense f32, whatever it is stored as.
    fn dense_f32(&self, name: &str) -> Result<Tensor>;
    /// `name` read in place by the dot engine, `None` when its storage has no such path.
    fn projection(&self, name: &str) -> Result<Option<Projection>>;
    /// The routed experts of `layer`, `n` of them, read on demand.
    fn experts(&self, layer: usize, n: usize) -> Result<Box<dyn ExpertLoader>>;
    /// The engram table `name`, rows read on demand when the storage allows it.
    fn engram_table(&self, name: &str) -> Result<EngramTable>;
}

/// The first of `names` the source carries, dense f32.
pub fn dense_first<S: WeightSource + ?Sized>(g: &S, names: &[String]) -> Result<Tensor> {
    match names.iter().find(|n| g.shape(n).is_some()) {
        Some(n) => g.dense_f32(n),
        None => Err(Error::msg(format!("weights: none of {names:?} present"))),
    }
}

/// The GGUF spellings of a reference name below `layers.N.`: the bare converter name, then the
/// legacy name the synthetic gates write. A mainline conversion appends `.weight` or `.bias` to
/// the bare name; that spelling is tried too.
fn gguf_block_names(rest: &str) -> Option<(&'static str, &'static str, &'static str)> {
    // (bare name, suffix a mainline conversion appends, legacy name)
    Some(match rest {
        "attn_norm.weight" => ("attn_norm", ".weight", "attn_norm.weight"),
        "ffn_norm.weight" => ("ffn_norm", ".weight", "ffn_norm.weight"),
        "hc_attn_fn" => ("hc_attn_fn", ".weight", "hc_attn_fn.weight"),
        "hc_attn_scale" => ("hc_attn_scale", ".weight", "hc_attn_scale"),
        "hc_attn_base" => ("hc_attn_base", ".weight", "hc_attn_base"),
        "hc_ffn_fn" => ("hc_ffn_fn", ".weight", "hc_ffn_fn.weight"),
        "hc_ffn_scale" => ("hc_ffn_scale", ".weight", "hc_ffn_scale"),
        "hc_ffn_base" => ("hc_ffn_base", ".weight", "hc_ffn_base"),
        "attn.wq_a.weight" => ("attn_q_a", ".weight", "attn_q_a.weight"),
        "attn.q_norm.weight" => ("attn_q_a_norm", ".weight", "attn_q_a_norm.weight"),
        "attn.wq_b.weight" => ("attn_q_b", ".weight", "attn_q_b.weight"),
        "attn.wkv.weight" => ("attn_kv", ".weight", "attn_kv.weight"),
        "attn.kv_norm.weight" => ("attn_kv_a_norm", ".weight", "attn_kv_norm.weight"),
        "attn.attn_sink" => ("attn_sinks", ".weight", "attn_sink"),
        "attn.wo_a.weight" => ("attn_output_a", ".weight", "attn_o_a.weight"),
        "attn.wo_b.weight" => ("attn_output_b", ".weight", "attn_o_b.weight"),
        "attn.compressor.norm.weight" => (
            "attn_compressor_norm",
            ".weight",
            "attn_compressor_norm.weight",
        ),
        "attn.compressor.wkv.weight" => {
            ("attn_compressor_kv", ".weight", "attn_compressor_kv.weight")
        }
        "attn.compressor.wgate.weight" => (
            "attn_compressor_gate",
            ".weight",
            "attn_compressor_gate.weight",
        ),
        "attn.indexer.wq_b.weight" => ("indexer.attn_q_b", ".weight", "attn_indexer_q_b.weight"),
        "attn.indexer.weights_proj.weight" => {
            ("indexer.proj", ".weight", "attn_indexer_weights.weight")
        }
        "attn.indexer.wk.weight" => ("indexer.attn_k", ".weight", "attn_indexer_k.weight"),
        "attn.indexer.k_norm.weight" => ("indexer.k_norm", ".weight", "attn_indexer_k_norm.weight"),
        "ffn.gate.weight" => ("ffn_gate_inp", ".weight", "ffn_gate_inp.weight"),
        "ffn.gate.bias" => ("exp_probs_b", ".bias", "ffn_gate_inp.bias"),
        "ffn.shared_experts.w1.weight" => ("ffn_gate_shexp", ".weight", "ffn_gate_shexp.weight"),
        "ffn.shared_experts.w2.weight" => ("ffn_down_shexp", ".weight", "ffn_down_shexp.weight"),
        "ffn.shared_experts.w3.weight" => ("ffn_up_shexp", ".weight", "ffn_up_shexp.weight"),
        "engram.embed.weight" => ("engram_embd", ".weight", "engram_embd.weight"),
        "engram.wkv.weight" => ("engram_wkv", ".weight", "engram_wkv.weight"),
        "engram.q_weight" => ("engram_q", ".weight", "engram_q.weight"),
        "engram.k_weight" => ("engram_k", ".weight", "engram_k.weight"),
        _ => return None,
    })
}

/// The GGUF stack a routed expert's projection sits in.
fn gguf_expert_stack(w: &str) -> Option<&'static str> {
    Some(match w {
        "w1" => "ffn_gate_exps",
        "w2" => "ffn_down_exps",
        "w3" => "ffn_up_exps",
        _ => return None,
    })
}

/// `layers.{l}.{rest}` taken apart.
fn layer_of(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("layers.")?;
    let (l, rest) = rest.split_once('.')?;
    Some((l.parse().ok()?, rest))
}

/// `ffn.experts.{e}.{w}.weight` taken apart.
fn expert_of(rest: &str) -> Option<(usize, &str)> {
    let rest = rest.strip_prefix("ffn.experts.")?;
    let (e, rest) = rest.split_once('.')?;
    let w = rest.strip_suffix(".weight")?;
    Some((e.parse().ok()?, w))
}

/// An engram table as the released checkpoint holds it: a row of fp8 values beside a row of e8m0
/// scales, both as raw bytes. A row spans hundreds to one, so the exponent each value carries
/// keeps what a block-scaled format loses, and a row with its scales takes fewer bytes than q8_0
/// would.
/// `None` when the file does not carry the pair.
fn released_fp8_rows<G: GgufSource + ?Sized>(g: &G, weight: &str) -> Result<Option<EngramTable>> {
    let scale = format!("{}.scale", weight.strip_suffix(".weight").unwrap_or(weight));
    let (Some(w), Some(s)) = (g.info(weight), g.info(&scale)) else {
        return Ok(None);
    };
    if w.ggml_dtype != GgmlDType::I8 || s.ggml_dtype != GgmlDType::I8 {
        return Ok(None);
    }
    let (&[rows, row_len], &[scale_rows, per_row]) = (w.shape.dims(), s.shape.dims()) else {
        return Err(Error::msg(format!(
            "{weight}: an engram table is two-dimensional"
        )));
    };
    if scale_rows != rows || per_row == 0 || row_len % per_row != 0 {
        return Err(Error::msg(format!(
            "{weight}: scales {:?} do not tile [{rows}, {row_len}] by rows",
            s.shape.dims()
        )));
    }
    let (Some((wm, wo, wl)), Some((sm, so, sl))) =
        (g.mapped_range(weight)?, g.mapped_range(&scale)?)
    else {
        return Ok(None);
    };
    let table = EngramTable::Fp8Rows {
        bytes: MappedBytes::new(wm, wo, wl)?,
        scales: MappedBytes::new(sm, so, sl)?,
        rows,
        row_len,
        block: row_len / per_row,
    };
    table.advise_random();
    Ok(Some(table))
}

/// Every GGUF spelling of a reference name, most likely first.
pub fn gguf_names(name: &str) -> Vec<String> {
    match name {
        "embed.weight" => return vec!["token_embd".into(), "token_embd.weight".into()],
        "norm.weight" => return vec!["output_norm".into(), "output_norm.weight".into()],
        "head.weight" => return vec!["output".into(), "output.weight".into()],
        _ => {}
    }
    let Some((l, rest)) = layer_of(name) else {
        return vec![name.to_string()];
    };
    if let Some((_, w)) = expert_of(rest) {
        // The stack; the expert inside it is addressed separately.
        return match gguf_expert_stack(w) {
            Some(s) => vec![format!("blk.{l}.{s}"), format!("blk.{l}.{s}.weight")],
            None => vec![],
        };
    }
    match gguf_block_names(rest) {
        Some((bare, suffix, legacy)) => vec![
            format!("blk.{l}.{bare}"),
            format!("blk.{l}.{bare}{suffix}"),
            format!("blk.{l}.{legacy}"),
        ],
        None => vec![name.to_string()],
    }
}

impl<G: GgufSource> WeightSource for G {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        let present = gguf_names(name)
            .into_iter()
            .find(|n| self.info(n).is_some())?;
        let dims = self.info(&present)?.shape.dims().to_vec();
        match layer_of(name).and_then(|(_, rest)| expert_of(rest)) {
            // One expert of a stack: its own rows.
            Some(_) if dims.len() == 3 => Some(dims[1..].to_vec()),
            _ => Some(dims),
        }
    }

    fn dense_f32(&self, name: &str) -> Result<Tensor> {
        let present = gguf_names(name)
            .into_iter()
            .find(|n| self.info(n).is_some())
            .ok_or_else(|| Error::msg(format!("gguf: no tensor for {name}")))?;
        match layer_of(name).and_then(|(_, rest)| expert_of(rest)) {
            Some((e, _)) => load_f32(self, &present)?
                .narrow(0, e, 1)?
                .squeeze(0)?
                .contiguous(),
            None => load_f32(self, &present),
        }
    }

    fn projection(&self, name: &str) -> Result<Option<Projection>> {
        let Some(present) = gguf_names(name)
            .into_iter()
            .find(|n| self.info(n).is_some())
        else {
            return Ok(None);
        };
        let Some(view) = self.mmap_view(&present)? else {
            return Ok(None);
        };
        match layer_of(name).and_then(|(_, rest)| expert_of(rest)) {
            Some((e, _)) => Ok(Some(Projection::Quant(Arc::new(
                crate::tensor::quant_view::expert_view(&Arc::new(view), e)?,
            )))),
            None => Ok(Some(Projection::Quant(Arc::new(view)))),
        }
    }

    fn experts(&self, layer: usize, n: usize) -> Result<Box<dyn ExpertLoader>> {
        let stack = |w: &str| -> Result<(String, Option<crate::tensor::quantized::QTensor>)> {
            let canonical = format!("layers.{layer}.ffn.experts.0.{w}.weight");
            let present = gguf_names(&canonical)
                .into_iter()
                .find(|n| self.info(n).is_some())
                .ok_or_else(|| Error::msg(format!("gguf: no expert stack for {canonical}")))?;
            let view = self.mmap_view(&present)?;
            Ok((present, view))
        };
        let (gname, gate) = stack("w1")?;
        let (uname, up) = stack("w3")?;
        let (dname, down) = stack("w2")?;
        if let (Some(gate), Some(up), Some(down)) = (gate, up, down) {
            // The file and the address its mapping starts at, so a released expert's pages
            // can be dropped from the page cache by file range, not only from this mapping.
            let anchor = |name: &str| -> Option<(Arc<std::fs::File>, usize)> {
                let (map, _, _) = self.mapped_range(name).ok()??;
                Some((self.mapped_file(name)?, map.as_ptr() as usize))
            };
            let anchors = [anchor(&gname), anchor(&uname), anchor(&dname)];
            return Ok(Box::new(
                QuantExpertLoader::new(Arc::new(gate), Arc::new(up), Arc::new(down), n)
                    .anchored(anchors),
            ));
        }
        // No view path for this dtype: the stacks are dequantised once and sliced.
        Ok(Box::new(
            crate::inference::offload::store::DenseStackLoader::new(
                load_f32(self, &gname)?,
                load_f32(self, &uname)?,
                load_f32(self, &dname)?,
                n,
            ),
        ))
    }

    fn engram_table(&self, name: &str) -> Result<EngramTable> {
        let present = gguf_names(name)
            .into_iter()
            .find(|n| self.info(n).is_some())
            .ok_or_else(|| Error::msg(format!("gguf: no tensor for {name}")))?;
        if let Some(table) = released_fp8_rows(self, &present)? {
            return Ok(table);
        }
        Ok(match self.mmap_view(&present)? {
            Some(view) => EngramTable::Streamed(Arc::new(view)),
            None => EngramTable::Resident(load_f32(self, &present)?),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reference name the loaders use has GGUF spellings, bare first, and an expert's
    /// name maps to its stack.
    /// An engram table the file carries as the release does - fp8 rows beside their e8m0 scales,
    /// both as raw bytes - is read as those rows, with the geometry the two shapes imply.
    #[test]
    fn engram_rows_are_read_as_released_fp8() {
        use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
        use crate::tensor::quantized::gguf_file::{open_mapped, Value};

        let (rows, row_len, per_row) = (7usize, 64usize, 2usize);
        let weight: Vec<u8> = (0..rows * row_len).map(|i| (i % 251) as u8).collect();
        let scales: Vec<u8> = (0..rows * per_row).map(|i| (120 + i % 9) as u8).collect();
        let md = vec![(
            "general.architecture".to_string(),
            Value::String("deepseek_v41".into()),
        )];
        let entries = vec![
            GgufEntry {
                name: "blk.1.engram_embd.weight".to_string(),
                dims: vec![rows, row_len],
                dtype: GgmlDType::I8,
                data: weight.clone(),
            },
            GgufEntry {
                name: "blk.1.engram_embd.scale".to_string(),
                dims: vec![rows, per_row],
                dtype: GgmlDType::I8,
                data: scales.clone(),
            },
        ];
        let path =
            std::env::temp_dir().join(format!("loken-dsv41-engram-{}.gguf", std::process::id()));
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let g = open_mapped(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let table = g.engram_table("layers.1.engram.embed.weight").unwrap();
        match table {
            EngramTable::Fp8Rows {
                bytes,
                scales: sc,
                rows: r,
                row_len: rl,
                block,
            } => {
                assert_eq!((r, rl, block), (rows, row_len, row_len / per_row));
                assert_eq!(bytes.as_slice(), weight.as_slice());
                assert_eq!(sc.as_slice(), scales.as_slice());
            }
            _ => panic!("the pair of byte tensors was not read as released fp8 rows"),
        }
    }

    #[test]
    fn reference_names_translate_to_gguf_spellings() {
        assert_eq!(
            gguf_names("embed.weight"),
            vec!["token_embd", "token_embd.weight"]
        );
        assert_eq!(
            gguf_names("layers.3.attn.kv_norm.weight"),
            vec![
                "blk.3.attn_kv_a_norm",
                "blk.3.attn_kv_a_norm.weight",
                "blk.3.attn_kv_norm.weight"
            ]
        );
        assert_eq!(
            gguf_names("layers.3.ffn.gate.bias"),
            vec![
                "blk.3.exp_probs_b",
                "blk.3.exp_probs_b.bias",
                "blk.3.ffn_gate_inp.bias"
            ]
        );
        assert_eq!(
            gguf_names("layers.3.ffn.experts.17.w2.weight"),
            vec!["blk.3.ffn_down_exps", "blk.3.ffn_down_exps.weight"]
        );
        assert_eq!(gguf_names("layers.3.attn.attn_sink")[2], "blk.3.attn_sink");
        assert!(gguf_names("layers.3.ffn.experts.17.w9.weight").is_empty());
    }
}
