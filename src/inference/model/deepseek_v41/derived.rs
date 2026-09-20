//! Routed experts requantised to a block format, derived from the checkpoint and regenerable.
//!
//! On a host whose disk is the wall, the bytes an expert occupies are the cost of routing to
//! it. The released experts weigh 4.25 bits per weight; the same experts requantised by the
//! block formats the dot engine reads in place weigh 2.6 (Q2_K) or 3.4 (Q3_K), so a token
//! reads that much less and the page cache holds that many more of them. A `LayeredSource`
//! answers the always-read path from the checkpoint and the routed experts from the derived
//! stacks where a layer has them, so the derived set can cover any subset of layers. What the
//! requantisation costs in quality is measured against the reference, never assumed.

use super::engram::EngramTable;
use super::source::WeightSource;
use super::DeepseekV41Config;
use crate::inference::offload::projection::Projection;
use crate::inference::offload::store::{ExpertLoader, QuantExpertLoader};
use crate::tensor::gguf_write::{GgufStreamWriter, PlannedEntry};
use crate::tensor::quant_cpu::from_float_bytes;
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::gguf_source::SplitGguf;
use crate::tensor::quantized::{GgmlDType, QTensor};
use crate::tensor::{Device, Error, Result, Tensor};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One layer's routed experts as three `[n, rows, cols]` stacks in `dtype`.
pub struct DerivedLayer {
    pub gate: Arc<QTensor>,
    pub up: Arc<QTensor>,
    pub down: Arc<QTensor>,
}

/// Requantise layer `layer`'s `n` routed experts from `base` into `dtype` stacks, expert by
/// expert in parallel.
pub fn requantize_layer<B: WeightSource + ?Sized + Sync>(
    base: &B,
    layer: usize,
    n: usize,
    dtype: GgmlDType,
) -> Result<DerivedLayer> {
    let stack = |w: &str| -> Result<Arc<QTensor>> {
        let name = |e: usize| format!("layers.{layer}.ffn.experts.{e}.{w}.weight");
        let shape = base
            .shape(&name(0))
            .ok_or_else(|| Error::msg(format!("derived: no {}", name(0))))?;
        let [rows, cols] = shape[..] else {
            return Err(Error::msg(format!("derived: {} is not 2-D", name(0))));
        };
        let parts: Vec<Vec<u8>> = (0..n)
            .into_par_iter()
            .map(|e| -> Result<Vec<u8>> {
                let w = base.dense_f32(&name(e))?.flatten_all()?.to_vec1::<f32>()?;
                from_float_bytes(dtype, &w).map_err(|err| Error::msg(err.0))
            })
            .collect::<Result<_>>()?;
        let bytes: Vec<u8> = parts.concat();
        Ok(Arc::new(QTensor::from_ggml_bytes(
            dtype,
            &bytes,
            vec![n, rows, cols],
            &Device::Cpu,
        )?))
    };
    Ok(DerivedLayer {
        gate: stack("w1")?,
        up: stack("w3")?,
        down: stack("w2")?,
    })
}

/// Where the derived experts are: stacks built in memory, or a sidecar file of them.
pub enum DerivedExperts {
    Memory(HashMap<usize, DerivedLayer>),
    File(SplitGguf),
}

/// The checkpoint for everything, the derived stacks for the routed experts of the layers that
/// have them.
pub struct LayeredSource<'a, B: WeightSource + ?Sized> {
    pub base: &'a B,
    pub derived: DerivedExperts,
}

/// The tensor names a derived layer's stacks carry in a sidecar: the GGUF spellings of the
/// reference names, so the GGUF source serves them as it serves any converted file.
fn stack_names(layer: usize) -> [String; 3] {
    [
        format!("blk.{layer}.ffn_gate_exps"),
        format!("blk.{layer}.ffn_up_exps"),
        format!("blk.{layer}.ffn_down_exps"),
    ]
}

/// The sidecar the derived experts of a checkpoint directory live in, by format.
pub fn sidecar_path(dir: &Path, dtype: GgmlDType) -> PathBuf {
    let tag = format!("{dtype:?}").to_ascii_lowercase();
    dir.join(format!("experts.{tag}.gguf"))
}

/// A sidecar serves only when it is no older than every shard it was derived from.
pub fn sidecar_fresh(sidecar: &Path, dir: &Path) -> bool {
    let Ok(side) = std::fs::metadata(sidecar).and_then(|m| m.modified()) else {
        return false;
    };
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
                .all(|p| {
                    std::fs::metadata(p)
                        .and_then(|m| m.modified())
                        .is_ok_and(|m| m <= side)
                })
        })
        .unwrap_or(false)
}

/// The freshest derived expert sidecar of `dir`, whichever format it was made in.
pub fn find_sidecar(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("experts.") && n.ends_with(".gguf"))
        })
        .filter(|p| sidecar_fresh(p, dir))
        .collect();
    found.sort();
    found.pop()
}

/// Requantise every layer's routed experts into `dtype` and write them as a sidecar GGUF of
/// `[n, rows, cols]` stacks, one layer at a time so no more than one layer's worth is held.
/// `progress` is told each layer written.
pub fn derive_experts<B: WeightSource + ?Sized + Sync>(
    base: &B,
    cfg: &DeepseekV41Config,
    dtype: GgmlDType,
    path: &Path,
    mut progress: impl FnMut(usize),
) -> Result<()> {
    let n = cfg.n_routed_experts;
    let planned_bytes = |name: &str| -> Result<(Vec<usize>, u64)> {
        let shape = base
            .shape(name)
            .ok_or_else(|| Error::msg(format!("derived: no {name}")))?;
        let [rows, cols] = shape[..] else {
            return Err(Error::msg(format!("derived: {name} is not 2-D")));
        };
        let bytes = (n * rows * cols / dtype.block_size() * dtype.type_size()) as u64;
        Ok((vec![n, rows, cols], bytes))
    };
    let mut entries = Vec::with_capacity(3 * cfg.n_layers);
    for l in 0..cfg.n_layers {
        for (name, w) in stack_names(l).into_iter().zip(["w1", "w3", "w2"]) {
            let (dims, byte_len) = planned_bytes(&format!("layers.{l}.ffn.experts.0.{w}.weight"))?;
            entries.push(PlannedEntry {
                name,
                dims,
                dtype,
                byte_len,
            });
        }
    }
    let metadata = vec![
        (
            "general.architecture".to_string(),
            Value::String(super::ARCH.into()),
        ),
        (
            "general.type".to_string(),
            Value::String("derived-experts".into()),
        ),
        (
            format!("{}.block_count", super::ARCH),
            Value::U32(cfg.n_layers as u32),
        ),
        (
            format!("{}.expert_count", super::ARCH),
            Value::U32(n as u32),
        ),
    ];
    let mut w = GgufStreamWriter::create(path, &metadata, &entries).map_err(|e| Error::msg(e.0))?;
    for l in 0..cfg.n_layers {
        let d = requantize_layer(base, l, n, dtype)?;
        for t in [&d.gate, &d.up, &d.down] {
            w.append(&t.data()?).map_err(|e| Error::msg(e.0))?;
        }
        progress(l);
    }
    w.finish().map_err(|e| Error::msg(e.0))
}

impl<B: WeightSource + ?Sized> WeightSource for LayeredSource<'_, B> {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.base.shape(name)
    }

    fn dense_f32(&self, name: &str) -> Result<Tensor> {
        self.base.dense_f32(name)
    }

    fn projection(&self, name: &str) -> Result<Option<Projection>> {
        self.base.projection(name)
    }

    fn experts(&self, layer: usize, n: usize) -> Result<Box<dyn ExpertLoader>> {
        match &self.derived {
            DerivedExperts::Memory(layers) => match layers.get(&layer) {
                Some(d) => Ok(Box::new(QuantExpertLoader::new(
                    d.gate.clone(),
                    d.up.clone(),
                    d.down.clone(),
                    n,
                ))),
                None => self.base.experts(layer, n),
            },
            DerivedExperts::File(g) => {
                // Present under whichever spelling the file uses.
                let has = |w: &str| {
                    g.shape(&format!("layers.{layer}.ffn.experts.0.{w}.weight"))
                        .is_some()
                };
                if has("w1") && has("w2") && has("w3") {
                    g.experts(layer, n)
                } else {
                    self.base.experts(layer, n)
                }
            }
        }
    }

    fn engram_table(&self, name: &str) -> Result<EngramTable> {
        self.base.engram_table(name)
    }
}
