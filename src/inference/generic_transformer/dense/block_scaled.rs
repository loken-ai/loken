//! A block-scaled safetensors checkpoint presented under the loader's tensor names.
//!
//! The checkpoint is a directory of `*.safetensors` shards beside `config.json` and
//! `inference/config.json`.
//! Every shard is mapped and its header listed.
//! A stored tensor is listed under every loader name of its reference name.
//! A routed expert's matrix is listed as the unit `blk.{l}.{stack}.{e}` of its stack.
//!
//! An fp8 e4m3 matrix `<stem>.weight` keeps 1 ue8m0 scale per square tile in `<stem>.scale`.
//! The tile side is `quantization_config.weight_block_size` of `config.json`.
//! An fp4 e2m1 matrix is stored `I8`, 2 values a byte, the low nibble first.
//! It keeps 1 ue8m0 scale per run of inputs of a row in `<stem>.scale`.
//!
//! A projection is a view of the mapping: fp8, fp4 and bf16 matrices are never decoded.
//! A dense read decodes the stored values to F32.

use super::{fp8_block_of, read_header};
use crate::inference::generic_transformer::wide_sparse_names::{
    expert_unit_name, gguf_names, EXPERT_STACKS,
};
use crate::tensor::blockscaled::bf16::Bf16Weight;
use crate::tensor::blockscaled::dequant::{
    dequant_fp4_blocked, dequant_fp8_blocked, e4m3_to_f32, e8m0_to_f32,
};
use crate::tensor::blockscaled::fp4::Fp4Weight;
use crate::tensor::blockscaled::fp8::Fp8Weight;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::mapped::MappedBytes;
use crate::tensor::{Device, Error, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// The tensor names a shard's header lists, `None` when the header cannot be read.
/// Only the header is read.
pub fn header_names(shard: &Path) -> Option<Vec<String>> {
    use std::io::Read;
    let mut f = std::fs::File::open(shard).ok()?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len).ok()?;
    let len = usize::try_from(u64::from_le_bytes(len)).ok()?;
    let limit = usize::try_from(f.metadata().ok()?.len()).ok()?;
    if len > limit {
        return None;
    }
    let mut header = vec![0u8; len];
    f.read_exact(&mut header).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&header).ok()?;
    Some(
        v.as_object()?
            .keys()
            .filter(|k| *k != "__metadata__")
            .cloned()
            .collect(),
    )
}

/// One mapped shard and the file it maps.
struct Shard {
    map: Arc<memmap2::Mmap>,
    file: Arc<std::fs::File>,
}

/// One stored tensor: its shard, stored type, stored dimensions and byte range in the mapping.
struct Stored {
    shard: usize,
    dtype: String,
    shape: Vec<usize>,
    start: usize,
    len: usize,
}

/// The shards of a block-scaled checkpoint and the names the loader reads them under.
pub struct BlockScaledCheckpoint {
    shards: Vec<Shard>,
    /// Every stored tensor by its stored name.
    stored: HashMap<String, Stored>,
    /// The stored name each loader name reads.
    names: HashMap<String, String>,
    /// Side of the square tiles of the fp8 matrices, `None` when the configuration gives none.
    fp8_block: Option<usize>,
    /// The reference arguments of the model, `inference/config.json`.
    pub reference: serde_json::Value,
    /// The model's own configuration, `config.json`.
    pub config: serde_json::Value,
    /// The directory the checkpoint was opened from.
    pub dir: std::path::PathBuf,
}

/// The rows of a stored table, as a conversion copies them.
pub enum TableRows {
    /// e4m3 values, 1 ue8m0 scale per `block` values of a row.
    Fp8 {
        bytes: MappedBytes,
        scales: MappedBytes,
        rows: usize,
        row_len: usize,
        block: usize,
    },
    /// Values decoded to F32, row-major.
    Dense {
        values: Vec<f32>,
        rows: usize,
        row_len: usize,
    },
}

impl TableRows {
    pub fn n_rows(&self) -> usize {
        match self {
            Self::Fp8 { rows, .. } | Self::Dense { rows, .. } => *rows,
        }
    }

    /// Rows `rows` decoded to F32, in row order.
    ///
    /// An e4m3 value is multiplied by the scale of its run of the row.
    pub fn values(&self, rows: std::ops::Range<usize>) -> Vec<f32> {
        use rayon::prelude::*;
        match self {
            Self::Fp8 {
                bytes,
                scales,
                row_len,
                block,
                ..
            } => {
                let per_row = row_len / block;
                rows.into_par_iter()
                    .map(|r| {
                        let row = &bytes.as_slice()[r * row_len..(r + 1) * row_len];
                        let sc = &scales.as_slice()[r * per_row..(r + 1) * per_row];
                        row.iter()
                            .enumerate()
                            .map(|(j, &b)| e4m3_to_f32(b) * e8m0_to_f32(sc[j / block]))
                            .collect::<Vec<f32>>()
                    })
                    .collect::<Vec<_>>()
                    .concat()
            }
            Self::Dense {
                values, row_len, ..
            } => values[rows.start * row_len..rows.end * row_len].to_vec(),
        }
    }

    /// Asks the kernel to read the stored rows in order.
    pub fn advise_sequential(&self) {
        if let Self::Fp8 { bytes, scales, .. } = self {
            bytes.advise_sequential();
            scales.advise_sequential();
        }
    }
}

/// Bytes 1 stored value takes, `None` for a type no read decodes.
fn width(dtype: &str) -> Option<usize> {
    Some(match dtype {
        "F32" => 4,
        "F16" | "BF16" => 2,
        "F8_E4M3" | "F8_E8M0" | "I8" | "U8" => 1,
        _ => return None,
    })
}

/// The unit name of the routed expert matrix `layers.{l}.ffn.experts.{e}.{w}.weight`.
fn expert_unit(name: &str) -> Option<String> {
    let rest = name.strip_prefix("layers.")?;
    let (layer, rest) = rest.split_once('.')?;
    let rest = rest.strip_prefix("ffn.experts.")?;
    let (expert, rest) = rest.split_once('.')?;
    let held = rest.strip_suffix(".weight")?;
    let (stack, _) = EXPERT_STACKS.iter().find(|(_, w)| *w == held)?;
    Some(expert_unit_name(
        layer.parse().ok()?,
        stack,
        expert.parse().ok()?,
    ))
}

/// Each loader name of the stored tensors, with the stored name it reads.
///
/// A routed expert's matrix takes its unit name alone.
/// Any other tensor takes every spelling of its reference name.
/// A spelling 2 stored tensors share reads 1 of them.
fn loader_names<'a>(stored: impl Iterator<Item = &'a String>) -> HashMap<String, String> {
    let mut names = HashMap::new();
    for name in stored {
        match expert_unit(name) {
            Some(unit) => {
                names.insert(unit, name.clone());
            }
            None => {
                for spelling in gguf_names(name) {
                    names.entry(spelling).or_insert_with(|| name.clone());
                }
            }
        }
    }
    names
}

impl BlockScaledCheckpoint {
    /// Maps every `*.safetensors` under `dir`, in name order, and reads both configurations.
    pub fn open(dir: &Path) -> Result<Self> {
        let read_json = |p: &Path| -> Result<serde_json::Value> {
            let text = std::fs::read_to_string(p)
                .map_err(|e| Error::msg(format!("{}: {e}", p.display())))?;
            serde_json::from_str(&text).map_err(|e| Error::msg(format!("{}: {e}", p.display())))
        };
        let config = read_json(&dir.join("config.json"))?;
        let reference = read_json(&dir.join("inference").join("config.json"))?;
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| Error::msg(format!("{}: {e}", dir.display())))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(Error::msg(format!(
                "{}: no safetensors shard",
                dir.display()
            )));
        }
        let mut shards = Vec::with_capacity(paths.len());
        let mut stored = HashMap::new();
        for (shard, path) in paths.iter().enumerate() {
            let shown = path.display().to_string();
            let file =
                std::fs::File::open(path).map_err(|e| Error::msg(format!("{shown}: {e}")))?;
            // Safety: the checkpoint is not written while served, the contract of every mapped loader.
            let map = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| Error::msg(format!("{shown}: {e}")))?;
            let (listing, data_start) = read_header(&map, &shown)?;
            for (name, dtype, shape, start) in listing {
                let len = shape.iter().product::<usize>() * width(&dtype).unwrap_or(0);
                stored.insert(
                    name,
                    Stored {
                        shard,
                        dtype,
                        shape,
                        start: (data_start + start) as usize,
                        len,
                    },
                );
            }
            shards.push(Shard {
                map: Arc::new(map),
                file: Arc::new(file),
            });
        }
        let names = loader_names(stored.keys());
        Ok(Self {
            shards,
            stored,
            names,
            fp8_block: fp8_block_of(&config),
            reference,
            config,
            dir: dir.to_path_buf(),
        })
    }

    /// Every stored name, scales included.
    pub fn stored_names(&self) -> Vec<&str> {
        self.stored.keys().map(String::as_str).collect()
    }

    /// The shape of the stored tensor `stored` in values.
    ///
    /// An fp4 tensor counts 2 values a stored byte: its last dimension is twice the stored one.
    pub fn stored_shape(&self, stored: &str) -> Option<Vec<usize>> {
        let e = self.stored.get(stored)?;
        let mut shape = e.shape.clone();
        if e.dtype == "I8" {
            if let Some(last) = shape.last_mut() {
                *last *= 2;
            }
        }
        Some(shape)
    }

    /// The values of the stored tensor `stored`, decoded to F32, row-major.
    pub fn stored_values(&self, stored: &str) -> Result<Vec<f32>> {
        self.decode(stored).map(|(values, _)| values)
    }

    /// Asks the kernel to read the stored tensor `stored`, and its scales, ahead of their use.
    pub fn will_need_stored(&self, stored: &str) {
        if let Ok((_, view)) = self.bytes(stored) {
            view.will_need();
        }
        if let Ok((_, view)) = self.scales(stored) {
            view.will_need();
        }
    }

    /// The stored 2-D table `stored`, read a run of rows at a time.
    ///
    /// An e4m3 table keeps its rows in the mapping, with 1 run of ue8m0 scales per row.
    /// Any other table is decoded to F32 whole.
    pub fn table(&self, stored: &str) -> Result<TableRows> {
        let (e, view) = self.bytes(stored)?;
        let shape = self.stored_shape(stored).unwrap_or_default();
        let [rows, row_len] = shape[..] else {
            return Err(Error::msg(format!("{stored}: table is not 2-D")));
        };
        if e.dtype != "F8_E4M3" {
            return Ok(TableRows::Dense {
                values: self.stored_values(stored)?,
                rows,
                row_len,
            });
        }
        let (scales, scale_view) = self.scales(stored)?;
        let per_row = scales.shape.get(1).copied().unwrap_or(0);
        if scales.shape.first().copied() != Some(rows) || per_row == 0 || row_len % per_row != 0 {
            return Err(Error::msg(format!(
                "{stored}: scales {:?} do not tile [{rows}, {row_len}] by rows",
                scales.shape
            )));
        }
        Ok(TableRows::Fp8 {
            bytes: view,
            scales: scale_view,
            rows,
            row_len,
            block: row_len / per_row,
        })
    }

    /// The 2-D table the loader reads as `name`, as [`Self::table`] reads it.
    pub fn table_of(&self, name: &str) -> Result<TableRows> {
        self.table(self.stored_name(name)?)
    }

    /// Whether the checkpoint holds a tensor the loader reads as `name`.
    pub fn has(&self, name: &str) -> bool {
        self.names.contains_key(name)
    }

    /// The stored name the loader name `name` reads.
    fn stored_name(&self, name: &str) -> Result<&str> {
        self.names
            .get(name)
            .map(String::as_str)
            .ok_or_else(|| Error::msg(format!("missing tensor: {name}")))
    }

    /// The stored tensor `stored` and a view of its bytes.
    fn bytes(&self, stored: &str) -> Result<(&Stored, MappedBytes)> {
        let e = self
            .stored
            .get(stored)
            .ok_or_else(|| Error::msg(format!("safetensors: no tensor {stored}")))?;
        if width(&e.dtype).is_none() {
            return Err(Error::msg(format!(
                "{stored}: unsupported dtype {}",
                e.dtype
            )));
        }
        let shard = &self.shards[e.shard];
        let view =
            MappedBytes::in_file(shard.map.clone(), Some(shard.file.clone()), e.start, e.len)?;
        Ok((e, view))
    }

    /// The ue8m0 scales `<stem>.scale` of the matrix `<stem>.weight`.
    fn scales(&self, weight: &str) -> Result<(&Stored, MappedBytes)> {
        let name = weight
            .strip_suffix(".weight")
            .map(|stem| format!("{stem}.scale"))
            .ok_or_else(|| {
                Error::msg(format!(
                    "{weight}: a block-scaled tensor without a .weight name"
                ))
            })?;
        let (e, view) = self.bytes(&name)?;
        if e.dtype != "F8_E8M0" {
            return Err(Error::msg(format!(
                "{name}: scales are {}, not ue8m0",
                e.dtype
            )));
        }
        Ok((e, view))
    }

    /// The scales of the matrix `weight`, decoded.
    fn scale_values(&self, weight: &str) -> Result<Vec<f32>> {
        let (_, view) = self.scales(weight)?;
        Ok(view.as_slice().iter().map(|&b| e8m0_to_f32(b)).collect())
    }

    /// The rows, columns and tile side of the fp8 matrix `weight`.
    ///
    /// Its scales tile it in square tiles of the configuration's side.
    fn fp8_tiles(&self, weight: &str, e: &Stored) -> Result<(usize, usize, usize)> {
        let [out, inp] = e.shape[..] else {
            return Err(Error::msg(format!("{weight}: fp8 tensor is not 2-D")));
        };
        let block = self.fp8_block.ok_or_else(|| {
            Error::msg(format!(
                "{weight}: fp8 and the configuration gives no square weight_block_size"
            ))
        })?;
        let (scales, _) = self.scales(weight)?;
        if scales.shape != [out.div_ceil(block), inp.div_ceil(block)] {
            return Err(Error::msg(format!(
                "{weight}: scales {:?} do not tile [{out}, {inp}] by {block}x{block}",
                scales.shape
            )));
        }
        Ok((out, inp, block))
    }

    /// The rows, inputs and inputs per scale of the fp4 matrix `weight`.
    ///
    /// Its scales hold the same count of scales for every row.
    fn fp4_runs(&self, weight: &str, e: &Stored) -> Result<(usize, usize, usize)> {
        let [out, packed] = e.shape[..] else {
            return Err(Error::msg(format!("{weight}: fp4 tensor is not 2-D")));
        };
        let inp = 2 * packed;
        let (scales, _) = self.scales(weight)?;
        let per_row = scales.shape.get(1).copied().unwrap_or(0);
        if scales.shape.first().copied() != Some(out) || per_row == 0 || inp % per_row != 0 {
            return Err(Error::msg(format!(
                "{weight}: scales {:?} do not tile [{out}, {inp}] by rows",
                scales.shape
            )));
        }
        Ok((out, inp, inp / per_row))
    }

    /// The tensor the loader reads as `name`, decoded to F32 on the host.
    ///
    /// An fp4 matrix counts its values: its last dimension is twice the stored one.
    pub fn dense_f32(&self, name: &str) -> Result<Tensor> {
        let (values, shape) = self.decode(self.stored_name(name)?)?;
        Tensor::from_vec(values, shape, &Device::Cpu)
    }

    /// The values of the stored tensor `stored` decoded to F32, with their shape.
    ///
    /// An fp4 matrix counts its values: its last dimension is twice the stored one.
    fn decode(&self, stored: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let (e, view) = self.bytes(stored)?;
        let bytes = view.as_slice();
        let (values, shape): (Vec<f32>, Vec<usize>) = match e.dtype.as_str() {
            "F32" => (
                bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
                e.shape.clone(),
            ),
            "BF16" => (
                bytes
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect(),
                e.shape.clone(),
            ),
            "F16" => (
                bytes
                    .chunks_exact(2)
                    .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect(),
                e.shape.clone(),
            ),
            "F8_E4M3" => {
                let (out, inp, block) = self.fp8_tiles(stored, e)?;
                let scales = self.scale_values(stored)?;
                (
                    dequant_fp8_blocked(bytes, &scales, out, inp, block),
                    e.shape.clone(),
                )
            }
            "I8" => {
                let (out, inp, run) = self.fp4_runs(stored, e)?;
                let scales = self.scale_values(stored)?;
                (
                    dequant_fp4_blocked(bytes, &scales, out, inp, run),
                    vec![out, inp],
                )
            }
            other => {
                return Err(Error::msg(format!("{stored}: unsupported dtype {other}")));
            }
        };
        Ok((values, shape))
    }

    /// The matrix the loader reads as `name`, in place where it is stored block-scaled or bf16.
    ///
    /// fp8, fp4 and bf16 matrices are views of the mapping.
    /// Any other tensor is decoded to F32.
    pub fn projection(&self, name: &str) -> Result<Projection> {
        let stored = self.stored_name(name)?;
        let (e, view) = self.bytes(stored)?;
        match e.dtype.as_str() {
            "F8_E4M3" => {
                let (out, inp, block) = self.fp8_tiles(stored, e)?;
                let (_, scales) = self.scales(stored)?;
                Ok(Projection::Fp8(Arc::new(Fp8Weight::new(
                    view, scales, out, inp, block,
                )?)))
            }
            "BF16" if e.shape.len() == 2 => Ok(Projection::Bf16(Arc::new(Bf16Weight::new(
                view, e.shape[0], e.shape[1],
            )?))),
            "I8" => {
                let [out, packed] = e.shape[..] else {
                    return Err(Error::msg(format!("{stored}: fp4 tensor is not 2-D")));
                };
                let (_, scales) = self.scales(stored)?;
                Ok(Projection::Fp4(Arc::new(Fp4Weight::new(
                    view,
                    scales,
                    out,
                    2 * packed,
                )?)))
            }
            _ => Ok(Projection::Dense(self.dense_f32(name)?)),
        }
    }
}
