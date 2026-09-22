//! The released checkpoint as a weight source: safetensors shards read in place.
//!
//! Every shard of the directory is mapped and its header indexed by name; a tensor's bytes are a
//! range of one mapping. What is stored dense (bf16, f32) is converted on read; what is stored
//! block-scaled fp8 (`<name>.weight` e4m3 with `<name>.scale` ue8m0 per `block x block`) is
//! dequantised on read or left in place; what is stored block-scaled fp4 (packed nibbles with a
//! ue8m0 scale per `block` inputs) - the routed experts - is read in place by `Fp4Weight`. The
//! block sizes come from the model's own `config.json`.

use super::engram::EngramTable;
use super::source::WeightSource;
use crate::inference::offload::experts::Expert;
use crate::inference::offload::projection::Projection;
use crate::inference::offload::store::ExpertLoader;
use crate::tensor::blockscaled::bf16::Bf16Weight;
use crate::tensor::blockscaled::dequant::{dequant_fp4_blocked, dequant_fp8_blocked, e8m0_to_f32};
use crate::tensor::blockscaled::fp4::Fp4Weight;
use crate::tensor::blockscaled::fp8::Fp8Weight;
use crate::tensor::mapped::MappedBytes;
use crate::tensor::{Device, Error, Result, Tensor};
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

struct Entry {
    map: usize,
    dtype: Dtype,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

struct Inner {
    maps: Vec<Arc<memmap2::Mmap>>,
    files: Vec<Arc<std::fs::File>>,
    index: HashMap<String, Entry>,
    /// The fp8 weight block, `[rows, cols]` per scale, from `quantization_config.weight_block_size`.
    fp8_block: (usize, usize),
}

/// The checkpoint directory: shards, `config.json`, `inference/config.json`.
pub struct SafeTensorsSource {
    inner: Arc<Inner>,
    /// The model's own `config.json`.
    pub config: serde_json::Value,
    /// The reference implementation's arguments, `inference/config.json`.
    pub inference_config: serde_json::Value,
}

impl SafeTensorsSource {
    /// Map every `*.safetensors` under `dir` and read both configs.
    pub fn open(dir: &Path) -> Result<Self> {
        let read_json = |p: &Path| -> Result<serde_json::Value> {
            let txt = std::fs::read_to_string(p)
                .map_err(|e| Error::msg(format!("{}: {e}", p.display())))?;
            serde_json::from_str(&txt).map_err(|e| Error::msg(format!("{}: {e}", p.display())))
        };
        let config = read_json(&dir.join("config.json"))?;
        let inference_config = read_json(&dir.join("inference").join("config.json"))?;
        let fp8_block = {
            let b = config
                .get("quantization_config")
                .and_then(|q| q.get("weight_block_size"))
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    Error::msg("config.json: no quantization_config.weight_block_size")
                })?;
            let n = |i: usize| b.get(i).and_then(|v| v.as_u64()).map(|v| v as usize);
            (
                n(0).ok_or_else(|| Error::msg("weight_block_size[0]"))?,
                n(1).ok_or_else(|| Error::msg("weight_block_size[1]"))?,
            )
        };
        let mut shards: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| Error::msg(format!("{}: {e}", dir.display())))?
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(Error::msg(format!(
                "{}: no safetensors shard",
                dir.display()
            )));
        }
        let mut maps = Vec::with_capacity(shards.len());
        let mut files = Vec::with_capacity(shards.len());
        let mut index = HashMap::new();
        for (i, p) in shards.iter().enumerate() {
            let file =
                std::fs::File::open(p).map_err(|e| Error::msg(format!("{}: {e}", p.display())))?;
            // Safety: the checkpoint is not written while served, the contract of every mapped loader.
            let map = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| Error::msg(format!("{}: {e}", p.display())))?;
            let (header_len, meta) = safetensors::SafeTensors::read_metadata(&map)
                .map_err(|e| Error::msg(format!("{}: {e}", p.display())))?;
            let data0 = 8 + header_len;
            for (name, info) in meta.tensors() {
                let (s, e) = info.data_offsets;
                index.insert(
                    name.to_string(),
                    Entry {
                        map: i,
                        dtype: info.dtype,
                        shape: info.shape.clone(),
                        start: data0 + s,
                        end: data0 + e,
                    },
                );
            }
            maps.push(Arc::new(map));
            files.push(Arc::new(file));
        }
        Ok(Self {
            inner: Arc::new(Inner {
                maps,
                files,
                index,
                fp8_block,
            }),
            config,
            inference_config,
        })
    }

    pub fn names(&self) -> Vec<&str> {
        self.inner.index.keys().map(|s| s.as_str()).collect()
    }

    /// A tensor's bytes exactly as the shard stores them, with its stored shape and type name: for a
    /// converter that carries a layout through unchanged.
    pub fn raw(&self, name: &str) -> Result<(String, Vec<usize>, MappedBytes)> {
        let (e, b) = self.inner.bytes(name)?;
        Ok((format!("{:?}", e.dtype), e.shape.clone(), b))
    }
}

impl Inner {
    fn entry(&self, name: &str) -> Result<&Entry> {
        self.index
            .get(name)
            .ok_or_else(|| Error::msg(format!("safetensors: no tensor {name}")))
    }

    fn bytes(&self, name: &str) -> Result<(&Entry, MappedBytes)> {
        let e = self.entry(name)?;
        Ok((
            e,
            MappedBytes::in_file(
                self.maps[e.map].clone(),
                Some(self.files[e.map].clone()),
                e.start,
                e.end - e.start,
            )?,
        ))
    }

    /// `<stem>.scale` for `<stem>.weight`.
    fn scale_name(name: &str) -> Option<String> {
        name.strip_suffix(".weight").map(|s| format!("{s}.scale"))
    }

    fn scales_f32(&self, name: &str) -> Result<Vec<f32>> {
        let (e, b) = self.bytes(name)?;
        if e.dtype != Dtype::F8_E8M0 {
            return Err(Error::msg(format!(
                "{name}: scales are {:?}, not ue8m0",
                e.dtype
            )));
        }
        Ok(b.as_slice().iter().map(|&x| e8m0_to_f32(x)).collect())
    }

    /// The logical shape: fp4 nibbles unpacked.
    fn logical_shape(&self, e: &Entry) -> Vec<usize> {
        let mut s = e.shape.clone();
        if e.dtype == Dtype::I8 {
            if let Some(last) = s.last_mut() {
                *last *= 2;
            }
        }
        s
    }

    fn dense_f32(&self, name: &str) -> Result<Tensor> {
        let (e, b) = self.bytes(name)?;
        let bytes = b.as_slice();
        let shape = self.logical_shape(e);
        let n: usize = shape.iter().product();
        let values: Vec<f32> = match e.dtype {
            Dtype::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::BF16 => bytes
                .chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            Dtype::F16 => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            Dtype::F8_E4M3 => {
                let scale_name = Inner::scale_name(name).ok_or_else(|| {
                    Error::msg(format!("{name}: fp8 tensor without a .weight name"))
                })?;
                let (se, _) = self.bytes(&scale_name)?;
                let [out, inp] = shape[..] else {
                    return Err(Error::msg(format!("{name}: fp8 tensor is not 2-D")));
                };
                let (br, bc) = self.fp8_block;
                if br != bc || se.shape != vec![out.div_ceil(br), inp.div_ceil(bc)] {
                    return Err(Error::msg(format!(
                        "{name}: scales {:?} do not tile [{out}, {inp}] by {br}x{bc}",
                        se.shape
                    )));
                }
                dequant_fp8_blocked(bytes, &self.scales_f32(&scale_name)?, out, inp, br)
            }
            Dtype::I8 => {
                let scale_name = Inner::scale_name(name).ok_or_else(|| {
                    Error::msg(format!("{name}: fp4 tensor without a .weight name"))
                })?;
                let (se, _) = self.bytes(&scale_name)?;
                let [out, inp] = shape[..] else {
                    return Err(Error::msg(format!("{name}: fp4 tensor is not 2-D")));
                };
                let per_row = se.shape.get(1).copied().unwrap_or(0);
                if se.shape.first().copied() != Some(out) || per_row == 0 || inp % per_row != 0 {
                    return Err(Error::msg(format!(
                        "{name}: scales {:?} do not tile [{out}, {inp}] by rows",
                        se.shape
                    )));
                }
                dequant_fp4_blocked(
                    bytes,
                    &self.scales_f32(&scale_name)?,
                    out,
                    inp,
                    inp / per_row,
                )
            }
            other => return Err(Error::msg(format!("{name}: unsupported dtype {other:?}"))),
        };
        if values.len() != n {
            return Err(Error::msg(format!(
                "{name}: {} values for shape {shape:?}",
                values.len()
            )));
        }
        Tensor::from_vec(values, shape, &Device::Cpu)
    }

    fn projection(&self, name: &str) -> Result<Option<Projection>> {
        let (e, b) = self.bytes(name)?;
        if e.dtype == Dtype::F8_E4M3 {
            let scale_name = Inner::scale_name(name)
                .ok_or_else(|| Error::msg(format!("{name}: fp8 tensor without a .weight name")))?;
            let (se, sb) = self.bytes(&scale_name)?;
            let [out, inp] = e.shape[..] else {
                return Err(Error::msg(format!("{name}: fp8 tensor is not 2-D")));
            };
            let (br, bc) = self.fp8_block;
            if br != bc || se.shape != vec![out.div_ceil(br), inp.div_ceil(bc)] {
                return Err(Error::msg(format!(
                    "{name}: scales {:?} do not tile [{out}, {inp}] by {br}x{bc}",
                    se.shape
                )));
            }
            return Ok(Some(Projection::Fp8(Arc::new(Fp8Weight::new(
                b, sb, out, inp, br,
            )?))));
        }
        if e.dtype == Dtype::BF16 && e.shape.len() == 2 {
            return Ok(Some(Projection::Bf16(Arc::new(Bf16Weight::new(
                b, e.shape[0], e.shape[1],
            )?))));
        }
        if e.dtype != Dtype::I8 {
            return Ok(None);
        }
        let scale_name = Inner::scale_name(name)
            .ok_or_else(|| Error::msg(format!("{name}: fp4 tensor without a .weight name")))?;
        let (_, sb) = self.bytes(&scale_name)?;
        let shape = self.logical_shape(e);
        let [out, inp] = shape[..] else {
            return Err(Error::msg(format!("{name}: fp4 tensor is not 2-D")));
        };
        Ok(Some(Projection::Fp4(Arc::new(Fp4Weight::new(
            b, sb, out, inp,
        )?))))
    }
}

impl WeightSource for SafeTensorsSource {
    fn shape(&self, name: &str) -> Option<Vec<usize>> {
        self.inner
            .index
            .get(name)
            .map(|e| self.inner.logical_shape(e))
    }

    fn dense_f32(&self, name: &str) -> Result<Tensor> {
        self.inner.dense_f32(name)
    }

    fn projection(&self, name: &str) -> Result<Option<Projection>> {
        self.inner.projection(name)
    }

    fn experts(&self, layer: usize, n: usize) -> Result<Box<dyn ExpertLoader>> {
        Ok(Box::new(ShardExpertLoader {
            inner: self.inner.clone(),
            prefix: format!("layers.{layer}.ffn.experts"),
            n,
            dense: false,
        }))
    }

    fn engram_table(&self, name: &str) -> Result<EngramTable> {
        let (e, b) = self.inner.bytes(name)?;
        let [rows, row_len] = e.shape[..] else {
            return Err(Error::msg(format!("{name}: engram table is not 2-D")));
        };
        if e.dtype != Dtype::F8_E4M3 {
            return Ok(EngramTable::Resident(self.inner.dense_f32(name)?));
        }
        let scale_name = Inner::scale_name(name)
            .ok_or_else(|| Error::msg(format!("{name}: fp8 table without a .weight name")))?;
        let (se, sb) = self.inner.bytes(&scale_name)?;
        let per_row = se.shape.get(1).copied().unwrap_or(0);
        if se.shape.first().copied() != Some(rows) || per_row == 0 || row_len % per_row != 0 {
            return Err(Error::msg(format!(
                "{name}: scales {:?} do not tile [{rows}, {row_len}] by rows",
                se.shape
            )));
        }
        let table = EngramTable::Fp8Rows {
            bytes: b,
            scales: sb,
            rows,
            row_len,
            block: row_len / per_row,
        };
        table.advise_random();
        Ok(table)
    }
}

/// Each routed expert from its own three tensors, read in place when fp4, dense otherwise. The
/// prefix names the block: `layers.{layer}.ffn.experts` for the backbone, `mtp.{stage}.ffn.experts`
/// for a DSpark stage.
struct ShardExpertLoader {
    inner: Arc<Inner>,
    prefix: String,
    n: usize,
    /// Dequantise to a dense f32 weight instead of keeping the block-scaled form. A DSpark draft
    /// runs its few experts on the cores; a dense dot vectorises where unpacking fp4 per element
    /// does not, so the draft is far cheaper for the handful of experts it touches.
    dense: bool,
}

impl ExpertLoader for ShardExpertLoader {
    fn build(&self, id: usize) -> Result<Expert> {
        let one = |w: &str| -> Result<Projection> {
            let name = format!("{}.{id}.{w}.weight", self.prefix);
            if self.dense {
                return Ok(Projection::Dense(self.inner.dense_f32(&name)?));
            }
            match self.inner.projection(&name)? {
                Some(p) => Ok(p),
                None => Ok(Projection::Dense(self.inner.dense_f32(&name)?)),
            }
        };
        Ok(Expert {
            w1: one("w1")?,
            w2: one("w2")?,
            w3: one("w3")?,
        })
    }

    fn count(&self) -> usize {
        self.n
    }
}

impl SafeTensorsSource {
    /// An expert loader for a block named by `prefix` (e.g. `mtp.0.ffn.experts`), for loading a
    /// DSpark stage's routed experts by their stored names. `dense` dequantises each to f32.
    pub fn experts_named(&self, prefix: &str, n: usize, dense: bool) -> Box<dyn ExpertLoader> {
        Box::new(ShardExpertLoader {
            inner: self.inner.clone(),
            prefix: prefix.to_string(),
            n,
            dense,
        })
    }
}
