//! The tensors of the checkpoint the two transformers do not read.
//!
//! A lookup table stays in the mapped file and is read by row on the host.
//! A projection is held F32 on the device it runs on.

use crate::tensor::{DType, Device, Error, Result, Tensor};
use std::sync::Arc;

/// One mapped shard: its bytes, where its data starts, and its listing.
pub struct Shard {
    map: Arc<memmap2::Mmap>,
    data_start: usize,
    listing: safetensors::tensor::Metadata,
}

impl Shard {
    /// The shards at `paths`, mapped.
    pub fn map_all(paths: &[String]) -> Result<Vec<Self>> {
        paths
            .iter()
            .map(|p| {
                let file = std::fs::File::open(p).map_err(|e| Error::msg(format!("{p}: {e}")))?;
                let map = unsafe { memmap2::Mmap::map(&file) }
                    .map_err(|e| Error::msg(format!("{p}: {e}")))?;
                let (len, listing) = safetensors::SafeTensors::read_metadata(&map)
                    .map_err(|e| Error::msg(format!("{p}: {e}")))?;
                Ok(Self {
                    map: Arc::new(map),
                    data_start: 8 + len,
                    listing,
                })
            })
            .collect()
    }
}

/// A `[rows, width]` table read by row from its mapped shard.
#[derive(Clone)]
pub struct HostTable {
    map: Arc<memmap2::Mmap>,
    start: usize,
    rows: usize,
    width: usize,
    dtype: safetensors::Dtype,
}

impl HostTable {
    /// The table `name` of `shards`.
    pub fn open(shards: &[Shard], name: &str) -> Result<Self> {
        let (shard, info) = shards
            .iter()
            .find_map(|s| Some((s, s.listing.info(name)?)))
            .ok_or_else(|| Error::msg(format!("the checkpoint has no {name}")))?;
        let [rows, width] = info.shape[..] else {
            return Err(Error::msg(format!("{name} is not a table")));
        };
        if !matches!(
            info.dtype,
            safetensors::Dtype::F32 | safetensors::Dtype::F16 | safetensors::Dtype::BF16
        ) {
            return Err(Error::msg(format!("{name} is {:?}", info.dtype)));
        }
        Ok(Self {
            map: shard.map.clone(),
            start: shard.data_start + info.data_offsets.0,
            rows,
            width,
            dtype: info.dtype,
        })
    }

    /// The width of a row.
    pub fn width(&self) -> usize {
        self.width
    }

    /// The row `id` added to `out` as F32.
    pub fn add_row(&self, id: u32, out: &mut [f32]) -> Result<()> {
        let id = id as usize;
        if id >= self.rows || out.len() != self.width {
            return Err(Error::msg(format!(
                "row {id} of a table of {} rows of {}",
                self.rows, self.width
            )));
        }
        let size = self.dtype.bitsize() / 8;
        let at = self.start + id * self.width * size;
        let bytes = &self.map[at..at + self.width * size];
        for (o, b) in out.iter_mut().zip(bytes.chunks_exact(size)) {
            *o += match self.dtype {
                safetensors::Dtype::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                safetensors::Dtype::F16 => half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
                _ => half::bf16::from_le_bytes([b[0], b[1]]).to_f32(),
            };
        }
        Ok(())
    }

    /// The rows of `ids`, `[ids.len(), width]` F32 on the host.
    pub fn rows(&self, ids: &[u32]) -> Result<Tensor> {
        let mut out = vec![0f32; ids.len() * self.width];
        for (id, row) in ids.iter().zip(out.chunks_exact_mut(self.width)) {
            self.add_row(*id, row)?;
        }
        Tensor::from_vec(out, (ids.len(), self.width), &Device::Cpu)
    }
}

/// `x W^T + b` of a stored projection, held F32 on its device.
pub struct Linear {
    /// `W^T`, `[in, out]`.
    wt: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    /// The projection `name` (`name.weight`, and `name.bias` when `bias`) of `st`, on `device`.
    pub fn load(
        st: &crate::tensor::safetensors::MmapedSafetensors,
        name: &str,
        bias: bool,
        device: &Device,
    ) -> Result<Self> {
        // CONVERSION: a projection of the glue is held F32 on the device it runs on.
        let load = |n: String| -> Result<Tensor> {
            st.load(&n, &Device::Cpu)?
                .to_dtype(DType::F32)?
                .to_device(device)
        };
        Ok(Self {
            wt: load(format!("{name}.weight"))?.t()?.contiguous()?,
            bias: match bias {
                true => Some(load(format!("{name}.bias"))?),
                false => None,
            },
        })
    }

    /// Bytes the projection named `name` takes held F32, read from the header of `st`.
    pub fn bytes(st: &crate::tensor::safetensors::MmapedSafetensors, name: &str) -> u64 {
        [format!("{name}.weight"), format!("{name}.bias")]
            .iter()
            .filter_map(|n| st.shape_of(n))
            .map(|s| (s.iter().product::<usize>() * std::mem::size_of::<f32>()) as u64)
            .sum()
    }

    /// The device the projection runs on.
    pub fn device(&self) -> Device {
        self.wt.device().clone()
    }

    /// `x` `[n, in]` projected to `[n, out]`, on the projection's device.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.to_device(&self.device())?.to_dtype(DType::F32)?;
        let y = x.matmul(&self.wt)?;
        match &self.bias {
            Some(b) => y.broadcast_add(b),
            None => Ok(y),
        }
    }
}
