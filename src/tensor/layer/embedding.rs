//! Token embedding: the lookup, and the transposed matmul that unties it for output.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;
use std::sync::Arc;

/// Token embedding (`[vocab, dim]` table; u32 id lookup).
///
/// The table is either a dense tensor, or the stored blocks of a checkpoint read row by row.
/// A stored table is never widened: a lookup decodes the `n` rows asked for, so the resident
/// cost is the file's own bytes (a view of the mapping) and the rest is `n * dim` floats.
#[derive(Clone, Debug)]
pub struct Embedding {
    /// The dense table. For a stored table, a one-row F32 anchor on the host that carries the
    /// row width and the device the looked-up rows land on.
    table: Tensor,
    /// The stored `[vocab, dim]` blocks, when the table is kept in the checkpoint's format.
    stored: Option<Arc<crate::tensor::quantized::QTensor>>,
}

impl Embedding {
    pub fn new(table: Tensor) -> Self {
        Self {
            table,
            stored: None,
        }
    }

    /// A table kept in the checkpoint's own format and decoded per lookup, on the host.
    /// The row length must be a whole number of blocks.
    pub fn from_stored(qt: Arc<crate::tensor::quantized::QTensor>) -> Result<Self> {
        let dims = qt.shape().dims().to_vec();
        let [_, dim] = dims[..] else {
            return Err(Error::msg(format!(
                "embedding table must be 2-D, got {dims:?}"
            )));
        };
        let block = qt.dtype().block_size();
        if dim % block != 0 {
            return Err(Error::msg(format!(
                "embedding row of {dim} is not whole {:?} blocks",
                qt.dtype()
            )));
        }
        let anchor = Tensor::zeros((1usize, dim), crate::tensor::DType::F32)?;
        Ok(Self {
            table: anchor,
            stored: Some(qt),
        })
    }

    /// Whether the table is the checkpoint's blocks rather than a dense tensor.
    pub fn is_stored(&self) -> bool {
        self.stored.is_some()
    }

    /// The device a lookup runs on and its rows land on.
    pub fn device(&self) -> Device {
        self.table.device().clone()
    }

    /// Bytes the table holds resident that the checkpoint file does not already hold: zero
    /// for a stored table, the dense size otherwise.
    pub fn owned_bytes(&self) -> u64 {
        if self.stored.is_some() {
            return 0;
        }
        (self.table.elem_count() * self.table.dtype().size_in_bytes()) as u64
    }

    /// Decode the rows `ids` names from the stored blocks into `[ids.len(), dim]` F32.
    fn stored_rows(&self, qt: &crate::tensor::quantized::QTensor, ids: &[u32]) -> Result<Tensor> {
        let dims = qt.shape().dims();
        let (vocab, dim) = (dims[0], dims[1]);
        let dtype = qt.dtype();
        let row_bytes = dim / dtype.block_size() * dtype.type_size();
        let data = qt.data()?;
        let mut out = vec![0f32; ids.len() * dim];
        // A row's byte offset is a multiple of its length, not of the block alignment, so each
        // row is decoded from an aligned copy of itself.
        let mut scratch = vec![0u32; row_bytes.div_ceil(4)];
        for (i, &id) in ids.iter().enumerate() {
            let id = id as usize;
            if id >= vocab {
                return Err(Error::msg(format!(
                    "token id {id} outside the table of {vocab} rows"
                )));
            }
            let bytes = unsafe {
                std::slice::from_raw_parts_mut(scratch.as_mut_ptr() as *mut u8, row_bytes)
            };
            bytes.copy_from_slice(&data[id * row_bytes..(id + 1) * row_bytes]);
            crate::tensor::quant_cpu::to_float_bytes(
                dtype,
                bytes,
                &mut out[i * dim..(i + 1) * dim],
            )?;
        }
        Tensor::from_vec(out, (ids.len(), dim), &Device::Cpu)
    }

    /// The raw `[vocab, dim]` table (weight-tied lm-heads read it back).
    /// The table, under the name the other implementation used for it.
    pub fn embeddings(&self) -> &Tensor {
        &self.table
    }

    /// The width of one row.
    ///
    /// Derived from the table rather than stored beside it: two fields that must agree is how
    /// a table and its declared width come apart, and the table is the one that is true.
    pub fn hidden_size(&self) -> usize {
        self.table.dims().last().copied().unwrap_or(0)
    }

    pub fn table(&self) -> &Tensor {
        &self.table
    }

    /// Move the embedding table onto `dev` (CPU↔CUDA).
    pub fn to_device(&self, dev: &Device) -> Result<Self> {
        if self.stored.is_some() {
            return Err(Error::msg("a stored embedding table stays on the host"));
        }
        Ok(Self::new(self.table.to_device(dev)?))
    }

    /// Copy on `dev` converted to `dtype`.
    pub fn to_dtype_on(&self, dev: &Device, dtype: crate::tensor::DType) -> Result<Self> {
        if self.stored.is_some() {
            return Err(Error::msg("a stored embedding table is not converted"));
        }
        Ok(Self::new(self.table().to_device(dev)?.to_dtype(dtype)?))
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let id_dims = ids.dims().to_vec();
        let flat = ids.reshape(vec![ids.elem_count()])?;
        let rows = match &self.stored {
            Some(qt) => {
                let host: Vec<u32> = flat
                    .to_device(&Device::Cpu)?
                    .to_dtype(crate::tensor::DType::U32)?
                    .to_vec1()?;
                self.stored_rows(qt, &host)?
            }
            None => self.table.index_select(&flat, 0)?,
        };
        let dim = self.table.dim(super::D::Minus1)?;
        let mut odims = id_dims;
        odims.push(dim);
        rows.reshape(odims)
    }
}

pub fn embedding(vocab: usize, dim: usize, vb: &VarBuilder) -> Result<Embedding> {
    Ok(Embedding::new(vb.get((vocab, dim), "weight")?))
}

// silence unused-import lint when cuda is off (Dim used via D::Minus1 path)

//
// The `Module` impl lives with the type it makes callable; it used to sit in another
// area of the crate entirely.
impl crate::tensor::Module for Embedding {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward(xs)
    }
}
