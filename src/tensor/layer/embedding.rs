//! Token embedding: the lookup, and the transposed matmul that unties it for output.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Token embedding (`[vocab, dim]` table; u32 id lookup).
#[derive(Clone, Debug)]
pub struct Embedding {
    table: Tensor,
}

impl Embedding {
    pub fn new(table: Tensor) -> Self {
        Self { table }
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
        Ok(Self {
            table: self.table.to_device(dev)?,
        })
    }

    /// Copy on `dev` converted to `dtype`.
    pub fn to_dtype_on(&self, dev: &Device, dtype: crate::tensor::DType) -> Result<Self> {
        Ok(Self::new(self.table().to_device(dev)?.to_dtype(dtype)?))
    }

    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        let id_dims = ids.dims().to_vec();
        let flat = ids.reshape(vec![ids.elem_count()])?;
        let rows = self.table.index_select(&flat, 0)?;
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
