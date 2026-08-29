//! Legacy-shaped `Module` (forward) + `IndexOp` (`.i()`) + `TensorId`, native-homed
//! (compat->native). `compat` re-exports them so the ~64 `.forward()` / 20 `.i()` / 14
//! `TensorId` sites resolve unchanged; the impls are for `native::Tensor` directly.
use crate::tensor::{Error, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(pub(crate) usize);

pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<M: Module> Module for &M {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        (*self).forward(xs)
    }
}

/// legacy-shaped indexing: `t.i(2)?`, `t.i((.., 3..7))?`, ...
#[derive(Clone, Debug)]
pub enum TensorIndexer {
    /// Pick one index (removes the dim).
    Select(usize),
    /// Range (keeps the dim).
    Narrow(std::ops::Bound<usize>, std::ops::Bound<usize>),
}

impl From<usize> for TensorIndexer {
    fn from(i: usize) -> Self {
        Self::Select(i)
    }
}

macro_rules! range_indexer {
    ($ty:ty) => {
        impl From<$ty> for TensorIndexer {
            fn from(r: $ty) -> Self {
                use std::ops::RangeBounds;
                Self::Narrow(r.start_bound().cloned(), r.end_bound().cloned())
            }
        }
    };
}

range_indexer!(std::ops::Range<usize>);
range_indexer!(std::ops::RangeFrom<usize>);
range_indexer!(std::ops::RangeTo<usize>);
range_indexer!(std::ops::RangeInclusive<usize>);
range_indexer!(std::ops::RangeToInclusive<usize>);
range_indexer!(std::ops::RangeFull);

fn apply_indexers(t: &Tensor, idx: &[TensorIndexer]) -> Result<Tensor> {
    use std::ops::Bound;
    let mut cur = t.clone();
    let mut dim = 0usize;
    for ix in idx {
        match ix {
            TensorIndexer::Select(i) => {
                cur = cur.narrow(dim, *i, 1)?.squeeze(dim)?;
                // dim stays (the squeezed dim is gone)
            }
            TensorIndexer::Narrow(s, e) => {
                let size = cur.dims()[dim];
                let start = match s {
                    Bound::Included(v) => *v,
                    Bound::Excluded(v) => v + 1,
                    Bound::Unbounded => 0,
                };
                let end = match e {
                    Bound::Included(v) => v + 1,
                    Bound::Excluded(v) => *v,
                    Bound::Unbounded => size,
                };
                if end < start || end > size {
                    return Err(Error::msg(format!(
                        "index {start}..{end} out of range for dim {dim} of size {size}"
                    )));
                }
                cur = cur.narrow(dim, start, end - start)?;
                dim += 1;
            }
        }
    }
    Ok(cur)
}

pub trait IndexOp<T> {
    fn i(&self, index: T) -> Result<Tensor>;
}

impl<A: Into<TensorIndexer>> IndexOp<A> for Tensor {
    fn i(&self, index: A) -> Result<Tensor> {
        apply_indexers(self, &[index.into()])
    }
}

macro_rules! index_op_tuple {
    ($($name:ident),+) => {
        #[allow(non_snake_case)]
        impl<$($name: Into<TensorIndexer>,)+> IndexOp<($($name,)+)> for Tensor {
            fn i(&self, ($($name,)+): ($($name,)+)) -> Result<Tensor> {
                apply_indexers(self, &[$($name.into(),)+])
            }
        }
    };
}

index_op_tuple!(A, B);
index_op_tuple!(A, B, C);
index_op_tuple!(A, B, C, D2);
index_op_tuple!(A, B, C, D2, E);
index_op_tuple!(A, B, C, D2, E, F);
