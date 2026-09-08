//! The layer types a model file loads its weights into and calls: `Linear`,
//! the norms, `Embedding`, `Conv2d`, and the quantised projection.
//!
//! A layer is a weight plus the shape of its call. That is all this module
//! holds. The operations layers are built from are in [`ops`](super::ops), the
//! loader that fills them in [`VarBuilder`](super::VarBuilder), and the KV
//! caches in [`kv_cache`](super::kv_cache) - each was here once, under the one
//! name `nn`, which could only ever describe the union.

// Split-out files reach these through `use super::*`.
use super::lora::{fuse_loras, LoraDelta};
use super::{DType, Device, Dim, Error, Result, Tensor, VarBuilder, D};

mod conv;
mod embedding;
mod linear;
mod norm;
pub mod qlinear;

pub use conv::*;
pub use embedding::*;
pub use linear::*;
pub use norm::*;
fn _assert_dim_used<I: Dim>(_: I) {}

#[cfg(test)]
mod tests;

/// Declare the fall-backs a `serde` config needs, as a table rather than as one function each.
///
/// A checkpoint states most of its configuration and leaves the rest to the reader, so every
/// optional field needs a path to a function returning its default. Written out, that is one
/// three-line function per field and a table nobody can read; here the values sit together
/// where they can be compared with the checkpoint that omitted them.
#[macro_export]
macro_rules! serde_defaults {
    ($($(#[$note:meta])* $name:ident: $ty:ty = $value:expr;)+) => {
        $(
            $(#[$note])*
            fn $name() -> $ty {
                $value
            }
        )+
    };
}
