//! Operations over tensors, as free functions.
//!
//! Separate from the layer types next door because the two families collide by name and mean
//! different things: `layer::rms_norm(size, eps, vb)` BUILDS a layer from a checkpoint, while
//! `ops::rms_norm(x, weight, eps)` APPLIES the normalisation to a tensor. Keeping them in
//! one namespace would force one of them to be called something it is not.

/// The traits a tensor answers to: indexing, `Module`, identity.
pub mod traits;

// Named imports rather than a glob: a glob would pull the layer BUILDERS in beside the ops
// and make `rms_norm` ambiguous inside this very module - the collision this split exists
// to resolve.
use super::{DType, Device, Result, Tensor, D};

mod activation;
mod attention;
mod host;
mod norm;
mod softmax;

pub use activation::*;
pub use attention::*;
pub use host::*;
pub use norm::*;
pub use softmax::*;

#[cfg(test)]
mod tests;
