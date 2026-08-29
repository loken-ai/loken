//! Z-Image.
//!
//! `crate::inference::model::zimage::<part>`

pub mod dit;
pub mod sampling;
pub mod text_encoder;
pub mod vae;

/// Splits the 34 blocks across CUDA, Arc and CPU. Already built on `dit`'s blocks.
#[cfg(feature = "image")]
pub mod hetero;
/// Intel Arc, via OpenCL kernels written against this architecture directly.
#[cfg(all(feature = "opencl", feature = "image"))]
pub mod opencl;
