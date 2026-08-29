//! FLUX.1: the shared config and ops, the DiT, the sampler, the VAE.
//!
//! `crate::inference::model::flux::<part>`

pub mod common;
pub mod sampling;
pub mod vae;

/// The transformer, and the two placements it runs under - whole on one device, or split
/// across several.
#[cfg(feature = "image")]
pub mod hetero;
/// Intel Arc: not a fallback but a device the substrate does not cover, so the blocks are
/// written against OpenCL kernels directly.
#[cfg(all(feature = "opencl", feature = "image"))]
pub mod opencl;
