//! Stable Diffusion XL: UNet, VAE, text encoders, ControlNet, sampler, pipeline.
//!
//! `crate::inference::model::sdxl::<part>`

pub mod controlnet;
pub mod pipeline;
pub mod sampling;
pub mod text;
pub mod unet;
pub mod vae;
