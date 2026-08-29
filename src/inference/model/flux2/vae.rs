//! FLUX.2 VAE (`AutoencoderKLFlux2`) - decode side.
//!
//! Architecturally this is the same KL autoencoder the rest of the fleet already runs: four
//! `[128, 256, 512, 512]` stages, two resnets each, 32 group-norm groups, one mid-block
//! attention. Only two things are its own, and both live OUTSIDE the conv stack:
//!
//! 1. **A 32-channel latent that the pipeline keeps PATCHIFIED at 2x2**, so the DiT sees 128
//!    channels over a half-resolution grid. Unpatchifying is therefore part of decoding, not a
//!    detail of the sampler.
//! 2. **The latent normalisation is a `BatchNorm2d`'s running statistics**, not the
//!    `scaling_factor` / `shift_factor` pair every other VAE here carries. The stats live in the
//!    checkpoint as `bn.running_mean` / `bn.running_var` over the 128 patchified channels, and
//!    the decode path un-normalises with `x * sqrt(var + eps) + mean`. Feeding the conv stack a
//!    latent that still carries those statistics does not fail - it desaturates.
//!
//! So the reuse is total on the conv stack (`native_zimage_vae`'s Decoder, which already walks
//! the diffusers names this checkpoint uses - no rewrite) and the new code is only the latent
//! handling above.

use crate::inference::model::zimage::vae as zvae;
use crate::inference::model::zimage::vae::{Decoder, VaeConfig};
use crate::tensor::layer::{conv2d, Conv2d, Conv2dConfig};
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Result, Tensor};

/// The VAE's own spatial reduction. The pipeline's 2x2 latent patchify sits ON TOP of this, so
/// pixels-to-DiT-grid is 16, not 8.
pub const VAE_STRIDE: usize = 8;
/// The pipeline's latent patch size, which is what turns 32 latent channels into the DiT's 128.
pub const LATENT_PATCH: usize = 2;

/// `AutoencoderKLFlux2` geometry. `scaling_factor` / `shift_factor` are deliberately the
/// identity: this family does NOT scale its latent that way (see the module docs), and leaving a
/// borrowed 0.3611/0.1159 here would silently wash every decode out.
pub fn flux2_vae_config() -> VaeConfig {
    VaeConfig {
        in_channels: 3,
        out_channels: 3,
        latent_channels: 32,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        scaling_factor: 1.0,
        shift_factor: 0.0,
        norm_num_groups: 32,
    }
}

pub struct Flux2Vae {
    decoder: Decoder,
    post_quant_conv: Conv2d,
    /// `bn.running_mean`, `[1, 128, 1, 1]` for broadcasting over the patchified latent.
    bn_mean: Tensor,
    /// `sqrt(bn.running_var + eps)`, same shape.
    bn_std: Tensor,
    device: Device,
}

impl Flux2Vae {
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// DiT tokens `[S, 128]` over a `gh x gw` grid -> image `[1, 3, gh*16, gw*16]` in `[-1, 1]`.
    ///
    /// Three steps that all belong together: re-grid the tokens, un-normalise with the batch-norm
    /// statistics, unpatchify 128 -> 32 channels at double resolution, then run the conv decoder.
    pub fn decode_tokens(&self, tokens: &Tensor, gh: usize, gw: usize) -> Result<Tensor> {
        let dims = tokens.shape().dims().to_vec();
        let c_patched = dims[dims.len() - 1];
        let z = tokens
            .to_device(&self.device)?
            .reshape((gh, gw, c_patched))?
            // [gh, gw, C] -> [1, C, gh, gw]
            .permute((2, 0, 1))?
            .contiguous()?
            .reshape((1, c_patched, gh, gw))?;
        let z = z
            .broadcast_mul(&self.bn_std)?
            .broadcast_add(&self.bn_mean)?;
        let z = unpatchify(&z)?;
        self.decoder.forward(&self.post_quant_conv.forward(&z)?)
    }
}

/// `[b, c*4, h, w] -> [b, c, h*2, w*2]`, where the packed channel index is
/// `c*4 + pi*2 + pj`. This is the inverse of the pipeline's `_patchify_latents`, and it is the
/// same packing the Qwen-Image port uses - getting the two patch axes the wrong way round
/// transposes every 2x2 cell, which reads as a fine cross-hatch rather than as an error.
fn unpatchify(z: &Tensor) -> Result<Tensor> {
    let d = z.shape().dims().to_vec();
    let (b, cp, h, w) = (d[0], d[1], d[2], d[3]);
    let c = cp / (LATENT_PATCH * LATENT_PATCH);
    z.reshape((b, c, LATENT_PATCH, LATENT_PATCH, h, w))?
        // (b, c, pi, pj, h, w) -> (b, c, h, pi, w, pj)
        .permute((0, 1, 4, 2, 5, 3))?
        .contiguous()?
        .reshape((b, c, h * LATENT_PATCH, w * LATENT_PATCH))
}

/// Load the decode side from the diffusers `vae/diffusion_pytorch_model.safetensors`.
pub fn load(path: &str, device: &Device, dtype: DType) -> Result<Flux2Vae> {
    let cfg = flux2_vae_config();
    // NO name rewrite. The shared conv stack already walks DIFFUSERS names
    // (`decoder.mid_block.resnets.0...`, `decoder.up_blocks.N.resnets.M...`) and this checkpoint
    // is diffusers-named too, so the two meet directly. The SDXL VAE maps because ITS checkpoint
    // is an LDM single-file, not because the stack wants LDM - borrowing that rewrite here turned
    // valid names into `decoder.mid.block_1.*`, which exists nowhere.
    let vb = unsafe { VarBuilder::from_files(&[path], dtype, device) }?;
    let raw = vb.clone();
    let eps = 1e-4f32; // `batch_norm_eps` from the VAE config.
    let n = cfg.latent_channels * LATENT_PATCH * LATENT_PATCH;
    let mean = raw.get(n, "bn.running_mean")?.reshape((1, n, 1, 1))?;
    let var = raw.get(n, "bn.running_var")?.reshape((1, n, 1, 1))?;
    let bn_std = var.affine(1.0, eps)?.sqrt()?;

    Ok(Flux2Vae {
        decoder: Decoder::new(&cfg.shape(), &zvae::NAMES, vb.pp("decoder"))?,
        post_quant_conv: conv2d(
            cfg.latent_channels,
            cfg.latent_channels,
            1,
            Conv2dConfig::default(),
            &vb.pp("post_quant_conv"),
        )?,
        bn_mean: mean,
        bn_std,
        device: device.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_ties_the_dit_grid_to_pixels() {
        let cfg = flux2_vae_config();
        // The DiT's 128 input channels ARE the 32-channel latent patchified 2x2.
        assert_eq!(cfg.latent_channels * LATENT_PATCH * LATENT_PATCH, 128);
        // ...so one DiT token covers 16 pixels a side, not 8.
        assert_eq!(VAE_STRIDE * LATENT_PATCH, 16);
    }

    /// The packed channel index is `c*4 + pi*2 + pj`. Swapping the patch axes still produces a
    /// correctly shaped image, just with every 2x2 cell transposed.
    #[test]
    fn unpatchify_places_each_patch_element() -> Result<()> {
        // One 1x1 grid, 4 channels -> 1 channel at 2x2. Values name their (pi, pj).
        let z = Tensor::from_vec_f32(vec![0.0, 1.0, 2.0, 3.0], (1usize, 4usize, 1usize, 1usize))?;
        let y = unpatchify(&z)?;
        assert_eq!(y.shape().dims(), &[1, 1, 2, 2]);
        // index c*4 + pi*2 + pj -> value at (pi, pj)
        assert_eq!(y.to_vec_f32(), vec![0.0, 1.0, 2.0, 3.0]);
        Ok(())
    }

    /// The identity scaling is load-bearing: this family normalises with batch-norm statistics,
    /// and a borrowed Flux scale/shift would desaturate every decode while still "working".
    #[test]
    fn latent_scaling_is_not_borrowed_from_flux() {
        let cfg = flux2_vae_config();
        assert!((cfg.scaling_factor - 1.0).abs() < f64::EPSILON);
        assert!(cfg.shift_factor.abs() < f64::EPSILON);
    }
}
