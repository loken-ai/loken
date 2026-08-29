//! SDXL VAE: the stock SD autoencoder (4-channel latent, 8x spatial), loaded from
//! a single-file checkpoint that uses the ORIGINAL LDM tensor names.
//!
//! There is no new network here on purpose. The SD/SDXL autoencoder is the same
//! architecture the Z-Image VAE already implements - `[128, 256, 512, 512]` blocks,
//! 2 resnets per block, group norm over 32 groups, one mid-block attention - so this
//! module contributes exactly two things the existing code lacked:
//!
//! 1. the SDXL CONFIG (4 latent channels instead of 16, the SDXL scaling factor and
//!    no shift term), and
//! 2. a NAME REWRITE from the LDM dialect the checkpoint ships to the diffusers
//!    dialect the module reads (`encoder.down.0.block.0.conv1` ->
//!    `encoder.down_blocks.0.resnets.0.conv1`), including the fact that LDM numbers
//!    the DECODER's up-blocks in the opposite order.
//!
//! The 1x1-conv-vs-linear difference in the attention projections is absorbed by the
//! VarBuilder's unit-dimension tolerance, since the two are the same operation.

use std::sync::Arc;

use crate::inference::model::zimage::vae as zvae;
use crate::inference::model::zimage::vae::{Decoder, Encoder, VaeConfig};
use crate::tensor::layer::{conv2d, Conv2d, Conv2dConfig};
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

/// Where the VAE lives inside an SDXL single-file checkpoint.
pub const CHECKPOINT_PREFIX: &str = "first_stage_model";

/// SDXL's autoencoder config. The architecture terms match the Z-Image defaults;
/// what differs is the latent width and the scaling convention.
pub fn sdxl_vae_config() -> VaeConfig {
    VaeConfig {
        in_channels: 3,
        out_channels: 3,
        // SD/SDXL keep a 4-channel latent (Flux/Z-Image use 16).
        latent_channels: 4,
        block_out_channels: vec![128, 256, 512, 512],
        layers_per_block: 2,
        // The SDXL VAE is trained with this scale and NO shift; using Flux's
        // 0.3611/0.1159 here would wash the decode out.
        scaling_factor: 0.13025,
        shift_factor: 0.0,
        norm_num_groups: 32,
    }
}

/// Number of resnets per encoder/decoder block, for the block-index arithmetic.
const BLOCKS: usize = 4;

/// Map ONE diffusers-style VAE path onto its LDM equivalent.
///
/// Written as a function over the whole path (not a table) because the mapping is
/// structural: block indices are re-derived, and the decoder's are mirrored.
fn ldm_name(diffusers: &str) -> String {
    // Strip the leading `encoder.`/`decoder.` so the rest can be matched once, then
    // put it back - both sides share the same tail grammar.
    let (side, tail) = match diffusers.split_once('.') {
        Some((s @ ("encoder" | "decoder"), t)) => (s, t),
        // Not a VAE sub-path (post_quant_conv / quant_conv): pass through.
        _ => return diffusers.to_string(),
    };
    let decoder = side == "decoder";
    let mapped = map_tail(tail, decoder);
    format!("{side}.{mapped}")
}

fn map_tail(tail: &str, decoder: bool) -> String {
    let parts: Vec<&str> = tail.split('.').collect();
    match parts.as_slice() {
        // conv_in / conv_out keep their names.
        ["conv_in", rest @ ..] | ["conv_out", rest @ ..] => {
            format!("{}.{}", parts[0], rest.join("."))
        }
        // The final norm is `conv_norm_out` in diffusers, `norm_out` in LDM.
        ["conv_norm_out", rest @ ..] => format!("norm_out.{}", rest.join(".")),
        // Resnets inside a down/up block.
        [group, idx, "resnets", j, rest @ ..]
            if *group == "down_blocks" || *group == "up_blocks" =>
        {
            let i: usize = idx.parse().unwrap_or(0);
            // LDM numbers the decoder's `up` blocks from the DEEPEST resolution
            // down, the reverse of diffusers' order.
            let li = if decoder { BLOCKS - 1 - i } else { i };
            let stage = if decoder { "up" } else { "down" };
            let name = rest.join(".");
            // diffusers calls the channel-matching 1x1 `conv_shortcut`; LDM
            // calls it `nin_shortcut`.
            let name = if name.starts_with("conv_shortcut") {
                name.replacen("conv_shortcut", "nin_shortcut", 1)
            } else {
                name
            };
            format!("{stage}.{li}.block.{j}.{name}")
        }
        // Down/up samplers.
        [group, idx, "downsamplers", _, rest @ ..] if *group == "down_blocks" => {
            format!("down.{idx}.downsample.{}", rest.join("."))
        }
        [group, idx, "upsamplers", _, rest @ ..] if *group == "up_blocks" => {
            let i: usize = idx.parse().unwrap_or(0);
            format!("up.{}.upsample.{}", BLOCKS - 1 - i, rest.join("."))
        }
        // Mid block: two resnets around one attention.
        ["mid_block", "resnets", j, rest @ ..] => {
            let n: usize = j.parse().unwrap_or(0);
            format!("mid.block_{}.{}", n + 1, rest.join("."))
        }
        ["mid_block", "attentions", _, rest @ ..] => {
            let name = rest.join(".");
            let ldm = match name.as_str() {
                n if n.starts_with("group_norm") => n.replacen("group_norm", "norm", 1),
                n if n.starts_with("to_q") => n.replacen("to_q", "q", 1),
                n if n.starts_with("to_k") => n.replacen("to_k", "k", 1),
                n if n.starts_with("to_v") => n.replacen("to_v", "v", 1),
                // diffusers stores the output projection as `to_out.0.*`.
                n if n.starts_with("to_out.0") => n.replacen("to_out.0", "proj_out", 1),
                n => n.to_string(),
            };
            format!("mid.attn_1.{ldm}")
        }
        _ => tail.to_string(),
    }
}

/// The SDXL autoencoder: the reused encoder/decoder PLUS the two 1x1 convolutions
/// the SD family keeps around its latent.
///
/// Those convs are why this is not simply `AutoEncoderKL`: Flux and Z-Image have no
/// `quant_conv`/`post_quant_conv`, so composing their AutoEncoderKL directly skipped
/// them and the decode came out as noise (round-trip correlation -0.05) even though
/// every weight loaded and the latent shape was right. A gate that only checked
/// shapes would have passed.
pub struct SdxlVae {
    encoder: Encoder,
    decoder: Decoder,
    /// 8 -> 8 over (mean, logvar), applied after the encoder.
    quant_conv: Conv2d,
    /// 4 -> 4, applied to the latent before the decoder.
    post_quant_conv: Conv2d,
    scaling_factor: f32,
    device: Device,
}

impl SdxlVae {
    /// Image `[b, 3, h, w]` in [-1, 1] -> scaled latent `[b, 4, h/8, w/8]`.
    ///
    /// The distribution's MEAN is used, not a sample: a round-trip must be
    /// reproducible, and sampling belongs to training.
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.encoder.forward(&x.to_device(&self.device)?)?;
        let moments = self.quant_conv.forward(&h)?;
        let mean = moments.chunk(2, 1)?.swap_remove(0);
        mean.affine(self.scaling_factor, 0.0)
    }

    /// Scaled latent -> image `[b, 3, h, w]`.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = z
            .to_device(&self.device)?
            .affine(1.0 / self.scaling_factor, 0.0)?;
        let z = self.post_quant_conv.forward(&z)?;
        self.decoder.forward(&z)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Load the VAE out of an SDXL single-file checkpoint.
///
/// `dtype` is the load dtype for the conv weights; the checkpoint stores them F32.
pub fn load(checkpoint: &str, device: &Device, dtype: crate::tensor::DType) -> Result<SdxlVae> {
    let cfg = sdxl_vae_config();
    let vb = unsafe { VarBuilder::from_files(&[checkpoint], dtype, device) }?;
    let vb = vb.pp(CHECKPOINT_PREFIX).with_rename(Arc::new(|full: &str| {
        // The rewrite sees the FULL path, prefix included.
        match full.split_once('.') {
            Some((CHECKPOINT_PREFIX, rest)) => {
                format!("{CHECKPOINT_PREFIX}.{}", ldm_name(rest))
            }
            _ => full.to_string(),
        }
    }));
    let one = Conv2dConfig::default();
    Ok(SdxlVae {
        encoder: Encoder::new(&cfg.shape(), &zvae::NAMES, vb.pp("encoder"))?,
        decoder: Decoder::new(&cfg.shape(), &zvae::NAMES, vb.pp("decoder"))?,
        quant_conv: conv2d(
            2 * cfg.latent_channels,
            2 * cfg.latent_channels,
            1,
            one,
            &vb.pp("quant_conv"),
        )?,
        post_quant_conv: conv2d(
            cfg.latent_channels,
            cfg.latent_channels,
            1,
            one,
            &vb.pp("post_quant_conv"),
        )?,
        scaling_factor: cfg.scaling_factor as f32,
        device: device.clone(),
    })
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// The mapping is structural, so pin the shapes of every rule - a wrong index
    /// or a missed rename fails the LOAD with "tensor not found", which is a much
    /// worse place to debug it from.
    #[test]
    fn encoder_paths_map_to_the_ldm_dialect() {
        assert_eq!(ldm_name("encoder.conv_in.weight"), "encoder.conv_in.weight");
        assert_eq!(
            ldm_name("encoder.down_blocks.0.resnets.1.conv1.weight"),
            "encoder.down.0.block.1.conv1.weight"
        );
        assert_eq!(
            ldm_name("encoder.down_blocks.2.resnets.0.conv_shortcut.weight"),
            "encoder.down.2.block.0.nin_shortcut.weight"
        );
        assert_eq!(
            ldm_name("encoder.down_blocks.1.downsamplers.0.conv.weight"),
            "encoder.down.1.downsample.conv.weight"
        );
        assert_eq!(
            ldm_name("encoder.conv_norm_out.weight"),
            "encoder.norm_out.weight"
        );
    }

    /// LDM numbers the decoder's blocks in the opposite order: index 0 in
    /// diffusers is the DEEPEST stage, which LDM calls 3.
    #[test]
    fn decoder_block_indices_are_mirrored() {
        assert_eq!(
            ldm_name("decoder.up_blocks.0.resnets.0.conv1.weight"),
            "decoder.up.3.block.0.conv1.weight"
        );
        assert_eq!(
            ldm_name("decoder.up_blocks.3.resnets.2.conv2.bias"),
            "decoder.up.0.block.2.conv2.bias"
        );
        assert_eq!(
            ldm_name("decoder.up_blocks.1.upsamplers.0.conv.weight"),
            "decoder.up.2.upsample.conv.weight"
        );
    }

    #[test]
    fn the_mid_block_maps_its_resnets_and_attention() {
        assert_eq!(
            ldm_name("decoder.mid_block.resnets.0.norm1.weight"),
            "decoder.mid.block_1.norm1.weight"
        );
        assert_eq!(
            ldm_name("decoder.mid_block.resnets.1.conv2.weight"),
            "decoder.mid.block_2.conv2.weight"
        );
        assert_eq!(
            ldm_name("encoder.mid_block.attentions.0.to_q.weight"),
            "encoder.mid.attn_1.q.weight"
        );
        assert_eq!(
            ldm_name("encoder.mid_block.attentions.0.to_out.0.bias"),
            "encoder.mid.attn_1.proj_out.bias"
        );
        assert_eq!(
            ldm_name("encoder.mid_block.attentions.0.group_norm.weight"),
            "encoder.mid.attn_1.norm.weight"
        );
    }

    /// SDXL's latent is 4-wide and its scaling has no shift term - getting either
    /// wrong decodes to a washed-out image rather than an error.
    #[test]
    fn the_config_is_sdxls_not_fluxs() {
        let c = sdxl_vae_config();
        assert_eq!(c.latent_channels, 4);
        assert!((c.scaling_factor - 0.13025).abs() < 1e-9);
        assert_eq!(c.shift_factor, 0.0);
    }

    /// THE gate for this module: load the VAE out of a real SDXL checkpoint and
    /// round-trip an image through it. A wrong name mapping fails the load; a wrong
    /// config (latent width, scaling) decodes to noise or a washed-out frame. Only a
    /// reconstruction that correlates with the source proves both are right.
    ///
    /// Ignored by default: it needs the checkpoint on disk.
    #[test]
    #[ignore = "needs an SDXL checkpoint under the configured models dir"]
    fn round_trips_a_real_image_through_a_real_checkpoint() {
        let dir = crate::config::Config::load_test().get_hf_models_dir();
        let ckpt = std::fs::read_dir(dir.join("raymnants"))
            .expect("raymnants dir")
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .expect("an SDXL .safetensors");
        let src = std::env::var("SDXL_VAE_TEST_IMAGE")
            .expect("set SDXL_VAE_TEST_IMAGE to a PNG/JPEG to round-trip");

        // Fastest probed GPU, CPU when there is none - never a fixed device.
        let dev = crate::inference::place::vram_manager::probe(0)
            .into_iter()
            .next()
            .map(|(_, _, d)| d)
            .unwrap_or(Device::Cpu);
        let vae = load(ckpt.to_str().unwrap(), &dev, crate::tensor::DType::F32)
            .expect("VAE load (a name-mapping gap surfaces here)");

        let img = image::open(&src).expect("test image").to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        // [1, 3, h, w] in [-1, 1], the autoencoder's input convention.
        let mut v = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let p = img.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    v[(c * h + y) * w + x] = p[c] as f32 / 127.5 - 1.0;
                }
            }
        }
        let x = Tensor::from_vec_f32(v.clone(), (1, 3, h, w)).expect("input tensor");
        let z = vae.encode(&x).expect("encode");
        let zd = z.dims().to_vec();
        assert_eq!(zd[1], 4, "SDXL latents are 4-wide, got {zd:?}");
        assert_eq!(
            (zd[2], zd[3]),
            (h / 8, w / 8),
            "8x spatial downsample, got {zd:?}"
        );

        let out = vae.decode(&z).expect("decode");
        let r = out.to_device(&Device::Cpu).unwrap().to_vec_f32();
        assert!(
            r.iter().all(|q| q.is_finite()),
            "decode produced non-finite values"
        );

        // Pearson correlation against the source: a working autoencoder reconstructs
        // the frame, so this is high. Noise or a channel/scale error collapses it.
        let n = v.len().min(r.len()) as f32;
        let (mx, my) = (
            v.iter().sum::<f32>() / n,
            r.iter().take(v.len()).sum::<f32>() / n,
        );
        let (mut sxy, mut sxx, mut syy) = (0f32, 0f32, 0f32);
        for (a, b) in v.iter().zip(r.iter()) {
            let (da, db) = (a - mx, b - my);
            sxy += da * db;
            sxx += da * da;
            syy += db * db;
        }
        let corr = sxy / (sxx.sqrt() * syy.sqrt() + 1e-12);
        // Write the reconstruction next to the source: a number can look fine while
        // the frame is subtly wrong (a channel swap correlates well), so the
        // round-trip is meant to be LOOKED at too.
        let recon = std::path::Path::new(&src).with_extension("recon.png");
        let mut out_img = image::RgbImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let px = [0usize, 1, 2]
                    .map(|c| (((r[(c * h + y) * w + x] + 1.0) * 127.5).clamp(0.0, 255.0)) as u8);
                out_img.put_pixel(x as u32, y as u32, image::Rgb(px));
            }
        }
        out_img.save(&recon).expect("write reconstruction");
        println!(
            "SDXL VAE round-trip: latent {zd:?}, corr {corr:.4}, wrote {}",
            recon.display()
        );
        assert!(
            corr > 0.9,
            "reconstruction correlation {corr:.4} is too low"
        );
    }
}
