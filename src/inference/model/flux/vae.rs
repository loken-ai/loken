//! Flux VAE on the native tensor substrate.
//! Conv-based encoder/decoder with resnet + spatial-attention blocks; the
//! generation hot path is `decode` (latent -> image), `encode` serves img2img.
//!
//! Runs F32 end-to-end (GPU convs go through im2col + cuBLAS, GroupNorm /
//! upsample / pad are dedicated kernels) - no F16 promote dance needed.
//! The public `encode`/`decode` boundary speaks the facade `Tensor` until the
//! flip; inputs are moved to the VAE's own device, outputs return on the
//! caller's input device.

use crate::inference::model::vae_blocks::{
    AttnNaming, Decoder, Encoder, Naming, ProjectionKind, Shape,
};
use crate::tensor::DType;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

/// This family's autoencoder: the shape the shared encoder and decoder are built from, plus
/// the affine that takes a latent into the range the transformer was trained on.
#[derive(Debug, Clone)]
pub struct Config {
    pub shape: Shape,
    pub scale_factor: f64,
    pub shift_factor: f64,
}

impl Config {
    /// Both published sizes ship the same autoencoder.
    pub fn schnell() -> Self {
        Self {
            shape: Shape {
                // A base width doubled twice and then held: 128, 256, 512, 512.
                stages: vec![128, 256, 512, 512],
                stem: 128,
                blocks_per_stage: 2,
                groups: 32,
                image_channels: 3,
                latent_channels: 16,
            },
            scale_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }

    pub fn dev() -> Self {
        Self::schnell()
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
    }
}

/// Single-head attention over the flattened spatial grid (`seq = h*w`,
/// `dim = c`). F32 throughout - no overflow risk. Delegates to the shared
/// [`native_acestep_ops::sdpa`] (same q.kᵀ.scale->softmax->.v math; its query
/// tiling is bit-exact and caps peak memory on large grids).
/// This lineage stores the middle block's projections as 1x1 convolutions, under a plain
/// `norm`, and normalises over thirty-two groups whatever the width.
const ATTN: AttnNaming = AttnNaming {
    kind: ProjectionKind::Conv1x1,
    norm: "norm",
    q: "q",
    k: "k",
    v: "v",
    out: "proj_out",
};

/// This lineage's naming: a stage's residual blocks under `block`, its resampler beside them,
/// the middle's three parts spelled out, and the decoder's stages keeping the encoder's
/// numbering so the walk runs backwards.
const NAMES: Naming = Naming {
    down: "down",
    up: "up",
    resnets: "block",
    downsample: "downsample",
    upsample: "upsample",
    shortcut: "nin_shortcut",
    mid_first: "mid.block_1",
    mid_attn: "mid.attn_1",
    mid_second: "mid.block_2",
    norm_out: "norm_out",
    attn: ATTN,
    up_keeps_encoder_numbering: true,
};

/// Latent regularizer: split `[b, 2z, h, w]` into mean/logvar and sample
/// `mean + exp(logvar/2) * eps` (Box-Muller normals, host-generated).
#[derive(Debug, Clone)]
struct DiagonalGaussian {
    sample: bool,
    chunk_dim: usize,
}

impl DiagonalGaussian {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(2, self.chunk_dim)?;
        if self.sample {
            let std = chunks[1].affine(0.5, 0.0)?.exp()?;
            let noise = std.randn_like()?;
            chunks[0].add(&std.mul(&noise)?)
        } else {
            Ok(chunks[0].clone())
        }
    }
}

pub struct AutoEncoder {
    encoder: Encoder,
    decoder: Decoder,
    reg: DiagonalGaussian,
    shift_factor: f64,
    scale_factor: f64,
    device: Device,
}

impl AutoEncoder {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        Ok(Self {
            encoder: Encoder::new(&cfg.shape(), &NAMES, vb.pp("encoder"))?,
            decoder: Decoder::new(&cfg.shape(), &NAMES, vb.pp("decoder"))?,
            reg: DiagonalGaussian {
                sample: true,
                chunk_dim: 1,
            },
            scale_factor: cfg.scale_factor,
            shift_factor: cfg.shift_factor,
            device,
        })
    }

    /// Image `[b, 3, h, w]` -> scaled latent. Facade boundary: input moves to
    /// the VAE's device, output returns on the input's device.
    pub fn encode(
        &self,
        xs: &crate::tensor::Tensor,
    ) -> crate::tensor::Result<crate::tensor::Tensor> {
        let x = xs.to_dtype(DType::F32)?.to_device(&self.device)?;
        let z = self
            .encoder
            .forward(&x)
            .and_then(|h| self.reg.forward(&h))
            .and_then(|z| {
                z.affine(
                    self.scale_factor as f32,
                    (-self.shift_factor * self.scale_factor) as f32,
                )
            })?;
        z.to_device(&xs.device())
    }

    /// Scaled latent `[b, z, h/8, w/8]` -> image. Same boundary contract as
    /// [`Self::encode`].
    pub fn decode(
        &self,
        xs: &crate::tensor::Tensor,
    ) -> crate::tensor::Result<crate::tensor::Tensor> {
        let x = xs.to_dtype(DType::F32)?.to_device(&self.device)?;
        let img = x
            .affine((1.0 / self.scale_factor) as f32, self.shift_factor as f32)
            .and_then(|x| self.decoder.forward(&x))?;
        img.to_device(&xs.device())
    }

    /// Tiled GPU decode for large latents: split the H/W grid into bands with `ov` latent-pixel
    /// overlap, decode each tile through the conv decoder (bounded peak VRAM), keep exactly its
    /// `[b_i,b_{i+1})` band (the overlap supplies seamless conv context), stitch by `cat`. Lets
    /// >512² decode on GPU instead of the ~55x slower whole-image CPU fallback.
    pub fn decode_tiled(
        &self,
        xs: &crate::tensor::Tensor,
        max_tile: usize,
        ov: usize,
    ) -> crate::tensor::Result<crate::tensor::Tensor> {
        let x = xs.to_dtype(DType::F32)?.to_device(&self.device)?;
        let x = x.affine((1.0 / self.scale_factor) as f32, self.shift_factor as f32)?;
        let d = x.dims();
        let (hl, wl) = (d[2], d[3]);
        let bounds = |len: usize| -> Vec<usize> {
            let n = len.div_ceil(max_tile).max(1);
            (0..=n).map(|i| (i * len) / n).collect()
        };
        let (hb, wb) = (bounds(hl), bounds(wl));
        let mut rows: Vec<Tensor> = Vec::new();
        for hi in 0..hb.len() - 1 {
            let (h0, h1) = (hb[hi], hb[hi + 1]);
            let (hs, he) = (h0.saturating_sub(ov), (h1 + ov).min(hl));
            let mut cols: Vec<Tensor> = Vec::new();
            for wi in 0..wb.len() - 1 {
                let (w0, w1) = (wb[wi], wb[wi + 1]);
                let (ws, we) = (w0.saturating_sub(ov), (w1 + ov).min(wl));
                let tin = x
                    .narrow(2, hs, he - hs)
                    .and_then(|t| t.narrow(3, ws, we - ws))?;
                let dec = self.decoder.forward(&tin)?;
                // keep exactly the [b_i,b_{i+1}) band (image space = latent.8)
                let dec = dec
                    .narrow(2, (h0 - hs) * 8, (h1 - h0) * 8)
                    .and_then(|t| t.narrow(3, (w0 - ws) * 8, (w1 - w0) * 8))?;
                cols.push(dec);
            }
            let refs: Vec<&Tensor> = cols.iter().collect();
            rows.push(Tensor::cat(&refs, 3)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let img = Tensor::cat(&refs, 2)?;
        img.to_device(&xs.device())
    }
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;
    use crate::tensor;

    /// The decode must produce a VARYING image at every resolution.
    ///
    /// Flux started returning pure black (`mean 0.00, std 0.00`) with no error anywhere
    /// in the pipeline - T5, denoise and VAE all reported success and timings. The
    /// evidence pointed at a resolution boundary: the same checkpoint rendered a real
    /// image at 1024 and black at 512. This isolates the VAE from the transformer, so a
    /// failure here says the decoder, and a pass says look upstream.
    ///
    /// A constant output is the whole signal: whatever the latent, the decoder must not
    /// collapse it to one value.
    #[test]
    #[ignore = "needs the FLUX schnell ae.safetensors under the configured models dir"]
    fn decode_varies_at_every_resolution() {
        let hub = crate::config::Config::load_test()
            .get_hf_models_dir()
            .join("hub");
        let Some(ae_file) = glob_one(&hub, "ae.safetensors") else {
            println!("no ae.safetensors under {}; skipping", hub.display());
            return;
        };
        let dev = crate::inference::place::vram_manager::probe(0)
            .into_iter()
            .next()
            .map(|(_, _, d)| d)
            .unwrap_or(tensor::Device::Cpu);
        let vb = unsafe { tensor::VarBuilder::from_files(&[&ae_file], tensor::DType::F32, &dev) }
            .expect("vae weights");
        let ae = AutoEncoder::new(&Config::schnell(), vb).expect("vae");

        // 64 latent = 512 px, 128 latent = 1024 px: the two sides of the boundary.
        for side in [64usize, 128] {
            let n = 16 * side * side;
            let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.017).sin() * 0.9).collect();
            let lat = tensor::Tensor::from_vec_f32(v, vec![1, 16, side, side])
                .expect("latent")
                .to_device(&dev)
                .expect("latent to device");
            let out = ae.decode(&lat).expect("decode");
            let host = out
                .to_device(&tensor::Device::Cpu)
                .expect("to cpu")
                .to_vec_f32();
            let mean = host.iter().sum::<f32>() / host.len() as f32;
            let var = host.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / host.len() as f32;
            println!(
                "latent {side}x{side} ({} px): mean {mean:.4} std {:.4}",
                side * 8,
                var.sqrt()
            );
            assert!(
                var.sqrt() > 1e-3,
                "latent {side}x{side}: decode returned a CONSTANT image (std {:.6}) - this is \
                 the black-image failure, and it is in the VAE",
                var.sqrt()
            );
        }
    }

    pub(super) fn glob_one(root: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
        let mut stack = vec![root.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).ok()?.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.file_name().is_some_and(|f| f == name) {
                    return Some(p);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod decode_reference {
    //! The decoder's answer on a fixed latent, on the real weights.
    //!
    //! Needs the checkpoint, so it does not run in the ordinary suite. It exists to be run
    //! either side of a change to how this VAE is BUILT: a rebuild that reads a different
    //! tensor into a different slot moves these numbers long before it produces a picture
    //! anyone would question.
    use super::tests::glob_one;
    use super::*;
    use crate::tensor;

    #[test]
    #[ignore = "needs the FLUX schnell ae.safetensors under the configured models dir"]
    fn the_decoder_answers_what_it_answered() {
        let hub = crate::config::Config::load_test()
            .get_hf_models_dir()
            .join("hub");
        let Some(ae_file) = glob_one(&hub, "ae.safetensors") else {
            println!(
                "no ae.safetensors under {}; this run judges nothing",
                hub.display()
            );
            return;
        };
        let dev = tensor::Device::Cpu;
        let vb = unsafe { tensor::VarBuilder::from_files(&[&ae_file], tensor::DType::F32, &dev) }
            .expect("vae weights");
        let ae = AutoEncoder::new(&Config::schnell(), vb).expect("vae");

        let cfg = Config::schnell();
        let (side, n) = (16usize, 16usize * 16 * 16);
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            data.push(((seed >> 40) as f32 / 8192.0) - 1.0);
        }
        let _ = cfg;
        let lat = tensor::Tensor::from_vec(data, (1, 16, side, side), &dev).expect("latent");
        let out = ae.decode(&lat).expect("decode");
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mean = v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len() as f64;
        let energy = v.iter().map(|x| f64::from(x * x)).sum::<f64>() / v.len() as f64;
        let lo = v.iter().fold(f32::INFINITY, |m, x| m.min(*x));
        let hi = v.iter().fold(f32::NEG_INFINITY, |m, x| m.max(*x));
        println!(
            "dims={:?} mean={mean:.9} energy={energy:.9} min={lo:.6} max={hi:.6}",
            out.dims()
        );
        assert_eq!(out.dims(), &[1, 3, 128, 128]);
        // Taken from this decoder before its middle attention became the shared one. The
        // Z-Image decoder answers the same to every digit - it is this autoencoder, under
        // another checkpoint's names.
        assert!((mean - 0.537_436_430).abs() < 1e-6, "mean {mean}");
        assert!((energy - 0.608_883_922).abs() < 1e-6, "energy {energy}");
        assert!((lo - (-2.189_235)).abs() < 1e-4, "min {lo}");
        assert!((hi - 2.280_594).abs() < 1e-4, "max {hi}");
    }
}
