//! Z-Image VAE (AutoEncoderKL, diffusers format) on the NATIVE tensor
//! substrate. Conv encoder/decoder with resnet +
//! Linear-based spatial attention; `decode` runs in generation, `encode`
//! serves img2img. F32 end-to-end; the public `encode`/`decode` boundary
//! speaks the facade `Tensor` until the flip (inputs move to the VAE's own
//! device, outputs return on the caller's input device).

// Re-exported: FLUX.2 and SDXL load this same autoencoder through this module.
use crate::inference::model::vae_blocks::{AttnNaming, Naming, ProjectionKind, Shape};
pub use crate::inference::model::vae_blocks::{Decoder, Encoder};
use crate::tensor::layer::{conv2d, group_norm, linear, Conv2d, Conv2dConfig, GroupNorm, Linear};
use crate::tensor::DType;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VaeConfig {
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    #[serde(default = "default_latent_channels")]
    pub latent_channels: usize,
    #[serde(default = "default_block_out_channels")]
    pub block_out_channels: Vec<usize>,
    #[serde(default = "default_layers_per_block")]
    pub layers_per_block: usize,
    #[serde(default = "default_scaling_factor")]
    pub scaling_factor: f64,
    #[serde(default = "default_shift_factor")]
    pub shift_factor: f64,
    #[serde(default = "default_norm_num_groups")]
    pub norm_num_groups: usize,
}

// What the published Z-Image autoencoder states when its config file does not.
crate::serde_defaults! {
    default_in_channels: usize = 3;
    default_out_channels: usize = 3;
    default_latent_channels: usize = 16;
    default_block_out_channels: Vec<usize> = vec![128, 256, 512, 512];
    default_layers_per_block: usize = 2;
    default_scaling_factor: f64 = 0.3611;
    default_shift_factor: f64 = 0.1159;
    default_norm_num_groups: usize = 32;
}

/// The published Z-Image autoencoder, which is what every default above states.
///
/// Written through those functions rather than beside them: a second list of the same values
/// is a second thing to keep right, and if the two disagreed a config file that omitted a
/// field would quietly build something other than the preset.
impl Default for VaeConfig {
    fn default() -> Self {
        Self {
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            latent_channels: default_latent_channels(),
            block_out_channels: default_block_out_channels(),
            layers_per_block: default_layers_per_block(),
            scaling_factor: default_scaling_factor(),
            shift_factor: default_shift_factor(),
            norm_num_groups: default_norm_num_groups(),
        }
    }
}

impl VaeConfig {
    pub fn z_image() -> Self {
        Self::default()
    }
}

/// q/k/v: `[b, seq, c]`. At 1024² the latent mid-block has seq = 128x128 =
/// 16384, so the full seq² score matrix is ~1 GB+ and (with intermediates)
/// OOMs the GPU. Delegates to the shared query-tiled
/// [`native_acestep_ops::sdpa`] - identical q.kᵀ.scale->softmax->.v math; its
/// per-row-softmax query tiling (tile 1024, vs the old local 2048 chunk) is
/// bit-exact and caps the materialized score matrix at `[b, TILE, seq]` so
/// the GPU VAE handles 1024² instead of falling back to the slow CPU decode.
/// This lineage stores the middle block's projections as linears, under a `group_norm`, and
/// takes the group count from the configuration.
const ATTN: AttnNaming = AttnNaming {
    kind: ProjectionKind::Linear,
    norm: "group_norm",
    q: "to_q",
    k: "to_k",
    v: "to_v",
    out: "to_out.0",
};

/// This lineage's naming: a stage's residual blocks under `resnets`, its resampler in a list of
/// its own, the middle's parts numbered inside one `mid_block`, and the decoder's stages
/// renumbered from the deepest.
pub const NAMES: Naming = Naming {
    down: "down_blocks",
    up: "up_blocks",
    resnets: "resnets",
    downsample: "downsamplers.0",
    upsample: "upsamplers.0",
    shortcut: "conv_shortcut",
    mid_first: "mid_block.resnets.0",
    mid_attn: "mid_block.attentions.0",
    mid_second: "mid_block.resnets.1",
    norm_out: "conv_norm_out",
    attn: ATTN,
    up_keeps_encoder_numbering: false,
};

impl VaeConfig {
    /// The channel schedule this configuration describes. The first stage takes what the input
    /// convolution produced, so the stem is the first width itself.
    pub fn shape(&self) -> Shape {
        Shape {
            stages: self.block_out_channels.clone(),
            stem: self.block_out_channels[0],
            blocks_per_stage: self.layers_per_block,
            groups: self.norm_num_groups,
            image_channels: self.in_channels,
            latent_channels: self.latent_channels,
        }
    }
}

/// Latent regularizer: split `[b, 2z, h, w]` into mean/logvar and sample
/// `mean + exp(logvar/2) * eps`.
#[derive(Debug, Clone)]
struct DiagonalGaussian {
    sample: bool,
}

impl DiagonalGaussian {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(2, 1)?;
        let mean = &chunks[0];
        let logvar = &chunks[1];
        if self.sample {
            let std = logvar.affine(0.5, 0.0)?.exp()?;
            mean.add(&std.mul(&std.randn_like()?)?)
        } else {
            Ok(mean.clone())
        }
    }
}

/// Z-Image VAE (AutoEncoderKL, diffusers format).
pub struct AutoEncoderKL {
    encoder: Encoder,
    decoder: Decoder,
    reg: DiagonalGaussian,
    scale_factor: f64,
    shift_factor: f64,
    device: Device,
}

impl AutoEncoderKL {
    pub fn new(cfg: &VaeConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        Ok(Self {
            encoder: Encoder::new(&cfg.shape(), &NAMES, vb.pp("encoder"))?,
            decoder: Decoder::new(&cfg.shape(), &NAMES, vb.pp("decoder"))?,
            reg: DiagonalGaussian { sample: true },
            scale_factor: cfg.scaling_factor,
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

    /// Scaled latent -> image. Same boundary contract as [`Self::encode`].
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

    /// Tiled decode: bound the peak instead of moving the work off the GPU.
    ///
    /// A VAE decode's cost is its widest full-resolution feature map, and at 1024^2
    /// that peaked at 5.9 GB here - more than a card holding another model could spare,
    /// so the decode fell to the CPU and took 116 s of a 176 s request.
    ///
    /// Splitting the DECODER across devices would be the wrong answer even though it
    /// sounds like the same problem as the transformer: a conv decoder has no natural
    /// seam, so a layer-wise split ships the whole feature map over PCIe at every
    /// boundary - gigabytes per layer at this resolution, and every layer is a
    /// synchronisation point. Decoding is spatially LOCAL instead: a tile needs only its
    /// own neighbourhood, so cutting the image into bands bounds the peak on ONE device
    /// with no cross-device traffic at all. That is why this tiles rather than spreads.
    ///
    /// `ov` latent pixels of overlap give the convolutions their context; only the
    /// interior band of each tile is kept, so the seams carry no boundary artefact.
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
                // Keep exactly the interior band; image space is latent x8.
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

    pub fn scale_factor(&self) -> f64 {
        self.scale_factor
    }

    pub fn shift_factor(&self) -> f64 {
        self.shift_factor
    }
}

#[cfg(test)]
mod tiled_decode_tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// Tiling is only a legitimate substitute for the whole decode if the seams and the
    /// narrowed attention cost nothing visible. The decoder's mid-block attends over the
    /// WHOLE latent map, so a tile attends only to itself - that is a real approximation,
    /// not a rounding difference, and the only way to know whether it matters is to
    /// measure it against the untiled decode on the actual weights.
    #[test]
    #[ignore = "needs the Z-Image VAE weights; runs on CPU"]
    fn tiled_decode_matches_the_whole_decode() {
        // Resolve through the same hub cache the engine loads from, and LOCALLY only -
        // a test must never reach for the network to decide whether it can run.
        let cache = hf_hub::Cache::default().model("Tongyi-MAI/Z-Image-Turbo".to_string());
        let Some(path) = cache.get("vae/diffusion_pytorch_model.safetensors") else {
            println!("Z-Image VAE weights not in the hub cache; skipping");
            return;
        };
        // Run on a GPU when one is free: the tiling does its narrow/cat/conv work on the
        // decode device, and a CPU-only check would leave that path unexercised.
        // Take a GPU only when one is comfortably free. A test must never compete for
        // VRAM with whatever the machine is actually serving; falling back to the CPU
        // costs this test 40 s and costs the user nothing.
        let dev = crate::inference::place::vram_manager::probe(0)
            .into_iter()
            .find(|(_, free, _)| *free >= 2_000_000_000)
            .map(|(_, _, d)| d)
            .unwrap_or(Device::Cpu);
        println!("decoding on {:?}", dev.location());
        let cfg = VaeConfig::z_image();
        let vb = unsafe {
            crate::tensor::VarBuilder::from_files(
                &[path.to_str().unwrap()],
                crate::tensor::DType::F32,
                &dev,
            )
        }
        .expect("varbuilder");
        let vae = AutoEncoderKL::new(&cfg, vb).expect("load");

        // 48x48 latent = 384^2 image: small enough to decode twice on the CPU, large
        // enough that a 16-px tile is a genuine 3x3 split rather than a no-op.
        let (hl, wl) = (48usize, 48);
        let n = cfg.latent_channels * hl * wl;
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            data.push(((seed >> 40) as f32 / 8192.0) - 1.0);
        }
        let lat = crate::tensor::Tensor::from_vec(
            data,
            (1, cfg.latent_channels, hl, wl),
            &crate::tensor::Device::Cpu,
        )
        .expect("latent");

        let whole = vae.decode(&lat).expect("whole decode");
        let tiled = vae.decode_tiled(&lat, 16, 8).expect("tiled decode");
        assert_eq!(
            whole.dims(),
            tiled.dims(),
            "tiling must not change the shape"
        );

        let a = whole.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = tiled.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let num: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| f64::from((x - y) * (x - y)))
            .sum();
        let den: f64 = a.iter().map(|x| f64::from(x * x)).sum::<f64>().max(1e-20);
        let rel = (num / den).sqrt();
        let max = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        // Output is in [-1, 1], so a max deviation of 0.02 is ~2.5 of 255 levels.
        println!("tiled vs whole: rel-RMS {rel:.3e}, max abs {max:.4}");
        assert!(
            rel < 0.02,
            "tiling changes the image too much (rel-RMS {rel:.3e})"
        );
    }

    /// The comparison that should have gated the tiling in the first place.
    ///
    /// The earlier test decoded a RANDOM latent and reported rel-RMS 8.7e-3, which is
    /// close to the one input for which tiling cannot fail: the decoder's mid-block
    /// attends over the whole map, and that global attention is what holds colour and
    /// brightness consistent ACROSS an image. Noise has no such structure to lose. A
    /// real latent does, so this encodes an actual photograph and looks for the failure
    /// a global RMS hides - each tile drifting on its OWN, which reads as blocks.
    ///
    /// Point ZIMG_REAL at an image file.
    #[test]
    #[ignore = "needs the Z-Image VAE weights and a real image; runs on CPU"]
    fn tiling_a_real_latent_does_not_shift_tiles_against_each_other() {
        let Ok(img_path) = std::env::var("ZIMG_REAL") else {
            println!("ZIMG_REAL not set; skipping");
            return;
        };
        let cache = hf_hub::Cache::default().model("Tongyi-MAI/Z-Image-Turbo".to_string());
        let Some(path) = cache.get("vae/diffusion_pytorch_model.safetensors") else {
            println!("Z-Image VAE weights not in the hub cache; skipping");
            return;
        };
        let dev = Device::Cpu;
        let cfg = VaeConfig::z_image();
        let vb = unsafe {
            crate::tensor::VarBuilder::from_files(
                &[path.to_str().unwrap()],
                crate::tensor::DType::F32,
                &dev,
            )
        }
        .expect("varbuilder");
        let vae = AutoEncoderKL::new(&cfg, vb).expect("load");

        // 512^2 keeps a CPU encode+decode+decode affordable while still giving the
        // 64x64 latent several tiles to disagree over.
        const SIDE: usize = 512;
        let rgb = image::open(&img_path)
            .expect("open ZIMG_REAL")
            .resize_exact(
                SIDE as u32,
                SIDE as u32,
                image::imageops::FilterType::Lanczos3,
            )
            .to_rgb8();
        let plane = SIDE * SIDE;
        let mut chw = vec![0f32; 3 * plane];
        for i in 0..plane {
            for c in 0..3 {
                chw[c * plane + i] = f32::from(rgb.as_raw()[3 * i + c]) / 127.5 - 1.0;
            }
        }
        let x =
            crate::tensor::Tensor::from_vec(chw, (1, 3, SIDE, SIDE), &crate::tensor::Device::Cpu)
                .expect("input");
        let lat = vae.encode(&x).expect("encode");

        let whole = vae.decode(&lat).expect("whole decode");
        let a = whole.flatten_all().unwrap().to_vec1::<f32>().unwrap();

        // Sweep the split RATIO, which is what governs the drift rather than the pixel
        // count. Production tiles a 128-px latent at 64 - a 2x2 split - while the first
        // measurement used a 3x3 split of a much smaller latent. Every GroupNorm in the
        // up-path takes its statistics over the spatial extent it is handed, so fewer,
        // larger tiles have statistics closer to the whole image's. The question this
        // answers is whether drift is inherent to tiling or only to cutting it fine.
        let latent_side = SIDE / 8;
        let mut coarse = f64::NAN;
        for div in [4usize, 3, 2] {
            let tile = latent_side / div;
            let tiled = vae.decode_tiled(&lat, tile, 8).expect("tiled decode");
            let b = tiled.flatten_all().unwrap().to_vec1::<f32>().unwrap();

            let num: f64 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| f64::from((x - y) * (x - y)))
                .sum();
            let den: f64 = a.iter().map(|x| f64::from(x * x)).sum::<f64>().max(1e-20);
            let rel = (num / den).sqrt();

            // The failure that matters is not the average error but a per-tile BIAS: a
            // block uniformly darker or bluer than its neighbour is obvious to the eye
            // at an RMS the average calls small.
            let tile_px = tile * 8;
            let tiles = (SIDE / tile_px).max(1);
            let mut means = Vec::new();
            for ty in 0..tiles {
                for tx in 0..tiles {
                    let (mut sum, mut n) = (0f64, 0u64);
                    for c in 0..3 {
                        for y in ty * tile_px..(ty + 1) * tile_px {
                            for x in tx * tile_px..(tx + 1) * tile_px {
                                let idx = c * plane + y * SIDE + x;
                                sum += f64::from(a[idx] - b[idx]);
                                n += 1;
                            }
                        }
                    }
                    means.push(sum / n as f64);
                }
            }
            let lo = means.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let spread = hi - lo;
            if div == 2 {
                coarse = spread;
            }
            println!(
                "latent {latent_side} tiled at {tile} ({tiles}x{tiles}): rel-RMS {rel:.3e}, \
                 per-tile bias spread {spread:.4}"
            );
        }
        // MEASURED, and the reason the bound is on the COARSE split only:
        //
        //   4x4 (tile 16)  spread 0.0338
        //   3x3 (tile 21)  spread 0.0185
        //   2x2 (tile 32)  spread 0.0067
        //
        // Drift is not inherent to tiling, it is inherent to cutting it FINE. Every
        // GroupNorm in the up-path takes its statistics over the extent it is handed,
        // so the more tiles, the further each one's statistics sit from the whole
        // image's - and the bias is per-tile, which is what reads as blocking.
        //
        // Output is [-1,1], so 0.02 of spread is ~2.5 of 255 levels between neighbours,
        // where blocking starts to show on flat areas like skin or sky. A 2x2 split is
        // an order of magnitude inside that; the finer splits are recorded above rather
        // than asserted, because they are the regime this must never enter.
        assert!(
            coarse < 0.02,
            "a 2x2 split drifts by {coarse:.4} - tiling is no longer safe at any ratio"
        );
    }

    /// The band arithmetic must partition the axis exactly: no gap, no overlap in what
    /// is KEPT, whatever the length and tile size.
    #[test]
    fn the_kept_bands_partition_the_axis() {
        for len in [7usize, 16, 31, 48, 64, 128] {
            for max_tile in [1usize, 3, 16, 32, 200] {
                let n = len.div_ceil(max_tile).max(1);
                let b: Vec<usize> = (0..=n).map(|i| (i * len) / n).collect();
                assert_eq!(b[0], 0);
                assert_eq!(*b.last().unwrap(), len);
                for w in b.windows(2) {
                    assert!(w[1] >= w[0], "bands must not go backwards");
                    assert!(w[1] - w[0] <= max_tile, "a band exceeds the tile budget");
                }
            }
        }
    }
}

#[cfg(test)]
mod decode_reference {
    //! The decoder's answer on a fixed latent, on the real weights.
    //!
    //! Needs the checkpoint, so it does not run in the ordinary suite. It exists to be run
    //! either side of a change to how this VAE is BUILT: the numbers below were taken from the
    //! implementation before that change, and a rebuild that reads a different tensor into a
    //! different slot moves them long before it produces a picture anyone would question.
    use super::*;

    fn fixed_latent(cfg: &VaeConfig, hl: usize, wl: usize) -> crate::tensor::Tensor {
        let n = cfg.latent_channels * hl * wl;
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            data.push(((seed >> 40) as f32 / 8192.0) - 1.0);
        }
        crate::tensor::Tensor::from_vec(
            data,
            (1, cfg.latent_channels, hl, wl),
            &crate::tensor::Device::Cpu,
        )
        .expect("latent")
    }

    #[test]
    #[ignore = "needs the Z-Image VAE weights; runs on CPU"]
    fn the_decoder_answers_what_it_answered() {
        let cache = hf_hub::Cache::default().model("Tongyi-MAI/Z-Image-Turbo".to_string());
        let Some(path) = cache.get("vae/diffusion_pytorch_model.safetensors") else {
            println!("Z-Image VAE weights not in the hub cache; this run judges nothing");
            return;
        };
        let cfg = VaeConfig::z_image();
        let vb = unsafe {
            crate::tensor::VarBuilder::from_files(
                &[path.to_str().unwrap()],
                crate::tensor::DType::F32,
                &Device::Cpu,
            )
        }
        .expect("varbuilder");
        let vae = AutoEncoderKL::new(&cfg, vb).expect("load");

        let out = vae.decode(&fixed_latent(&cfg, 16, 16)).expect("decode");
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
        // Taken from this decoder before the encoder and decoder were rebuilt from a shared
        // description. A tensor read into the wrong slot moves these long before it produces
        // a picture anyone would question.
        assert!((mean - 0.537_436_430).abs() < 1e-6, "mean {mean}");
        assert!((energy - 0.608_883_922).abs() < 1e-6, "energy {energy}");
        assert!((lo - (-2.189_235)).abs() < 1e-4, "min {lo}");
        assert!((hi - 2.280_594).abs() < 1e-4, "max {hi}");
    }
}
