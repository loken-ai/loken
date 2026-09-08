//! SDXL end to end: tokenize -> both CLIP towers -> UNet under CFG -> VAE decode.
//!
//! This is the piece that turns the four stages into an image. The sampler
//! arithmetic runs host-side on the latent (4 x h/8 x w/8 floats - 65 k values even
//! at 1024^2, so the transfers are noise next to a UNet forward), which keeps the
//! schedule readable and matches how the Qwen-Image engine drives its loop.

use crate::inference::model::sdxl::sampling::{
    cfg, denoised, euler_step, exponential_sigmas, input_scale, karras_sigmas, DpmPP2M,
    SamplerKind, Schedule, SchedulerKind,
};
use crate::inference::model::sdxl::text::SdxlTextEncoders;
use crate::inference::model::sdxl::unet::{SdxlUnet, LATENT_CHANNELS};
use crate::inference::model::sdxl::vae::SdxlVae;
use crate::tensor::{DType, Device, Result, Tensor};

/// SDXL's native resolution; the micro-conditioning is stated in these terms.
pub const NATIVE_SIDE: usize = 1024;

/// Everything a generation needs, resident.
pub struct SdxlPipeline {
    unet: SdxlUnet,
    text: SdxlTextEncoders,
    vae: SdxlVae,
    tokenizer: tokenizers::Tokenizer,
    schedule: Schedule,
    /// The structural conditioner, loaded on first use and kept.
    ///
    /// Answers a different question from the text conditioning: it says
    /// WHO, a ControlNet says WHERE. It is what constrains geometry, which no amount
    /// of prompt or guidance can - the reason "too many limbs" survives both.
    controlnet: Option<crate::inference::model::sdxl::controlnet::SdxlControlNet>,
    /// The pose/edge image for this render, already on the UNet's device, with how
    /// hard it pulls. `None` = an ordinary render, and the ControlNet is not run.
    control: Option<(Tensor, f32)>,
    /// Where the conditioner lives. Its own placement, because it is its own per-step
    /// component and 2.5 GB of it.
    control_device: Device,
    device: Device,
    /// Solver and sigma curve for the next render (see `set_sampling`).
    sampler: SamplerKind,
    scheduler: SchedulerKind,
    /// The adapter set currently attached, in request order. Requests carry their whole
    /// adapter set, so this is what makes a repeat request free - see `set_loras`.
    attached: Vec<(String, f32)>,
    /// Projections matched by `attached`, so a reused set still reports what it did.
    attached_matched: usize,
}

/// Architecture of this family, used to DERIVE what a render needs rather than to
/// state it: the VAE reduces space by 8 and its last decode stage carries 128
/// channels at full output resolution; the transformer blocks are built at a
/// feed-forward ratio of 4; the text towers see a 77-token window.
const VAE_STRIDE: usize = 8;
const VAE_DECODER_WIDTH: usize = 128;
const UNET_MLP_RATIO: f64 = 4.0;
const CLIP_CONTEXT_TOKENS: usize = 77;
/// Both text towers live under one wrapper in a single-file checkpoint.
const TEXT_PREFIX: &str = "conditioner.";
const CLIP_CONTEXT_DIM: usize = 1280;

/// Resident bytes of the part of a single-file checkpoint under `prefix`, from the
/// safetensors header alone - no tensor data is read.
///
/// Written-down parameter counts (2.6 B for the denoiser, 0.8 B for the towers) are
/// facts about ONE checkpoint, and this family is a family of checkpoints: a fine-tune
/// that widens a block makes them quietly wrong, in the direction that under-reserves.
/// The file is open at this point anyway, and it knows.
fn component_bytes(checkpoint: &str, prefix: &str, elem: u64) -> u64 {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(checkpoint) else {
        return 0;
    };
    let mut len = [0u8; 8];
    if f.read_exact(&mut len).is_err() {
        return 0;
    }
    let n = u64::from_le_bytes(len) as usize;
    // not-a-vram-size: a bound on a header length read out of the file.
    if n == 0 || n > (64 << 20) {
        return 0;
    }
    let mut buf = vec![0u8; n];
    if f.read_exact(&mut buf).is_err() {
        return 0;
    }
    let Ok(json) = serde_json::from_slice::<serde_json::Value>(&buf) else {
        return 0;
    };
    let Some(obj) = json.as_object() else {
        return 0;
    };
    let mut total = 0u64;
    for (name, info) in obj {
        if name == "__metadata__" || !name.starts_with(prefix) {
            continue;
        }
        let count: u64 = info
            .get("shape")
            .and_then(|s| s.as_array())
            .map(|dims| dims.iter().filter_map(|d| d.as_u64()).product())
            .unwrap_or(0);
        total += count;
    }
    total * elem
}

impl SdxlPipeline {
    /// Load all three networks from one single-file checkpoint.
    ///
    /// `tokenizer_json` is the CLIP BPE vocabulary - both towers share it.
    pub fn load(
        checkpoint: &str,
        tokenizer_json: &str,
        dtype: DType,
        geom: crate::inference::place::runtime_demand::RequestGeometry,
    ) -> Result<Self> {
        Self::load_inner(checkpoint, tokenizer_json, dtype, geom)
    }

    fn load_inner(
        checkpoint: &str,
        tokenizer_json: &str,
        dtype: DType,
        geom: crate::inference::place::runtime_demand::RequestGeometry,
    ) -> Result<Self> {
        // Placement follows the fleet contract, it is not a per-model choice: the
        // UNet is the HOT component (run twice per step under guidance) and takes
        // the fastest card that fits it; the text towers are ONE-SHOT and are placed
        // afterwards, on whatever capacity is left, so they never compete with the
        // denoiser for the fast card. Loading all three on one device is what OOMed
        // the first end-to-end run.
        let elem = if dtype == DType::F32 { 4u64 } else { 2 };
        let adapter_bytes = 0u64;
        // 2.6 B parameters, plus room for the activations of one forward AT THIS
        // REQUEST'S SIZE. The activation term was a flat 3 GiB, which is a different
        // amount of wrong at every resolution; it now comes from the UNet's own level
        // widths and attention stages, where the attention dominates by an order of
        // magnitude at anything above 512.
        let _unet_bytes = component_bytes(
            checkpoint,
            crate::inference::model::sdxl::unet::CHECKPOINT_PREFIX,
            elem,
        ) + crate::inference::place::runtime_demand::unet_with_attention_bytes(
            geom.height / VAE_STRIDE,
            geom.width / VAE_STRIDE,
            &crate::inference::model::sdxl::unet::level_channels(),
            &crate::inference::model::sdxl::unet::attention_levels(),
            crate::inference::model::sdxl::unet::HEAD_DIM,
            UNET_MLP_RATIO,
        ) + adapter_bytes;
        // EVERY card, fastest first - not the one card that happens to hold it whole.
        // Asking for a single device meant a UNet that fits nowhere went to the host
        // ENTIRELY, which is the worst of the three outcomes and the one the fleet
        // contract exists to prevent. It is spread across the cards instead, and only
        // the stages that fit nowhere land on the host.
        let probed = crate::inference::place::vram_manager::probe(0);
        let unet_devices: Vec<Device> = probed.iter().map(|(_, _, d)| d.clone()).collect();
        // What each card may hold in WEIGHTS: what is free on it, less the scratch this
        // request's denoise will need there. Every card running stages pays that
        // scratch, so it is deducted per card and not once.
        let per_card_scratch = crate::inference::place::runtime_demand::unet_with_attention_bytes(
            geom.height / VAE_STRIDE,
            geom.width / VAE_STRIDE,
            &crate::inference::model::sdxl::unet::level_channels(),
            &crate::inference::model::sdxl::unet::attention_levels(),
            crate::inference::model::sdxl::unet::HEAD_DIM,
            UNET_MLP_RATIO,
        );
        let budgets: Vec<u64> = probed
            .iter()
            .map(|(_, free, _)| free.saturating_sub(per_card_scratch))
            .collect();
        let unet_dev = unet_devices.first().cloned().unwrap_or(Device::Cpu);
        let unet = if unet_devices.is_empty() {
            SdxlUnet::load(checkpoint, &Device::Cpu, dtype)?
        } else {
            SdxlUnet::load_planned(checkpoint, &unet_devices, &budgets, dtype)?
        };

        // The text towers stay F32 even when the denoiser is halved. They were never
        // the memory problem - the UNet was - and their embedding lookup has no
        // half-precision path on the device, so a BF16 table falls back to the host
        // and fails there. 0.8 B parameters at F32 is 3.2 GB, which fits beside a
        // BF16 UNet; the conditioning they produce is converted at the UNet's door.
        let text_dtype = DType::F32;
        // The towers see a fixed, short prompt window, so their scratch is a function
        // of that window and their own width - not of the image being rendered.
        let text_bytes = component_bytes(checkpoint, TEXT_PREFIX, 4)
            + crate::inference::place::runtime_demand::dit_activation_bytes(
                CLIP_CONTEXT_TOKENS,
                crate::inference::model::sdxl::unet::CONTEXT_DIM,
                CLIP_CONTEXT_DIM / crate::inference::model::sdxl::unet::HEAD_DIM,
                UNET_MLP_RATIO,
            );
        let text_dev = crate::inference::place::vram_manager::pick_device_elsewhere(
            "sdxl text",
            text_bytes,
            &unet_dev,
        )
        .map(|(_, _, d)| d)
        .unwrap_or(Device::Cpu);
        let text = SdxlTextEncoders::load(checkpoint, &text_dev, text_dtype)?;

        // The VAE stays F32 - its decode is short and its convolutions are the part
        // most visible in the output - and it is placed against its DECODE peak, not
        // its weights: the widest full-resolution feature map plus its 3x3 im2col.
        let vae_bytes = component_bytes(
            checkpoint,
            crate::inference::model::sdxl::vae::CHECKPOINT_PREFIX,
            4,
        ) + crate::inference::place::runtime_demand::vae_decode_bytes(
            geom.height,
            geom.width,
            VAE_DECODER_WIDTH,
        );
        let vae_dev = crate::inference::place::vram_manager::pick_device_elsewhere(
            "sdxl vae", vae_bytes, &unet_dev,
        )
        .map(|(_, _, d)| d)
        .unwrap_or(Device::Cpu);
        let vae = crate::inference::model::sdxl::vae::load(checkpoint, &vae_dev, DType::F32)?;

        Ok(Self {
            unet,
            text,
            vae,
            controlnet: None,
            control: None,
            control_device: Device::Cpu,
            tokenizer: tokenizers::Tokenizer::from_file(tokenizer_json)
                .map_err(|e| crate::tensor::Error(format!("clip tokenizer: {e}")))?,
            schedule: Schedule::new(),
            device: unet_dev,
            sampler: SamplerKind::default(),
            scheduler: SchedulerKind::default(),
            attached: Vec::new(),
            attached_matched: 0,
        })
    }

    /// VAE-encode an image to the latent the sampler works in.
    ///
    /// The encoder emits the distribution's mean; the scaling the decoder undoes is
    /// applied here so the value lives on the same scale as a sampled latent. Skipping
    /// it leaves the source ~5x too small and the first step erases it.
    fn encode_image(&self, rgb: &[u8], width: usize, height: usize) -> Result<Vec<f32>> {
        if rgb.len() != 3 * width * height {
            return Err(crate::tensor::Error(format!(
                "sdxl img2img: expected {}x{} RGB, got {} bytes",
                width,
                height,
                rgb.len()
            )));
        }
        let plane = width * height;
        let mut chw = vec![0f32; 3 * plane];
        for i in 0..plane {
            for c in 0..3 {
                // The VAE is trained on [-1, 1].
                chw[c * plane + i] = f32::from(rgb[3 * i + c]) / 127.5 - 1.0;
            }
        }
        let x =
            Tensor::from_vec_f32(chw, vec![1, 3, height, width])?.to_device(self.vae.device())?;
        Ok(self.vae.encode(&x)?.to_device(&Device::Cpu)?.to_vec_f32())
    }

    /// BPE ids without the bracketing tokens (the encoders add those themselves).
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| crate::tensor::Error(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Generate one image. Returns interleaved RGB8 of `width * height`.
    pub fn generate(
        &self,
        prompt: &str,
        negative: &str,
        width: usize,
        height: usize,
        steps: usize,
        guidance: f32,
        seed: u64,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Vec<u8>> {
        self.render(
            prompt, negative, width, height, steps, guidance, seed, None, cancel,
        )
    }

    /// Load a LoRA and attach it to every projection of the UNet it names.
    ///
    /// `strength` scales the adapter's own alpha/rank factor, so 1.0 is "as trained".
    /// Errors when the file matches NOTHING, which is what an adapter for a different
    /// architecture does - and the alternative is an unchanged image with no explanation.
    ///
    /// A file can also match only in PART: convolution adapters are a different shape
    /// and are not merged. Partial is not silent - what was left out is counted and
    /// said, because an adapter that half-applies renders something the user asked for
    /// nowhere, and the only thing worse than that is not knowing.
    pub fn load_lora(&mut self, path: &str, strength: f32) -> Result<usize> {
        let file = crate::inference::load::lora::LoraFile::load(path, &self.device)?;
        let matched = self.unet.apply_lora(&file, strength)?;
        if matched == 0 {
            return Err(crate::inference::load::lora::no_match_error(path, &file));
        }
        if matched < file.len() {
            tracing::warn!(
                "sdxl: adapter '{path}' attached on {matched} of its {} modules - the rest \
                 are convolution adapters, which are not merged",
                file.len()
            );
        }
        Ok(matched)
    }

    /// Give this render a structural conditioner, from an encoded pose or edge image.
    ///
    /// The network is loaded on FIRST use and kept: it is 2.5 GB, and a render series
    /// carries the same control image on every request, so reloading it per render
    /// would cost more than the render. Passing `None` clears the control without
    /// unloading, so the next request that wants one pays nothing.
    ///
    /// `scale` attenuates the whole residual set; at 0 the base model is reproduced
    /// exactly, which is what makes the feature safe to leave wired in.
    pub fn set_control(&mut self, image: Option<&[u8]>, scale: f32) -> Result<()> {
        let Some(bytes) = image else {
            self.control = None;
            return Ok(());
        };
        if self.controlnet.is_none() {
            // Resolved from the shared config, like every other model file - the
            // engine does not carry the directory and should not have to.
            let path = crate::config::hf_models_dir()
                .join(crate::inference::model::sdxl::controlnet::OPENPOSE_SDXL);
            if !path.exists() {
                return Err(crate::tensor::Error(format!(
                    "structural conditioning needs {}; it is not installed",
                    path.display()
                )));
            }
            // PLACED, not dropped onto the UNet's card. This is a per-step component
            // in its own right - it runs beside the UNet on every step - and it is
            // 2.5 GB, so loading it wherever the pipeline happens to live exhausts
            // that card. Which is what it did: the first render with a control image
            // came back as an out-of-memory from the upload.
            //
            // The residuals cross to the card that consumes them; the injection
            // already moves them, so a different card costs a transfer per step and
            // not correctness.
            let weights = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            // Its activations are a subset of the UNet's - it runs the down path and
            // the middle, not the up path - so a share of its own weights bounds them.
            // A ratio, so it holds whatever ControlNet is dropped in.
            const SCRATCH_SHARE: u64 = 4;
            let want = weights + weights / SCRATCH_SHARE;
            // ANYWHERE BUT THE DENOISER'S CARD, first. Both run on every step, so they
            // compete for the same card's activations - and asking the ranked picker
            // simply put this one on top of the UNet and the upload ran out of memory.
            // Reading free VRAM there is not enough: the resident pipeline's own
            // scratch is not in that number.
            //
            // Sharing is allowed only when nothing else will have it AND the card can
            // hold both. Otherwise this returns nothing and the caller is told, which
            // is a better answer than an out-of-memory from a driver.
            let dev = crate::inference::place::vram_manager::pick_device_elsewhere(
                "sdxl controlnet",
                want,
                &self.device,
            )
            .or_else(|| {
                crate::inference::place::vram_manager::pick_device_for("sdxl controlnet", want)
            })
            .filter(|(_, free, _)| *free >= want)
            .map(|(_, _, d)| d)
            .ok_or_else(|| {
                crate::tensor::Error(
                    "structural conditioning needs a card with room for it and none has \
                     enough free; render without it, or at a smaller size"
                        .to_string(),
                )
            })?;
            tracing::info!("sdxl controlnet -> {:?}", dev.location());
            self.control_device = dev.clone();
            self.controlnet = Some(crate::inference::model::sdxl::controlnet::load(
                path.to_str().unwrap_or_default(),
                &dev,
                self.unet.dtype(),
            )?);
        }
        // The encoder wants the image at FULL resolution in 0..1; it brings it to the
        // latent grid itself, and doing that here would fight its own strides.
        // Through the oriented decoder: a raw decode ignores the EXIF rotation, so a
        // pose photographed on a phone would arrive sideways - and a sideways pose
        // conditions the render to a sideways subject, which is the one failure this
        // feature exists to prevent.
        let img = crate::inference::media::image_processor::decode_image_oriented(bytes)
            .map_err(|e| crate::tensor::Error(format!("control image: {e}")))?
            .to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let mut planes = vec![0f32; 3 * h * w];
        for (i, px) in img.pixels().enumerate() {
            planes[i] = px[0] as f32 / 255.0;
            planes[h * w + i] = px[1] as f32 / 255.0;
            planes[2 * h * w + i] = px[2] as f32 / 255.0;
        }
        let t = Tensor::from_vec_f32(planes, vec![1, 3, h, w])?
            .to_device(&self.control_device)?
            .to_dtype(self.unet.dtype())?;
        self.control = Some((t, scale));
        Ok(())
    }

    /// Drop every attached adapter.
    pub fn clear_loras(&mut self) {
        self.unet.clear_lora();
        self.attached.clear();
        self.attached_matched = 0;
    }

    /// Make `want` the attached adapter set, reusing the current one when it matches.
    ///
    /// Attaching is not cheap: the file is read from disk and its tensors uploaded to
    /// the render device, which for a full-rank SDXL adapter is hundreds of megabytes -
    /// measured at roughly 4 s, against 2.6 s for the render it precedes. Requests carry
    /// their whole adapter set rather than mutating a session, so the same set arrives
    /// again on every render of a series; comparing before rebuilding turns all but the
    /// first into a no-op.
    ///
    /// Returns the number of projections matched across the set, or 0 when nothing is
    /// attached.
    pub fn set_loras(&mut self, want: &[(String, f32)]) -> Result<usize> {
        if self.attached == want {
            return Ok(self.attached_matched);
        }
        self.clear_loras();
        let mut matched = 0;
        for (path, strength) in want {
            match self.load_lora(path, *strength) {
                Ok(n) => matched += n,
                Err(e) => {
                    // Leave the pipeline on its base weights: a partially attached set
                    // would render something neither the caller nor the next request
                    // asked for.
                    self.clear_loras();
                    return Err(e);
                }
            }
        }
        self.attached = want.to_vec();
        self.attached_matched = matched;
        Ok(matched)
    }

    /// Choose the solver and the sigma curve for subsequent renders.
    ///
    /// Set per request under the media gate, which serialises generations, so a plain
    /// field is enough - there is never a second render reading it concurrently.
    pub fn set_sampling(&mut self, sampler: SamplerKind, scheduler: SchedulerKind) {
        self.sampler = sampler;
        self.scheduler = scheduler;
    }

    /// Re-draw an existing image: the sampler starts partway down the ladder from the
    /// SOURCE latent instead of from noise.
    ///
    /// `strength` is how much of the ladder still runs - 0 keeps the input, 1 is a
    /// plain generation. That truncation IS img2img: the steps that were skipped are
    /// the ones that would have decided the composition, so what survives from the
    /// source is exactly what those early steps fix.
    ///
    /// `init` is interleaved RGB8 at `width * height`.
    pub fn generate_img2img(
        &self,
        prompt: &str,
        negative: &str,
        init: &[u8],
        width: usize,
        height: usize,
        steps: usize,
        guidance: f32,
        strength: f32,
        seed: u64,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Vec<u8>> {
        self.render(
            prompt,
            negative,
            width,
            height,
            steps,
            guidance,
            seed,
            Some((init, strength)),
            cancel,
        )
    }

    /// The one sampling loop, from noise or from an image.
    fn render(
        &self,
        prompt: &str,
        negative: &str,
        width: usize,
        height: usize,
        steps: usize,
        guidance: f32,
        seed: u64,
        init: Option<(&[u8], f32)>,
        cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    ) -> Result<Vec<u8>> {
        let (sampler, scheduler) = (self.sampler, self.scheduler);
        let (lw, lh) = (width / 8, height / 8);

        // Conditioning: both towers, for the prompt and the negative prompt.
        let (ctx_c, pooled_c) = self.text.encode(&self.tokenize(prompt)?)?;
        let (ctx_u, pooled_u) = self.text.encode(&self.tokenize(negative)?)?;
        // SDXL's micro-conditioning: the size the image claims to be, the crop it
        // claims to come from, and the size it targets.
        let sizes = [
            height as f32,
            width as f32,
            0.0,
            0.0,
            height as f32,
            width as f32,
        ];
        let label_c = self.unet.label_vector(&pooled_c, sizes)?;
        let label_u = self.unet.label_vector(&pooled_u, sizes)?;

        // A seeded draw, so the same request returns the same image.
        crate::tensor::rng::set_global_seed(seed);
        let full = match scheduler {
            SchedulerKind::Normal => self.schedule.sigma_ladder(steps),
            SchedulerKind::Karras => karras_sigmas(
                steps,
                self.schedule.sigma_min(),
                self.schedule.sigma_max(),
                7.0,
            ),
            SchedulerKind::Exponential => {
                exponential_sigmas(steps, self.schedule.sigma_min(), self.schedule.sigma_max())
            }
        };
        // How far down the ladder to start. A strength of 1 keeps the whole ladder and
        // is identical to a plain generation; anything less skips the early, high-noise
        // steps, which is what preserves the source's layout.
        let start = match init {
            Some((_, strength)) => {
                let s = strength.clamp(0.0, 1.0);
                let skipped = ((1.0 - s) * steps as f32).round() as usize;
                // Always leave at least one real step, or the call would decode its
                // own input back and look like it did nothing.
                skipped.min(steps.saturating_sub(1))
            }
            None => 0,
        };
        let sigmas = &full[start..];
        let mut solver = DpmPP2M::new();
        let noise = {
            let z = Tensor::zeros_on(vec![1, LATENT_CHANNELS, lh, lw], DType::F32, &Device::Cpu)?;
            z.randn_like()?
        };
        let mut x: Vec<f32> = match init {
            // x_t = x_0 + sigma * noise, the convention this ladder is written in.
            Some((rgb, _)) => {
                let latent = self.encode_image(rgb, width, height)?;
                let noised = noise.affine(sigmas[0], 0.0)?.to_vec_f32();
                latent.iter().zip(&noised).map(|(a, b)| a + b).collect()
            }
            None => noise.affine(sigmas[0], 0.0)?.to_vec_f32(),
        };
        crate::tensor::rng::clear_global_seed();

        for i in 0..sigmas.len() - 1 {
            if let Some(c) = cancel {
                c.bail()?;
            }
            let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
            // The model is fed the SCALED latent and the timestep matching sigma.
            let scaled: Vec<f32> = x.iter().map(|v| v * input_scale(sigma)).collect();
            // The sampler works in F32 on the host; the UNet meets it in its own
            // dtype, and hands the prediction back to the host as F32.
            let xin = Tensor::from_vec_f32(scaled, vec![1, LATENT_CHANNELS, lh, lw])?
                .to_device(&self.device)?
                .to_dtype(self.unet.dtype())?;
            let t = self.schedule.timestep_for(sigma);

            // The residuals depend on the latent AND the conditioning, so they are
            // recomputed per step and per guidance branch - the same rule the UNet
            // itself follows. Skipped entirely without a control image.
            let res_c = match (&self.controlnet, &self.control) {
                (Some(net), Some((img, scale))) => Some(net.forward(
                    &xin.to_device(&self.control_device)?,
                    t,
                    &ctx_c.to_device(&self.control_device)?,
                    &label_c.to_device(&self.control_device)?,
                    img,
                    *scale,
                )?),
                _ => None,
            };
            let res_u = match (guidance > 1.0, &self.controlnet, &self.control) {
                (true, Some(net), Some((img, scale))) => Some(net.forward(
                    &xin.to_device(&self.control_device)?,
                    t,
                    &ctx_u.to_device(&self.control_device)?,
                    &label_u.to_device(&self.control_device)?,
                    img,
                    *scale,
                )?),
                _ => None,
            };
            let eps_c = self
                .unet
                .forward_controlled(&xin, t, &ctx_c, &label_c, res_c.as_ref())?;
            let eps = if guidance > 1.0 {
                let eps_u =
                    self.unet
                        .forward_controlled(&xin, t, &ctx_u, &label_u, res_u.as_ref())?;
                cfg(
                    &eps_c
                        .to_dtype(DType::F32)?
                        .to_device(&Device::Cpu)?
                        .to_vec_f32(),
                    &eps_u
                        .to_dtype(DType::F32)?
                        .to_device(&Device::Cpu)?
                        .to_vec_f32(),
                    guidance,
                )
            } else {
                eps_c
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec_f32()
            };
            let d = denoised(&x, &eps, sigma);
            x = match sampler {
                SamplerKind::Euler => euler_step(&x, &d, sigma, sigma_next),
                SamplerKind::DpmPP2M => {
                    let prev = if i == 0 { None } else { Some(sigmas[i - 1]) };
                    solver.step(&x, &d, prev, sigma, sigma_next)
                }
            };
        }

        // Decode. The VAE applies the SDXL scaling itself.
        let lat = Tensor::from_vec_f32(x, vec![1, LATENT_CHANNELS, lh, lw])?;
        // The VAE stays F32 whatever the denoiser runs in, so the latent converts here.
        let img = self
            .vae
            .decode(&lat.to_dtype(DType::F32)?)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec_f32();
        let plane = height * width;
        let mut rgb = Vec::with_capacity(3 * plane);
        for y in 0..height {
            for xp in 0..width {
                for c in 0..3 {
                    let v = img[c * plane + y * width + xp];
                    rgb.push((((v + 1.0) * 127.5).clamp(0.0, 255.0)) as u8);
                }
            }
        }
        Ok(rgb)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// Same negative the engine conditions SDXL against.
    const SDXL_NEGATIVE_FOR_TEST: &str =
        "blurry, low quality, distorted, deformed, watermark, text";

    /// THE end-to-end gate: a real checkpoint, a real prompt, a real PNG.
    ///
    /// Everything before this validated a stage in isolation; only a generated
    /// image proves the four fit together - the conditioning order, the sigma
    /// ladder, the input scaling, the guidance and the decode scaling all have to
    /// agree or the result is noise or a grey field.
    #[test]
    #[ignore = "needs an SDXL checkpoint + the CLIP tokenizer in the HF cache"]
    fn generates_an_image_from_a_real_checkpoint() {
        let dir = crate::config::Config::load_test().get_hf_models_dir();
        let ckpt = std::fs::read_dir(dir.join("raymnants"))
            .expect("raymnants dir")
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .expect("an SDXL .safetensors");
        // The CLIP BPE vocabulary, already cached for the Flux pipeline.
        let tok = walk_for(&dir.join("hub"), "tokenizer.json", "clip-vit-large-patch14")
            .expect("the CLIP tokenizer in the HF cache");
        let t0 = std::time::Instant::now();
        // The engine runs this family in BF16; the gate follows it so a dtype the
        // production path uses cannot pass here and fail there.
        let dtype = if std::env::var("SDXL_F32").is_ok() {
            DType::F32
        } else {
            DType::BF16
        };
        println!("loading SDXL in {dtype:?}");
        let pipe = SdxlPipeline::load(
            ckpt.to_str().unwrap(),
            tok.to_str().unwrap(),
            dtype,
            crate::inference::place::runtime_demand::RequestGeometry::new(1024, 1024),
        )
        .expect("pipeline load");
        println!("SDXL pipeline loaded in {:.1}s", t0.elapsed().as_secs_f32());

        let t1 = std::time::Instant::now();
        let rgb = pipe
            .generate(
                "a red fox sitting on a mossy rock in a forest, detailed fur",
                "blurry, low quality",
                512,
                512,
                12,
                7.0,
                7,
                None,
            )
            .expect("generate");
        println!(
            "SDXL 512x512 / 12 steps in {:.1}s",
            t1.elapsed().as_secs_f32()
        );
        assert_eq!(rgb.len(), 3 * 512 * 512);

        let out = std::env::temp_dir().join("sdxl_first_image.png");
        image::RgbImage::from_raw(512, 512, rgb.clone())
            .expect("rgb buffer")
            .save(&out)
            .expect("write png");
        println!("wrote {}", out.display());

        // A failed pipeline returns a flat field (all-grey) or saturated noise; a
        // real image has structure. Check both ends.
        let mean = rgb.iter().map(|v| *v as f64).sum::<f64>() / rgb.len() as f64;
        let var = rgb.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / rgb.len() as f64;
        println!("SDXL image mean {mean:.1}, stddev {:.1}", var.sqrt());
        assert!(
            var.sqrt() > 10.0,
            "the image is flat (stddev {:.2})",
            var.sqrt()
        );
        assert!(
            (20.0..235.0).contains(&mean),
            "the image is saturated (mean {mean:.1})"
        );
    }

    /// img2img must actually be a DIAL: low strength keeps the source, high strength
    /// leaves it. A single strength proves nothing - an implementation that ignored
    /// the input entirely, or one that returned it untouched, each passes half of this.
    #[test]
    #[ignore = "needs an SDXL checkpoint + the CLIP tokenizer in the HF cache"]
    fn img2img_strength_moves_between_the_source_and_a_fresh_render() {
        let dir = crate::config::Config::load_test().get_hf_models_dir();
        let ckpt = std::fs::read_dir(dir.join("raymnants"))
            .expect("raymnants dir")
            .flatten()
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .expect("an SDXL .safetensors");
        let tok = walk_for(&dir.join("hub"), "tokenizer.json", "clip-vit-large-patch14")
            .expect("the CLIP tokenizer");
        let pipe = SdxlPipeline::load(
            ckpt.to_str().unwrap(),
            tok.to_str().unwrap(),
            DType::F32,
            crate::inference::place::runtime_demand::RequestGeometry::new(1024, 1024),
        )
        .expect("pipeline");

        let (w, h) = (512usize, 512usize);
        // A source with unmistakable structure: colour blocks the model would never
        // invent, so "did the source survive" is visible in the numbers.
        let mut src = vec![0u8; 3 * w * h];
        for y in 0..h {
            for x in 0..w {
                let i = 3 * (y * w + x);
                let (a, b) = (x * 2 / w, y * 2 / h);
                let c: [u8; 3] = match (a, b) {
                    (0, 0) => [220, 30, 30],
                    (1, 0) => [30, 220, 30],
                    (0, 1) => [30, 30, 220],
                    _ => [230, 230, 40],
                };
                src[i..i + 3].copy_from_slice(&c);
            }
        }
        let run = |strength: f32| -> Vec<u8> {
            pipe.generate_img2img(
                "a photograph of a forest clearing",
                SDXL_NEGATIVE_FOR_TEST,
                &src,
                w,
                h,
                12,
                7.0,
                strength,
                5,
                None,
            )
            .expect("img2img")
        };
        let mae = |a: &[u8], b: &[u8]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs())
                .sum::<f64>()
                / a.len() as f64
        };

        let gentle = run(0.2);
        let strong = run(0.9);
        let d_gentle = mae(&gentle, &src);
        let d_strong = mae(&strong, &src);
        println!("distance from the source: strength 0.2 -> {d_gentle:.1}, 0.9 -> {d_strong:.1}");
        for (img, name) in [(&gentle, "gentle"), (&strong, "strong")] {
            let out = std::env::temp_dir().join(format!("sdxl_img2img_{name}.png"));
            image::RgbImage::from_raw(w as u32, h as u32, img.clone())
                .expect("buf")
                .save(&out)
                .expect("write");
            println!("wrote {}", out.display());
        }
        assert!(
            d_gentle > 1.0,
            "strength 0.2 returned the input untouched ({d_gentle:.1})"
        );
        assert!(
            d_strong > d_gentle + 10.0,
            "strength did not move the result: 0.2 -> {d_gentle:.1}, 0.9 -> {d_strong:.1}"
        );
    }

    /// Find a file by name under an HF cache, restricted to a repo directory.
    fn walk_for(root: &std::path::Path, name: &str, repo: &str) -> Option<std::path::PathBuf> {
        let mut stack = vec![root.to_path_buf()];
        while let Some(d) = stack.pop() {
            for e in std::fs::read_dir(&d).ok()?.flatten() {
                let p = e.path();
                if p.is_dir() {
                    if p.to_string_lossy().contains(repo) || d == root {
                        stack.push(p);
                    }
                } else if p.file_name().is_some_and(|f| f == name)
                    && p.to_string_lossy().contains(repo)
                {
                    return Some(p);
                }
            }
        }
        None
    }
}
