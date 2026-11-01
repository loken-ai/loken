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
    /// Latent tile side and overlap of the next render's VAE decode, `None` for a whole
    /// decode (see `set_vae_tile`).
    vae_tile: Option<(usize, usize)>,
    /// The adapter set currently attached, in request order. Requests carry their whole
    /// adapter set, so this is what makes a repeat request free - see `set_loras`.
    attached: Vec<(String, f32)>,
    /// Projections matched by `attached`, so a reused set still reports what it did.
    attached_matched: usize,
    /// Bytes the denoise keeps free on every card a stage runs on, at the geometry the
    /// pipeline was loaded for.
    unet_scratch: u64,
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

/// One request of a batch render: what it conditions on, how hard it is guided, the
/// seed its noise is drawn from, and the token that stops it.
pub struct SdxlRow<'a> {
    pub prompt: &'a str,
    pub negative: &'a str,
    pub guidance: f32,
    pub seed: u64,
    pub cancel: Option<&'a crate::inference::serve::cancel::CancelToken>,
}

/// The conditional prediction of every row and, under guidance, the unconditional one,
/// each stacked in row order on the host.
type Predictions = (Vec<f32>, Option<Vec<f32>>);

/// Tensors of one row each, stacked along the batch axis in row order.
fn stack_rows(rows: &[Tensor]) -> Result<Tensor> {
    match rows {
        [one] => Ok(one.clone()),
        _ => Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0),
    }
}

/// The sampling loop over latents that share one sigma ladder, one per row.
///
/// `predict` takes the scaled latents of the rows still running, stacked, the sigma they sit
/// at, whether to answer the unconditional branch and the indices of those rows, and answers
/// [`Predictions`]; it is asked for the unconditional branch when any running row is guided.
/// Each row is guided at its own scale and keeps its own solver history, so a row's latent
/// depends on its own inputs alone.
///
/// `gone` is asked before every step for each running row. A row whose request went away
/// leaves the pass there: it is no longer computed, and its place in the answer is `None`.
/// The loop ends with an error when no row is left.
fn sample_rows(
    xs: Vec<Vec<f32>>,
    guidance: &[f32],
    sigmas: &[f32],
    sampler: SamplerKind,
    mut predict: impl FnMut(&[f32], f32, bool, &[usize]) -> Result<Predictions>,
    mut gone: impl FnMut(usize) -> bool,
    on_step: crate::inference::sample::StepSink,
) -> Result<Vec<Option<Vec<f32>>>> {
    let plane = xs.first().map_or(0, Vec::len);
    let total = xs.len();
    let mut running: Vec<usize> = (0..total).collect();
    let mut latents: Vec<Option<Vec<f32>>> = xs.into_iter().map(Some).collect();
    let mut solvers: Vec<DpmPP2M> = (0..total).map(|_| DpmPP2M::new()).collect();
    let total_steps = sigmas.len() - 1;
    on_step(0, total_steps);
    for i in 0..total_steps {
        running.retain(|&r| {
            let stays = !gone(r);
            if !stays {
                latents[r] = None;
            }
            stays
        });
        if running.is_empty() {
            return Err(crate::tensor::Error(
                "generation cancelled (client disconnected)".into(),
            ));
        }
        let guided = running.iter().any(|&r| guidance[r] > 1.0);
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // The model is fed the SCALED latent and the timestep matching sigma.
        let scale = input_scale(sigma);
        let scaled: Vec<f32> = running
            .iter()
            .flat_map(|&r| latents[r].iter().flatten().map(|v| v * scale))
            .collect();
        let (eps_c, eps_u) = predict(&scaled, sigma, guided, &running)?;
        for (slot, &r) in running.iter().enumerate() {
            let span = slot * plane..(slot + 1) * plane;
            let eps = match (&eps_u, guidance[r] > 1.0) {
                (Some(eps_u), true) => cfg(&eps_c[span.clone()], &eps_u[span], guidance[r]),
                _ => eps_c[span].to_vec(),
            };
            let x = latents[r].take().unwrap_or_default();
            let d = denoised(x.as_slice(), &eps, sigma);
            latents[r] = Some(match sampler {
                SamplerKind::Euler => euler_step(x.as_slice(), &d, sigma, sigma_next),
                SamplerKind::DpmPP2M => {
                    let prev = if i == 0 { None } else { Some(sigmas[i - 1]) };
                    solvers[r].step(x.as_slice(), &d, prev, sigma, sigma_next)
                }
            });
        }
        on_step(i + 1, total_steps);
    }
    Ok(latents)
}

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
        // EVERY card, fastest first - not the one card that happens to hold it whole.
        // Asking for a single device meant a UNet that fits nowhere went to the host
        // ENTIRELY, which is the worst of the three outcomes and the one the fleet
        // contract exists to prevent. The stages are spread across the cards instead,
        // each card keeping the scratch this request's denoise needs there, and only
        // the stages that fit nowhere land on the host.
        let per_card_scratch = crate::inference::place::runtime_demand::unet_with_attention_bytes(
            geom.height / VAE_STRIDE,
            geom.width / VAE_STRIDE,
            &crate::inference::model::sdxl::unet::level_channels(),
            &crate::inference::model::sdxl::unet::attention_levels(),
            crate::inference::model::sdxl::unet::HEAD_DIM,
            UNET_MLP_RATIO,
        );
        let unet = crate::inference::place::ladder::load_stages(
            "SDXL UNet",
            &crate::inference::model::sdxl::unet::stage_weight_bytes(dtype),
            per_card_scratch,
            |stages| SdxlUnet::load_planned(checkpoint, stages, dtype),
        )?;
        let unet_dev = unet.device().clone();
        // The one-shot towers and the decoder take the cards the UNet leaves alone first,
        // sharing one of its cards only when nothing else holds them, else the host.
        let hot = unet.cards();
        let aside = || {
            crate::inference::place::ladder::aside(
                &crate::inference::place::ladder::cards(0),
                &hot,
                0,
            )
        };
        let to_tensor = |e: anyhow::Error| crate::tensor::Error(format!("{e:#}"));

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
        let (text, _) = crate::inference::place::ladder::load_whole_on(
            "SDXL text",
            text_bytes,
            text_bytes,
            &aside(),
            None,
            |d| Ok(SdxlTextEncoders::load(checkpoint, d, text_dtype)?),
        )
        .map_err(to_tensor)?;

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
        let (vae, _) = crate::inference::place::ladder::load_whole_on(
            "SDXL VAE",
            vae_bytes,
            vae_bytes,
            &aside(),
            None,
            |d| {
                Ok(crate::inference::model::sdxl::vae::load(
                    checkpoint,
                    d,
                    DType::F32,
                )?)
            },
        )
        .map_err(to_tensor)?;

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
            vae_tile: None,
            attached: Vec::new(),
            attached_matched: 0,
            unet_scratch: per_card_scratch,
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
        on_step: crate::inference::sample::StepSink,
    ) -> Result<Vec<u8>> {
        self.render(
            prompt, negative, width, height, steps, guidance, seed, None, cancel, on_step,
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
            // The denoiser's cards are offered last, less the scratch its denoise keeps
            // free there.
            //
            // Sharing is allowed only when nothing else will have it AND the card can
            // hold both. Otherwise this returns nothing and the caller is told, which
            // is a better answer than an out-of-memory from a driver.
            let cards = crate::inference::place::ladder::whole_order(
                &crate::inference::place::ladder::aside(
                    &crate::inference::place::ladder::cards(0),
                    &self.unet.cards(),
                    self.unet_scratch,
                ),
                want,
                want,
            );
            if cards.is_empty() {
                return Err(crate::tensor::Error(
                    "structural conditioning needs a card with room for it and none has \
                     enough free; render without it, or at a smaller size"
                        .to_string(),
                ));
            }
            let dtype = self.unet.dtype();
            let (net, dev) = crate::inference::place::ladder::load_whole_on(
                "SDXL controlnet",
                want,
                want,
                &cards,
                None,
                |d| {
                    Ok(crate::inference::model::sdxl::controlnet::load(
                        path.to_str().unwrap_or_default(),
                        d,
                        dtype,
                    )?)
                },
            )
            .map_err(|e| crate::tensor::Error(format!("{e:#}")))?;
            tracing::info!("sdxl controlnet -> {:?}", dev.location());
            self.control_device = dev;
            self.controlnet = Some(net);
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

    /// Decode subsequent renders in overlapping `(side, overlap)` tiles, in latent pixels,
    /// or whole when `tile` is `None`. Set per request under the media gate, as
    /// `set_sampling` is.
    pub fn set_vae_tile(&mut self, tile: Option<(usize, usize)>) {
        self.vae_tile = tile;
    }

    /// The device the VAE decodes on.
    pub fn vae_device(&self) -> &Device {
        self.vae.device()
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
        on_step: crate::inference::sample::StepSink,
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
            on_step,
        )
    }

    /// Generate one image per row, all rows denoised together as one batch through the
    /// UNet. Returns interleaved RGB8 of `width * height` per row, in row order.
    ///
    /// The rows share the size, the step count, the solver and the sigma curve; each keeps
    /// its own prompt, negative prompt, guidance and seed, and gets the image it gets alone.
    /// A batch renders from noise and without a control image.
    pub fn generate_batch(
        &self,
        rows: &[SdxlRow<'_>],
        width: usize,
        height: usize,
        steps: usize,
        on_step: crate::inference::sample::StepSink,
    ) -> Result<Vec<Option<Vec<u8>>>> {
        self.render_rows(rows, width, height, steps, None, on_step)
    }

    /// The one sampling loop, from noise or from an image, for one request.
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
        on_step: crate::inference::sample::StepSink,
    ) -> Result<Vec<u8>> {
        let row = SdxlRow {
            prompt,
            negative,
            guidance,
            seed,
            cancel,
        };
        self.render_rows(&[row], width, height, steps, init, on_step)?
            .pop()
            .flatten()
            .ok_or_else(|| crate::tensor::Error("sdxl: the render produced no image".into()))
    }

    /// The ControlNet's residuals for one guidance branch, `None` without a control image.
    fn control_residuals(
        &self,
        xin: &Tensor,
        t: f32,
        ctx: &Tensor,
        label: &Tensor,
    ) -> Result<Option<crate::inference::model::sdxl::controlnet::ControlResiduals>> {
        match (&self.controlnet, &self.control) {
            (Some(net), Some((img, scale))) => Ok(Some(net.forward(
                &xin.to_device(&self.control_device)?,
                t,
                &ctx.to_device(&self.control_device)?,
                &label.to_device(&self.control_device)?,
                img,
                *scale,
            )?)),
            _ => Ok(None),
        }
    }

    /// The sampling loop over every row at once. `init` is only taken with a single row.
    fn render_rows(
        &self,
        rows: &[SdxlRow<'_>],
        width: usize,
        height: usize,
        steps: usize,
        init: Option<(&[u8], f32)>,
        on_step: crate::inference::sample::StepSink,
    ) -> Result<Vec<Option<Vec<u8>>>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        if rows.len() > 1 && (init.is_some() || self.control.is_some()) {
            return Err(crate::tensor::Error(
                "sdxl: a batch renders from noise, without a control image".into(),
            ));
        }
        let (sampler, scheduler) = (self.sampler, self.scheduler);
        let (lw, lh) = (width / 8, height / 8);

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
        // Conditioning: both towers, for each row's prompt and negative prompt, stacked
        // in row order.
        let mut ctx_c = Vec::with_capacity(rows.len());
        let mut ctx_u = Vec::with_capacity(rows.len());
        let mut label_c = Vec::with_capacity(rows.len());
        let mut label_u = Vec::with_capacity(rows.len());
        for row in rows {
            let (c, pooled_c) = self.text.encode(&self.tokenize(row.prompt)?)?;
            let (u, pooled_u) = self.text.encode(&self.tokenize(row.negative)?)?;
            label_c.push(self.unet.label_vector(&pooled_c, sizes)?);
            label_u.push(self.unet.label_vector(&pooled_u, sizes)?);
            ctx_c.push(c);
            ctx_u.push(u);
        }
        // The stack of every row, and the rows alone for the pass that loses some.
        let (stack_ctx_c, stack_ctx_u) = (stack_rows(&ctx_c)?, stack_rows(&ctx_u)?);
        let (stack_label_c, stack_label_u) = (stack_rows(&label_c)?, stack_rows(&label_u)?);
        let kept_stack = |alone: &[Tensor], whole: &Tensor, running: &[usize]| -> Result<Tensor> {
            if running.len() == alone.len() {
                return Ok(whole.clone());
            }
            let chosen: Vec<Tensor> = running.iter().map(|&r| alone[r].clone()).collect();
            stack_rows(&chosen)
        };

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
        let mut xs = Vec::with_capacity(rows.len());
        for row in rows {
            // A seeded draw per row, so the same request returns the same image, alone or
            // beside others.
            crate::tensor::rng::set_global_seed(row.seed);
            let noise = {
                let z =
                    Tensor::zeros_on(vec![1, LATENT_CHANNELS, lh, lw], DType::F32, &Device::Cpu)?;
                z.randn_like()?
            };
            xs.push(match init {
                // x_t = x_0 + sigma * noise, the convention this ladder is written in.
                Some((rgb, _)) => {
                    let latent = self.encode_image(rgb, width, height)?;
                    let noised = noise.affine(sigmas[0], 0.0)?.to_vec_f32();
                    latent.iter().zip(&noised).map(|(a, b)| a + b).collect()
                }
                None => noise.affine(sigmas[0], 0.0)?.to_vec_f32(),
            });
        }
        crate::tensor::rng::clear_global_seed();

        let n = rows.len();
        let guidance: Vec<f32> = rows.iter().map(|r| r.guidance).collect();
        // The sampler works in F32 on the host; the UNet meets it in its own dtype, and
        // hands the prediction back to the host as F32.
        let to_host = |eps: Tensor| -> Result<Vec<f32>> {
            Ok(eps
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .to_vec_f32())
        };
        let predict =
            |scaled: &[f32], sigma: f32, guided: bool, running: &[usize]| -> Result<Predictions> {
                let ctx_c = kept_stack(&ctx_c, &stack_ctx_c, running)?;
                let ctx_u = kept_stack(&ctx_u, &stack_ctx_u, running)?;
                let label_c = kept_stack(&label_c, &stack_label_c, running)?;
                let label_u = kept_stack(&label_u, &stack_label_u, running)?;
                let xin = Tensor::from_vec_f32(
                    scaled.to_vec(),
                    vec![running.len(), LATENT_CHANNELS, lh, lw],
                )?
                .to_device(&self.device)?
                .to_dtype(self.unet.dtype())?;
                let t = self.schedule.timestep_for(sigma);
                // The residuals depend on the latent AND the conditioning, so they are
                // recomputed per step and per guidance branch - the same rule the UNet
                // itself follows. Skipped entirely without a control image.
                let res_c = self.control_residuals(&xin, t, &ctx_c, &label_c)?;
                let eps_c =
                    self.unet
                        .forward_controlled(&xin, t, &ctx_c, &label_c, res_c.as_ref())?;
                let eps_u = if guided {
                    let res_u = self.control_residuals(&xin, t, &ctx_u, &label_u)?;
                    Some(to_host(self.unet.forward_controlled(
                        &xin,
                        t,
                        &ctx_u,
                        &label_u,
                        res_u.as_ref(),
                    )?)?)
                } else {
                    None
                };
                Ok((to_host(eps_c)?, eps_u))
            };
        // A row whose request went away leaves the pass at the next step, and the batch
        // stops when no row is left.
        let gone = |r: usize| rows[r].cancel.is_some_and(|c| c.is_cancelled());
        let xs = sample_rows(xs, &guidance, sigmas, sampler, predict, gone, on_step)?;

        xs.into_iter()
            .map(|x| x.map(|x| self.decode_latent(x, width, height)).transpose())
            .collect()
    }

    /// Decode one latent to interleaved RGB8. The VAE applies the SDXL scaling itself.
    fn decode_latent(&self, x: Vec<f32>, width: usize, height: usize) -> Result<Vec<u8>> {
        let (lw, lh) = (width / 8, height / 8);
        let lat = Tensor::from_vec_f32(x, vec![1, LATENT_CHANNELS, lh, lw])?;
        // The VAE stays F32 whatever the denoiser runs in, so the latent converts here.
        let lat = lat.to_dtype(DType::F32)?;
        let decoded = match self.vae_tile {
            Some((tile, overlap)) => self.vae.decode_tiled(&lat, tile, overlap)?,
            None => self.vae.decode(&lat)?,
        };
        let img = decoded
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

    /// A stand-in denoiser that reads each row's latent as a whole: its prediction for a
    /// value depends on that value and on the mean of its own row, so a row read at the
    /// wrong offset, or mixed with its neighbour, changes the answer.
    fn row_denoiser(
        plane: usize,
    ) -> impl FnMut(&[f32], f32, bool, &[usize]) -> crate::tensor::Result<Predictions> {
        move |scaled: &[f32], sigma: f32, guided: bool, _running: &[usize]| {
            let branch = |bias: f32| -> Vec<f32> {
                scaled
                    .chunks(plane)
                    .flat_map(|row| {
                        let mean = row.iter().sum::<f32>() / row.len() as f32;
                        row.iter()
                            .map(move |v| (v * 0.3 + mean * 0.05 + bias * sigma).sin())
                    })
                    .collect()
            };
            Ok((branch(0.1), guided.then(|| branch(-0.2))))
        }
    }

    /// A latent of `plane` values, distinct per `seed`.
    fn latent(seed: usize, plane: usize) -> Vec<f32> {
        (0..plane)
            .map(|i| (((i * 37 + seed * 101) % 89) as f32) * 0.05 - 2.2)
            .collect()
    }

    /// Two rows sampled together come out exactly as each comes out alone: the split of
    /// the stacked prediction, the guidance per row and the solver history per row.
    #[test]
    fn a_batch_samples_each_row_as_it_samples_alone() {
        let plane = LATENT_CHANNELS * 4 * 4;
        let sigmas = Schedule::new().sigma_ladder(6);
        let never = |_: usize| false;
        for sampler in [SamplerKind::DpmPP2M, SamplerKind::Euler] {
            let rows = [(latent(1, plane), 7.0f32), (latent(2, plane), 1.0)];
            let alone: Vec<Vec<f32>> = rows
                .iter()
                .map(|(x, g)| {
                    sample_rows(
                        vec![x.clone()],
                        &[*g],
                        &sigmas,
                        sampler,
                        row_denoiser(plane),
                        never,
                        &crate::inference::sample::no_progress,
                    )
                    .expect("alone")
                    .remove(0)
                    .expect("a row that ran")
                })
                .collect();
            let together = sample_rows(
                rows.iter().map(|(x, _)| x.clone()).collect(),
                &rows.iter().map(|(_, g)| *g).collect::<Vec<_>>(),
                &sigmas,
                sampler,
                row_denoiser(plane),
                never,
                &crate::inference::sample::no_progress,
            )
            .expect("together")
            .into_iter()
            .map(|x| x.expect("a row that ran"))
            .collect::<Vec<_>>();
            assert_eq!(together, alone, "a row's latent moved in the batch");
            assert_ne!(together[0], together[1], "the rows must stay distinct");
        }
    }

    /// A row whose request goes away leaves the pass and is answered `None`.
    /// The rows that stay come out exactly as they come out when no row leaves.
    #[test]
    fn a_row_that_leaves_the_pass_does_not_move_the_others() {
        let plane = LATENT_CHANNELS * 4 * 4;
        let sigmas = Schedule::new().sigma_ladder(6);
        let rows = [
            (latent(1, plane), 7.0f32),
            (latent(2, plane), 1.0),
            (latent(3, plane), 5.0),
        ];
        let run = |gone: &dyn Fn(usize, usize) -> bool| {
            let step = std::cell::Cell::new(0usize);
            let seen = std::cell::RefCell::new(Vec::new());
            let out = sample_rows(
                rows.iter().map(|(x, _)| x.clone()).collect(),
                &rows.iter().map(|(_, g)| *g).collect::<Vec<_>>(),
                &sigmas,
                SamplerKind::DpmPP2M,
                |scaled: &[f32], sigma: f32, guided: bool, running: &[usize]| {
                    seen.borrow_mut().push(running.to_vec());
                    row_denoiser(plane)(scaled, sigma, guided, running)
                },
                |r| gone(r, step.get()),
                &|done: usize, _| step.set(done),
            )
            .expect("a batch with rows left");
            (out, seen.into_inner())
        };
        let (whole, _) = run(&|_, _| false);
        let (out, seen) = run(&|r, step| r == 1 && step >= 2);
        assert_eq!(out[0], whole[0], "a row moved when another one left");
        assert_eq!(out[2], whole[2], "a row moved when another one left");
        assert!(out[1].is_none(), "the row that left has no latent");
        assert_eq!(seen[1], vec![0, 1, 2]);
        assert_eq!(
            seen[2],
            vec![0, 2],
            "the row that left is no longer computed"
        );
    }

    /// The pass stops with an error when no row is left to compute.
    #[test]
    fn a_batch_stops_when_no_row_is_left() {
        let plane = LATENT_CHANNELS;
        let sigmas = Schedule::new().sigma_ladder(4);
        let steps = std::cell::Cell::new(0usize);
        let counted = |_: usize, _: usize| steps.set(steps.get() + 1);
        // Both requests go away before the third step.
        let out = sample_rows(
            vec![latent(1, plane), latent(2, plane)],
            &[7.0, 7.0],
            &sigmas,
            SamplerKind::DpmPP2M,
            row_denoiser(plane),
            |_| steps.get() >= 3,
            &counted,
        );
        assert!(out.is_err());
        // The opening report and the two steps that ran before the stop.
        assert_eq!(steps.get(), 3);
    }

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
                &crate::inference::sample::no_progress,
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
                &crate::inference::sample::no_progress,
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
