//! Part of `impl ImageEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;
use crate::inference::sample::StepSink;

impl ImageEngine {
    /// Generate an image from a text prompt (non-streaming).
    /// Returns base64-encoded PNG.
    pub async fn generate_image(
        &self,
        prompt: &str,
        params: ImageGenParams,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        self.generate_image_seeded(prompt, params)
            .await
            .map(|r| r.png_base64)
    }

    /// Generate an image and report the seed it was drawn from, which is the caller's
    /// or a fresh one when the request named none (see [`ImageGenParams::resolve_seed`]).
    pub async fn generate_image_seeded(
        &self,
        prompt: &str,
        mut params: ImageGenParams,
    ) -> Result<RenderedImage, Box<dyn std::error::Error + Send + Sync>> {
        let model_state = self.model_state.clone();
        let prompt = prompt.to_string();
        let seed = params.resolve_seed();

        let result = tokio::task::spawn_blocking(move || -> AnyResult<String> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Image model not loaded"))?;
            render_on(
                state,
                &prompt,
                &params,
                seed,
                &crate::inference::sample::no_progress,
            )
        })
        .await?;

        result
            .map(|png_base64| RenderedImage { png_base64, seed })
            .map_err(boxed_engine_error)
    }

    /// Generate an image with streaming progress updates.
    ///
    /// A request with no seed renders from a fresh one; a caller that echoes the seed
    /// resolves it first with [`ImageGenParams::resolve_seed`].
    pub async fn generate_image_stream(
        &self,
        prompt: &str,
        mut params: ImageGenParams,
    ) -> Result<
        tokio::sync::mpsc::Receiver<ImageStreamEvent>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let model_state = self.model_state.clone();
        let presence = self.presence_cell();
        let prompt = prompt.to_string();
        let seed = params.resolve_seed();
        // Channel size sized to a comfortable HD-mode (Z-Image 18 steps,
        // Flux schnell HD 8 steps) plus a buffer for late receivers.
        // 64 is well above any sane num_steps and keeps the producer
        // from blocking when the SSE consumer is briefly slow.
        let (tx, rx) = tokio::sync::mpsc::channel::<ImageStreamEvent>(64);

        std::thread::spawn(move || {
            let mut guard = model_state.blocking_lock();
            let state = match guard.as_mut() {
                Some(s) => s,
                None => {
                    let _ =
                        tx.blocking_send(ImageStreamEvent::Error("Image model not loaded".into()));
                    return;
                }
            };
            super::remember(&presence, state);

            let progress = |completed, total| {
                let _ = tx.blocking_send(ImageStreamEvent::Progress { completed, total });
            };
            match render_on(state, &prompt, &params, seed, &progress) {
                Ok(base64_png) => {
                    let _ = tx.blocking_send(ImageStreamEvent::Complete {
                        image_base64: base64_png,
                    });
                }
                Err(e) => {
                    crate::inference::engine::log_engine_error("image", "generate", &e);
                    let _ = tx.blocking_send(ImageStreamEvent::Error(format!("{e:#}")));
                }
            }
        });

        Ok(rx)
    }
}

/// Render one image on the resident model: the one path both entry points share.
///
/// Every family draws its noise from `seed` and reports its denoise steps to `on_step`,
/// so a streaming caller sees the same progress whichever family is resident.
fn render_on(
    state: &mut LoadedImageModelState,
    prompt: &str,
    params: &ImageGenParams,
    seed: u64,
    on_step: StepSink,
) -> AnyResult<String> {
    MediaCapabilities::of(&state.model)
        .refuse_unsupported(family_of_model(&state.model), params)
        .map_err(anyhow::Error::new)?;
    state.device.set_seed(seed)?;

    let _precision = crate::inference::media::precision::GemmPrecision::pin(
        family_wants_reduced_precision(&state.model),
    );

    match &mut state.model {
        LoadedImageModel::Flux(flux_state) => generate_flux_image(
            flux_state,
            &state.device,
            state.dtype,
            prompt,
            params,
            on_step,
        ),
        LoadedImageModel::ZImage(zimg_state) => generate_zimage(
            &state.device,
            state.dtype,
            zimg_state,
            prompt,
            params,
            on_step,
        ),
        LoadedImageModel::Flux2(flux2_state) => crate::inference::engine::flux2_engine::generate(
            flux2_state,
            prompt,
            params.width,
            params.height,
            params.num_steps,
            params.guidance as f32,
            seed,
            &params.cancel,
            on_step,
        ),
        LoadedImageModel::QwenImage(qwen_state) => match &params.input_image {
            // Instruction edit: the loaded checkpoint IS the edit DiT, conditioned on the
            // source image.
            Some(src) => crate::inference::engine::qwen_image_engine::generate_edit(
                qwen_state,
                prompt,
                src,
                params.width,
                params.height,
                params.num_steps,
                params.guidance as f32,
                seed,
                &params.cancel,
                params.negative_prompt.as_deref(),
                on_step,
            ),
            None => crate::inference::engine::qwen_image_engine::generate(
                qwen_state,
                prompt,
                params.width,
                params.height,
                params.num_steps,
                params.guidance as f32,
                seed,
                &params.cancel,
                params.step_reuse,
                params.negative_prompt.as_deref(),
                on_step,
            ),
        },
        LoadedImageModel::Boogu(boogu_state) => crate::inference::engine::boogu_engine::generate(
            boogu_state,
            prompt,
            params.width,
            params.height,
            params.num_steps,
            params.guidance as f32,
            seed,
            &params.cancel,
            on_step,
        ),
        LoadedImageModel::Sdxl(pipe) => render_sdxl(pipe, prompt, params, seed, on_step),
    }
}

/// SDXL: attach this request's adapters, control image and solver, then sample.
///
/// Adapters and the control image are per REQUEST: whatever the last request left is
/// cleared before this one's is applied, so a later render is never styled or posed by
/// conditioning its caller did not ask for. The media gate runs one generation at a
/// time, so plain fields on the pipeline are enough.
fn render_sdxl(
    pipe: &mut crate::inference::model::sdxl::pipeline::SdxlPipeline,
    prompt: &str,
    params: &ImageGenParams,
    seed: u64,
    on_step: StepSink,
) -> AnyResult<String> {
    use crate::inference::model::sdxl::sampling::{SamplerKind, SchedulerKind};
    let n = pipe
        .set_loras(&params.loras)
        .map_err(|e| anyhow!("lora: {e}"))?;
    if !params.loras.is_empty() {
        info!("sdxl: {} lora(s) on {n} projections", params.loras.len());
    }
    pipe.set_control(params.control_image.as_deref(), params.control_scale)
        .map_err(|e| anyhow!("control image: {e}"))?;
    if params.control_image.is_some() {
        info!(
            "sdxl: structural conditioning at {:.2}",
            params.control_scale
        );
    }
    pipe.set_sampling(
        params
            .sampler
            .as_deref()
            .map(SamplerKind::parse)
            .unwrap_or_default(),
        params
            .scheduler
            .as_deref()
            .map(SchedulerKind::parse)
            .unwrap_or_default(),
    );
    let rgb = match params.input_image.as_deref() {
        // img2img: the source is resized to the render size, since the sampler works on
        // one latent grid.
        Some(src) => {
            let init = decode_base64_to_rgb8(src, params.width, params.height)?;
            pipe.generate_img2img(
                prompt,
                negative_of(params),
                &init,
                params.width,
                params.height,
                params.num_steps,
                params.guidance as f32,
                params.strength as f32,
                seed,
                Some(&params.cancel),
                on_step,
            )
            .map_err(|e| anyhow!("sdxl img2img: {e}"))?
        }
        None => pipe
            .generate(
                prompt,
                negative_of(params),
                params.width,
                params.height,
                params.num_steps,
                params.guidance as f32,
                seed,
                Some(&params.cancel),
                on_step,
            )
            .map_err(|e| anyhow!("sdxl generate: {e}"))?,
    };
    rgb_to_png_base64(rgb, params.width, params.height)
}
