//! Part of `impl ImageEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl ImageEngine {
    /// Generate an image from a text prompt (non-streaming).
    /// Returns base64-encoded PNG.
    pub async fn generate_image(
        &self,
        prompt: &str,
        params: ImageGenParams,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let model_state = self.model_state.clone();
        let prompt = prompt.to_string();

        let result = tokio::task::spawn_blocking(move || -> AnyResult<String> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Image model not loaded"))?;

            if let Some(seed) = params.seed {
                state.device.set_seed(seed)?;
            }

            set_render_precision(&state.device, family_wants_reduced_precision(&state.model));

            match &mut state.model {
                LoadedImageModel::Flux(flux_state) => {
                    generate_flux_image(flux_state, &state.device, state.dtype, &prompt, &params)
                }
                LoadedImageModel::ZImage(zimg_state) => generate_zimage(
                    &state.device,
                    state.dtype,
                    zimg_state,
                    &prompt,
                    &params,
                    None,
                ),
                LoadedImageModel::Flux2(flux2_state) => {
                    if params.input_image.is_some() {
                        Err(anyhow::anyhow!(
                            "flux2 has no img2img or instruction-edit path wired yet - use an \
                             image model with img2img (flux, z-image, sdxl) or an instruction \
                             editor (flux-kontext, qwen-image-edit)"
                        ))
                    } else {
                        crate::inference::engine::flux2_engine::generate(
                            flux2_state,
                            &prompt,
                            params.width,
                            params.height,
                            params.num_steps,
                            params.guidance as f32,
                            params.seed.unwrap_or(42),
                            &params.cancel,
                        )
                    }
                }
                LoadedImageModel::QwenImage(qwen_state) => match &params.input_image {
                    Some(src) => crate::inference::engine::qwen_image_engine::generate_edit(
                        qwen_state,
                        &prompt,
                        src,
                        params.width,
                        params.height,
                        params.num_steps,
                        params.guidance as f32,
                        params.seed.unwrap_or(42),
                        &params.cancel,
                        params.negative_prompt.as_deref(),
                    ),
                    None => crate::inference::engine::qwen_image_engine::generate(
                        qwen_state,
                        &prompt,
                        params.width,
                        params.height,
                        params.num_steps,
                        params.guidance as f32,
                        params.seed.unwrap_or(42),
                        &params.cancel,
                        params.step_reuse,
                        params.negative_prompt.as_deref(),
                    ),
                },
                LoadedImageModel::Boogu(boogu_state) => {
                    if params.input_image.is_some() {
                        Err(anyhow::anyhow!(
                            "boogu does not support image editing/variations yet - use an \
                             image model with img2img (flux, z-image) or an instruction \
                             editor (flux-kontext, qwen-image-edit)"
                        ))
                    } else {
                        crate::inference::engine::boogu_engine::generate(
                            boogu_state,
                            &prompt,
                            params.width,
                            params.height,
                            params.num_steps,
                            params.guidance as f32,
                            params.seed.unwrap_or(42),
                            &params.cancel,
                        )
                    }
                }
                LoadedImageModel::Sdxl(pipe) => {
                    // Per request, under the media gate (one generation at a time), so a
                    // plain field on the pipeline is enough.
                    use crate::inference::model::sdxl::sampling::{SamplerKind, SchedulerKind};
                    // Adapters are per REQUEST: clear whatever the last one left, then
                    // apply this one's. Leaving them attached would silently style every
                    // later render with a LoRA the caller did not ask for.
                    match pipe.set_loras(&params.loras) {
                        Ok(n) if !params.loras.is_empty() => {
                            info!("sdxl: {} lora(s) on {n} projections", params.loras.len())
                        }
                        Ok(_) => {}
                        Err(e) => return Err(anyhow::anyhow!("lora: {e}")),
                    }
                    // Structural conditioning, per request like the adapters: a control
                    // image left attached would silently pose every later render.
                    pipe.set_control(params.control_image.as_deref(), params.control_scale)
                        .map_err(|e| anyhow::anyhow!("control image: {e}"))?;
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
                    if let Some(src) = params.input_image.as_deref() {
                        // img2img: the source is resized to the render size, since the
                        // sampler works on one latent grid.
                        let init = decode_base64_to_rgb8(src, params.width, params.height)?;
                        let rgb = pipe
                            .generate_img2img(
                                &prompt,
                                negative_of(&params),
                                &init,
                                params.width,
                                params.height,
                                params.num_steps,
                                params.guidance as f32,
                                params.strength as f32,
                                params.seed.unwrap_or(42),
                                Some(&params.cancel),
                            )
                            .map_err(|e| anyhow::anyhow!("sdxl img2img: {e}"))?;
                        rgb_to_png_base64(&rgb, params.width, params.height)
                    } else {
                        let rgb = pipe
                            .generate(
                                &prompt,
                                negative_of(&params),
                                params.width,
                                params.height,
                                params.num_steps,
                                params.guidance as f32,
                                params.seed.unwrap_or(42),
                                Some(&params.cancel),
                            )
                            .map_err(|e| anyhow::anyhow!("sdxl generate: {e}"))?;
                        rgb_to_png_base64(&rgb, params.width, params.height)
                    }
                }
            }
        })
        .await?;

        result.map_err(std::convert::Into::into)
    }

    /// Generate an image with streaming progress updates.
    pub async fn generate_image_stream(
        &self,
        prompt: &str,
        params: ImageGenParams,
    ) -> Result<
        tokio::sync::mpsc::Receiver<ImageStreamEvent>,
        Box<dyn std::error::Error + Send + Sync>,
    > {
        let model_state = self.model_state.clone();
        let prompt = prompt.to_string();
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

            if let Some(seed) = params.seed {
                if let Err(e) = state.device.set_seed(seed) {
                    let _ = tx.blocking_send(ImageStreamEvent::Error(format!("Set seed: {e}")));
                    return;
                }
            }

            set_render_precision(&state.device, family_wants_reduced_precision(&state.model));

            let result = match &mut state.model {
                LoadedImageModel::Flux(flux_state) => generate_flux_image_stream(
                    flux_state,
                    &state.device,
                    state.dtype,
                    &prompt,
                    &params,
                    &tx,
                ),
                LoadedImageModel::ZImage(zimg_state) => generate_zimage(
                    &state.device,
                    state.dtype,
                    zimg_state,
                    &prompt,
                    &params,
                    Some(&tx),
                ),
                LoadedImageModel::Flux2(flux2_state) => {
                    if params.input_image.is_some() {
                        Err(anyhow::anyhow!(
                            "flux2 has no img2img or instruction-edit path wired yet - use an \
                             image model with img2img (flux, z-image, sdxl) or an instruction \
                             editor (flux-kontext, qwen-image-edit)"
                        ))
                    } else {
                        crate::inference::engine::flux2_engine::generate(
                            flux2_state,
                            &prompt,
                            params.width,
                            params.height,
                            params.num_steps,
                            params.guidance as f32,
                            params.seed.unwrap_or(42),
                            &params.cancel,
                        )
                    }
                }
                LoadedImageModel::QwenImage(qwen_state) => {
                    let _ = tx.blocking_send(ImageStreamEvent::Progress {
                        completed: 0,
                        total: params.num_steps,
                    });
                    match &params.input_image {
                        // Instruction edit: the loaded checkpoint IS the edit
                        // DiT - condition on the source image (it used to be
                        // silently dropped, degrading edits to txt2img).
                        Some(src) => crate::inference::engine::qwen_image_engine::generate_edit(
                            qwen_state,
                            &prompt,
                            src,
                            params.width,
                            params.height,
                            params.num_steps,
                            params.guidance as f32,
                            params.seed.unwrap_or(42),
                            &params.cancel,
                            params.negative_prompt.as_deref(),
                        ),
                        None => crate::inference::engine::qwen_image_engine::generate(
                            qwen_state,
                            &prompt,
                            params.width,
                            params.height,
                            params.num_steps,
                            params.guidance as f32,
                            params.seed.unwrap_or(42),
                            &params.cancel,
                            params.step_reuse,
                            params.negative_prompt.as_deref(),
                        ),
                    }
                }
                LoadedImageModel::Boogu(boogu_state) => {
                    let _ = tx.blocking_send(ImageStreamEvent::Progress {
                        completed: 0,
                        total: params.num_steps,
                    });
                    if params.input_image.is_some() {
                        // Honest failure beats a silent txt2img that discards
                        // the user's source image.
                        Err(anyhow::anyhow!(
                            "boogu does not support image editing/variations yet - use an \
                             image model with img2img (flux, z-image) or an instruction \
                             editor (flux-kontext, qwen-image-edit)"
                        ))
                    } else {
                        crate::inference::engine::boogu_engine::generate(
                            boogu_state,
                            &prompt,
                            params.width,
                            params.height,
                            params.num_steps,
                            params.guidance as f32,
                            params.seed.unwrap_or(42),
                            &params.cancel,
                        )
                    }
                }
                LoadedImageModel::Sdxl(pipe) => {
                    // Per request, under the media gate (one generation at a time), so a
                    // plain field on the pipeline is enough.
                    use crate::inference::model::sdxl::sampling::{SamplerKind, SchedulerKind};
                    // Adapters are per REQUEST: clear whatever the last one left, then
                    // apply this one's. Leaving them attached would silently style every
                    // later render with a LoRA the caller did not ask for.
                    let lora_err = match pipe.set_loras(&params.loras) {
                        Ok(n) => {
                            if !params.loras.is_empty() {
                                info!("sdxl: {} lora(s) on {n} projections", params.loras.len());
                            }
                            None
                        }
                        Err(e) => Some(format!("lora: {e}")),
                    };
                    // Report through the stream rather than rendering without the
                    // adapter: a caller who asked for a LoRA and silently got the base
                    // model has no way to tell, and would blame the adapter.
                    if let Some(e) = lora_err {
                        let _ = tx.blocking_send(ImageStreamEvent::Error(e));
                        return;
                    }
                    // Same contract on the streaming path: reported, never skipped.
                    if let Err(e) =
                        pipe.set_control(params.control_image.as_deref(), params.control_scale)
                    {
                        let _ = tx
                            .blocking_send(ImageStreamEvent::Error(format!("control image: {e}")));
                        return;
                    }
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
                    if let Some(src) = params.input_image.as_deref() {
                        decode_base64_to_rgb8(src, params.width, params.height)
                            .and_then(|init| {
                                pipe.generate_img2img(
                                    &prompt,
                                    negative_of(&params),
                                    &init,
                                    params.width,
                                    params.height,
                                    params.num_steps,
                                    params.guidance as f32,
                                    params.strength as f32,
                                    params.seed.unwrap_or(42),
                                    Some(&params.cancel),
                                )
                                .map_err(|e| anyhow::anyhow!("sdxl img2img: {e}"))
                            })
                            .and_then(|rgb| rgb_to_png_base64(&rgb, params.width, params.height))
                    } else {
                        pipe.generate(
                            &prompt,
                            negative_of(&params),
                            params.width,
                            params.height,
                            params.num_steps,
                            params.guidance as f32,
                            params.seed.unwrap_or(42),
                            Some(&params.cancel),
                        )
                        .map_err(|e| anyhow::anyhow!("sdxl generate: {e}"))
                        .and_then(|rgb| rgb_to_png_base64(&rgb, params.width, params.height))
                    }
                }
            };

            match result {
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
