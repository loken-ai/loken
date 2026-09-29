//! `/v1/images/edits` and `/v1/images/variations`: one multipart form, one set of checks
//! and one render loop for the two routes that take a source image.

use super::*;

/// The image model a request that names none renders with, on every image route.
pub(crate) const DEFAULT_IMAGE_MODEL: &str = "z-image-turbo";

/// Images one upload request may render.
const IMAGE_UPLOAD_N_MAX: u32 = 10;

/// Strength of an edit by default: the instruction may move the picture far.
const EDIT_STRENGTH: f64 = 0.75;
/// Strength of a variation by default: a near-replica with some diffuser variation.
const VARIATION_STRENGTH: f64 = 0.4;

/// An error answered by the image upload routes, in the OpenAI envelope.
fn form_error(code: axum::http::StatusCode, msg: impl Into<String>) -> axum::response::Response {
    use axum::response::IntoResponse;
    (code, Json(openai_error_body(code, msg.into()))).into_response()
}

/// An image upload goes to a node that holds the family on a card that fits it, the way a
/// generation does; otherwise its multipart reader is handed back to the route. The model
/// is the one the form names, else [`DEFAULT_IMAGE_MODEL`], the one the route renders
/// with.
async fn image_upload_where_the_model_is(
    state: &APIServer,
    headers: &axum::http::HeaderMap,
    body: axum::body::Bytes,
    path: &str,
) -> Result<(axum::extract::Multipart, String), axum::response::Response> {
    use crate::api::handlers::catalogue;
    let model_name = catalogue::upload_model(headers, &body)
        .await
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| DEFAULT_IMAGE_MODEL.to_string());
    let hot = match image_model_defaults(&state.huggingface_models_dir, &model_name) {
        Ok(d) => super::image_hot_bytes(
            state,
            &model_name,
            crate::inference::place::runtime_demand::RequestGeometry::new(d.size, d.size),
        ),
        Err(_) => 0,
    };
    if let Some(relayed) = crate::api::handlers::route_upload_to_holder(
        state,
        headers,
        &model_name,
        super::image_served_here(state, &model_name).await,
        state.card_holds(hot),
        hot,
        |peer| super::holds_image_model(peer, &model_name),
        path,
        &body,
    )
    .await
    {
        return Err(relayed);
    }
    let multipart = catalogue::upload_multipart(headers, body)
        .await
        .map_err(|e| form_error(axum::http::StatusCode::BAD_REQUEST, e))?;
    Ok((multipart, model_name))
}

/// The fields of an `/v1/images/edits` or `/v1/images/variations` multipart form, read
/// once for both routes. Unknown fields are drained and ignored: OpenAI's SDKs send some
/// this server has no use for.
pub(crate) struct ImageFormFields {
    pub image: Option<Vec<u8>>,
    /// The inpaint mask, kept as base64 so it travels with the other image inputs.
    pub mask_b64: Option<String>,
    pub prompt: Option<String>,
    /// What to steer away from. An empty one is a choice - steer away from nothing - and
    /// is kept as such rather than folded back into the default.
    pub negative_prompt: Option<String>,
    pub model: Option<String>,
    /// A region of the source to keep verbatim, as the caller wrote it (see
    /// [`parse_preserve`]).
    pub preserve: Option<String>,
    pub size: Option<String>,
    pub style: Option<String>,
    pub response_format: String,
    pub n: u32,
    pub strength: Option<f64>,
    pub seed: Option<u64>,
    pub num_steps: Option<u32>,
    pub guidance: Option<f64>,
    pub quality: Option<String>,
    /// Adapters, resolved by name inside the configured lora directory, never by path.
    pub loras: Vec<(String, f32)>,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub output_format: String,
    pub output_compression: u8,
}

impl Default for ImageFormFields {
    fn default() -> Self {
        Self {
            image: None,
            mask_b64: None,
            prompt: None,
            negative_prompt: None,
            model: None,
            preserve: None,
            size: None,
            style: None,
            response_format: "url".to_string(),
            n: 1,
            strength: None,
            seed: None,
            num_steps: None,
            guidance: None,
            quality: None,
            loras: Vec::new(),
            sampler: None,
            scheduler: None,
            output_format: "png".to_string(),
            output_compression: 85,
        }
    }
}

/// A form field's text, `None` when it is blank or unreadable.
async fn nonblank(field: axum::extract::multipart::Field<'_>) -> Option<String> {
    field.text().await.ok().filter(|s| !s.trim().is_empty())
}

/// A form field's text parsed as `T`, `None` when it does not parse.
async fn parsed<T: std::str::FromStr>(field: axum::extract::multipart::Field<'_>) -> Option<T> {
    field.text().await.ok()?.trim().parse().ok()
}

/// What both upload routes settle before rendering: the source image, the model, the
/// output size, the step count, the guidance and the seed.
pub(crate) struct ImageUploadJob {
    pub image_bytes: Vec<u8>,
    pub model_name: String,
    pub width: usize,
    pub height: usize,
    pub geom: crate::inference::place::runtime_demand::RequestGeometry,
    pub steps: usize,
    pub guidance: f64,
    pub seed: u64,
    /// A FLUX.1 checkpoint whose pipeline edits by instruction: the source is an
    /// instruction-edit context, not an img2img init latent.
    pub kontext: bool,
}

impl ImageFormFields {
    pub(crate) async fn read(
        multipart: &mut axum::extract::Multipart,
    ) -> Result<Self, axum::response::Response> {
        use axum::http::StatusCode;
        use base64::Engine as _;
        let mut f = Self::default();
        loop {
            let field = match multipart.next_field().await {
                Ok(Some(field)) => field,
                Ok(None) => break,
                Err(e) => {
                    return Err(form_error(
                        StatusCode::BAD_REQUEST,
                        format!("multipart parse: {e}"),
                    ))
                }
            };
            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                "image" => match field.bytes().await {
                    Ok(b) => f.image = Some(b.to_vec()),
                    Err(e) => {
                        return Err(form_error(
                            StatusCode::BAD_REQUEST,
                            format!("read image: {e}"),
                        ))
                    }
                },
                // A read failure means the caller sent a mask the body could not carry:
                // a 400, never a silent "there was no mask", which would return an
                // unmasked edit the client could not detect. A 0-byte part is how some
                // SDKs attach the field by default.
                "mask" => match field.bytes().await {
                    Ok(b) if b.is_empty() => {}
                    Ok(b) => {
                        if let Err(e) = validate_image_input_size(b.len()) {
                            return Err(form_error(
                                StatusCode::PAYLOAD_TOO_LARGE,
                                format!("mask: {e}"),
                            ));
                        }
                        f.mask_b64 = Some(base64::engine::general_purpose::STANDARD.encode(&b));
                    }
                    Err(e) => {
                        return Err(form_error(
                            StatusCode::BAD_REQUEST,
                            format!("read `mask` field: {e}"),
                        ))
                    }
                },
                "negative_prompt" | "negative" => f.negative_prompt = field.text().await.ok(),
                "prompt" => f.prompt = nonblank(field).await,
                "model" => f.model = nonblank(field).await,
                "preserve" => f.preserve = nonblank(field).await,
                "size" => f.size = nonblank(field).await,
                "style" => f.style = field.text().await.ok(),
                "response_format" => {
                    if let Ok(s) = field.text().await {
                        f.response_format = s.to_lowercase();
                    }
                }
                "n" => {
                    if let Some(v) = parsed(field).await {
                        f.n = v;
                    }
                }
                // The shapes the JSON endpoints take: a bare name, a list of names, or
                // {name, strength} objects. A form field carries text, so a JSON array is
                // parsed when it looks like one and taken as a single name otherwise.
                "loras" => {
                    if let Ok(t) = field.text().await {
                        let v = serde_json::from_str::<serde_json::Value>(&t)
                            .unwrap_or(serde_json::Value::String(t));
                        f.loras = parse_loras(Some(&v))
                            .map_err(|e| form_error(StatusCode::BAD_REQUEST, e))?;
                    }
                }
                "strength" => f.strength = parsed(field).await,
                "seed" => f.seed = parsed(field).await,
                "num_steps" => f.num_steps = parsed(field).await,
                "guidance" => f.guidance = parsed(field).await,
                "sampler" => f.sampler = nonblank(field).await,
                "scheduler" => f.scheduler = nonblank(field).await,
                "quality" => f.quality = field.text().await.ok(),
                "output_format" => {
                    if let Ok(s) = field.text().await {
                        f.output_format = s.to_lowercase();
                    }
                }
                "output_compression" => {
                    if let Some(v) = parsed(field).await {
                        f.output_compression = v;
                    }
                }
                _ => {
                    let _ = field.bytes().await;
                }
            }
        }
        Ok(f)
    }

    /// Validate the form and load the model, with the checks both routes apply.
    pub(crate) async fn job(
        &self,
        state: &APIServer,
    ) -> Result<ImageUploadJob, axum::response::Response> {
        use axum::http::StatusCode;
        use axum::response::IntoResponse;
        let image_bytes = match self.image.as_ref() {
            Some(b) if !b.is_empty() => b.clone(),
            _ => return Err(form_error(StatusCode::BAD_REQUEST, "missing `image` field")),
        };
        validate_image_input_size(image_bytes.len())
            .map_err(|e| form_error(StatusCode::PAYLOAD_TOO_LARGE, e))?;
        validate_image_response_format(&self.response_format)
            .map_err(|e| form_error(StatusCode::BAD_REQUEST, e))?;
        validate_image_output_format(&self.output_format)
            .map_err(|e| form_error(StatusCode::BAD_REQUEST, e))?;
        validate_image_quality(self.quality.as_deref())
            .map_err(|e| form_error(StatusCode::BAD_REQUEST, e))?;
        if self.n == 0 {
            return Err(form_error(StatusCode::BAD_REQUEST, "n must be >= 1"));
        }
        if self.n > IMAGE_UPLOAD_N_MAX {
            return Err(form_error(
                StatusCode::BAD_REQUEST,
                format!("n must be <= {IMAGE_UPLOAD_N_MAX}"),
            ));
        }
        if let Some(n) = self.num_steps {
            validate_image_num_steps(n).map_err(|e| form_error(StatusCode::BAD_REQUEST, e))?;
        }
        let model_name = self
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_IMAGE_MODEL.to_string());
        validate_model_id(&model_name).map_err(IntoResponse::into_response)?;
        let defaults = image_model_defaults(&state.huggingface_models_dir, &model_name)
            .map_err(|e| form_error(StatusCode::BAD_REQUEST, e))?;
        let (width, height) = match self.size.as_deref() {
            // A size the caller named still has to be one the model can halve evenly:
            // snapped, and said out loud, rather than a 500 raised deep inside the U-Net.
            Some(s) => match parse_image_size(s) {
                Ok((w, h)) => align_to(w, h, defaults.align, "size"),
                Err(e) => return Err(form_error(StatusCode::BAD_REQUEST, format!("size: {e}"))),
            },
            // No explicit size: follow the source image, not a square default.
            None => edit_output_size(&image_bytes, defaults.size, defaults.align),
        };
        let kontext = is_flux_instruction_editor(&state.huggingface_models_dir, &model_name);
        let (default_steps, default_guidance) = (defaults.steps, defaults.guidance);
        // Bound once: the placement reserves for this geometry, and a render that runs out
        // anyway re-plans against the same one.
        let geom = crate::inference::place::runtime_demand::RequestGeometry::new(width, height);
        if let Err(e) = ensure_image_model_loaded(state, &model_name, geom).await {
            return Err(form_error(http_status_for_load_error(&e), e));
        }
        let regime = regime_for(&self.loras, self.num_steps, self.guidance);
        let default_steps = regime.map(|r| r.steps).unwrap_or(default_steps);
        let default_guidance = regime.map(|r| r.guidance).unwrap_or(default_guidance);
        let steps = match self.num_steps {
            Some(n) => n as usize,
            None if is_hd_quality(self.quality.as_deref()) => default_steps * 2,
            None => default_steps,
        };
        Ok(ImageUploadJob {
            image_bytes,
            model_name,
            width,
            height,
            geom,
            steps,
            guidance: clamp_finite_f64(
                self.guidance.unwrap_or(default_guidance),
                0.0,
                30.0,
                default_guidance,
            ),
            seed: self.seed.unwrap_or_else(rand_u64),
            kontext,
        })
    }

    /// The render parameters every image of the job shares. `default_strength` is how far
    /// the route lets a render move from its source.
    pub(crate) fn params(
        &self,
        job: &ImageUploadJob,
        cancel: crate::inference::serve::cancel::CancelToken,
        default_strength: f64,
    ) -> ImageGenParams {
        use base64::Engine as _;
        ImageGenParams {
            // Regions compose a canvas from noise; an upload already HAS its composition,
            // and the source image decides where things are.
            regions: Vec::new(),
            // Every step is kept: a render from a real image shows an approximated step
            // far more than one from noise.
            step_reuse: 0.0,
            negative_prompt: self.negative_prompt.clone(),
            mask: self.mask_b64.clone(),
            control_image: None,
            control_scale: 1.0,
            cancel,
            width: job.width,
            height: job.height,
            num_steps: job.steps,
            guidance: job.guidance,
            seed: Some(job.seed),
            input_image: Some(base64::engine::general_purpose::STANDARD.encode(&job.image_bytes)),
            strength: clamp_finite_f64(
                self.strength.unwrap_or(default_strength),
                0.0,
                1.0,
                default_strength,
            ),
            kontext: job.kontext,
            sampler: self.sampler.clone(),
            scheduler: self.scheduler.clone(),
            loras: self.loras.clone(),
        }
    }
}

/// Run one render with the request's cancel guard armed across it: a client that goes
/// away mid-render drops the guard and stops the render at its next step; a render that
/// finishes disarms it and leaves the token clear for the next image of the batch.
pub(crate) async fn guarded_render<T>(
    cancel: &crate::inference::serve::cancel::CancelToken,
    render: impl std::future::Future<Output = T>,
) -> T {
    let guard = crate::inference::serve::cancel::CancelGuard::new(cancel.clone());
    let out = render.await;
    guard.disarm();
    out
}

/// Render the `n` images of an upload job, each seeded one past the last, pass each
/// through `finish`, and answer them in one response. One image failing does not fail
/// the batch: only a batch where nothing rendered is a failed request.
async fn render_upload_batch(
    state: &APIServer,
    headers: &axum::http::HeaderMap,
    form: &ImageFormFields,
    job: &ImageUploadJob,
    prompt: &str,
    base: ImageGenParams,
    finish: impl Fn(String) -> Result<String, String>,
) -> axum::response::Response {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    let generate_start = std::time::Instant::now();
    let bounces_before = crate::tensor::bounce::pressure_bounces();
    let mut data = Vec::with_capacity(form.n as usize);
    let mut failures: Vec<String> = Vec::new();
    for i in 0..form.n {
        let image_seed = job.seed.wrapping_add(u64::from(i));
        let params = ImageGenParams {
            seed: Some(image_seed),
            ..base.clone()
        };
        let cancel = params.cancel.clone();
        let rendered = guarded_render(
            &cancel,
            generate_image_resilient(state, &job.model_name, job.geom, prompt, params),
        )
        .await;
        match rendered {
            Ok(rendered) => match finish(rendered.png_base64) {
                Ok(b64) => data.push(image_response_entry(
                    b64,
                    &form.output_format,
                    form.output_compression,
                    Some(rendered.seed),
                )),
                Err(e) => return form_error(StatusCode::INTERNAL_SERVER_ERROR, e),
            },
            Err(e) => {
                if let Some(refusal) = refused_parameter(e.as_ref()) {
                    return refusal.into_response();
                }
                failures.push(format!("image {}: {e}", i + 1));
            }
        }
    }
    if data.is_empty() && !failures.is_empty() {
        // Said in the log as well as the response: the client has often dropped the
        // connection the body travels on.
        if all_cancelled(&failures) {
            info!("image: the client stopped the render before it finished");
            return form_error(
                StatusCode::from_u16(CLIENT_CLOSED_REQUEST).unwrap_or(StatusCode::BAD_REQUEST),
                "the render was cancelled",
            );
        }
        error!("image: every image failed - {}", failures.join("; "));
        return form_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("every image failed - {}", failures.join("; ")),
        );
    }
    if !failures.is_empty() {
        warn!(
            "image batch: {} of {} failed - {}",
            failures.len(),
            form.n,
            failures.join("; ")
        );
    }
    let generate_ms = generate_start.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Image edits/variations: model={} {}x{} n={} generate={generate_ms:.0}ms",
        job.model_name, job.width, job.height, form.n
    );
    // A pressure bounce is an op that could not allocate on its device and ran on the
    // host: correct, but a cliff worth reporting with the render.
    let bounced = crate::tensor::bounce::pressure_bounces() - bounces_before;
    if bounced > 0 {
        tracing::warn!(
            "{bounced} op(s) ran on the host under memory pressure during this render - \
             the placement did not leave enough room on the device"
        );
    }
    let data = match super::super::openai::images_as_urls(
        state,
        headers,
        form.response_format == "url",
        data,
    ) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let body = Json(serde_json::json!({
        "created": chrono::Utc::now().timestamp(),
        "data": data,
        "render_ms": generate_ms as u64,
    }));
    (image_response_headers(generate_ms), body).into_response()
}

/// OpenAI-compatible `/v1/images/edits`. Multipart form: `image` (PNG/JPEG bytes),
/// `prompt`, optional `model`, `n`, `size`, `response_format`, `mask`, plus our
/// extensions `strength` (0..1), `seed` (u64) and `preserve` (a region kept verbatim).
pub(crate) async fn images_edits(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> axum::response::Response {
    use axum::http::StatusCode;
    // The job goes where the family is held and a card holds it whole.
    let (mut multipart, upload_model) =
        match image_upload_where_the_model_is(&state, &headers, body, "/v1/images/edits").await {
            Ok(parsed) => parsed,
            Err(answer) => return answer,
        };
    // One media job at a time: two diffusion engines cannot share these cards.
    let _media_guard = state.media_lock_for(&upload_model, "image edit").await;
    let form = match ImageFormFields::read(&mut multipart).await {
        Ok(f) => f,
        Err(answer) => return answer,
    };
    let prompt = match form
        .prompt
        .clone()
        .map(|p| super::super::openai::with_style(p, form.style.as_deref()))
    {
        Some(p) if !p.trim().is_empty() => p,
        _ => return form_error(StatusCode::BAD_REQUEST, "missing or empty `prompt`"),
    };
    if prompt.chars().count() > IMAGE_PROMPT_MAX_CHARS {
        return form_error(
            StatusCode::BAD_REQUEST,
            format!(
                "prompt is {} chars; cap at {IMAGE_PROMPT_MAX_CHARS}",
                prompt.chars().count()
            ),
        );
    }
    let job = match form.job(&state).await {
        Ok(j) => j,
        Err(answer) => return answer,
    };
    let render_cancel = crate::inference::serve::cancel::CancelToken::new();
    let base = form.params(&job, render_cancel, EDIT_STRENGTH);

    // An edit re-renders the whole frame, so anything the instruction did not ask to
    // change can still drift. On its own this endpoint composites back a rectangle the
    // caller names; a registered assist can find the rectangle itself, say what the
    // picture shows, and put the region back accounting for the render having moved.
    let assist = crate::api::assist::edit_assist();
    let mut preserve: Option<(PreserveRegion, crate::api::assist::Chosen)> = None;
    let mut unresolved_preserve: Option<&str> = None;
    if let Some(spec) = form.preserve.as_deref() {
        match parse_preserve(spec) {
            Ok(r) => preserve = Some((r, crate::api::assist::Chosen::ByCaller)),
            Err(_) => unresolved_preserve = Some(spec),
        }
    }
    // Decoded once, and only when something is going to look at it.
    let source_rgb: Option<image::RgbImage> =
        (preserve.is_some() || unresolved_preserve.is_some() || assist.is_some())
            .then(|| {
                crate::inference::media::image_processor::decode_image_oriented(&job.image_bytes)
                    .ok()
                    .map(|i| i.to_rgb8())
            })
            .flatten();
    if let (Some(spec), Some(a), Some(src)) =
        (unresolved_preserve, assist.as_ref(), source_rgb.as_ref())
    {
        // Not a rectangle: an assist may read it; an unusable region protects nothing.
        if let Ok(r) = a.resolve_spec(spec, src) {
            preserve = Some((r, crate::api::assist::Chosen::ByAssist));
        }
    }
    // What is in the picture, folded into an instruction that names no subject. Never
    // blocks a render: with no assist the instruction goes through as written.
    let prompt = match (assist.as_ref(), source_rgb.as_ref()) {
        (Some(a), Some(src)) => match a.subject_clause(src) {
            Some(clause) => {
                let merged = format!("{clause}. {prompt}");
                info!("edit: the source shows {clause}; instruction now {merged:?}");
                merged
            }
            None => prompt,
        },
        _ => prompt,
    };
    if preserve.is_some() && source_rgb.is_none() {
        return form_error(
            StatusCode::BAD_REQUEST,
            "`preserve` was given but the source image could not be decoded",
        );
    }
    // The protected region is put back before the response is built, so the caller never
    // sees an intermediate where it drifted.
    let finish = |b64: String| match (preserve, source_rgb.as_ref()) {
        (Some((region, chosen)), Some(src)) => match assist.as_ref() {
            Some(a) => a.restore(&b64, src, region, chosen),
            None => preserve_in_b64(&b64, src, region),
        },
        _ => Ok(b64),
    };
    render_upload_batch(&state, &headers, &form, &job, &prompt, base, finish).await
}

/// OpenAI-compatible `/v1/images/variations`. Multipart form: `image` (PNG/JPEG bytes)
/// and the edit route's optional fields but `prompt`, `mask` and `preserve`: an img2img
/// run with an empty prompt and a lower default strength, so the output stays close to
/// the input.
pub(crate) async fn images_variations(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> axum::response::Response {
    // The job goes where the family is held and a card holds it whole.
    let (mut multipart, upload_model) = match image_upload_where_the_model_is(
        &state,
        &headers,
        body,
        "/v1/images/variations",
    )
    .await
    {
        Ok(parsed) => parsed,
        Err(answer) => return answer,
    };
    // One media job at a time: two diffusion engines cannot share these cards.
    let _media_guard = state.media_lock_for(&upload_model, "image variation").await;
    let mut form = match ImageFormFields::read(&mut multipart).await {
        Ok(f) => f,
        Err(answer) => return answer,
    };
    // A variation takes no prompt, and neither a mask nor a kept region applies to a
    // picture re-rendered whole.
    form.mask_b64 = None;
    form.preserve = None;
    let job = match form.job(&state).await {
        Ok(j) => ImageUploadJob {
            kontext: false,
            ..j
        },
        Err(answer) => return answer,
    };
    let render_cancel = crate::inference::serve::cancel::CancelToken::new();
    let base = form.params(&job, render_cancel, VARIATION_STRENGTH);
    render_upload_batch(&state, &headers, &form, &job, "", base, Ok).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::serve::cancel::CancelToken;
    use std::future::Future;

    /// Every image of a batch starts with a clear token: a finished render leaves no
    /// cancellation behind for the next one.
    #[tokio::test]
    async fn a_finished_render_does_not_cancel_the_next() {
        let token = CancelToken::new();
        for _ in 0..3 {
            let seen = guarded_render(&token, async { token.is_cancelled() }).await;
            assert!(!seen, "a render of the batch started cancelled");
        }
        assert!(!token.is_cancelled());
    }

    /// A second guard over the same token, left armed, cancels it when its scope ends:
    /// the reason a batch arms exactly one guard per render.
    #[test]
    fn a_second_armed_guard_cancels_the_batch() {
        use crate::inference::serve::cancel::CancelGuard;
        let token = CancelToken::new();
        {
            let _forgotten = CancelGuard::new(token.clone());
            let guard = CancelGuard::new(token.clone());
            guard.disarm();
        }
        assert!(token.is_cancelled());
    }

    /// A render whose client goes away mid-way is cancelled.
    #[tokio::test]
    async fn a_dropped_render_is_cancelled() {
        let token = CancelToken::new();
        {
            let render = guarded_render(&token, std::future::pending::<()>());
            let mut render = std::pin::pin!(render);
            let waker = futures::task::noop_waker();
            let mut cx = std::task::Context::from_waker(&waker);
            assert!(render.as_mut().poll(&mut cx).is_pending());
        }
        assert!(token.is_cancelled());
    }

    /// Both upload routes and the upload router fall back to one model.
    #[test]
    fn one_default_model_for_uploads() {
        let store = crate::api::handlers::media::tests::image_store("upload-default");
        assert!(image_model_defaults(&store.dir, DEFAULT_IMAGE_MODEL).is_ok());
        let f = ImageFormFields::default();
        assert!(f.model.is_none());
        assert_eq!(f.n, 1);
        assert_eq!(f.response_format, "url");
        assert_eq!(f.output_format, "png");
    }
}
