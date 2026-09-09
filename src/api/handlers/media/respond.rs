//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Build the response headers shared across all three /v1/images/*
/// endpoints (generations, edits, variations). One helper keeps the
/// three handlers in lockstep - previously each had its own inline
/// header build and was trivial to drift if one endpoint grew a new
/// header.
///
/// `Server-Timing: generate;dur=<ms>` lets browser DevTools surface
/// per-request generation latency without parsing the response body.
pub(super) fn image_response_headers(generate_ms: f64) -> axum::http::HeaderMap {
    let mut headers = axum::http::HeaderMap::new();
    let st = format!("generate;dur={generate_ms:.1}");
    if let Ok(hv) = axum::http::HeaderValue::from_str(&st) {
        headers.insert("server-timing", hv);
    }
    headers
}

/// Up-front validator for `output_format` so unknown values return a
/// clean 400 instead of silently falling back to PNG inside the
/// transcoder (which would mislead clients reading the response
/// Content-Type / file extension).
pub(super) fn validate_image_output_format(raw: &str) -> Result<(), String> {
    let fmt = raw.trim().to_ascii_lowercase();
    if fmt.is_empty() {
        return Ok(());
    }
    match fmt.as_str() {
        "png" | "jpeg" | "jpg" | "webp" => Ok(()),
        other => Err(format!(
            "output_format '{other}' not supported; use 'png'|'jpeg'|'webp'"
        )),
    }
}

/// Boundary check for `response_format` on the /v1/images/* endpoints.
/// Both `"url"` and `"b64_json"` are documented by OpenAI; everything
/// else gets a 400 instead of silent acceptance. The actual response
/// always carries `b64_json` regardless of the validated value - see
/// the `image_response_entry` docstring for the privacy rationale -
/// so this helper is the only place the input value is read.
///
/// Centralised so the three image endpoints (generations / edits /
/// variations) can't drift on the accepted set.
pub(super) fn validate_image_response_format(raw: &str) -> Result<(), String> {
    let rf = raw.trim().to_ascii_lowercase();
    if rf == "url" || rf == "b64_json" {
        Ok(())
    } else {
        Err(format!(
            "response_format '{rf}' not supported; use 'url' or 'b64_json'"
        ))
    }
}

/// Max accepted num_steps on the /v1/images/* and chat image-gen
/// paths. Past this the engine would happily loop denoising and
/// hold the GPU for minutes per request - a trivial DoS vector with
/// no benefit (Flux Schnell tops out around 8, Z-Image Turbo around
/// 20; even traditional SD checkpoints rarely exceed 100). 200
/// leaves headroom for distilled checkpoints we don't yet ship but
/// keeps `num_steps=100000` from hanging the server.
pub(crate) const IMAGE_NUM_STEPS_HARD_CAP: u32 = 200;

/// Max accepted bytes on the multipart `image` field of
/// /v1/images/edits + /v1/images/variations.
///
/// This is a backstop against a malformed or runaway body, NOT a judgement about what
/// a caller should be allowed to send. It used to sit at OpenAI's documented 25 MB,
/// which a single photograph off a current camera already exceeds - the server refused
/// ordinary source material and called it misuse. The old comment also justified the
/// cap by the VAE encoder's VRAM budget; that is the placement cascade's job, and
/// deciding it here by counting bytes of PNG was never going to be right.
pub(crate) const IMAGE_INPUT_MAX_BYTES: usize = 512 * 1024 * 1024;

/// Boundary check for caller-uploaded image bytes (the multipart
/// `image` field on /v1/images/edits + /variations, and the
/// base64-encoded `images[0]` on the /api/chat image-gen path).
/// Returns a formatted PAYLOAD_TOO_LARGE message when over the
/// cap; Ok(()) otherwise. Centralised so the entry points can't
/// drift on the cap or the error wording.
pub(crate) fn validate_image_input_size(byte_len: usize) -> Result<(), String> {
    if byte_len > IMAGE_INPUT_MAX_BYTES {
        Err(format!(
            "image too large: {byte_len} bytes > {} MB cap",
            IMAGE_INPUT_MAX_BYTES / (1024 * 1024)
        ))
    } else {
        Ok(())
    }
}

/// Render the image-gen seed-echo prefix the GUI parses with
/// `parse_seed_prefix` (crates/gui/src/chat_tab.rs). The prefix is
/// the wire-format contract that lets the GUI surface a copyable +
/// lockable seed badge on every image-gen assistant turn.
///
/// Centralised so the two server emit sites (streaming Complete
/// event + non-streaming response body) can't drift on the literal
/// brackets, the "seed:" key, or the surrounding whitespace. Any
/// change to this format breaks the GUI parser, so the unit test
/// pins the exact expected output.
pub(crate) fn format_seed_echo(seed: u64) -> String {
    format!("[seed: {seed}]")
}

/// Render the per-step progress line emitted on the streaming
/// image-gen response body. The GUI's `parse_image_step_progress`
/// helper (the GUI's chat tab (separate repository)) parses the literal
/// `Step <completed>/<total>` prefix to drive the chat tab's
/// progress bar + ETA chip. Single emit site today but extracted
/// so the wire format is greppable and the format string lives
/// next to its docstring; the unit test pins the exact form
/// the parser depends on.
pub(crate) fn format_image_step_progress(completed: usize, total: usize) -> String {
    format!("Step {completed}/{total}")
}

/// Boundary check for `num_steps` on the /v1/images/* endpoints
/// and the /api/chat image-gen path. Zero would emit no timesteps
/// (engine would either crash or return the input noise); values
/// above IMAGE_NUM_STEPS_HARD_CAP are rejected so a misconfigured
/// client (or a hostile one) can't pin the GPU on a single request.
pub(super) fn validate_image_num_steps(steps: u32) -> Result<(), String> {
    if steps == 0 {
        return Err("num_steps must be >= 1".to_string());
    }
    if steps > IMAGE_NUM_STEPS_HARD_CAP {
        return Err(format!(
            "num_steps must be <= {IMAGE_NUM_STEPS_HARD_CAP}; got {steps}"
        ));
    }
    Ok(())
}

/// `output_format` (png/jpeg/webp) re-encodes from the engine's native
/// PNG via the `image` crate when the client asks for a non-PNG format.
pub(super) fn image_response_entry(
    b64_png: String,
    output_format: &str,
    quality: u8,
    seed: Option<u64>,
) -> serde_json::Value {
    // PRIVACY (structural): this helper does not accept a
    // `response_format` parameter. The OpenAI API documents both
    // `url` and `b64_json` modes, but `url` would require persisting
    // generated bytes to disk - a privacy regression we removed in
    // 55493e7. Callers translating from `response_format=url`
    // requests funnel through this helper and silently receive
    // `b64_json`. Re-introducing a `url` mode now requires re-adding
    // the parameter (and the disk-cache machinery), which forces a
    // deliberate review rather than a one-line callsite change.
    //
    // `output_format` (png passthrough vs jpeg/webp re-encode) is
    // still honoured - that's an in-memory transcode, no disk.
    //
    // `seed` (when provided) is echoed in the response so the
    // client can reproduce a generation it liked. The server picks
    // a random seed when the caller doesn't supply one - without
    // echo, that random seed is lost and the generation is
    // un-reproducible.
    let fmt = output_format.trim().to_ascii_lowercase();
    let png_passthrough = fmt.is_empty() || fmt == "png";
    let bytes_b64 = if png_passthrough {
        b64_png
    } else {
        match transcode_png_b64(&b64_png, &fmt, quality) {
            Ok((bytes, _ext)) => {
                use base64::Engine as _;
                base64::engine::general_purpose::STANDARD.encode(&bytes)
            }
            // Transcode failure (bad output_format): fall back to
            // the engine's original PNG so the generation isn't lost.
            Err(_) => b64_png,
        }
    };
    let mut entry = serde_json::json!({ "b64_json": bytes_b64 });
    if let (Some(s), Some(obj)) = (seed, entry.as_object_mut()) {
        obj.insert("seed".to_string(), serde_json::json!(s));
    }
    entry
}

/// Decode the engine's base64 PNG and re-encode it in the requested
/// container. Returns `(bytes, extension)` for the URL handler.
/// `quality` is the JPEG/WebP quality knob (0-100); ignored for PNG
/// since it's lossless.
pub(super) fn transcode_png_b64(
    b64_png: &str,
    output_format: &str,
    quality: u8,
) -> anyhow::Result<(Vec<u8>, &'static str)> {
    use base64::Engine as _;
    let png_bytes = base64::engine::general_purpose::STANDARD
        .decode(b64_png)
        .map_err(|e| anyhow::anyhow!("base64 decode: {e}"))?;
    let fmt = output_format.trim().to_ascii_lowercase();
    if fmt == "png" || fmt.is_empty() {
        return Ok((png_bytes, "png"));
    }
    // ORIENTATION-OK: our own PNG output, which carries no EXIF.
    let img = image::load_from_memory_with_format(&png_bytes, image::ImageFormat::Png)
        .map_err(|e| anyhow::anyhow!("decode png: {e}"))?;
    let q = quality.clamp(1, 100);
    let mut out = Vec::with_capacity(png_bytes.len() / 4);
    use std::io::Cursor;
    match fmt.as_str() {
        "jpeg" | "jpg" => {
            // image::codecs::jpeg::JpegEncoder takes quality 1-100.
            let rgb = img.to_rgb8();
            let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, q);
            encoder
                .encode_image(&rgb)
                .map_err(|e| anyhow::anyhow!("jpeg encode: {e}"))?;
            Ok((out, "jpg"))
        }
        "webp" => {
            // image crate's WebP encoder is lossless-only; honour the
            // request but ignore `q` since the format dimension is
            // different. Sized similar to PNG in practice.
            img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::WebP)
                .map_err(|e| anyhow::anyhow!("webp encode: {e}"))?;
            Ok((out, "webp"))
        }
        other => anyhow::bail!("output_format '{other}' not supported; use 'png'|'jpeg'|'webp'"),
    }
}

/// Lightweight 64-bit seed from the system clock. Used when the
/// caller of /v1/images/* didn't supply `seed` - we pick one,
/// thread it through set_seed, and echo it in the response so the
/// caller can reproduce the result. Not cryptographic; this is
/// just to keep different requests in the same second from getting
/// the same seed by mixing in the nanos.
pub(super) fn rand_u64() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    // Combine seconds and nanos; XOR-fold so the high bits aren't
    // dominated by the slow-moving seconds counter.
    let s = d.as_secs();
    let n = d.subsec_nanos() as u64;
    (s << 32) ^ s ^ (n << 20) ^ n
}

pub(crate) const IMAGE_PROMPT_MAX_CHARS: usize = 4000;

/// Accept both a bare base64 payload and a `data:image/...;base64,` URL, which is
/// what a browser's FileReader hands a web client.
pub(super) fn strip_data_url(s: &str) -> &str {
    match s.strip_prefix("data:") {
        Some(rest) => rest.split_once("base64,").map_or(s, |(_, b)| b),
        None => s,
    }
}

/// Apply [`composite_preserved`] to a base64 PNG, returning a base64 PNG.
pub(super) fn preserve_in_b64(
    b64: &str,
    source: &image::RgbImage,
    region: PreserveRegion,
) -> Result<String, String> {
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| format!("preserve: decode result: {e}"))?;
    // ORIENTATION-OK: the model's own output, not a camera file.
    let edited = image::load_from_memory(&bytes)
        .map_err(|e| format!("preserve: parse result: {e}"))?
        .to_rgb8();
    let out = composite_preserved(&edited, source, region);
    let mut png = std::io::Cursor::new(Vec::new());
    out.write_to(&mut png, image::ImageFormat::Png)
        .map_err(|e| format!("preserve: encode: {e}"))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(png.into_inner()))
}

/// A normalized rectangle (0..1 of width/height) whose SOURCE pixels are kept when
/// an edit returns.
///
/// Defined in `api::extensions` rather than here: an edit assist registered from outside
/// this crate has to name the same rectangle the endpoint composites, and two structurally
/// identical types would have to be converted at every boundary for no reason.
pub(crate) use crate::api::assist::Region as PreserveRegion;

/// Parse `x,y,w,h` in 0..1. Anything malformed or degenerate is rejected rather
/// than silently ignored - a caller asking to protect a face must know if it did
/// not happen.
pub(crate) fn parse_preserve(spec: &str) -> Result<PreserveRegion, String> {
    let v: Vec<f32> = spec
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<f32>()
                .map_err(|_| format!("`preserve`: `{t}` is not a number"))
        })
        .collect::<Result<_, _>>()?;
    let [x, y, w, h] = v[..] else {
        return Err("`preserve` needs exactly x,y,w,h in 0..1".to_string());
    };
    if ![x, y, w, h].iter().all(|q| q.is_finite()) {
        return Err("`preserve` values must be finite".to_string());
    }
    if w <= 0.0 || h <= 0.0 {
        return Err("`preserve` width and height must be > 0".to_string());
    }
    if x < 0.0 || y < 0.0 || x + w > 1.0001 || y + h > 1.0001 {
        return Err("`preserve` must lie inside 0..1".to_string());
    }
    Ok(PreserveRegion { x, y, w, h })
}

/// Composite `region` of `source` back over `edited`, with a feathered border.
///
/// Instruction editors REGENERATE the whole frame conditioned on the source - there
/// is no mechanism anywhere in the pipeline forcing a region to stay identical, so a
/// face drifts as the strength rises. This is the exact remedy for the cases where
/// the answer is "keep this part": the returned pixels ARE the source's.
///
/// The border is blended over `feather` pixels so the seam does not read as a paste;
/// inside that band the result is a linear mix, and beyond it the source is exact.
pub(crate) fn composite_preserved(
    edited: &image::RgbImage,
    source: &image::RgbImage,
    region: PreserveRegion,
) -> image::RgbImage {
    let (w, h) = (edited.width(), edited.height());
    // The source may have been resized to the output: scale it once if so.
    let src = if source.dimensions() == (w, h) {
        std::borrow::Cow::Borrowed(source)
    } else {
        std::borrow::Cow::Owned(image::imageops::resize(
            source,
            w,
            h,
            image::imageops::FilterType::Lanczos3,
        ))
    };
    let x0 = (region.x * w as f32).round().clamp(0.0, w as f32) as u32;
    let y0 = (region.y * h as f32).round().clamp(0.0, h as f32) as u32;
    let rw = ((region.w * w as f32).round() as u32).min(w.saturating_sub(x0));
    let rh = ((region.h * h as f32).round() as u32).min(h.saturating_sub(y0));
    // A tenth of the shorter side, so the blend scales with the region.
    let feather = (rw.min(rh) / 10).max(1) as f32;

    let mut out = edited.clone();
    for y in y0..y0 + rh {
        for x in x0..x0 + rw {
            // Distance to the nearest edge of the region, in pixels.
            let d = (x - x0)
                .min(x0 + rw - 1 - x)
                .min(y - y0)
                .min(y0 + rh - 1 - y) as f32;
            let a = (d / feather).clamp(0.0, 1.0);
            let (e, sp) = (out.get_pixel(x, y), src.get_pixel(x, y));
            let mut px = [0u8; 3];
            for c in 0..3 {
                px[c] = (e[c] as f32 * (1.0 - a) + sp[c] as f32 * a).round() as u8;
            }
            out.put_pixel(x, y, image::Rgb(px));
        }
    }
    out
}

/// OpenAI-compatible `/v1/images/edits`. Multipart form: `image` (PNG/JPEG
/// bytes), `prompt`, optional `model`, `n`, `size`, `response_format`,
/// plus our extensions `strength` (0..1) and `seed` (u64).
/// An image upload goes to a node that holds the family on a card that fits it, the way a
/// generation does; otherwise its multipart reader is handed back to the route.
async fn image_upload_where_the_model_is(
    state: &APIServer,
    headers: &axum::http::HeaderMap,
    body: axum::body::Bytes,
    path: &str,
) -> Result<axum::extract::Multipart, axum::response::Response> {
    use crate::api::handlers::catalogue;
    use axum::response::IntoResponse;
    let model_name = catalogue::upload_model(headers, &body)
        .await
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| "z-image".to_string());
    let fits = match image_model_defaults(&model_name) {
        Ok(d) => super::image_fits_a_card(
            state,
            &model_name,
            crate::inference::place::runtime_demand::RequestGeometry::new(d.size, d.size),
        ),
        Err(_) => true,
    };
    if let Some(relayed) = crate::api::handlers::route_upload_to_holder(
        state,
        headers,
        &model_name,
        super::image_served_here(state, &model_name),
        fits,
        |peer| super::serves_image_family(peer, &model_name),
        path,
        &body,
    )
    .await
    {
        return Err(relayed);
    }
    catalogue::upload_multipart(headers, body)
        .await
        .map_err(|e| {
            let code = axum::http::StatusCode::BAD_REQUEST;
            (code, Json(openai_error_body(code, e))).into_response()
        })
}

pub(crate) async fn images_edits(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    // The job goes where the family is held and a card holds it whole.
    let mut multipart =
        match image_upload_where_the_model_is(&state, &headers, body, "/v1/images/edits").await {
            Ok(multipart) => multipart,
            Err(answer) => return answer,
        };
    // One media job at a time: two diffusion engines cannot share these cards,
    // and letting them try is what produced the OOM storm (see `media_gate`).
    let _media_guard = state.media_lock().await;
    use base64::Engine as _;

    let err_resp = |code: axum::http::StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };

    let mut image_bytes: Option<Vec<u8>> = None;
    // The inpaint mask, kept as base64 so it travels with the other image inputs.
    let mut mask_b64: Option<String> = None;
    let mut prompt: Option<String> = None;
    // What to steer AWAY from. The guidance families are conditioned on two prompts and
    // pushed away from the second, so this is half of what decides the picture - and it
    // is how a caller removes what a positive prompt cannot name: extra fingers, fused
    // limbs, a watermark. The pipeline took one all along; nothing carried the caller's.
    let mut negative_prompt: Option<String> = None;
    let mut model_name: Option<String> = None;
    let mut size: Option<String> = None;
    let mut style: Option<String> = None;
    let mut response_format = "url".to_string();
    let mut n: u32 = 1;
    let mut strength: Option<f64> = None;
    let mut seed: Option<u64> = None;
    // Our extension: a region of the SOURCE to keep verbatim (see `parse_preserve`).
    let mut preserve: Option<(PreserveRegion, crate::api::assist::Chosen)> = None;
    // A spec that is not a rectangle. Held until the source image has arrived, since
    // that is what anything else would have to be read from.
    let mut unresolved_preserve: Option<String> = None;
    let mut num_steps: Option<u32> = None;
    let mut guidance: Option<f64> = None;
    let mut quality: Option<String> = None;
    // Adapters apply here too. This was hard-coded empty on the grounds that "a
    // server-side path in a form field is a different trust question" - which stopped
    // being true in the same commit that added it: adapters are resolved by NAME inside
    // the server's configured lora directory, never by path, so a form field is exactly
    // as safe as a JSON one.
    let mut lora_list: Vec<(String, f32)> = Vec::new();
    let mut sampler_name: Option<String> = None;
    let mut scheduler_name: Option<String> = None;
    let mut output_format = "png".to_string();
    let mut output_compression: u8 = 85;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return err_resp(
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("multipart parse: {e}"),
                )
            }
        };
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "image" => match field.bytes().await {
                Ok(b) => image_bytes = Some(b.to_vec()),
                Err(e) => {
                    return err_resp(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!("read image: {e}"),
                    )
                }
            },
            "negative_prompt" | "negative" => {
                // An EMPTY one is a choice - "steer away from nothing" - and is kept as
                // such rather than folded back into the default.
                negative_prompt = field.text().await.ok();
            }
            "prompt" => {
                // Empty multipart prompt - None so the downstream
                // empty-prompt rejection (line ~5660) fires the
                // canonical "missing or empty prompt" error instead
                // of trying to generate from a zero-length string.
                prompt = field.text().await.ok().filter(|s| !s.trim().is_empty());
            }
            "model" => {
                model_name = field.text().await.ok().filter(|s| !s.trim().is_empty());
            }
            "preserve" => {
                if let Ok(spec) = field.text().await {
                    if !spec.trim().is_empty() {
                        match parse_preserve(&spec) {
                            Ok(r) => preserve = Some((r, crate::api::assist::Chosen::ByCaller)),
                            Err(_) => unresolved_preserve = Some(spec),
                        }
                    }
                }
            }
            "size" => size = field.text().await.ok().filter(|s| !s.trim().is_empty()),
            "response_format" => {
                if let Ok(s) = field.text().await {
                    response_format = s.to_lowercase();
                }
            }
            "n" => {
                if let Ok(s) = field.text().await {
                    if let Ok(v) = s.parse::<u32>() {
                        n = v;
                    }
                }
            }
            "loras" => {
                // Accepts the same shapes as the JSON endpoints: a bare name, a list of
                // names, or {name, strength} objects. A form field carries text, so a
                // JSON array is parsed when it looks like one and treated as a single
                // name otherwise.
                if let Ok(t) = field.text().await {
                    let v = serde_json::from_str::<serde_json::Value>(&t)
                        .unwrap_or_else(|_| serde_json::Value::String(t));
                    match parse_loras(Some(&v)) {
                        Ok(l) => lora_list = l,
                        Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
                    }
                }
            }
            "strength" => {
                if let Ok(s) = field.text().await {
                    strength = s.parse::<f64>().ok();
                }
            }
            "seed" => {
                if let Ok(s) = field.text().await {
                    seed = s.parse::<u64>().ok();
                }
            }
            "num_steps" => {
                if let Ok(s) = field.text().await {
                    num_steps = s.parse::<u32>().ok();
                }
            }
            "guidance" => {
                if let Ok(s) = field.text().await {
                    guidance = s.parse::<f64>().ok();
                }
            }
            "sampler" => sampler_name = field.text().await.ok().filter(|v| !v.trim().is_empty()),
            "scheduler" => {
                scheduler_name = field.text().await.ok().filter(|v| !v.trim().is_empty())
            }
            "quality" => quality = field.text().await.ok(),
            "output_format" => {
                if let Ok(s) = field.text().await {
                    output_format = s.to_lowercase();
                }
            }
            "output_compression" => {
                if let Ok(s) = field.text().await {
                    if let Ok(v) = s.parse::<u8>() {
                        output_compression = v;
                    }
                }
            }
            // Accept but ignore - OpenAI SDK compat shim.
            "style" => {
                style = field.text().await.ok();
            }
            "user" => {
                let _ = field.bytes().await;
            }
            // `mask` is the inpainting mask in OpenAI's spec - only
            // pixels matching the transparent areas should be edited.
            // Flux/Z-Image don't have a masked-inpaint path, so a
            // mask-bearing request can't actually fulfil what the
            // caller asked for. Reject when a non-empty mask is
            // attached so clients don't get unmasked output and
            // wonder why their constrained edit ignored the mask.
            "mask" => {
                // A read failure means the caller genuinely sent a mask but the body
                // was malformed - a 400, never a silent "there was no mask", which
                // would return an unmasked edit the client had no way to detect.
                match field.bytes().await {
                    Ok(bytes) if bytes.is_empty() => {
                        // Some SDKs attach the field as a 0-byte part by default.
                    }
                    Ok(bytes) => {
                        if let Err(e) = validate_image_input_size(bytes.len()) {
                            return err_resp(
                                axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                                format!("mask: {e}"),
                            );
                        }
                        use base64::Engine as _;
                        mask_b64 = Some(base64::engine::general_purpose::STANDARD.encode(&bytes));
                    }
                    Err(e) => {
                        return err_resp(
                            axum::http::StatusCode::BAD_REQUEST,
                            format!("read `mask` field: {e}"),
                        );
                    }
                }
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let prompt = prompt.map(|p| super::super::openai::with_style(p, style.as_deref()));
    let prompt = match prompt {
        Some(p) if !p.trim().is_empty() => p,
        _ => {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                "missing or empty `prompt`".into(),
            )
        }
    };
    if prompt.chars().count() > IMAGE_PROMPT_MAX_CHARS {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            format!("prompt is {} chars; cap at 4000", prompt.chars().count()),
        );
    }
    let image_bytes = match image_bytes {
        Some(b) if !b.is_empty() => b,
        _ => {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                "missing `image` field".into(),
            )
        }
    };
    if let Err(e) = validate_image_input_size(image_bytes.len()) {
        return err_resp(axum::http::StatusCode::PAYLOAD_TOO_LARGE, e);
    }

    if let Err(e) = validate_image_response_format(&response_format) {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
    }
    if let Err(e) = validate_image_output_format(&output_format) {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
    }
    if let Err(e) = validate_image_quality(quality.as_deref()) {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
    }
    if n == 0 {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, "n must be >= 1".into());
    }
    if n > 10 {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            "n must be <= 10".into(),
        );
    }

    let model_name = model_name.unwrap_or_else(|| "flux-schnell".to_string());
    if let Err(e) = validate_model_id(&model_name) {
        return e.into_response();
    }
    let defaults = match image_model_defaults(&model_name) {
        Ok(d) => d,
        Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
    };
    let (width, height) = if let Some(s) = size.as_deref() {
        match parse_image_size(s) {
            // A size the caller named still has to be one the model can halve evenly.
            // Refusing would be defensible; a 500 raised deep inside the U-Net is not,
            // and that is what an unaligned size produced. Snapped, and said out loud.
            Ok((w, h)) => align_to(w, h, defaults.align, "size"),
            Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, format!("size: {e}")),
        }
    } else {
        // No explicit size on an EDIT: follow the reference image, not a square default.
        edit_output_size(&image_bytes, defaults.size, defaults.align)
    };
    // FLUX Kontext is a dev-family (guidance-distilled) editor: it needs many more steps
    // than schnell (4) for a faithful, sharp edit, and a lower guidance. Override the flux
    // defaults (measured: 4 steps blurs edges, 28 steps ~matches the input's edge sharpness).
    let is_kontext_model = model_name.to_lowercase().contains("kontext");
    let default_steps: usize = if is_kontext_model { 28 } else { defaults.steps };
    let default_guidance: f64 = if is_kontext_model {
        2.5
    } else {
        defaults.guidance
    };

    // The image_engine consumes base64-encoded image bytes (PNG/JPEG); the
    // client sends raw bytes, so we re-encode once here.
    let input_b64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

    // Bound once: the placement below reserves for this geometry, and a render that
    // runs out anyway re-plans against the same one.
    let geom = crate::inference::place::runtime_demand::RequestGeometry::new(width, height);
    if let Err(e) = ensure_image_model_loaded(&state, &model_name, geom).await {
        return err_resp(http_status_for_load_error(&e), e);
    }

    if let Some(n) = num_steps {
        if let Err(e) = validate_image_num_steps(n) {
            return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
        }
    }
    let regime = regime_for(&lora_list, num_steps, guidance);
    let default_steps = regime.map(|r| r.steps).unwrap_or(default_steps);
    let default_guidance = regime.map(|r| r.guidance).unwrap_or(default_guidance);
    let resolved_steps = match num_steps {
        Some(n) => n as usize,
        None if is_hd_quality(quality.as_deref()) => default_steps * 2,
        None => default_steps,
    };
    let render_cancel = crate::inference::serve::cancel::CancelToken::new();
    let base = ImageGenParams {
        // Regions compose a canvas from noise; an edit already HAS its composition, and
        // the source image decides where things are. Left empty rather than plumbed.
        regions: Vec::new(),
        // Edits keep every step: they start from a real image, where an approximated
        // step is far more visible than in a from-noise render.
        step_reuse: 0.0,
        negative_prompt: negative_prompt.clone(),
        mask: mask_b64,
        control_image: None,
        control_scale: 1.0,
        cancel: render_cancel.clone(),
        width,
        height,
        num_steps: resolved_steps,
        guidance: clamp_finite_f64(
            guidance.unwrap_or(default_guidance),
            0.0,
            30.0,
            default_guidance,
        ),
        seed: Some(seed.unwrap_or_else(rand_u64)),
        input_image: Some(input_b64),
        strength: clamp_finite_f64(strength.unwrap_or(0.75), 0.0, 1.0, 0.75),
        // A Kontext checkpoint - treat the input as a fixed instruction-edit context
        // (not an img2img init latent).
        kontext: model_name.to_lowercase().contains("kontext"),
        sampler: sampler_name.clone(),
        scheduler: scheduler_name.clone(),
        loras: lora_list.clone(),
    };

    // An edit re-renders the whole frame, so anything the instruction did not ask to
    // change can still drift. On its own this endpoint offers the plain contract - the
    // caller names a rectangle and gets it composited back. A registered assist can do
    // more: find the rectangle itself, say what the picture shows, and put the region
    // back in a way that accounts for the render having moved.
    let assist = crate::api::assist::edit_assist();
    // Decoded ONCE, and only when something is going to look at it.
    let source_rgb: Option<image::RgbImage> =
        (preserve.is_some() || unresolved_preserve.is_some() || assist.is_some())
            .then(|| {
                crate::inference::media::image_processor::decode_image_oriented(&image_bytes)
                    .ok()
                    .map(|i| i.to_rgb8())
            })
            .flatten();

    if let Some(spec) = unresolved_preserve {
        // Not a rectangle. Whatever else it might mean is not this handler's to decide,
        // and an unusable region simply protects nothing.
        if let (Some(a), Some(src)) = (assist.as_ref(), source_rgb.as_ref()) {
            if let Ok(r) = a.resolve_spec(&spec, src) {
                preserve = Some((r, crate::api::assist::Chosen::ByAssist));
            }
        }
    }

    // WHAT IS IN THE PICTURE, folded into an instruction that names no subject. Never
    // blocks a render: the instruction goes through exactly as written, which is what
    // happens with no assist registered.
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
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            "`preserve` was given but the source image could not be decoded".into(),
        );
    }
    let generate_start = std::time::Instant::now();
    let bounces_before = crate::tensor::bounce::pressure_bounces();
    let mut data = Vec::with_capacity(n as usize);
    let mut failures: Vec<String> = Vec::new();
    for i in 0..n {
        let image_seed = base.seed.unwrap_or(0).wrapping_add(i as u64);
        let params = ImageGenParams {
            seed: Some(image_seed),
            sampler: sampler_name.clone(),
            scheduler: scheduler_name.clone(),
            loras: lora_list.clone(),
            ..base.clone()
        };
        let _cg = crate::inference::serve::cancel::CancelGuard::new(params.cancel.clone());
        // Cooperative cancellation: this future is dropped when the client
        // disconnects (the GUI Cancel button drops the request), the guard fires,
        // and the blocking render bails at its next step check instead of running
        // to completion as an orphan burning the GPU.
        let _cg = crate::inference::serve::cancel::CancelGuard::new(render_cancel.clone());
        let _res = generate_image_resilient(&state, &model_name, geom, &prompt, params).await;
        _cg.disarm();
        match _res {
            Ok(b64) => {
                // Put the protected region back BEFORE the response is built, so the
                // caller never sees an intermediate where the face drifted.
                let b64 = match (preserve, source_rgb.as_ref()) {
                    (Some((region, chosen)), Some(src)) => {
                        let restored = match assist.as_ref() {
                            Some(a) => a.restore(&b64, src, region, chosen),
                            // No assist: composite the rectangle, which is the whole of
                            // what this endpoint promises on its own.
                            None => preserve_in_b64(&b64, src, region),
                        };
                        match restored {
                            Ok(v) => v,
                            Err(e) => {
                                return err_resp(axum::http::StatusCode::INTERNAL_SERVER_ERROR, e)
                            }
                        }
                    }
                    _ => b64,
                };
                data.push(image_response_entry(
                    b64,
                    &output_format,
                    output_compression,
                    Some(image_seed),
                ))
            }
            // One variation failing does not fail the batch: losing the images that
            // DID render, because a later seed hit a problem of its own, is the worst
            // available reading of a partial success. Collected here and turned into an
            // error only if nothing rendered at all.
            Err(e) => failures.push(format!("image {}: {e}", i + 1)),
        }
    }
    // Only a batch where NOTHING rendered is a failed request.
    if data.is_empty() && !failures.is_empty() {
        // Say what actually happened, and say it in the LOG as well as in the response.
        // The message used to travel only in the body, over a connection the client had
        // often already dropped - so a failing render left nothing behind to diagnose.
        if all_cancelled(&failures) {
            info!("image: the client stopped the render before it finished");
            return err_resp(
                axum::http::StatusCode::from_u16(CLIENT_CLOSED_REQUEST)
                    .unwrap_or(axum::http::StatusCode::BAD_REQUEST),
                "the render was cancelled".to_string(),
            );
        }
        error!("image: every image failed - {}", failures.join("; "));
        return err_resp(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("every image failed - {}", failures.join("; ")),
        );
    }
    if !failures.is_empty() {
        warn!(
            "image batch: {} of {n} failed - {}",
            failures.len(),
            failures.join("; ")
        );
    }
    let generate_ms = generate_start.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Image edits/variations: model={model_name} {width}x{height} n={n} generate={generate_ms:.0}ms",
    );
    // A pressure bounce means an op could not allocate on its device and ran on the
    // host instead: correct, but a silent cliff worth an order of magnitude. Report
    // it with the render so a degraded path never again hides in a normal-looking
    // success.
    let bounced = crate::tensor::bounce::pressure_bounces() - bounces_before;
    if bounced > 0 {
        tracing::warn!(
            "{bounced} op(s) ran on the host under memory pressure during this render - \
             the placement did not leave enough room on the device"
        );
    }

    let data = match super::super::openai::images_as_urls(
        &state,
        &headers,
        response_format == "url",
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

/// OpenAI-compatible `/v1/images/variations`. Multipart form: `image`
/// (PNG/JPEG bytes) + optional `model`, `n`, `size`, `response_format`,
/// `strength`, `seed`. Internally an img2img run with an empty prompt
/// and lower default strength (0.4) so output stays close to the input.
pub(crate) async fn images_variations(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    // The job goes where the family is held and a card holds it whole.
    let mut multipart = match image_upload_where_the_model_is(
        &state,
        &headers,
        body,
        "/v1/images/variations",
    )
    .await
    {
        Ok(multipart) => multipart,
        Err(answer) => return answer,
    };
    // One media job at a time: two diffusion engines cannot share these cards,
    // and letting them try is what produced the OOM storm (see `media_gate`).
    let _media_guard = state.media_lock().await;
    use base64::Engine as _;

    let err_resp = |code: axum::http::StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };

    let mut image_bytes: Option<Vec<u8>> = None;
    let mut model_name: Option<String> = None;
    let mut size: Option<String> = None;
    let mut response_format = "url".to_string();
    let mut n: u32 = 1;
    let mut strength: Option<f64> = None;
    let mut seed: Option<u64> = None;
    let mut num_steps: Option<u32> = None;
    let mut guidance: Option<f64> = None;
    let mut quality: Option<String> = None;
    // Steering AWAY from something is useful without a positive prompt too: a
    // variation of a photograph still benefits from naming what must not appear.
    let mut negative_prompt: Option<String> = None;
    // Adapters apply here too. This was hard-coded empty on the grounds that "a
    // server-side path in a form field is a different trust question" - which stopped
    // being true in the same commit that added it: adapters are resolved by NAME inside
    // the server's configured lora directory, never by path, so a form field is exactly
    // as safe as a JSON one.
    let mut lora_list: Vec<(String, f32)> = Vec::new();
    let mut sampler_name: Option<String> = None;
    let mut scheduler_name: Option<String> = None;
    let mut output_format = "png".to_string();
    let mut output_compression: u8 = 85;

    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                return err_resp(
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("multipart parse: {e}"),
                )
            }
        };
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "image" => match field.bytes().await {
                Ok(b) => image_bytes = Some(b.to_vec()),
                Err(e) => {
                    return err_resp(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!("read image: {e}"),
                    )
                }
            },
            "model" => {
                model_name = field.text().await.ok().filter(|s| !s.trim().is_empty());
            }
            "size" => size = field.text().await.ok().filter(|s| !s.trim().is_empty()),
            "response_format" => {
                if let Ok(s) = field.text().await {
                    response_format = s.to_lowercase();
                }
            }
            "n" => {
                if let Ok(s) = field.text().await {
                    if let Ok(v) = s.parse::<u32>() {
                        n = v;
                    }
                }
            }
            "loras" => {
                // Accepts the same shapes as the JSON endpoints: a bare name, a list of
                // names, or {name, strength} objects. A form field carries text, so a
                // JSON array is parsed when it looks like one and treated as a single
                // name otherwise.
                if let Ok(t) = field.text().await {
                    let v = serde_json::from_str::<serde_json::Value>(&t)
                        .unwrap_or_else(|_| serde_json::Value::String(t));
                    match parse_loras(Some(&v)) {
                        Ok(l) => lora_list = l,
                        Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
                    }
                }
            }
            "strength" => {
                if let Ok(s) = field.text().await {
                    strength = s.parse::<f64>().ok();
                }
            }
            "seed" => {
                if let Ok(s) = field.text().await {
                    seed = s.parse::<u64>().ok();
                }
            }
            "num_steps" => {
                if let Ok(s) = field.text().await {
                    num_steps = s.parse::<u32>().ok();
                }
            }
            "guidance" => {
                if let Ok(s) = field.text().await {
                    guidance = s.parse::<f64>().ok();
                }
            }
            "sampler" => sampler_name = field.text().await.ok().filter(|v| !v.trim().is_empty()),
            "scheduler" => {
                scheduler_name = field.text().await.ok().filter(|v| !v.trim().is_empty())
            }
            "quality" => quality = field.text().await.ok(),
            "negative_prompt" | "negative" => negative_prompt = field.text().await.ok(),
            "output_format" => {
                if let Ok(s) = field.text().await {
                    output_format = s.to_lowercase();
                }
            }
            "output_compression" => {
                if let Ok(s) = field.text().await {
                    if let Ok(v) = s.parse::<u8>() {
                        output_compression = v;
                    }
                }
            }
            "style" | "user" => {
                let _ = field.bytes().await;
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let image_bytes = match image_bytes {
        Some(b) if !b.is_empty() => b,
        _ => {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                "missing `image` field".into(),
            )
        }
    };
    if let Err(e) = validate_image_input_size(image_bytes.len()) {
        return err_resp(axum::http::StatusCode::PAYLOAD_TOO_LARGE, e);
    }
    if let Err(e) = validate_image_response_format(&response_format) {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
    }
    if let Err(e) = validate_image_output_format(&output_format) {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
    }
    if let Err(e) = validate_image_quality(quality.as_deref()) {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
    }
    if n == 0 {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, "n must be >= 1".into());
    }
    if n > 10 {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            "n must be <= 10".into(),
        );
    }

    let model_name = model_name.unwrap_or_else(|| "flux-schnell".to_string());
    if let Err(e) = validate_model_id(&model_name) {
        return e.into_response();
    }
    let defaults = match image_model_defaults(&model_name) {
        Ok(d) => d,
        Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
    };
    let (width, height) = if let Some(s) = size.as_deref() {
        match parse_image_size(s) {
            // A size the caller named still has to be one the model can halve evenly.
            // Refusing would be defensible; a 500 raised deep inside the U-Net is not,
            // and that is what an unaligned size produced. Snapped, and said out loud.
            Ok((w, h)) => align_to(w, h, defaults.align, "size"),
            Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, format!("size: {e}")),
        }
    } else {
        // No explicit size on an EDIT: follow the reference image, not a square default.
        edit_output_size(&image_bytes, defaults.size, defaults.align)
    };
    // FLUX Kontext is a dev-family (guidance-distilled) editor: it needs many more steps
    // than schnell (4) for a faithful, sharp edit, and a lower guidance. Override the flux
    // defaults (measured: 4 steps blurs edges, 28 steps ~matches the input's edge sharpness).
    let is_kontext_model = model_name.to_lowercase().contains("kontext");
    let default_steps: usize = if is_kontext_model { 28 } else { defaults.steps };
    let default_guidance: f64 = if is_kontext_model {
        2.5
    } else {
        defaults.guidance
    };

    let input_b64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

    // Bound once: the placement below reserves for this geometry, and a render that
    // runs out anyway re-plans against the same one.
    let geom = crate::inference::place::runtime_demand::RequestGeometry::new(width, height);
    if let Err(e) = ensure_image_model_loaded(&state, &model_name, geom).await {
        return err_resp(http_status_for_load_error(&e), e);
    }

    // Empty prompt + low strength = "give me a near-replica with some
    // diffuser variation". OpenAI's spec doesn't accept a prompt here.
    if let Some(n) = num_steps {
        if let Err(e) = validate_image_num_steps(n) {
            return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
        }
    }
    let regime = regime_for(&lora_list, num_steps, guidance);
    let default_steps = regime.map(|r| r.steps).unwrap_or(default_steps);
    let default_guidance = regime.map(|r| r.guidance).unwrap_or(default_guidance);
    let resolved_steps = match num_steps {
        Some(n) => n as usize,
        None if is_hd_quality(quality.as_deref()) => default_steps * 2,
        None => default_steps,
    };
    let render_cancel = crate::inference::serve::cancel::CancelToken::new();
    let base = ImageGenParams {
        // Regions compose a canvas from noise; an edit already HAS its composition, and
        // the source image decides where things are. Left empty rather than plumbed.
        regions: Vec::new(),
        // Edits keep every step: they start from a real image, where an approximated
        // step is far more visible than in a from-noise render.
        step_reuse: 0.0,
        negative_prompt: negative_prompt.clone(),
        mask: None,
        control_image: None,
        control_scale: 1.0,
        cancel: render_cancel.clone(),
        width,
        height,
        num_steps: resolved_steps,
        guidance: clamp_finite_f64(
            guidance.unwrap_or(default_guidance),
            0.0,
            30.0,
            default_guidance,
        ),
        seed: Some(seed.unwrap_or_else(rand_u64)),
        input_image: Some(input_b64),
        strength: clamp_finite_f64(strength.unwrap_or(0.4), 0.0, 1.0, 0.4),
        kontext: false,
        sampler: sampler_name.clone(),
        scheduler: scheduler_name.clone(),
        loras: lora_list.clone(),
    };

    let generate_start = std::time::Instant::now();
    let bounces_before = crate::tensor::bounce::pressure_bounces();
    let mut data = Vec::with_capacity(n as usize);
    let mut failures: Vec<String> = Vec::new();
    for i in 0..n {
        let image_seed = base.seed.unwrap_or(0).wrapping_add(i as u64);
        let params = ImageGenParams {
            seed: Some(image_seed),
            sampler: sampler_name.clone(),
            scheduler: scheduler_name.clone(),
            loras: lora_list.clone(),
            ..base.clone()
        };
        let _cg = crate::inference::serve::cancel::CancelGuard::new(params.cancel.clone());
        // Cooperative cancellation: this future is dropped when the client
        // disconnects (the GUI Cancel button drops the request), the guard fires,
        // and the blocking render bails at its next step check instead of running
        // to completion as an orphan burning the GPU.
        let _cg = crate::inference::serve::cancel::CancelGuard::new(render_cancel.clone());
        let _res = generate_image_resilient(&state, &model_name, geom, "", params).await;
        _cg.disarm();
        match _res {
            Ok(b64) => data.push(image_response_entry(
                b64,
                &output_format,
                output_compression,
                Some(image_seed),
            )),
            // One variation failing does not fail the batch: losing the images that
            // DID render, because a later seed hit a problem of its own, is the worst
            // available reading of a partial success. Collected here and turned into an
            // error only if nothing rendered at all.
            Err(e) => failures.push(format!("image {}: {e}", i + 1)),
        }
    }
    // Only a batch where NOTHING rendered is a failed request.
    if data.is_empty() && !failures.is_empty() {
        // Say what actually happened, and say it in the LOG as well as in the response.
        // The message used to travel only in the body, over a connection the client had
        // often already dropped - so a failing render left nothing behind to diagnose.
        if all_cancelled(&failures) {
            info!("image: the client stopped the render before it finished");
            return err_resp(
                axum::http::StatusCode::from_u16(CLIENT_CLOSED_REQUEST)
                    .unwrap_or(axum::http::StatusCode::BAD_REQUEST),
                "the render was cancelled".to_string(),
            );
        }
        error!("image: every image failed - {}", failures.join("; "));
        return err_resp(
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("every image failed - {}", failures.join("; ")),
        );
    }
    if !failures.is_empty() {
        warn!(
            "image batch: {} of {n} failed - {}",
            failures.len(),
            failures.join("; ")
        );
    }
    let generate_ms = generate_start.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Image edits/variations: model={model_name} {width}x{height} n={n} generate={generate_ms:.0}ms",
    );
    let bounced = crate::tensor::bounce::pressure_bounces() - bounces_before;
    if bounced > 0 {
        tracing::warn!(
            "{bounced} op(s) ran on the host under memory pressure during this render - \
             the placement did not leave enough room on the device"
        );
    }

    let data = match super::super::openai::images_as_urls(
        &state,
        &headers,
        response_format == "url",
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

#[cfg(test)]
mod alignment_tests {
    use super::{align_to, image_model_defaults, regime_for};

    /// An adapter's regime fills in only what the CALLER left unset. A named step count
    /// is an instruction, and an adapter's preference does not outrank it.
    #[test]
    fn a_named_setting_outranks_the_adapter() {
        let loras = vec![("lcm-lora-sdxl".to_string(), 1.0f32)];
        assert!(
            regime_for(&loras, Some(30), Some(7.0)).is_none(),
            "both named"
        );
        // One of the two still leaves the other to fill.
        assert!(
            regime_for(&loras, Some(30), None).is_some(),
            "guidance still unset"
        );
        assert!(
            regime_for(&loras, None, Some(7.0)).is_some(),
            "steps still unset"
        );
    }

    /// A style adapter supplies nothing: its render keeps the model's own recipe.
    #[test]
    fn a_style_adapter_changes_no_sampling() {
        let loras = vec![("flux-realism-xlabs".to_string(), 1.0f32)];
        assert!(regime_for(&loras, None, None).is_none());
    }

    /// And no adapter at all changes nothing - the path every request without one takes.
    #[test]
    fn no_adapter_changes_nothing() {
        assert!(regime_for(&[], None, None).is_none());
    }

    /// THE REPORTED FAILURE. A source 784 pixels tall gives a latent of 98, which halves
    /// to 49, which halves to 24 by truncation - and the up path then hands 48 to a
    /// concatenation expecting 49. The error read `cat: shape mismatch [1, 640, 82, 49]`.
    /// At SDXL's own alignment the height never reaches an odd latent.
    #[test]
    fn the_size_that_broke_sdxl_is_brought_onto_its_alignment() {
        let sdxl = image_model_defaults("raymnants").unwrap();
        assert_eq!(sdxl.align, 64, "SDXL halves the latent three times over");
        let (w, h) = align_to(1312, 784, sdxl.align, "test");
        assert_eq!(h % 64, 0, "784 -> {h}");
        assert_eq!((w, h), (1280, 768));
        // And the latent survives three halvings without a remainder.
        let mut l = h / 8;
        for _ in 0..3 {
            assert_eq!(l % 2, 0, "latent {l} cannot be halved evenly");
            l /= 2;
        }
    }

    /// A size that is already aligned is untouched - the snap must not cost resolution
    /// for nothing, and every shape the studio offers is already a multiple of 64.
    #[test]
    fn an_aligned_size_is_left_alone() {
        for (w, h) in [
            (1024, 1024),
            (896, 1152),
            (832, 1216),
            (768, 1344),
            (1344, 768),
        ] {
            assert_eq!(align_to(w, h, 64, "test"), (w, h), "{w}x{h}");
        }
    }

    /// The requirement is the MODEL's, not one figure for everything: a plain DiT halves
    /// nothing inside its latent and 16 is enough, while forcing 64 on it would throw
    /// away resolution on every edit that follows its source.
    #[test]
    fn each_family_states_its_own_requirement() {
        assert_eq!(image_model_defaults("z-image").unwrap().align, 16);
        assert_eq!(image_model_defaults("flux-schnell").unwrap().align, 16);
        assert_eq!(image_model_defaults("qwen-image").unwrap().align, 16);
        assert_eq!(image_model_defaults("raymnants").unwrap().align, 64);
    }

    /// Snapping DOWN must never reach zero: a tiny source would otherwise ask for a
    /// zero-pixel render, which fails further along and less legibly.
    #[test]
    fn a_tiny_size_never_snaps_to_nothing() {
        let (w, h) = align_to(40, 30, 64, "test");
        assert!(w >= 64 && h >= 64, "{w}x{h}");
    }
}

#[cfg(test)]
mod preserve_tests {
    use super::{composite_preserved, parse_preserve, PreserveRegion};

    /// A caller protecting a face must be TOLD when the spec is unusable, never
    /// silently handed an edit that dropped the protection.
    #[test]
    fn malformed_regions_are_rejected_not_ignored() {
        for bad in [
            "0.1,0.1,0.5",         // too few
            "0.1,0.1,0.5,0.4,0.2", // too many
            "a,0.1,0.5,0.4",       // not a number
            "0.1,0.1,0,0.4",       // zero width
            "0.8,0.1,0.5,0.4",     // runs past the right edge
            "-0.1,0.1,0.5,0.4",    // negative origin
        ] {
            assert!(parse_preserve(bad).is_err(), "`{bad}` should be rejected");
        }
    }

    /// The centre of the protected region must come back EXACTLY from the source -
    /// that is the whole point - while the outside keeps the edit untouched.
    #[test]
    fn the_region_centre_is_the_source_and_the_outside_is_the_edit() {
        let (w, h) = (100u32, 100);
        let edited = image::RgbImage::from_pixel(w, h, image::Rgb([0, 0, 0]));
        let source = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
        let r = PreserveRegion {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        let out = composite_preserved(&edited, &source, r);
        // Dead centre: pure source.
        assert_eq!(out.get_pixel(50, 50), &image::Rgb([255, 255, 255]));
        // Well outside: untouched edit.
        assert_eq!(out.get_pixel(5, 5), &image::Rgb([0, 0, 0]));
        assert_eq!(out.get_pixel(95, 95), &image::Rgb([0, 0, 0]));
    }

    /// The border must FEATHER, and a linear feather is 0 at the outermost row -
    /// that is what makes the seam invisible - rising to pure source at the feather
    /// distance. A hard paste (source everywhere in the region) reads as a sticker.
    #[test]
    fn the_border_ramps_from_the_edit_to_the_source() {
        let (w, h) = (100u32, 100);
        let edited = image::RgbImage::from_pixel(w, h, image::Rgb([0, 0, 0]));
        let source = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
        let r = PreserveRegion {
            x: 0.2,
            y: 0.2,
            w: 0.6,
            h: 0.6,
        };
        let out = composite_preserved(&edited, &source, r);
        // The region is 60 px, so the feather band is 6 px.
        let ramp: Vec<u8> = (20..27).map(|x| out.get_pixel(x, 50)[0]).collect();
        assert_eq!(
            ramp[0], 0,
            "the outermost row is the edit, or the seam shows"
        );
        for pair in ramp.windows(2) {
            assert!(
                pair[1] >= pair[0],
                "the ramp must not go back down: {ramp:?}"
            );
        }
        assert_eq!(
            *ramp.last().unwrap(),
            255,
            "pure source past the feather band"
        );
    }

    /// A source at a different resolution is resized, not rejected: the edit output
    /// size is derived from the request, and callers do send other sizes.
    #[test]
    fn a_differently_sized_source_is_resized() {
        let edited = image::RgbImage::from_pixel(64, 64, image::Rgb([0, 0, 0]));
        let source = image::RgbImage::from_pixel(128, 128, image::Rgb([200, 100, 50]));
        let r = PreserveRegion {
            x: 0.25,
            y: 0.25,
            w: 0.5,
            h: 0.5,
        };
        let out = composite_preserved(&edited, &source, r);
        assert_eq!(out.dimensions(), (64, 64));
        assert_eq!(out.get_pixel(32, 32), &image::Rgb([200, 100, 50]));
    }
}

/// A fresh identifier for a render in flight.
///
/// Monotonic within the process, which is all the registry needs: it exists so a client can
/// name the work it started and stop it, not to be meaningful across restarts.
pub(super) fn next_render_id() -> String {
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    format!("r{}", N.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
}
