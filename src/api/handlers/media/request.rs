//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// `step_reuse` from an options map (see `ImageGenParams::step_reuse`); 0 when absent,
/// clamped to a sane band so a stray value cannot turn a render into one cached step.
pub(super) fn step_reuse_from(options: Option<&serde_json::Value>) -> f32 {
    options
        .and_then(|o| o.get("step_reuse").and_then(serde_json::Value::as_f64))
        .map(|v| clamp_finite_f64(v, 0.0, 0.5, 0.0) as f32)
        .unwrap_or(0.0)
}

#[derive(Debug, serde::Deserialize)]
pub(super) struct OpenAIImageRequest {
    #[serde(default)]
    pub(super) model: Option<String>,
    pub(super) prompt: String,
    #[serde(default)]
    pub(super) n: Option<u32>,
    #[serde(default)]
    pub(super) size: Option<String>,
    #[serde(default)]
    pub(super) response_format: Option<String>,
    /// Tolerate negative `seed` sentinels (e.g. -1 = "random").
    /// Same deserializer as ChatCompletionRequest.seed (e5a12db).
    #[serde(
        default,
        deserialize_with = "crate::api::types::deserialize_optional_seed"
    )]
    pub(super) seed: Option<u64>,
    #[serde(default)]
    pub(super) num_steps: Option<u32>,
    #[serde(default)]
    pub(super) guidance: Option<f64>,
    /// Adaptive step reuse for the sampler: the fraction of predicted output change a
    /// step may accumulate before the model must run again. An APPROXIMATION - omitted
    /// or 0 computes every step, which stays the default.
    #[serde(default)]
    pub(super) step_reuse: Option<f32>,
    /// What to steer the render AWAY from. Only the guidance families use it.
    pub(super) negative_prompt: Option<String>,
    /// Solver name: `euler` (default) or `dpmpp_2m`. The SAMPLER decides how the run
    /// moves between two noise levels; second-order solvers converge in fewer steps for
    /// the same cost per step, which is what checkpoint step-count recommendations assume.
    /// LoRA adapters for this render: `[{"path": "...", "strength": 0.8}]`, or plain
    /// strings for strength 1.0. Applied over the resident weights and dropped after, so
    /// the same checkpoint serves a different adapter next call without a reload.
    #[serde(default)]
    pub(super) loras: Option<serde_json::Value>,
    /// Areas of the canvas with their own prompt - see `parse_regions`.
    #[serde(default)]
    pub(super) regions: Option<serde_json::Value>,
    #[serde(default)]
    pub(super) sampler: Option<String>,
    /// Sigma-curve name: `normal` (default), `karras`, `exponential`. The SCHEDULER
    /// decides WHICH noise levels the run visits - karras spends more of the budget at
    /// low sigma, where detail is decided.
    #[serde(default)]
    pub(super) scheduler: Option<String>,
    /// A pose or edge image (base64), constraining WHERE things go.
    #[serde(default)]
    pub(super) control_image: Option<String>,
    /// How hard it pulls; 0 reproduces the base model.
    #[serde(default)]
    pub(super) control_scale: Option<f32>,
    /// OpenAI-style quality preset. "standard" (default) keeps the
    /// per-model `num_steps`; "hd" doubles them for higher fidelity at
    /// 2- generation cost. Ignored when `num_steps` is set explicitly.
    #[serde(default)]
    pub(super) quality: Option<String>,
    /// OpenAI-style style hint ("vivid"|"natural"). Validated for typos
    /// but currently a no-op - Flux/Z-Image don't have a dedicated
    /// style knob; users can fold the hint into `prompt` directly
    /// for explicit control.
    #[serde(default)]
    pub(super) style: Option<String>,
    /// OpenAI-style end-user identifier. Accepted for API
    /// compatibility but not yet used; could feed into per-user
    /// rate-limit / quota plumbing later.
    #[serde(default)]
    pub(super) user: Option<String>,
    /// When true, switch to Server-Sent Events (`text/event-stream`)
    /// and emit per-step progress events. Final event carries the
    /// finished image as `b64_json`. Not part of OpenAI's documented
    /// surface, but standard SDKs that wrap fetch can still consume it.
    #[serde(default)]
    pub(super) stream: Option<bool>,
    /// gpt-image-1's `output_format`: `png` (default), `jpeg`, `webp`.
    /// The engine always renders PNG natively; non-PNG formats are
    /// transcoded at the handler with the `image` crate.
    #[serde(default)]
    pub(super) output_format: Option<String>,
    /// gpt-image-1's `output_compression`: 0-100 quality knob for
    /// jpeg/webp. Default 85 (matches OpenAI's documented default).
    #[serde(default)]
    pub(super) output_compression: Option<u8>,
}

/// True when the client requested a high-quality preset that should
/// double the step count. Recognises both forms OpenAI documents:
///   * dall-e-3:    `"hd"` (vs `"standard"`)
///   * gpt-image-1: `"high"` or `"auto"` (vs `"low"` / `"medium"`)
///
/// Case-insensitive + trim-tolerant so `HD`, ` Hd `, ` HIGH `, etc.
/// all match. Unknown values fall through to false (default = 1x steps).
pub(super) fn is_hd_quality(q: Option<&str>) -> bool {
    matches!(
        q.map(|s| s.trim().to_ascii_lowercase()),
        Some(ref v) if v == "hd" || v == "high" || v == "auto"
    )
}

/// Reject unknown `quality` values up front. OpenAI's documented enum
/// (across both dall-e-3 + gpt-image-1) is finite - anything else is
/// almost always a typo that would silently land on the default
/// step count. Empty / unset / None passes through (uses default).
pub(super) fn validate_image_quality(q: Option<&str>) -> Result<(), String> {
    let Some(raw) = q else {
        return Ok(());
    };
    let trimmed = raw.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return Ok(());
    }
    match trimmed.as_str() {
        "standard" | "hd" | "low" | "medium" | "high" | "auto" => Ok(()),
        other => Err(format!(
            "quality '{other}' not supported; use 'standard'|'hd' (dall-e-3) or 'low'|'medium'|'high'|'auto' (gpt-image-1)"
        )),
    }
}

/// Hard cap on image-gen dimensions.
///
/// This exists to stop an absurd request (`width=100000`) from asking for terabytes of
/// latent, NOT to decide what a card can hold. It used to sit at 2048 because a Flux
/// render at 2048 takes about 12 GB and that "risks OOM on a 16-GB card" - a VRAM
/// judgement made in the request parser, blind to how many cards there are, which model
/// is being asked for, or that the placement cascade spreads and spills precisely so a
/// large render lands. Fitting is the planner's decision; this is only a sanity bound.
/// Kept public so the chat-path validator and the /v1/images/* parser share the number.
pub(crate) const IMAGE_MAX_DIM: usize = 8192;

/// Boundary check for `width` - `height` as consumed by ImageGenParams.
/// Applies to /v1/images/* (via parse_image_size) AND /api/chat with
/// caller-supplied options.width / options.height - without this gate,
/// `options.width=100000, options.height=100000` allocates ~40 GB of
/// latents and either OOMs or burns the GPU until the OS reaps it.
pub(super) fn validate_image_dimensions(w: usize, h: usize) -> Result<(), String> {
    if w == 0 || h == 0 {
        return Err(format!("size {w}x{h} must be non-zero"));
    }
    if !w.is_multiple_of(16) || !h.is_multiple_of(16) {
        return Err(format!(
            "size {w}x{h} must be a multiple of 16 (VAE stride)"
        ));
    }
    if w > IMAGE_MAX_DIM || h > IMAGE_MAX_DIM {
        return Err(format!(
            "size {w}x{h} exceeds the {IMAGE_MAX_DIM}x{IMAGE_MAX_DIM} sanity bound"
        ));
    }
    Ok(())
}

/// Output size for an EDIT when the client did not ask for one: keep the
/// REFERENCE IMAGE's dimensions. The square model default (1024x1024) re-framed
/// every edit - a 1536x864 photo came back square, cropped and stretched, which
/// is not an edit of that photo.
///
/// Rules: preserve the source aspect ratio; snap each side to a multiple of 16
/// (VAE downsample 8 x DiT patch 2, so the pipelines need that granularity);
/// downscale only when the long side exceeds the model's native resolution
/// (a bigger canvas costs quadratic VRAM and the model was not trained there);
/// An image's dimensions as they will actually be USED - after the EXIF orientation.
///
/// The quarter-turn orientations swap width and height, so reading the header alone
/// gives a portrait photo landscape dimensions and the edit comes back with its aspect
/// ratio transposed. Reads the header only; no pixels are decoded.
pub(super) fn oriented_dimensions(image_bytes: &[u8]) -> Option<(u32, u32)> {
    use image::ImageDecoder as _;
    let reader = image::ImageReader::new(std::io::Cursor::new(image_bytes))
        .with_guessed_format()
        .ok()?;
    let mut decoder = reader.into_decoder().ok()?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let (w, h) = decoder.dimensions();
    Some(match orientation {
        image::metadata::Orientation::Rotate90
        | image::metadata::Orientation::Rotate270
        | image::metadata::Orientation::Rotate90FlipH
        | image::metadata::Orientation::Rotate270FlipH => (h, w),
        _ => (w, h),
    })
}

/// never upscale, and never go below 256. Falls back to the square default when
/// the image header cannot be read.
/// Let an adapter that carries its own sampling regime supply it, when the caller has not.
///
/// A consistency adapter does not restyle a model, it retrains it to reach an image in a
/// handful of steps at almost no guidance. Run at the base model's recipe - twenty-five
/// steps at guidance seven - it does not give the same picture better, it gives an
/// overcooked one, and reads as an adapter that degrades. Nothing here adjusted for that,
/// so it could only ever disappoint.
///
/// It applies ONLY where the caller said nothing. A named step count or guidance is an
/// instruction, and an adapter's preference does not outrank it. What is decided is
/// logged, because a request that renders at four steps when its model's default is
/// twenty-five should not have to be guessed at.
pub(super) fn regime_for(
    loras: &[(String, f32)],
    steps: Option<u32>,
    guidance: Option<f64>,
) -> Option<crate::inference::load::lora::SamplingRegime> {
    if steps.is_some() && guidance.is_some() {
        return None;
    }
    let r = loras
        .iter()
        .find_map(|(name, _)| crate::inference::load::lora::sampling_regime(name))?;
    info!(
        "lora: this adapter carries its own sampling - {} steps at guidance {:.1}; using \
         it for whichever the request left unset",
        r.steps, r.guidance
    );
    Some(r)
}

/// Bring a requested size onto the model's own alignment, saying so when it moves.
///
/// Silence would be the worst of both: the caller asked for one thing, got another, and
/// nothing says which. Refusing outright is worse still - the difference is a few pixels,
/// and the alternative the caller faced was a 500 raised inside a skip concatenation.
pub(super) fn align_to(w: usize, h: usize, align: usize, what: &str) -> (usize, usize) {
    let align = align.max(1);
    let snap = |v: usize| (v / align * align).max(align);
    let (aw, ah) = (snap(w), snap(h));
    if (aw, ah) != (w, h) {
        info!("{what}: {w}x{h} is not a multiple of {align} for this model; rendering {aw}x{ah}");
    }
    (aw, ah)
}

pub(super) fn edit_output_size(image_bytes: &[u8], native: usize, align: usize) -> (usize, usize) {
    let dims = oriented_dimensions(image_bytes);
    let Some((w, h)) = dims else {
        return (native, native);
    };
    let (w, h) = (w as f64, h as f64);
    // Budget the AREA, not the long side: cost (tokens, activations, VRAM) scales
    // with pixel count, so a 16:9 source at the same area as the native square is
    // just as cheap - capping its long side instead threw away resolution for
    // nothing (1536x864 came back 1024x576 when 1360x768 fits the same budget).
    let budget = (native * native) as f64;
    let area = w * h;
    let scale = if area > budget {
        (budget / area).sqrt()
    } else {
        1.0
    };
    // Snap DOWN to the model's own alignment, never below one whole step of it.
    let align = align.max(1);
    let snap = |v: f64| -> usize {
        let px = (v * scale).round() as usize;
        (px / align * align).max(256_usize.div_ceil(align) * align)
    };
    (snap(w), snap(h))
}

/// Parse the `regions` field: areas of the canvas that carry their own prompt.
///
/// `[{"prompt": "...", "x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0, "strength": 1.0}]`, with
/// the rectangle in 0..1 of the image. Out-of-range values are clamped rather than
/// refused: a rectangle that runs off the edge is a reasonable thing to ask for, and the
/// weights are built from the clamped one.
///
/// A region with an empty prompt is dropped - it would denoise against nothing and pull
/// its area toward an unconditioned prediction, which is not what anyone means by it.
pub(super) fn parse_regions(
    v: Option<&serde_json::Value>,
) -> Vec<(String, f32, f32, f32, f32, f32)> {
    let Some(serde_json::Value::Array(a)) = v else {
        return Vec::new();
    };
    a.iter()
        .filter_map(|e| {
            let o = e.as_object()?;
            let prompt = o
                .get("prompt")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .trim();
            if prompt.is_empty() {
                return None;
            }
            let f = |k: &str, d: f64| {
                o.get(k)
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(d)
                    .clamp(0.0, 1.0) as f32
            };
            // Strength may exceed 1: it is a ratio against the base prompt, and pushing
            // a region past it is a legitimate way to say "here, this and not that".
            let strength = o
                .get("strength")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1.0)
                .clamp(0.0, 8.0) as f32;
            let (x, y) = (f("x", 0.0), f("y", 0.0));
            Some((
                prompt.to_string(),
                x,
                y,
                f("w", 1.0 - x as f64),
                f("h", 1.0 - y as f64),
                strength,
            ))
        })
        .collect()
}

/// The adapters this server can apply, by name.
///
/// Without this a caller has to guess file names, and the natural workaround is to send
/// a path - which is exactly what the name-only resolution above refuses.
pub(crate) async fn list_loras() -> axum::Json<serde_json::Value> {
    let names = crate::inference::load::lora::available();
    axum::Json(serde_json::json!({
        "object": "list",
        // Each adapter says what it was trained FOR. A client that offers every
        // adapter whatever model is selected lets the user pick a combination that
        // cannot work: the adapter matches no module, the render fails, and the only
        // clue is a message about tensor names. `family` is null when the layout is
        // not one this recognises - unknown is not a reason to hide it.
        "data": names
            .into_iter()
            .map(|n| {
                let fam = crate::inference::load::lora::target_family(&n);
                serde_json::json!({"id": n, "object": "lora", "family": fam})
            })
            .collect::<Vec<_>>(),
    }))
}

/// Parse the `loras` field into `(path, strength)` pairs.
///
/// Accepts a bare string, a list of strings, or a list of `{name, strength}` - the three
/// shapes callers actually send. Each name is resolved inside the server's configured
/// lora directory; a name that does not resolve is an ERROR rather than a skip, because
/// a caller who asked for an adapter and silently received the base model has no way to
/// tell and would blame the adapter.
pub(super) fn parse_loras(v: Option<&serde_json::Value>) -> Result<Vec<(String, f32)>, String> {
    fn one(v: &serde_json::Value) -> Result<Option<(String, f32)>, String> {
        let (name, strength) = match v {
            serde_json::Value::String(s) if s.trim().is_empty() => return Ok(None),
            serde_json::Value::String(s) => (s.trim().to_string(), 1.0f32),
            serde_json::Value::Object(o) => {
                // `name` is the documented key; `path` is accepted because that is what
                // every other tool calls the field, and rejecting it would only produce
                // a confusing silence. Either way it is resolved as a name.
                let n = o
                    .get("name")
                    .or_else(|| o.get("path"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .trim();
                if n.is_empty() {
                    return Ok(None);
                }
                let st = o.get("strength").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
                // Outside this range is a typo, not an intent: LoRA scales are
                // conventionally 0..1, and a stray 100 would swamp the base model.
                (n.to_string(), st.clamp(-2.0, 2.0))
            }
            _ => return Ok(None),
        };
        let path = crate::inference::load::lora::resolve(&name)?;
        Ok(Some((path.to_string_lossy().into_owned(), strength)))
    }
    let mut out = Vec::new();
    let items: Vec<&serde_json::Value> = match v {
        Some(serde_json::Value::Array(a)) => a.iter().collect(),
        Some(x) => vec![x],
        None => return Ok(out),
    };
    for it in items {
        if let Some(pair) = one(it)? {
            out.push(pair);
        }
    }
    Ok(out)
}

pub(super) fn parse_image_size(s: &str) -> anyhow::Result<(usize, usize)> {
    // Accept both lowercase 'x' (OpenAI canonical) and uppercase 'X'
    // since some clients capitalize. Also tolerate the Unicode
    // multiplication sign '-' which copy-paste from docs sometimes
    // produces.
    let parts: Vec<&str> = s.split(['x', 'X', '\u{00d7}']).collect();
    if parts.len() != 2 {
        anyhow::bail!("size must be WIDTHxHEIGHT, got '{s}'");
    }
    let w: usize = parts[0]
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("bad width in size '{s}'"))?;
    let h: usize = parts[1]
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("bad height in size '{s}'"))?;
    validate_image_dimensions(w, h).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((w, h))
}
