//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Encode RGB frames as an H.264 MP4 by piping rawvideo through ffmpeg. Returns the mp4 bytes.
/// Much smaller and higher-quality than the GIF fallback, but needs ffmpeg on PATH. Dims must be
/// even (the caller clamps to a multiple of 16, so yuv420p is always valid).
pub(super) fn encode_mp4(
    frames: &[Vec<u8>],
    w: usize,
    h: usize,
    fps: u32,
) -> std::io::Result<Vec<u8>> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let out = std::env::temp_dir().join(format!("wan_api_{}_{}.mp4", std::process::id(), nanos));
    let mut child = Command::new("ffmpeg")
        .args([
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            &format!("{w}x{h}"),
            "-r",
            &fps.to_string(),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            // Baseline, no B-frames: MEASURED, not preference. libx264 defaults to High
            // with 2 B-frames, and the in-app player's decoder (openh264) is a baseline
            // decoder - it silently returned 8 of 24 frames, dropping every B-frame and
            // showing a third of the clip. Baseline costs ~9% file size on a short clip
            // and is what plays everywhere, including mobile and older players.
            "-profile:v",
            "baseline",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ])
        .arg(&out)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| std::io::Error::other("ffmpeg stdin unavailable"))?;
        for fr in frames {
            stdin.write_all(fr)?;
        }
        // stdin dropped here - EOF so ffmpeg finalizes the file.
    }
    let status = child.wait()?;
    if !status.success() {
        let _ = std::fs::remove_file(&out);
        return Err(std::io::Error::other(format!("ffmpeg exited {status}")));
    }
    let bytes = std::fs::read(&out)?;
    let _ = std::fs::remove_file(&out);
    Ok(bytes)
}

/// Generative video (text-video) - Wan text-to-video. Returns H.264 MP4 by default (true color,
/// ~10- smaller than GIF, plays everywhere) when ffmpeg is available, else a self-contained
/// animated GIF; pass "format":"gif" to force GIF. JSON body: { "model": "wan", "prompt": "...",
/// "frames": 17, "size": "256x256", "steps": 20, "cfg": 6.0, "seed": 0, "format": "mp4"|"gif" }.
/// Returns { data: [{ b64_json, content_type: "video/mp4"|"image/gif" }] }. Heavy (loads
/// umT5+DiT+VAE, ~minutes per clip) - keep clips small.
#[cfg(feature = "video")]
pub(crate) async fn video_generations(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    // The job goes to a node whose catalogue holds the model when this one's does not.
    {
        let model = body
            .0
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("wan")
            .to_string();
        let served_here =
            crate::distributed::routing::can_serve(&state.local_node_state().await, &model);
        if let Some(relayed) = super::super::route_media_to_holder(
            &state,
            &headers,
            &model,
            served_here,
            true,
            |peer| crate::distributed::routing::can_serve(peer, &model),
            &super::super::VIDEO,
            &body.0,
        )
        .await
        {
            return relayed;
        }
    }
    // One media job at a time: two diffusion engines cannot share these cards,
    // and letting them try is what produced the OOM storm (see `media_gate`).
    let mut _media_guard = state
        .media_lock_for(
            body.0
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("wan"),
            "video",
        )
        .await;
    let err_resp = |code: axum::http::StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };
    let b = body.0;
    let prompt = b
        .get("prompt")
        .or_else(|| b.get("input"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if prompt.is_empty() {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            "prompt (or input) must not be empty".into(),
        );
    }
    // Multi-scene montage: an explicit `prompts` array, or one scene per non-empty
    // prompt line. Scenes render sequentially (models load ONCE) and are cut
    // together into a single clip - the way to get minutes of video out of a
    // seconds-per-shot model. Total-frame budget caps runaway requests.
    let mut prompts: Vec<String> = match b.get("prompts").and_then(|v| v.as_array()) {
        Some(arr) => arr
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        None => prompt
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect(),
    };
    if prompts.is_empty() {
        prompts.push(prompt.clone());
    }
    // Capped like every image sibling. This drives the SAME text encoder, over MANY
    // scenes, and was the one prompt surface with no bound at all - so the encoder
    // silently dropped whatever ran past its window rather than the caller being told.
    for (i, p) in prompts.iter().enumerate() {
        if p.chars().count() > IMAGE_PROMPT_MAX_CHARS {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                    "scene {} is {} characters; cap at {IMAGE_PROMPT_MAX_CHARS}",
                    i + 1,
                    p.chars().count()
                ),
            );
        }
    }
    // SAY WHEN THE ANSWER IS NOT WHAT WAS ASKED. A request for 200 frames answers with 81
    // and a montage of twenty scenes with nine; a clamp is a reasonable thing to do, but
    // silently is not, so both are reported in the response and the log.
    /// The longest clip a single request may ask for, across all its scenes. Not the
    /// model's window - it is a bound on one request's cost, and the note above says when
    /// a length was changed.
    const MAX_TOTAL_FRAMES: usize = 2048; // ~2 min at 16 fps
                                          // THE SIMPLE PATH. A request that says how LONG it wants, and not how, is answered
                                          // from the plan: resolution, frame count and step count derived from the duration, the
                                          // quality dial and which checkpoint was asked for. Anything the caller did state is
                                          // left exactly as stated - this fills silence, it does not overrule.
    let plan = b.get("seconds").and_then(|v| v.as_f64()).map(|secs| {
        use crate::inference::place::video_plan::{plan, Quality};
        let q = match b
            .get("quality")
            .and_then(|v| v.as_str())
            .unwrap_or("standard")
        {
            "draft" => Quality::Draft,
            "fine" => Quality::Fine,
            _ => Quality::Standard,
        };
        let wide = b
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .contains("14b");
        plan(secs as f32, q, wide)
    });
    if let Some(p) = plan.as_ref() {
        eprintln!(
            "[video] planned from {:.1}s at the requested quality: {}x{} {} frames, \
             {} steps, about {:.0}s of denoising ({})",
            p.seconds, p.width, p.height, p.frames, p.steps, p.estimated_seconds, p.basis
        );
    }
    let frames_asked = b
        .get("frames")
        .and_then(|v| v.as_u64())
        .or_else(|| plan.as_ref().map(|p| p.frames as u64))
        .unwrap_or(17);
    // EIGHTY-ONE WAS NEVER THE LIMIT. It is the reference's DEFAULT - its own node declares
    // `length: default=81, min=1, max=..., step=4` - and this clamped there, so a request
    // for a longer clip came back as five seconds with a line in a log the caller never
    // reads. Past 81 the model is outside the window it was trained on and the motion
    // drifts, which is a reason to say so, not to refuse.
    //
    // The `step=4` IS structural: the VAE compresses time by four, so only counts of the
    // form 4k+1 come out exactly, and a request for 50 renders 49.
    let aligned = {
        let n = frames_asked.max(1);
        let k = (n.saturating_sub(1)).div_ceil(4);
        (4 * k + 1).min(MAX_TOTAL_FRAMES as u64)
    };
    let frames = aligned as usize;
    if frames as u64 != frames_asked {
        info!(
            "video: {frames_asked} frames is not one the temporal VAE can express exactly \
             (it compresses time by four, so counts run 1, 5, 9, ...); rendering {frames} \
             ({:.1}s at 16 fps)",
            frames as f32 / 16.0
        );
    }
    let max_scenes = (MAX_TOTAL_FRAMES / frames.max(1)).max(1);
    if prompts.len() > max_scenes {
        info!(
            "video: {} scenes at {frames} frames each exceeds the {:.0}s budget; rendering \
             the first {max_scenes}",
            prompts.len(),
            MAX_TOTAL_FRAMES as f32 / 16.0
        );
        prompts.truncate(max_scenes);
    }
    let (mut width, mut height) = match plan.as_ref() {
        Some(p) => (p.width, p.height),
        None => (256usize, 256usize),
    };
    if let Some(s) = b.get("size").and_then(|v| v.as_str()) {
        if let Some((ws, hs)) = s.split_once('x') {
            if let (Ok(w), Ok(h)) = (ws.trim().parse::<usize>(), hs.trim().parse::<usize>()) {
                width = w;
                height = h;
            }
        }
    }
    // Wan requires spatial dims divisible by 16.
    width = (width / 16).max(1) * 16;
    height = (height / 16).max(1) * 16;
    let steps = b
        .get("steps")
        .and_then(|v| v.as_u64())
        .or_else(|| plan.as_ref().map(|p| p.steps as u64))
        .unwrap_or(20)
        .clamp(1, 60) as usize;
    // Resolution-aware default guidance, like the sampler default below: CFG 6 is the
    // reference setting at/above the model's native scale, but below it the velocity
    // field is noisy and 6 overdrives bright scenes into full-white saturation
    // (measured at 256x256: cfg 6 -> white frame, cfg 4 -> correct subject, and the
    // already-good subjects keep their quality at 4). An explicit "cfg" always wins.
    let cfg_default = if height * width >= 448 * 448 {
        6.0
    } else {
        4.0
    };
    let cfg = b.get("cfg").and_then(|v| v.as_f64()).unwrap_or(cfg_default) as f32;
    // Velocity reuse, off unless asked. It is worth twice here what it is on an image:
    // this sampler runs the DiT once per CFG branch, so a skipped step saves two forwards.
    // The threshold is per-model - measured on images, 0.10 bought 1.61x on one family and
    // REFRAMED the picture on another - so it stays the caller's decision.
    let step_reuse = b
        .get("step_reuse")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
        .clamp(0.0, 0.5) as f32;
    // What the guidance steers AWAY from. Video conditions on two prompts exactly as the
    // image families do, and this endpoint simply never read the field - a caller naming
    // an artefact to avoid was silently conditioning against the empty string.
    let negative_prompt = b
        .get("negative_prompt")
        .or_else(|| b.get("negative"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("")
        .to_string();
    let seed = b.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);
    // Optional sampler override ("unipc" / "heun"); the default is resolution-dependent
    // (see WanSampler::default_for).
    let sampler = match b.get("sampler").and_then(|v| v.as_str()) {
        Some(s) if s.eq_ignore_ascii_case("heun") => {
            crate::inference::model::wan::pipeline::WanSampler::Heun
        }
        Some(s) if s.eq_ignore_ascii_case("unipc") => {
            crate::inference::model::wan::pipeline::WanSampler::UniPc
        }
        _ => crate::inference::model::wan::pipeline::WanSampler::default_for(height, width),
    };

    // A REFERENCE FACE for the clip. The video models here take text, not an identity, so
    // it cannot condition the generation - it is transferred onto the frames afterwards,
    // with the same machinery a still image uses. `face` picks which face in a frame,
    // The frame an image-to-video checkpoint continues. Decoded here rather than deep in
    // the pipeline so a bad image is a 400 before several minutes of render, not after.
    let start_image: Option<(Vec<u8>, usize, usize)> = match b
        .get("image")
        .or_else(|| b.get("start_image"))
        .and_then(|v| v.as_str())
    {
        None => None,
        Some(sv) => {
            use base64::Engine as _;
            let bytes = match base64::engine::general_purpose::STANDARD.decode(strip_data_url(sv)) {
                Ok(b) => b,
                Err(e) => {
                    return err_resp(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!("start image: not base64: {e}"),
                    )
                }
            };
            // Through the orientation-aware decoder: a start frame is whatever the caller
            // had to hand, which includes phone photographs, and a clip that continues a
            // sideways frame is sideways for its whole length.
            match crate::inference::media::image_processor::decode_image_oriented(&bytes) {
                Ok(img) => {
                    let rgb = img.to_rgb8();
                    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
                    Some((rgb.into_raw(), w, h))
                }
                Err(e) => {
                    return err_resp(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!("start image: not an image: {e}"),
                    )
                }
            }
        }
    };

    // exactly as the swap endpoint spells it.
    // Extra conditioning for this clip, when something is registered that recognises the
    // request. Prepared up front: a render of several minutes that finishes without what
    // was asked for, and says nothing about why, is the failure this ordering avoids.
    let conditioner = crate::api::assist::request_conditioner();
    let prepared = match conditioner.as_ref() {
        Some(c) => match c.prepare(&b) {
            Ok(v) => v,
            Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
        },
        None => None,
    };

    // Default to H.264 MP4 (true color, ~10- smaller); "gif" forces the zero-dep fallback.
    // A format this cannot produce is REFUSED rather than quietly turned into MP4.
    // `"webm"`, or a typo like `"mp-4"`, used to render an MP4 and say nothing - the
    // caller gets a file their pipeline may not accept and no clue why. Its image
    // sibling has always answered with a 400 and the list.
    let want_gif = match b.get("format").and_then(|v| v.as_str()) {
        None => false,
        Some(f) if f.eq_ignore_ascii_case("gif") => true,
        Some(f) if f.eq_ignore_ascii_case("mp4") => false,
        Some(other) => {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                format!("format '{other}' not supported; use 'mp4' or 'gif'"),
            )
        }
    };
    let model_name = b
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("wan")
        .to_string();
    // Pick the DiT variant from the model name: anything containing "14" (e.g. "wan-14b")
    // selects the 14B Q8 GGUF (multi-GPU split, higher quality); otherwise the 1.3B default.
    let variant = crate::inference::model::wan::dit::WanVariant::from_token(&model_name);
    // A fine-tune dropped in `<models>/wan/`, selected by name exactly as the Ray families
    // are. Falls back to the variant's own checkpoint when nothing matches, so the plain
    // "wan" request is unchanged.
    let wan_ckpt = crate::inference::model::wan::dit::wan_dit_file_for(
        &state.huggingface_models_dir,
        &model_name,
    );

    // An image-to-video checkpoint renders a clip that CONTINUES a frame. If the caller did
    // not give one, MAKE one from the same prompt rather than refusing: the server already
    // generates images, the first frame of a clip is an image, and a picture from an image
    // model is sharper and better composed than a frame a video model invents.
    //
    // It happens HERE, before the denoiser is placed, because the image model needs the card
    // the denoiser is about to fill - and it is released before that placement, which is the
    // same order the text encoder already uses.
    let start_image = match &start_image {
        Some(_) => start_image,
        None if crate::inference::model::wan::dit::checkpoint_wants_reference(&wan_ckpt) => {
            let side_w = width;
            let side_h = height;
            // Which image model draws it: the caller's choice, or the family this server
            // renders fastest at a good size. Not invented at render time - a name the user
            // can read in the log and change.
            let img_model = b
                .get("start_image_model")
                .and_then(|v| v.as_str())
                .unwrap_or("z-image")
                .to_string();
            let geom =
                crate::inference::place::runtime_demand::RequestGeometry::new(side_w, side_h);
            tracing::info!(
                "video: {model_name} continues a frame and none was given - rendering one \
                 with {img_model} at {side_w}x{side_h}"
            );
            // The three calls report failure differently - a family with no recipe of its
            // own and a load both return a String, a render a boxed error - so all are
            // flattened to one message here rather than left to type inference, which had
            // them disagreeing about what `made` even holds.
            let made: Result<String, String> = match image_model_defaults(&img_model) {
                Err(e) => Err(e),
                Ok(d) => {
                    let params = crate::inference::engine::image_engine::ImageGenParams {
                        height: side_h,
                        width: side_w,
                        num_steps: d.steps,
                        guidance: d.guidance,
                        seed: Some(seed),
                        ..Default::default()
                    };
                    match ensure_image_model_loaded(&state, &img_model, geom).await {
                        Ok(()) => {
                            generate_image_resilient(&state, &img_model, geom, &prompt, params)
                                .await
                                .map_err(|e| e.to_string())
                        }
                        Err(e) => Err(e),
                    }
                }
            };
            // The card goes back to the denoiser by the pressure protocol immediately
            // below, which reclaims idle residents least-recently-used first - the image
            // model having just become one. Unloading it here by hand would duplicate that
            // and take it back even when there was room for both.
            match made {
                Ok(b64) => {
                    use base64::Engine as _;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(strip_data_url(&b64))
                        .unwrap_or_default();
                    // ORIENTATION-OK: this is the server's own render, not a camera file.
                    match image::load_from_memory(&bytes) {
                        Ok(img) => {
                            let rgb = img.to_rgb8();
                            let (w, h) = (rgb.width() as usize, rgb.height() as usize);
                            Some((rgb.into_raw(), w, h))
                        }
                        Err(e) => {
                            return err_resp(
                                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                                format!("video: the start frame it rendered is not an image: {e}"),
                            )
                        }
                    }
                }
                Err(e) => {
                    return err_resp(
                        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                        format!(
                            "video: {model_name} continues a frame, none was given, and one \
                             could not be rendered either: {e}"
                        ),
                    )
                }
            }
        }
        None => None,
    };

    // Pressure protocol (vram_manager): the probe inside trims the mempools so the Wan gates
    // see TRUE free memory; other engines' idle residents are reclaimed only if the DiT (the
    // hot per-step component) could not get a GPU otherwise - hetero placement around them is
    // always preferred (the umT5 spills to its cached CPU staging when no card fits it).
    {
        // Hot demand = the larger of (a) DiT weights + this clip's activation
        // bytes and (b) the VAE-decode peak (weights + the widest 192-channel
        // full-resolution stage) - the decode runs AFTER the last denoise step
        // and used to crawl on CPU when an idle co-resident kept its card full.
        let f_lat = (frames.saturating_sub(1)) / 4 + 1;
        let tokens = f_lat * (height / 16) * (width / 16);
        let dit_hot = crate::inference::model::wan::dit::hot_demand_bytes(variant, tokens);
        // The decode's widest full-resolution stage, from the request, plus the
        // decoder's own weights, from the file. Both were partly hand-written here:
        // the activation term was derived but the weights and the slack were not, so
        // the total was right only for the checkpoint it was written against.
        let vae_hot = crate::inference::place::runtime_demand::vae_encode_bytes(height, width)
            + std::fs::metadata(crate::inference::model::wan::vae::wan_file(
                "Wan2.1_VAE.pth",
            ))
            .map(|m| m.len())
            .unwrap_or(0);
        let hot = dit_hot.max(vae_hot);
        // A node whose cards cannot hold the render whole hands it to a peer that holds
        // the model; the gate is released first so the relay blocks no other render here.
        if !state.card_holds(hot) {
            let holds = |n: &crate::distributed::membership::NodeState| {
                crate::distributed::routing::can_serve(n, &model_name)
            };
            drop(_media_guard);
            if let Some(relayed) = super::super::route_media_to_holder(
                &state,
                &headers,
                &model_name,
                true,
                false,
                holds,
                &super::super::VIDEO,
                &b,
            )
            .await
            {
                return relayed;
            }
            _media_guard = state.media_lock_for(&model_name, "video").await;
        }
        crate::inference::place::vram_manager::ensure_gpu_headroom("video", hot, 0).await;
    }

    // Streaming (SSE): emit a per-DiT-step progress event, then a final "done" event carrying the
    // encoded clip - so a client watching a ~minute-long render sees live progress, not a blind wait.
    if b.get("stream").and_then(|v| v.as_bool()).unwrap_or(false) {
        use axum::response::sse::{Event, KeepAlive};
        let prompts_v = prompts.clone();
        let negative_stream = negative_prompt.clone();
        let ckpt_stream = wan_ckpt.clone();
        let start_image_s = start_image.clone();
        let model_s = model_name.clone();
        let cond_stream = conditioner.clone();
        let prepared_stream = prepared.clone();
        let (tx, mut rx) = tokio::sync::mpsc::channel::<(String, usize, usize)>(64);
        // Cooperative cancellation: if the SSE client disconnects, the stream (and this
        // handler's future) is dropped -> the guard fires -> the blocking render bails at its
        // next step check instead of burning GPU/CPU to completion as an orphan.
        let cancel = crate::inference::serve::cancel::CancelToken::new();
        let cancel_render = cancel.clone();
        // Publish it so the client can stop this render by name. Dropping the connection is
        // NOT a dependable signal - measured: a clip ran to completion for nobody after its
        // client had gone - so the client is given something it can act on instead.
        let render_id = next_render_id();
        let reg = crate::inference::serve::cancel::registry::Entry::new(&render_id, &cancel);
        let id_for_event = render_id.clone();
        let energy = crate::energy_report::begin();
        // The job record follows the render: every count and every placed part lands in
        // it, which is what a listing of this node shows while the render runs.
        let record = _media_guard.reporter();
        let place_fn = record.placement_fn();
        let handle = tokio::task::spawn_blocking(
            move || -> Result<(Vec<u8>, &'static str, usize, usize, usize), String> {
                let _placed = crate::inference::serve::progress::placement::publish(place_fn);
                // Progress across ALL scenes: each scene's denoise calls the cb once
                // per step, so a monotone call counter over scenes*steps tracks the
                // montage as one bar.
                // Forward what the phase counted, untouched.
                //
                // This used to keep its OWN denoise counter here, incremented on every
                // notification and compared against the phase's total multiplied by the
                // scene count. It counted notifications rather than work - the one
                // announcing the start included - so the step ran PAST its total, which on
                // a bar is worse than showing no number. And it discarded the count for
                // every other phase, so a load or a decode could only ever report a bare
                // name however much it knew: minutes of the word "Decoding" and nothing
                // else. A phase knows its own position; a relay's job is to relay it.
                let cb = move |phase: &str, done: usize, total: usize| {
                    record.note(phase, done, total);
                    let _ = tx.blocking_send((phase.to_string(), done, total));
                };
                let clips = crate::inference::model::wan::pipeline::render_many_sampled(
                    &prompts_v,
                    frames,
                    height,
                    width,
                    steps,
                    cfg,
                    seed,
                    variant,
                    sampler,
                    Some(&cb),
                    Some(&cancel_render),
                    &negative_stream,
                    Some(&ckpt_stream),
                    step_reuse,
                    start_image_s
                        .as_ref()
                        .map(|(p, w, h)| (p.as_slice(), *w, *h)),
                )
                .map_err(|e| e.to_string())?;
                let (w, h) = clips
                    .first()
                    .map(|c| (c.width, c.height))
                    .unwrap_or((width, height));
                let mut all_frames: Vec<Vec<u8>> =
                    clips.into_iter().flat_map(|c| c.frames).collect();
                if let (Some(c), Some(p)) = (cond_stream.as_ref(), prepared_stream.as_ref()) {
                    match c.apply_to_frames(p, &mut all_frames, w, h) {
                        Ok(changed) => info!(
                            "video: conditioning applied to {changed} of {} frame(s)",
                            all_frames.len()
                        ),
                        // The clip is real work already done; conditioning that could not
                        // be applied must not throw it away.
                        Err(e) => warn!("video: the conditioning could not be applied ({e})"),
                    }
                }
                let n = all_frames.len();
                if want_gif {
                    Ok((
                        crate::inference::media::gif::encode_gif_bytes(&all_frames, w, h, 6),
                        "image/gif",
                        n,
                        w,
                        h,
                    ))
                } else {
                    match encode_mp4(&all_frames, w, h, 16) {
                        Ok(mp4) => Ok((mp4, "video/mp4", n, w, h)),
                        Err(_) => Ok((
                            crate::inference::media::gif::encode_gif_bytes(&all_frames, w, h, 6),
                            "image/gif",
                            n,
                            w,
                            h,
                        )),
                    }
                }
            },
        );
        let t0 = std::time::Instant::now();
        let node_s = state.node_name();
        let stream = async_stream::stream! {
            // Owned by the STREAM future: dropped when the SSE client disconnects (or the
            // stream ends), firing the render's cancel token. Disarmed on normal completion
            // below is unnecessary - the render has already finished by then and the token is
            // no longer polled.
            let _cg = crate::inference::serve::cancel::CancelGuard::new(cancel.clone());
            // Owned by the STREAM, not by the handler: the handler returns as soon as the
            // response is built, and an entry scoped to it would be dropped before the render
            // took a single step, publishing an identifier already unknown.
            let _reg = reg;
            // The media gate and the job record last as long as the render, for the same
            // reason: dropped with the handler they would let a second render start on the
            // same cards, and list this node as idle while it renders.
            let _media_guard = _media_guard;
            yield Ok::<_, axum::Error>(Event::default().data(
                serde_json::json!({"status": "started", "model": model_s, "total": steps,
                                   "id": id_for_event, "node": node_s}).to_string()));
            while let Some((phase, step, total)) = rx.recv().await {
                // `phase` is new; `step`/`total` keep their meaning, so a client that
                // only reads those is unaffected and simply learns nothing new.
                yield Ok(Event::default().data(serde_json::json!({
                    "status": "rendering",
                    "phase": phase,
                    "phase_label": crate::inference::serve::progress::label(&phase),
                    "step": step, "total": total,
                    "elapsed_ms": t0.elapsed().as_millis() as u64,
                    "node": node_s,
                }).to_string()));
            }
            match handle.await {
                Ok(Ok((media, ctype, n, w, h))) => {
                    use base64::Engine;
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&media);
                    let energy_j = crate::energy_report::end_measured(
                        energy, "video", "[/v1/video/generations]");
                    let mut done = serde_json::json!({
                        "status": "done",
                        "data": [{"b64_json": b64, "content_type": ctype, "frames": n, "width": w, "height": h}],
                        "render_ms": t0.elapsed().as_millis() as u64,
                    });
                    if let Some(j) = energy_j {
                        done["energy_j"] = serde_json::json!((j * 10.0).round() / 10.0);
                    }
                    yield Ok(Event::default().data(done.to_string()));
                }
                Ok(Err(e)) => yield Ok(Event::default().data(serde_json::json!({"status": "error", "error": e}).to_string())),
                Err(e) => yield Ok(Event::default().data(serde_json::json!({"status": "error", "error": format!("render task panicked: {e}")}).to_string())),
            }
        };
        // KEEP-ALIVE, like every other stream here. A video render has long silent
        // stretches - the model load before the first step, a slow step, the VAE
        // decode and encode after the last one - and with nothing written to the
        // socket across them the client's read side gives up mid-render. It surfaces
        // as "error decoding response body", which reads like a protocol fault rather
        // than the silence it is.
        return axum::response::Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let t0 = std::time::Instant::now();
    let prompt_v = prompts.clone();
    let cancel = crate::inference::serve::cancel::CancelToken::new();
    let cancel_render = cancel.clone();
    // Registered for the same reason as the streaming route: this one has no early message
    // in which to hand the identifier back, so a caller finds it through GET /v1/renders.
    let render_id = next_render_id();
    let _reg = crate::inference::serve::cancel::registry::Entry::new(&render_id, &cancel);
    let _cg = crate::inference::serve::cancel::CancelGuard::new(cancel);
    let cond_plain = conditioner.clone();
    let prepared_plain = prepared.clone();
    let negative_plain = negative_prompt.clone();
    let ckpt_plain = wan_ckpt.clone();
    let start_image_p = start_image.clone();
    let record = _media_guard.reporter();
    let place_fn = record.placement_fn();
    let res = tokio::task::spawn_blocking(
        move || -> Result<(Vec<u8>, &'static str, usize, usize, usize), String> {
            let _placed = crate::inference::serve::progress::placement::publish(place_fn);
            let cb = move |phase: &str, done: usize, total: usize| record.note(phase, done, total);
            let clips = crate::inference::model::wan::pipeline::render_many_sampled(
                &prompt_v,
                frames,
                height,
                width,
                steps,
                cfg,
                seed,
                variant,
                sampler,
                Some(&cb),
                Some(&cancel_render),
                &negative_plain,
                Some(&ckpt_plain),
                step_reuse,
                start_image_p
                    .as_ref()
                    .map(|(p, w, h)| (p.as_slice(), *w, *h)),
            )
            .map_err(|e| e.to_string())?;
            let (w, h) = clips
                .first()
                .map(|c| (c.width, c.height))
                .unwrap_or((width, height));
            let mut all_frames: Vec<Vec<u8>> = clips.into_iter().flat_map(|c| c.frames).collect();
            if let (Some(c), Some(p)) = (cond_plain.as_ref(), prepared_plain.as_ref()) {
                match c.apply_to_frames(p, &mut all_frames, w, h) {
                    Ok(changed) => info!(
                        "video: conditioning applied to {changed} of {} frame(s)",
                        all_frames.len()
                    ),
                    Err(e) => warn!("video: the conditioning could not be applied ({e})"),
                }
            }
            let n = all_frames.len();
            // MP4 by default; fall back to GIF if ffmpeg is unavailable or fails. GIF at 16 fps
            // - 6 centiseconds/frame.
            if want_gif {
                Ok((
                    crate::inference::media::gif::encode_gif_bytes(&all_frames, w, h, 6),
                    "image/gif",
                    n,
                    w,
                    h,
                ))
            } else {
                match encode_mp4(&all_frames, w, h, 16) {
                    Ok(mp4) => Ok((mp4, "video/mp4", n, w, h)),
                    Err(_) => Ok((
                        crate::inference::media::gif::encode_gif_bytes(&all_frames, w, h, 6),
                        "image/gif",
                        n,
                        w,
                        h,
                    )),
                }
            }
        },
    )
    .await;
    let (media, ctype, nframes, w, h) = match res {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            return err_resp(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("wan render failed: {e}"),
            )
        }
        Err(e) => {
            return err_resp(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("render task panicked: {e}"),
            )
        }
    };
    use base64::Engine;
    let b64 = base64::engine::general_purpose::STANDARD.encode(&media);
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Json(serde_json::json!({
        "created": created,
        "model": model_name,
        "data": [{
            "b64_json": b64,
            "content_type": ctype,
            "frames": nframes,
            "width": w,
            "height": h,
        }],
        "render_ms": t0.elapsed().as_millis() as u64,
    }))
    .into_response()
}

pub(crate) async fn images_generations(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: Json<serde_json::Value>,
) -> axum::response::Response {
    use axum::response::IntoResponse;

    let err_resp = |code: axum::http::StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };

    let req: OpenAIImageRequest = match serde_json::from_value(body.0.clone()) {
        Ok(r) => r,
        Err(e) => {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                format!("invalid request body: {e}"),
            )
        }
    };

    if req.prompt.trim().is_empty() {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            "prompt must not be empty".into(),
        );
    }
    // OpenAI caps dall-e-3 prompts at 4000 chars; Flux T5 truncates
    // hard at 256 tokens (~1000 chars) anyway. Reject obviously-too-
    // long prompts up front instead of letting the T5 encoder silently
    // drop most of them.
    if req.prompt.chars().count() > IMAGE_PROMPT_MAX_CHARS {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            format!(
                "prompt is {} chars; cap at 4000 (Flux T5 truncates further at ~1000)",
                req.prompt.chars().count()
            ),
        );
    }

    // Default to z-image (Z-Image-Turbo): the working, higher-quality image
    // model. Flux-schnell has open issues at non-native res (black @512) and
    // OOMs at native 1024 (loads single-GPU, T5-XXL+flux+1024-activations >16GB).
    // Explicit `model:"flux-schnell"` still routes to flux for callers who want it.
    let model_name = req.model.clone().unwrap_or_else(|| "z-image".to_string());
    if let Err(e) = validate_model_id(&model_name) {
        return e.into_response();
    }

    // Picks: Z-Image defaults to 1024 @ 9 steps; Flux schnell to 512 @ 4.
    let defaults = match image_model_defaults(&model_name) {
        Ok(d) => d,
        Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
    };
    let (width, height) = if let Some(s) = req.size.as_deref() {
        match parse_image_size(s) {
            Ok(wh) => wh,
            Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, format!("size: {e}")),
        }
    } else {
        (defaults.size, defaults.size)
    };
    // The job goes where the family is held and a card holds it whole.
    let geom = crate::inference::place::runtime_demand::RequestGeometry::new(width, height);
    if let Some(relayed) = super::super::route_media_to_holder(
        &state,
        &headers,
        &model_name,
        super::image_served_here(&state, &model_name),
        super::image_fits_a_card(&state, &model_name, geom),
        |peer| super::serves_image_family(peer, &model_name),
        &super::super::IMAGES,
        &body.0,
    )
    .await
    {
        return relayed;
    }
    // One media job at a time: two diffusion engines cannot share these cards,
    // and letting them try is what produced the OOM storm (see `media_gate`).
    let _media_guard = state.media_lock_for(&model_name, "image").await;

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
    // Default to 1 when absent. The `n == 0` / `n > 10` guards below
    // catch explicit-zero + over-cap consistently across all three
    // image endpoints.
    let n = req.n.unwrap_or(1);
    if n == 0 {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, "n must be >= 1".into());
    }
    if n > 10 {
        return err_resp(
            axum::http::StatusCode::BAD_REQUEST,
            "n must be <= 10".into(),
        );
    }
    // Validate response_format at the request boundary so a typo like
    // "xml" still gets a 400. The validated value is then discarded:
    // see validate_image_response_format / image_response_entry for
    // why no downstream code threads it.
    if let Err(e) = validate_image_response_format(req.response_format.as_deref().unwrap_or("url"))
    {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
    }
    if let Some(raw) = req.output_format.as_deref() {
        if let Err(e) = validate_image_output_format(raw) {
            return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
        }
    }
    if let Err(e) = validate_image_quality(req.quality.as_deref()) {
        return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
    }
    // `style` is documented as `"vivid"|"natural"` (dall-e-3 only).
    // Currently a no-op for Flux/Z-Image (no dedicated style knob),
    // but reject typos so the no-op-ness is explicit instead of
    // silently swallowed.
    if let Some(s) = req.style.as_deref() {
        let trimmed = s.trim().to_ascii_lowercase();
        if !trimmed.is_empty() && trimmed != "vivid" && trimmed != "natural" {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                format!("style '{s}' not supported; use 'vivid' or 'natural' (currently no-op for Flux/Z-Image)"),
            );
        }
    }

    // Ensure the right image-model family is loaded (unloads + reloads
    // when the request switches between Flux and Z-Image).
    // Bound once: the placement below reserves for this geometry, and a render that
    // runs out anyway re-plans against the same one.
    let geom = crate::inference::place::runtime_demand::RequestGeometry::new(width, height);
    // A STREAMING request loads from inside the stream instead (see below). Loading here
    // makes the load invisible: no part of the response body exists until this returns, so
    // the minute spent reading a checkpoint reaches the client as a minute of silence.
    let will_stream = req.stream.unwrap_or(false);
    if !will_stream {
        if let Err(e) = ensure_image_model_loaded(&state, &model_name, geom).await {
            return err_resp(http_status_for_load_error(&e), e);
        }
    }

    // OpenAI's `quality: "hd"` doubles the step count for higher
    // fidelity. Explicit `num_steps` overrides quality.
    if let Some(n) = req.num_steps {
        if let Err(e) = validate_image_num_steps(n) {
            return err_resp(axum::http::StatusCode::BAD_REQUEST, e);
        }
    }
    let resolved_steps_base = match req.num_steps {
        Some(n) => n as usize,
        None if is_hd_quality(req.quality.as_deref()) => default_steps * 2,
        None => default_steps,
    };
    // Resolve seed: caller-provided wins, otherwise pick a random u64
    // and thread it through. Echoing the SERVER-PICKED seed in the
    // response lets clients reproduce a generation they liked even
    // when they didn't supply one themselves - without this, the
    // server would burn a random seed and the user could never
    // re-roll the same result.
    let base_seed: u64 = req.seed.unwrap_or_else(rand_u64);

    // Resolve the reference face to an identity embedding BEFORE the render, so a
    // portrait with no detectable face is a clear 400 rather than a picture of a
    // stranger. `None` here means the request simply did not ask for one.
    // A reference face is served two different ways, and which one applies is a
    // property of the model, not a preference.
    //
    // Families with an identity adapter CONDITION on it: the denoiser attends to the
    // face while it draws, so the person is generated into the scene's pose and light.
    // Every other family has no such adapter - the weights are trained against one
    // architecture's attention layers - so the face is applied AFTERWARDS, by swapping
    // it onto the rendered result. That is a weaker thing and it is labelled as such
    // in the response; what it is not is silent. Handing the reference to a family
    // that ignores it used to return a convincing picture of a stranger with nothing
    // anywhere to say the request had not been carried out.
    // Extra conditioning, when something is registered that recognises this request.
    //
    // Prepared BEFORE the render so a request that asks for conditioning which cannot be
    // prepared is refused up front, rather than after several minutes of work that
    // ignored what was asked for. What the conditioner reads out of the request, and
    // what it does with it, is its own business - this handler forwards an opaque token.
    let conditioner = crate::api::assist::request_conditioner();
    let prepared = match conditioner.as_ref() {
        Some(c) => match c.prepare(&body.0) {
            Ok(v) => v,
            Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
        },
        None => None,
    };
    let render_cancel = crate::inference::serve::cancel::CancelToken::new();
    let sampler_name = req.sampler.clone();
    let lora_list = match parse_loras(req.loras.as_ref()) {
        Ok(v) => v,
        Err(e) => return err_resp(axum::http::StatusCode::BAD_REQUEST, e),
    };
    // An adapter that carries its own sampling supplies it - only for whichever of the
    // two the caller left unset, and only now that the adapters are known.
    let regime = regime_for(&lora_list, req.num_steps, req.guidance);
    let default_guidance = regime.map(|r| r.guidance).unwrap_or(default_guidance);
    let resolved_steps = match (req.num_steps, regime) {
        (Some(n), _) => n as usize,
        (None, Some(r)) => r.steps,
        (None, None) => resolved_steps_base,
    };
    let scheduler_name = req.scheduler.clone();
    // A control image that failed to decode is an ERROR, not an absence. Silently
    // dropping it produced a perfectly normal-looking render with the pose ignored, and
    // the only way to find out was to notice the subject was not in the pose asked for.
    //
    // A `data:` prefix is accepted here too - it was stripped for the identity image and
    // not for this one, so the same paste worked in one field and did nothing in the other.
    let control_image_bytes = match req.control_image.as_deref() {
        None => None,
        Some(b64) => {
            use base64::Engine as _;
            match base64::engine::general_purpose::STANDARD.decode(strip_data_url(b64)) {
                Ok(v) => Some(v),
                Err(e) => {
                    return err_resp(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!("control_image is not valid base64: {e}"),
                    )
                }
            }
        }
    };
    if let Some(bytes) = control_image_bytes.as_ref() {
        if let Err(e) = validate_image_input_size(bytes.len()) {
            return err_resp(
                axum::http::StatusCode::PAYLOAD_TOO_LARGE,
                format!("control_image: {e}"),
            );
        }
    }
    let base_params = ImageGenParams {
        step_reuse: req.step_reuse.unwrap_or(0.0).clamp(0.0, 0.5),
        negative_prompt: req.negative_prompt.clone(),
        regions: parse_regions(req.regions.as_ref()),
        // Masking belongs to /v1/images/edits: a mask marks where to change an
        // EXISTING picture, and this endpoint generates from nothing.
        mask: None,
        control_image: control_image_bytes,
        control_scale: req
            .control_scale
            .map(|v| clamp_finite_f64(v as f64, 0.0, 2.0, 1.0) as f32)
            .unwrap_or(1.0),
        cancel: render_cancel.clone(),
        width,
        height,
        num_steps: resolved_steps,
        guidance: clamp_finite_f64(
            req.guidance.unwrap_or(default_guidance),
            0.0,
            30.0,
            default_guidance,
        ),
        seed: Some(base_seed),
        input_image: None,
        strength: 0.75,
        kontext: false,
        sampler: sampler_name.clone(),
        scheduler: scheduler_name.clone(),
        loras: lora_list.clone(),
    };

    // Stream-mode short-circuit. Drive the ImageEngine's progress channel
    // and forward each event as a Server-Sent Event. n>1 is not supported
    // in stream mode (would need to interleave channels). When stream is
    // requested but n>1, we still produce all images in series - the
    // stream emits per-step events for each one with an `index` field.
    if will_stream {
        use async_stream::stream;
        use axum::response::sse::{Event, Sse};

        let engine = state.image_engine.clone();
        let prompt = super::super::openai::with_style(req.prompt.clone(), req.style.as_deref());
        let base = base_params.clone();
        let out_format_stream = req
            .output_format
            .as_deref()
            .map(|s| s.trim().to_ascii_lowercase())
            .unwrap_or_else(|| "png".to_string());
        let out_quality_stream = req.output_compression.unwrap_or(85);
        let stream_started = std::time::Instant::now();
        let stream_model = model_name.clone();
        // Needed inside the generator to RE-PLAN a render that runs out of VRAM
        // mid-stream, the same way the non-streaming route does.
        let stream_state = state.0.clone();

        let stream = stream! {
            // Phase 0: get the model up, REPORTING. A cold checkpoint is the longest part
            // of a cold render and it used to happen before the response existed at all;
            // the loaders count every tensor they read (progress::scoped), and this is the
            // channel that carries those counts to whoever asked for the picture.
            {
                let (load_tx, mut load_rx) =
                    tokio::sync::mpsc::channel::<ImageEngineLoadingProgress>(32);
                let load_state = stream_state.clone();
                let load_model = stream_model.clone();
                // The load runs beside this generator so its counts can be forwarded while
                // it happens - and is ABORTED when the generator is dropped. Measured
                // without that: a client that stopped reading during the load left the
                // checkpoint loading to completion for nobody, and the next request was
                // told every GPU was busy by the ghost of the one that had gone. The
                // loader's own cancel guards fire on the abort, so it bails at its next
                // per-tensor check.
                let mut load_task = AbortOnDrop(tokio::spawn(async move {
                    ensure_image_model_loaded_reporting(
                        &load_state,
                        &load_model,
                        geom,
                        Some(load_tx),
                    )
                    .await
                }));
                // Drained to CHANNEL CLOSE, not to the loader's own "done": a load that
                // lost its card retries, and each attempt announces a completion of its
                // own. The sender lives in the task, so the close is the honest end.
                let mut load_failed: Option<String> = None;
                while let Some(ev) = load_rx.recv().await {
                    match ev {
                        ImageEngineLoadingProgress::Stage(msg) => {
                            yield Ok::<Event, std::convert::Infallible>(Event::default().data(
                                serde_json::json!({
                                    "status": "loading",
                                    "phase": crate::inference::serve::progress::phase::LOAD_MODEL,
                                    "phase_label": msg,
                                })
                                .to_string(),
                            ));
                        }
                        ImageEngineLoadingProgress::Error(msg) => load_failed = Some(msg),
                        ImageEngineLoadingProgress::Done => {}
                    }
                }
                let outcome = match (&mut load_task.0).await {
                    Ok(r) => r,
                    Err(e) => Err(format!("image load task failed: {e}")),
                };
                if let Err(e) = outcome.map_err(|e| load_failed.unwrap_or(e)) {
                    yield Ok(Event::default().data(
                        serde_json::json!({"error": {"message": e}}).to_string(),
                    ));
                    yield Ok(Event::default().data("[DONE]"));
                    return;
                }
            }
            for i in 0..n {
                let params = ImageGenParams {
                    seed: base.seed.map(|s| s.wrapping_add(i as u64)),
                    sampler: sampler_name.clone(),
                    scheduler: scheduler_name.clone(),
                    loras: lora_list.clone(),
            ..base.clone()
                };
                // Running out of VRAM is not a rendering failure, it is a placement
                // that stopped being true - another engine or another process took
                // the card between the decision and the denoise. Dropping the
                // resident returns its memory and lets the loader plan again against
                // what is actually free. Once: a second exhaustion means the pressure
                // is not transient, and then the error is the honest answer.
                let mut replanned = false;
                'render: loop {
                let attempt_params = params.clone();
                let _cg = crate::inference::serve::cancel::CancelGuard::new(attempt_params.cancel.clone());
                let _res = engine.generate_image_stream(&prompt, attempt_params).await;
                _cg.disarm();
                let mut rx = match _res {
                    Ok(rx) => rx,
                    Err(e) => {
                        if !replanned && is_vram_exhaustion(e.as_ref()) {
                            replanned = true;
                            let level = crate::inference::place::vram_manager::vram_degrade();
                            tracing::warn!("image stream: {stream_model} out of VRAM ({e}) - re-planning at pressure {level}");
                            engine.unload().await;
                            let _ = ensure_image_model_loaded(&stream_state, &stream_model, geom).await;
                            continue 'render;
                        }
                        yield Ok::<Event, std::convert::Infallible>(Event::default().data(
                            serde_json::json!({"error":{"message":format!("stream init: {e}")}}).to_string()
                        ));
                        return;
                    }
                };
                while let Some(ev) = rx.recv().await {
                    // The denoise reports its own exhaustion as an event rather than
                    // as a failed call, so the retry has to be reachable from here
                    // too - otherwise only the setup was ever recoverable.
                    if let ImageStreamEvent::Error(msg) = &ev {
                        if !replanned
                            && (msg.contains("[oom]")
                                || msg.contains("out of memory")
                                || msg.contains("out_of_memory"))
                        {
                            replanned = true;
                            let level = crate::inference::place::vram_manager::vram_degrade();
                            tracing::warn!("image stream: {stream_model} out of VRAM mid-render ({msg}) - re-planning at pressure {level}");
                            engine.unload().await;
                            let _ = ensure_image_model_loaded(&stream_state, &stream_model, geom).await;
                            continue 'render;
                        }
                    }
                    let payload = match ev {
                        ImageStreamEvent::Progress { completed, total } => serde_json::json!({
                            "index": i,
                            "step": completed,
                            "total": total,
                        }),
                        ImageStreamEvent::Complete { image_base64 } => {
                            // Honour response_format on the terminal
                            // chunk: `url` mode caches the PNG to disk
                            // and yields a URL; `b64_json` (or default)
                            // passes the engine's b64 through. The
                            // transcode helper handles output_format
                            // (jpeg/webp) consistently with the
                            // non-stream path.
                            let entry = image_response_entry(
                                image_base64,
                                &out_format_stream,
                                out_quality_stream,
                                Some(base.seed.unwrap_or(0).wrapping_add(i as u64)),
                            );
                            let mut v = serde_json::json!({
                                "index": i,
                                "done": true,
                            });
                            if let (Some(obj), Some(entry_obj)) =
                                (v.as_object_mut(), entry.as_object())
                            {
                                for (k, val) in entry_obj {
                                    obj.insert(k.clone(), val.clone());
                                }
                            }
                            v
                        }
                        ImageStreamEvent::Error(e) => serde_json::json!({
                            "index": i,
                            "error": {"message": e},
                        }),
                    };
                    yield Ok(Event::default().data(payload.to_string()));
                }
                break;
                }
            }
            let total_ms = stream_started.elapsed().as_millis();
            tracing::info!(
                "Image generations stream: model={stream_model} n={n} total={total_ms}ms"
            );
            // Final terminator event so clients can cleanly close on `[DONE]`.
            yield Ok(Event::default().data("[DONE]"));
        };
        return Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response();
    }

    let out_format = req
        .output_format
        .as_deref()
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "png".to_string());
    let out_quality = req.output_compression.unwrap_or(85);
    let generate_start = std::time::Instant::now();
    let bounces_before = crate::tensor::bounce::pressure_bounces();
    let energy = crate::energy_report::begin();
    let mut data = Vec::with_capacity(n as usize);
    let mut failures: Vec<String> = Vec::new();
    for i in 0..n {
        // Step the seed so each image in the batch is distinct.
        // base_params.seed is always Some(_) at this point (resolved
        // above to either the caller's seed or a server-picked
        // random u64).
        let image_seed = base_params.seed.unwrap_or(0).wrapping_add(i as u64);
        let params = ImageGenParams {
            seed: Some(image_seed),
            sampler: sampler_name.clone(),
            scheduler: scheduler_name.clone(),
            loras: lora_list.clone(),
            ..base_params.clone()
        };
        let _cg = crate::inference::serve::cancel::CancelGuard::new(params.cancel.clone());
        // Cooperative cancellation: this future is dropped when the client
        // disconnects (the GUI Cancel button drops the request), the guard fires,
        // and the blocking render bails at its next step check instead of running
        // to completion as an orphan burning the GPU.
        let _cg = crate::inference::serve::cancel::CancelGuard::new(render_cancel.clone());
        let _res = generate_image_resilient(&state, &model_name, geom, &req.prompt, params).await;
        _cg.disarm();
        match _res {
            Ok(b64) => {
                let b64 = match (conditioner.as_ref(), prepared.as_ref()) {
                    (Some(c), Some(p)) => match c.apply_to_frame(p, &b64) {
                        Ok(v) => v,
                        Err(e) => return err_resp(axum::http::StatusCode::UNPROCESSABLE_ENTITY, e),
                    },
                    _ => b64,
                };
                data.push(image_response_entry(
                    b64,
                    &out_format,
                    out_quality,
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
    let energy_j = crate::energy_report::end_measured(energy, "image", "[/v1/images/generations]");
    let generate_ms = generate_start.elapsed().as_secs_f64() * 1000.0;
    // GUIDANCE AND SOLVER TOO. They decide what the picture looks like at least as
    // much as the step count, and neither was recorded - so "the output is imprecise
    // despite 50 steps" could not be answered from the log at all. On a
    // guidance-distilled checkpoint the guidance value dominates and more steps add
    // nothing, which is exactly the case this was asked about.
    info!(
        "Image generations: model={model_name} {width}x{height} steps={resolved_steps} \
         guidance={:.2} sampler={} scheduler={} n={n} generate={generate_ms:.0}ms",
        base_params.guidance,
        sampler_name.as_deref().unwrap_or("default"),
        scheduler_name.as_deref().unwrap_or("default"),
    );
    // A pressure bounce means an op could not allocate on its device and ran on the
    // host instead: correct, but a silent cliff worth an order of magnitude. Report
    // it with the render so a degraded path never again hides in a normal-looking
    // success. The counter is process-wide, and LLM generations are deliberately not
    // gated against media, so a concurrent chat request can add to this figure - it
    // is a signal that something ran degraded, not a per-request attribution.
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
        req.response_format.as_deref().unwrap_or("url") == "url",
        data,
    ) {
        Ok(d) => d,
        Err(e) => return e.into_response(),
    };
    let mut body_v = serde_json::json!({
        "created": chrono::Utc::now().timestamp(),
        "data": data,
        "render_ms": generate_ms as u64,
    });
    if let Some(j) = energy_j {
        body_v["energy_j"] = serde_json::json!((j * 10.0).round() / 10.0);
    }
    // Whatever the conditioner wants the caller to know about what it did. It cannot
    // see the render, and a request whose conditioning quietly did less than asked looks
    // exactly like one that did all of it.
    if let (Some(c), Some(p)) = (conditioner.as_ref(), prepared.as_ref()) {
        if let Some(note) = c.note(p) {
            body_v["notes"] = serde_json::json!([note]);
        }
    }
    let body = Json(body_v);
    (image_response_headers(generate_ms), body).into_response()
}
