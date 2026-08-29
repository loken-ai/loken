//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Ensure the AudioEngine has the requested whisper checkpoint loaded.
/// Unloads + reloads when the request asks for a different one (small
/// -> medium, large-v3, custom HF id) so the next transcribe call uses
/// the right model.
pub(super) async fn ensure_whisper_model_loaded(
    state: &APIServer,
    requested: Option<&str>,
) -> Result<(), String> {
    let loaded = state.audio_engine.loaded_name().await;
    let needs_reload = match (loaded.as_deref(), requested) {
        (Some(cur), Some(req)) => {
            // Resolve the `whisper-1` alias and bare names the same way
            // load_whisper does, then compare suffixes.
            let normalize = |s: &str| -> String {
                let mapped = match s {
                    "whisper-1" => "whisper-small",
                    other => other,
                };
                mapped.rsplit('/').next().unwrap_or(mapped).to_string()
            };
            normalize(cur) != normalize(req)
        }
        _ => false,
    };
    if needs_reload {
        state.audio_engine.unload().await;
    }
    if !state.audio_engine.is_loaded().await {
        state
            .audio_engine
            .load_whisper(requested)
            .await
            .map_err(|e| format!("whisper load: {e}"))?;
    }
    Ok(())
}

/// Synthesize speech from the request's prompt and return an Ollama-
/// shaped response with the audio payload in `audios[0]` (base64-encoded
/// WAV). Mirrors the image-gen wiring so the GUI's chat tab can route
/// TTS-modality models through /api/chat or /api/generate transparently.
pub(crate) async fn handle_chat_tts(
    state: &APIServer,
    model_name: &str,
    prompt: &str,
    options: Option<&serde_json::Value>,
) -> Result<Response, ApiError> {
    // Same boundary check /v1/audio/speech uses: rejects empty AND
    // over-cap inputs with the documented "split client-side" hint.
    // Without the upper cap a 100 MB chat message would pin the synth
    // engine for many minutes (each char ≈ 5-15 ms on parler-mini-v1).
    if let Err(e) = validate_tts_input(prompt) {
        return Err(ApiError::Validation(e));
    }

    // Load (or swap) the right TTS checkpoint. ensure_tts_model_loaded
    // compares the loaded checkpoint's bare-name suffix against the
    // requested one and only unloads+reloads when they differ, so
    // back-to-back calls to the same model hit the warm path.
    let load_name = if model_name.is_empty() {
        None
    } else {
        Some(model_name)
    };
    ensure_tts_model_loaded(state, load_name)
        .await
        .map_err(|e| ApiError::Internal(format!("TTS load: {e:#}")))?;

    // Time the synth so the chat tab's timing footer shows a duration
    // (format_timing_line collapses to just-duration when token_count
    // is 0, which is what we want here - no token-rate display for
    // audio synth).
    let start = std::time::Instant::now();

    // Build per-call params from caller-supplied options. Voice +
    // speed are GUI-thread-able from the chat tab; absence falls
    // back to the documented Parler-TTS defaults (TtsSynthParams::
    // default). Validation:
    //   - voice: must be one of KNOWN_VOICES (else 400 with the
    //     accepted set in the message - same wording as /v1/audio/
    //     speech for consistency across surfaces).
    //   - speed: clamped to the documented [0.25, 4.0] range via
    //     clamp_finite_f64 (NaN/Inf fall back to 1.0).
    let mut params = TtsSynthParams::default();
    let voice_str = options
        .and_then(|o| o.get("voice"))
        .and_then(serde_json::Value::as_str)
        .map(|s| s.trim().to_ascii_lowercase());
    let speed = options
        .and_then(|o| o.get("speed"))
        .and_then(serde_json::Value::as_f64)
        .map(|v| clamp_finite_f64(v, 0.25, 4.0, 1.0) as f32)
        .unwrap_or(1.0);
    if let Some(ref v) = voice_str {
        if !KNOWN_VOICES.contains(&v.as_str()) {
            return Err(ApiError::Validation(format!(
                "voice '{v}' is unknown; valid presets: {} (or pass `voice_description` to override)",
                KNOWN_VOICES.join(", "),
            )));
        }
        // Map OpenAI preset -> Parler voice_description, folding the
        // speed-tier phrasing into the description so Parler's
        // delivery matches the requested speed (the actual time-
        // domain resample still happens AFTER synth via
        // apply_speed_linear so duration is exact).
        params.voice_description = openai_voice_to_description(v, speed);
    }

    let result = state
        .tts_engine
        .synthesize(prompt.to_string(), params)
        .await
        .map_err(|e| ApiError::Internal(format!("TTS synthesize: {e}")))?;

    // Apply playback-speed time-domain resampling so duration changes
    // match what /v1/audio/speech callers see when they pass speed.
    let pcm = if (speed - 1.0).abs() < 1e-3 {
        result.pcm
    } else {
        apply_speed_linear(&result.pcm, speed)
    };
    let result = TtsResult {
        pcm,
        sample_rate: result.sample_rate,
    };

    let elapsed_ns = start.elapsed().as_nanos() as u64;

    // Encode raw f32 mono PCM into a WAV envelope so the GUI can play
    // the bytes back directly (single-format simplifies the chat-tab
    // attachment renderer; /v1/audio/speech still offers wav/pcm
    // selection for non-chat clients).
    let wav = pcm_to_wav_base64(&result.pcm, result.sample_rate);
    let secs = result.pcm.len() as f32 / result.sample_rate.max(1) as f32;

    // Put a human-readable summary in the content too so chat clients
    // that don't yet render the audios[] field still show *something*
    // useful in the assistant turn. GUI audio-playback wiring lands
    // separately; this keeps the chat tab from looking broken until
    // then.
    let summary = format!(
        "Synthesised {secs:.1}s of audio ({} kHz, {} samples).",
        result.sample_rate / 1000,
        result.pcm.len(),
    );
    let mut response = OllamaChatResponse::new(
        model_name.to_string(),
        Message::new("assistant".to_string(), summary),
    );
    response.done = true;
    response.done_reason = Some("stop".to_string());
    response.message.audios = Some(vec![wav]);
    // total_duration / eval_duration are nanoseconds in the Ollama
    // wire format. eval_count = 0 signals "non-token output" - the
    // GUI's format_timing_line collapses that to just the duration
    // string (no fake tok/s number for synth).
    response.total_duration = Some(elapsed_ns);
    response.eval_duration = Some(elapsed_ns);
    response.eval_count = Some(0);

    Ok(Json(response).into_response())
}

/// /api/generate counterpart to handle_chat_tts. Synthesises from the
/// `prompt` field and returns an OllamaGenerateResponse with the WAV
/// payload in `audios[0]` instead of `images[0]` (mirroring the existing
/// audios field on the chat response message).
pub(crate) async fn handle_generate_tts(
    state: &APIServer,
    model_name: &str,
    prompt: &str,
) -> Result<Response, ApiError> {
    if prompt.trim().is_empty() {
        return Err(ApiError::Validation(
            "TTS request requires non-empty input text".to_string(),
        ));
    }

    // Same load-or-swap pattern as handle_chat_tts so back-to-back
    // /api/generate calls with the same TTS model hit the warm path.
    let load_name = if model_name.is_empty() {
        None
    } else {
        Some(model_name)
    };
    ensure_tts_model_loaded(state, load_name)
        .await
        .map_err(|e| ApiError::Internal(format!("TTS load: {e:#}")))?;

    let start = std::time::Instant::now();
    let result = state
        .tts_engine
        .synthesize(prompt.to_string(), Default::default())
        .await
        .map_err(|e| ApiError::Internal(format!("TTS synthesize: {e}")))?;
    let elapsed_ns = start.elapsed().as_nanos() as u64;

    let wav = pcm_to_wav_base64(&result.pcm, result.sample_rate);

    let mut response = OllamaGenerateResponse::new(model_name.to_string(), String::new());
    response.done = true;
    response.done_reason = Some("stop".to_string());
    response.audios = Some(vec![wav]);
    response.total_duration = Some(elapsed_ns);
    response.eval_duration = Some(elapsed_ns);
    response.eval_count = Some(0);
    Ok(Json(response).into_response())
}
