//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Wrap raw f32 mono PCM into a 16-bit WAV envelope, base64-encoded.
/// Used by handle_chat_tts so the GUI receives a self-contained playable
/// blob without needing to know the sample rate out of band.
pub(super) fn pcm_to_wav_base64(samples: &[f32], sample_rate: u32) -> String {
    pcm_to_wav_base64_ch(samples, sample_rate, 1)
}

/// Interleaved PCM -> 16-bit WAV base64, `channels` ∈ {1, 2}.
pub(super) fn pcm_to_wav_base64_ch(samples: &[f32], sample_rate: u32, channels: u16) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(pcm_to_wav_ch(samples, sample_rate, channels))
}

/// Interleaved PCM -> 16-bit WAV bytes, `channels` ∈ {1, 2}.
pub(super) fn pcm_to_wav_ch(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let mut wav: Vec<u8> = Vec::with_capacity(44 + samples.len() * 2);
    let block_align = channels * 2;
    let byte_rate = sample_rate * block_align as u32;
    let data_bytes = (samples.len() * 2) as u32;
    let riff_size = 36 + data_bytes;
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&riff_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // subchunk1 size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits/sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_bytes.to_le_bytes());
    // Reserve the data region in one shot and write directly into it.
    // The previous extend_from_slice-per-sample loop did N bounds-check
    // + N grow-check calls; for 10 s @ 24 kHz that's 240k extends.
    let data_start = wav.len();
    wav.resize(data_start + samples.len() * 2, 0);
    let dst = &mut wav[data_start..];
    for (i, s) in samples.iter().enumerate() {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        let bytes = v.to_le_bytes();
        dst[2 * i] = bytes[0];
        dst[2 * i + 1] = bytes[1];
    }
    wav
}

/// Pull `key` out of a JSON request body as base64 audio: extract the
/// (non-empty) string field, base64-decode it, and enforce the shared
/// audio-input size cap (`validate_audio_input_size`). Error strings are
/// user-facing and BAD_REQUEST material; wording matches what the JSON
/// audio endpoints have always emitted.
pub(super) fn decode_b64_field(req: &serde_json::Value, key: &str) -> Result<Vec<u8>, String> {
    use base64::Engine as _;
    let b64 = req
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("`{key}` (base64 audio) is required"))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .map_err(|e| format!("`{key}` is not valid base64: {e}"))?;
    validate_audio_input_size(bytes.len())?;
    Ok(bytes)
}

/// Decode compressed audio bytes (WAV/MP3/FLAC/...) to mono f32 PCM @16 kHz
/// on the blocking pool, mapping both failure modes. `what` names the input
/// in error messages ("audio", "reference", ...). A decode failure is the
/// caller's fault -> BAD_REQUEST; a panicked decode worker is ours ->
/// INTERNAL_SERVER_ERROR.
pub(super) async fn decode_audio_blocking(
    bytes: Vec<u8>,
    what: &str,
) -> Result<(Vec<f32>, u32), (StatusCode, String)> {
    match tokio::task::spawn_blocking(move || decode_audio_to_mono_f32_16k(bytes)).await {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err((
            StatusCode::BAD_REQUEST,
            format!("{what} could not be decoded: {e}. {ACCEPTED_AUDIO_FORMATS}"),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("{what} decode join: {e}"),
        )),
    }
}

/// Whisper-via-chat: take the first attached "image" (the GUI's chat-
/// attach button stores audio bytes in images[] for ASR models - see
/// the GUI's chat tab (separate repository)), decode, run whisper, return the
/// transcript as the assistant message content.
///
/// Returns 400 with a clear hint when no audio was attached so users
/// don't see a generic "empty input" error.
pub(crate) async fn handle_chat_asr(
    state: &APIServer,
    model_name: &str,
    images_b64: &[String],
) -> Result<Response, ApiError> {
    let b64 = images_b64.first().ok_or_else(|| {
        ApiError::Validation(
            "Whisper requires an audio attachment. Use the chat tab's Attach \
         button to select a WAV / MP3 / FLAC / OGG / M4A / AAC file."
                .to_string(),
        )
    })?;

    // Apply the same per-attachment size cap the /v1/audio/transcriptions
    // multipart path uses, so the two surfaces cannot disagree about what they
    // accept. Without this an Ollama-shape POST bypasses the cap entirely
    // and reach the symphonia decoder, which would either OOM on the
    // resample path or tie up the audio decode worker for minutes. Estimate
    // decoded bytes from the b64 length (4/3 expansion) so we reject
    // before paying the base64 decode cost.
    let estimated_bytes = b64.len().saturating_mul(3) / 4;
    if let Err(e) = validate_audio_input_size(estimated_bytes) {
        return Err(ApiError::Validation(e));
    }

    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| ApiError::Validation(format!("attached audio is not valid base64: {e}")))?;

    // Decode to mono f32 PCM at whisper's required 16 kHz. Returns the
    // sample rate too even though we already constrain to 16k - keeps
    // the call symmetric with the /v1/audio/transcriptions handler.
    let (samples, sample_rate) =
        decode_audio_blocking(bytes, "audio")
            .await
            .map_err(|(code, e)| {
                if code.is_server_error() {
                    ApiError::Internal(e)
                } else {
                    ApiError::Validation(format!(
                        "{e}. Supported formats: WAV / MP3 / FLAC / OGG / M4A / AAC."
                    ))
                }
            })?;

    // Lazy whisper load (defaults to openai/whisper-small when no
    // model_name override is set up front).
    ensure_whisper_model_loaded(state, Some(model_name))
        .await
        .map_err(ApiError::Internal)?;

    let params = AudioTranscribeParams {
        language: None,
        temperature: 0.0,
        task: crate::inference::engine::audio_engine::WhisperTask::Transcribe,
        timestamps: false,
        collect_metrics: false,
        initial_prompt: None,
    };
    let start = std::time::Instant::now();
    let result = state
        .audio_engine
        .transcribe(samples, sample_rate, params)
        .await
        .map_err(|e| ApiError::Internal(format!("transcribe: {e}")))?;
    let elapsed_ns = start.elapsed().as_nanos() as u64;

    let mut response = OllamaChatResponse::new(
        model_name.to_string(),
        Message::new("assistant".to_string(), result.text),
    );
    response.done = true;
    response.done_reason = Some("stop".to_string());
    // Surface duration in the chat tab's timing footer
    // (format_timing_line collapses to just-duration when token_count
    // is 0 - no fake tok/s for transcription).
    response.total_duration = Some(elapsed_ns);
    response.eval_duration = Some(elapsed_ns);
    response.eval_count = Some(0);
    Ok(Json(response).into_response())
}

/// Health check
/// OpenAI-compatible `/v1/audio/transcriptions`.
///
/// Accepts multipart/form-data with at minimum a `file` part (WAV bytes,
/// 16 kHz mono). Optional fields:
///   * `model`           - HF repo id (default `openai/whisper-small`).
///   * `language`        - BCP-47 code. Auto-detected when omitted.
///   * `temperature`     - non-negative f32 (default 0.0 = greedy).
///   * `response_format` - `json` (default), `text`, or `verbose_json`.
pub(crate) async fn audio_transcriptions(
    state: axum::extract::State<APIServer>,
    multipart: axum::extract::Multipart,
) -> axum::response::Response {
    audio_decode_endpoint(state, multipart, WhisperTask::Transcribe).await
}

/// OpenAI-compatible `/v1/audio/translations`. Same multipart contract as
/// transcriptions, but whisper's `<|translate|>` prompt token forces
/// English output regardless of source language.
pub(crate) async fn audio_translations(
    state: axum::extract::State<APIServer>,
    multipart: axum::extract::Multipart,
) -> axum::response::Response {
    audio_decode_endpoint(state, multipart, WhisperTask::Translate).await
}
