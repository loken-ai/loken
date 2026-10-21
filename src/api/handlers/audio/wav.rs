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

/// The 44-byte header of a 16-bit PCM WAV. `data_bytes` is `None` for a stream of
/// unknown length, whose RIFF and data sizes are written as 0xFFFFFFFF: most players,
/// ffmpeg and libsndfile among them, then read until the connection closes.
pub(super) fn wav_header(sample_rate: u32, channels: u16, data_bytes: Option<u32>) -> Vec<u8> {
    const BYTES_PER_SAMPLE: u16 = 2;
    let block_align = channels * BYTES_PER_SAMPLE;
    let byte_rate = sample_rate * u32::from(block_align);
    let riff_size = data_bytes.map_or(u32::MAX, |d| 36 + d);
    let mut hdr = Vec::with_capacity(44);
    hdr.extend_from_slice(b"RIFF");
    hdr.extend_from_slice(&riff_size.to_le_bytes());
    hdr.extend_from_slice(b"WAVE");
    hdr.extend_from_slice(b"fmt ");
    hdr.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    hdr.extend_from_slice(&1u16.to_le_bytes()); // PCM
    hdr.extend_from_slice(&channels.to_le_bytes());
    hdr.extend_from_slice(&sample_rate.to_le_bytes());
    hdr.extend_from_slice(&byte_rate.to_le_bytes());
    hdr.extend_from_slice(&block_align.to_le_bytes());
    hdr.extend_from_slice(&(BYTES_PER_SAMPLE * 8).to_le_bytes()); // bits per sample
    hdr.extend_from_slice(b"data");
    hdr.extend_from_slice(&data_bytes.unwrap_or(u32::MAX).to_le_bytes());
    hdr
}

/// Interleaved PCM -> 16-bit WAV bytes, `channels` ∈ {1, 2}. The samples are encoded
/// by [`write_i16_le_into`], as the raw PCM stream is, so the two never differ.
pub(super) fn pcm_to_wav_ch(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let data_bytes = samples.len() * 2;
    let mut wav = wav_header(sample_rate, channels, Some(data_bytes as u32));
    let data_start = wav.len();
    wav.resize(data_start + data_bytes, 0);
    write_i16_le_into(&mut wav[data_start..], samples);
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
        .decode(crate::api::types::base64_payload(b64))
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

/// A transcription with no option set: language detected, greedy, no timestamps.
pub(super) fn plain_transcription() -> AudioTranscribeParams {
    AudioTranscribeParams {
        language: None,
        temperature: 0.0,
        task: WhisperTask::Transcribe,
        timestamps: false,
        collect_metrics: false,
        initial_prompt: None,
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
        .decode(crate::api::types::base64_payload(b64))
        .map_err(|e| ApiError::Validation(format!("attached audio is not valid base64: {e}")))?;

    let start = std::time::Instant::now();
    let result = decode_and_transcribe(state, bytes, Some(model_name), plain_transcription())
        .await
        .map_err(|(code, e)| match code {
            c if c.is_server_error() => ApiError::Internal(e),
            StatusCode::NOT_FOUND => ApiError::NotFound(e),
            _ => ApiError::Validation(e),
        })?;
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
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> axum::response::Response {
    decode_where_the_model_is(
        state,
        headers,
        body,
        WhisperTask::Transcribe,
        "/v1/audio/transcriptions",
    )
    .await
}

/// An upload goes to a node whose catalogue holds the transcription model when this
/// one's does not; otherwise it is decoded here.
async fn decode_where_the_model_is(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
    task: WhisperTask,
    path: &str,
) -> axum::response::Response {
    use crate::api::handlers::catalogue;
    let requested = catalogue::upload_model(&headers, &body)
        .await
        .unwrap_or_default();
    let holds = |n: &crate::distributed::membership::NodeState| {
        catalogue::serves_transcription(n, &requested)
    };
    let served_here = holds(&state.local_node_state().await);
    if let Some(relayed) = crate::api::handlers::route_upload_to_holder(
        &state,
        &headers,
        if requested.is_empty() {
            "whisper"
        } else {
            &requested
        },
        served_here,
        true,
        0,
        holds,
        path,
        &body,
    )
    .await
    {
        return relayed;
    }
    // The model loads whole on one card: an idle resident is reclaimed to make room, and a
    // node whose cards cannot hold it hands the upload to a peer that has the model.
    let demand = catalogue::listed_size(&state, |id| {
        catalogue::transcription_entry_matches(id, &requested)
    })
    .await
    .map(catalogue::whole_load_demand)
    .unwrap_or(0);
    if let Some(relayed) = crate::api::handlers::route_upload_to_holder(
        &state,
        &headers,
        if requested.is_empty() {
            "whisper"
        } else {
            &requested
        },
        true,
        state.card_holds(demand),
        demand,
        holds,
        path,
        &body,
    )
    .await
    {
        return relayed;
    }
    crate::inference::place::vram_manager::ensure_gpu_headroom(
        "asr",
        demand,
        catalogue::WHOLE_LOAD_RESERVE,
    )
    .await;
    let _job = state.media_note(
        if requested.is_empty() {
            "whisper"
        } else {
            &requested
        },
        "transcription",
    );
    match catalogue::upload_multipart(&headers, body).await {
        Ok(multipart) => audio_decode_endpoint(state, headers, multipart, task).await,
        Err(e) => {
            use axum::response::IntoResponse;
            let code = axum::http::StatusCode::BAD_REQUEST;
            (
                code,
                Json(crate::api::handlers::openai::openai_error_body(code, e)),
            )
                .into_response()
        }
    }
}

/// OpenAI-compatible `/v1/audio/translations`. Same multipart contract as
/// transcriptions, but whisper's `<|translate|>` prompt token forces
/// English output regardless of source language.
pub(crate) async fn audio_translations(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> axum::response::Response {
    decode_where_the_model_is(
        state,
        headers,
        body,
        WhisperTask::Translate,
        "/v1/audio/translations",
    )
    .await
}
