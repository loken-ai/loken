//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Speech bytes (any container the server reads) to text, the way every route that
/// transcribes does it: an empty upload and an oversized one refused, the audio decoded
/// to whisper's mono 16 kHz, the model loaded, the transcript produced. The error carries
/// the status it answers with.
pub(super) async fn decode_and_transcribe(
    state: &APIServer,
    bytes: Vec<u8>,
    model_name: Option<&str>,
    params: AudioTranscribeParams,
) -> Result<TranscribeResult, (StatusCode, String)> {
    // Zero bytes would reach the decoder as an unhelpful "unsupported format"; most
    // clients send them when an aborted streaming upload still fires the request.
    if bytes.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "the audio is empty".to_string()));
    }
    validate_audio_input_size(bytes.len()).map_err(|e| (StatusCode::PAYLOAD_TOO_LARGE, e))?;
    let (samples, sr) = decode_audio_blocking(bytes, "audio").await?;
    ensure_whisper_model_loaded(state, model_name)
        .await
        .map_err(|e| (http_status_for_load_error(&e), e))?;
    state
        .audio_engine
        .transcribe(samples, sr, params)
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("transcribe: {e}"),
            )
        })
}

/// Speech to text on a model that takes the clip in its prompt (an instruction-format
/// speech tower), in the model's transcription mode: the clip alone in the user turn, the
/// language and the transcription marker after it. `None` when `model` is not such a model,
/// so the caller can try a speech recogniser of that name.
async fn transcribe_with_audio_model(
    state: &APIServer,
    headers: &axum::http::HeaderMap,
    model: &str,
    bytes: &[u8],
    language: Option<&str>,
    temperature: f32,
) -> Result<Option<TranscribeResult>, ApiError> {
    use crate::api::handlers::admission::{ensure_prompt_fits, LoadRequest};
    use base64::Engine as _;
    if bytes.is_empty() {
        return Err(ApiError::Validation("the audio is empty".to_string()));
    }
    validate_audio_input_size(bytes.len()).map_err(ApiError::Validation)?;
    let load = LoadRequest::from_headers(headers, state.get_effective_keep_alive(None));
    let engine = match state.load_for_request(model, load).await {
        Ok(engine) => engine,
        Err(ApiError::NotFound(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    let Some(tower) = engine.audio_input().await.filter(|t| t.instruction_format) else {
        return Ok(None);
    };
    let prompt = crate::api::handlers::chat_prompt::transcription_prompt(&tower, language);
    ensure_prompt_fits(&engine, &prompt).await?;
    let priority = crate::api::gate::Priority::from_headers(headers)
        .unwrap_or(crate::api::gate::Priority::Interactive);
    let _admitted = state
        .admit(
            engine.clone(),
            model,
            priority,
            "/v1/audio/transcriptions",
            &[],
        )
        .await?;
    let clip = base64::engine::general_purpose::STANDARD.encode(bytes);
    state.admit_audio(&engine, Some(&clip)).await?;
    let mut params = GenerationParams {
        temperature: Some(temperature),
        repeat_penalty: Some(1.0),
        ..Default::default()
    };
    let _cancel = engine.cancel_guard(&mut params);
    let out = engine
        .generate(&prompt, params)
        .await
        .map_err(|e| ApiError::Internal(format!("transcribe: {e}")))?;
    let text = out.text.trim().to_string();
    let duration_s = audio_header_seconds(bytes).unwrap_or(0.0) as f32;
    Ok(Some(TranscribeResult {
        language: language.unwrap_or_default().to_string(),
        duration_s,
        tokens: Vec::new(),
        segments: vec![crate::inference::engine::audio_engine::WhisperSegment {
            start: 0.0,
            end: duration_s,
            text: text.clone(),
            tokens: Vec::new(),
            compression_ratio: 0.0,
            temperature,
            avg_logprob: 0.0,
            no_speech_prob: 0.0,
        }],
        text,
    }))
}

pub(super) async fn audio_decode_endpoint(
    axum::extract::State(state): axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
    mut multipart: axum::extract::Multipart,
    task: WhisperTask,
) -> axum::response::Response {
    let _inflight = crate::distributed::rate_meter::InFlight::enter();
    use axum::response::IntoResponse;

    let mut file_bytes: Option<Vec<u8>> = None;
    let mut model_name: Option<String> = None;
    let mut language: Option<String> = None;
    let mut temperature: f32 = 0.0;
    let mut response_format = "json".to_string();
    let mut initial_prompt: Option<String> = None;
    // OpenAI accepts `timestamp_granularities[]` as a multipart array.
    // We treat any non-empty value (`segment` or `word`) as a request
    // for segment-level timestamps; word-level is not yet implemented.
    let mut request_timestamps = false;

    let err_resp = |code: axum::http::StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };

    let mut multipart_err: Option<axum::response::Response> = None;
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                multipart_err = Some(err_resp(
                    axum::http::StatusCode::BAD_REQUEST,
                    format!("multipart parse: {e}"),
                ));
                break;
            }
        };
        let name = field.name().unwrap_or("").to_string();
        match name.as_str() {
            "file" => match field.bytes().await {
                Ok(b) => file_bytes = Some(b.to_vec()),
                Err(e) => {
                    multipart_err = Some(err_resp(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!("read file: {e}"),
                    ));
                    break;
                }
            },
            "model" => match field.text().await {
                Ok(t) => {
                    // Empty `model` field -> fall back to the default
                    // (currently whisper-small). Some SDK clients
                    // attach an empty model field by default when the
                    // caller hasn't set one.
                    let trimmed = t.trim();
                    if !trimmed.is_empty() {
                        model_name = Some(trimmed.to_string());
                    }
                }
                Err(e) => {
                    multipart_err = Some(err_resp(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!("read model: {e}"),
                    ));
                    break;
                }
            },
            "language" => match field.text().await {
                Ok(t) => {
                    // Treat empty/whitespace-only as "not provided"
                    // so whisper falls back to auto-detect. Some SDKs
                    // include the field as an empty string by default.
                    let trimmed = t.trim();
                    if !trimmed.is_empty() {
                        language = Some(trimmed.to_string());
                    }
                }
                Err(e) => {
                    multipart_err = Some(err_resp(
                        axum::http::StatusCode::BAD_REQUEST,
                        format!("read language: {e}"),
                    ));
                    break;
                }
            },
            "temperature" => {
                if let Ok(t) = field.text().await {
                    // OpenAI bounds whisper temperature to [0, 1]; we
                    // clamp defensively so a stray negative or 100 value
                    // doesn't break the StdRng softmax path.
                    temperature = clamp_finite_f32(t.parse::<f32>().unwrap_or(0.0), 0.0, 1.0, 0.0);
                }
            }
            "response_format" => {
                if let Ok(t) = field.text().await {
                    // Lowercase normalize so JSON/Text/Verbose_JSON all
                    // hit the same arms as their OpenAI canonical
                    // lowercase form.
                    response_format = t.to_lowercase();
                }
            }
            "timestamp_granularities" | "timestamp_granularities[]" | "timestamps" => {
                if let Ok(t) = field.text().await {
                    let lower = t.to_lowercase();
                    if !lower.is_empty() && lower != "false" && lower != "0" && lower != "none" {
                        request_timestamps = true;
                    }
                }
            }
            // OpenAI's documented `prompt` field - text the model
            // sees as `<|startofprev|>` context to bias vocabulary
            // / continue from a previous segment.
            "prompt" => {
                if let Ok(t) = field.text().await {
                    if !t.trim().is_empty() {
                        initial_prompt = Some(t);
                    }
                }
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    // SRT/VTT only make sense with timestamps; auto-enable so the
    // formatter doesn't fall back to a single full-duration cue.
    if response_format == "srt" || response_format == "vtt" {
        request_timestamps = true;
    }
    // text/json formats don't surface segments - skip the per-segment
    // logprob + timestamp parsing work even if the caller passed
    // `timestamp_granularities[]`. Matches OpenAI's "ignored without
    // verbose_json" semantics and keeps the fast path fast.
    let surface_segments = matches!(response_format.as_str(), "verbose_json" | "srt" | "vtt");
    if !surface_segments {
        request_timestamps = false;
    }
    // Reject unknown response_format up front (matches OpenAI spec),
    // rather than silently falling through to plain JSON in the
    // formatter - clients passing typos get a clear 400 instead.
    match response_format.as_str() {
        "json" | "text" | "verbose_json" | "srt" | "vtt" => {}
        other => {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                format!(
                    "response_format '{other}' not supported; use json|text|verbose_json|srt|vtt"
                ),
            );
        }
    }
    if let Some(r) = multipart_err {
        return r;
    }

    let bytes = match file_bytes {
        Some(b) => b,
        None => {
            return err_resp(
                axum::http::StatusCode::BAD_REQUEST,
                "missing `file` field".to_string(),
            )
        }
    };
    if let Some(name) = model_name.as_deref() {
        if let Err(e) = validate_model_id(name) {
            return e.into_response();
        }
    }
    // A model that takes the clip in its prompt transcribes in its own mode; any other name
    // is a speech recogniser and takes the path below.
    if let Some(name) = model_name
        .as_deref()
        .filter(|n| !family::is_asr(&state.huggingface_models_dir, n))
        .filter(|_| matches!(task, WhisperTask::Transcribe))
    {
        match transcribe_with_audio_model(
            &state,
            &headers,
            name,
            &bytes,
            language.as_deref(),
            temperature,
        )
        .await
        {
            Ok(Some(result)) => return format_transcribe_response(&response_format, &result, task),
            Ok(None) => {}
            Err(e) => return e.into_response(),
        }
    }
    let collect_metrics = matches!(response_format.as_str(), "verbose_json" | "srt" | "vtt");
    let params = AudioTranscribeParams {
        language,
        temperature,
        task,
        timestamps: request_timestamps,
        collect_metrics,
        initial_prompt,
    };
    // The wall time goes out as `Server-Timing` (RFC 8673), for clients diagnosing a slow
    // transcription without server-side log access.
    let transcribe_start = std::time::Instant::now();
    let result = match decode_and_transcribe(&state, bytes, model_name.as_deref(), params).await {
        Ok(r) => r,
        Err((code, e)) => return err_resp(code, e),
    };
    let transcribe_ms = transcribe_start.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Audio {task:?}: duration={duration_s:.1}s lang={lang} transcribe={transcribe_ms:.0}ms",
        duration_s = result.duration_s,
        lang = result.language,
    );

    let mut resp = format_transcribe_response(&response_format, &result, task);
    let st = format!("transcribe;dur={transcribe_ms:.1}");
    if let Ok(hv) = axum::http::HeaderValue::from_str(&st) {
        resp.headers_mut().insert("server-timing", hv);
    }
    resp
}

pub(super) fn format_transcribe_response(
    fmt: &str,
    r: &TranscribeResult,
    task: WhisperTask,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match fmt {
        "text" => (
            [(
                axum::http::header::CONTENT_TYPE,
                "text/plain; charset=utf-8",
            )],
            r.text.clone(),
        )
            .into_response(),
        "srt" => (
            [
                (
                    axum::http::header::CONTENT_TYPE,
                    "application/x-subrip; charset=utf-8",
                ),
                (
                    axum::http::header::CONTENT_DISPOSITION,
                    "inline; filename=\"transcript.srt\"",
                ),
            ],
            format_srt(r),
        )
            .into_response(),
        "vtt" => (
            [
                (axum::http::header::CONTENT_TYPE, "text/vtt; charset=utf-8"),
                (
                    axum::http::header::CONTENT_DISPOSITION,
                    "inline; filename=\"transcript.vtt\"",
                ),
            ],
            format_vtt(r),
        )
            .into_response(),
        "verbose_json" => {
            let task_str = match task {
                WhisperTask::Transcribe => "transcribe",
                WhisperTask::Translate => "translate",
            };
            // When timestamps were requested, `r.segments` carries the
            // real per-utterance boundaries parsed from whisper's
            // timestamp tokens. Otherwise the engine fills in a single
            // full-duration placeholder so SDKs always see `segments`.
            let segments: Vec<serde_json::Value> = r
                .segments
                .iter()
                .enumerate()
                .map(|(i, seg)| {
                    serde_json::json!({
                        "id": i,
                        "seek": (seg.start * 100.0) as u32,
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "tokens": seg.tokens,
                        "temperature": seg.temperature,
                        "avg_logprob": seg.avg_logprob,
                        "compression_ratio": seg.compression_ratio,
                        "no_speech_prob": seg.no_speech_prob,
                    })
                })
                .collect();
            Json(serde_json::json!({
                "task": task_str,
                "language": r.language,
                "duration": r.duration_s,
                "text": r.text,
                "segments": segments,
            }))
            .into_response()
        }
        _ => {
            // Default: json
            Json(serde_json::json!({ "text": r.text })).into_response()
        }
    }
}

/// SRT (SubRip) subtitle format. One cue per parsed whisper segment.
pub(super) fn format_srt(r: &TranscribeResult) -> String {
    let mut out = String::new();
    for (i, seg) in r.segments.iter().enumerate() {
        out.push_str(&(i + 1).to_string());
        out.push('\n');
        out.push_str(&hms_comma(seg.start));
        out.push_str(" --> ");
        out.push_str(&hms_comma(seg.end));
        out.push('\n');
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

/// WebVTT subtitle format. One cue per parsed whisper segment.
pub(super) fn format_vtt(r: &TranscribeResult) -> String {
    let mut out = String::new();
    out.push_str("WEBVTT\n\n");
    for seg in r.segments.iter() {
        out.push_str(&hms_dot(seg.start));
        out.push_str(" --> ");
        out.push_str(&hms_dot(seg.end));
        out.push('\n');
        out.push_str(seg.text.trim());
        out.push_str("\n\n");
    }
    out
}

pub(super) fn hms_comma(t: f32) -> String {
    let ms = (t * 1000.0).round() as u64;
    let h = ms / 3_600_000;
    let m = (ms / 60_000) % 60;
    let s = (ms / 1000) % 60;
    let ms = ms % 1000;
    format!("{h:02}:{m:02}:{s:02},{ms:03}")
}

pub(super) fn hms_dot(t: f32) -> String {
    let ms = (t * 1000.0).round() as u64;
    let h = ms / 3_600_000;
    let m = (ms / 60_000) % 60;
    let s = (ms / 1000) % 60;
    let ms = ms % 1000;
    format!("{h:02}:{m:02}:{s:02}.{ms:03}")
}
