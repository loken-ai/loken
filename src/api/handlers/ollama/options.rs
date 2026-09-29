//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Parse an optional `kv_quant` override from the Ollama `options` field.
/// Returns Some(kv_quant) if the caller asked for a specific KV cache format
/// different from the server default. Accepts "off"/"none", "q8", "q4".
pub(super) fn extract_kv_quant_override(
    options: Option<&serde_json::Value>,
) -> Option<crate::inference::engine::llm_engine::KvQuant> {
    use crate::inference::engine::llm_engine::KvQuant;
    let s = options?.get("kv_quant")?.as_str()?;
    match s.to_ascii_lowercase().as_str() {
        "off" | "none" | "f16" | "f32" => Some(KvQuant::Off),
        "q8" | "q8_0" => Some(KvQuant::Q8),
        "q4" | "q4_0" => Some(KvQuant::Q4),
        _ => None,
    }
}

/// How an Ollama request wants its model loaded: `options.num_ctx` and
/// `options.kv_quant` first, the headers every route reads otherwise.
pub(super) fn ollama_load_request(
    state: &APIServer,
    options: Option<&serde_json::Value>,
    headers: &axum::http::HeaderMap,
    keep_alive: Option<&str>,
) -> crate::api::handlers::admission::LoadRequest {
    let from_headers = crate::api::handlers::admission::LoadRequest::from_headers(
        headers,
        state.get_effective_keep_alive(keep_alive),
    );
    crate::api::handlers::admission::LoadRequest {
        window: options
            .and_then(|o| o.get("num_ctx"))
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize)
            .or(from_headers.window),
        kv_quant: extract_kv_quant_override(options).or(from_headers.kv_quant),
        ..from_headers
    }
}

/// The gate priority an Ollama request asks for: `options.priority`, else the header
/// every route reads, else Fim for a fill-in-the-middle request and Interactive for the
/// rest.
pub(super) fn ollama_priority(
    options: Option<&serde_json::Value>,
    headers: &axum::http::HeaderMap,
    fim: bool,
) -> crate::api::gate::Priority {
    options
        .and_then(|v| v.get("priority"))
        .and_then(serde_json::Value::as_str)
        .and_then(crate::api::gate::Priority::parse)
        .or_else(|| crate::api::gate::Priority::from_headers(headers))
        .unwrap_or(if fim {
            crate::api::gate::Priority::Fim
        } else {
            crate::api::gate::Priority::Interactive
        })
}

/// Range-validate the Ollama options bag with the bounds every dialect shares.
#[cfg(test)]
pub(super) fn validate_ollama_options(options: Option<&serde_json::Value>) -> Result<(), ApiError> {
    crate::api::handlers::sampling::SamplingRequest::from_ollama(options, None, None).validate()
}

/// The engine parameters an Ollama options bag asks for.
#[cfg(test)]
pub(super) fn extract_generation_options(options: Option<&serde_json::Value>) -> GenerationParams {
    crate::api::handlers::sampling::SamplingRequest::from_ollama(options, None, None).into_params()
}

/// Chat (POST /api/chat) - Ollama format
pub(crate) async fn ollama_chat(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    OllamaJson(request): OllamaJson<OllamaChatRequest>,
) -> Result<Response, ApiError> {
    validate_model_id(&request.model)?;
    // Type-level validators on OllamaChatRequest enforce:
    //   - messages.len() <= 4096 (count cap)
    //   - each Message.role / Message.content non-empty (nested)
    //   - each Message.images.len() <= 16 (per-message vision cap)
    // The aggregate-chars cap below is still imperative - validator
    // crate can't express "sum of nested field chars across a Vec".
    validate_request(&request)?;
    let sampling = crate::api::handlers::sampling::SamplingRequest::from_ollama(
        request.options.as_ref(),
        request.logprobs,
        request.top_logprobs,
    );
    sampling.validate()?;
    // Normalize model ID
    let model_name = normalize_model_id(&request.model);
    // A conversation, or a load instruction for a model held elsewhere, goes where the
    // model is; an empty conversation for a model held here stays here. Peers are asked
    // about the prompt this node would prefill, which is the text their caches key on.
    if state.cluster_handle().is_some() {
        let prompt_text = state
            .prepare_prompt(
                &model_name,
                None,
                &crate::api::handlers::chat_prompt::PromptSpec {
                    messages: &request.messages,
                    tools: request.tools.as_ref(),
                    tool_choice: request.tool_choice.as_ref(),
                    thinking: request.thinking_preference(),
                    output_grammar: None,
                },
            )
            .await
            .prompt;
        let local_state = state.local_node_state().await;
        let served_here = crate::distributed::routing::can_serve(&local_state, &model_name);
        if !request.messages.is_empty() || !served_here {
            let max_tokens = request
                .options
                .as_ref()
                .and_then(|o| o.get("num_predict"))
                .and_then(serde_json::Value::as_u64)
                .map_or_else(
                    || crate::distributed::rate_meter::expected_output_tokens(&model_name),
                    |m| m as u32,
                );
            let body = serde_json::to_value(&request)
                .map_err(|e| ApiError::Internal(format!("forward: {e}")))?;
            if let Some(relayed) = crate::api::handlers::route_to_holder(
                &state,
                &headers,
                &model_name,
                &prompt_text,
                max_tokens,
                &crate::api::handlers::OLLAMA_CHAT,
                &body,
            )
            .await
            {
                return Ok(relayed);
            }
        }
    }
    info!(
        "💬 Chat request for model: {} (normalized from: {}), stream: {}",
        model_name, request.model, request.stream
    );

    // Aggregate-chars guard (1024 KiB). Mirror /v1/chat/completions
    // from 5e13fbe. The body limit is sized for media uploads and would
    // otherwise let a pathological multi-megabyte history reach prefill.
    const MESSAGES_TOTAL_MAX_CHARS: usize = 1024 * 1024;
    let total_chars: usize = request
        .messages
        .iter()
        .map(|m| m.content.chars().count())
        .sum();
    if total_chars > MESSAGES_TOTAL_MAX_CHARS {
        return Err(ApiError::Validation(format!(
            "aggregate `messages[*].content` is {total_chars} chars; cap at {MESSAGES_TOTAL_MAX_CHARS} (~256k tokens). Trim history client-side."
        )));
    }
    // Per-message role/content non-empty check (parity with the
    // OpenAI `nested` validator from cde788d). Without this, a
    // request with `[{"role":"user","content":""}]` would format
    // into an empty user turn and the model would generate from
    // a degenerate prompt.
    //
    // Role allowlist matches the /v1/chat/completions handler so an
    // unknown role like "cthulhu" gets a 400 instead of being folded
    // into the chat template as a strange role tag (which most
    // templates either error on or silently fold into the system
    // turn - both are surprising to the caller).
    for (i, m) in request.messages.iter().enumerate() {
        if m.role.trim().is_empty() {
            return Err(ApiError::Validation(format!(
                "messages[{i}].role must not be empty"
            )));
        }
        match m.role.as_str() {
            "system" | "user" | "assistant" | "tool" | "function" | "developer" => {}
            other => {
                return Err(ApiError::Validation(format!(
                    "messages[{i}].role '{other}' not allowed; use system|user|assistant|tool|function|developer"
                )));
            }
        }
        // Empty content is allowed only on assistant turns that carry
        // tool_calls (the OpenAI/Ollama agentic round-trip shape) - every
        // other role still needs text.
        if m.content.is_empty()
            && !(m.role == "assistant" && m.tool_calls.as_ref().is_some_and(|t| !t.is_empty()))
        {
            return Err(ApiError::Validation(format!(
                "messages[{i}].content must not be empty"
            )));
        }
    }
    // `tools` are handled below (injected into the prompt, parsed back
    // out of the model output by crate::api::tool_calls).

    // Handle empty messages (load/unload cases per Ollama spec)
    if request.messages.is_empty() {
        let keep_alive_minutes = state.get_effective_keep_alive(request.keep_alive.as_deref());

        if keep_alive_minutes == 0 {
            // UNLOAD. The outcome is REPORTED, not assumed: answering "unload" whatever
            // happened makes a name nothing recognised read as freed VRAM.
            let outcome = state.unload_model(&model_name).await?;
            // The freed memory may now serve a resident faster; the answer does not wait for it.
            let server = state.clone();
            tokio::spawn(async move { server.migrate_idle_models().await });
            let mut response = OllamaChatResponse::new(
                model_name,
                Message::new("assistant".to_string(), String::new()),
            );
            response.done = true;
            response.done_reason = Some(outcome.done_reason().to_string());
            return Ok(Json(response).into_response());
        } else if keep_alive_minutes > 0 {
            // LOAD - model load via the dedup'd loader (per-model lock,
            // re-check after acquire, no double-engine on race).
            let _ = state.ensure_loaded(&model_name, keep_alive_minutes).await;

            // Return load response
            let mut response = OllamaChatResponse::new(
                model_name,
                Message::new("assistant".to_string(), String::new()),
            );
            response.done = true;
            response.done_reason = Some("load".to_string());
            return Ok(Json(response).into_response());
        }
    }

    // Reject non-decoder pipeline components (CLIP, T5, whisper, parler)
    // before they reach the text engine, where the panic-on-missing-
    // weights path would tear down the worker thread.
    if let Some(hint) = non_chat_pipeline_component(&model_name) {
        return Err(ApiError::Validation(format!(
            "model '{model_name}' is not a chat model: {hint}",
        )));
    }

    // TTS: route to the audio engine with the last user message's text.
    // Returns audio embedded in the response (audios[0] WAV base64) so
    // the chat tab can render an inline playback control. No streaming
    // - TTS engines synth the full utterance up front.
    if crate::api::handlers::family::is_tts(&model_name) {
        let last_user_msg = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .ok_or_else(|| {
                ApiError::Validation(
                    "TTS models require at least one user message with the text to synthesise"
                        .to_string(),
                )
            })?;
        #[cfg(feature = "audio")]
        return handle_chat_tts(
            &state,
            &model_name,
            &last_user_msg.content,
            request.options.as_ref(),
        )
        .await;
    }

    // ASR: take the most recent user-attached audio (carried in
    // images[] for the Ollama chat shape) and return its transcript
    // as the assistant message content.
    #[cfg(feature = "audio")]
    if crate::api::handlers::family::is_asr(&model_name) {
        let last_user_msg = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .ok_or_else(|| {
                ApiError::Validation(
                    "Whisper requires at least one user message with an audio attachment"
                        .to_string(),
                )
            })?;
        let images = last_user_msg.images.clone().unwrap_or_default();
        return handle_chat_asr(&state, &model_name, &images).await;
    }

    // Check if this is an image generation model (e.g., Flux, Z-Image)
    #[cfg(feature = "image")]
    if crate::api::handlers::family::is_image(&model_name) {
        // Extract prompt and images from last user message. Reject
        // explicitly when no user turn exists - the diffusion engine
        // would otherwise generate from an empty prompt and produce
        // nonsense, and the silent unwrap_or_default() hid that
        // misuse from the caller.
        let last_user_msg = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .ok_or_else(|| {
                ApiError::Validation(
                    "image-generation models require at least one user message with a prompt"
                        .to_string(),
                )
            })?;
        let prompt = last_user_msg.content.clone();
        let images = last_user_msg.images.clone();
        // Reuse generate handler logic via a synthetic generate request
        let mut gen_request = OllamaGenerateRequest::new(request.model.clone(), prompt);
        gen_request.stream = request.stream;
        gen_request.options = request.options.clone();
        gen_request.keep_alive = request.keep_alive.clone();
        gen_request.images = images;
        return handle_image_generation(&state, &model_name, &gen_request).await;
    }

    // Extract generation options
    let mut params = sampling.into_params();
    let wants_logprobs = request.logprobs == Some(true);
    // Ollama's `format` field: `"json"` -> json_object grammar; a JSON
    // Schema object -> strict json_schema grammar. Bridges Ollama's
    // structured-output knob to the same llguidance path the OpenAI
    // `response_format` uses. `params.grammar` from options takes
    // precedence so explicit overrides via the options bag still win.
    if params.grammar.is_none() {
        params.grammar = ollama_format_to_grammar(request.format.as_ref());
    }

    // Loaded on first use (Ollama-compatible), in the context and cache format the
    // options or the headers ask for.
    let engine = state
        .load_for_request(
            &model_name,
            ollama_load_request(
                &state,
                request.options.as_ref(),
                &headers,
                request.keep_alive.as_deref(),
            ),
        )
        .await?;

    // The model's template with the tools in its own syntax, the thinking preference,
    // the Qwen3-VL vision sentinel, and the call object as the grammar when a call is
    // required and the caller's `format` set none.
    let crate::api::handlers::chat_prompt::PreparedPrompt {
        prompt,
        tools_active,
        tool_format,
        grammar,
        thinking_opened,
    } = state
        .prepare_prompt(
            &model_name,
            Some(&engine),
            &crate::api::handlers::chat_prompt::PromptSpec {
                messages: &request.messages,
                tools: request.tools.as_ref(),
                tool_choice: request.tool_choice.as_ref(),
                thinking: request.thinking_preference(),
                output_grammar: params.grammar.take(),
            },
        )
        .await;
    params.grammar = grammar;
    // Length only: the prompt itself is user content and is never written to a log.
    debug!("Chat prompt: {} chars", prompt.len());

    // Log applied parameters
    if params.max_tokens.is_some() {
        info!("Chat: max_tokens override: {:?}", params.max_tokens);
    }
    if params.temperature.is_some() {
        info!("Chat: temperature override: {:?}", params.temperature);
    }
    if params.top_p.is_some() {
        info!("Chat: top_p override: {:?}", params.top_p);
    }
    if !params.stop_sequences.is_empty() {
        info!("Chat: {} stop sequence(s)", params.stop_sequences.len());
    }

    crate::api::handlers::admission::ensure_prompt_fits(&engine, &prompt).await?;
    // The gate, the in-flight count and the last user turn's images.
    let admission = state
        .admit(
            engine.clone(),
            &model_name,
            ollama_priority(request.options.as_ref(), &headers, false),
            "/api/chat",
            &crate::api::handlers::chat_prompt::last_user_images(&request.messages),
        )
        .await?;

    // Check if streaming is requested
    if request.stream {
        let max_tokens_cap = params.max_tokens;
        // Use streaming generation (NDJSON format, matching Ollama's API)
        match engine.generate_stream(&prompt, params.clone()).await {
            Ok(mut rx) => {
                let model_name_clone = model_name.clone();
                let start_time = std::time::Instant::now();
                let prompt_clone = prompt.clone();
                let engine_for_stats = engine.clone();
                let stream = async_stream::stream! {
                    let _admitted = admission;
                    let mut accumulated_text = String::new();
                    let mut chunk_count: usize = 0;
                    let mut errored = false;
                    let mut tool_scanner = if tools_active {
                        Some(crate::api::tool_calls::StreamToolScanner::new(tool_format))
                    } else {
                        None
                    };
                    let mut splitter = crate::api::thinking::ThinkSplit::opened(thinking_opened);
                    let mut thought = false;
                    let mut thinking_ns: Option<u64> = None;

                    while let Some(result) = rx.recv().await {
                        match result {
                            Ok(chunk) => {
                                accumulated_text.push_str(&chunk);
                                chunk_count += 1;
                                for seg in splitter.push(&chunk) {
                                    let message = match seg {
                                        crate::api::thinking::Segment::Thinking(t) => {
                                            thought = true;
                                            serde_json::json!({"role": "assistant", "content": "", "thinking": t})
                                        }
                                        crate::api::thinking::Segment::Content(c) => {
                                            if thought && thinking_ns.is_none() {
                                                thinking_ns = Some(start_time.elapsed().as_nanos() as u64);
                                            }
                                            let emit = match tool_scanner.as_mut() {
                                                Some(sc) => sc.push(&c),
                                                None => c,
                                            };
                                            if emit.is_empty() {
                                                continue;
                                            }
                                            serde_json::json!({"role": "assistant", "content": emit})
                                        }
                                    };
                                    let mut response_chunk = serde_json::json!({
                                        "model": model_name_clone,
                                        "created_at": chrono::Utc::now().to_rfc3339(),
                                        "message": message,
                                        "done": false
                                    });
                                    if wants_logprobs {
                                        let drawn = engine_for_stats.take_logprobs();
                                        if !drawn.is_empty() {
                                            response_chunk["logprobs"] = ollama_logprobs(&drawn);
                                        }
                                    }
                                    let mut line = response_chunk.to_string();
                                    line.push('\n');
                                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                                }
                            }
                            Err(e) => {
                                error!("Streaming chat chunk error: {}", e);
                                let err_chunk = serde_json::json!({ "error": e });
                                let mut line = err_chunk.to_string();
                                line.push('\n');
                                yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                                errored = true;
                                break;
                            }
                        }
                    }

                    // Final done message with timing and token counts.
                    // Skipped on error so SDK clients don't see two
                    // terminal markers (`error` + `done`) on one stream.
                    if !errored {
                        for seg in splitter.finish() {
                            let message = match seg {
                                crate::api::thinking::Segment::Thinking(t) => {
                                    serde_json::json!({"role": "assistant", "content": "", "thinking": t})
                                }
                                crate::api::thinking::Segment::Content(c) => {
                                    let emit = match tool_scanner.as_mut() {
                                        Some(sc) => sc.push(&c),
                                        None => c,
                                    };
                                    if emit.is_empty() {
                                        continue;
                                    }
                                    serde_json::json!({"role": "assistant", "content": emit})
                                }
                            };
                            let frame = serde_json::json!({
                                "model": model_name_clone,
                                "created_at": chrono::Utc::now().to_rfc3339(),
                                "message": message,
                                "done": false
                            });
                            let mut line = frame.to_string();
                            line.push('\n');
                            yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                        }
                        let wall_total_ns = start_time.elapsed().as_nanos() as u64;
                        let stats = engine_for_stats.take_last_stream_stats();
                        let (
                            eval_count, eval_duration, prompt_eval_count,
                            prompt_eval_duration, total_duration, done_from_engine,
                        ) = match stats {
                            Some(s) => (
                                s.eval_count,
                                s.eval_duration_ns,
                                s.prompt_eval_count,
                                s.prompt_eval_duration_ns,
                                s.total_duration_ns,
                                Some(s.finish_reason.ollama()),
                            ),
                            None => (
                                chunk_count as u64,
                                wall_total_ns,
                                estimate_token_count(&prompt_clone),
                                0u64,
                                wall_total_ns,
                                None,
                            ),
                        };
                        let load_duration =
                            total_duration.saturating_sub(prompt_eval_duration + eval_duration);
                        // `done_reason`: "length" when chunk_count
                        // (≈ tokens emitted) reached the requested cap,
                        // "stop" otherwise. Same semantics as /v1/*
                        // streams - matches Ollama's documented field.
                        // Tool calls (if any) are surfaced on the final
                        // message - Ollama clients read message.tool_calls
                        // off the terminal done frame.
                        let tool_calls = tool_scanner
                            .as_ref()
                            .map(|sc| sc.finalize())
                            .unwrap_or_default();
                        let used_tools = !tool_calls.is_empty();
                        // Flush any withheld buffer that turned out not to
                        // be a tool call (bare-JSON false trigger).
                        if !used_tools {
                            if let Some(tail) = tool_scanner
                                .as_ref()
                                .map(|sc| sc.unstreamed())
                                .filter(|t| !t.is_empty())
                            {
                                let flush = serde_json::json!({
                                    "model": model_name_clone,
                                    "created_at": chrono::Utc::now().to_rfc3339(),
                                    "message": {"role": "assistant", "content": tail},
                                    "done": false
                                });
                                let mut l = flush.to_string();
                                l.push('\n');
                                yield Ok::<_, std::io::Error>(axum::body::Bytes::from(l));
                            }
                        }
                        let done_reason = done_from_engine.unwrap_or(match max_tokens_cap {
                            Some(cap) if chunk_count >= cap => "length",
                            _ => "stop",
                        });
                        let mut message = serde_json::json!({
                            "role": "assistant",
                            "content": ""
                        });
                        if used_tools {
                            message["tool_calls"] = serde_json::to_value(&tool_calls)
                                .unwrap_or(serde_json::Value::Null);
                        }
                        let final_chunk = serde_json::json!({
                            "model": model_name_clone,
                            "created_at": chrono::Utc::now().to_rfc3339(),
                            "message": message,
                            "done": true,
                            "thinking_duration": thinking_ns,
                            "done_reason": done_reason,
                            "total_duration": total_duration,
                            "load_duration": load_duration,
                            "prompt_eval_count": prompt_eval_count,
                            "prompt_eval_duration": prompt_eval_duration,
                            "eval_count": eval_count,
                            "eval_duration": eval_duration
                        });
                        let mut line = final_chunk.to_string();
                        line.push('\n');
                        yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                    }
                };

                Ok(Response::builder()
                    .header("content-type", "application/x-ndjson")
                    .body(axum::body::Body::from_stream(stream))
                    .unwrap())
            }
            Err(e) => {
                error!("Generation error (streaming chat): {}", e);
                Err(ApiError::Internal(format!("generate_stream: {e}")))
            }
        }
    } else {
        // Use non-streaming generation
        let start = std::time::Instant::now();
        let _admitted = admission;
        let energy = crate::energy_report::begin();

        let _cancel = engine.cancel_guard();
        match engine
            .generate(&prompt, params)
            .await
            .map_err(|e| e.to_string())
        {
            Ok(result) => {
                crate::energy_report::end(energy, "text", "[/api/chat]");
                let total_duration = start.elapsed().as_nanos() as u64;

                // Tool calling: lift markers out of the output into
                // structured tool_calls (Ollama surfaces these on
                // message.tool_calls). Plain answers pass through.
                let (chat_thinking, chat_body) =
                    crate::api::thinking::split_thinking_opened(thinking_opened, &result.text);
                let (msg, _tool_called) = if tools_active {
                    let parsed = crate::api::tool_calls::parse_tool_calls(tool_format, &chat_body);
                    if parsed.calls.is_empty() {
                        (
                            Message::new("assistant".to_string(), chat_body.clone()),
                            false,
                        )
                    } else {
                        let mut m = Message::new("assistant".to_string(), parsed.content);
                        m.tool_calls = Some(parsed.calls);
                        (m, true)
                    }
                } else {
                    (
                        Message::new("assistant".to_string(), chat_body.clone()),
                        false,
                    )
                };
                let mut msg = msg;
                let thinking_duration = thinking_share_ns(
                    &engine,
                    chat_thinking.as_deref(),
                    result.eval_count,
                    result.eval_duration,
                )
                .await;
                msg.thinking = chat_thinking;
                let mut response = OllamaChatResponse::new(model_name, msg);
                if !result.logprobs.is_empty() {
                    response.logprobs = Some(ollama_logprobs(&result.logprobs));
                }
                response.thinking_duration = thinking_duration;
                // "length" when we hit max_tokens, "stop" otherwise  -
                // mirrors the streaming-path fix in a338be1.
                let done_reason = result.finish_reason.ollama();
                response.done_reason = Some(done_reason.to_string());
                response.total_duration = Some(total_duration);
                response.load_duration = Some(
                    total_duration
                        .saturating_sub(result.prompt_eval_duration + result.eval_duration),
                );
                response.prompt_eval_count = Some(result.prompt_eval_count);
                response.prompt_eval_duration = Some(result.prompt_eval_duration);
                response.eval_count = Some(result.eval_count);
                response.eval_duration = Some(result.eval_duration);
                Ok(Json(response).into_response())
            }
            Err(e) => {
                error!("Generation error (non-streaming chat): {}", e);
                Err(ApiError::Internal(format!("generate: {e}")))
            }
        }
    }
}

/// Generate (POST /api/generate) - Ollama format
/// The prompt as the engine will actually see it: the model's own chat template applied.
///
/// Shared with the cluster's prefix endpoint, which is the whole reason it is a function. A peer
/// asked "how much of this prompt do you hold" has to tokenise the SAME text the generation path
/// would, or the two sequences differ from their first token and the honest answer is always
/// zero - which is what the endpoint reported until this was pulled out of the handler.
pub(crate) async fn templated_prompt(
    state: &APIServer,
    model_name: &str,
    prompt: &str,
    system: Option<&str>,
    template: Option<&str>,
) -> String {
    // A template given with the request comes first: Jinja rendered as such, the Go
    // form rendered for this one exchange, anything else falling through to the model's.
    if let Some(t) = template.filter(|t| !t.trim().is_empty()) {
        let msgs = generate_messages(system, prompt);
        if t.contains("{%") {
            return format_chat_prompt(&msgs, Some(t));
        }
        if let Some(rendered) = super::super::prompt_format::render_go_template(t, system, prompt) {
            return rendered;
        }
    }
    // A vocabulary-chosen template stands in when the model declares none: Ollama applies
    // the template by default and takes raw=true as the opt-out.
    let chat_template = state.chat_template_for(model_name).await;
    match chat_template.as_deref() {
        Some(tmpl) if !prompt_already_templated(prompt, tmpl) => {
            let wrapped = format_chat_prompt(&generate_messages(system, prompt), Some(tmpl));
            debug!(
                "Generate: applied chat template ({} -> {} chars)",
                prompt.len(),
                wrapped.len()
            );
            wrapped
        }
        _ => prompt.to_string(),
    }
}

/// The prompt `/api/generate` prefills. A fill-in-the-middle request (a `suffix`, or the
/// sentinels already in the prompt) bypasses the template; a raw or empty prompt is taken
/// as given; anything else is rendered in the model's chat template, which a chat-tuned
/// model needs or it can answer EOS first. The thinking preference is applied as on the
/// chat route.
pub(crate) async fn generate_prompt(
    state: &APIServer,
    model_name: &str,
    request: &OllamaGenerateRequest,
) -> String {
    let suffix = request.suffix.as_deref().filter(|s| !s.is_empty());
    let mut prompt = if let Some(suffix) = suffix {
        format!(
            "<|fim_prefix|>{}<|fim_suffix|>{suffix}<|fim_middle|>",
            request.prompt
        )
    } else if request.prompt.contains("<|fim_middle|>")
        || request.raw.unwrap_or(false)
        || request.prompt.is_empty()
    {
        request.prompt.clone()
    } else {
        templated_prompt(
            state,
            model_name,
            &request.prompt,
            request.system.as_deref(),
            request.template.as_deref(),
        )
        .await
    };
    let template = match request.template.as_deref().filter(|t| !t.trim().is_empty()) {
        Some(t) => Some(t.to_string()),
        None => state.chat_template_for(model_name).await,
    };
    apply_thinking_preference(
        &mut prompt,
        request.thinking_preference(),
        template.as_deref(),
    );
    prompt
}

/// The one exchange `/api/generate` renders: an optional system turn and the prompt.
fn generate_messages(system: Option<&str>, prompt: &str) -> Vec<Message> {
    let mut msgs = Vec::with_capacity(2);
    if let Some(sys) = system.filter(|s| !s.trim().is_empty()) {
        msgs.push(Message::new("system".to_string(), sys.to_string()));
    }
    msgs.push(Message::new("user".to_string(), prompt.to_string()));
    msgs
}

pub(crate) async fn ollama_generate(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    OllamaJson(request): OllamaJson<OllamaGenerateRequest>,
) -> Result<Response, ApiError> {
    validate_model_id(&request.model)?;
    // Type-level validators on OllamaGenerateRequest enforce:
    //   - prompt <= 262144 bytes (256 KiB ≈ 64k tokens; byte cap is
    //     stricter than the original char cap for multi-byte UTF-8
    //     but that's the right belt-and-braces direction).
    //   - images.len() <= 16 (vision encoder cost gate).
    // validate_request humanizes the error so clients see
    // "`prompt` length must be in [..., 262144]" instead of the raw
    // validator JSON debug form.
    validate_request(&request)?;
    let sampling = crate::api::handlers::sampling::SamplingRequest::from_ollama(
        request.options.as_ref(),
        request.logprobs,
        request.top_logprobs,
    );
    sampling.validate()?;
    // Normalize model ID
    let model_name = normalize_model_id(&request.model);

    // -- Cluster: serve here, or hand the whole request to a better-placed peer --------
    //
    // Read BEFORE any decision: the marker is what stops two nodes that each prefer the
    // other from passing a request back and forth until something times out, and a hang is
    // a far worse failure than an imperfect placement.
    // Counted from the first instant, so concurrent arrivals see each other in `busy` when
    // they decide - the admission gate never sees this path, and a count taken any later
    // publishes an idle node under any load. Released explicitly on hand-over: a forwarded
    // request is the peer's work, not ours.
    let inflight_guard = crate::distributed::rate_meter::InFlight::enter();
    if let Some(cluster) = state.cluster_handle() {
        // An empty prompt is a load or unload instruction addressed to THIS node, and stays
        // here as long as this node holds the model. A model it does not hold can only be
        // loaded where it is, so that instruction travels to a holder like a generation.
        let local_state = state.local_node_state().await;
        let served_here = crate::distributed::routing::can_serve(&local_state, &model_name);
        if !request.prompt.is_empty() || !served_here {
            // `options` is free-form JSON here, so the cap is read by name; a missing value
            // falls back to the default the generation path uses.
            let max_tokens = request
                .options
                .as_ref()
                .and_then(|o| o.get("num_predict"))
                .and_then(serde_json::Value::as_u64)
                .map_or_else(
                    || crate::distributed::rate_meter::expected_output_tokens(&model_name),
                    |m| m as u32,
                );
            // The request is re-serialised from the parsed form rather than relayed as raw
            // bytes. Both nodes run the same build, so a field one understands the other
            // does too.
            let body = serde_json::to_value(&request)
                .map_err(|e| ApiError::Internal(format!("forward: {e}")))?;
            if let Some(relayed) = crate::api::handlers::route_to_holder(
                &state,
                &headers,
                &model_name,
                &generate_prompt(&state, &model_name, &request).await,
                max_tokens,
                &crate::api::handlers::OLLAMA_GENERATE,
                &body,
            )
            .await
            {
                drop(inflight_guard);
                return Ok(relayed);
            }
        }
        let _ = cluster;
    }
    info!(
        "📤 Generate request for model: {} (normalized from: {}), stream: {}",
        model_name, request.model, request.stream
    );

    // Handle empty prompt (load/unload cases per Ollama spec)
    if request.prompt.is_empty() {
        let keep_alive_minutes = state.get_effective_keep_alive(request.keep_alive.as_deref());

        if keep_alive_minutes == 0 {
            // UNLOAD. The result is REPORTED, not swallowed: `.ok()` here answered
            // "unload" with a 200 even when nothing matched and the weights stayed in
            // VRAM, so the one way to free a card looked like it worked while the card
            // stayed full. A model that was not loaded is not an error - there is
            // nothing to free - but it is not an unload either, and a failure to unload
            // one that IS loaded must say so.
            let outcome = state.unload_model(&model_name).await?;
            // The freed memory may now serve a resident faster; the answer does not wait for it.
            let server = state.clone();
            tokio::spawn(async move { server.migrate_idle_models().await });
            let mut response = OllamaGenerateResponse::new(model_name, String::new());
            response.done = true;
            response.done_reason = Some(outcome.done_reason().to_string());
            return Ok(Json(response).into_response());
        } else if keep_alive_minutes > 0 {
            // Image models (Flux, Z-Image) don't use the GGUF engine path.
            // Return a "load" response and let the actual generation call handle loading.
            #[cfg(feature = "image")]
            if crate::api::handlers::family::is_image(&model_name) {
                info!(
                    "Image model load request for {} - will load on first generation",
                    model_name
                );
                let mut response =
                    OllamaGenerateResponse::new(model_name, "Image model ready".to_string());
                response.done = true;
                response.done_reason = Some("load".to_string());
                return Ok(Json(response).into_response());
            }

            // LOAD - model load (LLM path) via the dedup'd loader.
            let mut load_successful = false;
            let mut load_error = String::new();
            match state.ensure_loaded(&model_name, keep_alive_minutes).await {
                Ok(()) => {
                    load_successful = true;
                    info!("✅ Model {} ready", model_name);
                }
                Err(e) => {
                    load_error = e;
                    info!("❌ Model {} load failed: {}", model_name, load_error);
                }
            }

            // Return load response with actual status and error details
            let response_message = if load_successful {
                format!("Model {} loaded successfully", model_name)
            } else if !load_error.is_empty() {
                load_error
            } else {
                format!("Model {} load failed", model_name)
            };
            let mut response = OllamaGenerateResponse::new(model_name.clone(), response_message);
            response.done = true;

            if load_successful {
                response.done_reason = Some("load".to_string());
                info!(
                    "📋 Load operation completed successfully for {}",
                    model_name
                );
            } else {
                response.done_reason = Some("error".to_string());
                info!("⚠️  Load operation failed for {}", model_name);
            }

            return Ok(Json(response).into_response());
        }
    }

    // Same guard as ollama_chat - reject component-only repos before
    // they panic the text engine.
    if let Some(hint) = non_chat_pipeline_component(&model_name) {
        return Err(ApiError::Validation(format!(
            "model '{model_name}' is not a generation model: {hint}",
        )));
    }

    // TTS routing - same shape as image-gen, returns audio in the
    // response payload. /api/generate's `prompt` field maps directly
    // to the TTS input text.
    if crate::api::handlers::family::is_tts(&model_name) {
        #[cfg(feature = "audio")]
        return handle_generate_tts(&state, &model_name, &request.prompt).await;
    }

    // Check if this is an image generation model (e.g., Flux, Z-Image)
    #[cfg(feature = "image")]
    if crate::api::handlers::family::is_image(&model_name) {
        return handle_image_generation(&state, &model_name, &request).await;
    }

    // Loaded on first use (Ollama-compatible), in the context and cache format the
    // options or the headers ask for.
    let engine = state
        .load_for_request(
            &model_name,
            ollama_load_request(
                &state,
                request.options.as_ref(),
                &headers,
                request.keep_alive.as_deref(),
            ),
        )
        .await?;

    // Detect fill-in-the-middle (FIM) intent. Either the caller set `suffix`
    // (Ollama-style FIM - we auto-wrap), or the prompt already contains the
    // `<|fim_middle|>` sentinel (caller pre-wrapped). FIM bypasses the chat
    // template regardless of `raw`.
    let has_suffix = request
        .suffix
        .as_deref()
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let has_sentinels = request.prompt.contains("<|fim_middle|>");
    let is_fim = has_suffix || has_sentinels;
    if is_fim {
        info!(
            "Generate: FIM request (suffix={}, sentinels_in_prompt={})",
            has_suffix, has_sentinels
        );
    }

    // Ollama parity: the model's chat template unless the caller set `raw: true` or the
    // prompt is already templated, the sentinels for FIM, and the thinking preference.
    let effective_prompt = generate_prompt(&state, &model_name, &request).await;
    let thinking_opened = crate::api::thinking::prompt_opens_thinking(&effective_prompt);

    // Extract generation options
    let mut params = sampling.into_params();
    let wants_logprobs = request.logprobs == Some(true);
    // Ollama continuation: the context a previous reply returned is the token prefix
    // this generation resumes from - replayed as tokens, never re-detokenised.
    if let Some(ctx) = request.context.as_ref() {
        params.prefix_tokens = Some(ctx.iter().map(|&t| t as u32).collect());
    }
    // Honour Ollama's `format` field as a grammar selector. See the
    // matching wire-up in ollama_chat (line ~1623).
    if params.grammar.is_none() {
        params.grammar = ollama_format_to_grammar(request.format.as_ref());
    }

    // FIM fast-path defaults: short max_tokens, low temperature, sentinel stops.
    if is_fim {
        if params.max_tokens.is_none() {
            params.max_tokens = Some(128);
        }
        if params.temperature.is_none() {
            params.temperature = Some(0.2);
        }
        for stop in [
            "<|endoftext|>",
            "<|fim_pad|>",
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|file_sep|>",
        ] {
            if !params.stop_sequences.iter().any(|s| s == stop) {
                params.stop_sequences.push(stop.to_string());
            }
        }
    }

    if params.max_tokens.is_some() {
        info!("Generate: max_tokens override: {:?}", params.max_tokens);
    }
    if params.temperature.is_some() {
        info!("Generate: temperature override: {:?}", params.temperature);
    }
    if params.top_p.is_some() {
        info!("Generate: top_p override: {:?}", params.top_p);
    }
    if !params.stop_sequences.is_empty() {
        info!("Generate: {} stop sequence(s)", params.stop_sequences.len());
    }

    crate::api::handlers::admission::ensure_prompt_fits(&engine, &effective_prompt).await?;
    // The gate (FIM auto-promotes its priority), the in-flight count and the images.
    let admission = state
        .admit(
            engine.clone(),
            &model_name,
            ollama_priority(request.options.as_ref(), &headers, is_fim),
            "/api/generate",
            request.images.as_deref().unwrap_or_default(),
        )
        .await?;
    drop(inflight_guard);

    // Check if streaming is requested
    if request.stream {
        let max_tokens_cap_gen = params.max_tokens;
        // Use streaming generation (NDJSON format, matching Ollama's API)
        match engine
            .generate_stream(&effective_prompt, params.clone())
            .await
        {
            Ok(mut rx) => {
                let model_name_clone = model_name.clone();
                let start_time = std::time::Instant::now();
                let prompt_clone = effective_prompt.clone();
                // Clone engine handle so the stream closure can read the
                // compute-only stats after consuming all chunks
                // (bench-fairness fix - eval_duration matches Ollama).
                let engine_for_stats = engine.clone();
                let stream = async_stream::stream! {
                    let _admitted = admission; // released when the stream ends
                    let mut splitter = crate::api::thinking::ThinkSplit::opened(thinking_opened);
                    let mut thought = false;
                    let mut thinking_ns: Option<u64> = None;
                    let mut accumulated_text = String::new();
                    let mut chunk_count: usize = 0;
                    let mut errored = false;
                    while let Some(result) = rx.recv().await {
                        match result {
                            Ok(chunk) => {
                                accumulated_text.push_str(&chunk);
                                chunk_count += 1;
                                for seg in splitter.push(&chunk) {
                                    let (response, thinking) = match seg {
                                        crate::api::thinking::Segment::Thinking(t) => {
                                            thought = true;
                                            (String::new(), Some(t))
                                        }
                                        crate::api::thinking::Segment::Content(c) => {
                                            if thought && thinking_ns.is_none() {
                                                thinking_ns = Some(start_time.elapsed().as_nanos() as u64);
                                            }
                                            (c, None)
                                        }
                                    };
                                    let mut response_chunk = serde_json::json!({
                                        "model": model_name_clone,
                                        "created_at": chrono::Utc::now().to_rfc3339(),
                                        "response": response,
                                        "done": false
                                    });
                                    if wants_logprobs {
                                        let drawn = engine_for_stats.take_logprobs();
                                        if !drawn.is_empty() {
                                            response_chunk["logprobs"] = ollama_logprobs(&drawn);
                                        }
                                    }
                                    if let Some(t) = thinking {
                                        response_chunk["thinking"] = serde_json::Value::String(t);
                                    }
                                    let mut line = response_chunk.to_string();
                                    line.push('\n');
                                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                                }
                            }
                            Err(e) => {
                                error!("Streaming generate chunk error: {}", e);
                                let err_chunk = serde_json::json!({
                                    "error": format!("Generation error: {}", e)
                                });
                                let mut line = err_chunk.to_string();
                                line.push('\n');
                                yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                                errored = true;
                                break;
                            }
                        }
                    }

                    // Final done message with timing and token counts;
                    // skipped on error to avoid emitting two terminal
                    // markers on the same stream (mirrors /api/chat).
                    if !errored {
                        for seg in splitter.finish() {
                            let (response, thinking) = match seg {
                                crate::api::thinking::Segment::Thinking(t) => (String::new(), Some(t)),
                                crate::api::thinking::Segment::Content(c) => (c, None),
                            };
                            let mut frame = serde_json::json!({
                                "model": model_name_clone,
                                "created_at": chrono::Utc::now().to_rfc3339(),
                                "response": response,
                                "done": false
                            });
                            if let Some(t) = thinking {
                                frame["thinking"] = serde_json::Value::String(t);
                            }
                            let mut line = frame.to_string();
                            line.push('\n');
                            yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                        }
                        let wall_total_ns = start_time.elapsed().as_nanos() as u64;
                        let done_reason = match max_tokens_cap_gen {
                            Some(cap) if chunk_count >= cap => "length",
                            _ => "stop",
                        };
                        // Report the engine's own compute timing rather than wall-clock.
                        // A caller with no server-side figure has to fall back to the time
                        // it observed, which includes the HTTP yield - so the same work
                        // reads as slower for reasons that have nothing to do with the
                        // engine. Every Ollama-compatible client expects these fields.
                        let stats = engine_for_stats.take_last_stream_stats();
                        let (done_from_engine, context_tokens) = match stats.as_ref() {
                            Some(s) => (Some(s.finish_reason.ollama()), s.context_tokens.clone()),
                            None => (None, Vec::new()),
                        };
                        let done_reason = done_from_engine.unwrap_or(done_reason);
                        let (
                            eval_count, eval_duration, prompt_eval_count,
                            prompt_eval_duration, total_duration,
                        ) = match stats {
                            Some(s) => (
                                s.eval_count,
                                s.eval_duration_ns,
                                s.prompt_eval_count,
                                s.prompt_eval_duration_ns,
                                s.total_duration_ns,
                            ),
                            None => (
                                chunk_count as u64,
                                wall_total_ns,
                                estimate_token_count(&prompt_clone),
                                0u64,
                                wall_total_ns,
                            ),
                        };
                        let final_chunk = serde_json::json!({
                            "model": model_name_clone,
                            "created_at": chrono::Utc::now().to_rfc3339(),
                            "response": "",
                            "done": true,
                            "thinking_duration": thinking_ns,
                            "context": context_tokens,
                            "done_reason": done_reason,
                            "total_duration": total_duration,
                            "prompt_eval_count": prompt_eval_count,
                            "prompt_eval_duration": prompt_eval_duration,
                            "eval_count": eval_count,
                            "eval_duration": eval_duration
                        });
                        let mut line = final_chunk.to_string();
                        line.push('\n');
                        yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                    }
                };

                Ok(Response::builder()
                    .header("content-type", "application/x-ndjson")
                    .body(axum::body::Body::from_stream(stream))
                    .unwrap())
            }
            Err(e) => {
                error!("Generation error (streaming generate): {}", e);
                Err(ApiError::Internal(format!("generate_stream: {e}")))
            }
        }
    } else {
        // Use non-streaming generation
        let start = std::time::Instant::now();
        let _admitted = admission;
        let energy = crate::energy_report::begin();

        let _cancel = engine.cancel_guard();
        match engine
            .generate(&effective_prompt, params)
            .await
            .map_err(|e| e.to_string())
        {
            Ok(result) => {
                crate::energy_report::end(energy, "text", "[/api/generate]");
                let total_duration = start.elapsed().as_nanos() as u64;

                let (thinking, answer) =
                    crate::api::thinking::split_thinking_opened(thinking_opened, &result.text);
                let mut response = OllamaGenerateResponse::new(model_name, answer);
                if !result.logprobs.is_empty() {
                    response.logprobs = Some(ollama_logprobs(&result.logprobs));
                }
                response.thinking_duration = thinking_share_ns(
                    &engine,
                    thinking.as_deref(),
                    result.eval_count,
                    result.eval_duration,
                )
                .await;
                response.thinking = thinking;
                let done_reason = result.finish_reason.ollama();
                response.done_reason = Some(done_reason.to_string());
                response.total_duration = Some(total_duration);
                response.load_duration = Some(
                    total_duration
                        .saturating_sub(result.prompt_eval_duration + result.eval_duration),
                );
                // The field existed and was never filled, so every reply carried
                // `null` - a valid value, which is why a client continuing a
                // conversation from it lost its history each turn, in silence.
                if !result.tokens.is_empty() {
                    response.context = Some(result.tokens.iter().map(|&t| t as i32).collect());
                }
                response.prompt_eval_count = Some(result.prompt_eval_count);
                response.prompt_eval_duration = Some(result.prompt_eval_duration);
                response.eval_count = Some(result.eval_count);
                response.eval_duration = Some(result.eval_duration);
                Ok(Json(response).into_response())
            }
            Err(e) => {
                error!("Generation error (non-streaming generate): {}", e);
                Err(ApiError::Internal(format!("generate: {e}")))
            }
        }
    }
}

/// The share of a whole generation's decode time that went to the reasoning, by its
/// share of the tokens: a whole answer arrives at once, so the boundary is not timed.
async fn thinking_share_ns(
    engine: &std::sync::Arc<LlmEngine>,
    thinking: Option<&str>,
    eval_count: u64,
    eval_duration: u64,
) -> Option<u64> {
    let text = thinking?;
    let thought = engine.count_tokens(text).await? as u64;
    if eval_count == 0 {
        return None;
    }
    Some(eval_duration.saturating_mul(thought.min(eval_count)) / eval_count)
}
