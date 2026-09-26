//! OpenAI-compatible /v1/* handlers: chat/completions, embeddings, rerank,
//! models listing, and grammar (response_format) helpers.

use super::*;

/// The string a request field holds, empty when it is absent or is not a string.
///
/// The endpoints in this module read raw bodies, and to all of them an absent field and a
/// field of the wrong type mean the same thing - the caller said nothing - with validation
/// downstream deciding whether that was allowed. Stating it once keeps a field that is
/// checked on one route from being silently optional on another.
fn text_field(body: &serde_json::Value, key: &str) -> String {
    body.get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default()
        .to_string()
}

/// Milliseconds from the stream opening to its first token.
///
/// The wait before that token is prefill and is what a user actually feels; keeping it
/// apart from the decode rate is what tells a slow prefill from a slow decode. Zero when
/// no token was ever produced.
fn time_to_first_token_ms(
    stream_started: std::time::Instant,
    first_token_at: Option<std::time::Instant>,
) -> u128 {
    first_token_at.map_or(0, |t| t.duration_since(stream_started).as_millis())
}

/// The decode rate to append to a stream's summary line, or nothing to append.
///
/// Measured from the first token onward, for the reason above. A stream that produced one
/// token has no interval to divide by, and reports no rate rather than an invented one.
fn decode_rate_suffix(first_token_at: Option<std::time::Instant>, tokens: u64) -> String {
    let Some(first) = first_token_at.filter(|_| tokens > 1) else {
        return String::new();
    };
    let decode_ms = first.elapsed().as_millis();
    if decode_ms == 0 {
        return String::new();
    }
    format!(
        " ({:.1} tok/s)",
        ((tokens - 1) as f64) * 1000.0 / decode_ms as f64
    )
}

/// Whole answers of `runs` generations gathered into one response: the first answer's
/// fields, every answer's choices indexed in order, and the usage summed - the prompt
/// once when every run read the same prompt, per run when `sum_prompts`.
async fn merge_runs<F, Fut>(
    runs: usize,
    sum_prompts: bool,
    mut run: F,
) -> Result<Response, ApiError>
where
    F: FnMut(usize) -> Fut,
    Fut: std::future::Future<Output = Result<Response, ApiError>>,
{
    let mut merged: Option<serde_json::Value> = None;
    let mut choices = Vec::new();
    let (mut prompt_tokens, mut completion_tokens) = (0i64, 0i64);
    for i in 0..runs {
        let resp = run(i).await?;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .map_err(|e| ApiError::Internal(format!("n: {e}")))?;
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| ApiError::Internal(format!("n: {e}")))?;
        for c in v["choices"].as_array().into_iter().flatten() {
            let mut c = c.clone();
            c["index"] = serde_json::json!(choices.len());
            choices.push(c);
        }
        let count = |key: &str| {
            v.pointer(&format!("/usage/{key}"))
                .and_then(serde_json::Value::as_i64)
                .unwrap_or(0)
        };
        if sum_prompts || i == 0 {
            prompt_tokens += count("prompt_tokens");
        }
        completion_tokens += count("completion_tokens");
        merged.get_or_insert(v);
    }
    let mut out = merged.unwrap_or_else(|| serde_json::json!({}));
    out["choices"] = serde_json::Value::Array(choices);
    if !out["usage"].is_object() {
        out["usage"] = serde_json::json!({});
    }
    out["usage"]["prompt_tokens"] = serde_json::json!(prompt_tokens);
    out["usage"]["completion_tokens"] = serde_json::json!(completion_tokens);
    out["usage"]["total_tokens"] = serde_json::json!(prompt_tokens + completion_tokens);
    Ok(Json(out).into_response())
}

/// A completions `prompt`: the string, or the strings of an array joined by newlines;
/// `None` for any other shape.
fn completion_prompt_text(body: &serde_json::Value) -> Option<String> {
    match body.get("prompt")? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(arr) => Some(
            arr.iter()
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        _ => None,
    }
}

/// The prompt with a non-empty `suffix` rendered as a fill-in-the-middle request, with
/// the sentinels the coder models read, as `/api/generate` does.
fn fim_wrapped(body: &serde_json::Value, prompt: String) -> String {
    match body.get("suffix").and_then(serde_json::Value::as_str) {
        Some(suffix) if !suffix.is_empty() => {
            format!("<|fim_prefix|>{prompt}<|fim_suffix|>{suffix}<|fim_middle|>")
        }
        _ => prompt,
    }
}

/// What a chat-completion request says about its prompt: the tools in the model's own
/// syntax, the reasoning effort, and the grammar a structured output imposes.
fn chat_prompt_spec(request: &ChatCompletionRequest) -> super::chat_prompt::PromptSpec<'_> {
    super::chat_prompt::PromptSpec {
        messages: &request.messages,
        tools: request.tools.as_ref(),
        tool_choice: request.tool_choice.as_ref(),
        thinking: super::chat_prompt::thinking_from_effort(request.reasoning_effort.as_deref()),
        output_grammar: response_format_to_grammar(request.response_format.as_ref()),
    }
}

/// Chat completion (OpenAI-compatible format)
pub(crate) async fn chat_completion(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    OpenAIJson(request): OpenAIJson<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    validate_request(&request)?;

    // Cap aggregate prompt size up front (mirror /v1/completions guard).
    // 1024k chars ≈ 256k tokens, comfortably within the largest
    // practical context any perimeter model supports. Stops a pathological
    // multi-megabyte history from tying up the prefill path even when
    // the body fits inside the DefaultBodyLimit.
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
    // The 4096-message count cap now lives as #[validate(length(max=4096))]
    // on ChatCompletionRequest.messages (231b3a4); validate_request above
    // returns the same 400 envelope ("messages length must be in [1, 4096]")
    // before we get here. Imperative check removed.

    // The sampling knobs, `stop`, `top_logprobs` and the `user` session key, bounded
    // as on every other route.
    let sampling = super::sampling::SamplingRequest::from_openai(&request);
    sampling.validate()?;

    // OpenAI's allowed message roles. Anything else is a 400 - the
    // chat template would otherwise silently fold an unknown role
    // ("invalid: hi") into the prompt and the model would respond
    // to garbage. Also catches typos like "asistant".
    for (i, m) in request.messages.iter().enumerate() {
        match m.role.as_str() {
            "system" | "user" | "assistant" | "tool" | "function" | "developer" => {}
            other => {
                return Err(ApiError::Validation(format!(
                    "messages[{i}].role '{other}' not allowed; use system|user|assistant|tool|function|developer"
                )));
            }
        }
    }

    // Audio output is produced by /v1/audio/speech, not here.
    if let Some(m) = request.modalities.as_ref() {
        if m.iter().any(|x| x.eq_ignore_ascii_case("audio")) {
            return Err(ApiError::Validation(
                "`modalities` with audio is not produced here; use /v1/audio/speech".into(),
            ));
        }
    }
    // `n` > 1 is n completions of the same request, each with its own seed, gathered
    // into one response as on /v1/completions; a stream carries one completion.
    let n = super::sampling::completions_asked(request.n.map(u64::from))?;
    if n > 1 {
        if request.stream.unwrap_or(false) {
            return Err(ApiError::Validation(
                "`n` > 1 is not supported with `stream`".into(),
            ));
        }
        return merge_runs(n, false, |i| {
            let mut one = request.clone();
            one.n = Some(1);
            one.seed = one.seed.map(|s| s.wrapping_add(i as u64));
            Box::pin(chat_completion(
                State(state.clone()),
                headers.clone(),
                OpenAIJson(one),
            ))
        })
        .await;
    }
    // `tools` are handled below (injected into the prompt, then parsed
    // back out of the model output by crate::api::tool_calls). Nothing to
    // reject here anymore.
    // Validate service_tier against OpenAI's documented enum. The
    // field echoes back into the response (b0f9301), so a typo like
    // "premium" would surface in usage logs as if it were a real
    // tier instead of getting rejected up front.
    if let Some(tier) = request.service_tier.as_deref() {
        match tier {
            "auto" | "default" | "scale" => {}
            other => {
                return Err(ApiError::Validation(format!(
                    "`service_tier` '{other}' not supported; use 'auto', 'default', or 'scale'"
                )));
            }
        }
    }

    validate_model_id(&request.model)?;
    // Normalize model ID
    let model_name = normalize_model_id(&request.model);
    // The conversation goes where the model is. Peers are asked about the prompt this
    // node would prefill, which is the text their caches are keyed on.
    let clustered = state.cluster_handle().is_some();
    if let Some(body) = clustered
        .then(|| serde_json::to_value(&request).ok())
        .flatten()
    {
        let prompt_text = state
            .prepare_prompt(&model_name, None, &chat_prompt_spec(&request))
            .await
            .prompt;
        let max_tokens = body["max_completion_tokens"]
            .as_u64()
            .or_else(|| body["max_tokens"].as_u64())
            .map_or_else(
                || crate::distributed::rate_meter::expected_output_tokens(&model_name),
                |m| m as u32,
            );
        if let Some(relayed) = super::route_to_holder(
            &state,
            &headers,
            &model_name,
            &prompt_text,
            max_tokens,
            &super::OPENAI_CHAT,
            &body,
        )
        .await
        {
            return Ok(relayed);
        }
    }

    // Reject non-decoder pipeline components (CLIP, T5, whisper, parler)
    // before they reach the text engine, matching the /api/chat and
    // /api/generate guards. Same rationale as 857388d: routing CLIP to
    // the chat path panicked the tensor-op worker thread on its missing-
    // weights branch (blocking_lock in async).
    if let Some(hint) = non_chat_pipeline_component(&model_name) {
        return Err(ApiError::Validation(format!(
            "model '{model_name}' is not a chat model: {hint}",
        )));
    }

    // TTS via the OpenAI shape: the canonical OpenAI endpoint for text
    // -to-speech is /v1/audio/speech. /api/chat (the Ollama shape) has
    // its own handle_chat_tts that embeds audio in message.audios, but
    // chat-completion responses don't have a documented audios field  -
    // direct callers to the right endpoint instead of fabricating one
    // and risking silent SDK breakage.
    if crate::api::handlers::family::is_tts(&model_name) {
        return Err(ApiError::Validation(format!(
            "model '{model_name}' is a TTS pipeline. \
             POST text to /v1/audio/speech (OpenAI shape) for synthesis, \
             or use /api/chat (Ollama shape) for inline audio in the chat response."
        )));
    }

    // ASR via the OpenAI shape: same rationale as TTS - the OpenAI
    // ChatCompletionRequest has no input-bytes field, so we can't
    // accept audio here. Direct callers to /v1/audio/transcriptions
    // (the canonical OpenAI endpoint). /api/chat handles whisper
    // via its images[] field.
    if crate::api::handlers::family::is_asr(&model_name) {
        return Err(ApiError::Validation(format!(
            "model '{model_name}' is a speech-to-text pipeline. \
             POST audio bytes to /v1/audio/transcriptions (OpenAI shape), \
             or use /api/chat (Ollama shape) with an audio attachment."
        )));
    }

    let stream = request.stream.unwrap_or(false);
    // OpenAI's spec: `stream_options` is only valid when stream=true.
    // Reject mid-flight rather than silently dropping the bag - the
    // mistake usually means the caller forgot to flip stream and would
    // be confused by missing usage when they passed include_usage.
    if !stream && request.stream_options.is_some() {
        return Err(ApiError::Validation(
            "`stream_options` is only valid when `stream: true`".into(),
        ));
    }
    // Emit the trailing usage-only chunk iff stream_options.include_usage
    // = true. Default off - many older SDKs get tripped up by the
    // extra chunk.
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|o| o.include_usage)
        .unwrap_or(false);
    info!(
        "Chat completion request: model={model_name} stream={stream} n_messages={n_messages}{normalized}",
        n_messages = request.messages.len(),
        normalized = if request.model != model_name {
            format!(" (from {})", request.model)
        } else {
            String::new()
        },
    );

    // Auto-load the model on demand, matching /api/chat and the
    // Anthropic shim. OpenAI-compatible coding agents (Cline, Continue,
    // Aider) just name a model and expect it to be served - a 404 on the
    // first request breaks them. Falls through to the 404 below only if
    // the model genuinely can't be loaded (missing weights, bad arch). The window and
    // cache format asked in the headers are opened by the load itself.
    let engine = state
        .load_for_request(
            &model_name,
            super::admission::LoadRequest::from_headers(
                &headers,
                state.get_effective_keep_alive(None),
            ),
        )
        .await?;

    // What the model is actually served with, echoed back in the same header the request
    // may have asked in. A client that asks and is quietly clamped - by the checkpoint or
    // by what the cards hold - would otherwise size its prompts for a window it never got,
    // and the server would truncate them from the front without either side saying so.
    let served_window = engine.context_window().await;

    // The model's template with the tools in its own syntax, the caller's reasoning
    // effort, and the grammar a structured output or a required call imposes.
    let super::chat_prompt::PreparedPrompt {
        prompt,
        tools_active,
        tool_format,
        grammar,
        thinking_opened,
    } = state
        .prepare_prompt(&model_name, Some(&engine), &chat_prompt_spec(&request))
        .await;
    let single_tool_call = request.parallel_tool_calls == Some(false);

    let params = GenerationParams {
        grammar,
        ..sampling.into_params()
    };

    // The body has no field for this, so it is read from the headers: a client with
    // background work needs a way to say so on the route it speaks, and a client that
    // says nothing is interactive.
    let oai_priority = crate::api::gate::Priority::from_headers(&headers)
        .unwrap_or(crate::api::gate::Priority::Interactive);

    // The gate, the in-flight count and the last user turn's images.
    let admission = state
        .admit(
            engine.clone(),
            &model_name,
            oai_priority,
            "/v1/chat/completions",
            &super::chat_prompt::last_user_images(&request.messages),
        )
        .await?;

    if stream {
        let max_tokens_cap = request.completion_cap();
        // Estimate prompt token count once for the trailing usage chunk
        // (when stream_options.include_usage=true). Cheap heuristic  -
        // engine doesn't expose the real tokenizer count in this path.
        let prompt_tokens = estimate_token_count(&prompt) as i32;
        // Echo the client's `service_tier` request into each chunk
        // (mirrors the non-stream fix in b0f9301). Cloned once into
        // the closure so we don't re-borrow `request` per chunk.
        let service_tier = request
            .service_tier
            .clone()
            .unwrap_or_else(|| "default".to_string());
        // Streaming response using SSE (OpenAI-compatible format)
        let engine_for_stats = engine.clone();
        let wants_logprobs = request.logprobs == Some(true);
        match engine.generate_stream(&prompt, params.clone()).await {
            Ok(mut rx) => {
                let completion_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());
                let model_name_clone = model_name.clone();
                // Fixed at request time so every chunk in this stream
                // carries the same `created` field (OpenAI spec - SDKs
                // treat it as a request id; the previous behaviour of
                // re-sampling `now()` per chunk made it drift seconds
                // over a long generation).
                let created_at = chrono::Utc::now().timestamp();
                let stream_started = std::time::Instant::now();
                let stream = async_stream::stream! {
                    let _admitted = admission;
                    // First chunk: role indicator
                    let mut role_chunk = ChatCompletionChunk::new_delta_at(
                        &completion_id,
                        &model_name_clone,
                        created_at,
                        0,
                        ChunkDelta::role("assistant"),
                        None,
                    );
                    role_chunk.service_tier = service_tier.clone();
                    yield Ok::<_, axum::Error>(Event::default().data(to_json_string!(&role_chunk)));

                    let mut token_count: i32 = 0;
                    let mut first_token_at: Option<std::time::Instant> = None;
                    let mut errored = false;
                    // When tools are active, route content through a scanner
                    // that streams natural text until a tool-call marker,
                    // then buffers the tool region for end-of-stream parse.
                    let mut tool_scanner = if tools_active {
                        Some(crate::api::tool_calls::StreamToolScanner::new(tool_format))
                    } else {
                        None
                    };
                    let mut splitter = crate::api::thinking::ThinkSplit::opened(thinking_opened);
                    let mut reasoning_acc = String::new();

                    while let Some(result) = rx.recv().await {
                        match result {
                            Ok(chunk_text) => {
                                if first_token_at.is_none() {
                                    first_token_at = Some(std::time::Instant::now());
                                }
                                token_count += 1;
                                for seg in splitter.push(&chunk_text) {
                                    let delta = match seg {
                                        crate::api::thinking::Segment::Thinking(t) => {
                                            reasoning_acc.push_str(&t);
                                            ChunkDelta::reasoning(t)
                                        }
                                        crate::api::thinking::Segment::Content(c) => {
                                            let emit = match tool_scanner.as_mut() {
                                                Some(sc) => sc.push(&c),
                                                None => c,
                                            };
                                            if emit.is_empty() {
                                                continue;
                                            }
                                            ChunkDelta::content(emit)
                                        }
                                    };
                                    let mut chunk = ChatCompletionChunk::new_delta_at(
                                        &completion_id,
                                        &model_name_clone,
                                        created_at,
                                        0,
                                        delta,
                                        None,
                                    );
                                    chunk.service_tier = service_tier.clone();
                                    if wants_logprobs {
                                        let drawn = engine_for_stats.take_logprobs();
                                        if !drawn.is_empty() {
                                            if let Some(c) = chunk.choices.first_mut() {
                                                c.logprobs = Some(serde_json::json!({
                                                    "content": openai_logprobs_content(&drawn)
                                                }));
                                            }
                                        }
                                    }
                                    yield Ok(Event::default().data(to_json_string!(&chunk)));
                                }
                            }
                            Err(e) => {
                                // Surface the error as the terminal
                                // chunk: include the message in
                                // content + close out with the
                                // OpenAI-spec finish_reason for failed
                                // generation. Single terminal chunk,
                                // no duplicate stop after this.
                                let mut err_chunk = ChatCompletionChunk::new_delta_at(
                                    &completion_id,
                                    &model_name_clone,
                                    created_at,
                                    0,
                                    ChunkDelta::content(format!("[Error: {}]", e)),
                                    Some("stop".to_string()),
                                );
                                err_chunk.service_tier = service_tier.clone();
                                yield Ok(Event::default().data(to_json_string!(&err_chunk)));
                                errored = true;
                                break;
                            }
                        }
                    }

                    // Final chunk with finish_reason. "length" when we
                    // produced the requested max_tokens (model would
                    // have continued otherwise), "stop" when the
                    // engine emitted an EOS / stop sequence and closed
                    // the channel on its own. Skipped on the error
                    // path - the err_chunk already carried a
                    // finish_reason, so this would be a duplicate.
                    // Parse the buffered tool region (if any) into
                    // structured calls and emit them as a single delta
                    // before the terminal chunk. finish_reason flips to
                    // "tool_calls" when the model invoked functions.
                    for seg in splitter.finish() {
                        let delta = match seg {
                            crate::api::thinking::Segment::Thinking(t) => ChunkDelta::reasoning(t),
                            crate::api::thinking::Segment::Content(c) => {
                                let emit = match tool_scanner.as_mut() {
                                    Some(sc) => sc.push(&c),
                                    None => c,
                                };
                                if emit.is_empty() {
                                    continue;
                                }
                                ChunkDelta::content(emit)
                            }
                        };
                        let mut chunk = ChatCompletionChunk::new_delta_at(
                            &completion_id,
                            &model_name_clone,
                            created_at,
                            0,
                            delta,
                            None,
                        );
                        chunk.service_tier = service_tier.clone();
                        yield Ok(Event::default().data(to_json_string!(&chunk)));
                    }
                    let mut tool_calls = tool_scanner
                        .as_ref()
                        .map(|sc| sc.finalize())
                        .unwrap_or_default();
                    if single_tool_call {
                        tool_calls.truncate(1);
                    }
                    let used_tools = !tool_calls.is_empty();
                    if !errored {
                        if used_tools {
                            let deltas: Vec<_> = tool_calls
                                .iter()
                                .enumerate()
                                .map(|(i, c)| crate::api::types::ToolCallDelta::from_tool_call(i, c))
                                .collect();
                            let mut tc_chunk = ChatCompletionChunk::new_delta_at(
                                &completion_id,
                                &model_name_clone,
                                created_at,
                                0,
                                ChunkDelta::tool_calls(deltas),
                                None,
                            );
                            tc_chunk.service_tier = service_tier.clone();
                            yield Ok(Event::default().data(to_json_string!(&tc_chunk)));
                        } else if let Some(tail) =
                            tool_scanner.as_ref().map(|sc| sc.unstreamed()).filter(|t| !t.is_empty())
                        {
                            // Bare-JSON trigger fired on non-tool text:
                            // flush the withheld buffer as content so it
                            // isn't lost.
                            let mut chunk = ChatCompletionChunk::new_delta_at(
                                &completion_id,
                                &model_name_clone,
                                created_at,
                                0,
                                ChunkDelta::content(tail.to_string()),
                                None,
                            );
                            chunk.service_tier = service_tier.clone();
                            yield Ok(Event::default().data(to_json_string!(&chunk)));
                        }
                    }
                    let stats = engine_for_stats.take_last_stream_stats();
                    let finish_reason = if used_tools {
                        "tool_calls"
                    } else if let Some(s) = stats.as_ref() {
                        s.finish_reason.openai()
                    } else {
                        match max_tokens_cap {
                            Some(cap) if (token_count as usize) >= cap => "length",
                            _ => "stop",
                        }
                    };
                    if !errored {
                        let mut stop_chunk = ChatCompletionChunk::new_delta_at(
                            &completion_id,
                            &model_name_clone,
                            created_at,
                            0,
                            ChunkDelta::empty(),
                            Some(finish_reason.to_string()),
                        );
                        stop_chunk.service_tier = service_tier.clone();
                        yield Ok(Event::default().data(to_json_string!(&stop_chunk)));
                    }

                    // OpenAI spec: the trailing usage-only chunk
                    // (choices=[], usage populated) ships only when the
                    // client passes `stream_options.include_usage: true`.
                    // Older SDKs treat an unexpected chunk as a hard
                    // protocol error, so default-off is the safer
                    // behaviour.
                    // What the engine actually read, where it said; the word-count
                    // estimate only where it did not. Both the usage chunk and the log
                    // line below take this figure: reporting the estimate in the log made
                    // every streamed request look slower per token than it was, by
                    // whatever margin the estimate was wrong - on source code, nearly
                    // double.
                    // The cached share is reported as the whole answer reports it.
                    let mut usage = match stats.as_ref() {
                        Some(s) => Usage::from_counts(
                            s.prompt_eval_count,
                            s.cached_prompt_tokens,
                            s.eval_count,
                        ),
                        None => Usage::new(prompt_tokens, token_count),
                    };
                    let (in_tokens, out_tokens) = (usage.prompt_tokens, usage.completion_tokens);
                    if include_usage {
                        if !reasoning_acc.is_empty() {
                            if let (Some(n), Some(d)) = (
                                engine_for_stats.count_tokens(&reasoning_acc).await,
                                usage.completion_tokens_details.as_mut(),
                            ) {
                                d.reasoning_tokens = n as i32;
                            }
                        }
                        let mut usage_chunk = ChatCompletionChunk::new_final_at(
                            &completion_id,
                            &model_name_clone,
                            created_at,
                            usage,
                        );
                        usage_chunk.service_tier = service_tier.clone();
                        yield Ok(Event::default().data(to_json_string!(&usage_chunk)));
                    }

                    // Stream summary log - mirrors the non-stream
                    // "Chat completion:" line so a single grep covers
                    // both paths. `total_ms` is wall time from stream
                    // start to terminal marker, including network
                    // backpressure on slow clients.
                    let total_ms = stream_started.elapsed().as_millis();
                    let ttft_ms = time_to_first_token_ms(stream_started, first_token_at);
                    let rate = decode_rate_suffix(first_token_at, token_count as u64);
                    tracing::info!(
                        "Chat completion stream: prompt_tokens={in_tokens} completion_tokens={out_tokens} finish={finish_reason} ttft={ttft_ms}ms total={total_ms}ms{rate}"
                    );
                    // Terminal marker
                    yield Ok(Event::default().data("[DONE]"));
                };

                let mut response = Sse::new(stream)
                    .keep_alive(KeepAlive::default())
                    .into_response();
                if let Some(n) = served_window {
                    if let Ok(hv) = axum::http::HeaderValue::from_str(&n.to_string()) {
                        response
                            .headers_mut()
                            .insert(crate::api::gate::Window::HEADER, hv);
                    }
                }
                Ok(response)
            }
            Err(e) => {
                error!(
                    "Generation error (OpenAI chat completion stream-init): {}",
                    e
                );
                Err(ApiError::Internal(format!("generate_stream: {e}")))
            }
        }
    } else {
        let _admitted = admission;
        // `mut` bindings with `None` defaults trigger an
        // "assigned-never-read" warning because the only path that
        // reads timing is the Ok arm (which always overwrites).
        // Declared without initial value here - the Err arm returns
        // before the read sites, so the compiler's still happy.
        let timing_prefill_ms: Option<f64>;
        let timing_decode_ms: Option<f64>;
        let _cancel = engine.cancel_guard();
        let result_logprobs: Vec<crate::inference::engine::llm_engine::TokenLogprob>;
        let (content, finish_reason, completion_tokens, prompt_tokens, cached_tokens) =
            match engine.generate(&prompt, params).await {
                Ok(result) => {
                    result_logprobs = result.logprobs.clone();
                    // OpenAI's `finish_reason`: `length` when we hit
                    // max_tokens, `stop` when the model emitted an EOS
                    // token / stop sequence. Inferred from eval_count vs
                    // the request cap.
                    let reason = result.finish_reason.openai();
                    // Convert nanos -> ms for Server-Timing header below.
                    timing_prefill_ms = Some(result.prompt_eval_duration as f64 / 1_000_000.0);
                    timing_decode_ms = Some(result.eval_duration as f64 / 1_000_000.0);
                    (
                        result.text,
                        reason.to_string(),
                        result.eval_count as i32,
                        result.prompt_eval_count,
                        result.cached_prompt_tokens,
                    )
                }
                Err(e) => {
                    // Engine error during generation: return 500 with
                    // the OpenAI error envelope rather than a 200 OK
                    // whose assistant content is the error string  -
                    // SDK clients otherwise treat it as model output.
                    error!("Generation error (OpenAI chat completion): {}", e);
                    return Err(ApiError::Internal(format!("generate: {e}")));
                }
            };

        // When tools were injected, lift any tool-call markers out of the
        // model output into structured tool_calls and switch finish_reason
        // to "tool_calls" (OpenAI's contract). Plain answers (no markers)
        // pass through unchanged.
        let (reasoning, content) =
            crate::api::thinking::split_thinking_opened(thinking_opened, &content);
        let (message, finish_reason) = if tools_active {
            let mut parsed = crate::api::tool_calls::parse_tool_calls(tool_format, &content);
            if single_tool_call {
                parsed.calls.truncate(1);
            }
            if parsed.calls.is_empty() {
                (
                    Message::new("assistant".to_string(), content),
                    finish_reason,
                )
            } else {
                (
                    Message::with_tool_calls(parsed.content, parsed.calls),
                    "tool_calls".to_string(),
                )
            }
        } else {
            (
                Message::new("assistant".to_string(), content),
                finish_reason,
            )
        };
        let mut message = message;
        message.reasoning_content = reasoning.clone();
        let finish_for_log = finish_reason.clone();
        let mut choice = Choice::new(0, message, finish_reason);
        if !result_logprobs.is_empty() {
            choice.logprobs = Some(serde_json::json!({
                "content": openai_logprobs_content(&result_logprobs)
            }));
        }
        let mut response = ChatCompletionResponse::new(
            format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            model_name,
            vec![choice],
            Usage::from_counts(prompt_tokens, cached_tokens, completion_tokens as u64),
        );
        if let Some(r) = reasoning.as_deref() {
            if let (Some(n), Some(d)) = (
                engine.count_tokens(r).await,
                response.usage.completion_tokens_details.as_mut(),
            ) {
                d.reasoning_tokens = n as i32;
            }
        }
        // Echo the client's `service_tier` request back so SDKs that
        // round-trip the value see what they asked for. Falls back to
        // "default" when unset, matching OpenAI's behaviour for free-
        // tier accounts.
        if let Some(tier) = request.service_tier.as_deref() {
            response.service_tier = tier.to_string();
        }

        info!(
            "Chat completion: prompt_tokens={} cached={cached_tokens} completion_tokens={completion_tokens} finish={finish_for_log}{timing}",
            response.usage.prompt_tokens,
            timing =match (timing_prefill_ms, timing_decode_ms) {
                (Some(p), Some(d)) => {
                    // Sub-millisecond decode is almost always
                    // `max_tokens=1` or an early EOS - reporting a
                    // tok/s computed from <0.5ms produces ridiculous
                    // numbers (1.6M tok/s on a 0ms-rounded decode).
                    // Skip the rate when the floor isn't usable.
                    let rate = if d >= 1.0 {
                        format!(" ({:.1} tok/s)", (completion_tokens as f64) * 1000.0 / d)
                    } else {
                        String::new()
                    };
                    format!(" prefill={p:.0}ms decode={d:.0}ms{rate}")
                }
                _ => String::new(),
            },
        );
        // Server-Timing header (RFC 8673-style) lets clients see prefill
        // + decode latencies without server-side log access. Standard
        // dev-tools surface this in the Network panel. Skipped on the
        // error path (no real timings to report).
        let mut headers = axum::http::HeaderMap::new();
        if let (Some(pf), Some(dc)) = (timing_prefill_ms, timing_decode_ms) {
            let v = format!(
                "prefill;dur={pf:.1}, decode;dur={dc:.1}, total;dur={:.1}",
                pf + dc
            );
            if let Ok(hv) = axum::http::HeaderValue::from_str(&v) {
                headers.insert("server-timing", hv);
            }
        }
        if let Some(n) = served_window {
            if let Ok(hv) = axum::http::HeaderValue::from_str(&n.to_string()) {
                headers.insert(crate::api::gate::Window::HEADER, hv);
            }
        }
        Ok((headers, Json(response)).into_response())
    }
}

/// Cap on the serialized JSON Schema attached to `response_format` /
/// `format`. Real-world schemas top out around ~50 KB even for complex
/// OpenAPI-derived shapes; 64 KiB matches the modelfile / parameters
/// caps from /api/create. Bigger inputs would just hand a giant string
/// to the grammar engine to re-parse for nothing useful.
const JSON_SCHEMA_MAX_BYTES: usize = 64 * 1024;

/// Format a JSON Schema as `json_schema:<schema>` for the grammar
/// engine, returning None when the serialized form exceeds the cap.
/// Centralised so both response_format and Ollama format paths apply
/// the same bound.
pub(crate) fn json_schema_grammar_str(schema: &serde_json::Value) -> Option<String> {
    let serialized = schema.to_string();
    if serialized.len() > JSON_SCHEMA_MAX_BYTES {
        // Falling back to None matches the "unknown shape ⇒ None" policy
        // both helpers already follow - generator runs free, no surprise
        // 500 from the grammar engine choking on a megabyte input.
        return None;
    }
    Some(format!("json_schema:{serialized}"))
}

/// Map OpenAI's `response_format` object onto the grammar-spec string
/// that `parse_grammar_spec` knows how to consume.
///
///   `{"type":"text"}`                                     -> None
///   `{"type":"json_object"}`                              -> "json_object"
///   `{"type":"json_schema","json_schema":{"schema":...}}` -> `"json_schema:<schema>"`
///
/// Unknown shapes return None - the generator runs free, matching
/// OpenAI's lenient handling of unrecognized response_format types.
fn response_format_to_grammar(rf: Option<&serde_json::Value>) -> Option<String> {
    let rf = rf?;
    let kind = rf.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match kind {
        "json_object" => Some("json_object".to_string()),
        "json_schema" => {
            let schema = rf
                .get("json_schema")
                .and_then(|s| s.get("schema"))
                .or_else(|| rf.get("schema"))?;
            json_schema_grammar_str(schema)
        }
        _ => None,
    }
}

/// Map Ollama's `format` field onto the same grammar-spec string the
/// OpenAI `response_format` route uses. Two shapes accepted:
///
///   "json"              -> "json_object" (bare JSON object)
///   `<JSON Schema obj>` -> `"json_schema:<schema>"` (strict schema)
///
/// Returns `None` for missing / unrecognized values - generator runs
/// free, matching Ollama's lenient handling.
pub(crate) fn ollama_format_to_grammar(format: Option<&serde_json::Value>) -> Option<String> {
    let f = format?;
    if let Some(s) = f.as_str() {
        return match s {
            "json" => Some("json_object".to_string()),
            _ => None,
        };
    }
    // JSON Schema object - same 64 KiB cap as the OpenAI path.
    if f.is_object() {
        return json_schema_grammar_str(f);
    }
    None
}

/// Legacy `POST /v1/completions` - OpenAI's pre-chat text-completion
/// endpoint. Older SDKs (LangChain's default, llamaindex, raw-prompt
/// scripts) still call it. Body shape:
///   { "model", "prompt": String|[String], "max_tokens", "temperature",
///     "top_p", "stream", "user", "seed", "stream_options" }
/// Response is the `text_completion` object form (`choices[*].text`),
/// not the chat object form (`choices[*].message.content`).
pub(crate) async fn text_completions(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let model = text_field(&body, "model");
    // The completion goes where the model is.
    {
        let model_name = crate::api::handlers::normalize_model_id(&model);
        let prompt_text = fim_wrapped(&body, completion_prompt_text(&body).unwrap_or_default());
        let max_tokens = body["max_tokens"].as_u64().map_or_else(
            || crate::distributed::rate_meter::expected_output_tokens(&model_name),
            |m| m as u32,
        );
        if let Some(relayed) = super::route_to_holder(
            &state,
            &headers,
            &model_name,
            &prompt_text,
            max_tokens,
            &super::OPENAI_COMPLETIONS,
            &body,
        )
        .await
        {
            return Ok(relayed);
        }
    }
    // Several prompts are several completions, one per prompt, gathered into one
    // response with their indexes; a stream carries one prompt.
    if let Some(arr) = body.get("prompt").and_then(serde_json::Value::as_array) {
        if arr.len() > 1 {
            if body
                .get("stream")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
            {
                return Err(ApiError::Validation(
                    "an array `prompt` is not supported with `stream`; send one prompt per stream"
                        .into(),
                ));
            }
            return merge_runs(arr.len(), true, |i| {
                let mut single = body.clone();
                single["prompt"] = arr[i].clone();
                Box::pin(text_completions(
                    State(state.clone()),
                    headers.clone(),
                    Json(single),
                ))
            })
            .await;
        }
    }
    // Empty / whitespace-only / oversized / path-traversal model IDs are
    // all rejected by `validate_model_id` below with the same 400 envelope  -
    // no separate early check needed.
    // OpenAI accepts a string or an array of strings for `prompt`; an array of several
    // was fanned out above.
    let Some(prompt) = completion_prompt_text(&body) else {
        return Err(ApiError::Validation(
            "`prompt` must be a string or array of strings".to_string(),
        ));
    };
    if prompt.trim().is_empty() {
        return Err(ApiError::Validation(
            "`prompt` must not be empty".to_string(),
        ));
    }
    // Cap prompt length up front so a pathological multi-megabyte
    // prompt doesn't tie up the prefill path (the body limit sizes
    // media uploads and would otherwise let one through). 256k chars ~ 64k tokens at
    // 4 chars/token, fits within the largest practical context the
    // perimeter models support.
    const PROMPT_MAX_CHARS: usize = 256 * 1024;
    if prompt.chars().count() > PROMPT_MAX_CHARS {
        return Err(ApiError::Validation(format!(
            "`prompt` is {} chars; cap at {PROMPT_MAX_CHARS} (~64k tokens). Split or summarize client-side.",
            prompt.chars().count()
        )));
    }

    let stream = body
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    // Same `stream_options`-without-`stream` guard as /v1/chat/completions
    // (a453bdb). Reject early so a caller who forgot to flip `stream`
    // doesn't silently lose the trailing usage chunk.
    if !stream && body.get("stream_options").is_some() {
        return Err(ApiError::Validation(
            "`stream_options` is only valid when `stream: true`".into(),
        ));
    }
    let include_usage = body
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    // The sampling knobs, `stop`, `logprobs` and the `user` session key, bounded as on
    // every other route. `logprobs: n` asks for each token's log-probability with `n`
    // alternatives.
    let sampling = super::sampling::SamplingRequest::from_completions(&body);
    sampling.validate()?;
    let completion_logprobs = sampling.top_logprobs.map(|v| v as usize);

    // Fail-fast for `best_of` (sample N, return top 1) - would
    // require Nx decode + ranking. Default 1 / null is fine; > 1 is
    // rejected so clients don't think they're getting best-of-N output
    // when they're really getting a single sample.
    if let Some(b) = body.get("best_of").and_then(serde_json::Value::as_u64) {
        if b > 1 {
            return Err(ApiError::Validation(format!(
                "`best_of` = {b} not supported; only the single-sample path is implemented"
            )));
        }
    }
    // `echo: true` would prepend the prompt to the response. Reject
    // explicitly rather than silently dropping the prompt-prefix
    // (clients consuming `choices[0].text` would then see the model
    // continuation without the prompt and might mis-parse it).
    // `echo` puts the prompt back in front of the completion, in the first chunk of a
    // stream; `suffix` makes a fill-in-the-middle request, rendered with the sentinels
    // the coder models read, as `/api/generate` does.
    let echo = body
        .get("echo")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let prompt = fim_wrapped(&body, prompt);
    let echo_text = if echo {
        body.get("prompt")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("")
            .to_string()
    } else {
        String::new()
    };
    // `n` > 1 is n completions of the same prompt, each with its own seed, gathered into
    // one response as on /v1/chat/completions; a stream carries one completion.
    let n = super::sampling::completions_asked(body.get("n").and_then(serde_json::Value::as_u64))?;
    if n > 1 {
        if stream {
            return Err(ApiError::Validation(
                "`n` > 1 is not supported with `stream`".into(),
            ));
        }
        return merge_runs(n, false, |i| {
            let mut one = body.clone();
            one["n"] = serde_json::json!(1);
            if let Some(seed) = body.get("seed").and_then(serde_json::Value::as_u64) {
                one["seed"] = serde_json::json!(seed.wrapping_add(i as u64));
            }
            Box::pin(text_completions(
                State(state.clone()),
                headers.clone(),
                Json(one),
            ))
        })
        .await;
    }

    validate_model_id(&model)?;
    let model_name = normalize_model_id(&model);

    // Reject non-decoder pipeline components (parity with the chat path).
    if let Some(hint) = non_chat_pipeline_component(&model_name) {
        return Err(ApiError::Validation(format!(
            "model '{model_name}' is not a text-completion model: {hint}",
        )));
    }

    // TTS: same routing rationale as chat_completion - /v1/completions
    // doesn't carry an audio payload field, so direct callers to the
    // canonical /v1/audio/speech endpoint instead.
    if crate::api::handlers::family::is_tts(&model_name) {
        return Err(ApiError::Validation(format!(
            "model '{model_name}' is a TTS pipeline. \
             POST text to /v1/audio/speech for synthesis."
        )));
    }
    // ASR: /v1/completions has no audio-input field either.
    if crate::api::handlers::family::is_asr(&model_name) {
        return Err(ApiError::Validation(format!(
            "model '{model_name}' is a speech-to-text pipeline. \
             POST audio bytes to /v1/audio/transcriptions instead."
        )));
    }

    let prompt_chars = prompt.chars().count();
    info!(
        "Text completion request: model={model_name} stream={stream} prompt_chars={prompt_chars}{normalized}",
        normalized = if model != model_name {
            format!(" (from {model})")
        } else {
            String::new()
        },
    );

    // Loaded on first use, as on the chat route, in the window and cache format the
    // headers ask for: an editor sending a long prefix needs the same say.
    let engine = state
        .load_for_request(
            &model_name,
            super::admission::LoadRequest::from_headers(
                &headers,
                state.get_effective_keep_alive(None),
            ),
        )
        .await?;

    // `response_format` is documented on /v1/chat/completions and several SDKs send it
    // here too: the same grammar engine serves both.
    let params = GenerationParams {
        grammar: response_format_to_grammar(body.get("response_format")),
        ..sampling.into_params()
    };

    let oai_priority = crate::api::gate::Priority::from_headers(&headers)
        .unwrap_or(crate::api::gate::Priority::Interactive);
    let completion_id = format!("cmpl-{}", uuid::Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    // The prompt count a stream reports when the engine publishes no statistics.
    let prompt_tokens_est = estimate_token_count(&prompt) as u32;
    // The gate (a continuously batching model takes concurrent requests in its own
    // worker), the in-flight count, and no images: a completion carries none, and what an
    // earlier request encoded must not answer this one.
    let admission = state
        .admit(
            engine.clone(),
            &model_name,
            oai_priority,
            "/v1/completions",
            &[],
        )
        .await?;

    if stream {
        // Capture max_tokens before moving params so the terminal
        // chunk can report `finish_reason: "length"` when we hit the
        // cap (was hardcoded "stop" regardless).
        let max_tokens_cap = params.max_tokens;
        let engine_for_stats = engine.clone();
        let rx = engine
            .generate_stream(&prompt, params.clone())
            .await
            .map_err(|e| ApiError::Internal(format!("generate_stream: {e}")))?;
        let cid = completion_id.clone();
        let mn = model_name.clone();
        let stream_started = std::time::Instant::now();
        let mut echo_pending = echo_text.clone();
        let mut text_offset = echo_text.len();
        let stream = async_stream::stream! {
            let _admitted = admission;
            let mut rx = rx;
            let mut tokens: u32 = 0;
            let mut first_token_at: Option<std::time::Instant> = None;
            let mut errored = false;
            while let Some(result) = rx.recv().await {
                match result {
                    Ok(text) => {
                        if first_token_at.is_none() {
                            first_token_at = Some(std::time::Instant::now());
                        }
                        tokens += 1;
                        let piece = format!("{echo_pending}{text}");
                        echo_pending.clear();
                        let drawn = if completion_logprobs.is_some() {
                            engine_for_stats.take_logprobs()
                        } else {
                            Vec::new()
                        };
                        let logprobs_value = if drawn.is_empty() {
                            serde_json::Value::Null
                        } else {
                            completions_logprobs(&drawn, text_offset)
                        };
                        text_offset += text.len();
                        let chunk = serde_json::json!({
                            "id": cid,
                            "object": "text_completion",
                            "created": created,
                            "model": mn,
                            "choices": [{
                                "text": piece,
                                "index": 0,
                                "logprobs": logprobs_value,
                                "finish_reason": serde_json::Value::Null,
                            }],
                        });
                        yield Ok::<_, axum::Error>(Event::default().data(chunk.to_string()));
                    }
                    Err(e) => {
                        let err_chunk = serde_json::json!({
                            "id": cid,
                            "object": "text_completion",
                            "created": created,
                            "model": mn,
                            "choices": [{
                                "text": format!("[Error: {e}]"),
                                "index": 0,
                                "logprobs": serde_json::Value::Null,
                                "finish_reason": "stop",
                            }],
                        });
                        yield Ok(Event::default().data(err_chunk.to_string()));
                        errored = true;
                        break;
                    }
                }
            }
            // Final stop chunk with empty text + finish_reason=stop.
            // Skipped on error - the err_chunk already carried
            // finish_reason="stop", so otherwise the stream ends with
            // two terminal markers (same fix as the chat-completion +
            // /api/chat streams).
            let stats = engine_for_stats.take_last_stream_stats();
            if !errored {
                // Match the non-stream path's `finish_reason` logic:
                // "length" when we produced exactly `max_tokens`,
                // "stop" otherwise (EOS / stop sequence / natural end).
                let finish_reason = match stats.as_ref() {
                    Some(st) => st.finish_reason.openai(),
                    None => match max_tokens_cap {
                        Some(cap) if (tokens as usize) >= cap => "length",
                        _ => "stop",
                    },
                };
                let stop = serde_json::json!({
                    "id": cid,
                    "object": "text_completion",
                    "created": created,
                    "model": mn,
                    "choices": [{
                        "text": "",
                        "index": 0,
                        "logprobs": serde_json::Value::Null,
                        "finish_reason": finish_reason,
                    }],
                });
                yield Ok(Event::default().data(stop.to_string()));
            }
            // What the engine counted, the estimate only where it said nothing; the
            // cached share as the whole answer reports it.
            let usage = match stats.as_ref() {
                Some(s) => Usage::from_counts(s.prompt_eval_count, s.cached_prompt_tokens, s.eval_count),
                None => Usage::new(prompt_tokens_est as i32, tokens as i32),
            };
            if include_usage {
                let usage = serde_json::json!({
                    "id": cid,
                    "object": "text_completion",
                    "created": created,
                    "model": mn,
                    "choices": [],
                    "usage": usage,
                });
                yield Ok(Event::default().data(usage.to_string()));
            }
            let total_ms = stream_started.elapsed().as_millis();
            let ttft_ms = time_to_first_token_ms(stream_started, first_token_at);
            let rate = decode_rate_suffix(first_token_at, tokens as u64);
            tracing::info!(
                "Text completion stream: completion_tokens={tokens} ttft={ttft_ms}ms total={total_ms}ms{rate}"
            );
            yield Ok(Event::default().data("[DONE]"));
        };
        return Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let _admitted = admission;
    let _cancel = engine.cancel_guard();
    let result = engine
        .generate(&prompt, params)
        .await
        .map_err(|e| ApiError::Internal(format!("generate: {e}")))?;
    let finish = result.finish_reason.openai();
    let prefill_ms = result.prompt_eval_duration as f64 / 1_000_000.0;
    let decode_ms = result.eval_duration as f64 / 1_000_000.0;
    let response = serde_json::json!({
        "id": completion_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "text": format!("{echo_text}{}", result.text),
            "index": 0,
            "logprobs": if result.logprobs.is_empty() {
                serde_json::Value::Null
            } else {
                completions_logprobs(&result.logprobs, echo_text.len())
            },
            "finish_reason": finish,
        }],
        "usage": Usage::from_counts(
            result.prompt_eval_count,
            result.cached_prompt_tokens,
            result.eval_count,
        ),
    });
    let rate = if decode_ms >= 1.0 {
        format!(
            " ({:.1} tok/s)",
            (result.eval_count as f64) * 1000.0 / decode_ms
        )
    } else {
        String::new()
    };
    info!(
        "Text completion: prompt_tokens={} completion_tokens={} finish={finish} prefill={prefill_ms:.0}ms decode={decode_ms:.0}ms{rate}",
        result.prompt_eval_count, result.eval_count,
    );
    let mut headers = axum::http::HeaderMap::new();
    let st = format!(
        "prefill;dur={prefill_ms:.1}, decode;dur={decode_ms:.1}, total;dur={:.1}",
        prefill_ms + decode_ms
    );
    if let Ok(hv) = axum::http::HeaderValue::from_str(&st) {
        headers.insert("server-timing", hv);
    }
    Ok((headers, Json(response)).into_response())
}

/// OpenAI-shaped error envelope:
///   { "error": { "message", "type", "param", "code" } }
/// `type` is derived from the HTTP status: 4xx -> invalid_request_error,
/// 5xx -> server_error. `param`/`code` stay null until callers carry them.
///
/// This is the canonical error body for every dialect EXCEPT Anthropic:
/// the /v1/* OpenAI routes build it via their local `err_resp` closures /
/// `openai_error_response`, and the JSON media endpoints reach it through
/// `conv_err`. Anthropic-protocol clients need `anthropic_error_response`.
pub(crate) fn openai_error_body(
    status: axum::http::StatusCode,
    message: impl Into<String>,
) -> serde_json::Value {
    let kind = if status.is_client_error() {
        "invalid_request_error"
    } else if status.is_server_error() {
        "server_error"
    } else {
        "api_error"
    };
    serde_json::json!({
        "error": {
            "message": message.into(),
            "type": kind,
            "param": serde_json::Value::Null,
            "code": serde_json::Value::Null,
        }
    })
}

/// Returns true when an HF repo's snapshot directory exists locally
/// AND contains at least one non-empty file - a cheap proxy for "won't
/// trigger a multi-GB download on first call". Doesn't validate file
/// integrity; that's the loader's job.
fn hf_cache_has_model(repo: &str) -> bool {
    let dir_name = format!("models--{}", repo.replace('/', "--"));
    // Probe every plausible cache root in priority order, matching the
    // resolution the hf-hub crate itself does. Stop on the first hit
    // that passes the non-empty / >1 MB sanity check below.
    //   1. HF_HUB_CACHE - direct override of the hub cache subdir.
    //   2. HF_HOME/hub - community-standard root (set when the user
    //      relocates the cache to e.g. a larger disk).
    //   3. ~/.cache/huggingface/hub - the platform default.
    let mut roots: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(v) = std::env::var("HF_HUB_CACHE") {
        roots.push(std::path::PathBuf::from(v));
    }
    if let Ok(v) = std::env::var("HF_HOME") {
        roots.push(std::path::PathBuf::from(v).join("hub"));
    }
    if let Some(h) = dirs::home_dir() {
        roots.push(h.join(".cache/huggingface/hub"));
    }
    for root in roots {
        let dir = root.join(&dir_name).join("snapshots");
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        // Require at least 2 non-empty files in any snapshot dir +
        // total size > 1 MB. Avoids the false-positive where a partial
        // download left only config.json (small) but not the weights.
        //
        // HF snapshots/<sha>/* are symlinks into ../blobs; use
        // `fs::metadata(path)` (follows symlinks) instead of
        // `DirEntry::metadata()` which on Unix returns the symlink's
        // own metadata and would report tiny sizes for everything.
        for snap in rd.flatten() {
            if let Ok(inner) = std::fs::read_dir(snap.path()) {
                let mut nonempty = 0u32;
                let mut total: u64 = 0;
                for f in inner.flatten() {
                    if let Ok(m) = std::fs::metadata(f.path()) {
                        let len = m.len();
                        if len > 0 {
                            nonempty += 1;
                            total = total.saturating_add(len);
                        }
                    }
                }
                if nonempty >= 2 && total > 1_000_000 {
                    return true;
                }
            }
        }
    }
    false
}

/// OpenAI-style /v1/models - lists local LLM checkpoints plus the
/// multimodal models this server can spin up on demand. Each entry
/// follows OpenAI's `Model` shape (`id`, `object`, `owned_by`,
/// `created`) plus a `kind` field flagging the modality so clients
/// can filter (chat / asr / tts / image), plus a `cached` boolean for
/// the multimodal entries.
pub(crate) async fn openai_list_models(
    state: axum::extract::State<APIServer>,
    headers: axum::http::HeaderMap,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let now = chrono::Utc::now().timestamp();

    let mut data: Vec<serde_json::Value> = Vec::new();

    // Snapshot the set of warm engines once so the response doesn't
    // race the engine map per-row. SDK clients can use `is_loaded`
    // to skip the cold-start latency for un-warmed models.
    let loaded_ids: std::collections::HashSet<String> = state
        .engines
        .read()
        .await
        .iter()
        .map(|e| e.model_id.clone())
        .collect();

    // LLM models pulled from the local model manager. Parse the
    // RFC3339 download timestamp into a unix `created` so the field
    // matches what most SDKs expect (file age, not request time).
    if let Ok(mut models) = state.model_manager.list_models().await {
        // Only models this node can actually load: present weights, and a format the text
        // loader reads. A manifest that names a deleted blob is not one a client can ask for.
        models.retain(|m| state.serves_weights(m));
        // Stable alphabetical order so SDK UIs render the catalog
        // consistently across calls (filesystem walk order is
        // platform-dependent).
        models.sort_by(|a, b| a.id.cmp(&b.id));
        for m in models {
            let created = chrono::DateTime::parse_from_rfc3339(&m.downloaded_at)
                .map(|t| t.timestamp())
                .unwrap_or(now);
            let is_loaded = loaded_ids.contains(&m.id);
            data.push(serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": created,
                "owned_by": "local",
                "kind": "chat",
                "size_bytes": m.size,
                "is_loaded": is_loaded,
            }));
        }
    }

    // What the peers hold and this node does not, owned by the node that holds it: a
    // request naming one is forwarded there.
    let local: std::collections::HashSet<String> = data
        .iter()
        .filter_map(|m| m["id"].as_str().map(str::to_string))
        .collect();
    let peers = super::cluster_catalogue::peer_models(&state, &headers).await;
    for m in super::cluster_catalogue::merge(Vec::new(), peers) {
        if local.contains(&m.name) {
            continue;
        }
        let created = chrono::DateTime::parse_from_rfc3339(&m.modified_at)
            .map(|t| t.timestamp())
            .unwrap_or(now);
        data.push(serde_json::json!({
            "id": m.name,
            "object": "model",
            "created": created,
            "owned_by": m.node.unwrap_or_default(),
            "kind": "chat",
            "size_bytes": m.size,
            "is_loaded": false,
        }));
    }

    // Multimodal entries the OpenAI-shaped endpoints can dispatch to.
    // Mark `cached` so SDK consumers know which ones won't trigger a
    // multi-GB HF download on first call.
    for (id, kind, owner, endpoint, hf_repo) in MULTIMODAL_EXTRAS {
        let cached = hf_cache_has_model(hf_repo);
        let is_loaded = multimodal_entry_is_loaded(&state, kind, hf_repo, id).await;
        data.push(serde_json::json!({
            "id": id,
            "object": "model",
            "created": now,
            "owned_by": owner,
            "kind": kind,
            "endpoint": endpoint,
            "hf_repo": hf_repo,
            "cached": cached,
            "is_loaded": is_loaded,
        }));
    }

    Json(serde_json::json!({ "object": "list", "data": data })).into_response()
}

/// Pure family->catalog-id matcher extracted from
/// `multimodal_entry_is_loaded` so the (family, id) -> bool table
/// can be unit-tested without standing up an engine state.
///
/// image_engine.loaded_family() returns "flux" or "zimage"  -
/// coarser than the per-id catalog (flux-schnell, z-image-turbo)
/// since the engine only tracks family-level state. We currently
/// only ship one checkpoint per family server-side, so the
/// family->id mapping is 1:1; this matcher pins that contract.
pub(crate) fn image_family_matches_catalog_id(family: Option<&str>, catalog_id: &str) -> bool {
    match family {
        Some("flux") => catalog_id == "flux-schnell",
        Some("zimage") => catalog_id == "z-image-turbo",
        _ => false,
    }
}

/// Is a specific MULTIMODAL_EXTRAS catalog entry currently warm?
/// Shared between /v1/models (the listing) and /v1/models/{id}
/// (the retrieve) so the two surfaces can't drift on which family
/// names count as loaded.
async fn multimodal_entry_is_loaded(
    state: &APIServer,
    kind: &str,
    hf_repo: &str,
    id: &str,
) -> bool {
    match kind {
        #[cfg(feature = "audio")]
        "asr" => state.audio_engine.loaded_name().await.as_deref() == Some(hf_repo),
        #[cfg(feature = "audio")]
        "tts" => state.tts_engine.loaded_name().await.as_deref() == Some(hf_repo),
        #[cfg(feature = "image")]
        "image" => {
            let fam = state.image_engine.loaded_family().await;
            image_family_matches_catalog_id(fam, id)
        }
        _ => false,
    }
}

/// Multimodal-model catalog surfaced by /v1/models + the retrieve
/// endpoint. Tuple shape: (id, kind, owner, endpoint, hf_repo).
const MULTIMODAL_EXTRAS: &[(&str, &str, &str, &str, &str)] = &[
    (
        "whisper-small",
        "asr",
        "openai",
        "/v1/audio/transcriptions",
        "openai/whisper-small",
    ),
    (
        "whisper-medium",
        "asr",
        "openai",
        "/v1/audio/transcriptions",
        "openai/whisper-medium",
    ),
    (
        "whisper-large-v3",
        "asr",
        "openai",
        "/v1/audio/transcriptions",
        "openai/whisper-large-v3",
    ),
    (
        "parler-tts-mini-v1",
        "tts",
        "parler-tts",
        "/v1/audio/speech",
        "parler-tts/parler-tts-mini-v1",
    ),
    (
        "parler-tts-large-v1",
        "tts",
        "parler-tts",
        "/v1/audio/speech",
        "parler-tts/parler-tts-large-v1",
    ),
    (
        "flux-schnell",
        "image",
        "black-forest-labs",
        "/v1/images/generations",
        "black-forest-labs/FLUX.1-schnell",
    ),
    // Z-Image: the loader in `inference/engine/image_engine/` actually downloads
    // \`Tongyi-MAI/Z-Image-Turbo\` from HF, not the placeholder
    // \`Tencent/Z-Image\` this entry used to list. With the wrong repo
    // here, \`cached\` lookups against the HF cache always returned
    // false even when the right snapshot was already on disk, and
    // SDK clients writing the repo back into a download URL would
    // 404. Owner / repo names now match the actual upstream model.
    (
        "z-image-turbo",
        "image",
        "tongyi-mai",
        "/v1/images/generations",
        "Tongyi-MAI/Z-Image-Turbo",
    ),
    // Generative media (text->sound/music/MIDI/video), all reachable via the API.
    (
        "ezaudio",
        "audio",
        "opensound",
        "/v1/audio/generations",
        "OpenSound/EzAudio",
    ),
    (
        "ace-step",
        "music",
        "ace-step",
        "/v1/audio/generations",
        "ACE-Step/ACE-Step-v1-3.5B",
    ),
    (
        "midi",
        "midi",
        "slseanwu",
        "/v1/audio/generations",
        "slseanwu/MIDI-LLM_Llama-3.2-1B",
    ),
    (
        "wan",
        "video",
        "wan-ai",
        "/v1/video/generations",
        "Wan-AI/Wan2.1-T2V-1.3B",
    ),
];

/// OpenAI's "retrieve model" - `GET /v1/models/{id}`. Returns the
/// single-entry shape (same fields as one element of /v1/models'
/// `data` array). 404 + envelope when the id isn't known.
pub(crate) async fn openai_retrieve_model(
    state: axum::extract::State<APIServer>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if let Err(e) = validate_model_id(&model_id) {
        return e.into_response();
    }
    let normalized = normalize_model_id(&model_id);
    let now = chrono::Utc::now().timestamp();

    // Look up among locally-cached LLM checkpoints first.
    if let Ok(models) = state.model_manager.list_models().await {
        if let Some(m) = models
            .iter()
            .find(|m| m.id == model_id || m.id == normalized)
        {
            let created = chrono::DateTime::parse_from_rfc3339(&m.downloaded_at)
                .map(|t| t.timestamp())
                .unwrap_or(now);
            let is_loaded = state
                .engines
                .read()
                .await
                .iter()
                .any(|e| e.model_id == m.id);
            return Json(serde_json::json!({
                "id": m.id,
                "object": "model",
                "created": created,
                "owned_by": "local",
                "kind": "chat",
                "size_bytes": m.size,
                "is_loaded": is_loaded,
            }))
            .into_response();
        }
    }

    // Then the multimodal entries we'd spin up on demand.
    if let Some((id, kind, owner, endpoint, hf_repo)) = MULTIMODAL_EXTRAS
        .iter()
        .find(|(id, ..)| *id == model_id || *id == normalized)
    {
        let is_loaded = multimodal_entry_is_loaded(&state, kind, hf_repo, id).await;
        return Json(serde_json::json!({
            "id": id,
            "object": "model",
            "created": now,
            "owned_by": owner,
            "kind": kind,
            "endpoint": endpoint,
            "hf_repo": hf_repo,
            "cached": hf_cache_has_model(hf_repo),
            "is_loaded": is_loaded,
        }))
        .into_response();
    }

    let code = axum::http::StatusCode::NOT_FOUND;
    (
        code,
        Json(openai_error_body(
            code,
            format!("Model '{model_id}' not found"),
        )),
    )
        .into_response()
}

/// Cohere/Jina-style `/v1/rerank` (+ `/rerank`). Scores each document against the
/// query with the loaded model used as a cross-encoder reranker (Qwen3-Reranker
/// recipe) and returns results sorted by descending relevance, optionally capped
/// to `top_n`. Body: `{model, query, documents:[..], top_n?, instruction?,
/// return_documents?}`.
pub(crate) async fn openai_rerank(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    // The request goes to a node whose catalogue holds the model when this one's does not.
    {
        let model = normalize_model_id(&text_field(&body, "model"));
        let holds = |n: &crate::distributed::membership::NodeState| {
            crate::distributed::routing::can_serve(n, &model)
        };
        let served_here = holds(&state.local_node_state().await);
        if let Some(relayed) = super::route_media_to_holder(
            &state,
            &headers,
            &model,
            served_here,
            true,
            0,
            holds,
            &super::RERANK,
            &body,
        )
        .await
        {
            return relayed;
        }
    }
    use axum::response::IntoResponse;
    let err = |code: axum::http::StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };

    let model = text_field(&body, "model");
    let query = match body.get("query").and_then(|v| v.as_str()) {
        Some(q) if !q.trim().is_empty() => q.to_string(),
        _ => {
            return err(
                axum::http::StatusCode::BAD_REQUEST,
                "`query` must be a non-empty string".into(),
            )
        }
    };
    let documents: Vec<String> = match body.get("documents") {
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => {
            return err(
                axum::http::StatusCode::BAD_REQUEST,
                "`documents` must be an array of strings".into(),
            )
        }
    };
    if documents.is_empty() {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            "`documents` must not be empty".into(),
        );
    }
    const RERANK_MAX_DOCS: usize = 1000;
    if documents.len() > RERANK_MAX_DOCS {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            format!(
                "`documents` has {} entries; limit is {RERANK_MAX_DOCS}",
                documents.len()
            ),
        );
    }
    let instruction = text_field(&body, "instruction");
    let return_documents = body
        .get("return_documents")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if let Err(e) = validate_model_id(&model) {
        return e.into_response();
    }
    let model_name = normalize_model_id(&model);
    let engine = match state.get_engine(&model_name).await {
        Ok(e) => e,
        Err(_) => {
            return err(
                axum::http::StatusCode::NOT_FOUND,
                format!("Model '{model_name}' not loaded. Pull it first with POST /api/pull."),
            )
        }
    };
    state.reset_expiration(&model_name).await;

    let t0 = std::time::Instant::now();
    let mut scored: Vec<(usize, f32)> = Vec::with_capacity(documents.len());
    for (i, doc) in documents.iter().enumerate() {
        match engine
            .rerank_score(query.clone(), doc.clone(), instruction.clone())
            .await
        {
            Ok(s) => scored.push((i, s)),
            Err(e) => {
                return err(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("rerank (idx={i}): {e}"),
                )
            }
        }
    }
    // Sort by descending relevance; cap to top_n if provided.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(n) = body.get("top_n").and_then(serde_json::Value::as_u64) {
        scored.truncate(n as usize);
    }
    let results: Vec<serde_json::Value> = scored
        .iter()
        .map(|&(idx, score)| {
            let mut o = serde_json::json!({ "index": idx, "relevance_score": score });
            if return_documents {
                o["document"] = serde_json::json!({ "text": documents[idx] });
            }
            o
        })
        .collect();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Rerank: model={model_name} n={} top={} {ms:.0}ms",
        documents.len(),
        results.len()
    );
    (
        axum::http::StatusCode::OK,
        Json(serde_json::json!({
            "model": model_name,
            "object": "list",
            "results": results,
        })),
    )
        .into_response()
}

/// OpenAI-compatible `/v1/embeddings`. Shares the same engine path as
/// /api/embed but reshapes the response to OpenAI's documented
/// envelope so SDKs (langchain, openai-python, llama-index) parse it
/// directly.
pub(crate) async fn openai_embeddings(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> axum::response::Response {
    // The request goes to a node whose catalogue holds the model when this one's does not.
    {
        let model = normalize_model_id(&text_field(&body, "model"));
        let holds = |n: &crate::distributed::membership::NodeState| {
            crate::distributed::routing::can_serve(n, &model)
        };
        let served_here = holds(&state.local_node_state().await);
        if let Some(relayed) = super::route_media_to_holder(
            &state,
            &headers,
            &model,
            served_here,
            true,
            0,
            holds,
            &super::EMBEDDINGS,
            &body,
        )
        .await
        {
            return relayed;
        }
    }
    use axum::response::IntoResponse;

    let err = |code: axum::http::StatusCode, msg: String| -> axum::response::Response {
        (code, Json(openai_error_body(code, msg))).into_response()
    };

    let model = text_field(&body, "model");
    // Empty / whitespace-only / oversized / path-traversal model IDs are
    // rejected by `validate_model_id` below with the same 400 envelope.
    let inputs: Vec<String> = match body.get("input") {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(std::string::ToString::to_string))
            .collect(),
        _ => {
            return err(
                axum::http::StatusCode::BAD_REQUEST,
                "`input` must be a string or array of strings".to_string(),
            )
        }
    };
    if inputs.is_empty() {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            "`input` must not be empty".to_string(),
        );
    }
    // Reject empty strings - OpenAI returns 400 here. The engine
    // tolerates them and returns a degenerate constant embedding,
    // which is worse than failing fast for the caller.
    if let Some(idx) = inputs.iter().position(|s| s.trim().is_empty()) {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            format!("`input[{idx}]` is empty; embeddings require non-empty text"),
        );
    }
    // Cap each input at 8192 chars (OpenAI's documented per-input
    // length for text-embedding-3-* is 8191 tokens; clamping here
    // prevents pathological inputs from tying up the embed path).
    // Cap array size at 2048 entries (OpenAI's documented batch cap).
    const EMBED_MAX_CHARS: usize = 8192;
    const EMBED_MAX_BATCH: usize = 2048;
    if let Some((idx, s)) = inputs
        .iter()
        .enumerate()
        .find(|(_, s)| s.chars().count() > EMBED_MAX_CHARS)
    {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            format!(
                "`input[{idx}]` is {} chars; limit is {EMBED_MAX_CHARS}",
                s.chars().count()
            ),
        );
    }
    if inputs.len() > EMBED_MAX_BATCH {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            format!(
                "`input` has {} entries; limit is {EMBED_MAX_BATCH}",
                inputs.len()
            ),
        );
    }

    // Validate request shape BEFORE the model lookup so payload bugs
    // surface as 400 directly (otherwise an unloaded model returns
    // 404 first and the caller has no idea their other params were
    // also bad).
    //
    // OpenAI's per-request encoding_format: "float" (default - JSON
    // array of f32) or "base64" (compact - little-endian f32 bytes
    // -> base64 string). Compact form roughly halves the JSON payload
    // size for high-dim embeddings.
    let encoding_format = body
        .get("encoding_format")
        .and_then(|v| v.as_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| "float".to_string());
    if encoding_format != "float" && encoding_format != "base64" {
        return err(
            axum::http::StatusCode::BAD_REQUEST,
            format!("encoding_format '{encoding_format}' not supported; use 'float' or 'base64'"),
        );
    }
    // Optional dimensions: trim each embedding to the first N coords.
    // OpenAI's text-embedding-3-* uses Matryoshka representation
    // (early dims carry the most info); plain truncation is a
    // reasonable approximation that loses minimal quality.
    let dims_req = body
        .get("dimensions")
        .and_then(serde_json::Value::as_u64)
        .map(|v| v as usize);
    if let Some(d) = dims_req {
        if d == 0 {
            return err(
                axum::http::StatusCode::BAD_REQUEST,
                "`dimensions` must be >= 1".to_string(),
            );
        }
    }

    if let Err(e) = validate_model_id(&model) {
        return e.into_response();
    }
    let model_name = normalize_model_id(&model);
    // Loads on demand, as the chat and completion endpoints do.
    let keep_alive = state.get_effective_keep_alive(None);
    if let Err(e) = state.ensure_loaded(&model_name, keep_alive).await {
        return ApiError::NotFound(format!("model '{model_name}' could not be loaded: {e}"))
            .into_response();
    }
    let engine = match state.get_engine(&model_name).await {
        Ok(e) => e,
        Err(_) => {
            return err(
                axum::http::StatusCode::NOT_FOUND,
                format!("Model '{model_name}' not loaded. Pull it first with POST /api/pull."),
            )
        }
    };
    state.reset_expiration(&model_name).await;

    let embed_start = std::time::Instant::now();
    let mut data = Vec::with_capacity(inputs.len());
    // Capture the true embedding vector length (post-truncation) once
    // so the log reports the actual dim regardless of encoding format
    // - base64 strings have a length that's unrelated to the vector
    // size, which used to mislead the log.
    let mut emb_dim: usize = 0;
    for (index, input) in inputs.iter().enumerate() {
        match engine.generate_embeddings(input).await {
            Ok(mut embedding) => {
                if let Some(d) = dims_req {
                    if d < embedding.len() {
                        embedding.truncate(d);
                    }
                }
                if emb_dim == 0 {
                    emb_dim = embedding.len();
                }
                let emb_value = if encoding_format == "base64" {
                    use base64::Engine as _;
                    let mut bytes = Vec::with_capacity(embedding.len() * 4);
                    for f in &embedding {
                        bytes.extend_from_slice(&f.to_le_bytes());
                    }
                    serde_json::Value::String(
                        base64::engine::general_purpose::STANDARD.encode(&bytes),
                    )
                } else {
                    serde_json::json!(embedding)
                };
                data.push(serde_json::json!({
                    "object": "embedding",
                    "embedding": emb_value,
                    "index": index,
                }));
            }
            Err(e) => {
                return err(
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("embeddings (idx={index}): {e}"),
                )
            }
        }
    }
    let embed_ms = embed_start.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Embeddings: model={model_name} n={} dim={emb_dim} encoding={encoding_format} embed={embed_ms:.0}ms",
        inputs.len(),
    );

    // OpenAI's `usage.{prompt_tokens, total_tokens}` - embeddings don't
    // generate, so completion_tokens isn't included. Estimate prompt
    // tokens with the same word-count heuristic the chat path uses.
    let prompt_tokens: u64 = inputs.iter().map(|s| estimate_token_count(s)).sum();

    let body = Json(serde_json::json!({
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens,
        },
    }));
    let mut headers = axum::http::HeaderMap::new();
    let st = format!("embed;dur={embed_ms:.1}");
    if let Ok(hv) = axum::http::HeaderValue::from_str(&st) {
        headers.insert("server-timing", hv);
    }
    (headers, body).into_response()
}

mod files;
pub(crate) use files::*;
mod moderation;
pub(crate) use moderation::*;
mod responses;
pub(crate) use responses::*;
#[cfg(test)]
mod tests;

/// `DELETE /v1/models/{id}`: unloads the model if it is resident and removes it from
/// the store, the way `/api/delete` does.
pub(crate) async fn openai_delete_model(
    State(state): State<APIServer>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_model_id(&model_id)?;
    let name = normalize_model_id(&model_id);
    {
        let mut engines = state.engines.write().await;
        if let Some(pos) = engines.iter().position(|e| e.model_id == name) {
            let entry = engines.remove(pos);
            if let Some(handle) = entry.expire_handle {
                handle.abort();
            }
            let _ = entry.engine.unload().await;
        }
    }
    state
        .model_manager
        .delete_model(&name)
        .await
        .map_err(|e| ApiError::NotFound(format!("model '{model_id}': {e}")))?;
    Ok(Json(
        serde_json::json!({"id": model_id, "object": "model", "deleted": true}),
    ))
}
