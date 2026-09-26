//! Anthropic-compatible /v1/messages handlers.

use super::*;

/// Build an Anthropic-shaped error response (`{type:error,error:{...}}`)
/// so Anthropic-protocol clients (Claude Code) parse failures correctly
/// instead of choking on the OpenAI envelope. Use ONLY on the /v1/messages
/// path; everything else speaks the OpenAI shape (`openai_error_body`,
/// reached via `conv_err` / the per-handler `err_resp` closures).
pub(crate) fn anthropic_error_response(
    status: axum::http::StatusCode,
    kind: &str,
    message: String,
) -> Response {
    let body = crate::api::anthropic::AnthropicError::new(kind, message);
    (status, Json(body)).into_response()
}

/// An [`ApiError`] in the Messages API's envelope, with the error type its status
/// carries there.
pub(crate) fn anthropic_api_error(e: ApiError) -> Response {
    use axum::http::StatusCode;
    let (status, kind, message) = match e {
        ApiError::Validation(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m),
        ApiError::NotFound(m) => (StatusCode::NOT_FOUND, "not_found_error", m),
        ApiError::Busy(m) => (StatusCode::SERVICE_UNAVAILABLE, "overloaded_error", m),
        ApiError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, "api_error", m),
    };
    anthropic_error_response(status, kind, message)
}

/// Anthropic Messages API shim (`POST /v1/messages`). Translates the
/// Anthropic request into the internal Message/Tool shapes, runs the same
/// engine + tool-calling pipeline as the OpenAI path, and lifts the output
/// back into Anthropic content blocks. Supports streaming via the
/// Anthropic SSE event protocol.
pub(crate) async fn anthropic_messages(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    AnthropicJson(req): AnthropicJson<crate::api::anthropic::AnthropicMessagesRequest>,
) -> Response {
    use crate::api::anthropic;
    use axum::http::StatusCode;

    if let Err(e) = validate_model_id(&req.model) {
        return anthropic_api_error(e);
    }
    let model_name = normalize_model_id(&req.model);
    // The conversation goes where the model is. Peers are asked about the prompt this
    // node would prefill, which is the text their caches are keyed on.
    let clustered = state.cluster_handle().is_some();
    if let Some(body) = clustered.then(|| serde_json::to_value(&req).ok()).flatten() {
        let tool_choice = req.tool_choice.clone();
        let thinking = req.thinking_preference();
        let (messages, tools) = req.clone().into_internal();
        let prompt_text = state
            .prepare_prompt(
                &model_name,
                None,
                &super::chat_prompt::PromptSpec {
                    messages: &messages,
                    tools: Some(&tools),
                    tool_choice: tool_choice.as_ref(),
                    thinking,
                    output_grammar: None,
                },
            )
            .await
            .prompt;
        let max_tokens = req
            .max_tokens
            .unwrap_or(state.default_inference_config.max_tokens) as u32;
        if let Some(relayed) = super::route_to_holder(
            &state,
            &headers,
            &model_name,
            &prompt_text,
            max_tokens,
            &super::MESSAGES,
            &body,
        )
        .await
        {
            return relayed;
        }
    }
    if let Some(why) = req.unsupported_input() {
        return anthropic_error_response(StatusCode::BAD_REQUEST, "invalid_request_error", why);
    }
    // The same sampling bounds as every other route.
    let sampling = super::sampling::SamplingRequest::from_anthropic(&req);
    if let Err(e) = sampling.validate() {
        return anthropic_api_error(e);
    }

    if let Some(hint) = non_chat_pipeline_component(&model_name) {
        return anthropic_error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("model '{model_name}' is not a chat model: {hint}"),
        );
    }

    // Loaded on demand (Claude Code expects to just name a model), in the window and
    // cache format the headers ask for.
    let load =
        super::admission::LoadRequest::from_headers(&headers, state.get_effective_keep_alive(None));
    let engine = match state.load_for_request(&model_name, load).await {
        Ok(e) => e,
        Err(e) => return anthropic_api_error(e),
    };

    // Pull scalar fields before `into_internal` consumes the request.
    let stream = req.is_stream();
    let max_tokens = req.effective_max_tokens();
    let model_for_resp = model_name.clone();
    let single_tool_call = req
        .tool_choice
        .as_ref()
        .and_then(|c| c.get("disable_parallel_tool_use"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let tool_choice = req.tool_choice.clone();
    let thinking = req.thinking_preference();
    let output_grammar = req.output_schema().and_then(json_schema_grammar_str);
    let (messages, tools) = req.into_internal();
    let super::chat_prompt::PreparedPrompt {
        prompt,
        tools_active,
        tool_format,
        grammar,
        thinking_opened,
    } = state
        .prepare_prompt(
            &model_name,
            Some(&engine),
            &super::chat_prompt::PromptSpec {
                messages: &messages,
                tools: Some(&tools),
                tool_choice: tool_choice.as_ref(),
                thinking,
                output_grammar,
            },
        )
        .await;
    debug!("Anthropic /v1/messages: model={model_name} stream={stream} tools={tools_active} prompt_chars={}", prompt.len());

    let params = GenerationParams {
        grammar,
        ..sampling.into_params()
    };

    // The gate, the in-flight count and the last user turn's images.
    let priority = crate::api::gate::Priority::from_headers(&headers)
        .unwrap_or(crate::api::gate::Priority::Interactive);
    let admission = match state
        .admit(
            engine.clone(),
            &model_name,
            priority,
            "/v1/messages",
            &super::chat_prompt::last_user_images(&messages),
        )
        .await
    {
        Ok(a) => a,
        Err(e) => return anthropic_api_error(e),
    };

    if !stream {
        let _admitted = admission;
        let _cancel = engine.cancel_guard();
        match engine.generate(&prompt, params).await {
            Ok(result) => {
                let (thinking, answer) =
                    crate::api::thinking::split_thinking_opened(thinking_opened, &result.text);
                let (text, calls) = if tools_active {
                    let mut parsed = crate::api::tool_calls::parse_tool_calls(tool_format, &answer);
                    if single_tool_call {
                        parsed.calls.truncate(1);
                    }
                    (parsed.content, parsed.calls)
                } else {
                    (answer.clone(), Vec::new())
                };
                let (stop_kind, stop_seq) = result.finish_reason.anthropic();
                let hit_max = stop_kind == "max_tokens" && calls.is_empty();
                let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
                let body = anthropic::build_response(
                    &id,
                    &model_for_resp,
                    thinking.as_deref(),
                    &text,
                    &calls,
                    anthropic::MessageUsage::from_counts(
                        result.prompt_eval_count,
                        result.cached_prompt_tokens,
                        result.eval_count,
                    ),
                    hit_max,
                    stop_seq,
                );
                (StatusCode::OK, Json(body)).into_response()
            }
            Err(e) => anthropic_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                format!("generate: {e}"),
            ),
        }
    } else {
        let input_tokens = engine
            .count_tokens(&prompt)
            .await
            .map(|n| n as i32)
            .unwrap_or_else(|| estimate_token_count(&prompt) as i32);
        let engine_for_stats = engine.clone();
        match engine.generate_stream(&prompt, params).await {
            Ok(mut rx) => {
                let model_clone = model_for_resp.clone();
                let event_stream = async_stream::stream! {
                    let _admitted = admission;
                    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
                    // Anthropic event prologue: message_start, open text
                    // block 0, a ping.
                    yield Ok::<_, axum::Error>(Event::default()
                        .event("message_start")
                        .data(anthropic::sse::message_start(&id, &model_clone, input_tokens).to_string()));
                    yield Ok(Event::default()
                        .event("ping")
                        .data(anthropic::sse::ping().to_string()));
                    let mut scanner = if tools_active {
                        Some(crate::api::tool_calls::StreamToolScanner::new(tool_format))
                    } else {
                        None
                    };
                    // Blocks are numbered as they open: a thinking block first when the
                    // model reasons, then the text, then the tool uses.
                    let mut splitter = crate::api::thinking::ThinkSplit::opened(thinking_opened);
                    let mut next_idx = 0usize;
                    let mut think_idx: Option<usize> = None;
                    let mut text_idx: Option<usize> = None;
                    let mut output_tokens = 0i32;
                    let mut segments: Vec<crate::api::thinking::Segment> = Vec::new();
                    let mut ended = false;
                    loop {
                        if segments.is_empty() {
                            if ended {
                                break;
                            }
                            match rx.recv().await {
                                Some(Ok(tok)) => {
                                    output_tokens += 1;
                                    segments = splitter.push(&tok);
                                    segments.reverse();
                                }
                                Some(Err(e)) => {
                                    yield Ok(Event::default()
                                        .event("error")
                                        .data(anthropic::sse::error("api_error", &format!("generate: {e}")).to_string()));
                                    // The error event ends the stream: no message_delta or
                                    // message_stop follows it.
                                    return;
                                }
                                None => {
                                    ended = true;
                                    segments = splitter.finish();
                                    segments.reverse();
                                }
                            }
                            continue;
                        }
                        let Some(seg) = segments.pop() else { continue };
                        match seg {
                            crate::api::thinking::Segment::Thinking(t) => {
                                if think_idx.is_none() {
                                    think_idx = Some(next_idx);
                                    next_idx += 1;
                                    yield Ok(Event::default()
                                        .event("content_block_start")
                                        .data(anthropic::sse::thinking_block_start(next_idx - 1).to_string()));
                                }
                                yield Ok(Event::default()
                                    .event("content_block_delta")
                                    .data(anthropic::sse::thinking_delta(think_idx.unwrap_or(0), &t).to_string()));
                            }
                            crate::api::thinking::Segment::Content(c) => {
                                let emit = match scanner.as_mut() {
                                    Some(sc) => sc.push(&c),
                                    None => c,
                                };
                                if emit.is_empty() {
                                    continue;
                                }
                                if text_idx.is_none() {
                                    if let Some(ti) = think_idx {
                                        yield Ok(Event::default()
                                            .event("content_block_delta")
                                            .data(anthropic::sse::signature_delta(ti).to_string()));
                                        yield Ok(Event::default()
                                            .event("content_block_stop")
                                            .data(anthropic::sse::block_stop(ti).to_string()));
                                    }
                                    text_idx = Some(next_idx);
                                    next_idx += 1;
                                    yield Ok(Event::default()
                                        .event("content_block_start")
                                        .data(anthropic::sse::text_block_start(next_idx - 1).to_string()));
                                }
                                yield Ok(Event::default()
                                    .event("content_block_delta")
                                    .data(anthropic::sse::text_delta(text_idx.unwrap_or(0), &emit).to_string()));
                            }
                        }
                    }
                    let mut calls = scanner.as_ref().map(|s| s.finalize()).unwrap_or_default();
                    if single_tool_call {
                        calls.truncate(1);
                    }
                    let used_tools = !calls.is_empty();
                    let tail = if used_tools {
                        None
                    } else {
                        scanner.as_ref().map(|s| s.unstreamed()).filter(|t| !t.is_empty())
                    };
                    if tail.is_some() && text_idx.is_none() {
                        if let Some(ti) = think_idx {
                            yield Ok(Event::default()
                                .event("content_block_delta")
                                .data(anthropic::sse::signature_delta(ti).to_string()));
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(anthropic::sse::block_stop(ti).to_string()));
                        }
                        text_idx = Some(next_idx);
                        next_idx += 1;
                        yield Ok(Event::default()
                            .event("content_block_start")
                            .data(anthropic::sse::text_block_start(next_idx - 1).to_string()));
                    }
                    if let Some(t) = tail {
                        yield Ok(Event::default()
                            .event("content_block_delta")
                            .data(anthropic::sse::text_delta(text_idx.unwrap_or(0), t).to_string()));
                    }
                    match (think_idx, text_idx) {
                        (Some(ti), None) => {
                            yield Ok(Event::default()
                                .event("content_block_delta")
                                .data(anthropic::sse::signature_delta(ti).to_string()));
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(anthropic::sse::block_stop(ti).to_string()));
                        }
                        (_, Some(xi)) => {
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(anthropic::sse::block_stop(xi).to_string()));
                        }
                        (None, None) => {
                            // Nothing streamed: an empty text block keeps the message well formed.
                            yield Ok(Event::default()
                                .event("content_block_start")
                                .data(anthropic::sse::text_block_start(next_idx).to_string()));
                            yield Ok(Event::default()
                                .event("content_block_stop")
                                .data(anthropic::sse::block_stop(next_idx).to_string()));
                            next_idx += 1;
                        }
                    }
                    let mut idx = next_idx;
                    for c in &calls {
                        yield Ok(Event::default()
                            .event("content_block_start")
                            .data(anthropic::sse::tool_block_start(idx, c).to_string()));
                        let args = c.function.as_ref()
                            .and_then(|f| f.arguments.clone())
                            .unwrap_or_else(|| "{}".to_string());
                        yield Ok(Event::default()
                            .event("content_block_delta")
                            .data(anthropic::sse::tool_input_delta(idx, &args).to_string()));
                        yield Ok(Event::default()
                            .event("content_block_stop")
                            .data(anthropic::sse::block_stop(idx).to_string()));
                        idx += 1;
                    }

                    let stats = engine_for_stats.take_last_stream_stats();
                    // What the engine counted; the chunk count and the prompt estimate only
                    // where it published nothing.
                    let (stop_seq, hit_max, usage) = match stats.as_ref() {
                        Some(s) => {
                            let (kind, seq) = s.finish_reason.anthropic();
                            (
                                seq.map(str::to_string),
                                kind == "max_tokens" && !used_tools,
                                anthropic::MessageUsage::from_counts(
                                    s.prompt_eval_count,
                                    s.cached_prompt_tokens,
                                    s.eval_count,
                                ),
                            )
                        }
                        None => (
                            None,
                            (output_tokens as usize) >= max_tokens && !used_tools,
                            anthropic::MessageUsage {
                                input: input_tokens,
                                output: output_tokens,
                                ..Default::default()
                            },
                        ),
                    };
                    yield Ok(Event::default()
                        .event("message_delta")
                        .data(anthropic::sse::message_delta(
                            used_tools,
                            hit_max,
                            stop_seq.as_deref(),
                            usage,
                        ).to_string()));
                    yield Ok(Event::default()
                        .event("message_stop")
                        .data(anthropic::sse::message_stop().to_string()));
                };
                Sse::new(event_stream)
                    .keep_alive(KeepAlive::default())
                    .into_response()
            }
            Err(e) => anthropic_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "api_error",
                format!("generate_stream: {e}"),
            ),
        }
    }
}

/// `POST /v1/messages/count_tokens`: the prompt a request would prefill, counted by
/// the model's own tokenizer once the model is loaded, estimated before.
pub(crate) async fn anthropic_count_tokens(
    State(state): State<APIServer>,
    AnthropicJson(req): AnthropicJson<crate::api::anthropic::AnthropicMessagesRequest>,
) -> Response {
    use axum::http::StatusCode;
    if let Err(e) = validate_model_id(&req.model) {
        return anthropic_api_error(e);
    }
    let model_name = normalize_model_id(&req.model);
    let tool_choice = req.tool_choice.clone();
    let thinking = req.thinking_preference();
    let (messages, tools) = req.into_internal();
    let engine = state.get_engine(&model_name).await.ok();
    // The same prompt the Messages route would prefill, thinking preference included.
    let prompt = state
        .prepare_prompt(
            &model_name,
            engine.as_deref(),
            &super::chat_prompt::PromptSpec {
                messages: &messages,
                tools: Some(&tools),
                tool_choice: tool_choice.as_ref(),
                thinking,
                output_grammar: None,
            },
        )
        .await
        .prompt;
    let counted = match engine {
        Some(engine) => engine.count_tokens(&prompt).await,
        None => None,
    };
    let input_tokens = counted.unwrap_or_else(|| estimate_token_count(&prompt) as usize);
    Json(serde_json::json!({ "input_tokens": input_tokens })).into_response()
}

/// `GET /v1/models` in the shape of the client asking: an `anthropic-version` header
/// gets the Messages API's list, anyone else OpenAI's.
pub(crate) async fn list_models_by_dialect(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
) -> Response {
    if !headers.contains_key("anthropic-version") {
        return super::openai::openai_list_models(State(state), headers).await;
    }
    let mut data: Vec<serde_json::Value> = Vec::new();
    let listed = chrono::Utc::now().to_rfc3339();
    let mut local: Vec<String> = Vec::new();
    if let Ok(mut models) = state.model_manager.list_models().await {
        models.sort_by(|a, b| a.id.cmp(&b.id));
        for m in models {
            local.push(m.id.clone());
            data.push(serde_json::json!({
                "type": "model",
                "id": m.id,
                "display_name": m.id,
                "created_at": listed,
            }));
        }
    }
    // What the peers hold and this node does not; a request naming one is forwarded.
    let peers = super::cluster_catalogue::peer_models(&state, &headers).await;
    for m in super::cluster_catalogue::merge(Vec::new(), peers) {
        if local.contains(&m.name) {
            continue;
        }
        data.push(serde_json::json!({
            "type": "model",
            "id": m.name,
            "display_name": m.name,
            "created_at": listed,
        }));
    }
    let first = data
        .first()
        .and_then(|m| m["id"].as_str().map(str::to_string));
    let last = data
        .last()
        .and_then(|m| m["id"].as_str().map(str::to_string));
    Json(serde_json::json!({
        "data": data,
        "has_more": false,
        "first_id": first,
        "last_id": last,
    }))
    .into_response()
}
