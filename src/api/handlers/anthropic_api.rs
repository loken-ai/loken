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
        return anthropic_error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{e:?}"),
        );
    }
    let model_name = normalize_model_id(&req.model);
    // The conversation goes where the model is.
    if let Ok(body) = serde_json::to_value(&req) {
        let prompt_text = body["messages"].to_string();
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

    if let Some(hint) = non_chat_pipeline_component(&model_name) {
        return anthropic_error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("model '{model_name}' is not a chat model: {hint}"),
        );
    }

    // Auto-load on demand (Claude Code expects to just name a model).
    let keep_alive = state.get_effective_keep_alive(None);
    if let Err(e) = state.ensure_loaded(&model_name, keep_alive).await {
        return anthropic_error_response(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("model '{model_name}' could not be loaded: {e}"),
        );
    }
    let engine = match state.get_engine(&model_name).await {
        Ok(e) => e,
        Err(_) => {
            return anthropic_error_response(
                StatusCode::NOT_FOUND,
                "not_found_error",
                format!("model '{model_name}' not loaded"),
            );
        }
    };
    state.reset_expiration(&model_name).await;

    let chat_template = state
        .read_chat_template(&model_name)
        .or_else(|| infer_template_from_model_name(&model_name));

    // Pull scalar fields before `into_internal` consumes the request.
    let stream = req.is_stream();
    let max_tokens = req.effective_max_tokens();
    let temperature = req.temperature;
    let top_p = req.top_p;
    let stop_sequences = req.stop_sequences.clone().unwrap_or_default();
    let tools_active = req.tools_enabled();
    let tool_format =
        crate::api::tool_calls::detect_tool_format(chat_template.as_deref(), &model_name);
    let tool_directive = crate::api::tool_calls::tool_choice_directive(req.tool_choice.as_ref());
    let model_for_resp = model_name.clone();
    let top_k = req.top_k;
    let seed = req.seed;
    let thinking_on = req
        .thinking
        .as_ref()
        .and_then(|t| t.get("type"))
        .and_then(serde_json::Value::as_str)
        == Some("enabled");
    let single_tool_call = req
        .tool_choice
        .as_ref()
        .and_then(|c| c.get("disable_parallel_tool_use"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    // A required call (`any`, or one named function) constrains the generation to the
    // call object itself; a structured output constrains it to the client's schema.
    let forced_call: Option<Option<String>> = match req
        .tool_choice
        .as_ref()
        .and_then(|c| c.get("type"))
        .and_then(serde_json::Value::as_str)
    {
        Some("any") => Some(None),
        Some("tool") => Some(
            req.tool_choice
                .as_ref()
                .and_then(|c| c.get("name"))
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
        ),
        _ => None,
    };
    let output_schema = req
        .output_format
        .as_ref()
        .filter(|f| f.get("type").and_then(serde_json::Value::as_str) == Some("json_schema"))
        .and_then(|f| f.get("schema").cloned());

    let (messages, tools) = req.into_internal();
    let grammar = if let Some(schema) = output_schema {
        Some(format!("json_schema:{schema}"))
    } else if let (Some(only), true) = (forced_call.as_ref(), tools_active) {
        Some(format!(
            "json_schema:{}",
            crate::api::tool_calls::forced_call_schema(&tools, only.as_deref())
        ))
    } else {
        None
    };
    let prompt = if tools_active {
        let flat = crate::api::tool_calls::flatten_messages(
            &messages,
            tool_format,
            &tools,
            tool_directive.as_deref(),
        );
        if template_takes_tools(chat_template.as_deref()) {
            format_chat_prompt_with_tools(&messages, chat_template.as_deref(), &tools)
        } else {
            format_chat_prompt(&flat, chat_template.as_deref())
        }
    } else {
        format_chat_prompt(&messages, chat_template.as_deref())
    };
    let mut prompt = prompt;
    if !thinking_on {
        super::prompt_format::apply_thinking_preference(&mut prompt, Some("disabled"));
    }
    // Whether the template already opened the thinking block; the splitter must know.
    let thinking_opened = crate::api::thinking::prompt_opens_thinking(&prompt);
    debug!("Anthropic /v1/messages: model={model_name} stream={stream} tools={tools_active} prompt_chars={}", prompt.len());

    let params = GenerationParams {
        prefix_tokens: None,
        max_tokens: Some(max_tokens),
        temperature,
        top_p,
        top_k: top_k.map(|k| k as _),
        seed: seed.map(|s| s as _),
        stop_sequences,
        early_exit_threshold: None,
        repeat_penalty: None,
        repeat_last_n: None,
        context_length: None,
        session_id: None,
        grammar,
        logit_bias: None,
        top_logprobs: None,
    };

    let gate_guard = match state
        .request_gate
        .acquire_with_info_timeout(
            crate::api::gate::Priority::Interactive,
            model_name.clone(),
            "/v1/messages",
            GATE_WAIT,
        )
        .await
    {
        Some(g) => g,
        None => {
            return anthropic_error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "overloaded_error",
                format!(
                    "no inference slot became available within {}s; retry later",
                    GATE_WAIT.as_secs()
                ),
            );
        }
    };
    // The last user turn's images go to the vision encoder; none clears what an earlier
    // request left there. After the gate: both take the model lock.
    let images: Vec<String> = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.images.clone())
        .unwrap_or_default();
    if images.is_empty() {
        engine.clear_images().await;
    } else if let Err(e) = engine.set_images(&images).await {
        return anthropic_error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("images: {e}"),
        );
    }

    if !stream {
        let _g = gate_guard;
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
                    (result.prompt_eval_count + result.cached_prompt_tokens) as i32,
                    result.eval_count as i32,
                    hit_max,
                    stop_seq,
                    result.cached_prompt_tokens as i32,
                    result.prompt_eval_count as i32,
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
                    let _gate_held = gate_guard;
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
                                    break;
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
                    let (stop_seq, hit_max, in_tokens, cache_read, cache_creation) = match stats.as_ref() {
                        Some(s) => {
                            let (kind, seq) = s.finish_reason.anthropic();
                            (
                                seq.map(str::to_string),
                                kind == "max_tokens" && !used_tools,
                                (s.prompt_eval_count + s.cached_prompt_tokens) as i32,
                                s.cached_prompt_tokens as i32,
                                s.prompt_eval_count as i32,
                            )
                        }
                        None => (None, (output_tokens as usize) >= max_tokens && !used_tools, input_tokens, 0, 0),
                    };
                    yield Ok(Event::default()
                        .event("message_delta")
                        .data(anthropic::sse::message_delta(
                            used_tools,
                            hit_max,
                            stop_seq.as_deref(),
                            output_tokens,
                            in_tokens,
                            cache_read,
                            cache_creation,
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
        return anthropic_error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            format!("{e:?}"),
        );
    }
    let model_name = normalize_model_id(&req.model);
    let chat_template = state
        .read_chat_template(&model_name)
        .or_else(|| infer_template_from_model_name(&model_name));
    let tools_active = req.tools_enabled();
    let tool_format =
        crate::api::tool_calls::detect_tool_format(chat_template.as_deref(), &model_name);
    let tool_directive = crate::api::tool_calls::tool_choice_directive(req.tool_choice.as_ref());
    let (messages, tools) = req.into_internal();
    let prompt = if tools_active {
        let flat = crate::api::tool_calls::flatten_messages(
            &messages,
            tool_format,
            &tools,
            tool_directive.as_deref(),
        );
        if template_takes_tools(chat_template.as_deref()) {
            format_chat_prompt_with_tools(&messages, chat_template.as_deref(), &tools)
        } else {
            format_chat_prompt(&flat, chat_template.as_deref())
        }
    } else {
        format_chat_prompt(&messages, chat_template.as_deref())
    };
    let counted = match state.get_engine(&model_name).await {
        Ok(engine) => engine.count_tokens(&prompt).await,
        Err(_) => None,
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
