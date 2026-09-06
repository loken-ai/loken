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

/// Extract generation options from Ollama options field
/// Range-validate the Ollama options bag before extraction. Mirrors
/// the chat/text-completion sampling-knob bounds enforced via
/// validator macros on the OpenAI shape (e7a96bb). Returns 400 on
/// out-of-range values so /api/chat + /api/generate produce the same
/// errors as /v1/* for identical inputs.
pub(super) fn validate_ollama_options(options: Option<&serde_json::Value>) -> Result<(), ApiError> {
    let Some(opts) = options else {
        return Ok(());
    };
    if let Some(t) = opts.get("temperature").and_then(serde_json::Value::as_f64) {
        if !(0.0..=2.0).contains(&t) || !t.is_finite() {
            return Err(ApiError::Validation(format!(
                "options.temperature must be in [0, 2]; got {t}"
            )));
        }
    }
    if let Some(p) = opts.get("top_p").and_then(serde_json::Value::as_f64) {
        if !(0.0..=1.0).contains(&p) || !p.is_finite() {
            return Err(ApiError::Validation(format!(
                "options.top_p must be in [0, 1]; got {p}"
            )));
        }
    }
    if let Some(rp) = opts
        .get("repeat_penalty")
        .and_then(serde_json::Value::as_f64)
    {
        // Ollama documents repeat_penalty in roughly [0.5, 2.0]. Allow
        // a wider [0, 4] to tolerate user experimentation but reject
        // negative / non-finite values that would invert preferences.
        if !(0.0..=4.0).contains(&rp) || !rp.is_finite() {
            return Err(ApiError::Validation(format!(
                "options.repeat_penalty must be in [0, 4]; got {rp}"
            )));
        }
    }
    if let Some(np) = opts.get("num_predict").and_then(serde_json::Value::as_i64) {
        // -1 means "until EOS" in Ollama; 0 / >0 set a hard cap.
        if np < -1 {
            return Err(ApiError::Validation(format!(
                "options.num_predict must be >= -1; got {np}"
            )));
        }
        // Same upper bound as ChatCompletionRequest::max_tokens (50d9ca1).
        // Without it, np = i64::MAX parses past the as_u64 consumer below
        // and the decode loop spins until process exit.
        if np > 131072 {
            return Err(ApiError::Validation(format!(
                "options.num_predict is {np}; cap at 131072 (128K). Use -1 for unbounded-until-EOS."
            )));
        }
    }
    if let Some(nc) = opts.get("num_ctx").and_then(serde_json::Value::as_i64) {
        if nc < 1 {
            return Err(ApiError::Validation(format!(
                "options.num_ctx must be >= 1; got {nc}"
            )));
        }
        // num_ctx allocates KV cache proportional to context size; an
        // unbounded value lets a single request OOM the GPU. 1M tokens
        // is well past anything the perimeter actually supports.
        if nc > 1_048_576 {
            return Err(ApiError::Validation(format!(
                "options.num_ctx is {nc}; cap at 1048576 (1M tokens) - anything larger would OOM the KV cache"
            )));
        }
    }
    // top_k bounds the per-step softmax candidate pool. Practical values
    // sit in [1, vocab_size]; vocabularies are 32K-256K. A request like
    // top_k = u64::MAX parses straight through the as_u64 consumer and
    // its only effect is to misallocate buffers in the sampler.
    if let Some(k) = opts.get("top_k").and_then(serde_json::Value::as_i64) {
        if k < 0 {
            return Err(ApiError::Validation(format!(
                "options.top_k must be >= 0; got {k} (0 = disable top_k filtering)"
            )));
        }
        if k > 1_048_576 {
            return Err(ApiError::Validation(format!(
                "options.top_k is {k}; cap at 1048576 - practical vocabularies top out at ~256K tokens"
            )));
        }
    }
    // repeat_last_n is the rolling window of past tokens scanned for
    // repetition penalty. Engine cost is O(repeat_last_n x vocab_size)
    // per step; without a cap a request like u64::MAX brings decode
    // to a halt at first token. Same upper bound as max_tokens.
    if let Some(rl) = opts
        .get("repeat_last_n")
        .and_then(serde_json::Value::as_i64)
    {
        if rl < 0 {
            return Err(ApiError::Validation(format!(
                "options.repeat_last_n must be >= 0; got {rl} (0 = disable repeat penalty)"
            )));
        }
        if rl > 131_072 {
            return Err(ApiError::Validation(format!(
                "options.repeat_last_n is {rl}; cap at 131072 (128K)"
            )));
        }
    }
    // Stop-sequence count cap, parity with the /v1/chat/completions
    // and /v1/completions guards. Engine's per-step stop scan is
    // O(stop_count x generated_tokens); OpenAI's documented limit
    // is 4.
    if let Some(arr) = opts.get("stop").and_then(|v| v.as_array()) {
        if arr.len() > 4 {
            return Err(ApiError::Validation(format!(
                "options.stop has {} entries; cap at 4",
                arr.len()
            )));
        }
        // Per-entry char cap matches the chat + completions paths so an
        // unbounded 1 GB stop string can't tie up the per-step matcher.
        const STOP_ENTRY_MAX_CHARS: usize = 256;
        for (i, v) in arr.iter().enumerate() {
            if let Some(s) = v.as_str() {
                if s.chars().count() > STOP_ENTRY_MAX_CHARS {
                    return Err(ApiError::Validation(format!(
                        "options.stop[{i}] is {} chars; cap at {STOP_ENTRY_MAX_CHARS}",
                        s.chars().count()
                    )));
                }
            }
        }
    }
    // Same per-session-key cap as /v1/{chat,}completions (validate_user_id):
    // session_id is the KV-cache HashMap key, must be bounded.
    if let Some(s) = opts.get("session_id").and_then(|v| v.as_str()) {
        if !s.is_empty() && s.len() > 256 {
            return Err(ApiError::Validation(format!(
                "options.session_id is {} chars; cap at 256",
                s.len()
            )));
        }
    }
    Ok(())
}

pub(super) fn extract_generation_options(options: Option<&serde_json::Value>) -> GenerationParams {
    let options = match options {
        Some(o) => o,
        None => return GenerationParams::default(),
    };

    GenerationParams {
        prefix_tokens: None,
        logit_bias: None,
        top_logprobs: None,
        max_tokens: options
            .get("num_predict")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize),
        temperature: options
            .get("temperature")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32),
        top_p: options
            .get("top_p")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32),
        top_k: options
            .get("top_k")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize),
        seed: options.get("seed").and_then(serde_json::Value::as_u64),
        stop_sequences: options
            .get("stop")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(std::string::ToString::to_string))
                    .collect()
            })
            .unwrap_or_default(),
        early_exit_threshold: options
            .get("early_exit_threshold")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32),
        repeat_penalty: options
            .get("repeat_penalty")
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32),
        repeat_last_n: options
            .get("repeat_last_n")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize),
        context_length: options
            .get("num_ctx")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as usize),
        session_id: options
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string),
        grammar: options
            .get("grammar")
            .and_then(|v| v.as_str())
            .map(std::string::ToString::to_string),
    }
}

/// Chat (POST /api/chat) - Ollama format
pub(crate) async fn ollama_chat(
    State(state): State<APIServer>,
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
    validate_ollama_options(request.options.as_ref())?;
    // Normalize model ID
    let model_name = normalize_model_id(&request.model);
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
    let mut params = extract_generation_options(request.options.as_ref());
    params.top_logprobs = if request.logprobs == Some(true) {
        Some(request.top_logprobs.unwrap_or(0).min(20))
    } else {
        None
    };
    let wants_logprobs = request.logprobs == Some(true);
    // Ollama's `format` field: `"json"` -> json_object grammar; a JSON
    // Schema object -> strict json_schema grammar. Bridges Ollama's
    // structured-output knob to the same llguidance path the OpenAI
    // `response_format` uses. `params.grammar` from options takes
    // precedence so explicit overrides via the options bag still win.
    if params.grammar.is_none() {
        params.grammar = ollama_format_to_grammar(request.format.as_ref());
    }

    // Per-request kv_quant override (reload if the loaded engine differs).
    if let Some(want) = extract_kv_quant_override(request.options.as_ref()) {
        if let Err(e) = state.ensure_engine_kv_quant(&model_name, want).await {
            return Err(ApiError::Internal(format!(
                "kv_quant reload for '{model_name}': {e}"
            )));
        }
    }
    if let Some(want) = request
        .options
        .as_ref()
        .and_then(|o| o.get("num_ctx"))
        .and_then(serde_json::Value::as_u64)
    {
        if let Err(e) = state.ensure_engine_context(&model_name, want as usize).await {
            return Err(ApiError::Internal(format!(
                "num_ctx reload for '{model_name}': {e}"
            )));
        }
    }

    // Auto-load on first use (Ollama-compatible).
    let keep_alive_minutes_chat = state.get_effective_keep_alive(request.keep_alive.as_deref());
    if let Err(e) = state
        .ensure_loaded(&model_name, keep_alive_minutes_chat)
        .await
    {
        return Err(ApiError::NotFound(format!(
            "Model '{model_name}' could not be auto-loaded: {e}"
        )));
    }
    let engine = match state.get_engine(&model_name).await {
        Ok(engine) => engine,
        Err(_) => {
            return Err(ApiError::NotFound(format!(
                "Model '{model_name}' not loaded after auto-load attempt"
            )));
        }
    };

    // Reset expiration timer on use (like Ollama)
    state.reset_expiration(&model_name).await;

    // Build prompt using model-specific chat template (read from Ollama manifest)
    let chat_template = {
        let engines = state.engines.read().await;
        engines
            .iter()
            .find(|e| e.model_id == model_name)
            .and_then(|e| e.chat_template.clone())
    };
    // Lazy-load template on first use if not yet cached
    let chat_template = chat_template.or_else(|| {
        let tmpl = state.read_chat_template(&model_name);
        if tmpl.is_some() {
            // Cache it for next time (fire-and-forget)
            let state2 = state.clone();
            let name2 = model_name.clone();
            let tmpl2 = tmpl.clone();
            tokio::spawn(async move {
                let mut engines = state2.engines.write().await;
                if let Some(entry) = engines.iter_mut().find(|e| e.model_id == name2) {
                    entry.chat_template = tmpl2;
                }
            });
        }
        tmpl
    });
    // Fallback: if no template found in manifest, synthesize one based on
    // model name. Some Ollama manifests omit the template layer (e.g.,
    // gemma4) - falling through to the [INST] default would produce
    // gibberish on a model that expects <start_of_turn>.
    let chat_template = chat_template.or_else(|| infer_template_from_model_name(&model_name));
    // Tool calling: when a non-empty `tools` array is present, inject the
    // tool definitions + round-trip prior calls/results into the prompt
    // and remember the family so the response can be parsed back out.
    // Ollama has no `tool_choice`, so injection is gated only on presence.
    let tools_active = crate::api::tool_calls::should_inject_tools(
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
    );
    let tool_directive =
        crate::api::tool_calls::tool_choice_directive(request.tool_choice.as_ref());
    // `required` or a named function constrains the generation to the call object.
    let forced_call: Option<Option<String>> = match request.tool_choice.as_ref() {
        Some(serde_json::Value::String(s)) if s == "required" => Some(None),
        Some(serde_json::Value::Object(o))
            if o.get("type").and_then(serde_json::Value::as_str) == Some("function") =>
        {
            Some(
                o.get("function")
                    .and_then(|f| f.get("name"))
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string),
            )
        }
        _ => None,
    };
    // `required` or a named function: the call object is the grammar, unless the
    // caller's `format` already set one.
    if params.grammar.is_none() && tools_active {
        if let (Some(only), Some(tools)) = (forced_call.as_ref(), request.tools.as_ref()) {
            params.grammar = Some(format!(
                "json_schema:{}",
                crate::api::tool_calls::forced_call_schema(tools, only.as_deref())
            ));
        }
    }
    let tool_format =
        crate::api::tool_calls::detect_tool_format(chat_template.as_deref(), &model_name);
    // qwen35moe (Qwen3-VL): the simple ChatML formatter doesn't render the
    // template's vision branch, so inject ONE `<|image_pad|>` (wrapped in
    // vision_start/end) into the last user message carrying images. The engine
    // expands that single sentinel to n_merged placeholders at prefill (once
    // preprocessing reveals the patch grid) and splices the ViT output there.
    let qwen35_vision_active = request
        .messages
        .iter()
        .any(|m| m.role == "user" && m.images.as_ref().is_some_and(|i| !i.is_empty()))
        && engine.is_qwen35_vision().await;
    let eff_messages = if qwen35_vision_active {
        let mut msgs = request.messages.clone();
        if let Some(um) = msgs
            .iter_mut()
            .rev()
            .find(|m| m.role == "user" && m.images.as_ref().is_some_and(|i| !i.is_empty()))
        {
            um.content = format!("<|vision_start|><|image_pad|><|vision_end|>{}", um.content);
        }
        msgs
    } else {
        request.messages.clone()
    };
    let mut prompt = if tools_active {
        let tools = request.tools.as_ref().unwrap();
        // Ollama has no tool_choice field -> no forcing directive.
        let flattened =
            crate::api::tool_calls::flatten_messages(
                &eff_messages,
                tool_format,
                tools,
                tool_directive.as_deref(),
            );
        if template_takes_tools(chat_template.as_deref()) {
            format_chat_prompt_with_tools(&eff_messages, chat_template.as_deref(), tools)
        } else {
            format_chat_prompt(&flattened, chat_template.as_deref())
        }
    } else {
        format_chat_prompt(&eff_messages, chat_template.as_deref())
    };
    // `thinking: "disabled"` - until now the field was parsed and consumed by
    // NOBODY. For reasoning families whose template opens the assistant turn in
    // ChatML style (qwen3 & co), the official enable_thinking=false rendering
    // appends an EMPTY think block after the assistant opener; the model then
    // answers directly. Only applied when the rendered prompt actually ends
    // with the ChatML opener, so non-reasoning templates are untouched.
    apply_thinking_preference(&mut prompt, request.thinking_preference());
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
        info!("Chat: stop sequences: {:?}", params.stop_sequences);
    }

    // Resolve priority for the gate. Chat has no `suffix` FIM auto-promote
    // path, so we only honor an explicit options.priority; default Interactive.
    let chat_priority = request
        .options
        .as_ref()
        .and_then(|v| v.get("priority"))
        .and_then(|v| v.as_str())
        .and_then(crate::api::gate::Priority::parse)
        .unwrap_or(crate::api::gate::Priority::Interactive);

    // Acquire gate before touching engine.set_images / clear_images (they
    // grab the model_state mutex held by the active decode). See the same
    // fix in ollama_generate.
    // CB-served models decode in their own background worker (not under the
    // model_state mutex), so they skip the single-request gate - letting
    // concurrent requests reach the worker and batch (the throughput win).
    let chat_gate_guard = if engine.is_continuous().await {
        None
    } else {
        Some(acquire_gate(&state, chat_priority, model_name.clone(), "/api/chat").await?)
    };

    // Process images from the last user message for vision models.
    let last_user_images: Vec<String> = request
        .messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .and_then(|m| m.images.clone())
        .unwrap_or_default();
    if !last_user_images.is_empty() {
        if let Err(e) = engine.set_images(&last_user_images).await {
            warn!("Failed to process images for vision model: {}", e);
        } else {
            debug!(
                "Vision: {} image(s) encoded for chat",
                last_user_images.len()
            );
        }
    } else {
        engine.clear_images().await;
    }

    // Check if streaming is requested
    if request.stream {
        let gate_guard = chat_gate_guard;
        let max_tokens_cap = params.max_tokens;
        // Use streaming generation (NDJSON format, matching Ollama's API)
        match engine.generate_stream(&prompt, params.clone()).await {
            Ok(mut rx) => {
                let model_name_clone = model_name.clone();
                let start_time = std::time::Instant::now();
                let prompt_clone = prompt.clone();
                let engine_for_stats = engine.clone();
                let stream = async_stream::stream! {
                    let _gate_held = gate_guard;
                    let mut accumulated_text = String::new();
                    let mut chunk_count: usize = 0;
                    let mut errored = false;
                    let mut tool_scanner = if tools_active {
                        Some(crate::api::tool_calls::StreamToolScanner::new(tool_format))
                    } else {
                        None
                    };
                    let mut splitter = crate::api::thinking::ThinkSplit::new();
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
        let _gate_guard = chat_gate_guard;
        let energy = crate::energy_report::begin();

        let _cancel = engine.cancel_guard();
        match engine.generate(&prompt, params).await.map_err(|e| e.to_string()) {
            Ok(result) => {
                crate::energy_report::end(energy, "text", "[/api/chat]");
                let total_duration = start.elapsed().as_nanos() as u64;

                // Tool calling: lift markers out of the output into
                // structured tool_calls (Ollama surfaces these on
                // message.tool_calls). Plain answers pass through.
                let (chat_thinking, chat_body) = crate::api::thinking::split_thinking(&result.text);
                let (msg, tool_called) = if tools_active {
                    let parsed =
                        crate::api::tool_calls::parse_tool_calls(tool_format, &chat_body);
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
    let cached = {
        let engines = state.engines.read().await;
        engines
            .iter()
            .find(|e| e.model_id == model_name)
            .and_then(|e| e.chat_template.clone())
    };
    let chat_template = cached.or_else(|| {
        let tmpl = state.read_chat_template(model_name);
        if tmpl.is_some() {
            let state2 = state.clone();
            let name2 = model_name.to_string();
            let tmpl2 = tmpl.clone();
            tokio::spawn(async move {
                let mut engines = state2.engines.write().await;
                if let Some(entry) = engines.iter_mut().find(|e| e.model_id == name2) {
                    entry.chat_template = tmpl2;
                }
            });
        }
        tmpl
    });
    // Fallback when the manifest is missing the template layer (e.g. gemma4): synthesize one
    // from the model name. Without it, /api/generate would emit `[INST]`-formatted gibberish
    // that mirrors `ollama --raw=true`, while Ollama's own endpoint applies the template by
    // default and takes raw=true as the opt-out. We match that contract.
    let chat_template = chat_template.or_else(|| infer_template_from_model_name(model_name));
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
    validate_ollama_options(request.options.as_ref())?;
    // Normalize model ID
    let model_name = normalize_model_id(&request.model);

    // -- Cluster: serve here, or hand the whole request to a better-placed peer --------
    //
    // Read BEFORE any decision: the marker is what stops two nodes that each prefer the
    // other from passing a request back and forth until something times out, and a hang is
    // a far worse failure than an imperfect placement.
    let already_forwarded = headers.contains_key(crate::distributed::cluster::FORWARDED_HEADER);
    // Counted from the first instant, so concurrent arrivals see each other in `busy` when
    // they decide - the admission gate never sees this path, and a count taken any later
    // publishes an idle node under any load. Released explicitly on hand-over: a forwarded
    // request is the peer's work, not ours.
    let inflight_guard = crate::distributed::rate_meter::InFlight::enter();
    if let Some(cluster) = state.cluster_handle() {
        // An empty prompt is a load or unload instruction addressed to THIS node. Forwarding
        // it would make a peer load a model the caller asked this one to hold.
        if !request.prompt.is_empty() {
            let shape = crate::distributed::routing::RequestShape {
                model: model_name.clone(),
                // Bytes over four is a rough token count, and rough is enough: it is
                // compared against what a peer says it holds, and both sides describe the
                // SAME prompt, so an estimate biases every candidate identically.
                prompt_tokens: (request.prompt.len() / 4).max(1) as u32,
                // `options` is free-form JSON here, so the field is read by name rather
                // than through a struct. A missing or malformed value falls back to the
                // same default the generation path uses, so the estimate matches what will
                // actually run.
                max_tokens: request
                    .options
                    .as_ref()
                    .and_then(|o| o.get("num_predict"))
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(state.default_inference_config.max_tokens as u64) as u32,
            };
            let now = crate::distributed::cluster_runtime::now_ms(state.cluster_started());
            // This node has to sit in its own routing table, described exactly as peers
            // describe it. Without this it is compared as a default state - nothing resident,
            // no catalogue, no load - so it judges itself by facts that are not its own.
            cluster.publish_local(state.local_node_state().await);
            // Ask the peers, rather than guess for them: only the node owning a model can
            // tokenise for it and look in its own cache, so the question travels and no
            // hashing convention has to be shared. A peer that does not answer is priced as
            // holding none of the prompt - the safe direction.
            let cached = cluster
                .ask_peers_what_they_hold(&model_name, &request.prompt, now)
                .await;
            if let crate::distributed::cluster::Decision::Forward { peer, url, reason } =
                cluster.decide(&shape, now, already_forwarded, &cached)
            {
                info!("🔀 forwarding to {peer}: {reason}");
                // The request is re-serialised from the parsed form rather than relayed as
                // raw bytes. Both nodes run the same build, so a field one understands the
                // other does too; a mixed-version cluster would need the raw body kept.
                let body = serde_json::to_value(&request)
                    .map_err(|e| ApiError::Internal(format!("forward: {e}")))?;
                drop(inflight_guard);
                match crate::api::handlers::forward_to_peer(&url, &body).await {
                    Ok(relayed) => return Ok(relayed),
                    // The peer never got as far as answering. Nothing has been sent to the
                    // client yet, so the request can still be served here - failing it would
                    // hand the caller someone else's outage. Once bytes are flowing the choice
                    // is gone, which is why this is decided before the body starts.
                    Err(e) => {
                        warn!("cluster: hand-over to {peer} failed ({e}) - serving here instead");
                        cluster.note_handover_failed(&peer, now);
                    }
                }
            }
        }
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

    // Per-request kv_quant override. If the caller asked for a format that
    // differs from what's currently loaded, reload before proceeding.
    if let Some(want) = extract_kv_quant_override(request.options.as_ref()) {
        if let Err(e) = state.ensure_engine_kv_quant(&model_name, want).await {
            return Err(ApiError::Internal(format!(
                "kv_quant reload for '{model_name}': {e}"
            )));
        }
    }
    if let Some(want) = request
        .options
        .as_ref()
        .and_then(|o| o.get("num_ctx"))
        .and_then(serde_json::Value::as_u64)
    {
        if let Err(e) = state.ensure_engine_context(&model_name, want as usize).await {
            return Err(ApiError::Internal(format!(
                "num_ctx reload for '{model_name}': {e}"
            )));
        }
    }

    // Auto-load on first use (Ollama-compatible). If the model is on
    // disk but not in `engines`, ensure_loaded resolves the path, runs
    // any LRU eviction, and pushes a fresh entry. Only fail if the
    // load itself fails.
    let keep_alive_minutes = state.get_effective_keep_alive(request.keep_alive.as_deref());
    if let Err(e) = state.ensure_loaded(&model_name, keep_alive_minutes).await {
        return Err(ApiError::NotFound(format!(
            "Model '{model_name}' could not be auto-loaded: {e}"
        )));
    }
    let engine = match state.get_engine(&model_name).await {
        Ok(engine) => engine,
        Err(_) => {
            return Err(ApiError::NotFound(format!(
                "Model '{model_name}' not loaded after auto-load attempt"
            )));
        }
    };

    // Reset expiration timer on use (like Ollama)
    state.reset_expiration(&model_name).await;

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

    // Ollama parity: auto-apply the model's chat template for chat-tuned models
    // on /api/generate, unless the caller set `raw: true` or the prompt already
    // contains template markers from this model's family. Without this, chat-tuned
    // models like deepseek-r1 can emit EOS as their first token on instruction-
    // shaped prompts.
    let effective_prompt = if is_fim {
        if has_suffix {
            format!(
                "<|fim_prefix|>{}<|fim_suffix|>{}<|fim_middle|>",
                request.prompt,
                request.suffix.as_deref().unwrap_or("")
            )
        } else {
            request.prompt.clone()
        }
    } else {
        let raw_mode = request.raw.unwrap_or(false);
        if raw_mode || request.prompt.is_empty() {
            request.prompt.clone()
        } else {
            templated_prompt(
                &state,
                &model_name,
                &request.prompt,
                request.system.as_deref(),
                request.template.as_deref(),
            )
            .await
        }
    };
    // The same request field the chat endpoint honours: a caller asking for no reasoning
    // gets none here too.
    let mut effective_prompt = effective_prompt;
    apply_thinking_preference(&mut effective_prompt, request.thinking_preference());

    // Extract generation options
    let mut params = extract_generation_options(request.options.as_ref());
    params.top_logprobs = if request.logprobs == Some(true) {
        Some(request.top_logprobs.unwrap_or(0).min(20))
    } else {
        None
    };
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
        info!("Generate: stop sequences: {:?}", params.stop_sequences);
    }

    // Resolve request priority: FIM auto-promotes to Fim; explicit
    // options.priority overrides; default Interactive.
    let priority = request
        .options
        .as_ref()
        .and_then(|v| v.get("priority"))
        .and_then(|v| v.as_str())
        .and_then(crate::api::gate::Priority::parse)
        .unwrap_or(if is_fim {
            crate::api::gate::Priority::Fim
        } else {
            crate::api::gate::Priority::Interactive
        });

    // Acquire the priority gate BEFORE touching engine.set_images /
    // clear_images. Both grab the model_state mutex which is held by the
    // active spawn_blocking decode, so doing them before the gate would
    // serialize all pending requests on that mutex - making the gate
    // invisible to /api/inflight (queueing happens at the wrong layer).
    // CB-served models skip the single-request gate (they batch in their own
    // worker) so concurrent /api/generate requests reach the worker together.
    let gate_guard_owned = if engine.is_continuous().await {
        None
    } else {
        Some(acquire_gate(&state, priority, model_name.clone(), "/api/generate").await?)
    };

    // Process images for vision models (encode before generation).
    // Now serialized correctly: only the gate-holder runs this.
    if let Some(ref images) = request.images {
        if !images.is_empty() {
            if let Err(e) = engine.set_images(images).await {
                warn!("Failed to process images for vision model: {}", e);
            } else {
                debug!("Vision: {} image(s) encoded for generation", images.len());
            }
        } else {
            engine.clear_images().await;
        }
    } else {
        engine.clear_images().await;
    }

    // Draft dispatch: if the target has a draft attached and the
    // request is greedy (temperature=0) without grammar/FIM-special
    // overrides, prefer the spec-decode path. Falls back to regular
    // generate_stream otherwise.
    let draft_pair: Option<(Arc<LlmEngine>, usize)> =
        if !is_fim && params.grammar.is_none() && (params.temperature.unwrap_or(1.0) == 0.0) {
            let attached = {
                let engines = state.engines.read().await;
                engines
                    .iter()
                    .find(|e| e.model_id == model_name)
                    .and_then(|e| e.draft.as_ref())
                    .map(|d| (d.engine.clone(), d.k))
            };
            // A drafter named in the configuration takes the same path as an attached one,
            // with the attach endpoint's default k.
            match attached {
                Some(p) => Some(p),
                None if engine.has_config_drafter() => {
                    // Loaded on a blocking thread, the way the plain paths do it: the load
                    // future is not Send and cannot be awaited from this handler.
                    let e = engine.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        tokio::runtime::Handle::current().block_on(e.ensure_draft_loaded())
                    })
                    .await;
                    engine.config_drafter().await.map(|d| (d, 4))
                }
                None => None,
            }
        } else {
            None
        };

    // Check if streaming is requested
    if request.stream {
        let gate_guard = gate_guard_owned;
        // Spec-decode streaming path: used when a draft is attached.
        if let Some((draft_engine, k)) = draft_pair {
            let engine_for_stats = engine.clone();
            match engine
                .generate_stream_with_draft(
                    effective_prompt.clone(),
                    params.clone(),
                    draft_engine,
                    k,
                )
                .await
            {
                Ok(mut rx) => {
                    let model_name_clone = model_name.clone();
                    let start_time = std::time::Instant::now();
                    let prompt_clone = effective_prompt.clone();
                    let stream = async_stream::stream! {
                        let _gate_held = gate_guard;
                        let mut accumulated_text = String::new();
                        let mut chunk_count: usize = 0;
                        let mut splitter = crate::api::thinking::ThinkSplit::new();
                        let mut thought = false;
                        let mut thinking_ns: Option<u64> = None;
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
                                    error!("Spec-decode stream chunk error: {}", e);
                                    let err_chunk = serde_json::json!({"error": format!("{}", e)});
                                    let mut line = err_chunk.to_string();
                                    line.push('\n');
                                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                                    break;
                                }
                            }
                        }
                        // The draft loop publishes the same statistics as the plain stream.
                        // The fallbacks are what this message used to carry on its own:
                        // chunk_count is the exact emitted-token count (an estimate over the
                        // accumulated text returned ~1, chunks being words with no spaces).
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
                        let stats = engine_for_stats.take_last_stream_stats();
                        let (done_from_engine, context_tokens) = match stats.as_ref() {
                            Some(s) => (Some(s.finish_reason.ollama()), s.context_tokens.clone()),
                            None => (None, Vec::new()),
                        };
                        let done_reason = done_from_engine.unwrap_or("stop");
                        let (eval_count, eval_duration, prompt_eval_count, prompt_eval_duration, total_duration) =
                            match stats {
                                Some(s) => (
                                    s.eval_count,
                                    s.eval_duration_ns,
                                    s.prompt_eval_count,
                                    s.prompt_eval_duration_ns,
                                    s.total_duration_ns,
                                ),
                                None => (
                                    chunk_count as u64,
                                    0,
                                    estimate_token_count(&prompt_clone),
                                    0,
                                    start_time.elapsed().as_nanos() as u64,
                                ),
                            };
                        let final_chunk = serde_json::json!({
                            "model": model_name_clone,
                            "created_at": chrono::Utc::now().to_rfc3339(),
                            "response": "",
                            "done": true,
                            "thinking_duration": thinking_ns,
                            "done_reason": done_reason,
                            "context": context_tokens,
                            "total_duration": total_duration,
                            "prompt_eval_count": prompt_eval_count,
                            "prompt_eval_duration": prompt_eval_duration,
                            "eval_count": eval_count,
                            "eval_duration": eval_duration
                        });
                        let mut line = final_chunk.to_string();
                        line.push('\n');
                        yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                    };
                    return Ok(Response::builder()
                        .header("content-type", "application/x-ndjson")
                        .body(axum::body::Body::from_stream(stream))
                        .unwrap());
                }
                Err(e) => {
                    error!(
                        "Spec-decode setup error: {} (falling back to non-spec path)",
                        e
                    );
                    // fall through to regular generate_stream below
                }
            }
        }
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
                    let _gate_held = gate_guard; // released when stream ends
                    let mut splitter = crate::api::thinking::ThinkSplit::new();
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
        let _gate_guard = gate_guard_owned;
        let energy = crate::energy_report::begin();

        let _cancel = engine.cancel_guard();
        match engine.generate(&effective_prompt, params).await.map_err(|e| e.to_string()) {
            Ok(result) => {
                crate::energy_report::end(energy, "text", "[/api/generate]");
                let total_duration = start.elapsed().as_nanos() as u64;

                let (thinking, answer) = crate::api::thinking::split_thinking(&result.text);
                let mut response = OllamaGenerateResponse::new(model_name, answer);
                if !result.logprobs.is_empty() {
                    response.logprobs = Some(ollama_logprobs(&result.logprobs));
                }
                response.thinking_duration =
                    thinking_share_ns(&engine, thinking.as_deref(), result.eval_count, result.eval_duration).await;
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
