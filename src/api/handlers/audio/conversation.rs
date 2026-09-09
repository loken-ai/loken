//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

pub(super) fn route_intent(text: &str, has_image: bool) -> ConvRoute {
    if has_image {
        return ConvRoute::Vision;
    }
    let t = text.trim().to_lowercase();
    const IMG_VERBS: &[&str] = &[
        "draw", "generate", "create", "make", "paint", "render", "design", "sketch",
    ];
    const IMG_NOUNS: &[&str] = &[
        "image",
        "picture",
        "photo",
        "drawing",
        "painting",
        "illustration",
        "artwork",
        "logo",
        "poster",
    ];
    let starts_img = IMG_VERBS.iter().any(|v| t.starts_with(v));
    let has_img_noun = IMG_NOUNS.iter().any(|n| t.contains(n));
    if (starts_img && has_img_noun)
        || t.contains("image of")
        || t.contains("picture of")
        || t.contains("photo of")
    {
        return ConvRoute::ImageGen;
    }
    if t.starts_with("say ")
        || t.starts_with("speak ")
        || t.contains("read aloud")
        || t.contains("read this aloud")
        || t.contains("read it aloud")
        || t.contains("out loud")
        || t.contains("text to speech")
        || t.contains("text-to-speech")
    {
        return ConvRoute::Tts;
    }
    // Sound generation: an explicit audio noun, either with a media verb
    // ("make a drum loop") or in a "sound of ..." phrasing.
    const SND_NOUNS: &[&str] = &[
        "sound effect",
        "sound of",
        "sfx",
        "audio clip",
        "drum loop",
        "music loop",
        "audio loop",
        "seamless loop",
        "jingle",
        "soundscape",
        "ambience of",
    ];
    if SND_NOUNS.iter().any(|n| t.contains(n))
        && (IMG_VERBS.iter().any(|v| t.starts_with(v)) || t.contains("sound of"))
    {
        return ConvRoute::SoundGen;
    }
    ConvRoute::Chat
}

/// Strip a sound-gen command prefix, leaving the sound to render.
pub(super) fn sound_subject(text: &str) -> String {
    let t = text.trim();
    let lower = t.to_lowercase();
    for pat in [
        "a sound effect of ",
        "the sound of ",
        "a sound of ",
        "sound effect of ",
        "sound of ",
        "an audio clip of ",
        "the ambience of ",
    ] {
        if let Some(i) = lower.find(pat) {
            return t[i + pat.len()..].trim().to_string();
        }
    }
    for v in ["make ", "generate ", "create ", "render ", "design "] {
        if lower.starts_with(v) {
            return t[v.len()..].trim().to_string();
        }
    }
    t.to_string()
}

/// Strip an image-gen command prefix, leaving the subject to render.
pub(super) fn image_subject(text: &str) -> String {
    let t = text.trim();
    let lower = t.to_lowercase();
    for pat in [
        "an image of ",
        "a picture of ",
        "a photo of ",
        "a drawing of ",
        "a painting of ",
        "image of ",
        "picture of ",
        "photo of ",
    ] {
        if let Some(i) = lower.find(pat) {
            return t[i + pat.len()..].trim().to_string();
        }
    }
    for v in [
        "draw ",
        "generate ",
        "create ",
        "make ",
        "paint ",
        "render ",
        "design ",
        "sketch ",
    ] {
        if lower.starts_with(v) {
            return t[v.len()..].trim().to_string();
        }
    }
    t.to_string()
}

/// Strip a TTS command, leaving the text to speak (falls back to the last assistant
/// message when the user says e.g. "read that aloud").
pub(super) fn tts_text(text: &str, messages: &[crate::api::types::Message]) -> String {
    let t = text.trim();
    let lower = t.to_lowercase();
    for pat in ["say ", "speak "] {
        if lower.starts_with(pat) {
            let rest = t[pat.len()..].trim().trim_start_matches([':', '"']).trim();
            if !rest.is_empty() {
                return rest.to_string();
            }
        }
    }
    for pat in [
        "read aloud:",
        "read this aloud:",
        "text to speech:",
        "text-to-speech:",
    ] {
        if let Some(i) = lower.find(pat) {
            let rest = t[i + pat.len()..].trim();
            if !rest.is_empty() {
                return rest.to_string();
            }
        }
    }
    // "read that/the previous aloud" -> speak the last assistant message.
    if let Some(m) = messages.iter().rev().find(|m| m.role == "assistant") {
        if !m.content.trim().is_empty() {
            return m.content.clone();
        }
    }
    t.to_string()
}

/// 200 wrapper for the JSON media/conversation endpoints (/conversation, /voice, ...).
/// Success bodies are endpoint-specific; only errors share an envelope (see `conv_err`).
pub(super) fn conv_ok(v: serde_json::Value) -> axum::response::Response {
    (StatusCode::OK, Json(v)).into_response()
}
/// Error envelope for the JSON media/conversation endpoints (the `conv_ok`
/// family: /conversation, /voice, ...). Delegates
/// to `openai_error_body` so these endpoints emit the exact same
/// `{"error":{message,type,param,code}}` shape as the OpenAI-dialect
/// routes - one parser serves both. Use `anthropic_error_response` only on
/// the /v1/messages (Anthropic-protocol) path.
pub(crate) fn conv_err(code: StatusCode, msg: &str) -> axum::response::Response {
    (code, Json(openai_error_body(code, msg))).into_response()
}

// -- Stage B: classifier-LLM routing -------------------------------------
// The rule matcher (`route_intent`) confidently catches an attached image
// (->vision) and explicit command phrasings ("draw an image of...", "read
// aloud:..."). Everything else falls into the Chat bucket - which is exactly
// where mis-phrased media intents hide ("could you paint me a sunset",
// "narrate this poem"). For those, a tiny constrained-decode classifier on
// a small always-warm model disambiguates chat vs image vs speak. Grammar
// forces one of the three bare labels, so the model can't wander (nor
// "think") - one token, greedy. Any failure -> caller keeps the rule result.
pub(super) const ROUTER_DEFAULT_MODEL: &str = "qwen3:0.6b";
pub(super) const ROUTER_GRAMMAR: &str = "lark:start: \"chat\" | \"image\" | \"speak\" | \"sound\"";

/// LLM intent classification for the ambiguous (rule=Chat) bucket. Returns
/// `None` (-> caller falls back to the rule result) on any load/generate
/// error so smart-routing never breaks a turn.
pub(super) async fn classify_route_llm(
    state: &APIServer,
    router_model: &str,
    text: &str,
) -> Option<ConvRoute> {
    // Keep the router tiny + warm; if it isn't present this loads it once.
    let keep_alive = state.get_effective_keep_alive(None);
    if let Err(e) = state.ensure_loaded(router_model, keep_alive).await {
        warn!("smart-routing: router model '{router_model}' unavailable ({e}); using rules");
        return None;
    }
    let engine = state.get_engine(router_model).await.ok()?;
    state.reset_expiration(router_model).await;
    // Raw instruction (no chat template) - the grammar guarantees a valid
    // label regardless, and a raw prompt sidesteps think-token injection.
    let snippet: String = text.chars().take(400).collect();
    let prompt = format!(
        "Classify the user's message into exactly one label. Reply with only the label word.\n\n\
         Labels:\n\
         - image: the user explicitly asks to CREATE/GENERATE/DRAW a visual picture, image, photo, drawing, or artwork.\n\
         - speak: the user explicitly asks to have text spoken or read ALOUD as audio (text-to-speech, narration).\n\
         - sound: the user asks to CREATE a sound effect, ambience, noise, or a short musical loop/jingle (not spoken text).\n\
         - chat: everything else - questions, conversation, explanations, or writing/generating TEXT (poems, stories, code).\n\n\
         Examples:\n\
         Message: \"draw a cat wearing a hat\"\nLabel: image\n\
         Message: \"generate a photo of a beach at sunset\"\nLabel: image\n\
         Message: \"read this paragraph aloud\"\nLabel: speak\n\
         Message: \"say good morning to me\"\nLabel: speak\n\
         Message: \"please narrate the following text softly\"\nLabel: speak\n\
         Message: \"make the sound of rain on a tin roof\"\nLabel: sound\n\
         Message: \"I need a seamless techno loop\"\nLabel: sound\n\
         Message: \"what is the capital of France?\"\nLabel: chat\n\
         Message: \"write me a poem about the sea\"\nLabel: chat\n\
         Message: \"explain how photosynthesis works\"\nLabel: chat\n\n\
         Message: \"{snippet}\"\nLabel:"
    );
    let params = GenerationParams {
        max_tokens: Some(4),
        temperature: Some(0.0),
        grammar: Some(ROUTER_GRAMMAR.to_string()),
        logit_bias: None,
        top_logprobs: None,
        ..Default::default()
    };
    let out = engine.generate(&prompt, params).await.ok()?;
    let label = out.text.trim().to_lowercase();
    let route = if label.contains("image") {
        ConvRoute::ImageGen
    } else if label.contains("speak") {
        ConvRoute::Tts
    } else if label.contains("sound") {
        ConvRoute::SoundGen
    } else {
        ConvRoute::Chat
    };
    // The user's words never reach the log; the length is what a reader needs.
    let chars = snippet.chars().count();
    info!("🧭 smart-routing: {chars} chars -> {label}");
    Some(route)
}

// -- Stage B: conversation-aware residency -------------------------------
// A conversation's working set (the models it has used) should stay warm so
// LRU doesn't evict a model the SAME conversation will need again next turn.
// We ride the existing LRU: each turn we refresh `last_used` on every
// resident model in the conversation's set, so an active conversation
// out-ranks one-off requests from other callers. Self-expiring - a
// conversation not seen for CONV_TTL_MS is pruned, its models fall back to
// normal LRU. No new eviction policy, no leak, no struct plumbing.
pub(super) const CONV_TTL_MS: i64 = 30 * 60 * 1000;

pub(super) struct ConvWorkingSet {
    pub(super) last_seen_ms: i64,
    pub(super) models: std::collections::HashSet<String>,
}

pub(super) fn conv_registry(
) -> &'static std::sync::Mutex<std::collections::HashMap<String, ConvWorkingSet>> {
    static R: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, ConvWorkingSet>>,
    > = std::sync::OnceLock::new();
    R.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Record `model` in `conv_id`'s working set and return the current set
/// (for warm-touch). Prunes conversations idle past CONV_TTL_MS.
pub(super) fn conv_note_model(conv_id: &str, model: &str) -> Vec<String> {
    let now = now_millis();
    let mut reg = match conv_registry().lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    reg.retain(|_, ws| now - ws.last_seen_ms <= CONV_TTL_MS);
    let ws = reg
        .entry(conv_id.to_string())
        .or_insert_with(|| ConvWorkingSet {
            last_seen_ms: now,
            models: std::collections::HashSet::new(),
        });
    ws.last_seen_ms = now;
    ws.models.insert(model.to_string());
    ws.models.iter().cloned().collect()
}

/// The image route, as events rather than one late payload.
///
/// It forwards exactly what `/v1/images/generations` forwards - the loader's stage lines, then
/// the denoise progress - and ends with the same JSON body the non-streaming branch returns, so
/// a client reads one final event instead of parsing a second shape.
///
/// The load runs beside this generator so its stage lines can be forwarded while it happens,
/// and is aborted when the generator is dropped: a client that stops reading must not leave
/// gigabytes loading for nobody.
#[cfg(feature = "image")]
pub(super) async fn conversation_image_stream(
    state: APIServer,
    model: String,
    user_text: String,
    conversation_id: Option<String>,
    routed_by: &'static str,
) -> axum::response::Response {
    use crate::inference::engine::image_engine::ImageStreamEvent;
    use axum::response::sse::Event;

    let side = match crate::api::handlers::media::image_model_defaults(&model) {
        Ok(d) => d.size,
        Err(e) => return conv_err(StatusCode::BAD_REQUEST, &e),
    };
    let geom = crate::inference::place::runtime_demand::RequestGeometry::new(side, side);
    let subject = image_subject(&user_text);
    // One media job at a time, and this render listed for as long as it runs: the guard
    // moves into the stream, which is what renders after this call returns.
    let media_guard = state.media_lock_for(&model, "image").await;

    let stream = async_stream::stream! {
        let _media_guard = media_guard;
        let (load_tx, mut load_rx) =
            tokio::sync::mpsc::channel::<ImageEngineLoadingProgress>(32);
        let load_state = state.clone();
        let load_model = model.clone();
        let mut load_task = crate::api::handlers::media::AbortOnDrop(tokio::spawn(async move {
            crate::api::handlers::media::ensure_image_model_loaded_reporting(
                &load_state, &load_model, geom, Some(load_tx),
            )
            .await
        }));
        while let Some(ev) = load_rx.recv().await {
            if let ImageEngineLoadingProgress::Stage(msg) = ev {
                let chunk = serde_json::json!({ "route": "image_gen", "stage": msg });
                yield Ok::<Event, std::convert::Infallible>(
                    Event::default().data(chunk.to_string()),
                );
            }
        }
        if let Ok(Err(e)) = (&mut load_task.0).await {
            let chunk = serde_json::json!({ "route": "image_gen", "error": format!("image model load: {e}") });
            yield Ok(Event::default().data(chunk.to_string()));
            return;
        }

        let mut rx = match state
            .image_engine
            .generate_image_stream(&subject, ImageGenParams::default())
            .await
        {
            Ok(rx) => rx,
            Err(e) => {
                let chunk = serde_json::json!({ "route": "image_gen", "error": format!("image gen: {e:#}") });
                yield Ok(Event::default().data(chunk.to_string()));
                return;
            }
        };
        while let Some(event) = rx.recv().await {
            let chunk = match event {
                ImageStreamEvent::Progress { completed, total } => {
                    serde_json::json!({ "route": "image_gen", "completed": completed, "total": total })
                }
                ImageStreamEvent::Complete { image_base64 } => serde_json::json!({
                    "route": "image_gen",
                    "routed_by": routed_by,
                    "conversation_id": conversation_id,
                    "model": model,
                    "message": { "role": "assistant", "content": format!("[Generated an image: {subject}]") },
                    "image": image_base64,
                }),
                ImageStreamEvent::Error(e) => {
                    serde_json::json!({ "route": "image_gen", "error": e })
                }
            };
            yield Ok(Event::default().data(chunk.to_string()));
        }
    };

    axum::response::Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

/// POST /conversation - see the block comment above.
pub(crate) async fn conversation_handler(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    Json(req): Json<serde_json::Value>,
) -> axum::response::Response {
    let messages: Vec<crate::api::types::Message> = match serde_json::from_value(
        req.get("messages")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    ) {
        Ok(m) => m,
        Err(e) => return conv_err(StatusCode::BAD_REQUEST, &format!("invalid `messages`: {e}")),
    };
    if messages.is_empty() {
        return conv_err(StatusCode::BAD_REQUEST, "`messages` must not be empty");
    }
    let last_user = messages.iter().rev().find(|m| m.role == "user");
    let (user_text, images) = match last_user {
        Some(m) => (m.content.clone(), m.images.clone().unwrap_or_default()),
        None => (String::new(), Vec::new()),
    };
    let s = |k: &str| req.get(k).and_then(|v| v.as_str()).map(String::from);
    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    // Stage B: opt-out smart routing (default ON). The rule matcher decides
    // confident cases; the classifier only disambiguates the Chat bucket.
    let smart_routing = req
        .get("smart_routing")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    let conversation_id = s("conversation_id");
    // Honoured, not ignored. This route used to accept `stream: true` and answer with the
    // whole body at once - so a client that asked for events showed its "drawing..." state,
    // waited a minute and a half for a render it was never told about, and then received a
    // payload its event reader could not parse. Nothing failed; nothing arrived either.
    let want_stream = req.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let rule_route = route_intent(&user_text, !images.is_empty());
    let (route, routed_by) =
        if rule_route == ConvRoute::Chat && smart_routing && !user_text.trim().is_empty() {
            let router_model =
                s("router_model").unwrap_or_else(|| ROUTER_DEFAULT_MODEL.to_string());
            match classify_route_llm(&state, &router_model, &user_text).await {
                Some(r) => (r, "classifier"),
                None => (rule_route, "rules"),
            }
        } else {
            (rule_route, "rules")
        };

    // An image turn goes to the node that holds the family on a card that fits it, the
    // way a direct image request does; the peer classifies the turn again and renders.
    #[cfg(feature = "image")]
    if route == ConvRoute::ImageGen {
        let model = s("image_model").unwrap_or_else(|| "z-image".to_string());
        let hot = match crate::api::handlers::media::image_model_defaults(&model) {
            Ok(d) => crate::api::handlers::media::image_hot_bytes(
                &state,
                &model,
                crate::inference::place::runtime_demand::RequestGeometry::new(d.size, d.size),
            ),
            Err(_) => 0,
        };
        if let Some(relayed) = crate::api::handlers::route_media_to_holder(
            &state,
            &headers,
            &model,
            crate::api::handlers::media::image_served_here(&state, &model).await,
            state.card_holds(hot),
            hot,
            |peer| crate::api::handlers::media::serves_image_family(peer, &model),
            &crate::api::handlers::CONVERSATION,
            &req,
        )
        .await
        {
            return relayed;
        }
    }

    // A sound turn goes to a node whose catalogue holds a sound model when this one's
    // does not; a speech turn naming a model likewise. The peer classifies the turn again.
    {
        use crate::distributed::routing::can_serve;
        let holds = |n: &crate::distributed::membership::NodeState, model: &str| match route {
            ConvRoute::SoundGen => crate::api::handlers::catalogue::serves_sound(n, model),
            ConvRoute::Tts => crate::api::handlers::catalogue::serves_speech(n, model),
            _ => can_serve(n, model),
        };
        let wanted: Option<String> = match route {
            ConvRoute::SoundGen => Some(String::new()),
            ConvRoute::Tts => s("tts_model"),
            ConvRoute::Chat | ConvRoute::Vision => s("model"),
            _ => None,
        };
        if let Some(model) = wanted {
            let local = state.local_node_state().await;
            if let Some(relayed) = crate::api::handlers::route_media_to_holder(
                &state,
                &headers,
                if model.is_empty() {
                    "a sound model"
                } else {
                    &model
                },
                holds(&local, &model),
                true,
                0,
                |peer| holds(peer, &model),
                &crate::api::handlers::CONVERSATION,
                &req,
            )
            .await
            {
                return relayed;
            }
        }
    }

    // Warm-touch the conversation's whole working set so LRU keeps it
    // resident across turns. `cid` also echoes into the response.
    let cid = conversation_id.clone().unwrap_or_default();
    let note_and_touch = |state: &APIServer, model: String| {
        let cid = cid.clone();
        let state = state.clone();
        async move {
            if !cid.is_empty() {
                let set = conv_note_model(&cid, &model);
                state.touch_models(&set).await;
            }
        }
    };

    match route {
        ConvRoute::Chat | ConvRoute::Vision => {
            // The `model` hint is a chat-model PREFERENCE, not a hard
            // requirement: clients (the GUI's Auto mode among them) forward
            // whatever model was last selected, which may be any checkpoint
            // the server advertises (image, audio, ...). The LLM loader is
            // the single authority on what can chat - if the hint does not
            // load as an LLM, fall back to the default chat model instead of
            // failing the turn. No model-name knowledge here.
            let hint = s("model");
            let default_model = state.default_inference_config.model_id.clone();
            let keep_alive = state.get_effective_keep_alive(None);
            let mut model = hint.clone().unwrap_or_else(|| default_model.clone());
            if let Err(e) = state.ensure_loaded(&model, keep_alive).await {
                if hint.is_some() && model != default_model {
                    warn!("/conversation: hint '{model}' not loadable as an LLM ({e}); using default '{default_model}'");
                    model = default_model.clone();
                    if let Err(e) = state.ensure_loaded(&model, keep_alive).await {
                        return conv_err(
                            StatusCode::NOT_FOUND,
                            &format!("model '{model}' could not be auto-loaded: {e}"),
                        );
                    }
                } else {
                    return conv_err(
                        StatusCode::NOT_FOUND,
                        &format!("model '{model}' could not be auto-loaded: {e}"),
                    );
                }
            }
            let engine = match state.get_engine(&model).await {
                Ok(e) => e,
                Err(e) => return e.into_response(),
            };
            state.reset_expiration(&model).await;
            // Stage B: keep this conversation's LLM working set warm.
            note_and_touch(&state, model.clone()).await;
            if route == ConvRoute::Vision && !images.is_empty() {
                if let Err(e) = engine.set_images(&images).await {
                    return conv_err(
                        StatusCode::INTERNAL_SERVER_ERROR,
                        &format!("vision encode: {e}"),
                    );
                }
            }
            // Use the model's real chat template (read from the Ollama
            // manifest) so we don't leak generic [INST] scaffolding. Fall
            // back to a name-inferred template when the manifest omits one  -
            // same chain as /api/chat. Without this, moondream (whose
            // manifest template isn't always reachable) gets the [INST]
            // default and its base-phi2 LM emits gibberish ("!!! ") instead
            // of the " Question: ...\n\n Answer: " convention it was tuned on.
            let template = state
                .read_chat_template(&model)
                .or_else(|| infer_template_from_model_name(&model));
            let prompt = format_chat_prompt(&messages, template.as_deref());
            let params = GenerationParams {
                max_tokens: max_tokens.or(Some(512)),
                ..Default::default()
            };
            match engine.generate(&prompt, params).await {
                Ok(out) => conv_ok(serde_json::json!({
                    "route": if route == ConvRoute::Vision { "vision" } else { "chat" },
                    "routed_by": routed_by,
                    "conversation_id": conversation_id,
                    "model": model,
                    "message": { "role": "assistant", "content": out.text },
                })),
                Err(e) => conv_err(StatusCode::INTERNAL_SERVER_ERROR, &format!("generate: {e}")),
            }
        }
        #[cfg(feature = "image")]
        ConvRoute::ImageGen if want_stream => {
            let model = s("image_model").unwrap_or_else(|| "z-image".to_string());
            conversation_image_stream(state, model, user_text, conversation_id, routed_by).await
        }
        #[cfg(feature = "image")]
        ConvRoute::ImageGen => {
            let model = s("image_model").unwrap_or_else(|| "z-image".to_string());
            // This route names no size, so it renders at the family default -
            // which is what the placement must reserve scratch for.
            let side = match crate::api::handlers::media::image_model_defaults(&model) {
                Ok(d) => d.size,
                Err(e) => return conv_err(StatusCode::BAD_REQUEST, &e),
            };
            if let Err(e) = ensure_image_model_loaded(
                &state,
                &model,
                crate::inference::place::runtime_demand::RequestGeometry::new(side, side),
            )
            .await
            {
                return conv_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("image model load: {e}"),
                );
            }
            let subject = image_subject(&user_text);
            let _media_guard = state.media_lock_for(&model, "image").await;
            match state
                .image_engine
                .generate_image(&subject, ImageGenParams::default())
                .await
            {
                Ok(b64) => conv_ok(serde_json::json!({
                    "route": "image_gen",
                    "routed_by": routed_by,
                    "conversation_id": conversation_id,
                    "model": model,
                    // Bridge: fold a text note into the shared history so later text
                    // turns know an image was produced (KV can't cross models).
                    "message": { "role": "assistant", "content": format!("[Generated an image: {subject}]") },
                    "image": b64,
                })),
                Err(e) => conv_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("image gen: {e}"),
                ),
            }
        }
        // The router can still recognise the intent - it reads text - but nothing in this
        // build can answer it, and saying so is more use than a chat reply about drawing.
        #[cfg(not(feature = "image"))]
        ConvRoute::ImageGen => {
            conv_err(StatusCode::NOT_IMPLEMENTED, "this build renders no images")
        }
        ConvRoute::Tts => {
            let model = s("tts_model");
            if let Some(wanted) = model.as_deref() {
                use crate::api::handlers::catalogue;
                let demand = catalogue::listed_size(&state, |id| {
                    catalogue::speech_entry_matches(id, wanted)
                })
                .await
                .map(catalogue::whole_load_demand)
                .unwrap_or(0);
                crate::inference::place::vram_manager::ensure_gpu_headroom(
                    "tts",
                    demand,
                    catalogue::WHOLE_LOAD_RESERVE,
                )
                .await;
            }
            if let Err(e) = ensure_tts_model_loaded(&state, model.as_deref()).await {
                return conv_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("tts model load: {e}"),
                );
            }
            let text = tts_text(&user_text, &messages);
            let _job = state.media_note(model.as_deref().unwrap_or("speech"), "speech");
            match state
                .tts_engine
                .synthesize(text.clone(), TtsSynthParams::default())
                .await
            {
                Ok(res) => {
                    let wav = pcm_to_wav_base64(&res.pcm, res.sample_rate);
                    conv_ok(serde_json::json!({
                        "route": "tts",
                        "routed_by": routed_by,
                        "conversation_id": conversation_id,
                        "message": { "role": "assistant", "content": format!("[Spoke aloud: {text}]") },
                        "audio": wav,
                    }))
                }
                Err(e) => conv_err(StatusCode::INTERNAL_SERVER_ERROR, &format!("tts: {e}")),
            }
        }
        ConvRoute::SoundGen => {
            // Stable Audio (stereo 44.1 kHz) with an EzAudio fallback when its
            // weights are absent. Short clip + moderate steps keeps the chat
            // turn responsive (~1 s warm, a few seconds cold).
            let subject = sound_subject(&user_text);
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1);
            let subj = subject.clone();
            let media_guard = state.media_lock_for("stable-audio", "sound").await;
            let place_fn = media_guard.reporter().placement_fn();
            // A dropped request must stop the sampler: it runs in spawn_blocking,
            // so the step hook is the only place it can observe the cancellation.
            let cancel = crate::inference::serve::cancel::CancelToken::new();
            let guard = crate::inference::serve::cancel::CancelGuard::new(cancel.clone());
            let rendered = tokio::task::spawn_blocking(move || {
                let _placed = crate::inference::serve::progress::placement::publish(place_fn);
                match crate::inference::model::stable_audio::render(
                    &subj,
                    "",
                    10.0,
                    25,
                    7.0,
                    seed,
                    |_, _, _| cancel.bail(),
                ) {
                    Ok((pcm, rate)) => Ok((pcm, rate, 2u16, "stable-audio")),
                    Err(e) if e.to_string().contains("weights not found") => {
                        crate::inference::model::ezaudio::pipeline::render(
                            &subj, 10.0, 50, 3.0, seed, 1.0, "",
                        )
                        .map(|(pcm, _)| (pcm, 24_000u32, 1u16, "ezaudio"))
                        .map_err(|e| e.to_string())
                    }
                    Err(e) => Err(e.to_string()),
                }
            })
            .await;
            guard.disarm();
            match rendered {
                Ok(Ok((pcm, rate, chans, model))) => conv_ok(serde_json::json!({
                    "route": "sound_gen",
                    "routed_by": routed_by,
                    "conversation_id": conversation_id,
                    "model": model,
                    "message": { "role": "assistant", "content": format!("[Generated a sound: {subject}]") },
                    "audio": pcm_to_wav_base64_ch(&pcm, rate, chans),
                })),
                Ok(Err(e)) => conv_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("sound gen: {e}"),
                ),
                Err(e) => conv_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("sound gen task: {e}"),
                ),
            }
        }
    }
}

/// POST /voice - fully-local voice assistant: speech in -> ASR -> LLM -> speech out.
/// Chains three CO-RESIDENT engines that live in separate slots so none evicts
/// another: whisper (ASR) + the chat LLM (registry) + pocket-tts. Reuses the
/// /conversation prompt-template chain so the LLM turn is formatted correctly
/// per model.
///
/// Request (JSON):
///   audio         base64 WAV/MP3/FLAC/... of the user's speech        (required)
///   model         chat LLM id            (default: server default)
///   asr_model     whisper id             (default: "whisper-small")
///   voice         bundled pocket-tts voice ("alba" -> pocket-tts-alba); default "default"
///   messages      prior turns for context                          (optional)
///   max_tokens    reply cap              (default 200)
/// Response: { transcript, reply, audio (base64 WAV), sample_rate, model, voice }
pub(crate) async fn voice_handler(
    State(state): State<APIServer>,
    Json(req): Json<serde_json::Value>,
) -> axum::response::Response {
    let s = |k: &str| req.get(k).and_then(|v| v.as_str()).map(String::from);

    // -- 1. Speech in -> ASR (whisper) -------------------------------------
    let bytes = match decode_b64_field(&req, "audio") {
        Ok(b) => b,
        Err(e) => return conv_err(StatusCode::BAD_REQUEST, &e),
    };
    let (samples, sr) = match decode_audio_blocking(bytes, "audio").await {
        Ok(v) => v,
        Err((code, e)) => return conv_err(code, &e),
    };
    let asr_model = s("asr_model").unwrap_or_else(|| "whisper-small".to_string());
    if let Err(e) = ensure_whisper_model_loaded(&state, Some(&asr_model)).await {
        return conv_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            &format!("asr model load: {e}"),
        );
    }
    let asr_params = AudioTranscribeParams {
        language: None,
        temperature: 0.0,
        task: WhisperTask::Transcribe,
        timestamps: false,
        collect_metrics: false,
        initial_prompt: None,
    };
    let asr_job = state.media_note(&asr_model, "transcription");
    let (transcript, detected_lang) =
        match state.audio_engine.transcribe(samples, sr, asr_params).await {
            Ok(r) => (r.text.trim().to_string(), r.language),
            Err(e) => {
                return conv_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("transcribe: {e}"),
                )
            }
        };

    // -- 2. LLM reply (transcript folded into the shared history) ---------
    drop(asr_job);
    let model = s("model").unwrap_or_else(|| state.default_inference_config.model_id.clone());
    let max_tokens = req
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .map(|n| n as usize);
    let mut messages: Vec<crate::api::types::Message> = serde_json::from_value(
        req.get("messages")
            .cloned()
            .unwrap_or(serde_json::Value::Null),
    )
    .unwrap_or_default();
    messages.push(crate::api::types::Message::new(
        "user".to_string(),
        transcript.clone(),
    ));
    let keep_alive = state.get_effective_keep_alive(None);
    if let Err(e) = state.ensure_loaded(&model, keep_alive).await {
        return conv_err(
            StatusCode::NOT_FOUND,
            &format!("model '{model}' could not be auto-loaded: {e}"),
        );
    }
    let engine = match state.get_engine(&model).await {
        Ok(e) => e,
        Err(e) => return e.into_response(),
    };
    state.reset_expiration(&model).await;
    let template = state
        .read_chat_template(&model)
        .or_else(|| infer_template_from_model_name(&model));
    let prompt = format_chat_prompt(&messages, template.as_deref());
    let params = GenerationParams {
        max_tokens: max_tokens.or(Some(200)),
        ..Default::default()
    };
    let reply = match engine.generate(&prompt, params).await {
        Ok(out) => out.text,
        Err(e) => return conv_err(StatusCode::INTERNAL_SERVER_ERROR, &format!("generate: {e}")),
    };
    // Speak the answer, not the chain-of-thought: strip a leading
    // <think>...</think> block (thinking models like qwen3) so the TTS voices
    // the actual reply. Falls back to the raw text if there's no closing tag.
    let spoken = match reply.split_once("</think>") {
        Some((_, after)) if reply.trim_start().starts_with("<think>") => after.trim(),
        _ => reply.trim(),
    };

    // -- 3. Reply -> speech ------------------------------------------------
    // Selection priority:
    //   1. a registered speech post-processor that recognises this request - it names the
    //      base voice to synthesize with, and transforms the result afterwards;
    //   2. explicit `voice` - a full backend id ("piper/fr_FR-tom-medium",
    //      "pocket-tts-alba", "parler-...") passed straight through; a bare name
    //      -> pocket-tts;
    //   3. else a default voice for the ASR-DETECTED language - multilingual
    //      output via per-language piper voices, English -> pocket-tts.
    let voice = s("voice");
    let voice_used: String;
    // Whether the reply is handed to a post-processor once it has been spoken. Absent on a
    // server with nothing registered, in which case the request is served by voice alone.
    let post = crate::api::assist::speech_postprocessor();
    let base_voice = match post.as_ref() {
        Some(p) => match p.base_voice(&req, &detected_lang) {
            Ok(v) => v,
            Err(e) => return conv_err(StatusCode::BAD_REQUEST, &e),
        },
        None => None,
    };
    // Set only on the post-processed path, so the decision is made ONCE here rather than
    // re-derived after synthesis by comparing voice names - which would post-process, or
    // fail to, on a string match that has nothing to do with what was asked.
    let postprocess = base_voice.is_some();
    if let Some(base) = base_voice {
        let _job = state.media_note(&base, "speech");
        if let Err(e) = ensure_tts_model_loaded(&state, Some(&base)).await {
            return conv_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("base tts load ({base}): {e}"),
            );
        }
        voice_used = post.as_ref().map(|p| p.label()).unwrap_or(base);
    } else {
        let tts_model = match voice.as_deref() {
            Some(v)
                if v.contains('/')
                    || v.contains("pocket")
                    || v.contains("piper")
                    || v.contains("parler")
                    || v.contains("kyutai") =>
            {
                v.to_string()
            }
            Some(v) => format!("pocket-tts-{v}"),
            None => default_voice_for_language(&detected_lang),
        };
        voice_used = tts_model.clone();
        if let Err(e) = ensure_tts_model_loaded(&state, Some(&tts_model)).await {
            return conv_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("tts model load ({tts_model}): {e}"),
            );
        }
    }
    if spoken.is_empty() {
        return conv_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "empty reply - nothing to speak",
        );
    }
    let res = match state
        .tts_engine
        .synthesize(spoken.to_string(), TtsSynthParams::default())
        .await
    {
        Ok(r) => r,
        Err(e) => return conv_err(StatusCode::INTERNAL_SERVER_ERROR, &format!("tts: {e}")),
    };
    // Hand the spoken reply to the post-processor, when one recognised this request.
    let (pcm, sr) = match post.as_ref().filter(|_| postprocess) {
        Some(p) => match p.apply(&req, res.pcm, res.sample_rate).await {
            Ok(v) => v,
            Err(e) => return conv_err(StatusCode::INTERNAL_SERVER_ERROR, &e),
        },
        None => (res.pcm, res.sample_rate),
    };
    conv_ok(serde_json::json!({
        "transcript": transcript,
        "reply": reply,
        "spoken": spoken,
        "audio": pcm_to_wav_base64(&pcm, sr),
        "sample_rate": sr,
        "model": model,
        "language": detected_lang,
        "voice": voice_used,
    }))
}
