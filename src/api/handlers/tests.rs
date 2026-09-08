/// The gate must refuse by default and accept only what was configured.
///
/// Written because the whole feature is one boolean away from being decorative:
/// `configure_auth` is called after construction, and an implementation that
/// defaulted to "accept" would pass every functional test in this file while
/// leaving the API open.
#[test]
fn the_auth_gate_accepts_only_configured_keys() {
    let keys = vec!["alpha-secret".to_string(), "beta-secret".to_string()];
    let gate = super::AuthGate::new(true, &keys, &[], 0, 10);
    assert!(gate.accepts("alpha-secret"));
    assert!(gate.accepts("beta-secret"));
    // Surrounding whitespace is trimmed on both sides, so a config with a stray
    // newline still works and a header with one is not a different key.
    assert!(gate.accepts("  alpha-secret\n"));
    assert!(!gate.accepts("alpha-secre"), "a prefix must not pass");
    assert!(!gate.accepts("alpha-secrets"), "an extension must not pass");
    assert!(!gate.accepts(""), "empty must not pass");
    assert!(!gate.accepts("ALPHA-SECRET"), "keys are case sensitive");
}

/// require_auth with NO keys must refuse everything. A typo in the config should
/// lock the door, never open it.
#[test]
fn requiring_auth_without_keys_refuses_everything() {
    let gate = super::AuthGate::new(true, &[], &[], 0, 10);
    assert!(!gate.accepts(""));
    assert!(!gate.accepts("anything"));
}

/// CORS stays wide open while unauthenticated - where it protects nothing - and
/// narrows to the named origins once the API carries credentials.
#[test]
fn cors_narrows_once_the_api_is_credentialed() {
    let open = super::AuthGate::new(false, &[], &[], 0, 10);
    // AllowOrigin has no public accessor, so this asserts the branch we can see:
    // the gate's own view of whether it is credentialed.
    assert!(
        !open.required,
        "unauthenticated gate must not be credentialed"
    );
    let closed = super::AuthGate::new(
        true,
        &["k".to_string()],
        &["https://example.test".to_string()],
        0,
        10,
    );
    assert!(closed.required);
    assert_eq!(
        closed.allowed_origins,
        vec!["https://example.test".to_string()]
    );
}
use super::*;

#[test]
fn available_endpoints_list_includes_critical_routes() {
    // Pins the discoverability hint surfaced in 404 envelopes to
    // the actual route set. A new endpoint registered in
    // create_router but forgotten in the hint will be caught only
    // if it's listed here - so this test acts as a checklist of
    // user-facing routes the GUI / SDK depend on.
    let list = available_endpoints_list();
    let entries: Vec<&str> = list
        .as_array()
        .expect("hint is a JSON array")
        .iter()
        .map(|v| v.as_str().expect("each entry is a string"))
        .collect();

    // Endpoints the GUI / SDK / docs depend on. If any of these
    // disappear from the hint, the user-facing surface lost a
    // discoverability hook - add the route back or update this
    // test to acknowledge its removal.
    for required in [
        "GET /api/tags",
        "POST /api/chat",
        "POST /api/generate",
        "GET /api/models",
        "GET /api/models/loaded",
        "GET /api/inflight",
        "GET /api/layer_perf",
        "GET /api/distributed/devices",
        "POST /v1/chat/completions",
        "POST /v1/completions",
        "POST /v1/embeddings",
        "POST /v1/audio/transcriptions",
        "POST /v1/audio/speech",
        "POST /v1/images/generations",
        "GET /v1/models",
        "GET /health",
        "GET /api/version",
    ] {
        assert!(
            entries.contains(&required),
            "available_endpoints hint missing required route: {required}",
        );
    }
}

#[test]
fn validate_user_id_caps_at_256_chars() {
    // None / empty are fine - represents "no session reuse".
    assert!(validate_user_id(None).is_ok());
    assert!(validate_user_id(Some("")).is_ok());
    // Realistic OpenAI user-ids pass.
    assert!(validate_user_id(Some("user-abc-123")).is_ok());
    assert!(validate_user_id(Some(&"a".repeat(256))).is_ok());
    // Past cap rejects.
    assert!(validate_user_id(Some(&"a".repeat(257))).is_err());
    assert!(validate_user_id(Some(&"x".repeat(10_000))).is_err());
    // Control characters rejected (NUL, newline, tab, escape) so they
    // can't break structured logs or wedge terminals via log display.
    assert!(validate_user_id(Some("user\0null")).is_err());
    assert!(validate_user_id(Some("user\nnewline")).is_err());
    assert!(validate_user_id(Some("user\ttab")).is_err());
    assert!(validate_user_id(Some("user\x1bescape")).is_err());
}

#[test]
fn validate_model_id_accepts_normal_names() {
    assert!(validate_model_id("gemma4:26b").is_ok());
    assert!(validate_model_id("Qwen/Qwen2-7B-Instruct").is_ok());
    assert!(validate_model_id("llama3:latest").is_ok());
    assert!(validate_model_id("openai/whisper-small").is_ok());
    assert!(validate_model_id("parler-tts/parler-tts-mini-v1").is_ok());
}

#[test]
fn validate_model_id_rejects_traversal() {
    assert!(validate_model_id("../etc/passwd").is_err());
    assert!(validate_model_id("foo/../bar").is_err());
    assert!(validate_model_id("..").is_err());
    assert!(validate_model_id("/absolute").is_err());
    assert!(validate_model_id(".hidden").is_err());
    assert!(validate_model_id("foo\\bar").is_err());
    assert!(validate_model_id("foo\x00bar").is_err());
    assert!(validate_model_id("foo\nbar").is_err());
    assert!(validate_model_id("").is_err());
    // Whitespace-only (post fe7d952 + this commit).
    assert!(validate_model_id("   ").is_err());
    assert!(validate_model_id("\t").is_err());
    // Embedded space - model names shouldn't contain spaces.
    assert!(validate_model_id("foo bar").is_err());
    // 257 chars > 256 cap.
    assert!(validate_model_id(&"a".repeat(257)).is_err());
}

#[test]
fn clamp_finite_replaces_nan_and_infinity_with_default() {
    // Finite values pass through clamp.
    assert_eq!(clamp_finite_f64(5.0, 0.0, 30.0, 4.0), 5.0);
    assert_eq!(clamp_finite_f64(-1.0, 0.0, 30.0, 4.0), 0.0);
    assert_eq!(clamp_finite_f64(99.0, 0.0, 30.0, 4.0), 30.0);
    // NaN / ±Infinity collapse to the (clamped) default.
    assert_eq!(clamp_finite_f64(f64::NAN, 0.0, 30.0, 4.0), 4.0);
    assert_eq!(clamp_finite_f64(f64::INFINITY, 0.0, 30.0, 4.0), 4.0);
    assert_eq!(clamp_finite_f64(f64::NEG_INFINITY, 0.0, 30.0, 4.0), 4.0);
    // Default itself out-of-range still gets clamped (defensive).
    assert_eq!(clamp_finite_f64(f64::NAN, 0.0, 30.0, 99.0), 30.0);

    // f32 sibling.
    assert_eq!(clamp_finite_f32(0.5_f32, 0.0, 1.0, 0.0), 0.5);
    assert_eq!(clamp_finite_f32(f32::NAN, 0.0, 1.0, 0.0), 0.0);
    assert_eq!(clamp_finite_f32(f32::INFINITY, 0.0, 1.0, 0.0), 0.0);
}

// -- clamp_finite_f64 (image-gen guidance/strength sanitiser) ----
// The chat path threads caller-supplied JSON floats straight into
// ImageGenParams.{guidance,strength}. NaN strength produces a
// black image (NaN in the noise mix); Inf guidance propagates
// NaN through Flux's embedded-guidance tensor. Pin both pathological
// inputs fall back to the explicit default, and finite values
// outside [lo,hi] get clamped (not rejected) so callers don't
// see 400s for harmless out-of-range values.

#[test]
fn clamp_finite_f64_passes_through_finite_in_range() {
    assert_eq!(clamp_finite_f64(0.5, 0.0, 1.0, 0.75), 0.5);
    assert_eq!(clamp_finite_f64(10.0, 0.0, 30.0, 4.0), 10.0);
}

#[test]
fn clamp_finite_f64_clamps_finite_out_of_range() {
    // Caller-passed 2.0 with strength range [0,1] gets pulled back
    // to 1.0 - not rejected, so a sloppy client doesn't get a 400.
    assert_eq!(clamp_finite_f64(2.0, 0.0, 1.0, 0.75), 1.0);
    // Negative guidance pulled up to 0.
    assert_eq!(clamp_finite_f64(-5.0, 0.0, 30.0, 4.0), 0.0);
}

#[test]
fn clamp_finite_f64_substitutes_default_for_nan_and_inf() {
    // NaN must fall back to the default - propagating NaN through
    // the noise mix would yield a black image with no diagnostic.
    assert_eq!(clamp_finite_f64(f64::NAN, 0.0, 1.0, 0.75), 0.75);
    // +Inf and -Inf would either saturate to the cap or break
    // tensor ops downstream; both go to default.
    assert_eq!(clamp_finite_f64(f64::INFINITY, 0.0, 30.0, 4.0), 4.0);
    assert_eq!(clamp_finite_f64(f64::NEG_INFINITY, 0.0, 30.0, 4.0), 4.0);
}

/// A bad nested Message field used to surface the generic "request
/// validation failed" fallback. Post-68d2e30, the humanizer walks the
/// error tree and surfaces the indexed path + reason. (Uses an empty
/// `role` - `content` is intentionally no longer length-validated so
/// assistant tool-call turns can carry empty content; see the
/// `deserialize_nullable_string` note on Message::content.)
#[test]
fn humanize_validation_walks_nested_messages() {
    use crate::api::types::{ChatCompletionRequest, Message};
    use validator::Validate;

    let mut req = ChatCompletionRequest::new(
        "gpt-4o".to_string(),
        vec![Message::new(String::new(), "hi".to_string())],
    );
    // Trigger top-level validator error too so both paths run.
    req.top_p = Some(2.0);
    let err = req.validate().expect_err("should fail");
    let msg = humanize_validation_error(&err);

    // Nested message path is walked and surfaced.
    assert!(
        msg.contains("messages[0].role"),
        "nested message path missing: {msg}"
    );
    // Top-level: "top_p must be in [0, 1]; got 2"
    assert!(
        msg.contains("top_p") && msg.contains("must be in"),
        "top_level message missing: {msg}"
    );
}

#[test]
fn estimate_token_count_uses_word_count_times_1_3() {
    // Empty string -> 0; ceil(0 * 1.3) = 0.
    assert_eq!(estimate_token_count(""), 0);
    // 1 word -> ceil(1 * 1.3) = 2.
    assert_eq!(estimate_token_count("hello"), 2);
    // 10 words -> ceil(10 * 1.3) = 13.
    let ten = "the quick brown fox jumps over the lazy dog now";
    assert_eq!(estimate_token_count(ten), 13);
    // Whitespace folding: multi-space + leading/trailing trims to 3 words.
    assert_eq!(estimate_token_count("  a  b   c  "), 4);
}

#[test]
fn non_chat_pipeline_component_routes_helpfully() {
    // Each rejection must explain WHAT the model is and WHERE to use
    // it (or that it can't be used standalone). A previous user
    // attempt to chat with CLIP panicked the text-engine worker
    // thread because the loader bailed with blocking_lock on a
    // tokio runtime - see the loader in `inference/engine/llm_engine/`.
    let clip = non_chat_pipeline_component("openai/clip-vit-large-patch14").unwrap();
    assert!(clip.contains("CLIP"), "got: {clip}");
    assert!(
        clip.contains("Flux"),
        "should suggest using Flux instead: {clip}"
    );

    let t5 = non_chat_pipeline_component("google/t5-v1_1-xxl").unwrap();
    assert!(t5.contains("T5"), "got: {t5}");

    // Whisper is NOT rejected here on the Ollama /api/chat path  -
    // it routes to handle_chat_asr (which decodes attached audio
    // from images[] and returns the transcript). The OpenAI-shape
    // chat_completion / text_completions handlers still 400 on
    // whisper via their is_asr_model branch since the OpenAI
    // request envelope has no input-bytes field.
    assert!(
        non_chat_pipeline_component("openai/whisper-small").is_none(),
        "whisper must reach handle_chat_asr on /api/chat, not the early-reject path"
    );

    // Parler / other TTS models are NOT rejected here - they're
    // routed to handle_chat_tts on /api/chat and /api/generate so
    // the GUI's "type text, get audio" UX works through the chat
    // tab. Whisper still rejects (ASR is post-binary-audio, not
    // text-driven generation).
    assert!(
        non_chat_pipeline_component("parler-tts/parler-tts-mini-v1").is_none(),
        "TTS models must reach handle_chat_tts, not the early-reject path"
    );
    assert!(
        non_chat_pipeline_component("tts-1-hd").is_none(),
        "OpenAI-style tts-1 aliases also must reach handle_chat_tts"
    );

    let tk = non_chat_pipeline_component("lmz/mt5-tokenizers").unwrap();
    assert!(tk.contains("tokenizer"), "got: {tk}");

    // Generation models pass through cleanly.
    assert!(non_chat_pipeline_component("Tongyi-MAI/Z-Image-Turbo").is_none());
    assert!(non_chat_pipeline_component("lmz/candle-flux").is_none());
    assert!(non_chat_pipeline_component("qwen3-coder:latest").is_none());
    assert!(non_chat_pipeline_component("gemma4:26b").is_none());
    // Edge: empty input -> not rejected (defer to normal validation).
    assert!(non_chat_pipeline_component("").is_none());
}

// -- humanize_validation_error --
//
// validate_request() runs validator::Validate on every typed
// /api/* request body. The default ValidationErrors Debug form
// is unreadable JSON; humanize_validation_error turns it into
// a single-line, dotted-path, plain-English message the client
// can show the user. Drift in the message shape would surface
// to users in the 422 response body - pin the canonical forms.

use validator::Validate;

// Min-only ("non-empty" hint) - exercises the specialised
// "must not be empty" branch in humanize_one.
#[derive(Validate)]
struct NonEmpty {
    #[validate(length(min = 1))]
    name: String,
}

// Max-only - exercises the "length must be <= N" branch.
#[derive(Validate)]
struct MaxLen {
    #[validate(length(max = 4))]
    name: String,
}

// Both bounds - exercises the "length must be in [lo, hi]" branch.
#[derive(Validate)]
struct LenBoth {
    #[validate(length(min = 1, max = 4))]
    name: String,
}

#[derive(Validate)]
struct RangeStruct {
    #[validate(range(min = 0, max = 100))]
    temp: i32,
}

#[derive(Validate)]
struct NestedListStruct {
    #[validate(nested)]
    items: Vec<NonEmpty>,
}

#[test]
fn humanize_validation_error_length_empty_field_specialised_message() {
    // length min=1 + actual len 0 (empty string) -> specialised
    // "must not be empty" message. Pin so a regression doesn't
    // surface the generic "length must be >= 1" form, which is
    // less helpful to humans.
    let bad = NonEmpty {
        name: String::new(),
    };
    let err = bad.validate().unwrap_err();
    let msg = humanize_validation_error(&err);
    assert!(msg.contains("name must not be empty"), "got: {msg}");
}

#[test]
fn humanize_validation_error_length_too_long_includes_actual_and_bounds() {
    // length max=4, actual 7 -> "length must be <= 4; got 7"
    // with the actual value so the caller knows by how much.
    let bad = MaxLen {
        name: "abcdefg".to_string(),
    };
    let err = bad.validate().unwrap_err();
    let msg = humanize_validation_error(&err);
    assert!(msg.contains("name length"));
    assert!(msg.contains("<= 4"));
    assert!(msg.contains("got 7"), "got: {msg}");
}

#[test]
fn humanize_validation_error_length_both_bounds_includes_bracket_form() {
    // min + max both set, actual out of range -> "length must be
    // in [min, max]; got N". Pin so the bracket-form arm doesn't
    // get accidentally swapped with the single-bound form.
    let bad = LenBoth {
        name: "abcdefg".to_string(),
    };
    let err = bad.validate().unwrap_err();
    let msg = humanize_validation_error(&err);
    assert!(msg.contains("name length must be in [1, 4]"), "got: {msg}");
    assert!(msg.contains("got 7"), "got: {msg}");
}

#[test]
fn humanize_validation_error_range_includes_bounds_and_actual() {
    // range min=0, max=100, actual -5 -> "must be in [0, 100]; got -5"
    let bad = RangeStruct { temp: -5 };
    let err = bad.validate().unwrap_err();
    let msg = humanize_validation_error(&err);
    assert!(msg.contains("temp must be in [0, 100]"), "got: {msg}");
    assert!(msg.contains("got -5"), "got: {msg}");
}

#[test]
fn humanize_validation_error_walks_nested_list_with_dotted_index_path() {
    // Nested-list errors get the dotted path with the index, so
    // a client knows WHICH item in the array failed. Pin the
    // shape - e.g. "items[1].name must not be empty".
    let bad = NestedListStruct {
        items: vec![
            NonEmpty {
                name: "ok".to_string(),
            },
            NonEmpty {
                name: String::new(),
            }, // index 1 fails
        ],
    };
    let err = bad.validate().unwrap_err();
    let msg = humanize_validation_error(&err);
    assert!(
        msg.contains("items[1].name"),
        "must dot-path into the failing list item; got: {msg}"
    );
    assert!(msg.contains("must not be empty"), "got: {msg}");
}

#[test]
fn humanize_validation_error_empty_errors_falls_back_to_generic() {
    // If somehow the ValidationErrors set is empty (shouldn't
    // happen, but the helper is defensive), use a generic
    // fallback string rather than emitting an empty response.
    let empty = validator::ValidationErrors::new();
    let msg = humanize_validation_error(&empty);
    assert!(!msg.is_empty(), "must never emit empty message");
    assert!(
        msg.contains("validation"),
        "fallback should mention validation; got: {msg}"
    );
}

// -- model_capabilities -----------------------------------------
// /api/show used to hardcode ["completion","embedding"] for every
// model, even for image-gen / TTS / ASR checkpoints that can't
// fulfil either. Pin the modality-aware split so a future
// additional classifier (video-gen, embedding-only model, ...)
// can't silently regress to the old hardcoded list.

#[test]
fn is_vision_model_by_name_matches_known_families_not_plain_text() {
    // Known vision-LLM families (the direct-path HF fallback for
    // model_has_vision when there is no Ollama projector layer).
    for name in [
        "moondream:1.8b",
        "llava:13b",
        "bakllava",
        "pixtral-12b",
        "minicpm-v:8b",
        "qwen2.5-vl:7b",
        "qwen3-vl:8b",
        "internvl-8b",
        "llama3.2-vision:11b",
        "granite3.2-vision",
        "smolvlm",
    ] {
        assert!(is_vision_model_by_name(name), "{name} should be vision");
    }
    // Plain text / non-vision models must NOT be tagged (a false
    // positive would offer image upload for a text-only model).
    for name in [
        "qwen3-coder:latest",
        "deepseek-r1:32b",
        "llama3:8b",
        "mistral-nemo",
        "gemma4:9b",
        "nomic-embed-text",
        "",
    ] {
        assert!(
            !is_vision_model_by_name(name),
            "{name} should NOT be vision"
        );
    }
}

#[test]
fn a_text_model_is_offered_for_chat() {
    // What `/api/tags` advertises drives what an SDK will route at a model. A text LLM that
    // does not say `chat` disappears from every client's model picker.
    for name in [
        "qwen3-coder:latest",
        "deepseek-r1:32b",
        "llama3:8b",
        "moondream:1.8b", // vision, but still text-capable
        "",               // unknown -> defaults to text
    ] {
        let caps = model_capabilities(name);
        assert!(
            caps.contains(&"completion".to_string()),
            "{name} should be offered for chat; it advertises {caps:?}"
        );
    }
}

#[test]
fn a_model_that_cannot_chat_is_never_offered_for_it() {
    // The invariant that matters, and it holds across the whole non-LLM range rather than
    // one family at a time: whatever an image, speech or music model advertises, routing a
    // conversation at it fails, so `chat` must not be among it.
    for name in [
        "flux-schnell",
        "flux-dev:fp16",
        "Tongyi-MAI/Z-Image-Turbo",
        "stable-diffusion-3-medium",
        "whisper-large-v3",
        "openai/whisper-small",
        "parler-tts/parler-tts-mini-v1",
        "openai/tts-1",
        "stable-audio-open-1.0",
        "ace-step-v1",
        "wan2.2-t2v",
    ] {
        let caps = model_capabilities(name);
        assert!(
            !caps.is_empty(),
            "{name} advertises nothing at all, so no client can reach it"
        );
        assert!(
            !caps.contains(&"completion".to_string()),
            "{name} is not a conversational model but advertises chat: {caps:?}"
        );
    }
}

#[test]
fn an_embedding_model_offers_embedding_and_nothing_else() {
    // Nomic / BGE / general "bert"-family models accept embeddings
    // requests but produce garbage on completion. Pin the
    // single-cap "embedding"-only response so SDKs (Open-WebUI's
    // embedder picker, etc.) don't offer the wrong toggle.
    for name in [
        "nomic-embed-text:latest",
        "nomic-embed-text-v1.5",
        "bge-large-en-v1.5",
        "snowflake-arctic-embed-l", // "embed" substring
    ] {
        let caps = model_capabilities(name);
        assert_eq!(
            caps,
            vec!["embedding".to_string()],
            "{name} should expose only embedding; got {caps:?}"
        );
    }
}

#[test]
fn normalize_model_id_passes_tagged_and_hf_paths() {
    // Ollama tagged name -> unchanged
    assert_eq!(normalize_model_id("qwen3:latest"), "qwen3:latest");
    assert_eq!(normalize_model_id("gemma4:26b"), "gemma4:26b");
    // HuggingFace path -> unchanged (contains '/')
    assert_eq!(
        normalize_model_id("TheBloke/Qwen3-Coder-7B"),
        "TheBloke/Qwen3-Coder-7B",
    );
    // Bare ollama name -> :latest appended
    assert_eq!(normalize_model_id("qwen3"), "qwen3:latest");
    // Edge: empty string -> ":latest" (not ideal but matches current contract
    // - caller validates via validate_model_id first so this can't escape).
    assert_eq!(normalize_model_id(""), ":latest");
}

/// An untagged image model must still match its resident engine.
///
/// Ollama ids are normalised to `name:latest` on the way in, while the image
/// engine records the name it was asked to load. Comparing them raw meant the
/// unload never matched for any untagged image model - and the caller was told it
/// had worked. This pins the shape of the comparison the unload path relies on.
#[test]
fn a_normalised_id_still_matches_its_untagged_resident() {
    let bare = |s: &str| s.strip_suffix(":latest").unwrap_or(s).to_string();
    assert_eq!(bare(&normalize_model_id("qwen-image")), bare("qwen-image"));
    assert_eq!(
        bare(&normalize_model_id("flux-schnell")),
        bare("flux-schnell")
    );
    // A tagged id keeps its tag, so it must not be confused with a different one.
    assert_ne!(
        bare(&normalize_model_id("qwen-image:v2")),
        bare("qwen-image")
    );
    // HuggingFace paths pass through untouched.
    assert_eq!(
        bare(&normalize_model_id("Qwen/Qwen-Image")),
        "Qwen/Qwen-Image"
    );
}

#[test]
fn only_a_streamed_answer_is_watched_for_its_end() {
    use serde_json::json;
    assert!(super::OLLAMA_CHAT.streams(&json!({})));
    assert!(!super::OLLAMA_GENERATE.streams(&json!({"stream": false})));
    assert!(!super::OPENAI_CHAT.streams(&json!({})));
    assert!(super::OPENAI_COMPLETIONS.streams(&json!({"stream": true})));
    assert!(!super::MESSAGES.streams(&json!({"max_tokens": 8})));
}

/// A log line carries lengths and counts, never what a user wrote or a model answered.
#[test]
fn no_log_line_carries_user_text() {
    fn visit(dir: &std::path::Path, hits: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, hits);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            for (offset, _) in source.match_indices('!') {
                let head = &source[..offset];
                let Some(name) = head
                    .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                else {
                    continue;
                };
                if !matches!(name, "info" | "warn" | "debug" | "trace" | "error") {
                    continue;
                }
                let tail = &source[offset..];
                let Some(end) = tail.find(");") else { continue };
                let call = &tail[..end];
                let suspect = [
                    "snippet",
                    "user_text",
                    ".prompt,",
                    ".prompt)",
                    "prompt.chars",
                    "text.chars",
                    "content.chars",
                    "stop_sequences)",
                    "messages)",
                    ".content)",
                    "transcript,",
                ];
                if suspect.iter().any(|s| call.contains(s)) {
                    let line = head.matches('\n').count() + 1;
                    hits.push(format!("{}:{line}", path.display()));
                }
            }
        }
    }
    let mut hits = Vec::new();
    visit(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut hits,
    );
    assert!(hits.is_empty(), "log lines carrying user text: {hits:?}");
}
