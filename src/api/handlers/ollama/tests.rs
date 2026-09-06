/// EVERY image family must advertise `txt2img`.
///
/// `model_capabilities` matches on the family name, and a family added to
/// `infer_model_family` but not here falls through to the CHAT default - which is
/// how SDXL came to describe itself to clients as a chat model, so it never
/// appeared in any image picker. The failure is silent on the server: the model
/// loads and renders perfectly when asked directly; it simply cannot be found.
#[test]
fn every_image_family_advertises_that_it_generates_images() {
    // One representative id per image family the server actually serves.
    for id in [
        "flux-schnell",
        "flux-kontext",
        "rayflux",
        "Tongyi-MAI/Z-Image-Turbo",
        "rayzist",
        "qwen-image",
        "rayqwest",
        "boogu",
        "raymnants",
        "rayctifier",
        "rayburn",
    ] {
        let caps = super::models::model_capabilities(id);
        assert!(
            caps.iter().any(|c| c == "txt2img"),
            "{id} (family {}) does not advertise txt2img; it advertises {caps:?}",
            infer_model_family(id)
        );
        assert!(
            !caps.iter().any(|c| c == "completion"),
            "{id} is an image model but advertises chat: {caps:?}"
        );
    }
}

use super::*;

#[test]
fn parse_modelfile_from_basic() {
    assert_eq!(
        parse_modelfile_from("FROM llama3:latest"),
        Some("llama3:latest")
    );
    // Lowercase directive accepted.
    assert_eq!(
        parse_modelfile_from("from llama3:latest"),
        Some("llama3:latest")
    );
    // Mixed case + leading whitespace.
    assert_eq!(
        parse_modelfile_from("  From  llama3:latest  "),
        Some("llama3:latest")
    );
    // First FROM line wins; subsequent ones are ignored.
    let mf = "# comment\nFROM mistral:7b\nPARAMETER temperature 0.7\nFROM ignored:tag";
    assert_eq!(parse_modelfile_from(mf), Some("mistral:7b"));
    // CRLF line endings.
    assert_eq!(
        parse_modelfile_from("PARAMETER x 1\r\nFROM gemma:2b\r\n"),
        Some("gemma:2b")
    );
}

#[test]
fn parse_modelfile_from_returns_none() {
    assert_eq!(parse_modelfile_from(""), None);
    assert_eq!(parse_modelfile_from("PARAMETER temperature 0.7"), None);
    // "FROM" without a trailing space is not the directive.
    assert_eq!(parse_modelfile_from("FROMM mistral"), None);
    // Bare "FROM " (trailing whitespace only) - trim() strips the space,
    // so the line no longer matches the "FROM " starts_with check and
    // we never enter the slice path.
    assert_eq!(parse_modelfile_from("FROM "), None);
    assert_eq!(parse_modelfile_from("FROM   "), None);
    // Tab-separated "FROM\tname" is not accepted (spec is space-separated).
    assert_eq!(parse_modelfile_from("FROM\tllama3"), None);
}

#[test]
fn parse_modelfile_from_traversal_passes_back_to_validator() {
    // Parser itself doesn't filter - it surfaces whatever the modelfile
    // contains. validate_model_id is the gate. Confirm the path-traversal
    // string round-trips so the handler's validate_model_id call has
    // something to reject.
    let mf = "FROM ../../../etc/passwd";
    assert_eq!(parse_modelfile_from(mf), Some("../../../etc/passwd"));
    assert!(validate_model_id("../../../etc/passwd").is_err());
}

#[test]
fn normalize_blob_digest_canonicalizes_to_dash_lowerhex() {
    let hex_lc = "a".repeat(64);
    let hex_uc = "A".repeat(64);
    // sha256-lowerhex is the canonical form: identity.
    assert_eq!(
        normalize_blob_digest(&format!("sha256-{hex_lc}")),
        format!("sha256-{hex_lc}")
    );
    // Legacy colon form gets rewritten to the dash form.
    assert_eq!(
        normalize_blob_digest(&format!("sha256:{hex_lc}")),
        format!("sha256-{hex_lc}")
    );
    // Uppercase hex gets lowercased so downstream filesystem lookups
    // hit the same name regardless of how the client cased it.
    assert_eq!(
        normalize_blob_digest(&format!("sha256-{hex_uc}")),
        format!("sha256-{hex_lc}")
    );
    assert_eq!(
        normalize_blob_digest(&format!("sha256:{hex_uc}")),
        format!("sha256-{hex_lc}")
    );
}

#[test]
fn validate_blob_digest_accepts_known_forms() {
    let hex = "a".repeat(64);
    assert!(validate_blob_digest(&format!("sha256-{hex}")).is_ok());
    assert!(validate_blob_digest(&format!("sha256:{hex}")).is_ok());
}

#[test]
fn validate_blob_digest_rejects_traversal_and_bad_hex() {
    assert!(validate_blob_digest("").is_err());
    assert!(validate_blob_digest("../etc").is_err());
    assert!(validate_blob_digest("sha256-").is_err()); // empty hex
    assert!(validate_blob_digest("sha256-short").is_err());
    assert!(validate_blob_digest(&format!("md5-{}", "a".repeat(64))).is_err());
    // Hex with non-hex char.
    let bad_hex = format!("g{}", "a".repeat(63));
    assert!(validate_blob_digest(&format!("sha256-{bad_hex}")).is_err());
}

#[test]
fn estimate_parameter_size_known_sizes() {
    // 4.3 GB GGUF (typical llama-7B Q4_K) -> ~7.2B.
    assert_eq!(estimate_parameter_size(4_300_000_000, "gguf"), "7.2B");
    // 13 GB safetensors at F16 -> ~6.5B.
    assert_eq!(
        estimate_parameter_size(13_000_000_000, "safetensors"),
        "6.5B"
    );
    // Empty / unknown.
    assert_eq!(estimate_parameter_size(0, "gguf"), "unknown");
    // Small under-1B model uses M suffix.
    assert_eq!(estimate_parameter_size(500_000_000, "gguf"), "833M");
}

#[test]
fn infer_model_family_known_models() {
    // The video family, which used to fall through to the text default so a client
    // could not tell a video model from an LLM.
    assert_eq!(infer_model_family("wan"), "video");
    assert_eq!(infer_model_family("wan-14b"), "video");
    assert_eq!(
        infer_model_family("wan-photoreal"),
        "video",
        "a fine-tune's own tag"
    );
    // And it must not swallow an unrelated name that merely contains the letters.
    assert_ne!(infer_model_family("swan-lake-7b"), "video");
    assert_eq!(infer_model_family("gemma4:26b"), "gemma");
    assert_eq!(infer_model_family("qwen3-coder:latest"), "qwen");
    assert_eq!(infer_model_family("mistral-7b"), "mistral");
    assert_eq!(infer_model_family("mixtral-8x7b"), "mistral");
    assert_eq!(infer_model_family("deepseek-r1"), "deepseek");
    assert_eq!(infer_model_family("phi3:mini"), "phi");
    assert_eq!(infer_model_family("moondream"), "moondream");
    assert_eq!(infer_model_family("nomic-embed-text"), "bert");
    // Fallback for unknown.
    assert_eq!(infer_model_family("custom-model-xyz"), "llama");
}

#[test]
fn infer_model_family_non_llm_modalities() {
    // Image-gen families. Without these, /api/show used to return
    // family="llama" for flux/z-image checkpoints, misleading SDK
    // model-card UIs that switch behaviour on the family name.
    assert_eq!(infer_model_family("flux-schnell"), "flux");
    assert_eq!(infer_model_family("flux-dev:fp16"), "flux");
    assert_eq!(
        infer_model_family("black-forest-labs/FLUX.1-schnell"),
        "flux"
    );
    assert_eq!(infer_model_family("Tongyi-MAI/Z-Image-Turbo"), "z-image");
    assert_eq!(infer_model_family("z_image_local"), "z-image");

    // ASR + TTS families
    assert_eq!(infer_model_family("whisper-large-v3"), "whisper");
    assert_eq!(infer_model_family("openai/whisper-small"), "whisper");
    assert_eq!(infer_model_family("parler-tts/parler-tts-mini-v1"), "tts");
    assert_eq!(infer_model_family("tts-1"), "tts");
    assert_eq!(infer_model_family("openai/tts-1-hd"), "tts");
}

#[test]
fn infer_model_family_stable_diffusion_does_not_collide_with_stablelm() {
    // The historical "stable" -> "stablelm" arm swallowed
    // "stable-diffusion" and "stable-cascade". Pin the resolution
    // order so a future bump can't reintroduce the collision.
    assert_eq!(
        infer_model_family("stable-diffusion-3-medium"),
        "stable-diffusion"
    );
    assert_eq!(
        infer_model_family("stable-cascade-prior"),
        "stable-diffusion"
    );
    // StableLM proper (text LLM) still resolves correctly.
    assert_eq!(infer_model_family("stablelm-3b"), "stablelm");
    assert_eq!(infer_model_family("stable-code-3b"), "stablelm");
}

#[test]
fn http_status_for_load_error_maps_not_found_keywords() {
    use axum::http::StatusCode;
    assert_eq!(
        http_status_for_load_error("HTTP 404 returned"),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        http_status_for_load_error("model not found"),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        http_status_for_load_error("No such file or directory"),
        StatusCode::NOT_FOUND
    );
    assert_eq!(
        http_status_for_load_error("missing config.json"),
        StatusCode::NOT_FOUND
    );
    // Anything else -> 500 so retryable infra issues don't get masked
    // as client errors.
    assert_eq!(
        http_status_for_load_error("CUDA OOM"),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(
        http_status_for_load_error(""),
        StatusCode::INTERNAL_SERVER_ERROR
    );
}

// -- extract_generation_options --
//
// Maps Ollama's `options` JSON to GenerationParams. Every Ollama
// sampling-knob name -> struct-field mapping is documented; a
// silent rename or type-cast bug would mean callers send e.g.
// `temperature: 0.3` and the server ignores it. Pin the full
// table so a future refactor can't drop a field.

#[test]
fn extract_generation_options_returns_defaults_for_none() {
    let p = extract_generation_options(None);
    // All optional fields are None; stop_sequences is empty Vec.
    assert!(p.max_tokens.is_none());
    assert!(p.temperature.is_none());
    assert!(p.top_p.is_none());
    assert!(p.top_k.is_none());
    assert!(p.seed.is_none());
    assert!(p.stop_sequences.is_empty());
    assert!(p.repeat_penalty.is_none());
    assert!(p.repeat_last_n.is_none());
    assert!(p.context_length.is_none());
    assert!(p.session_id.is_none());
    assert!(p.grammar.is_none());
}

#[test]
fn extract_generation_options_returns_defaults_for_empty_object() {
    let empty = serde_json::json!({});
    let p = extract_generation_options(Some(&empty));
    assert!(p.max_tokens.is_none());
    assert!(p.temperature.is_none());
    assert!(p.session_id.is_none());
}

#[test]
fn extract_generation_options_maps_every_documented_ollama_knob() {
    // Pin the full Ollama -> GenerationParams table. The field
    // names on the JSON side are the wire contract - `num_predict`
    // (not `max_tokens`), `num_ctx` (not `context_length`) - and
    // can't drift without breaking SDK clients.
    let opts = serde_json::json!({
        "num_predict": 256,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "seed": 42_u64,
        "stop": ["</s>", "Human:"],
        "early_exit_threshold": 0.95,
        "repeat_penalty": 1.15,
        "repeat_last_n": 128,
        "num_ctx": 8192,
        "session_id": "abc-123",
        "grammar": "json_object"
    });
    let p = extract_generation_options(Some(&opts));
    assert_eq!(p.max_tokens, Some(256));
    assert_eq!(p.temperature, Some(0.3));
    assert_eq!(p.top_p, Some(0.9));
    assert_eq!(p.top_k, Some(40));
    assert_eq!(p.seed, Some(42));
    assert_eq!(p.stop_sequences, vec!["</s>", "Human:"]);
    assert_eq!(p.early_exit_threshold, Some(0.95));
    assert_eq!(p.repeat_penalty, Some(1.15));
    assert_eq!(p.repeat_last_n, Some(128));
    assert_eq!(p.context_length, Some(8192));
    assert_eq!(p.session_id.as_deref(), Some("abc-123"));
    assert_eq!(p.grammar.as_deref(), Some("json_object"));
}

#[test]
fn extract_generation_options_ignores_wrong_type_silently() {
    // Wrong-type values fall through to None - matches Ollama's
    // lenient behaviour. Pin so a refactor doesn't start
    // 400-ing on a typoed-type that previously worked.
    let opts = serde_json::json!({
        "temperature": "not-a-number",  // wrong: string
        "top_p": [0.9],                 // wrong: array
        "seed": -5,                     // wrong: signed (u64 only)
        "stop": "single",               // wrong: string (need array)
    });
    let p = extract_generation_options(Some(&opts));
    assert!(p.temperature.is_none());
    assert!(p.top_p.is_none());
    assert!(p.seed.is_none(), "negative int can't fit u64");
    assert!(
        p.stop_sequences.is_empty(),
        "non-array stop falls back to empty"
    );
}

#[test]
fn extract_generation_options_stop_filters_non_string_entries() {
    // `stop` is a mixed-type array - filter to strings only,
    // dropping garbage entries silently (Ollama-compatible).
    let opts = serde_json::json!({
        "stop": ["A", 42, "B", null, "C"]
    });
    let p = extract_generation_options(Some(&opts));
    assert_eq!(p.stop_sequences, vec!["A", "B", "C"]);
}

#[test]
fn estimate_parameter_size_scales_with_format() {
    // GGUF: ~0.6 bytes/param. 4 GB / 0.6 ≈ 6.67B params -> "6.7B".
    assert_eq!(
        estimate_parameter_size(4 * 1024 * 1024 * 1024, "gguf"),
        "7.2B", // 4 GiB / 0.6 = 7.158B
    );
    // safetensors F16: 2 bytes/param. 14 GB / 2 = 7B -> "7.5B".
    // 14 GiB = 14 * 2^30 ≈ 1.503e10 / 2 ≈ 7.516e9
    assert_eq!(
        estimate_parameter_size(14 * 1024 * 1024 * 1024, "safetensors"),
        "7.5B",
    );
    // Small model - under 1B uses 'M' suffix.
    assert_eq!(estimate_parameter_size(600_000_000, "safetensors"), "300M");
    // 0 bytes -> 'unknown' rather than '0M' (prevents misleading
    // "this is a 0-param model" output on /api/show for partial loads).
    assert_eq!(estimate_parameter_size(0, "gguf"), "unknown");
    assert_eq!(estimate_parameter_size(0, "safetensors"), "unknown");
}

#[test]
fn infer_model_family_orders_specific_before_generic() {
    // gemma > generic, even when name embeds 'gemma4' (would not match
    // any of the other arms anyway, but pins the routing).
    assert_eq!(infer_model_family("gemma4:26b"), "gemma");
    assert_eq!(infer_model_family("gemma3-it"), "gemma");
    // qwen
    assert_eq!(infer_model_family("qwen3-coder"), "qwen");
    // mistral and mixtral collapse to 'mistral'
    assert_eq!(infer_model_family("mistral-7b"), "mistral");
    assert_eq!(infer_model_family("mixtral-8x7b"), "mistral");
    // embedding/sentence-encoder families collapse to 'bert'
    assert_eq!(infer_model_family("nomic-embed-text"), "bert");
    assert_eq!(infer_model_family("BAAI/bge-large-en"), "bert");
    // Unknown -> llama fallback (never panic, always a usable family
    // string for prompt-template selection).
    assert_eq!(infer_model_family("mystery-coder-v2"), "llama");
    assert_eq!(infer_model_family(""), "llama");
}

#[test]
fn extract_kv_quant_override_maps_aliases_case_insensitively() {
    use crate::inference::engine::llm_engine::KvQuant;
    let pick = |s: &str| {
        let v = serde_json::json!({"kv_quant": s});
        extract_kv_quant_override(Some(&v))
    };
    assert!(matches!(pick("off"), Some(KvQuant::Off)));
    assert!(matches!(pick("NONE"), Some(KvQuant::Off)));
    assert!(matches!(pick("f16"), Some(KvQuant::Off)));
    assert!(matches!(pick("F32"), Some(KvQuant::Off)));
    assert!(matches!(pick("q8"), Some(KvQuant::Q8)));
    assert!(matches!(pick("Q8_0"), Some(KvQuant::Q8)));
    assert!(matches!(pick("q4"), Some(KvQuant::Q4)));
    assert!(matches!(pick("q4_0"), Some(KvQuant::Q4)));
    // Unknown values yield None (caller treats as 'no override' and
    // falls back to the model config default).
    assert!(pick("q2").is_none());
    assert!(pick("garbage").is_none());
    // Wrong shapes return None gracefully - never error.
    assert!(extract_kv_quant_override(None).is_none());
    assert!(extract_kv_quant_override(Some(&serde_json::json!({}))).is_none());
    assert!(extract_kv_quant_override(Some(&serde_json::json!({"kv_quant": 4}))).is_none());
}

#[test]
fn validate_ollama_options_accepts_in_range_and_omitted_fields() {
    // None options -> always Ok.
    assert!(validate_ollama_options(None).is_ok());
    // Empty object -> no keys to validate.
    let empty = serde_json::json!({});
    assert!(validate_ollama_options(Some(&empty)).is_ok());
    // All knobs at sane values.
    let ok = serde_json::json!({
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_predict": 256,
        "num_ctx": 8192,
        "top_k": 40,
        "repeat_last_n": 64,
        "stop": ["\n\n", "<end>"],
        "session_id": "abc-123",
    });
    assert!(validate_ollama_options(Some(&ok)).is_ok());
    // num_predict = -1 (until EOS) is the documented sentinel.
    let neg_one = serde_json::json!({"num_predict": -1});
    assert!(validate_ollama_options(Some(&neg_one)).is_ok());
}

#[test]
fn validate_ollama_options_rejects_each_out_of_range_knob() {
    let bad = |obj: serde_json::Value, key: &str| {
        let err =
            validate_ollama_options(Some(&obj)).expect_err(&format!("expected {key} rejection"));
        match err {
            ApiError::Validation(msg) => assert!(
                msg.contains(key),
                "validation message for {key} should mention the field; got: {msg}"
            ),
            other => panic!("expected Validation, got {other:?}"),
        }
    };
    // Temperature: out of [0, 2]. (NaN can't appear here because
    // serde_json's `Value::from(f64::NAN)` already collapses to Null,
    // which short-circuits the field's `as_f64()` to None - so the
    // validator's `!t.is_finite()` guard is belt-and-suspenders only
    // and not exercisable through this JSON-shaped entry point.)
    bad(serde_json::json!({"temperature": 3.0}), "temperature");
    bad(serde_json::json!({"temperature": -0.1}), "temperature");
    // top_p: out of [0, 1].
    bad(serde_json::json!({"top_p": 1.5}), "top_p");
    // repeat_penalty: negative.
    bad(
        serde_json::json!({"repeat_penalty": -0.5}),
        "repeat_penalty",
    );
    // num_predict: < -1 or above the 128K cap.
    bad(serde_json::json!({"num_predict": -2}), "num_predict");
    bad(serde_json::json!({"num_predict": 200_000}), "num_predict");
    // num_ctx: < 1 or above 1M.
    bad(serde_json::json!({"num_ctx": 0}), "num_ctx");
    bad(serde_json::json!({"num_ctx": 2_000_000}), "num_ctx");
    // top_k: negative or above 1M.
    bad(serde_json::json!({"top_k": -1}), "top_k");
    bad(serde_json::json!({"top_k": 2_000_000}), "top_k");
    // repeat_last_n: same shape.
    bad(serde_json::json!({"repeat_last_n": -1}), "repeat_last_n");
    bad(
        serde_json::json!({"repeat_last_n": 200_000}),
        "repeat_last_n",
    );
    // stop: too many entries.
    bad(
        serde_json::json!({"stop": ["a", "b", "c", "d", "e"]}),
        "stop",
    );
    // stop[0]: entry too long.
    bad(serde_json::json!({"stop": ["x".repeat(300)]}), "stop[0]");
    // session_id: too long.
    bad(
        serde_json::json!({"session_id": "x".repeat(300)}),
        "session_id",
    );
}

/// `keep_alive:0` answers 200 whatever happens, so `done_reason` IS the report.
/// It used to say "unload" unconditionally: a model no engine was holding - every
/// TTS model, before the walk knew that engine - was reported as freed VRAM, and a
/// caller emptying a card for a render had no way to learn otherwise.
#[tokio::test]
async fn keep_alive_zero_does_not_claim_an_unload_it_did_not_do() {
    let state = APIServer::new(
        "/nonexistent-ollama-models".to_string(),
        "/nonexistent-hf-models".to_string(),
    );
    let mut request =
        OllamaGenerateRequest::new("parler-tts/parler-tts-mini-v1".to_string(), String::new());
    request.keep_alive = Some("0".to_string());
    // No forwarding header: this exercises the local path, which is what an unload is -
    // an instruction addressed to THIS node.
    let resp = ollama_generate(
        axum::extract::State(state),
        axum::http::HeaderMap::new(),
        OllamaJson(request),
    )
    .await
    .expect("an unload of a model that is not loaded is not an error");
    let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["done"], true);
    assert_eq!(
        json["done_reason"], "not_loaded",
        "nothing was resident, so nothing was freed: {json}"
    );
}

#[test]
fn classify_pull_error_routes_404_keywords_to_notfound() {
    // 404 in the body of an upstream response.
    match classify_pull_error("foo", "HTTP 404 Not Found") {
        ApiError::NotFound(m) => assert!(m.contains("'foo'") && m.contains("not found")),
        other => panic!("expected NotFound, got {other:?}"),
    }
    // 'not found' substring without a numeric code.
    match classify_pull_error("bar", "manifest not found at registry") {
        ApiError::NotFound(_) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
    // Local fs error from a private registry fall-through.
    match classify_pull_error("baz", "No such file or directory") {
        ApiError::NotFound(_) => {}
        other => panic!("expected NotFound, got {other:?}"),
    }
    // Anything else -> Internal so retryable network/disk errors
    // surface as 500 rather than misleading 404s.
    match classify_pull_error("qux", "connection reset by peer") {
        ApiError::Internal(m) => assert!(m.contains("Failed to pull")),
        other => panic!("expected Internal, got {other:?}"),
    }
}
