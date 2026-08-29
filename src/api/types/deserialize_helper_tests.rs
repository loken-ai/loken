//! Lock in the SDK-compat quirks for the three custom deserializers.
//! Every case here was triggered by a real third-party client at some
//! point - a regression here usually shows up as `422 Unprocessable
//! Entity` from this server while the same payload works against Ollama.
use super::*;

// ---------- keep_alive ----------

#[derive(Deserialize)]
struct KA {
    #[serde(default, deserialize_with = "deserialize_keep_alive")]
    keep_alive: Option<String>,
}

fn parse_ka(json: &str) -> Option<String> {
    serde_json::from_str::<KA>(json).unwrap().keep_alive
}

#[test]
fn keep_alive_accepts_string_int_and_null() {
    assert_eq!(parse_ka(r#"{"keep_alive":"5m"}"#).as_deref(), Some("5m"));
    // Integer form - bench / ollama go-client send this.
    assert_eq!(parse_ka(r#"{"keep_alive":0}"#).as_deref(), Some("0"));
    assert_eq!(parse_ka(r#"{"keep_alive":-1}"#).as_deref(), Some("-1"));
    // null / omitted -> None.
    assert_eq!(parse_ka(r#"{"keep_alive":null}"#), None);
    assert_eq!(parse_ka(r#"{}"#), None);
    // Garbage types short-circuit to None - never propagate a serde error
    // that would 422 the whole request.
    assert_eq!(parse_ka(r#"{"keep_alive":true}"#), None);
    assert_eq!(parse_ka(r#"{"keep_alive":{"nested":"obj"}}"#), None);
}

// ---------- optional_seed ----------

#[derive(Deserialize)]
struct SD {
    #[serde(default, deserialize_with = "deserialize_optional_seed")]
    seed: Option<u64>,
}

fn parse_sd(json: &str) -> Option<u64> {
    serde_json::from_str::<SD>(json).unwrap().seed
}

#[test]
fn optional_seed_maps_negatives_and_floats_to_none() {
    assert_eq!(parse_sd(r#"{"seed":42}"#), Some(42));
    assert_eq!(parse_sd(r#"{"seed":0}"#), Some(0));
    // The common "use a random seed" convention from many SDKs.
    assert_eq!(parse_sd(r#"{"seed":-1}"#), None);
    // Floats are not valid seeds.
    assert_eq!(parse_sd(r#"{"seed":3.14}"#), None);
    // Strings + null + omitted -> None.
    assert_eq!(parse_sd(r#"{"seed":"abc"}"#), None);
    assert_eq!(parse_sd(r#"{"seed":null}"#), None);
    assert_eq!(parse_sd(r#"{}"#), None);
}

// ---------- optional_thinking ----------

#[derive(Deserialize)]
struct TK {
    #[serde(default, deserialize_with = "deserialize_optional_thinking")]
    thinking: Option<String>,
}

fn parse_tk(json: &str) -> Option<String> {
    serde_json::from_str::<TK>(json).unwrap().thinking
}

// ---------- Ollama request/response constructors ----------

#[test]
fn ollama_chat_request_new_defaults_nullable_fields_to_none() {
    let req = OllamaChatRequest::new(
        "model".into(),
        vec![Message::new("user".into(), "hi".into())],
    );
    assert_eq!(req.model, "model");
    assert_eq!(req.messages.len(), 1);
    // stream defaults to false - caller flips this for SSE responses.
    // OllamaChatRequest::new is for handler tests / in-process callers
    // who default to single-shot mode.
    assert!(!req.stream);
    assert!(req.format.is_none());
    assert!(req.options.is_none());
    assert!(req.keep_alive.is_none());
    assert!(req.thinking.is_none());
    assert!(req.tools.is_none());
}

#[test]
fn ollama_generate_request_new_defaults_match_chat() {
    let req = OllamaGenerateRequest::new("model".into(), "hello".into());
    assert_eq!(req.model, "model");
    assert_eq!(req.prompt, "hello");
    assert!(!req.stream);
    assert!(req.images.is_none());
    assert!(req.suffix.is_none(), "FIM suffix omitted by default");
    assert!(req.raw.is_none(), "raw mode opt-in only");
}

#[test]
fn ollama_pull_request_new_defaults_source_to_ollama() {
    // Pull defaults to the Ollama registry; HuggingFace requires
    // an explicit source override. Pin so a refactor doesn't flip
    // the default and pull from the wrong registry silently.
    let req = OllamaPullRequest::new("llama3:latest".into());
    assert_eq!(req.name, "llama3:latest");
    assert_eq!(
        req.source, "ollama",
        "default source must be 'ollama' (not 'huggingface' or '')"
    );
    // stream defaults to false in the constructor - distinct from
    // the chat/generate requests' `default_stream_true`. Pull is
    // a one-shot operation; SSE streaming is opt-in.
    assert!(!req.stream);
    assert!(req.insecure.is_none());
}

#[test]
fn ollama_chat_response_new_marks_done_with_rfc3339_timestamp() {
    let resp = OllamaChatResponse::new("m".into(), Message::new("assistant".into(), "ok".into()));
    assert_eq!(resp.model, "m");
    assert!(resp.done, "single-shot constructor marks done=true");
    // created_at is RFC3339 - pin shape so a future refactor
    // doesn't switch to Unix epoch or a custom format.
    // Real Ollama clients rely on RFC3339 for log correlation.
    let parsed = chrono::DateTime::parse_from_rfc3339(&resp.created_at);
    assert!(
        parsed.is_ok(),
        "created_at must be RFC3339; got {}",
        resp.created_at
    );
    // All timing + optional fields default to None.
    assert!(resp.total_duration.is_none());
    assert!(resp.tool_calls.is_none());
    assert!(resp.vision_processed.is_none());
    assert!(resp.done_reason.is_none());
}

#[test]
fn ollama_generate_response_new_includes_audios_and_images_as_none() {
    // /api/generate response carries optional images (Flux) +
    // audios (TTS) - both must default to None so the wire-shape
    // for text-only responses isn't bloated with empty arrays.
    let resp = OllamaGenerateResponse::new("m".into(), "text".into());
    assert!(resp.images.is_none());
    assert!(resp.audios.is_none());
    // RFC3339 timestamp, same as chat response.
    assert!(chrono::DateTime::parse_from_rfc3339(&resp.created_at).is_ok());
}

#[test]
fn ollama_response_done_reason_skip_serializing_when_none() {
    // The chat + generate responses use `skip_serializing_if`
    // on every optional field. A regression that dropped that
    // attribute would emit `"total_duration": null` on every
    // intermediate stream chunk, which real Ollama omits - and
    // some go-client parsers reject as a contract drift.
    let resp = OllamaChatResponse::new("m".into(), Message::new("assistant".into(), "hi".into()));
    let json = serde_json::to_value(&resp).unwrap();
    assert!(
        json.get("done_reason").is_none(),
        "done_reason=None must be elided, not emitted as null"
    );
    assert!(
        json.get("total_duration").is_none(),
        "total_duration=None must be elided"
    );
    assert!(json.get("tool_calls").is_none());
    assert!(json.get("vision_processed").is_none());
    // Required fields must still appear.
    assert!(json["model"].is_string());
    assert!(json["created_at"].is_string());
    assert!(json["done"].as_bool() == Some(true));
    assert!(json["message"].is_object());
}

// ---------- StructuredOutput / Tool constructors ----------

#[test]
fn structured_output_json_uses_lowercase_discriminator() {
    // The `output_type` string is the OpenAI-compatible wire
    // discriminator. SDKs key off it ("json" / "json_schema");
    // any case change (e.g. "JSON" or "Json") silently breaks
    // every client.
    let s = StructuredOutput::json();
    assert_eq!(s.output_type, "json");
    assert!(s.json_schema.is_none(), "json variant carries no schema");
}

#[test]
fn structured_output_json_schema_carries_schema_verbatim() {
    let schema = serde_json::json!({"type": "object", "required": ["x"]});
    let s = StructuredOutput::json_schema(schema.clone());
    assert_eq!(
        s.output_type, "json_schema",
        "schema variant uses snake_case discriminator"
    );
    assert_eq!(
        s.json_schema,
        Some(schema),
        "schema must be threaded through verbatim"
    );
}

#[test]
fn tool_function_constructor_sets_type_to_function_lowercase() {
    // "type": "function" is OpenAI's tool-call discriminator;
    // lowercase, not "Function". Pin so a refactor doesn't
    // silently break tool-calling for every existing SDK.
    let t = Tool::function(
        "get_weather".to_string(),
        Some("Get the current weather".to_string()),
        Some(serde_json::json!({"type": "object"})),
    );
    assert_eq!(
        t.r#type, "function",
        "tool type discriminator must be lowercase 'function'"
    );
    let func = t.function.expect("function payload present");
    assert_eq!(func.name, "get_weather");
    assert_eq!(func.description.as_deref(), Some("Get the current weather"));
    assert!(func.parameters.is_some());
}

#[test]
fn tool_function_constructor_accepts_minimal_form() {
    // Just the name - description + parameters omitted.
    let t = Tool::function("simple".to_string(), None, None);
    let func = t.function.expect("function payload present");
    assert_eq!(func.name, "simple");
    assert!(func.description.is_none());
    assert!(func.parameters.is_none());
}

#[test]
fn tool_function_serializes_with_required_type_field() {
    // The struct's r#type field MUST serialize to JSON key "type"
    // (not "r#type" or "type_") so it matches OpenAI's tool
    // schema. Round-trip JSON to pin this.
    let t = Tool::function("f".to_string(), None, None);
    let json = serde_json::to_value(&t).unwrap();
    assert_eq!(json["type"], "function");
    assert!(json["function"]["name"].as_str() == Some("f"));
    // description/parameters elided when None.
    assert!(json["function"].get("description").is_none());
    assert!(json["function"].get("parameters").is_none());
}

#[test]
fn optional_thinking_normalises_bool_and_strings() {
    // Ollama 0.5+ bool form.
    assert_eq!(parse_tk(r#"{"thinking":true}"#).as_deref(), Some("enabled"));
    assert_eq!(
        parse_tk(r#"{"thinking":false}"#).as_deref(),
        Some("disabled")
    );
    // Legacy string form passes through unchanged so custom values
    // (e.g. "high"/"low" reasoning levels) reach the engine intact.
    assert_eq!(
        parse_tk(r#"{"thinking":"enabled"}"#).as_deref(),
        Some("enabled")
    );
    assert_eq!(parse_tk(r#"{"thinking":"high"}"#).as_deref(), Some("high"));
    // null + omitted -> None.
    assert_eq!(parse_tk(r#"{"thinking":null}"#), None);
    assert_eq!(parse_tk(r#"{}"#), None);
    // Other types ignored - no 422.
    assert_eq!(parse_tk(r#"{"thinking":42}"#), None);
}
