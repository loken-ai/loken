use super::*;

#[test]
fn response_format_to_grammar_handles_all_documented_shapes() {
    // None input -> None output (no response_format constraint).
    assert_eq!(response_format_to_grammar(None), None);

    // {"type":"text"} is the OpenAI default - explicitly no
    // grammar constraint. Pin so a silent rewrite doesn't
    // start forcing JSON-mode on text responses.
    let text = serde_json::json!({"type": "text"});
    assert_eq!(response_format_to_grammar(Some(&text)), None);

    // {"type":"json_object"} -> json_object (bare JSON, no schema).
    let obj = serde_json::json!({"type": "json_object"});
    assert_eq!(
        response_format_to_grammar(Some(&obj)).as_deref(),
        Some("json_object"),
    );

    // {"type":"json_schema", "json_schema":{"schema":{...}}}
    // - the strict-schema path. The grammar engine sees
    // `json_schema:<schema>` where <schema> is the inner schema
    // JSON, not the wrapper.
    let nested = serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": "Person",
            "schema": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    });
    let out = response_format_to_grammar(Some(&nested)).unwrap();
    assert!(out.starts_with("json_schema:"), "got: {out}");
    // Inner schema should be threaded through; outer wrapper keys
    // (name) should NOT be in the serialized form.
    assert!(
        out.contains("\"type\":\"object\""),
        "inner schema missing: {out}"
    );
    assert!(
        !out.contains("Person"),
        "wrapper key leaked into schema: {out}"
    );

    // Some clients flatten the wrapper and put `schema` directly
    // under response_format. Accept that too.
    let flat = serde_json::json!({
        "type": "json_schema",
        "schema": {"type": "integer"},
    });
    let out = response_format_to_grammar(Some(&flat)).unwrap();
    assert!(out.starts_with("json_schema:"), "got: {out}");
    assert!(out.contains("integer"));

    // json_schema with no schema field at all -> None (the inner
    // ?-chain short-circuits when neither location resolves).
    let no_schema = serde_json::json!({
        "type": "json_schema",
        "json_schema": {"name": "Foo"},
    });
    assert_eq!(response_format_to_grammar(Some(&no_schema)), None);

    // Unknown type -> None (lenient match: OpenAI ignores unrecognized
    // response_format types rather than 400ing).
    let unknown = serde_json::json!({"type": "yaml"});
    assert_eq!(response_format_to_grammar(Some(&unknown)), None);

    // Missing type field -> None (treat as unconstrained).
    let no_type = serde_json::json!({"foo": "bar"});
    assert_eq!(response_format_to_grammar(Some(&no_type)), None);
}

#[test]
fn ollama_format_to_grammar_string_and_object() {
    // "json" -> json_object.
    let json_str = serde_json::Value::String("json".to_string());
    assert_eq!(
        ollama_format_to_grammar(Some(&json_str)).as_deref(),
        Some("json_object")
    );
    // Object -> json_schema:<serialized>.
    let schema = serde_json::json!({"type": "object", "properties": {}});
    let out = ollama_format_to_grammar(Some(&schema)).unwrap();
    assert!(out.starts_with("json_schema:"));
    // Unknown string + None.
    let other = serde_json::Value::String("xml".to_string());
    assert_eq!(ollama_format_to_grammar(Some(&other)), None);
    assert_eq!(ollama_format_to_grammar(None), None);
}

#[test]
fn json_schema_grammar_str_caps_huge_schemas() {
    // Small schema fits - returns the prefixed form.
    let small = serde_json::json!({"type": "string"});
    assert!(json_schema_grammar_str(&small)
        .unwrap()
        .starts_with("json_schema:"));

    // Build a schema whose serialized size exceeds the 64 KiB cap
    // by stuffing a long string into a `description` field. Cheap
    // to build (one big-string allocation, no nested traversal).
    let huge_desc = "x".repeat(64 * 1024 + 100);
    let huge = serde_json::json!({"type": "string", "description": huge_desc});
    // Should fall back to None rather than handing a 64 KiB+ string
    // to the grammar engine. Same policy as "unknown shape ⇒ None"
    // - generator runs free, no surprise 500.
    assert_eq!(json_schema_grammar_str(&huge), None);
    // Both helpers share the cap.
    assert_eq!(ollama_format_to_grammar(Some(&huge)), None);
    let rf_huge = serde_json::json!({"type": "json_schema", "json_schema": {"schema": huge}});
    assert_eq!(response_format_to_grammar(Some(&rf_huge)), None);
}

// -- openai_error_body --
//
// Every /v1/* handler that returns an error builds its response
// body via this helper. The OpenAI SDK + OpenAI-compatible clients
// expect the literal envelope `{"error":{"message","type","param",
// "code"}}` - any drift in the wrapping or the `type` discriminator
// breaks `client.error.message` style access patterns.

#[test]
fn openai_error_body_status_400_yields_invalid_request_error_type() {
    // 4xx -> invalid_request_error (OpenAI's documented client-side
    // error discriminator).
    let body = openai_error_body(axum::http::StatusCode::BAD_REQUEST, "missing field");
    assert_eq!(body["error"]["message"], "missing field");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(body["error"]["param"].is_null());
    assert!(body["error"]["code"].is_null());
}

#[test]
fn openai_error_body_status_500_yields_server_error_type() {
    // 5xx -> server_error.
    let body = openai_error_body(
        axum::http::StatusCode::INTERNAL_SERVER_ERROR,
        "model crashed",
    );
    assert_eq!(body["error"]["message"], "model crashed");
    assert_eq!(body["error"]["type"], "server_error");
}

#[test]
fn openai_error_body_status_other_yields_generic_api_error() {
    // Non-4xx, non-5xx -> fallback "api_error". The only realistic
    // shape is 3xx (redirects) which we shouldn't emit but the
    // handler must be safe.
    let body = openai_error_body(axum::http::StatusCode::MOVED_PERMANENTLY, "moved");
    assert_eq!(body["error"]["type"], "api_error");
}

#[test]
fn openai_error_body_envelope_shape_is_stable() {
    // Pin the literal wrapping `{"error":{...}}`. SDK clients
    // access `response.json()["error"]["message"]`; a refactor
    // that drops the outer "error" key or adds a sibling field
    // would break every existing client.
    let body = openai_error_body(axum::http::StatusCode::BAD_REQUEST, "x");
    let obj = body.as_object().expect("top-level is an object");
    assert_eq!(obj.len(), 1, "only `error` key at top level");
    assert!(obj.contains_key("error"));
    let err = body["error"].as_object().expect("error is an object");
    // Exactly 4 documented fields: message / type / param / code.
    assert_eq!(
        err.len(),
        4,
        "got fields: {:?}",
        err.keys().collect::<Vec<_>>()
    );
    for key in ["message", "type", "param", "code"] {
        assert!(err.contains_key(key), "missing field: {key}");
    }
}

#[test]
fn openai_error_body_accepts_owned_string_and_str() {
    // `impl Into<String>` - both String and &str must compile and
    // surface the same body.
    let from_str = openai_error_body(axum::http::StatusCode::BAD_REQUEST, "msg");
    let from_string = openai_error_body(axum::http::StatusCode::BAD_REQUEST, String::from("msg"));
    assert_eq!(from_str, from_string);
}

/// `n` is bounded the same on chat and completions: 1 when absent, refused at 0 and
/// past the cap.
#[test]
fn n_is_bounded_on_both_routes() {
    use super::super::sampling::{completions_asked, N_MAX};
    assert_eq!(completions_asked(None).unwrap(), 1);
    assert_eq!(completions_asked(Some(3)).unwrap(), 3);
    assert!(completions_asked(Some(0)).is_err());
    assert!(completions_asked(Some(N_MAX + 1)).is_err());
    // The chat request type does not refuse what the handler serves.
    let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m", "messages": [{"role": "user", "content": "hi"}], "n": 3
    }))
    .unwrap();
    assert!(validate_request(&req).is_ok());
}

/// Runs gather into one response: choices re-indexed in order, completions summed, the
/// prompt counted once or per run.
#[tokio::test]
async fn runs_merge_into_one_response() {
    let answer = |text: &str| -> Result<Response, ApiError> {
        Ok(Json(serde_json::json!({
            "id": "x",
            "choices": [{"index": 0, "text": text}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}
        }))
        .into_response())
    };
    for (sum_prompts, prompt) in [(false, 10), (true, 30)] {
        let out = merge_runs(3, sum_prompts, |i| {
            let r = answer(&format!("run {i}"));
            async move { r }
        })
        .await
        .unwrap();
        let bytes = axum::body::to_bytes(out.into_body(), usize::MAX)
            .await
            .unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["choices"].as_array().unwrap().len(), 3);
        assert_eq!(v["choices"][2]["index"], 2);
        assert_eq!(v["choices"][2]["text"], "run 2");
        assert_eq!(v["usage"]["prompt_tokens"], prompt);
        assert_eq!(v["usage"]["completion_tokens"], 6);
        assert_eq!(v["usage"]["total_tokens"], prompt + 6);
    }
}
