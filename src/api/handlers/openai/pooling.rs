//! The scoring and embedding surfaces other servers name differently: Cohere's `/v2/rerank` and
//! `/v2/embed`, and vLLM's `/score`. Each reads its own request shape, runs the handler that
//! already serves the work, and writes its own response shape.

use super::*;
use axum::response::IntoResponse;

/// The JSON a handler answered with, and its status.
async fn answer(resp: Response) -> (axum::http::StatusCode, serde_json::Value) {
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null),
    )
}

/// A request id in the shape the clients of these surfaces expect to read.
fn request_id() -> String {
    format!("{:032x}", rand::random::<u128>())
}

/// The strings of a Cohere `documents` array: plain strings, or objects carrying `text`.
fn document_texts(documents: &serde_json::Value) -> Option<Vec<String>> {
    documents
        .as_array()?
        .iter()
        .map(|d| match d {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(o) => o.get("text")?.as_str().map(str::to_string),
            _ => None,
        })
        .collect()
}

/// Cohere `/v2/rerank`: `{model, query, documents, top_n, return_documents}` answered with
/// `{id, results: [{index, relevance_score}], meta}`.
pub(crate) async fn cohere_rerank(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let Some(documents) = body.get("documents").and_then(document_texts) else {
        return ApiError::Validation(
            "`documents` must be an array of strings or of objects with a `text`".into(),
        )
        .into_response();
    };
    let mut adapted = serde_json::json!({
        "model": body.get("model").cloned().unwrap_or_default(),
        "query": body.get("query").cloned().unwrap_or_default(),
        "documents": documents,
    });
    if let Some(n) = body.get("top_n") {
        adapted["top_n"] = n.clone();
    }
    let return_documents = body
        .get("return_documents")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    adapted["return_documents"] = serde_json::json!(return_documents);
    let (status, v) = answer(openai_rerank(State(state), headers, Json(adapted)).await).await;
    if !status.is_success() {
        return (status, Json(v)).into_response();
    }
    Json(serde_json::json!({
        "id": request_id(),
        "results": v["results"],
        "meta": {
            "api_version": {"version": "2"},
            "billed_units": {"search_units": 1},
        },
    }))
    .into_response()
}

/// Cohere `/v2/embed`: `{model, texts, input_type, embedding_types}` answered with
/// `{id, embeddings: {float: [[..]]}, texts, meta}`. Only float embeddings are served.
pub(crate) async fn cohere_embed(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let texts: Vec<String> = match body.get("texts").and_then(serde_json::Value::as_array) {
        Some(items) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        None => {
            return ApiError::Validation("`texts` must be an array of strings".into())
                .into_response()
        }
    };
    let types: Vec<String> = body
        .get("embedding_types")
        .and_then(serde_json::Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_else(|| vec!["float".to_string()]);
    if let Some(other) = types.iter().find(|t| *t != "float") {
        return ApiError::Validation(format!(
            "embedding type `{other}` is not served; ask for `float`"
        ))
        .into_response();
    }
    let adapted = serde_json::json!({
        "model": body.get("model").cloned().unwrap_or_default(),
        "input": texts,
    });
    let (status, v) = answer(openai_embeddings(State(state), headers, Json(adapted)).await).await;
    if !status.is_success() {
        return (status, Json(v)).into_response();
    }
    let vectors: Vec<serde_json::Value> = v["data"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|d| d["embedding"].clone())
        .collect();
    Json(serde_json::json!({
        "id": request_id(),
        "embeddings": {"float": vectors},
        "texts": texts,
        "meta": {"api_version": {"version": "2"}},
    }))
    .into_response()
}

/// The pairs a score request names: `text_1` against each of `text_2` when one is a string, or
/// element by element when both are lists of the same length.
pub(crate) fn score_pairs(
    first: &serde_json::Value,
    second: &serde_json::Value,
) -> Result<Vec<(String, String)>, String> {
    let list = |v: &serde_json::Value| -> Option<Vec<String>> {
        match v {
            serde_json::Value::String(s) => Some(vec![s.clone()]),
            serde_json::Value::Array(a) => {
                a.iter().map(|x| x.as_str().map(str::to_string)).collect()
            }
            _ => None,
        }
    };
    let (Some(a), Some(b)) = (list(first), list(second)) else {
        return Err("`text_1` and `text_2` must be strings or arrays of strings".into());
    };
    if a.is_empty() || b.is_empty() {
        return Err("`text_1` and `text_2` must not be empty".into());
    }
    match (a.len(), b.len()) {
        (1, _) => Ok(b.into_iter().map(|d| (a[0].clone(), d)).collect()),
        (_, 1) => Ok(a.into_iter().map(|q| (q, b[0].clone())).collect()),
        (x, y) if x == y => Ok(a.into_iter().zip(b).collect()),
        (x, y) => Err(format!(
            "`text_1` has {x} entries and `text_2` has {y}: send one of them alone, or equal lengths"
        )),
    }
}

/// vLLM `/score` and `/v1/score`: `{model, text_1, text_2}` answered with one relevance score
/// per pair, `{id, object: "list", model, data: [{index, object: "score", score}], usage}`.
pub(crate) async fn score(
    State(state): State<APIServer>,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let model = text_field(&body, "model");
    let pairs = match score_pairs(
        body.get("text_1").unwrap_or(&serde_json::Value::Null),
        body.get("text_2").unwrap_or(&serde_json::Value::Null),
    ) {
        Ok(p) => p,
        Err(why) => return ApiError::Validation(why).into_response(),
    };
    if let Err(e) = validate_model_id(&model) {
        return e.into_response();
    }
    let model_name = normalize_model_id(&model);
    let keep_alive = state.get_effective_keep_alive(None);
    if let Err(e) = state.ensure_loaded(&model_name, keep_alive).await {
        return ApiError::NotFound(format!("model '{model_name}' could not be loaded: {e}"))
            .into_response();
    }
    let Ok(engine) = state.get_engine(&model_name).await else {
        return ApiError::NotFound(format!("Model '{model_name}' not loaded")).into_response();
    };
    state.reset_expiration(&model_name).await;
    let mut data = Vec::with_capacity(pairs.len());
    let mut tokens = 0usize;
    for (i, (query, document)) in pairs.into_iter().enumerate() {
        tokens += engine.count_tokens(&query).await.unwrap_or(0)
            + engine.count_tokens(&document).await.unwrap_or(0);
        match engine.rerank_score(query, document, String::new()).await {
            Ok(s) => data.push(serde_json::json!({"index": i, "object": "score", "score": s})),
            Err(e) => {
                return ApiError::Internal(format!("score (idx={i}): {e}")).into_response();
            }
        }
    }
    Json(serde_json::json!({
        "id": format!("score-{}", request_id()),
        "object": "list",
        "model": model_name,
        "data": data,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn documents_are_strings_or_objects_with_text() {
        let got = document_texts(&json!(["a", {"text": "b"}])).unwrap();
        assert_eq!(got, ["a", "b"]);
        assert!(document_texts(&json!(["a", 3])).is_none());
        assert!(document_texts(&json!({"text": "a"})).is_none());
    }

    #[test]
    fn a_score_pairs_one_text_with_each_of_the_others() {
        let p = score_pairs(&json!("q"), &json!(["a", "b"])).unwrap();
        assert_eq!(p, [("q".into(), "a".into()), ("q".into(), "b".into())]);
        let p = score_pairs(&json!(["q1", "q2"]), &json!("d")).unwrap();
        assert_eq!(p, [("q1".into(), "d".into()), ("q2".into(), "d".into())]);
        let p = score_pairs(&json!(["q1", "q2"]), &json!(["a", "b"])).unwrap();
        assert_eq!(p, [("q1".into(), "a".into()), ("q2".into(), "b".into())]);
        assert!(score_pairs(&json!(["q1", "q2"]), &json!(["a", "b", "c"])).is_err());
        assert!(score_pairs(&json!([]), &json!("a")).is_err());
        assert!(score_pairs(&json!(1), &json!("a")).is_err());
    }
}
