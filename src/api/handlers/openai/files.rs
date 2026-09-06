//! `/v1/files`, `/v1/batches`, `/v1/messages/batches` and `/v1/moderations`: the
//! endpoints around a request rather than in it. Files live in a directory of the store,
//! one blob and one JSON record each; a batch reads a JSONL file of requests, runs them
//! one after another through the same handlers a client would call, and writes their
//! answers to another file. Moderation is a judgement by a configured model.

use super::*;
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

/// Where the files go: the configured directory, else `files/` beside the model store.
fn files_dir(state: &APIServer) -> PathBuf {
    state
        .default_inference_config
        .files_dir
        .as_ref()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(&state.ollama_models_dir).join("files"))
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

fn record_path(dir: &std::path::Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.json"))
}

fn blob_path(dir: &std::path::Path, id: &str) -> PathBuf {
    dir.join(format!("{id}.bin"))
}

/// Writes `bytes` as a file object; returns the record.
pub(crate) fn store_file(
    state: &APIServer,
    filename: &str,
    purpose: &str,
    bytes: &[u8],
) -> Result<Value, ApiError> {
    let dir = files_dir(state);
    std::fs::create_dir_all(&dir).map_err(|e| ApiError::Internal(format!("files: {e}")))?;
    let id = new_id("file");
    std::fs::write(blob_path(&dir, &id), bytes).map_err(|e| ApiError::Internal(format!("files: {e}")))?;
    let record = json!({
        "id": id,
        "object": "file",
        "bytes": bytes.len(),
        "created_at": now(),
        "filename": filename,
        "purpose": purpose,
        "status": "processed",
    });
    std::fs::write(record_path(&dir, &id), record.to_string())
        .map_err(|e| ApiError::Internal(format!("files: {e}")))?;
    Ok(record)
}

fn read_record(state: &APIServer, id: &str) -> Option<Value> {
    if !id.starts_with("file_") || id.contains('/') || id.contains("..") {
        return None;
    }
    let bytes = std::fs::read(record_path(&files_dir(state), id)).ok()?;
    serde_json::from_slice(&bytes).ok()
}

pub(crate) fn read_file_bytes(state: &APIServer, id: &str) -> Option<Vec<u8>> {
    read_record(state, id)?;
    std::fs::read(blob_path(&files_dir(state), id)).ok()
}

/// `POST /v1/files`: multipart with `file` and `purpose`.
pub(crate) async fn files_upload(
    State(state): State<APIServer>,
    mut multipart: axum::extract::Multipart,
) -> Result<Json<Value>, ApiError> {
    let mut purpose = "user_data".to_string();
    let mut filename = "upload".to_string();
    let mut bytes: Option<Vec<u8>> = None;
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::Validation(format!("multipart: {e}")))?
    {
        match field.name().unwrap_or("") {
            "purpose" => purpose = field.text().await.unwrap_or_default(),
            "file" => {
                if let Some(n) = field.file_name() {
                    filename = n.to_string();
                }
                bytes = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| ApiError::Validation(format!("file: {e}")))?
                        .to_vec(),
                );
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }
    let bytes = bytes.ok_or_else(|| ApiError::Validation("`file` is required".into()))?;
    Ok(Json(store_file(&state, &filename, &purpose, &bytes)?))
}

/// `GET /v1/files`
pub(crate) async fn files_list(State(state): State<APIServer>) -> Json<Value> {
    let mut data = Vec::new();
    if let Ok(entries) = std::fs::read_dir(files_dir(&state)) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) == Some("json") {
                if let Ok(Ok(v)) = std::fs::read(&p).map(|b| serde_json::from_slice::<Value>(&b)) {
                    data.push(v);
                }
            }
        }
    }
    data.sort_by_key(|v| -v.get("created_at").and_then(Value::as_i64).unwrap_or(0));
    Json(json!({"object": "list", "data": data, "has_more": false}))
}

/// `GET /v1/files/{id}`
pub(crate) async fn files_get(
    State(state): State<APIServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Value>, ApiError> {
    read_record(&state, &id)
        .map(Json)
        .ok_or_else(|| ApiError::NotFound(format!("file '{id}' not found")))
}

/// `GET /v1/files/{id}/content`
pub(crate) async fn files_content(
    State(state): State<APIServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Response, ApiError> {
    let record = read_record(&state, &id)
        .ok_or_else(|| ApiError::NotFound(format!("file '{id}' not found")))?;
    let bytes = read_file_bytes(&state, &id)
        .ok_or_else(|| ApiError::NotFound(format!("file '{id}' has no content")))?;
    // The upload named the file; a header must not carry quotes, separators or control
    // characters from it, so the name is reduced to a safe charset before it is echoed.
    let raw = record.get("filename").and_then(Value::as_str).unwrap_or("file");
    let safe: String = raw
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .take(128)
        .collect();
    let name = if safe.is_empty() { "file" } else { safe.as_str() };
    let mime = match name.rsplit('.').next().map(str::to_ascii_lowercase).as_deref() {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        Some("jsonl") => "application/jsonl",
        Some("json") => "application/json",
        Some("txt") | Some("md") => "text/plain; charset=utf-8",
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        _ => "application/octet-stream",
    };
    Ok(Response::builder()
        .header("content-type", mime)
        .header("content-disposition", format!("attachment; filename=\"{name}\""))
        .body(axum::body::Body::from(bytes))
        .unwrap())
}

/// `DELETE /v1/files/{id}`
pub(crate) async fn files_delete(
    State(state): State<APIServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Value>, ApiError> {
    read_record(&state, &id).ok_or_else(|| ApiError::NotFound(format!("file '{id}' not found")))?;
    let dir = files_dir(&state);
    let _ = std::fs::remove_file(record_path(&dir, &id));
    let _ = std::fs::remove_file(blob_path(&dir, &id));
    Ok(Json(json!({"id": id, "object": "file", "deleted": true})))
}

// ---------------------------------------------------------------- batches

#[derive(Clone)]
struct Batch {
    object: Value,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// The most requests one batch may hold: the limit the Batch API documents, so a client
/// written against it never trips here first.
const BATCH_MAX_REQUESTS: usize = 50_000;
/// How long a finished batch stays listable: the API's 24-hour completion window.
const BATCH_RETENTION_SECS: i64 = 24 * 60 * 60;

/// Drops finished batches past the retention window so the table cannot grow forever.
fn evict_finished_batches() {
    let cutoff = now() - BATCH_RETENTION_SECS;
    if let Ok(mut g) = batches().lock() {
        g.retain(|_, b| {
            let done = matches!(
                b.object.get("status").and_then(Value::as_str),
                Some("completed") | Some("cancelled") | Some("failed")
            ) || b.object.get("processing_status").and_then(Value::as_str) == Some("ended");
            // OpenAI batches stamp an epoch, Messages batches an RFC 3339 string.
            let created = match b.object.get("created_at") {
                Some(Value::Number(n)) => n.as_i64().unwrap_or(i64::MAX),
                Some(Value::String(t)) => chrono::DateTime::parse_from_rfc3339(t)
                    .map(|d| d.timestamp())
                    .unwrap_or(i64::MAX),
                _ => i64::MAX,
            };
            !(done && created < cutoff)
        });
    }
}

fn batches() -> &'static Mutex<HashMap<String, Batch>> {
    static B: OnceLock<Mutex<HashMap<String, Batch>>> = OnceLock::new();
    B.get_or_init(|| Mutex::new(HashMap::new()))
}

fn batch_get(id: &str) -> Option<Batch> {
    batches().lock().ok()?.get(id).cloned()
}

fn batch_update(id: &str, f: impl FnOnce(&mut Value)) {
    if let Ok(mut g) = batches().lock() {
        if let Some(b) = g.get_mut(id) {
            f(&mut b.object);
        }
    }
}

/// One request line of a batch, run through the handler its `url` names.
async fn run_batch_line(state: APIServer, url: &str, body: Value) -> (u16, Value) {
    let resp: Result<Response, ApiError> = match url {
        "/v1/chat/completions" => match serde_json::from_value::<ChatCompletionRequest>(body) {
            Ok(req) => chat_completion(State(state), OpenAIJson(req)).await,
            Err(e) => Err(ApiError::Validation(e.to_string())),
        },
        "/v1/completions" => text_completions(State(state), Json(body)).await,
        "/v1/responses" => openai_responses(State(state), OpenAIJson(body)).await,
        "/v1/embeddings" => Ok(openai_embeddings(State(state), Json(body)).await),
        other => Err(ApiError::Validation(format!("batch endpoint '{other}' is not served"))),
    };
    match resp {
        Ok(r) => {
            let status = r.status().as_u16();
            let bytes = axum::body::to_bytes(r.into_body(), usize::MAX).await.unwrap_or_default();
            let body = serde_json::from_slice::<Value>(&bytes)
                .unwrap_or_else(|_| json!(String::from_utf8_lossy(&bytes)));
            (status, body)
        }
        Err(e) => {
            let r = e.into_response();
            let status = r.status().as_u16();
            let bytes = axum::body::to_bytes(r.into_body(), usize::MAX).await.unwrap_or_default();
            (status, serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null))
        }
    }
}

/// `POST /v1/batches`: `{input_file_id, endpoint, completion_window, metadata}`.
pub(crate) async fn batches_create(
    State(state): State<APIServer>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    let input_id = body
        .get("input_file_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::Validation("`input_file_id` is required".into()))?
        .to_string();
    let endpoint = body
        .get("endpoint")
        .and_then(Value::as_str)
        .unwrap_or("/v1/chat/completions")
        .to_string();
    let input = read_file_bytes(&state, &input_id)
        .ok_or_else(|| ApiError::NotFound(format!("input file '{input_id}' not found")))?;
    let lines: Vec<Value> = String::from_utf8_lossy(&input)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Value>(l).unwrap_or_else(|e| json!({"parse_error": e.to_string()})))
        .collect();
    if lines.len() > BATCH_MAX_REQUESTS {
        return Err(ApiError::Validation(format!(
            "batch holds {} requests; at most {BATCH_MAX_REQUESTS} are served",
            lines.len()
        )));
    }
    evict_finished_batches();
    let id = new_id("batch");
    let object = json!({
        "id": id,
        "object": "batch",
        "endpoint": endpoint,
        "input_file_id": input_id,
        "completion_window": body.get("completion_window").cloned().unwrap_or(json!("24h")),
        "status": "in_progress",
        "output_file_id": Value::Null,
        "error_file_id": Value::Null,
        "created_at": now(),
        "in_progress_at": now(),
        "completed_at": Value::Null,
        "request_counts": {"total": lines.len(), "completed": 0, "failed": 0},
        "metadata": body.get("metadata").cloned().unwrap_or(json!({})),
    });
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    if let Ok(mut g) = batches().lock() {
        g.insert(id.clone(), Batch { object: object.clone(), cancel: cancel.clone() });
    }
    let runner_state = state.clone();
    let runner_id = id.clone();
    tokio::spawn(async move {
        let mut out = String::new();
        let mut errs = String::new();
        let mut completed = 0usize;
        let mut failed = 0usize;
        for line in lines {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }
            let custom_id = line.get("custom_id").cloned().unwrap_or(Value::Null);
            let url = line
                .get("url")
                .and_then(Value::as_str)
                .unwrap_or(&endpoint)
                .to_string();
            let req_body = line.get("body").cloned().unwrap_or(Value::Null);
            let (status, resp) = run_batch_line(runner_state.clone(), &url, req_body).await;
            let ok = (200..300).contains(&status);
            let record = json!({
                "id": new_id("batch_req"),
                "custom_id": custom_id,
                "response": {"status_code": status, "request_id": new_id("req"), "body": resp},
                "error": Value::Null,
            });
            if ok {
                completed += 1;
                out.push_str(&record.to_string());
                out.push('\n');
            } else {
                failed += 1;
                errs.push_str(&record.to_string());
                errs.push('\n');
            }
            batch_update(&runner_id, |o| {
                o["request_counts"]["completed"] = json!(completed);
                o["request_counts"]["failed"] = json!(failed);
            });
        }
        let output = store_file(&runner_state, "batch_output.jsonl", "batch_output", out.as_bytes()).ok();
        let errors = if errs.is_empty() {
            None
        } else {
            store_file(&runner_state, "batch_errors.jsonl", "batch_output", errs.as_bytes()).ok()
        };
        let cancelled = cancel.load(std::sync::atomic::Ordering::Relaxed);
        batch_update(&runner_id, |o| {
            o["status"] = json!(if cancelled { "cancelled" } else { "completed" });
            o["completed_at"] = json!(now());
            o["output_file_id"] = output.as_ref().and_then(|f| f.get("id").cloned()).unwrap_or(Value::Null);
            o["error_file_id"] = errors.as_ref().and_then(|f| f.get("id").cloned()).unwrap_or(Value::Null);
        });
    });
    Ok(Json(object))
}

pub(crate) async fn batches_get(
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Value>, ApiError> {
    batch_get(&id)
        .map(|b| Json(b.object))
        .ok_or_else(|| ApiError::NotFound(format!("batch '{id}' not found")))
}

pub(crate) async fn batches_list() -> Json<Value> {
    let mut data: Vec<Value> = batches()
        .lock()
        .map(|g| g.values().map(|b| b.object.clone()).collect())
        .unwrap_or_default();
    data.sort_by_key(|v| -v.get("created_at").and_then(Value::as_i64).unwrap_or(0));
    Json(json!({"object": "list", "data": data, "has_more": false}))
}

pub(crate) async fn batches_cancel(
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Result<Json<Value>, ApiError> {
    let b = batch_get(&id).ok_or_else(|| ApiError::NotFound(format!("batch '{id}' not found")))?;
    b.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    batch_update(&id, |o| {
        if o["status"] == "in_progress" {
            o["status"] = json!("cancelling");
        }
    });
    Ok(Json(batch_get(&id).map(|b| b.object).unwrap_or(Value::Null)))
}

// ------------------------------------------------- Anthropic message batches

/// `POST /v1/messages/batches`: `{requests: [{custom_id, params}]}`.
pub(crate) async fn anthropic_batches_create(
    State(state): State<APIServer>,
    Json(body): Json<Value>,
) -> Response {
    let requests: Vec<Value> = body
        .get("requests")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if requests.is_empty() {
        return super::super::anthropic_api::anthropic_error_response(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "`requests` must hold at least one entry".into(),
        );
    }
    let id = new_id("msgbatch");
    let object = json!({
        "id": id,
        "type": "message_batch",
        "processing_status": "in_progress",
        "request_counts": {"processing": requests.len(), "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
        "ended_at": Value::Null,
        "created_at": chrono::Utc::now().to_rfc3339(),
        "expires_at": (chrono::Utc::now() + chrono::Duration::hours(24)).to_rfc3339(),
        "cancel_initiated_at": Value::Null,
        "results_url": format!("/v1/messages/batches/{id}/results"),
    });
    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    if let Ok(mut g) = batches().lock() {
        g.insert(id.clone(), Batch { object: object.clone(), cancel: cancel.clone() });
    }
    let runner_state = state.clone();
    let runner_id = id.clone();
    tokio::spawn(async move {
        let mut out = String::new();
        let (mut succeeded, mut errored, mut canceled) = (0usize, 0usize, 0usize);
        let total = requests.len();
        for r in requests {
            let custom_id = r.get("custom_id").cloned().unwrap_or(Value::Null);
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                canceled += 1;
                out.push_str(&json!({"custom_id": custom_id, "result": {"type": "canceled"}}).to_string());
                out.push('\n');
                continue;
            }
            let params = r.get("params").cloned().unwrap_or(Value::Null);
            let result = match serde_json::from_value::<crate::api::anthropic::AnthropicMessagesRequest>(params) {
                Ok(req) => {
                    let resp = super::super::anthropic_api::anthropic_messages(State(runner_state.clone()), AnthropicJson(req)).await;
                    let status = resp.status().as_u16();
                    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap_or_default();
                    let body = serde_json::from_slice::<Value>(&bytes).unwrap_or(Value::Null);
                    if (200..300).contains(&status) {
                        succeeded += 1;
                        json!({"type": "succeeded", "message": body})
                    } else {
                        errored += 1;
                        json!({"type": "errored", "error": body})
                    }
                }
                Err(e) => {
                    errored += 1;
                    json!({"type": "errored", "error": {"type": "error", "error": {"type": "invalid_request_error", "message": e.to_string()}}})
                }
            };
            out.push_str(&json!({"custom_id": custom_id, "result": result}).to_string());
            out.push('\n');
            batch_update(&runner_id, |o| {
                o["request_counts"] = json!({"processing": total - succeeded - errored - canceled, "succeeded": succeeded, "errored": errored, "canceled": canceled, "expired": 0});
            });
        }
        let results = store_file(&runner_state, "batch_results.jsonl", "batch_output", out.as_bytes()).ok();
        batch_update(&runner_id, |o| {
            o["processing_status"] = json!("ended");
            o["ended_at"] = json!(chrono::Utc::now().to_rfc3339());
            o["request_counts"]["processing"] = json!(0);
            if let Some(f) = results.as_ref().and_then(|f| f.get("id").cloned()) {
                o["results_file_id"] = f;
            }
        });
    });
    Json(object).into_response()
}

pub(crate) async fn anthropic_batches_get(
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match batch_get(&id) {
        Some(b) => Json(b.object).into_response(),
        None => super::super::anthropic_api::anthropic_error_response(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("batch '{id}' not found"),
        ),
    }
}

pub(crate) async fn anthropic_batches_list() -> Json<Value> {
    let data: Vec<Value> = batches()
        .lock()
        .map(|g| g.values().filter(|b| b.object["type"] == "message_batch").map(|b| b.object.clone()).collect())
        .unwrap_or_default();
    Json(json!({"data": data, "has_more": false, "first_id": data.first().and_then(|v| v.get("id").cloned()), "last_id": data.last().and_then(|v| v.get("id").cloned())}))
}

pub(crate) async fn anthropic_batches_cancel(
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    match batch_get(&id) {
        Some(b) => {
            b.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
            batch_update(&id, |o| {
                if o["processing_status"] == "in_progress" {
                    o["processing_status"] = json!("canceling");
                    o["cancel_initiated_at"] = json!(chrono::Utc::now().to_rfc3339());
                }
            });
            Json(batch_get(&id).map(|b| b.object).unwrap_or(Value::Null)).into_response()
        }
        None => super::super::anthropic_api::anthropic_error_response(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("batch '{id}' not found"),
        ),
    }
}

/// `GET /v1/messages/batches/{id}/results`: the JSONL of results, once ended.
pub(crate) async fn anthropic_batches_results(
    State(state): State<APIServer>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let Some(b) = batch_get(&id) else {
        return super::super::anthropic_api::anthropic_error_response(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("batch '{id}' not found"),
        );
    };
    let Some(file) = b.object.get("results_file_id").and_then(Value::as_str) else {
        return super::super::anthropic_api::anthropic_error_response(
            StatusCode::NOT_FOUND,
            "not_found_error",
            format!("batch '{id}' has not ended"),
        );
    };
    match read_file_bytes(&state, file) {
        Some(bytes) => Response::builder()
            .header("content-type", "application/x-jsonl")
            .body(axum::body::Body::from(bytes))
            .unwrap(),
        None => super::super::anthropic_api::anthropic_error_response(
            StatusCode::NOT_FOUND,
            "not_found_error",
            "results file missing".into(),
        ),
    }
}

// --------------------------------------------------------------- moderations

const MODERATION_CATEGORIES: [&str; 13] = [
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
];

/// `POST /v1/moderations`: each input judged by the configured moderation model, which
/// answers the category flags as JSON under a grammar. Without one, 501 and the key
/// to set.
pub(crate) async fn moderations(
    State(state): State<APIServer>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let Some(model) = state
        .default_inference_config
        .moderation_model
        .clone()
        .or_else(|| body.get("model").and_then(Value::as_str).map(str::to_string))
    else {
        return Ok((
            StatusCode::NOT_IMPLEMENTED,
            Json(openai_error_body(
                StatusCode::NOT_IMPLEMENTED,
                "no moderation model is configured; set `[inference] moderation_model` to the model that judges, or pass `model`".to_string(),
            )),
        )
            .into_response());
    };
    let inputs: Vec<String> = match body.get("input") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
        _ => return Err(ApiError::Validation("`input` must be a string or an array of strings".into())),
    };
    let schema = json!({
        "type": "object",
        "properties": MODERATION_CATEGORIES.iter().map(|c| (c.to_string(), json!({"type": "number", "minimum": 0, "maximum": 1}))).collect::<serde_json::Map<_, _>>(),
        "required": MODERATION_CATEGORIES,
        "additionalProperties": false
    });
    let mut results = Vec::new();
    for text in inputs {
        let req: ChatCompletionRequest = serde_json::from_value(json!({
            "model": model,
            "messages": [
                {"role": "system", "content": format!("You are a content moderation classifier. For the user's text, give for each category a score from 0 to 1, the probability that the text belongs to it. Categories: {}. Answer with the JSON object only.", MODERATION_CATEGORIES.join(", "))},
                {"role": "user", "content": text}
            ],
            "temperature": 0,
            "max_completion_tokens": 400,
            "response_format": {"type": "json_schema", "json_schema": {"name": "moderation", "schema": schema}}
        }))
        .map_err(|e| ApiError::Internal(format!("moderation: {e}")))?;
        let resp = Box::pin(chat_completion(State(state.clone()), OpenAIJson(req))).await?;
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .map_err(|e| ApiError::Internal(format!("moderation: {e}")))?;
        let v: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        let answer = v
            .pointer("/choices/0/message/content")
            .and_then(Value::as_str)
            .and_then(|s| serde_json::from_str::<Value>(s).ok())
            .unwrap_or(json!({}));
        let scores: serde_json::Map<String, Value> = MODERATION_CATEGORIES
            .iter()
            .map(|c| (c.to_string(), json!(answer.get(*c).and_then(Value::as_f64).unwrap_or(0.0))))
            .collect();
        let categories: serde_json::Map<String, Value> = scores
            .iter()
            .map(|(k, v)| (k.clone(), json!(v.as_f64().unwrap_or(0.0) >= 0.5)))
            .collect();
        let flagged = categories.values().any(|v| v.as_bool().unwrap_or(false));
        results.push(json!({"flagged": flagged, "categories": categories, "category_scores": scores}));
    }
    Ok(Json(json!({"id": new_id("modr"), "model": model, "results": results})).into_response())
}

/// Turns `b64_json` image entries into stored files served by URL, when the request
/// asked for `url`. The URL is built on the request's `Host`, or on `public_url` when
/// the configuration names one.
pub(crate) fn images_as_urls(
    state: &APIServer,
    headers: &axum::http::HeaderMap,
    wanted: bool,
    data: Vec<Value>,
) -> Result<Vec<Value>, ApiError> {
    if !wanted {
        return Ok(data);
    }
    let base = state
        .default_inference_config
        .public_url
        .clone()
        .or_else(|| {
            // `Host` is client-supplied. Without a configured `public_url` it is the only
            // origin to hand back, so it is accepted only when it looks like a host and
            // port, never as arbitrary text that would land in every URL of the answer.
            headers
                .get(axum::http::header::HOST)
                .and_then(|h| h.to_str().ok())
                .filter(|h| {
                    !h.is_empty()
                        && h.len() <= 255
                        && h.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | ':' | '[' | ']'))
                })
                .map(|h| format!("http://{h}"))
        })
        .unwrap_or_default();
    let mut out = Vec::with_capacity(data.len());
    for mut entry in data {
        let Some(b64) = entry.get("b64_json").and_then(Value::as_str) else {
            out.push(entry);
            continue;
        };
        use base64::Engine as _;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| ApiError::Internal(format!("image: {e}")))?;
        let ext = match entry.get("content_type").and_then(Value::as_str) {
            Some("image/jpeg") => "jpg",
            Some("image/webp") => "webp",
            Some("image/gif") => "gif",
            _ => "png",
        };
        let record = store_file(state, &format!("image.{ext}"), "generated", &bytes)?;
        let id = record.get("id").and_then(Value::as_str).unwrap_or("").to_string();
        if let Some(obj) = entry.as_object_mut() {
            obj.remove("b64_json");
            obj.insert("url".to_string(), json!(format!("{base}/v1/files/{id}/content")));
        }
        out.push(entry);
    }
    Ok(out)
}

/// OpenAI's `style` as words the generator reads, appended to the prompt.
pub(crate) fn with_style(prompt: String, style: Option<&str>) -> String {
    match style.map(str::to_ascii_lowercase).as_deref() {
        Some("vivid") => format!("{prompt}, vivid, dramatic lighting, saturated color"),
        Some("natural") => format!("{prompt}, natural, realistic, understated"),
        _ => prompt,
    }
}
