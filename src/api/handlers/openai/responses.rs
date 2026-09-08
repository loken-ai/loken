//! `POST /v1/responses`, the Responses API, as an adapter over `/v1/chat/completions`:
//! the request is lowered to a chat completion, the completion lifted back into
//! output items, whole or as the Responses event stream. A bounded store keeps the
//! conversation of each response so `previous_response_id` can continue it.

use super::*;
use axum::response::sse::{Event, KeepAlive, Sse};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};

/// Responses kept for `previous_response_id`, newest last.
const STORE_CAPACITY: usize = 256;

struct Stored {
    id: String,
    messages: Vec<Value>,
    response: Value,
}

fn store() -> &'static Mutex<VecDeque<Stored>> {
    static STORE: OnceLock<Mutex<VecDeque<Stored>>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(VecDeque::new()))
}

fn store_response(id: &str, messages: Vec<Value>, response: Value) {
    if let Ok(mut g) = store().lock() {
        g.retain(|s| s.id != id);
        if g.len() >= STORE_CAPACITY {
            g.pop_front();
        }
        g.push_back(Stored {
            id: id.to_string(),
            messages,
            response,
        });
    }
}

fn stored_messages(id: &str) -> Option<Vec<Value>> {
    store()
        .lock()
        .ok()?
        .iter()
        .find(|s| s.id == id)
        .map(|s| s.messages.clone())
}

fn error(status: StatusCode, message: String) -> Response {
    (status, Json(openai_error_body(status, message))).into_response()
}

/// A Responses `content` value as chat completion content: a string stays one, an
/// array of input parts becomes an array of chat parts.
fn lower_content(content: &Value) -> Result<Value, String> {
    match content {
        Value::String(_) | Value::Null => Ok(content.clone()),
        Value::Array(parts) => {
            let mut out = Vec::new();
            for p in parts {
                let kind = p.get("type").and_then(Value::as_str).unwrap_or("");
                match kind {
                    "input_text" | "output_text" | "text" => out.push(json!({
                        "type": "text",
                        "text": p.get("text").and_then(Value::as_str).unwrap_or("")
                    })),
                    "input_image" => {
                        let url = match p.get("image_url") {
                            Some(Value::String(u)) => u.clone(),
                            Some(Value::Object(o)) => o
                                .get("url")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            _ => String::new(),
                        };
                        out.push(json!({"type": "image_url", "image_url": {"url": url}}));
                    }
                    other => return Err(format!("unsupported input part type '{other}'")),
                }
            }
            Ok(Value::Array(out))
        }
        _ => Err("`content` must be a string or an array of parts".to_string()),
    }
}

/// The chat messages a Responses request stands for.
fn lower_input(req: &Value) -> Result<Vec<Value>, String> {
    let mut messages: Vec<Value> = Vec::new();
    if let Some(prev) = req.get("previous_response_id").and_then(Value::as_str) {
        match stored_messages(prev) {
            Some(m) => messages.extend(m),
            None => return Err(format!("previous_response_id '{prev}' is not known")),
        }
    }
    if let Some(instr) = req.get("instructions").and_then(Value::as_str) {
        messages.push(json!({"role": "system", "content": instr}));
    }
    match req.get("input") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => messages.push(json!({"role": "user", "content": s})),
        Some(Value::Array(items)) => {
            for item in items {
                let kind = item
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("message");
                match kind {
                    "message" => {
                        let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                        let content = lower_content(item.get("content").unwrap_or(&Value::Null))?;
                        messages.push(json!({"role": role, "content": content}));
                    }
                    "function_call" => {
                        let call = json!({
                            "id": item.get("call_id").cloned().unwrap_or(Value::Null),
                            "type": "function",
                            "function": {
                                "name": item.get("name").cloned().unwrap_or(Value::Null),
                                "arguments": item.get("arguments").cloned().unwrap_or(json!("{}")),
                            }
                        });
                        // Consecutive calls belong to one assistant turn.
                        let merged = messages
                            .last_mut()
                            .filter(|m| m["role"] == "assistant" && m.get("tool_calls").is_some())
                            .and_then(|m| m["tool_calls"].as_array_mut())
                            .map(|calls| calls.push(call.clone()))
                            .is_some();
                        if !merged {
                            messages.push(
                                json!({"role": "assistant", "content": "", "tool_calls": [call]}),
                            );
                        }
                    }
                    "function_call_output" => {
                        let output = match item.get("output") {
                            Some(Value::String(s)) => s.clone(),
                            Some(v) => v.to_string(),
                            None => String::new(),
                        };
                        messages.push(json!({
                            "role": "tool",
                            "tool_call_id": item.get("call_id").cloned().unwrap_or(Value::Null),
                            "content": output,
                        }));
                    }
                    "reasoning" => {}
                    other => return Err(format!("unsupported input item type '{other}'")),
                }
            }
        }
        Some(_) => return Err("`input` must be a string or an array of items".to_string()),
    }
    Ok(messages)
}

/// Responses tools are flat; chat completion tools nest the function.
fn lower_tools(req: &Value) -> Result<Option<Vec<Value>>, String> {
    let Some(tools) = req.get("tools").and_then(Value::as_array) else {
        return Ok(None);
    };
    let mut out = Vec::new();
    for t in tools {
        match t.get("type").and_then(Value::as_str).unwrap_or("") {
            "function" => out.push(json!({
                "type": "function",
                "function": {
                    "name": t.get("name").cloned().unwrap_or(Value::Null),
                    "description": t.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": t.get("parameters").cloned().unwrap_or(Value::Null),
                }
            })),
            other => return Err(format!("unsupported tool type '{other}'")),
        }
    }
    Ok(Some(out))
}

fn lower_tool_choice(v: Option<&Value>) -> Value {
    match v {
        Some(Value::Object(o)) if o.get("type").and_then(Value::as_str) == Some("function") => {
            json!({"type": "function", "function": {"name": o.get("name").cloned().unwrap_or(Value::Null)}})
        }
        Some(v) => v.clone(),
        None => Value::Null,
    }
}

fn lower_text_format(req: &Value) -> Value {
    let Some(f) = req.get("text").and_then(|t| t.get("format")) else {
        return Value::Null;
    };
    match f.get("type").and_then(Value::as_str) {
        Some("json_schema") => json!({
            "type": "json_schema",
            "json_schema": {
                "name": f.get("name").cloned().unwrap_or(json!("response")),
                "schema": f.get("schema").cloned().unwrap_or(Value::Null),
                "strict": f.get("strict").cloned().unwrap_or(Value::Null),
            }
        }),
        Some("json_object") => json!({"type": "json_object"}),
        _ => Value::Null,
    }
}

/// One Responses event: the body stamped with its type and sequence number.
fn event(kind: &str, mut body: Value, seq: &mut u64) -> Event {
    body["type"] = json!(kind);
    body["sequence_number"] = json!(*seq);
    *seq += 1;
    Event::default().event(kind).data(body.to_string())
}

fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

/// The output items of a finished chat message: reasoning, text, then the calls.
fn output_items(message: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(r) = message.get("reasoning_content").and_then(Value::as_str) {
        if !r.is_empty() {
            out.push(json!({
                "type": "reasoning",
                "id": new_id("rs"),
                "summary": [{"type": "summary_text", "text": r}],
            }));
        }
    }
    if let Some(text) = message.get("content").and_then(Value::as_str) {
        if !text.is_empty() {
            out.push(json!({
                "type": "message",
                "id": new_id("msg"),
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }));
        }
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            out.push(json!({
                "type": "function_call",
                "id": new_id("fc"),
                "call_id": c.get("id").cloned().unwrap_or(Value::Null),
                "name": c.pointer("/function/name").cloned().unwrap_or(Value::Null),
                "arguments": c.pointer("/function/arguments").cloned().unwrap_or(json!("{}")),
                "status": "completed",
            }));
        }
    }
    out
}

fn usage_object(usage: Option<&Value>) -> Value {
    let input = usage
        .and_then(|u| u.get("prompt_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let output = usage
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    json!({
        "input_tokens": input,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input + output,
    })
}

/// The response object, with `status` and `output` as far as they are known.
#[allow(clippy::too_many_arguments)]
fn response_object(
    id: &str,
    created: i64,
    req: &Value,
    model: &str,
    status: &str,
    output: Vec<Value>,
    usage: Value,
    incomplete: Option<&str>,
) -> Value {
    json!({
        "id": id,
        "object": "response",
        "created_at": created,
        "status": status,
        "error": Value::Null,
        "incomplete_details": incomplete.map(|r| json!({"reason": r})),
        "instructions": req.get("instructions").cloned().unwrap_or(Value::Null),
        "max_output_tokens": req.get("max_output_tokens").cloned().unwrap_or(Value::Null),
        "model": model,
        "output": output,
        "parallel_tool_calls": req.get("parallel_tool_calls").cloned().unwrap_or(json!(true)),
        "previous_response_id": req.get("previous_response_id").cloned().unwrap_or(Value::Null),
        "reasoning": req.get("reasoning").cloned().unwrap_or(json!({"effort": Value::Null, "summary": Value::Null})),
        "store": req.get("store").cloned().unwrap_or(json!(true)),
        "temperature": req.get("temperature").cloned().unwrap_or(Value::Null),
        "text": req.get("text").cloned().unwrap_or(json!({"format": {"type": "text"}})),
        "tool_choice": req.get("tool_choice").cloned().unwrap_or(json!("auto")),
        "tools": req.get("tools").cloned().unwrap_or(json!([])),
        "top_p": req.get("top_p").cloned().unwrap_or(Value::Null),
        "truncation": req.get("truncation").cloned().unwrap_or(json!("disabled")),
        "usage": usage,
        "user": req.get("user").cloned().unwrap_or(Value::Null),
        "metadata": req.get("metadata").cloned().unwrap_or(json!({})),
    })
}

pub(crate) async fn openai_responses(
    State(state): State<APIServer>,
    OpenAIJson(req): OpenAIJson<Value>,
) -> Result<Response, ApiError> {
    let model = req
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::Validation("`model` is required".to_string()))?
        .to_string();
    let messages = lower_input(&req).map_err(ApiError::Validation)?;
    if messages.is_empty() {
        return Err(ApiError::Validation("`input` is empty".to_string()));
    }
    let tools = lower_tools(&req).map_err(ApiError::Validation)?;
    let stream = req.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let keep = req.get("store").and_then(Value::as_bool).unwrap_or(true);
    let mut chat = json!({
        "model": model,
        "messages": messages,
        "stream": stream,
        "stream_options": {"include_usage": true},
        "max_completion_tokens": req.get("max_output_tokens").cloned().unwrap_or(Value::Null),
        "temperature": req.get("temperature").cloned().unwrap_or(Value::Null),
        "top_p": req.get("top_p").cloned().unwrap_or(Value::Null),
        "user": req.get("user").cloned().unwrap_or(Value::Null),
        "tool_choice": lower_tool_choice(req.get("tool_choice")),
        "response_format": lower_text_format(&req),
    });
    if let Some(t) = tools {
        chat["tools"] = Value::Array(t);
    }
    if let Some(obj) = chat.as_object_mut() {
        obj.retain(|_, v| !v.is_null());
    }
    let chat_req: ChatCompletionRequest = serde_json::from_value(chat)
        .map_err(|e| ApiError::Validation(format!("responses: {e}")))?;
    let id = new_id("resp");
    let created = chrono::Utc::now().timestamp();
    let history = messages.clone();
    let inner = chat_completion(State(state), OpenAIJson(chat_req)).await?;
    if !inner.status().is_success() {
        return Ok(inner);
    }
    if !stream {
        let bytes = axum::body::to_bytes(inner.into_body(), usize::MAX)
            .await
            .map_err(|e| ApiError::Internal(format!("responses: {e}")))?;
        let completion: Value = serde_json::from_slice(&bytes)
            .map_err(|e| ApiError::Internal(format!("responses: {e}")))?;
        let choice = completion
            .pointer("/choices/0")
            .cloned()
            .unwrap_or(Value::Null);
        let message = choice.get("message").cloned().unwrap_or(Value::Null);
        let finish = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .unwrap_or("stop");
        let (status, incomplete) = if finish == "length" {
            ("incomplete", Some("max_output_tokens"))
        } else {
            ("completed", None)
        };
        let output = output_items(&message);
        let body = response_object(
            &id,
            created,
            &req,
            &model,
            status,
            output,
            usage_object(completion.get("usage")),
            incomplete,
        );
        if keep {
            let mut convo = history;
            convo.push(message);
            store_response(&id, convo, body.clone());
        }
        return Ok(Json(body).into_response());
    }

    // Streamed: the chat completion's SSE frames become Responses events.
    use futures::StreamExt;
    let mut frames = inner.into_body().into_data_stream();
    let req_for_stream = req.clone();
    let events = async_stream::stream! {
        let mut seq = 0u64;
        let pending = response_object(&id, created, &req_for_stream, &model, "in_progress", vec![], usage_object(None), None);
        yield Ok::<_, axum::Error>(event("response.created", json!({"response": pending}), &mut seq));
        yield Ok::<_, axum::Error>(event("response.in_progress", json!({"response": pending}), &mut seq));
        let mut output: Vec<Value> = Vec::new();
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut msg_id: Option<String> = None;
        let mut rs_id: Option<String> = None;
        let mut finish = "stop".to_string();
        let mut usage = Value::Null;
        let mut buf = String::new();
        let mut tool_calls: Vec<Value> = Vec::new();
        'frames: while let Some(chunk) = frames.next().await {
            let Ok(bytes) = chunk else { break };
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(nl) = buf.find('\n') {
                let line = buf[..nl].trim().to_string();
                buf = buf[nl + 1..].to_string();
                let Some(data) = line.strip_prefix("data:") else { continue };
                let data = data.trim();
                if data == "[DONE]" {
                    break 'frames;
                }
                let Ok(v) = serde_json::from_str::<Value>(data) else { continue };
                if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                    usage = u.clone();
                }
                let Some(choice) = v.pointer("/choices/0") else { continue };
                if let Some(f) = choice.get("finish_reason").and_then(Value::as_str) {
                    finish = f.to_string();
                }
                let delta = choice.get("delta").cloned().unwrap_or(Value::Null);
                if let Some(r) = delta.get("reasoning_content").and_then(Value::as_str) {
                    if rs_id.is_none() {
                        let rid = new_id("rs");
                        yield Ok::<_, axum::Error>(event("response.output_item.added", json!({"output_index": output.len(), "item": {"type": "reasoning", "id": rid, "summary": []}}), &mut seq));
                        yield Ok::<_, axum::Error>(event("response.reasoning_summary_part.added", json!({"item_id": rid, "output_index": output.len(), "summary_index": 0, "part": {"type": "summary_text", "text": ""}}), &mut seq));
                        rs_id = Some(rid);
                    }
                    reasoning.push_str(r);
                    yield Ok::<_, axum::Error>(event("response.reasoning_summary_text.delta", json!({"item_id": rs_id.clone(), "output_index": output.len(), "summary_index": 0, "delta": r}), &mut seq));
                }
                if let Some(c) = delta.get("content").and_then(Value::as_str) {
                    if !c.is_empty() {
                        if let (Some(rid), None) = (rs_id.clone(), msg_id.as_ref()) {
                            if !output.iter().any(|o| o["id"] == rid) {
                                yield Ok::<_, axum::Error>(event("response.reasoning_summary_text.done", json!({"item_id": rid, "output_index": output.len(), "summary_index": 0, "text": reasoning}), &mut seq));
                                yield Ok::<_, axum::Error>(event("response.reasoning_summary_part.done", json!({"item_id": rid, "output_index": output.len(), "summary_index": 0, "part": {"type": "summary_text", "text": reasoning}}), &mut seq));
                                let item = json!({"type": "reasoning", "id": rid, "summary": [{"type": "summary_text", "text": reasoning}]});
                                yield Ok::<_, axum::Error>(event("response.output_item.done", json!({"output_index": output.len(), "item": item}), &mut seq));
                                output.push(item);
                            }
                        }
                        if msg_id.is_none() {
                            let mid = new_id("msg");
                            yield Ok::<_, axum::Error>(event("response.output_item.added", json!({"output_index": output.len(), "item": {"type": "message", "id": mid, "status": "in_progress", "role": "assistant", "content": []}}), &mut seq));
                            yield Ok::<_, axum::Error>(event("response.content_part.added", json!({"item_id": mid, "output_index": output.len(), "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}}), &mut seq));
                            msg_id = Some(mid);
                        }
                        text.push_str(c);
                        yield Ok::<_, axum::Error>(event("response.output_text.delta", json!({"item_id": msg_id.clone(), "output_index": output.len(), "content_index": 0, "delta": c}), &mut seq));
                    }
                }
                if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    tool_calls.extend(calls.iter().cloned());
                }
            }
        }
        if let Some(rid) = rs_id.clone() {
            if !output.iter().any(|o| o["id"] == rid) {
                yield Ok::<_, axum::Error>(event("response.reasoning_summary_text.done", json!({"item_id": rid, "output_index": output.len(), "summary_index": 0, "text": reasoning}), &mut seq));
                yield Ok::<_, axum::Error>(event("response.reasoning_summary_part.done", json!({"item_id": rid, "output_index": output.len(), "summary_index": 0, "part": {"type": "summary_text", "text": reasoning}}), &mut seq));
                let item = json!({"type": "reasoning", "id": rid, "summary": [{"type": "summary_text", "text": reasoning}]});
                yield Ok::<_, axum::Error>(event("response.output_item.done", json!({"output_index": output.len(), "item": item}), &mut seq));
                output.push(item);
            }
        }
        if let Some(mid) = msg_id.clone() {
            yield Ok::<_, axum::Error>(event("response.output_text.done", json!({"item_id": mid, "output_index": output.len(), "content_index": 0, "text": text}), &mut seq));
            yield Ok::<_, axum::Error>(event("response.content_part.done", json!({"item_id": mid, "output_index": output.len(), "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": []}}), &mut seq));
            let item = json!({"type": "message", "id": mid, "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]});
            yield Ok::<_, axum::Error>(event("response.output_item.done", json!({"output_index": output.len(), "item": item}), &mut seq));
            output.push(item);
        }
        for c in &tool_calls {
            let fid = new_id("fc");
            let args = c.pointer("/function/arguments").and_then(Value::as_str).unwrap_or("{}").to_string();
            let item = json!({"type": "function_call", "id": fid, "call_id": c.get("id").cloned().unwrap_or(Value::Null), "name": c.pointer("/function/name").cloned().unwrap_or(Value::Null), "arguments": "", "status": "in_progress"});
            yield Ok::<_, axum::Error>(event("response.output_item.added", json!({"output_index": output.len(), "item": item}), &mut seq));
            yield Ok::<_, axum::Error>(event("response.function_call_arguments.delta", json!({"item_id": fid, "output_index": output.len(), "delta": args}), &mut seq));
            yield Ok::<_, axum::Error>(event("response.function_call_arguments.done", json!({"item_id": fid, "output_index": output.len(), "arguments": args}), &mut seq));
            let mut done = item;
            done["arguments"] = json!(args);
            done["status"] = json!("completed");
            yield Ok::<_, axum::Error>(event("response.output_item.done", json!({"output_index": output.len(), "item": done}), &mut seq));
            output.push(done);
        }
        let (status, incomplete) = if finish == "length" { ("incomplete", Some("max_output_tokens")) } else { ("completed", None) };
        let final_obj = response_object(&id, created, &req_for_stream, &model, status, output, usage_object(if usage.is_null() { None } else { Some(&usage) }), incomplete);
        if keep {
            let mut convo = history.clone();
            let mut assistant = json!({"role": "assistant", "content": text});
            if !tool_calls.is_empty() {
                assistant["tool_calls"] = Value::Array(tool_calls.clone());
            }
            convo.push(assistant);
            store_response(&id, convo, final_obj.clone());
        }
        let kind = if status == "completed" { "response.completed" } else { "response.incomplete" };
        yield Ok::<_, axum::Error>(event(kind, json!({"response": final_obj}), &mut seq));
    };
    Ok(Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response())
}

/// `GET /v1/responses/{id}`: a stored response.
pub(crate) async fn openai_get_response(
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let found = store()
        .lock()
        .ok()
        .and_then(|g| g.iter().find(|s| s.id == id).map(|s| s.response.clone()));
    match found {
        Some(r) => Json(r).into_response(),
        None => error(StatusCode::NOT_FOUND, format!("response '{id}' not found")),
    }
}

/// `DELETE /v1/responses/{id}`: forgets a stored response.
pub(crate) async fn openai_delete_response(
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    let removed = store()
        .lock()
        .map(|mut g| {
            let before = g.len();
            g.retain(|s| s.id != id);
            before != g.len()
        })
        .unwrap_or(false);
    if removed {
        Json(json!({"id": id, "object": "response", "deleted": true})).into_response()
    } else {
        error(StatusCode::NOT_FOUND, format!("response '{id}' not found"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_items_lower_to_chat_messages() {
        let req = json!({
            "instructions": "be brief",
            "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "42"}
            ]
        });
        let m = lower_input(&req).unwrap();
        assert_eq!(m.len(), 4);
        assert_eq!(m[0]["role"], "system");
        assert_eq!(m[1]["content"][0]["type"], "text");
        assert_eq!(m[2]["tool_calls"][0]["function"]["name"], "f");
        assert_eq!(m[3]["role"], "tool");
    }

    #[test]
    fn flat_tools_nest_and_unknown_kinds_are_refused() {
        let ok = json!({"tools": [{"type": "function", "name": "f", "parameters": {}}]});
        assert_eq!(
            lower_tools(&ok).unwrap().unwrap()[0]["function"]["name"],
            "f"
        );
        let no = json!({"tools": [{"type": "web_search"}]});
        assert!(lower_tools(&no).is_err());
    }

    #[test]
    fn a_completion_lifts_to_output_items_in_order() {
        let msg = json!({"content": "hello", "reasoning_content": "think", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]});
        let items = output_items(&msg);
        assert_eq!(items[0]["type"], "reasoning");
        assert_eq!(items[1]["type"], "message");
        assert_eq!(items[2]["type"], "function_call");
    }
}
