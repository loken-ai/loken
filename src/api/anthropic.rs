//! Anthropic Messages API (`POST /v1/messages`) compatibility shim.
//!
//! Lets clients that speak the Anthropic Messages protocol - notably
//! Claude Code - point at loken natively. This module holds the wire
//! types and the pure translation logic; the orchestrating handler lives
//! in `api/handlers/` (it needs the crate-private engine/prompt helpers).
//!
//! Translation strategy: an Anthropic request is lowered to the same
//! internal `Message` + `Tool` shapes the OpenAI/Ollama paths use, so tool
//! injection, generation and tool-call parsing are shared end-to-end. The
//! model output is then lifted back into Anthropic content blocks
//! (`text` + `tool_use`) for the response.

use super::types::{Message, Tool, ToolCall};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// `system` accepts either a bare string or an array of text blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicSystem {
    Text(String),
    Blocks(Vec<Value>),
}

/// The `text` a block carries, or None when it carries none.
fn block_text(b: &Value) -> Option<String> {
    b.get("text")
        .and_then(|t| t.as_str())
        .map(std::string::ToString::to_string)
}

/// The text an array of content blocks amounts to, one block to a line.
///
/// `text_of` decides what a single block contributes; a block that contributes nothing is
/// dropped rather than left as a blank line. That closure is the only place the callers
/// differ - `system` is typed blocks and nothing else, while a `tool_result` may hold bare
/// strings beside them - so the flattening itself is stated once.
fn join_block_text(blocks: &[Value], text_of: impl Fn(&Value) -> Option<String>) -> String {
    blocks
        .iter()
        .filter_map(text_of)
        .collect::<Vec<_>>()
        .join("\n")
}

impl AnthropicSystem {
    /// Flatten to a single system string (concatenating block `text`).
    fn into_text(self) -> String {
        match self {
            AnthropicSystem::Text(s) => s,
            AnthropicSystem::Blocks(blocks) => join_block_text(&blocks, block_text),
        }
    }
}

/// A message's `content`: a bare string or an array of typed blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicMessageContent {
    Text(String),
    Blocks(Vec<Value>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicMessageContent,
}

/// Anthropic tool definition: `{name, description, input_schema}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    /// Anthropic requires `max_tokens`; default defensively if a client
    /// omits it rather than 422'ing.
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub system: Option<AnthropicSystem>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Accepted so a request carrying it is not refused; nothing here reads it.
    #[serde(default)]
    #[allow(dead_code)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// `{type: "enabled", budget_tokens}`; absent or disabled, a reasoning model is told
    /// not to think. The budget is not enforced: a local model's reasoning has no cap
    /// of its own.
    #[serde(default)]
    pub thinking: Option<Value>,
    /// Structured output: `{type: "json_schema", schema}` constrains the answer.
    #[serde(default)]
    pub output_format: Option<Value>,
}

/// One block of a message's content array, after extraction.
fn block_type(b: &Value) -> &str {
    b.get("type").and_then(|t| t.as_str()).unwrap_or("")
}

/// Render a `tool_result` block's `content` (string or array of text
/// blocks) to a flat string.
fn tool_result_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        // Either a bare string or a `{type:text,text:..}` block.
        Value::Array(items) => join_block_text(items, |it| {
            it.as_str()
                .map(std::string::ToString::to_string)
                .or_else(|| block_text(it))
        }),
        other => other.to_string(),
    }
}

impl AnthropicMessagesRequest {
    /// Effective max_tokens (Anthropic field, defaulting when absent).
    /// The first content block this server cannot take, with the reason: an image
    /// given by URL (nothing is fetched from here) or a document (nothing reads PDFs).
    pub fn unsupported_input(&self) -> Option<String> {
        let messages = serde_json::to_value(&self.messages).ok()?;
        for m in messages.as_array()? {
            let Some(blocks) = m.get("content").and_then(Value::as_array) else {
                continue;
            };
            for b in blocks {
                match b.get("type").and_then(Value::as_str) {
                    Some("image")
                        if b.pointer("/source/type").and_then(Value::as_str) == Some("url") =>
                    {
                        return Some(
                            "image blocks given by URL are not fetched; send the image as base64"
                                .to_string(),
                        );
                    }
                    Some("document") => {
                        return Some(
                            "document blocks are not read; send the document's text".to_string(),
                        );
                    }
                    _ => {}
                }
            }
        }
        None
    }

    pub fn effective_max_tokens(&self) -> usize {
        self.max_tokens.unwrap_or(2048)
    }

    pub fn is_stream(&self) -> bool {
        self.stream.unwrap_or(false)
    }

    /// True unless `tool_choice.type == "none"` and tools are present.
    pub fn tools_enabled(&self) -> bool {
        let has = self.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
        if !has {
            return false;
        }
        !matches!(
            self.tool_choice
                .as_ref()
                .and_then(|v| v.get("type"))
                .and_then(|t| t.as_str()),
            Some("none")
        )
    }

    /// Lower the Anthropic request to internal `Message`s and `Tool`s.
    /// System prompt becomes a leading system message; assistant
    /// `tool_use` blocks become `Message.tool_calls`; user `tool_result`
    /// blocks become `role: "tool"` messages; images are carried via the
    /// Ollama-style base64 `images` field.
    pub fn into_internal(self) -> (Vec<Message>, Vec<Tool>) {
        let mut messages: Vec<Message> = Vec::with_capacity(self.messages.len() + 1);

        if let Some(sys) = self.system {
            let text = sys.into_text();
            if !text.is_empty() {
                messages.push(Message::new("system".to_string(), text));
            }
        }

        for msg in self.messages {
            match msg.content {
                AnthropicMessageContent::Text(text) => {
                    messages.push(Message::new(msg.role.clone(), text));
                }
                AnthropicMessageContent::Blocks(blocks) => {
                    let mut text = String::new();
                    let mut tool_calls: Vec<ToolCall> = Vec::new();
                    let mut images: Vec<String> = Vec::new();
                    let mut tool_results: Vec<(String, String)> = Vec::new();

                    for b in &blocks {
                        match block_type(b) {
                            "text" => {
                                if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                                    if !text.is_empty() {
                                        text.push('\n');
                                    }
                                    text.push_str(t);
                                }
                            }
                            "tool_use" => {
                                let name = b.get("name").and_then(|n| n.as_str()).unwrap_or("");
                                let id = b
                                    .get("id")
                                    .and_then(|i| i.as_str())
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| {
                                        format!("call_{}", uuid::Uuid::new_v4().simple())
                                    });
                                let input = b.get("input").cloned().unwrap_or(json!({}));
                                tool_calls.push(ToolCall {
                                    id,
                                    r#type: "function".to_string(),
                                    function: Some(super::types::ToolCallFunction {
                                        name: name.to_string(),
                                        arguments: Some(input.to_string()),
                                    }),
                                });
                            }
                            "tool_result" => {
                                let id = b
                                    .get("tool_use_id")
                                    .and_then(|i| i.as_str())
                                    .unwrap_or("")
                                    .to_string();
                                let content = b.get("content").cloned().unwrap_or(Value::Null);
                                tool_results.push((id, tool_result_text(&content)));
                            }
                            "image" => {
                                // {source:{type:base64,media_type,data}}
                                if let Some(data) = b
                                    .get("source")
                                    .and_then(|s| s.get("data"))
                                    .and_then(|d| d.as_str())
                                {
                                    images.push(data.to_string());
                                }
                            }
                            _ => {}
                        }
                    }

                    // Emit tool_result blocks as standalone `tool` messages
                    // (they precede the textual user turn in Anthropic's
                    // model). Order: results first, then any user text.
                    for (id, res) in tool_results {
                        let mut m = Message::new("tool".to_string(), res);
                        if !id.is_empty() {
                            m.tool_call_id = Some(id);
                        }
                        messages.push(m);
                    }

                    // Skip a fully-empty assistant/user turn (e.g. a
                    // message that held only tool_results, already emitted).
                    if text.is_empty() && tool_calls.is_empty() && images.is_empty() {
                        continue;
                    }

                    let mut m = Message::new(msg.role.clone(), text);
                    if !tool_calls.is_empty() {
                        m.tool_calls = Some(tool_calls);
                    }
                    if !images.is_empty() {
                        m.images = Some(images);
                    }
                    messages.push(m);
                }
            }
        }

        let tools = self
            .tools
            .unwrap_or_default()
            .into_iter()
            .map(|t| Tool::function(t.name, t.description, t.input_schema))
            .collect();

        (messages, tools)
    }
}

/// Map an internal finish state to an Anthropic `stop_reason`.
pub fn stop_reason(used_tools: bool, hit_max: bool) -> &'static str {
    if used_tools {
        "tool_use"
    } else if hit_max {
        "max_tokens"
    } else {
        "end_turn"
    }
}

/// A tool call's arguments the way this protocol states them.
///
/// The two shapes disagree about the type: `input` is an object here, while the internal
/// call carries the same arguments as the JSON string the other protocol asks for. Absent
/// arguments and arguments that will not parse both come out as an empty object, because a
/// tool invoked with none is an ordinary thing and a block without `input` is not valid.
fn tool_input(func: &super::types::ToolCallFunction) -> Value {
    match func.arguments.as_deref().map(serde_json::from_str::<Value>) {
        Some(Ok(input)) => input,
        _ => json!({}),
    }
}

/// Build the Anthropic content-block array from parsed text + tool calls.
fn content_blocks(thinking: Option<&str>, text: &str, calls: &[ToolCall]) -> Vec<Value> {
    let mut blocks = Vec::new();
    // A local model signs nothing: the signature is present, and empty.
    if let Some(t) = thinking {
        blocks.push(json!({"type": "thinking", "thinking": t, "signature": ""}));
    }
    if !text.is_empty() {
        blocks.push(json!({"type": "text", "text": text}));
    }
    for call in calls {
        if let Some(func) = call.function.as_ref() {
            let input = tool_input(func);
            blocks.push(json!({
                "type": "tool_use",
                "id": call.id,
                "name": func.name,
                "input": input,
            }));
        }
    }
    // Anthropic always returns at least one block.
    if blocks.is_empty() {
        blocks.push(json!({"type": "text", "text": ""}));
    }
    blocks
}

/// The Messages object itself: the field set, spelling and order this protocol defines.
///
/// It is the whole non-streaming body, and it is also what the stream's opening event
/// carries - with its content still empty and its stop reason not yet known. One shape,
/// so one statement of it: written out at both places, a field added to the response
/// would silently be missing from the stream.
fn message_object(
    id: &str,
    model: &str,
    content: Value,
    stop_reason: Value,
    stop_sequence: Option<&str>,
    input_tokens: i32,
    output_tokens: i32,
    cache_read: i32,
    cache_creation: i32,
) -> Value {
    json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_creation,
        },
    })
}

/// Build the non-streaming Anthropic Messages response body.
/// `stop_sequence` names the sequence that ended the answer, when one did; the cache
/// counts say how much of the prompt the resident KV served and how much was prefilled.
pub fn build_response(
    id: &str,
    model: &str,
    thinking: Option<&str>,
    text: &str,
    calls: &[ToolCall],
    input_tokens: i32,
    output_tokens: i32,
    hit_max: bool,
    stop_sequence: Option<&str>,
    cache_read: i32,
    cache_creation: i32,
) -> Value {
    let used_tools = !calls.is_empty();
    let reason = match stop_sequence {
        Some(_) if !used_tools => "stop_sequence",
        _ => stop_reason(used_tools, hit_max),
    };
    message_object(
        id,
        model,
        json!(content_blocks(thinking, text, calls)),
        json!(reason),
        stop_sequence.filter(|_| !used_tools),
        input_tokens,
        output_tokens,
        cache_read,
        cache_creation,
    )
}

/// SSE event payloads for the streaming protocol. Each returns the JSON
/// `data:` body; the handler prefixes the matching `event:` line.
pub mod sse {
    use super::*;

    pub fn message_start(id: &str, model: &str, input_tokens: i32) -> Value {
        json!({
            "type": "message_start",
            "message": message_object(id, model, json!([]), Value::Null, None, input_tokens, 0, 0, 0),
        })
    }

    pub fn text_block_start(index: usize) -> Value {
        json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "text", "text": ""}
        })
    }

    pub fn thinking_block_start(index: usize) -> Value {
        json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "thinking", "thinking": ""}
        })
    }
    pub fn thinking_delta(index: usize, text: &str) -> Value {
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": text}
        })
    }
    /// Closes a thinking block the way the official API does, with a signature; a
    /// local model has none to give.
    pub fn signature_delta(index: usize) -> Value {
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "signature_delta", "signature": ""}
        })
    }
    pub fn error(kind: &str, message: &str) -> Value {
        json!({"type": "error", "error": {"type": kind, "message": message}})
    }
    pub fn text_delta(index: usize, text: &str) -> Value {
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text}
        })
    }

    pub fn tool_block_start(index: usize, call: &ToolCall) -> Value {
        let (id, name) = match call.function.as_ref() {
            Some(f) => (call.id.as_str(), f.name.as_str()),
            None => (call.id.as_str(), ""),
        };
        json!({
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}
        })
    }

    pub fn tool_input_delta(index: usize, partial_json: &str) -> Value {
        json!({
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json}
        })
    }

    pub fn block_stop(index: usize) -> Value {
        json!({"type": "content_block_stop", "index": index})
    }

    pub fn message_delta(
        used_tools: bool,
        hit_max: bool,
        stop_sequence: Option<&str>,
        output_tokens: i32,
        input_tokens: i32,
        cache_read: i32,
        cache_creation: i32,
    ) -> Value {
        let reason = match stop_sequence {
            Some(_) if !used_tools => "stop_sequence",
            _ => stop_reason(used_tools, hit_max),
        };
        json!({
            "type": "message_delta",
            "delta": {"stop_reason": reason, "stop_sequence": stop_sequence.filter(|_| !used_tools)},
            "usage": {
                "output_tokens": output_tokens,
                "input_tokens": input_tokens,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_creation,
            }
        })
    }

    pub fn message_stop() -> Value {
        json!({"type": "message_stop"})
    }

    pub fn ping() -> Value {
        json!({"type": "ping"})
    }
}

/// Anthropic error envelope (`{type:error,error:{type,message}}`).
#[derive(Debug, Serialize)]
pub struct AnthropicError {
    pub r#type: &'static str,
    pub error: AnthropicErrorBody,
}

#[derive(Debug, Serialize)]
pub struct AnthropicErrorBody {
    pub r#type: String,
    pub message: String,
}

impl AnthropicError {
    pub fn new(kind: &str, message: String) -> Value {
        json!({"type": "error", "error": {"type": kind, "message": message}})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_string_content_and_system() {
        let req: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 100,
            "system": "be terse",
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        let (msgs, tools) = req.into_internal();
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[0].content, "be terse");
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[1].content, "hi");
        assert!(tools.is_empty());
    }

    #[test]
    fn lowers_tool_use_and_tool_result_blocks() {
        let req: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "m",
            "max_tokens": 100,
            "tools": [{"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}],
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "checking"},
                    {"type": "tool_use", "id": "tu_1", "name": "get_weather", "input": {"city": "Paris"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "tu_1", "content": "sunny"}
                ]}
            ]
        }))
        .unwrap();
        assert!(req.tools_enabled());
        let (msgs, tools) = req.into_internal();
        assert_eq!(tools.len(), 1);
        let asst = msgs.iter().find(|m| m.role == "assistant").unwrap();
        assert_eq!(
            asst.tool_calls.as_ref().unwrap()[0]
                .function
                .as_ref()
                .unwrap()
                .name,
            "get_weather"
        );
        let tool = msgs.iter().find(|m| m.role == "tool").unwrap();
        assert_eq!(tool.content, "sunny");
        assert_eq!(tool.tool_call_id.as_deref(), Some("tu_1"));
    }

    #[test]
    fn tool_choice_none_disables() {
        let req: AnthropicMessagesRequest = serde_json::from_value(json!({
            "model": "m", "max_tokens": 10,
            "tools": [{"name": "f"}],
            "tool_choice": {"type": "none"},
            "messages": [{"role": "user", "content": "hi"}]
        }))
        .unwrap();
        assert!(!req.tools_enabled());
    }

    #[test]
    fn response_has_tool_use_block() {
        let calls = vec![ToolCall {
            id: "call_1".into(),
            r#type: "function".into(),
            function: Some(super::super::types::ToolCallFunction {
                name: "f".into(),
                arguments: Some("{\"x\":1}".into()),
            }),
        }];
        let v = build_response("msg_1", "m", None, "", &calls, 5, 7, false, None, 0, 0);
        assert_eq!(v["stop_reason"], "tool_use");
        let blocks = v["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "tool_use");
        assert_eq!(blocks[0]["input"]["x"], 1);
    }

    #[test]
    fn response_text_only() {
        let v = build_response("msg_1", "m", None, "hello", &[], 3, 2, false, None, 0, 0);
        assert_eq!(v["stop_reason"], "end_turn");
        assert_eq!(v["content"][0]["text"], "hello");
    }
}
