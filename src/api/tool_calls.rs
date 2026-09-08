//! Tool / function calling support.
//!
//! The text engine has no native tool-call decode path: it emits raw
//! tokens. This module bridges the OpenAI/Anthropic structured-tool
//! contract onto that raw stream in three steps:
//!
//!   1. **Inject** - turn the request's `tools` array into a system-prompt
//!      block that lists the available functions and shows the model the
//!      exact marker syntax to emit (`build_tool_system_prompt`).
//!   2. **Round-trip** - render prior-turn assistant tool calls and tool
//!      results back into the prompt in the same marker syntax so a
//!      multi-turn agentic loop keeps its history coherent
//!      (`flatten_messages`).
//!   3. **Parse** - scan the generated text for those markers and lift
//!      them back into structured `ToolCall`s (`parse_tool_calls`), with a
//!      streaming variant (`StreamToolScanner`) that streams natural
//!      content until the first tool marker, then buffers the tool region.
//!
//! Marker syntaxes follow the reference implementations (llama.cpp
//! `common/chat.cpp`, vllm `tool_parsers/*`, ollama `template/`): Hermes/
//! Qwen `<tool_call>{json}</tool_call>`, Mistral `[TOOL_CALLS] [..]`,
//! Llama-3 bare JSON (optionally `<|python_tag|>`-prefixed). Unknown
//! families fall back to the Hermes shape, which instruct models follow
//! reliably from the system instruction even when not natively trained on
//! it - the parser, not the model's training, is the correctness lever.

use super::types::{Message, Tool, ToolCall, ToolCallFunction};
use serde_json::Value;

/// Native tool-call marker family for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFormat {
    /// `<tool_call>{"name":..,"arguments":{..}}</tool_call>` - Qwen,
    /// Hermes-2-Pro, qwen3-coder and every ChatML model. Also the
    /// fallback for unknown families.
    Hermes,
    /// `[TOOL_CALLS] [{"name":..,"arguments":{..}}]` - Mistral, Mixtral,
    /// Devstral.
    Mistral,
    /// Bare JSON `{"name":..,"parameters":{..}}`, optionally prefixed by
    /// `<|python_tag|>` - Llama 3.1 / 3.2.
    Llama3,
    /// `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{..}<|call|>`
    /// - gpt-oss, the harmony format.
    Harmony,
    /// `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME` then a
    /// ```json block, closed by `<｜tool▁call▁end｜><｜tool▁calls▁end｜>` - DeepSeek V3 / R1.
    DeepSeek,
}

impl ToolFormat {
    /// Start markers that, once fully present in the output, indicate the
    /// model has begun a tool call. Used by the streaming scanner to know
    /// when to stop streaming natural content. Ordered longest-first so a
    /// scan can hold back the maximal partial prefix.
    fn start_markers(self) -> &'static [&'static str] {
        match self {
            ToolFormat::Hermes => &["<tool_call>"],
            ToolFormat::Mistral => &["[TOOL_CALLS]"],
            // Llama emits bare JSON; `<|python_tag|>` is the only
            // unambiguous start marker. Bare-JSON detection mid-stream is
            // unreliable, so for Llama we only treat the python tag as a
            // streaming trigger and rely on the end-of-stream parse to
            // catch tag-less JSON.
            ToolFormat::Llama3 => &["<|python_tag|>"],
            ToolFormat::Harmony => &["<|channel|>commentary"],
            ToolFormat::DeepSeek => &[DS_CALLS_BEGIN],
        }
    }
}

/// Decide whether tools should be injected for this request: true when a
/// non-empty `tools` array is present AND `tool_choice` is not the string
/// `"none"`. (`required`/specific-function choices degrade to auto, since
/// the engine can't be forced to emit a particular call - the model is
/// merely instructed.)
pub fn should_inject_tools(tools: Option<&Vec<Tool>>, tool_choice: Option<&Value>) -> bool {
    let has_tools = tools.map(|t| !t.is_empty()).unwrap_or(false);
    if !has_tools {
        return false;
    }
    !matches!(tool_choice.and_then(|v| v.as_str()), Some("none"))
}

/// Detect the tool-call family from the model's chat template (preferred,
/// it reflects what the model was trained to emit) falling back to the
/// model name. Mirrors the family detection in `format_chat_prompt`.
pub fn detect_tool_format(template: Option<&str>, model_name: &str) -> ToolFormat {
    if let Some(t) = template {
        if t.contains("<|channel|>") {
            return ToolFormat::Harmony;
        }
        if t.contains(DS_CALLS_BEGIN) || t.contains("<｜tool▁call▁begin｜>") {
            return ToolFormat::DeepSeek;
        }
        if t.contains("[TOOL_CALLS]") || t.contains("[AVAILABLE_TOOLS]") {
            return ToolFormat::Mistral;
        }
        if t.contains("<tool_call>") {
            return ToolFormat::Hermes;
        }
        if t.contains("<|python_tag|>")
            || (t.contains("ipython") && t.contains("<|start_header_id|>"))
        {
            return ToolFormat::Llama3;
        }
        // ChatML template without an explicit tool marker still emits
        // Hermes-style tool calls when instructed.
        if t.contains("<|im_start|>") {
            return ToolFormat::Hermes;
        }
    }
    let m = model_name.to_ascii_lowercase();
    if m.contains("gpt-oss") {
        ToolFormat::Harmony
    } else if m.contains("deepseek") {
        ToolFormat::DeepSeek
    } else if m.contains("mistral")
        || m.contains("mixtral")
        || m.contains("devstral")
        || m.contains("codestral")
    {
        ToolFormat::Mistral
    } else if m.contains("llama") {
        ToolFormat::Llama3
    } else {
        // qwen, hermes, deepseek-coder, phi, gemma, glm and anything
        // unknown: Hermes is the most widely-followed marker shape.
        ToolFormat::Hermes
    }
}

/// Serialize the tools array to the per-line JSON the Hermes/Qwen system
/// block expects: one `{"type":"function","function":{...}}` object per
/// line.
fn tools_as_json_lines(tools: &[Tool]) -> String {
    tools
        .iter()
        .map(|t| serde_json::to_string(t).unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Compact JSON array of the tools (Mistral `[AVAILABLE_TOOLS]` shape).
fn tools_as_json_array(tools: &[Tool]) -> String {
    serde_json::to_string(tools).unwrap_or_else(|_| "[]".to_string())
}

/// Derive an extra instruction from OpenAI's `tool_choice`. The engine
/// can't be *forced* to emit a specific call, but a stronger instruction
/// markedly improves compliance:
///   - `"required"` -> must call at least one tool,
///   - `{"type":"function","function":{"name":"X"}}` -> must call `X`,
///   - `"auto"`/`"none"`/absent -> no extra directive (None).
pub fn tool_choice_directive(tool_choice: Option<&Value>) -> Option<String> {
    let tc = tool_choice?;
    // OpenAI string form: "auto" | "none" | "required".
    if let Some(s) = tc.as_str() {
        return match s {
            "required" => Some("You MUST call at least one of the provided tools.".to_string()),
            _ => None,
        };
    }
    // Object form - handle both OpenAI ({"type":"function","function":
    // {"name":"X"}}) and Anthropic ({"type":"any"} / {"type":"tool",
    // "name":"X"}).
    // Specific function by name (OpenAI nests under .function.name,
    // Anthropic puts .name at top level).
    if let Some(name) = tc
        .get("function")
        .and_then(|f| f.get("name"))
        .and_then(|n| n.as_str())
        .or_else(|| tc.get("name").and_then(|n| n.as_str()))
    {
        return Some(format!("You MUST call the `{name}` function."));
    }
    // Anthropic "any" / OpenAI "required" expressed as a type.
    match tc.get("type").and_then(|t| t.as_str()) {
        Some("any") | Some("required") => {
            Some("You MUST call at least one of the provided tools.".to_string())
        }
        _ => None,
    }
}

/// Build the system-prompt block that advertises the available tools and
/// shows the model the exact marker syntax to emit. `directive` is an
/// optional extra instruction derived from `tool_choice`. Prepended to (or
/// merged into) the system message by `flatten_messages`.
pub fn build_tool_system_prompt(
    format: ToolFormat,
    tools: &[Tool],
    directive: Option<&str>,
) -> String {
    let base = build_tool_system_prompt_base(format, tools);
    match directive {
        Some(d) if !d.is_empty() => format!("{base}\n\n{d}"),
        _ => base,
    }
}

fn build_tool_system_prompt_base(format: ToolFormat, tools: &[Tool]) -> String {
    match format {
        ToolFormat::Hermes => format!(
            "You are a function calling AI model. You are provided with function \
signatures within <tools></tools> XML tags. You may call one or more functions \
to assist with the user query. Don't make assumptions about what values to plug \
into functions.\n\n<tools>\n{tools}\n</tools>\n\nFor each function call, return a \
JSON object with the function name and arguments within <tool_call></tool_call> \
XML tags, one object per call:\n<tool_call>\n{{\"name\": <function-name>, \
\"arguments\": <args-json-object>}}\n</tool_call>",
            tools = tools_as_json_lines(tools)
        ),
        ToolFormat::Mistral => format!(
            "You have access to the following functions. To call a function, emit a \
line beginning with [TOOL_CALLS] followed by a JSON array of objects, each with \
\"name\" and \"arguments\" keys, then stop. Available functions:\n\
[AVAILABLE_TOOLS] {tools} [/AVAILABLE_TOOLS]\n\
Example: [TOOL_CALLS] [{{\"name\": \"fn\", \"arguments\": {{\"x\": 1}}}}]",
            tools = tools_as_json_array(tools)
        ),
        ToolFormat::Llama3 => format!(
            "You have access to the following functions. When you need to call a \
function, respond with a JSON object of the form {{\"name\": <function-name>, \
\"parameters\": <args-json-object>}} and nothing else. You may emit one JSON \
object per function call. Available functions:\n{tools}",
            tools = tools_as_json_lines(tools)
        ),
        ToolFormat::Harmony => format!(
            "# Tools\n\n## functions\n\nnamespace functions {{\n{tools}\n}}\n\nTo call a \
function, write on the commentary channel: <|channel|>commentary \
to=functions.<function-name> <|constrain|>json<|message|><args-json-object><|call|>",
            tools = tools_as_json_lines(tools)
        ),
        ToolFormat::DeepSeek => format!(
            "You have access to the following functions:\n{tools}\n\nTo call one or more \
functions, write exactly: {begin_all}{begin}function{sep}<function-name>\n```json\n\
<args-json-object>\n```{end}{end_all}",
            tools = tools_as_json_lines(tools),
            begin_all = DS_CALLS_BEGIN,
            begin = DS_CALL_BEGIN,
            sep = DS_SEP,
            end = DS_CALL_END,
            end_all = DS_CALLS_END,
        ),
    }
}

/// Render an assistant turn's prior tool calls back into the model's
/// native marker syntax, so replaying conversation history reproduces what
/// the model originally emitted.
fn render_assistant_tool_calls(format: ToolFormat, calls: &[ToolCall]) -> String {
    let mut out = String::new();
    for call in calls {
        let Some(func) = call.function.as_ref() else {
            continue;
        };
        let name = &func.name;
        let args = func.arguments.as_deref().unwrap_or("{}");
        match format {
            ToolFormat::Hermes => {
                out.push_str(&format!(
                    "<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {args}}}\n</tool_call>\n"
                ));
            }
            ToolFormat::Mistral => {
                // One [TOOL_CALLS] line carrying a single-element array;
                // sequential calls produce sequential lines, which the
                // parser re-merges.
                out.push_str(&format!(
                    "[TOOL_CALLS] [{{\"name\": \"{name}\", \"arguments\": {args}}}]\n"
                ));
            }
            ToolFormat::Llama3 => {
                out.push_str(&format!(
                    "{{\"name\": \"{name}\", \"parameters\": {args}}}\n"
                ));
            }
            ToolFormat::Harmony => {
                out.push_str(&format!(
                    "<|channel|>commentary to=functions.{name} <|constrain|>json<|message|>{args}<|call|>\n"
                ));
            }
            ToolFormat::DeepSeek => {
                out.push_str(&format!(
                    "{DS_CALLS_BEGIN}{DS_CALL_BEGIN}function{DS_SEP}{name}\n```json\n{args}\n```{DS_CALL_END}{DS_CALLS_END}\n"
                ));
            }
        }
    }
    out
}

/// Render a tool-result message (role `tool`/`function`) into the native
/// observation syntax.
fn render_tool_result(format: ToolFormat, content: &str) -> String {
    match format {
        ToolFormat::Hermes => format!("<tool_response>\n{content}\n</tool_response>"),
        ToolFormat::Mistral => format!("[TOOL_RESULTS] {content} [/TOOL_RESULTS]"),
        ToolFormat::Llama3 => content.to_string(),
        ToolFormat::Harmony => {
            format!(
                "<|start|>functions to=assistant<|channel|>commentary<|message|>{content}<|end|>"
            )
        }
        ToolFormat::DeepSeek => format!("<｜tool▁output▁begin｜>{content}<｜tool▁output▁end｜>"),
    }
}

/// Rewrite the message list so the existing `format_chat_prompt` family
/// formatters can render a tool-augmented conversation without any change:
///   - the tool definitions are injected into the system message (merged
///     if one exists, otherwise prepended),
///   - assistant turns carrying `tool_calls` get those calls appended to
///     their textual content in native marker syntax,
///   - `tool`/`function` result turns get their content wrapped in the
///     native observation syntax.
pub fn flatten_messages(
    messages: &[Message],
    format: ToolFormat,
    tools: &[Tool],
    directive: Option<&str>,
) -> Vec<Message> {
    let tool_block = build_tool_system_prompt(format, tools, directive);
    let mut out: Vec<Message> = Vec::with_capacity(messages.len() + 1);
    let mut injected = false;

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                // Merge the tool block into the first system message so the
                // model sees a single coherent system turn.
                let merged = if injected {
                    msg.content.clone()
                } else {
                    injected = true;
                    if msg.content.is_empty() {
                        tool_block.clone()
                    } else {
                        format!("{}\n\n{}", msg.content, tool_block)
                    }
                };
                out.push(Message::new("system".to_string(), merged));
            }
            "assistant" => {
                let mut content = msg.content.clone();
                if let Some(calls) = msg.tool_calls.as_ref() {
                    if !calls.is_empty() {
                        if !content.is_empty() {
                            content.push('\n');
                        }
                        content.push_str(&render_assistant_tool_calls(format, calls));
                    }
                }
                out.push(Message::new("assistant".to_string(), content));
            }
            "tool" | "function" => {
                let rendered = render_tool_result(format, &msg.content);
                // ChatML carries arbitrary role names; keep `tool` so the
                // model sees an observation turn. The [INST] fallback
                // tags it `(tool)` which is still legible.
                out.push(Message::new("tool".to_string(), rendered));
            }
            _ => out.push(msg.clone()),
        }
    }

    if !injected {
        // No system message existed - prepend one carrying the tool block.
        out.insert(0, Message::new("system".to_string(), tool_block));
    }
    out
}

/// Result of parsing a completed generation for tool calls.
pub struct ToolParseResult {
    /// Natural-language content with tool-call markers stripped out.
    pub content: String,
    /// Structured tool calls lifted from the output.
    pub calls: Vec<ToolCall>,
}

/// Fresh OpenAI-style tool-call id (`call_<hex>`).
fn new_call_id() -> String {
    format!("call_{}", uuid::Uuid::new_v4().simple())
}

/// Build a `ToolCall` from a parsed name + arguments value. `arguments`
/// is serialized to a JSON string per the OpenAI contract (the function
/// arguments field is a string, not an object).
fn make_call(name: &str, arguments: &Value) -> ToolCall {
    let args_str = if arguments.is_null() {
        "{}".to_string()
    } else if let Some(s) = arguments.as_str() {
        // Already a JSON string - keep verbatim.
        s.to_string()
    } else {
        serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_string())
    };
    ToolCall {
        id: new_call_id(),
        r#type: "function".to_string(),
        function: Some(ToolCallFunction {
            name: name.to_string(),
            arguments: Some(args_str),
        }),
    }
}

/// Lift a single `{"name":..,"arguments"/"parameters":..}` object into a
/// `ToolCall`, accepting either argument key.
fn call_from_object(obj: &Value) -> Option<ToolCall> {
    let name = obj.get("name").and_then(|v| v.as_str())?;
    let args = obj
        .get("arguments")
        .or_else(|| obj.get("parameters"))
        .cloned()
        .unwrap_or(Value::Null);
    Some(make_call(name, &args))
}

/// Parse a completed generation for tool calls in the given family.
/// Returns the residual natural-language content and any calls found.
/// A JSON schema for one call object, `{"name", "arguments"}`, to constrain a
/// generation to a call when the client requires one (`tool_choice` required, any, or a
/// named function). The bare object is what every format's parser falls back to.
pub fn forced_call_schema(tools: &[Tool], only: Option<&str>) -> Value {
    let names: Vec<&str> = tools
        .iter()
        .filter_map(|t| t.function.as_ref().map(|f| f.name.as_str()))
        .filter(|n| only.is_none_or(|o| o == *n))
        .collect();
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": names},
            "arguments": {"type": "object"}
        },
        "required": ["name", "arguments"],
        "additionalProperties": false
    })
}

pub fn parse_tool_calls(format: ToolFormat, raw: &str) -> ToolParseResult {
    let parsed = parse_tool_calls_native(format, raw);
    if parsed.calls.is_empty() && raw.trim_start().starts_with('{') {
        // A constrained generation answers with the bare call object, whatever
        // the model's own syntax.
        let bare = scan_bare_json_calls(raw);
        if !bare.calls.is_empty() {
            return bare;
        }
    }
    parsed
}

fn parse_tool_calls_native(format: ToolFormat, raw: &str) -> ToolParseResult {
    match format {
        ToolFormat::Hermes => parse_hermes(raw),
        ToolFormat::Mistral => parse_mistral(raw),
        ToolFormat::Llama3 => parse_llama3(raw),
        ToolFormat::Harmony => parse_harmony(raw),
        ToolFormat::DeepSeek => parse_deepseek(raw),
    }
}

const DS_CALLS_BEGIN: &str = "<｜tool▁calls▁begin｜>";
const DS_CALLS_END: &str = "<｜tool▁calls▁end｜>";
const DS_CALL_BEGIN: &str = "<｜tool▁call▁begin｜>";
const DS_CALL_END: &str = "<｜tool▁call▁end｜>";
const DS_SEP: &str = "<｜tool▁sep｜>";

/// The arguments of a call as JSON, whether the model wrote them bare or fenced.
fn call_arguments(fragment: &str) -> Value {
    let t = fragment.trim();
    if let Ok(v) = serde_json::from_str::<Value>(t) {
        return v;
    }
    first_json_object(t)
        .and_then(|span| serde_json::from_str::<Value>(span).ok())
        .unwrap_or(Value::Object(Default::default()))
}

/// gpt-oss: every `to=functions.NAME` header opens a call whose arguments follow
/// `<|message|>` up to `<|call|>`; the channel header itself is not content.
fn parse_harmony(raw: &str) -> ToolParseResult {
    const TO: &str = "to=functions.";
    const MSG: &str = "<|message|>";
    const CALL: &str = "<|call|>";
    if !raw.contains(TO) {
        return scan_bare_json_calls(raw);
    }
    let mut calls = Vec::new();
    let mut content = String::new();
    let mut rest = raw;
    while let Some(i) = rest.find(TO) {
        let head = rest[..i].rfind("<|channel|>").unwrap_or(i);
        content.push_str(&rest[..head]);
        let after = &rest[i + TO.len()..];
        let name_end = after
            .find(|c: char| c.is_whitespace() || c == '<')
            .unwrap_or(after.len());
        let name = &after[..name_end];
        let Some(m) = after.find(MSG) else {
            rest = "";
            break;
        };
        let body = &after[m + MSG.len()..];
        let (json, consumed) = match body.find(CALL) {
            Some(c) => (&body[..c], m + MSG.len() + c + CALL.len()),
            None => (body, after.len()),
        };
        calls.push(make_call(name, &call_arguments(json)));
        rest = &after[consumed.min(after.len())..];
    }
    content.push_str(rest);
    ToolParseResult {
        content: content.trim().to_string(),
        calls,
    }
}

/// DeepSeek: the calls sit between the begin/end markers, each naming its function
/// after the separator and carrying its arguments in a fenced JSON block.
fn parse_deepseek(raw: &str) -> ToolParseResult {
    if !raw.contains(DS_CALL_BEGIN) {
        return scan_bare_json_calls(raw);
    }
    let start = raw
        .find(DS_CALLS_BEGIN)
        .or_else(|| raw.find(DS_CALL_BEGIN))
        .unwrap_or(0);
    let mut content = raw[..start].to_string();
    let after_all = raw
        .find(DS_CALLS_END)
        .map(|e| &raw[e + DS_CALLS_END.len()..])
        .unwrap_or("");
    content.push_str(after_all);
    let mut calls = Vec::new();
    let mut rest = &raw[start..];
    while let Some(b) = rest.find(DS_CALL_BEGIN) {
        let seg = &rest[b + DS_CALL_BEGIN.len()..];
        let (seg, next) = match seg.find(DS_CALL_END) {
            Some(e) => (&seg[..e], b + DS_CALL_BEGIN.len() + e + DS_CALL_END.len()),
            None => (seg, rest.len()),
        };
        if let Some(sep) = seg.find(DS_SEP) {
            let named = &seg[sep + DS_SEP.len()..];
            let name_end = named.find('\n').unwrap_or(named.len());
            let name = named[..name_end].trim();
            let args = named[name_end..]
                .trim()
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_end_matches("```");
            if !name.is_empty() {
                calls.push(make_call(name, &call_arguments(args)));
            }
        }
        rest = &rest[next.min(rest.len())..];
    }
    ToolParseResult {
        content: content.trim().to_string(),
        calls,
    }
}

/// Hermes: extract every `<tool_call> ... </tool_call>` block (tolerating a
/// missing final `</tool_call>` when the model hit EOS mid-tag) and parse
/// the inner JSON object. If the model emitted the call as bare JSON
/// without the tags (common - instruct models often ignore the marker
/// instruction and just print `{"name":..,"arguments":..}`), fall back to
/// a bare-JSON scan.
fn parse_hermes(raw: &str) -> ToolParseResult {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    if !raw.contains(OPEN) {
        return scan_bare_json_calls(raw);
    }
    let mut calls = Vec::new();
    let mut content = String::new();
    let mut rest = raw;
    while let Some(open_idx) = rest.find(OPEN) {
        content.push_str(&rest[..open_idx]);
        let after_open = &rest[open_idx + OPEN.len()..];
        let (inner, consumed) = match after_open.find(CLOSE) {
            Some(close_idx) => (
                &after_open[..close_idx],
                open_idx + OPEN.len() + close_idx + CLOSE.len(),
            ),
            // No closing tag: take the remainder.
            None => (after_open, raw.len()),
        };
        if let Some(call) = parse_json_call_object(inner) {
            calls.push(call);
        }
        if consumed >= rest.len() {
            rest = "";
            break;
        }
        rest = &rest[consumed..];
    }
    content.push_str(rest);
    ToolParseResult {
        content: content.trim().to_string(),
        calls,
    }
}

/// Parse a fragment that should hold one `{"name":..,"arguments":..}`
/// object, tolerating leading/trailing whitespace or stray text by
/// scanning for the first balanced JSON object.
fn parse_json_call_object(fragment: &str) -> Option<ToolCall> {
    let trimmed = fragment.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return call_from_object(&v);
    }
    // Fall back to extracting the first balanced {...} span.
    let span = first_json_object(trimmed)?;
    let v: Value = serde_json::from_str(span).ok()?;
    call_from_object(&v)
}

/// Mistral: `[TOOL_CALLS]` followed by a JSON array (or a single object)
/// of `{"name":..,"arguments":..}`.
fn parse_mistral(raw: &str) -> ToolParseResult {
    const MARK: &str = "[TOOL_CALLS]";
    if !raw.contains(MARK) {
        // Model emitted bare JSON instead of the [TOOL_CALLS] marker.
        return scan_bare_json_calls(raw);
    }
    let mut calls = Vec::new();
    let mut content = String::new();
    let mut rest = raw;
    while let Some(idx) = rest.find(MARK) {
        content.push_str(&rest[..idx]);
        let after = rest[idx + MARK.len()..].trim_start();
        // Strip an optional [/TOOL_CALLS] trailer when locating the JSON.
        let after = after.strip_prefix('[').map(|_| after).unwrap_or(after);
        if let Some(span) = first_json_array(after).or_else(|| first_json_object(after)) {
            if let Ok(v) = serde_json::from_str::<Value>(span) {
                match v {
                    Value::Array(items) => {
                        for it in &items {
                            if let Some(c) = call_from_object(it) {
                                calls.push(c);
                            }
                        }
                    }
                    Value::Object(_) => {
                        if let Some(c) = call_from_object(&v) {
                            calls.push(c);
                        }
                    }
                    _ => {}
                }
            }
            // Advance past the parsed span.
            let span_end = (span.as_ptr() as usize + span.len()) - after.as_ptr() as usize;
            let abs = idx
                + MARK.len()
                + (after.as_ptr() as usize - rest[idx + MARK.len()..].as_ptr() as usize)
                + span_end;
            if abs >= rest.len() {
                rest = "";
                break;
            }
            rest = &rest[abs..];
        } else {
            // No JSON after the marker - drop the marker, keep scanning.
            rest = &rest[idx + MARK.len()..];
        }
    }
    content.push_str(rest);
    // Drop a dangling [/TOOL_CALLS] closer if present in residual content.
    let content = content.replace("[/TOOL_CALLS]", "");
    ToolParseResult {
        content: content.trim().to_string(),
        calls,
    }
}

/// Llama 3: bare JSON object(s), optionally prefixed by `<|python_tag|>`.
fn parse_llama3(raw: &str) -> ToolParseResult {
    scan_bare_json_calls(raw)
}

/// Scan free text for bare-JSON tool calls - `{"name":..,"arguments":..}`
/// or `{"name":..,"parameters":..}` objects, optionally `<|python_tag|>`-
/// prefixed. Used directly for Llama 3 and as the fallback for every
/// family when the native markers are absent (models frequently emit bare
/// JSON regardless of the marker instruction). Non-call text is preserved
/// as content.
fn scan_bare_json_calls(raw: &str) -> ToolParseResult {
    let stripped = raw.replace("<|python_tag|>", "");
    let trimmed = stripped.trim();
    let mut calls = Vec::new();

    // Fast path: the entire output is one JSON object/array.
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        match &v {
            Value::Object(_) => {
                if let Some(c) = call_from_object(&v) {
                    calls.push(c);
                    return ToolParseResult {
                        content: String::new(),
                        calls,
                    };
                }
            }
            Value::Array(items) => {
                for it in items {
                    if let Some(c) = call_from_object(it) {
                        calls.push(c);
                    }
                }
                if !calls.is_empty() {
                    return ToolParseResult {
                        content: String::new(),
                        calls,
                    };
                }
            }
            _ => {}
        }
    }

    // Otherwise scan for balanced JSON objects that look like calls and
    // keep the surrounding text as content.
    let mut content = String::new();
    let mut search_from = 0usize;
    let bytes = trimmed.as_bytes();
    while let Some(rel) = trimmed[search_from..].find('{') {
        let start = search_from + rel;
        content.push_str(&trimmed[search_from..start]);
        if let Some(span) = first_json_object(&trimmed[start..]) {
            let end = start + span.len();
            if let Ok(v) = serde_json::from_str::<Value>(span) {
                if v.get("name").and_then(|n| n.as_str()).is_some() {
                    if let Some(c) = call_from_object(&v) {
                        calls.push(c);
                    }
                } else {
                    content.push_str(span);
                }
            } else {
                content.push_str(span);
            }
            search_from = end;
        } else {
            // Unbalanced - keep the rest as content.
            content.push_str(&trimmed[start..]);
            search_from = bytes.len();
            break;
        }
    }
    content.push_str(&trimmed[search_from.min(trimmed.len())..]);
    ToolParseResult {
        content: content.trim().to_string(),
        calls,
    }
}

/// Return the first balanced `{...}` substring (respecting nesting and
/// quoted strings), or None if no balanced object exists.
fn first_json_object(s: &str) -> Option<&str> {
    first_balanced(s, b'{', b'}')
}

/// Return the first balanced `[...]` substring.
fn first_json_array(s: &str) -> Option<&str> {
    first_balanced(s, b'[', b']')
}

/// Scan for the first balanced span delimited by `open`/`close`, honoring
/// JSON string quoting and escapes so braces inside strings don't throw
/// off the depth count. Returns a slice of `s`.
fn first_balanced(s: &str, open: u8, close: u8) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = bytes.iter().position(|&b| b == open)?;
    let mut depth = 0i32;
    let mut in_str = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let b = bytes[i];
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            x if x == open => depth += 1,
            x if x == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Largest `k < marker.len()` such that the last `k` bytes of `buf` equal
/// `marker[..k]` - i.e. how many trailing bytes might be the start of a
/// not-yet-complete marker and must be held back from streaming. Markers
/// are ASCII, and ASCII bytes never appear inside a multi-byte UTF-8
/// sequence, so byte-level matching can't split a codepoint.
fn partial_marker_overlap(buf: &str, marker: &str) -> usize {
    let b = buf.as_bytes();
    let m = marker.as_bytes();
    let max = b.len().min(m.len().saturating_sub(1));
    for k in (1..=max).rev() {
        if b[b.len() - k..] == m[..k] {
            return k;
        }
    }
    0
}

/// Streaming scanner: feed it generation deltas; it returns the content
/// that is safe to stream now (holding back any trailing bytes that could
/// be the start of a tool marker), and switches to buffering once a tool
/// region begins. Call `finalize` at end-of-stream to lift the buffered
/// region into structured calls.
pub struct StreamToolScanner {
    format: ToolFormat,
    /// Entire raw output seen so far (content + tool region).
    buf: String,
    /// Bytes of `buf` already returned to the caller as streamed content.
    emitted: usize,
    /// Once a tool start marker is seen, all further bytes are buffered.
    in_tool_region: bool,
    /// Family markers plus the bare-JSON trigger (`{"name"`), since models
    /// frequently emit tool calls as bare JSON without the family marker.
    markers: Vec<&'static str>,
}

/// Bare-JSON tool-call trigger for streaming. Specific enough that normal
/// prose rarely contains it, so false positives are unlikely; if one does
/// fire, `finalize` finds no call and the caller flushes `unstreamed`.
const BARE_JSON_MARKER: &str = "{\"name\"";

impl StreamToolScanner {
    pub fn new(format: ToolFormat) -> Self {
        let mut markers: Vec<&'static str> = format.start_markers().to_vec();
        markers.push(BARE_JSON_MARKER);
        Self {
            format,
            buf: String::new(),
            emitted: 0,
            in_tool_region: false,
            markers,
        }
    }

    /// Append a delta and return the slice of natural content that is now
    /// safe to stream (empty string when nothing is releasable yet or the
    /// stream has entered a tool region).
    pub fn push(&mut self, delta: &str) -> String {
        self.buf.push_str(delta);
        if self.in_tool_region {
            return String::new();
        }
        let markers: &[&str] = &self.markers;
        // Earliest fully-present marker after the emitted boundary.
        let mut first_marker_at: Option<usize> = None;
        for m in markers {
            if let Some(rel) = self.buf[self.emitted..].find(m) {
                let abs = self.emitted + rel;
                first_marker_at = Some(first_marker_at.map_or(abs, |cur| cur.min(abs)));
            }
        }
        if let Some(abs) = first_marker_at {
            // Stream content up to the marker, then buffer the rest.
            let out = self.buf[self.emitted..abs].to_string();
            self.emitted = abs;
            self.in_tool_region = true;
            return out;
        }
        // No complete marker: hold back the largest possible partial
        // marker prefix at the tail so we never stream half a marker.
        let mut hold = 0usize;
        for m in markers {
            hold = hold.max(partial_marker_overlap(&self.buf, m));
        }
        let safe_end = self.buf.len().saturating_sub(hold);
        if safe_end <= self.emitted {
            return String::new();
        }
        let out = self.buf[self.emitted..safe_end].to_string();
        self.emitted = safe_end;
        out
    }

    /// True once the stream has entered a tool-call region (the caller
    /// should stop emitting content deltas and prepare for tool_calls).
    pub fn in_tool_region(&self) -> bool {
        self.in_tool_region
    }

    /// Parse the full buffered output into structured calls. Any content
    /// already streamed via `push` is not repeated here.
    pub fn finalize(&self) -> Vec<ToolCall> {
        parse_tool_calls(self.format, &self.buf).calls
    }

    /// Buffered bytes that were withheld from streaming (the suspected
    /// tool region). When `finalize` yields no calls - e.g. a bare-JSON
    /// trigger fired on text that wasn't actually a tool call - the caller
    /// should flush this as content so nothing is lost.
    pub fn unstreamed(&self) -> &str {
        &self.buf[self.emitted.min(self.buf.len())..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool(name: &str) -> Tool {
        Tool::function(
            name.to_string(),
            Some("desc".into()),
            Some(serde_json::json!({"type":"object"})),
        )
    }

    #[test]
    fn detect_from_template() {
        assert_eq!(
            detect_tool_format(Some("foo [TOOL_CALLS] bar"), "x"),
            ToolFormat::Mistral
        );
        assert_eq!(
            detect_tool_format(Some("<tool_call>"), "x"),
            ToolFormat::Hermes
        );
        assert_eq!(
            detect_tool_format(Some("<|im_start|>"), "x"),
            ToolFormat::Hermes
        );
        assert_eq!(
            detect_tool_format(None, "devstral:latest"),
            ToolFormat::Mistral
        );
        assert_eq!(detect_tool_format(None, "llama-3.1-8b"), ToolFormat::Llama3);
        assert_eq!(detect_tool_format(None, "qwen3-coder"), ToolFormat::Hermes);
    }

    #[test]
    fn parse_hermes_single() {
        let raw = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
        let r = parse_tool_calls(ToolFormat::Hermes, raw);
        assert_eq!(r.calls.len(), 1);
        let f = r.calls[0].function.as_ref().unwrap();
        assert_eq!(f.name, "get_weather");
        assert_eq!(f.arguments.as_deref().unwrap(), "{\"city\":\"Paris\"}");
        assert_eq!(r.content, "Let me check.");
    }

    #[test]
    fn parse_hermes_multiple() {
        let raw = "<tool_call>{\"name\":\"a\",\"arguments\":{}}</tool_call><tool_call>{\"name\":\"b\",\"arguments\":{\"x\":1}}</tool_call>";
        let r = parse_tool_calls(ToolFormat::Hermes, raw);
        assert_eq!(r.calls.len(), 2);
        assert_eq!(r.calls[1].function.as_ref().unwrap().name, "b");
    }

    #[test]
    fn parse_hermes_unclosed() {
        let raw = "<tool_call>{\"name\":\"a\",\"arguments\":{\"k\":\"v\"}}";
        let r = parse_tool_calls(ToolFormat::Hermes, raw);
        assert_eq!(r.calls.len(), 1);
        assert_eq!(r.calls[0].function.as_ref().unwrap().name, "a");
    }

    #[test]
    fn parse_mistral_array() {
        let raw = "[TOOL_CALLS] [{\"name\": \"f\", \"arguments\": {\"a\": 1}}, {\"name\": \"g\", \"arguments\": {}}]";
        let r = parse_tool_calls(ToolFormat::Mistral, raw);
        assert_eq!(r.calls.len(), 2);
        assert_eq!(r.calls[0].function.as_ref().unwrap().name, "f");
        assert_eq!(r.calls[1].function.as_ref().unwrap().name, "g");
    }

    #[test]
    fn parse_llama3_object() {
        let raw = "{\"name\": \"search\", \"parameters\": {\"q\": \"rust\"}}";
        let r = parse_tool_calls(ToolFormat::Llama3, raw);
        assert_eq!(r.calls.len(), 1);
        assert_eq!(
            r.calls[0]
                .function
                .as_ref()
                .unwrap()
                .arguments
                .as_deref()
                .unwrap(),
            "{\"q\":\"rust\"}"
        );
    }

    #[test]
    fn parse_llama3_python_tag() {
        let raw = "<|python_tag|>{\"name\": \"f\", \"parameters\": {}}";
        let r = parse_tool_calls(ToolFormat::Llama3, raw);
        assert_eq!(r.calls.len(), 1);
    }

    #[test]
    fn parse_hermes_bare_json_fallback() {
        // Model ignored the <tool_call> instruction and emitted bare JSON.
        let raw = "\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Berlin\"}}\n";
        let r = parse_tool_calls(ToolFormat::Hermes, raw);
        assert_eq!(r.calls.len(), 1, "bare-JSON fallback should catch it");
        assert_eq!(r.calls[0].function.as_ref().unwrap().name, "get_weather");
        assert_eq!(r.content, "");
    }

    #[test]
    fn parse_mistral_with_surrounding_text() {
        // Exercises the marker-advance + content extraction (text before
        // and after the [TOOL_CALLS] block).
        let raw =
            "Sure, let me check.\n[TOOL_CALLS] [{\"name\":\"f\",\"arguments\":{\"a\":1}}]\nDone.";
        let r = parse_tool_calls(ToolFormat::Mistral, raw);
        assert_eq!(r.calls.len(), 1);
        assert_eq!(r.calls[0].function.as_ref().unwrap().name, "f");
        assert!(r.content.contains("Sure, let me check."));
        assert!(r.content.contains("Done."));
    }

    #[test]
    fn parse_mistral_with_close_trailer() {
        let raw = "[TOOL_CALLS] [{\"name\":\"f\",\"arguments\":{}}][/TOOL_CALLS]";
        let r = parse_tool_calls(ToolFormat::Mistral, raw);
        assert_eq!(r.calls.len(), 1);
        assert!(!r.content.contains("[/TOOL_CALLS]"));
    }

    #[test]
    fn parse_mistral_bare_json_fallback() {
        let raw = "{\"name\": \"f\", \"arguments\": {\"a\": 1}}";
        let r = parse_tool_calls(ToolFormat::Mistral, raw);
        assert_eq!(r.calls.len(), 1);
    }

    #[test]
    fn stream_bare_json_detected() {
        let mut sc = StreamToolScanner::new(ToolFormat::Hermes);
        // Pure bare-JSON tool call split across deltas; nothing streamed.
        assert_eq!(sc.push("{\"name\""), "");
        let _ = sc.push(": \"f\", \"arguments\": {\"x\": 1}}");
        assert!(sc.in_tool_region());
        assert_eq!(sc.finalize().len(), 1);
    }

    #[test]
    fn stream_bare_json_false_trigger_flushes() {
        let mut sc = StreamToolScanner::new(ToolFormat::Hermes);
        // Marker fires but the object never closes (not a real call).
        let _ = sc.push("{\"name\": \"x\"");
        assert!(sc.in_tool_region());
        // No balanced call object -> finalize empty; buffer is recoverable.
        assert!(sc.finalize().is_empty());
        assert_eq!(sc.unstreamed(), "{\"name\": \"x\"");
    }

    #[test]
    fn no_tool_call_is_plain_content() {
        let raw = "Just a normal answer with no tools.";
        let r = parse_tool_calls(ToolFormat::Hermes, raw);
        assert!(r.calls.is_empty());
        assert_eq!(r.content, raw);
    }

    #[test]
    fn balanced_object_ignores_braces_in_strings() {
        let s = "prefix {\"k\": \"a}b{c\"} suffix";
        assert_eq!(first_json_object(s).unwrap(), "{\"k\": \"a}b{c\"}");
    }

    #[test]
    fn stream_holds_partial_marker() {
        let mut sc = StreamToolScanner::new(ToolFormat::Hermes);
        // Content then a partial marker prefix split across deltas.
        assert_eq!(sc.push("Hello <to"), "Hello ");
        // "<to" is a prefix of "<tool_call>" - held back, nothing new safe.
        assert_eq!(sc.push("ol"), "");
        // Completes the marker - content already flushed, now in tool region.
        let _ = sc.push("_call>{\"name\":\"f\",\"arguments\":{}}</tool_call>");
        assert!(sc.in_tool_region());
        assert_eq!(sc.finalize().len(), 1);
    }

    #[test]
    fn stream_plain_content_flows() {
        let mut sc = StreamToolScanner::new(ToolFormat::Hermes);
        assert_eq!(sc.push("Hello "), "Hello ");
        assert_eq!(sc.push("world"), "world");
        assert!(!sc.in_tool_region());
        assert!(sc.finalize().is_empty());
    }

    #[test]
    fn flatten_injects_system() {
        let msgs = vec![Message::new("user".into(), "hi".into())];
        let out = flatten_messages(&msgs, ToolFormat::Hermes, &[tool("f")], None);
        assert_eq!(out[0].role, "system");
        assert!(out[0].content.contains("<tools>"));
        assert!(out[0].content.contains("\"f\""));
    }

    #[test]
    fn tool_choice_directive_shapes() {
        use serde_json::json;
        assert!(tool_choice_directive(None).is_none());
        assert!(tool_choice_directive(Some(&json!("auto"))).is_none());
        assert!(tool_choice_directive(Some(&json!("none"))).is_none());
        assert!(tool_choice_directive(Some(&json!("required")))
            .unwrap()
            .contains("at least one"));
        // OpenAI specific-function form.
        let d = tool_choice_directive(Some(&json!({"type":"function","function":{"name":"f"}})))
            .unwrap();
        assert!(d.contains("`f`"));
        // Anthropic forms.
        assert!(tool_choice_directive(Some(&json!({"type":"any"})))
            .unwrap()
            .contains("at least one"));
        assert!(
            tool_choice_directive(Some(&json!({"type":"tool","name":"g"})))
                .unwrap()
                .contains("`g`")
        );
    }

    #[test]
    fn directive_appended_to_system_block() {
        let msgs = vec![Message::new("user".into(), "hi".into())];
        let out = flatten_messages(
            &msgs,
            ToolFormat::Hermes,
            &[tool("f")],
            Some("You MUST call the `f` function."),
        );
        assert!(out[0].content.contains("You MUST call the `f` function."));
    }

    #[test]
    fn flatten_roundtrips_tool_calls() {
        let assistant = Message {
            role: "assistant".into(),
            content: String::new(),
            images: None,
            audios: None,
            thinking: None,
            reasoning_content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_1".into(),
                r#type: "function".into(),
                function: Some(ToolCallFunction {
                    name: "f".into(),
                    arguments: Some("{\"x\":1}".into()),
                }),
            }]),
            tool_call_id: None,
            name: None,
        };
        let tool_res = Message {
            role: "tool".into(),
            content: "42".into(),
            images: None,
            audios: None,
            thinking: None,
            reasoning_content: None,
            tool_calls: None,
            tool_call_id: Some("call_1".into()),
            name: None,
        };
        let out = flatten_messages(
            &[assistant, tool_res],
            ToolFormat::Hermes,
            &[tool("f")],
            None,
        );
        // system injected, then assistant with rendered call, then tool obs.
        assert!(out
            .iter()
            .any(|m| m.role == "assistant" && m.content.contains("<tool_call>")));
        assert!(out
            .iter()
            .any(|m| m.role == "tool" && m.content.contains("<tool_response>")));
    }

    #[test]
    fn parse_harmony_call_and_content() {
        let raw = "<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\": \"Paris\"}<|call|>";
        let r = parse_tool_calls(ToolFormat::Harmony, raw);
        assert_eq!(r.calls.len(), 1);
        let f = r.calls[0].function.as_ref().unwrap();
        assert_eq!(f.name, "get_weather");
        assert!(f.arguments.as_deref().unwrap().contains("Paris"));
        assert_eq!(r.content, "");
        assert_eq!(detect_tool_format(None, "gpt-oss:20b"), ToolFormat::Harmony);
    }

    #[test]
    fn parse_deepseek_fenced_call() {
        let raw = "Let me check.<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{\"city\": \"Paris\"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>";
        let r = parse_tool_calls(ToolFormat::DeepSeek, raw);
        assert_eq!(r.calls.len(), 1);
        assert_eq!(r.calls[0].function.as_ref().unwrap().name, "get_weather");
        assert_eq!(r.content, "Let me check.");
        assert_eq!(
            detect_tool_format(None, "deepseek-r1:70b"),
            ToolFormat::DeepSeek
        );
    }
}
