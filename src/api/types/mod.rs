//! API types for the LLM server
//!
//! This module defines the data structures used in the API.
//! Supports Ollama-compatible API format.

use serde::{Deserialize, Deserializer, Serialize};
use validator::Validate;

/// Deserialise a `keep_alive` field that may arrive as either a string
/// ("0", "5m", "10s") or an integer (0, 60, -1) - Ollama's HTTP API
/// accepts both forms, and the bench / many third-party clients send
/// integers. Without this, the server rejects integer values with
/// `invalid type: integer 0, expected a string` and the model never
/// gets unloaded - which then strands its VRAM and corrupts subsequent
/// bench runs.
fn deserialize_keep_alive<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(match v {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::String(s)) => Some(s),
        Some(serde_json::Value::Number(n)) => Some(n.to_string()),
        // Bool/Array/Object: nothing parse_keep_alive() can use. Returning
        // the JSON-stringified form (`true` -> "true", `{...}` -> JSON blob)
        // would fall through to the parser's `value.parse().ok()` failure
        // path and end up as None anyway - but for an Object payload that
        // means re-serializing the whole nested value just to throw it
        // away. Short-circuit to None.
        Some(_) => None,
    })
}

/// Deserialize a seed field that some SDKs send as a negative integer
/// (e.g. `-1` to mean "use a random seed"). Plain `Option<u64>` would
/// 422 the whole request because u64 can't hold a negative. Accept
/// any integer, map negatives + null to None (engine then picks its
/// own seed), positives get cast.
pub fn deserialize_optional_seed<'de, D>(d: D) -> Result<Option<u64>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(match v {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Number(n)) => {
            if let Some(u) = n.as_u64() {
                Some(u)
            } else if n.as_i64().is_some_and(|i| i < 0) {
                None
            } else {
                // Float or out-of-range - treat as no-seed.
                None
            }
        }
        // Strings/objects are not valid seed types - ignore.
        _ => None,
    })
}

/// Map a JSON string or `null`/absent into a `String`, with null/absent
/// becoming `""`. Lets assistant turns that only carry `tool_calls` (and
/// send `content: null`, per the OpenAI tool-calling contract) deserialize
/// instead of failing on the non-optional `content: String` field.
fn deserialize_nullable_string<'de, D>(d: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<String>::deserialize(d)?;
    Ok(v.unwrap_or_default())
}

/// Ollama 0.5+ accepts `thinking: true|false` (bool) AND
/// `thinking: "enabled"|"disabled"` (string) for thinking-model
/// preferences. Plain `Option<String>` would 422 the bool form.
/// Normalise to canonical strings: `Some("enabled")` / `Some("disabled")`
/// / passthrough custom strings. Null + unset -> None.
pub fn deserialize_optional_thinking<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(match v {
        None | Some(serde_json::Value::Null) => None,
        Some(serde_json::Value::Bool(true)) => Some("enabled".to_string()),
        Some(serde_json::Value::Bool(false)) => Some("disabled".to_string()),
        Some(serde_json::Value::String(s)) => Some(s),
        // Other types (number/array/object) are ignored - caller
        // should pass a documented form.
        _ => None,
    })
}

// ============================================================================
// Common Types
// ============================================================================

/// Model source for pull/list operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ModelSource {
    /// Ollama models
    Ollama,
    /// HuggingFace models
    HuggingFace,
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelSource::Ollama => write!(f, "ollama"),
            ModelSource::HuggingFace => write!(f, "huggingface"),
        }
    }
}

/// Structured output configuration for JSON schema enforcement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredOutput {
    /// Type of structured output: "json" for basic JSON, "json_schema" for schema validation
    pub output_type: String,
    /// JSON schema definition (when output_type is "json_schema")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<serde_json::Value>,
}

impl StructuredOutput {
    /// Create a simple JSON output request
    pub fn json() -> Self {
        Self {
            output_type: "json".to_string(),
            json_schema: None,
        }
    }

    /// Create a JSON schema-constrained output request
    pub fn json_schema(schema: serde_json::Value) -> Self {
        Self {
            output_type: "json_schema".to_string(),
            json_schema: Some(schema),
        }
    }
}

/// Tool definition for tool calling / function calling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub r#type: String, // "function"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolFunction>,
}

/// Tool function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>, // JSON schema for parameters
}

impl Tool {
    /// Create a function tool
    pub fn function(
        name: String,
        description: Option<String>,
        parameters: Option<serde_json::Value>,
    ) -> Self {
        Self {
            r#type: "function".to_string(),
            function: Some(ToolFunction {
                name,
                description,
                parameters,
            }),
        }
    }
}

/// Model's tool call in response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String, // "function"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunction>,
}

/// Function call details in model response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>, // JSON string of arguments
}

// ============================================================================
// Ollama-compatible Types
// ============================================================================

/// Ollama API default for `stream`: true if not specified.
fn default_stream_true() -> bool {
    true
}

/// Ollama chat request (POST /api/chat)
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct OllamaChatRequest {
    pub model: String,
    /// Report each token's log-probability, with `top_logprobs` alternatives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,
    /// Same 4096-entry cap + per-Message nested validation as
    /// ChatCompletionRequest.messages (231b3a4). Recursively validates
    /// each Message (role/content min length, images max length).
    #[validate(length(max = 4096), nested)]
    pub messages: Vec<Message>,
    #[serde(default = "default_stream_true")]
    pub stream: bool,
    /// Ollama's `format`: accepts the literal string `"json"` (simple
    /// JSON-object mode) OR a JSON Schema object for structured output.
    /// Stored as Value so both shapes deserialize cleanly - was
    /// `Option<String>` which rejected the object form with a 422.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Value>,
    /// Keep model in memory for this duration (in minutes).
    /// 0 = unload immediately, -1 = keep forever, None = use default
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_keep_alive"
    )]
    pub keep_alive: Option<String>,
    /// Thinking preferences for thinking models - accepts both
    /// `true`/`false` (Ollama 0.5+ bool form) and `"enabled"`/`"disabled"`
    /// (legacy string form). Custom deserializer normalises both.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_thinking"
    )]
    pub thinking: Option<String>,
    /// The name an Ollama client actually sends. Carried as its own field rather than an
    /// alias, because a caller may send both and an alias makes that a duplicate-field
    /// rejection. Read through [`Self::thinking_preference`], never directly.
    #[serde(
        default,
        rename = "think",
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_thinking"
    )]
    pub think: Option<String>,
    /// Tools available for the model to call (function calling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// OpenAI's `tool_choice`, taken here too: `none`, `auto`, `required`, or one
    /// named function.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

impl OllamaChatRequest {
    pub fn new(model: String, messages: Vec<Message>) -> Self {
        Self {
            model,
            messages,
            stream: false,
            format: None,
            options: None,
            keep_alive: None,
            thinking: None,
            think: None,
            tools: None,
            logprobs: None,
            top_logprobs: None,
            tool_choice: None,
        }
    }
}

/// Ollama chat response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    pub created_at: String,
    pub message: Message,
    pub done: bool,
    /// Per token, when the request asked for them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    /// Reason request ended: "stop", "load", "unload" (or null if streaming/not done)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    // `skip_serializing_if` keeps load/unload + intermediate-stream
    // responses lean - without it, every `None` timing emits as
    // `"total_duration": null` etc., which real Ollama omits from
    // the wire. `#[serde(default)]` stays so deserialization still
    // tolerates the field's absence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
    /// Model's thinking output (for thinking models like Deepseek-R1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Time spent thinking (nanoseconds, for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_duration: Option<u64>,
    /// Structured output info if format was requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<String>,
    /// Tool calls made by the model (for tool calling / function calling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Vision capabilities info if images were processed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_processed: Option<bool>,
}

impl OllamaChatResponse {
    pub fn new(model: String, message: Message) -> Self {
        Self {
            model,
            created_at: chrono::Utc::now().to_rfc3339(),
            message,
            done: true,
            done_reason: None,
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
            thinking: None,
            thinking_duration: None,
            logprobs: None,
            structured_output: None,
            tool_calls: None,
            vision_processed: None,
        }
    }
}

/// Ollama generate request (POST /api/generate)
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct OllamaGenerateRequest {
    pub model: String,
    /// Report each token's log-probability, with `top_logprobs` alternatives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<usize>,
    /// System prompt, rendered as the model's system turn as Ollama renders it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// A per-request template: Jinja, or the Go form Ollama modelfiles use, of which
    /// the `.System`, `.Prompt`, `.Response` fields and `if`/`else`/`end` are rendered.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    /// Cap matches handler-side guard at 256 KiB chars (~64k tokens at
    /// ~4 chars/token). Comfortable within any practical context the
    /// perimeter models support. validate(length) is byte-based on
    /// String - a CJK-heavy prompt under 64K chars can still trip the
    /// 256 KiB byte cap, which is the intended belt-and-braces behaviour.
    ///
    /// OPTIONAL, because `{"model": X, "keep_alive": 0}` with no prompt is the
    /// documented way to unload a model - and the handler below already implements
    /// it under `prompt.is_empty()`. Requiring the field made that branch
    /// unreachable: the call was rejected at deserialisation with "missing field
    /// `prompt`", so the one supported way to free VRAM did not work.
    #[validate(length(max = 262144))]
    #[serde(default)]
    pub prompt: String,
    #[serde(default = "default_stream_true")]
    pub stream: bool,
    /// Ollama's `format`: accepts the literal string `"json"` (simple
    /// JSON-object mode) OR a JSON Schema object for structured output.
    /// See OllamaChatRequest.format for the same rationale.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Value>,
    /// Keep model in memory for this duration (in minutes).
    /// 0 = unload immediately, -1 = keep forever, None = use default
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_keep_alive"
    )]
    pub keep_alive: Option<String>,
    /// Thinking preferences for thinking models - accepts both
    /// `true`/`false` (Ollama 0.5+ bool form) and `"enabled"`/`"disabled"`
    /// (legacy string form). Custom deserializer normalises both.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_thinking"
    )]
    pub thinking: Option<String>,
    /// The name an Ollama client actually sends. Carried as its own field rather than an
    /// alias, because a caller may send both and an alias makes that a duplicate-field
    /// rejection. Read through [`Self::thinking_preference`], never directly.
    #[serde(
        default,
        rename = "think",
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_thinking"
    )]
    pub think: Option<String>,
    /// Images for vision models (base64-encoded strings, Ollama format).
    /// Same per-request 16-image cap as Message.images / handler-side guard.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(length(max = 16))]
    pub images: Option<Vec<String>>,
    /// If true, bypass chat template application (Ollama spec).
    /// When false/unset, chat-tuned models auto-wrap the prompt as a single user turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<bool>,
    /// Ollama's continuation `context`: the token sequence a previous reply returned,
    /// replayed as the conversation so far. Prepended at TOKEN level - a detokenise /
    /// retokenise round trip does not survive every tokenizer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    /// Fill-in-the-middle: when set, `prompt` is the prefix and `suffix` is the
    /// text that should follow the model-generated middle. The handler wraps
    /// them with the model's FIM sentinel tokens and applies FIM-tuned defaults
    /// (short max_tokens, low temperature, sentinel stop sequences).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
}

impl OllamaChatRequest {
    /// What the caller asked for about reasoning, from whichever field they used.
    ///
    /// `think` is the Ollama name and wins; `thinking` is accepted for the clients that
    /// send it. One accessor so a handler cannot read only one of the two - which is how
    /// the flag came to be honoured on one endpoint and dropped on the other.
    pub fn thinking_preference(&self) -> Option<&str> {
        self.think.as_deref().or(self.thinking.as_deref())
    }
}

impl OllamaGenerateRequest {
    /// See [`OllamaChatRequest::thinking_preference`].
    pub fn thinking_preference(&self) -> Option<&str> {
        self.think.as_deref().or(self.thinking.as_deref())
    }
}

impl OllamaGenerateRequest {
    pub fn new(model: String, prompt: String) -> Self {
        Self {
            model,
            prompt,
            stream: false,
            format: None,
            options: None,
            keep_alive: None,
            thinking: None,
            think: None,
            images: None,
            raw: None,
            context: None,
            suffix: None,
            logprobs: None,
            top_logprobs: None,
            system: None,
            template: None,
        }
    }
}

/// Ollama generate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaGenerateResponse {
    pub model: String,
    pub created_at: String,
    pub response: String,
    pub done: bool,
    /// Per token, when the request asked for them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
    /// Reason request ended: "stop", "load", "unload" (or null if streaming/not done)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub done_reason: Option<String>,
    // Same `skip_serializing_if = is_none` sweep as OllamaChatResponse  -
    // keeps load/unload + intermediate-stream responses lean.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<Vec<i32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_duration: Option<u64>,
    /// Model's thinking output (for thinking models like Deepseek-R1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// Time spent thinking (nanoseconds, for thinking models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_duration: Option<u64>,
    /// Structured output info if format was requested
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_output: Option<String>,
    /// Tool calls made by the model (for tool calling / function calling)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Vision capabilities info if images were processed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vision_processed: Option<bool>,
    /// Generated images (base64-encoded, for image generation models like Flux)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub images: Option<Vec<String>>,
    /// Generated audio payloads (base64-encoded WAV) for TTS pipeline
    /// responses. Mirror of `Message.audios` on the chat-shape response;
    /// surfaced here too so /api/generate clients can drive TTS without
    /// needing the chat envelope.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audios: Option<Vec<String>>,
}

impl OllamaGenerateResponse {
    pub fn new(model: String, response: String) -> Self {
        Self {
            model,
            created_at: chrono::Utc::now().to_rfc3339(),
            response,
            done: true,
            done_reason: None,
            context: None,
            total_duration: None,
            load_duration: None,
            prompt_eval_count: None,
            prompt_eval_duration: None,
            eval_count: None,
            eval_duration: None,
            thinking: None,
            thinking_duration: None,
            logprobs: None,
            structured_output: None,
            tool_calls: None,
            vision_processed: None,
            images: None,
            audios: None,
        }
    }
}

/// Ollama model information (from /api/tags)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModel {
    pub name: String,
    /// The same identifier under the key current clients read.
    pub model: String,
    pub modified_at: String,
    pub size: u64,
    #[serde(default)]
    pub digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<OllamaModelDetails>,
    /// Model source
    #[serde(default = "default_model_source_str")]
    pub source: String,
    /// What this model can DO ("chat", "txt2img", "img2img", "edit", "sfx",
    /// "music", "video", "tts", "asr", "embedding", ...). The server is the
    /// single authority; clients build their pickers from this instead of
    /// hardcoding model lists. Absent on Ollama-compat consumers' radar
    /// (extra JSON fields are ignored).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<String>,
    /// Recommended per-model defaults (e.g. {"steps": 27, "cfg": 1.0}) so
    /// clients can preset knobs without knowing checkpoints by name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub defaults: Option<serde_json::Value>,
}

fn default_model_source_str() -> String {
    "ollama".to_string()
}

impl OllamaModel {
    pub fn new(name: String, size: u64, modified_at: String) -> Self {
        Self {
            capabilities: Vec::new(),
            defaults: None,
            model: name.clone(),
            name,
            modified_at,
            size,
            digest: String::new(),
            details: None,
            source: "ollama".to_string(),
        }
    }

    pub fn with_source(mut self, source: ModelSource) -> Self {
        self.source = source.to_string();
        self
    }

    pub fn with_digest(mut self, digest: String) -> Self {
        self.digest = digest;
        self
    }

    pub fn with_details(mut self, details: OllamaModelDetails) -> Self {
        self.details = Some(details);
        self
    }
}

/// Ollama model details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaModelDetails {
    pub format: String,
    pub family: String,
    /// Every family the model belongs to. A vision model lists its tower beside its language
    /// family; most models list one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub families: Option<Vec<String>>,
    pub parameter_size: String,
    /// The label the file is stored under, "Q4_K_M" and the like. Omitted rather than sent as
    /// null when the checkpoint does not say.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization_level: Option<String>,
}

/// Ollama list models response (GET /api/tags)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaListModelsResponse {
    pub models: Vec<OllamaModel>,
}

impl OllamaListModelsResponse {
    pub fn new(models: Vec<OllamaModel>) -> Self {
        Self { models }
    }
}

/// Ollama pull request (POST /api/pull)
fn pull_stream_default() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPullRequest {
    #[serde(alias = "model")]
    pub name: String,
    /// Progress frames by default, as Ollama streams them; `false` answers once.
    #[serde(default = "pull_stream_default")]
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub insecure: Option<bool>,
    /// Model source: "ollama" or "huggingface"
    #[serde(default = "default_model_source")]
    pub source: String,
}

fn default_model_source() -> String {
    "ollama".to_string()
}

impl OllamaPullRequest {
    pub fn new(name: String) -> Self {
        Self {
            name,
            stream: false,
            insecure: None,
            source: "ollama".to_string(),
        }
    }

    pub fn with_source(mut self, source: ModelSource) -> Self {
        self.source = source.to_string();
        self
    }
}

/// Ollama pull response (streaming status)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPullResponse {
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed: Option<u64>,
}

/// Ollama show model request (POST /api/show)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaShowRequest {
    /// Primary field for model name (preferred over name)
    #[serde(default)]
    pub model: String,
    /// Deprecated field for model name (use model instead)
    #[serde(default)]
    pub name: String,
    /// With it, `model_info` carries the tokenizer's vocabulary arrays too.
    #[serde(default)]
    pub verbose: bool,
}

/// Ollama show model response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaShowResponse {
    // Most fields are optional in the spec - emit only when populated
    // so /api/show responses for models without a license/modelfile/
    // parameters layer don't carry explicit nulls. Real Ollama omits
    // the missing layers entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modelfile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub template: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<OllamaModelDetails>,
    /// The checkpoint's own metadata, keyed as the file keys it.
    ///
    /// Clients read architecture, context length and head geometry from here rather than from
    /// a field per property, so what a file declares reaches them whether or not this server
    /// has a name for it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_info: Option<serde_json::Map<String, serde_json::Value>>,
    /// Ollama's documented capabilities array - modern SDK clients
    /// read this to decide whether to surface "Chat with image",
    /// "Use as embedder", etc. in their UI. Omit when None so the
    /// response shape stays compact for clients that don't read it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Vec<String>>,
}

/// Ollama delete request (DELETE /api/delete)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaDeleteRequest {
    /// `model` is what current clients send; `name` what earlier ones did.
    #[serde(alias = "model")]
    pub name: String,
}

// ============================================================================
// Common Types (used by both Ollama and OpenAI-compatible APIs)
// ============================================================================

/// Image content for multimodal messages (vision models)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageContent {
    /// URL to the image (can be HTTP URL or data:image/... URI)
    pub url: String,
    /// Optional base64-encoded image data (for inline images)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base64: Option<String>,
}

impl ImageContent {}

/// Chat message (compatible with both Ollama and OpenAI)
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
#[serde(try_from = "MessageWire")]
pub struct Message {
    #[validate(length(min = 1))]
    pub role: String,
    /// Message text. Tolerant of absent/null/empty content: an assistant
    /// turn that only invokes tools carries `tool_calls` with empty
    /// `content`, and OpenAI clients send `content: null` on such turns.
    /// `deserialize_nullable_string` maps null->"" so these round-trip
    /// instead of 422'ing; the `min=1` length check was dropped for the
    /// same reason. Empty content is harmless downstream - the chat
    /// formatters just emit the role markers.
    pub content: String,
    /// The model's reasoning, in the field the Ollama surface returns it in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking: Option<String>,
    /// The same reasoning, in the field the OpenAI surface returns it in.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Images for vision models (base64-encoded strings, Ollama format).
    /// Per-message cap of 16 matches the handler-side guard (fbe25c5).
    /// Vision encoder cost is O(images x tokens_per_image); typical use
    /// is 1-2 images. The 16-entry ceiling matches the /v1/images n-cap.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(length(max = 16))]
    pub images: Option<Vec<String>>,
    /// Audio payloads (base64-encoded WAV) returned for TTS pipeline
    /// responses so the chat tab can render a playback control inline.
    /// Same cap shape as images for symmetry; in practice always
    /// length-1 since TTS produces a single utterance per request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(length(max = 16))]
    pub audios: Option<Vec<String>>,
    /// Tool calls emitted by the model (assistant turns) or replayed from
    /// conversation history. Populated in responses when the model invokes
    /// a function; accepted in requests so a multi-turn agentic loop can
    /// send back the assistant's prior calls. Serialized inside the
    /// `message` object per the OpenAI contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// For `role: "tool"` result messages - the id of the tool call this
    /// message answers. Accepted (and echoed) for round-trip fidelity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Optional function/tool name on `role: "tool"`/`"function"` result
    /// messages (legacy OpenAI function-calling shape).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A message as OpenAI clients send it: `content` is a string, null, or an array of
/// parts (`text`, `image_url`). Parts are folded into `Message`: the texts joined, the
/// images carried as base64. What is not carried is refused rather than dropped, so a
/// client learns at once that a remote image or an audio part reached a server that
/// does not take them.
#[derive(Deserialize)]
struct MessageWire {
    role: String,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    thinking: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
    #[serde(default)]
    images: Option<Vec<String>>,
    #[serde(default)]
    audios: Option<Vec<String>>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

/// Text and base64 images of a `content` value.
fn fold_content_parts(v: Option<serde_json::Value>) -> Result<(String, Vec<String>), String> {
    use serde_json::Value;
    let parts = match v {
        None | Some(Value::Null) => return Ok((String::new(), Vec::new())),
        Some(Value::String(s)) => return Ok((s, Vec::new())),
        Some(Value::Array(parts)) => parts,
        Some(_) => return Err("`content` must be a string or an array of parts".to_string()),
    };
    let mut text = String::new();
    let mut images = Vec::new();
    for part in parts {
        let kind = part.get("type").and_then(Value::as_str).unwrap_or("");
        match kind {
            "text" => {
                let t = part
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or("a `text` part needs a `text` string")?;
                if !text.is_empty() {
                    text.push('\n');
                }
                text.push_str(t);
            }
            "image_url" => {
                let url = match part.get("image_url") {
                    Some(Value::String(u)) => u.as_str(),
                    Some(Value::Object(o)) => o.get("url").and_then(Value::as_str).unwrap_or(""),
                    _ => "",
                };
                let Some(b64) = data_url_base64(url) else {
                    return Err(
                        "`image_url` must be a data: URL carrying base64; remote images are not fetched"
                            .to_string(),
                    );
                };
                images.push(b64.to_string());
            }
            other => return Err(format!("unsupported content part type '{other}'")),
        }
    }
    Ok((text, images))
}

/// The base64 payload of a `data:<type>;base64,<payload>` URL.
pub(crate) fn data_url_base64(url: &str) -> Option<&str> {
    let rest = url.strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(',')?;
    meta.ends_with(";base64").then_some(payload)
}

impl TryFrom<MessageWire> for Message {
    type Error = String;
    fn try_from(w: MessageWire) -> Result<Self, String> {
        let (content, mut images) = fold_content_parts(w.content)?;
        if let Some(more) = w.images {
            images.extend(more);
        }
        Ok(Self {
            role: w.role,
            content,
            thinking: w.thinking,
            reasoning_content: w.reasoning_content,
            images: (!images.is_empty()).then_some(images),
            audios: w.audios,
            tool_calls: w.tool_calls,
            tool_call_id: w.tool_call_id,
            name: w.name,
        })
    }
}

impl Message {
    pub fn new(role: String, content: String) -> Self {
        Self {
            role,
            content,
            thinking: None,
            reasoning_content: None,
            images: None,
            audios: None,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    /// Create an assistant message carrying tool calls (used to build
    /// tool-calling responses). Content may be empty when the model emits
    /// only tool calls.
    pub fn with_tool_calls(content: String, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            role: "assistant".to_string(),
            content,
            thinking: None,
            reasoning_content: None,
            images: None,
            audios: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
        }
    }
}

/// Chat completion request (OpenAI-compatible format, also accepted)
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ChatCompletionRequest {
    #[validate(length(min = 1))]
    pub model: String,
    /// Recursively validates each Message - without `nested`, the
    /// per-Message role/content min-length checks don't fire when
    /// the chat request validator runs, and an empty content would
    /// reach format_chat_prompt unguarded.
    ///
    /// `max = 4096` matches the handler-side count cap (chat_completion
    /// in api/handlers/). A pathological 100k-message history is almost
    /// certainly a client bug or attack; per-message overhead (template
    /// tokens, role headers) compounds even for short content.
    #[validate(length(min = 1, max = 4096), nested)]
    pub messages: Vec<Message>,
    #[validate(range(min = 0.0, max = 2.0))]
    pub temperature: Option<f32>,
    /// Upper bound matches the largest practical decode budget across
    /// the perimeter (128K-context models). Without a max, a client
    /// sending `max_tokens: usize::MAX` parses fine here, then the
    /// engine's `for _ in 0..max_tokens` loop never terminates.
    #[validate(range(min = 1, max = 131072))]
    pub max_tokens: Option<usize>,
    /// The cap current OpenAI clients send instead of `max_tokens`; the two are
    /// read through `completion_cap`, this one first.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[validate(range(min = 1, max = 131072))]
    pub max_completion_tokens: Option<usize>,
    /// Token id (as a string, as OpenAI keys it) to a bias added to its logit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// OpenAI's reasoning effort; `none` and `minimal` switch a reasoning model's
    /// thinking off, the other levels leave it on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Output modalities; only text is produced here, audio is refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modalities: Option<Vec<String>>,
    pub stream: Option<bool>,
    /// OpenAI-compatible session identifier. When the same id is sent on
    /// consecutive turns, the KV cache is reused and only new tokens get
    /// prefilled.
    #[serde(default)]
    pub user: Option<String>,
    /// OpenAI's stream_options object. Only field consumed today is
    /// `include_usage`; presence triggers emission of the trailing
    /// usage-only chunk per spec.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    /// Nucleus sampling probability mass. OpenAI clamps to [0, 1];
    /// values < 1 narrow the candidate set during sampling.
    #[validate(range(min = 0.0, max = 1.0))]
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Per-request RNG seed. With `temperature > 0` the same seed
    /// produces the same continuation given the same context.
    /// Custom deserializer accepts any integer (including negative
    /// "random" sentinels like -1 that some SDKs send) and clamps
    /// to None for negative values, instead of 422'ing the whole
    /// request at the serde layer for an `Option<u64>`.
    #[serde(default, deserialize_with = "deserialize_optional_seed")]
    pub seed: Option<u64>,
    /// Stop sequences: when the generator emits any of these strings,
    /// streaming halts and `finish_reason="stop"`. Accepts a single
    /// string or an array of strings.
    #[serde(default)]
    pub stop: Option<StopSequences>,
    /// OpenAI's structured-output knob. Currently supports
    /// `{"type":"json_object"}` (biases output to a valid JSON object
    /// via the llguidance grammar engine) and the new
    /// `{"type":"json_schema","json_schema":{"schema":...}}` shape.
    /// `{"type":"text"}` is a no-op (the default).
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    /// OpenAI's repetition penalty by *frequency* of past tokens.
    /// Range [-2.0, 2.0]; default 0.0. Folded into our llama.cpp-style
    /// `repeat_penalty` knob.
    #[validate(range(min = -2.0, max = 2.0))]
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// OpenAI's repetition penalty by *presence* of past tokens.
    /// Range [-2.0, 2.0]; default 0.0. Same knob mapping as
    /// frequency_penalty - we only have one repetition lever.
    #[validate(range(min = -2.0, max = 2.0))]
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// OpenAI's `n`: number of completions to return per call. The
    /// engine generates one sequence per request; supporting `n > 1`
    /// requires either repeating the sampler or batched-sampling, both
    /// of which are out of scope for the current decode loop.
    /// `max = 1` enforces the cap at the type level - handler-side
    /// returns the same 400 envelope with a friendlier "issue {n}
    /// requests in parallel" hint.
    #[validate(range(min = 1, max = 1))]
    #[serde(default)]
    pub n: Option<u32>,
    /// OpenAI's `logprobs` (chat-shape boolean). The sampler doesn't
    /// surface per-token top-K data, so the handler rejects `true`
    /// with a 400 - preferable to silently returning `"logprobs":null`
    /// for callers expecting populated values. Explicit `false`/null
    /// is accepted.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// OpenAI's `top_logprobs`: integer count of top alternative tokens
    /// to report per position. Rejected with 400 when > 0 for the same
    /// reason as `logprobs`.
    #[serde(default)]
    pub top_logprobs: Option<u32>,
    /// OpenAI's `tools` array - function-calling spec. Accepted but a
    /// non-empty array is rejected with 400 by the handler since the
    /// engine doesn't have a tool-call decode path. Empty array passes
    /// (SDKs often include it by default).
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// OpenAI's `tool_choice`: `"auto"` | `"none"` | `"required"` | a
    /// specific `{"type":"function","function":{"name":...}}` object.
    /// Stored as a raw Value; the handler treats `"none"` as "don't inject
    /// tools" and everything else as "auto" (the engine can't force a
    /// specific call, so `required`/specific degrade to auto).
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    /// OpenAI's `parallel_tool_calls` - accepted and ignored (the parser
    /// already lifts every emitted call, so parallel calls work either
    /// way). Stored to avoid `deny_unknown_fields`-style rejections.
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    /// OpenAI's `service_tier`: "auto" | "default" | "scale". This
    /// server only operates one tier, but we reflect the requested
    /// value back in the response so SDK clients see the round-trip
    /// match OpenAI's behaviour. Stored as a String so future tier
    /// names don't require a struct change.
    #[serde(default)]
    pub service_tier: Option<String>,
}

impl ChatCompletionRequest {
    /// The completion cap a request asks for, `max_completion_tokens` before the
    /// `max_tokens` it superseded.
    pub fn completion_cap(&self) -> Option<usize> {
        self.max_completion_tokens.or(self.max_tokens)
    }
}

/// OpenAI's `stop` field accepts either a single string or an array.
/// We deserialize into either shape and flatten downstream.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StopSequences {
    One(String),
    Many(Vec<String>),
}

impl StopSequences {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            StopSequences::One(s) => vec![s],
            StopSequences::Many(v) => v,
        }
    }
}

/// Streaming options object - matches OpenAI's
/// `stream_options.include_usage`. Defaults to None / false so the
/// trailing usage chunk is omitted unless the client opts in.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

impl ChatCompletionRequest {
    pub fn new(model: String, messages: Vec<Message>) -> Self {
        Self {
            model,
            messages,
            temperature: None,
            max_tokens: None,
            stream: None,
            user: None,
            stream_options: None,
            top_p: None,
            seed: None,
            stop: None,
            response_format: None,
            frequency_penalty: None,
            presence_penalty: None,
            n: None,
            logprobs: None,
            top_logprobs: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            service_tier: None,
            max_completion_tokens: None,
            logit_bias: None,
            reasoning_effort: None,
            modalities: None,
        }
    }

    /// Convert to Ollama format. Maps the OpenAI-shape sampling
    /// knobs into the Ollama `options` bag so the converted request
    /// preserves user intent - previously this dropped every
    /// parameter except model + messages.
    pub fn to_ollama(self) -> OllamaChatRequest {
        let mut opts = serde_json::Map::new();
        if let Some(t) = self.temperature {
            opts.insert("temperature".to_string(), serde_json::json!(t));
        }
        if let Some(p) = self.top_p {
            opts.insert("top_p".to_string(), serde_json::json!(p));
        }
        if let Some(n) = self.max_tokens {
            opts.insert("num_predict".to_string(), serde_json::json!(n));
        }
        if let Some(s) = self.seed {
            opts.insert("seed".to_string(), serde_json::json!(s));
        }
        if let Some(ref stop) = self.stop {
            let v = match stop {
                StopSequences::One(s) => vec![s.clone()],
                StopSequences::Many(vs) => vs.clone(),
            };
            opts.insert("stop".to_string(), serde_json::json!(v));
        }
        // Fold OpenAI's two repetition penalties into Ollama's single
        // repeat_penalty knob using the same +0.0 -> 1.0 / +2.0 -> 3.0 /
        // -2.0 -> -1.0 linear mapping the chat handler uses. Clamp the
        // sum to [-2, 2] before adding 1.0 - without the clamp the
        // converted Ollama request could carry repeat_penalty = 5.0
        // (both penalties at +2 sum to +4, +1 = 5), which is outside
        // the engine's expected range. Matches api/handlers/::chat_completion
        // (line ~3643) so direct-call vs cross-format paths produce
        // the same engine input.
        let oa_pen = (self.frequency_penalty.unwrap_or(0.0) + self.presence_penalty.unwrap_or(0.0))
            .clamp(-2.0, 2.0);
        if oa_pen.abs() > f32::EPSILON {
            opts.insert(
                "repeat_penalty".to_string(),
                serde_json::json!(1.0_f32 + oa_pen),
            );
        }
        if let Some(ref u) = self.user {
            opts.insert("session_id".to_string(), serde_json::json!(u));
        }
        let mut req = OllamaChatRequest::new(self.model, self.messages);
        if !opts.is_empty() {
            req.options = Some(serde_json::Value::Object(opts));
        }
        req.stream = self.stream.unwrap_or(false);
        req
    }
}

/// Chat completion response (OpenAI-compatible format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    /// OpenAI's deterministic-output fingerprint. SDKs sometimes
    /// check it to detect backend changes between calls. We don't
    /// run different backend versions, so a fixed identifying string
    /// is fine - clients that compare across calls just see no
    /// change.
    pub system_fingerprint: String,
    /// OpenAI's `service_tier` - "default" / "auto" / "scale" for
    /// the new tiered routing. We don't tier traffic; emitting
    /// "default" keeps SDK shape tests passing without misleading
    /// callers about non-existent tiers.
    pub service_tier: String,
}

impl ChatCompletionResponse {
    pub fn new(id: String, model: String, choices: Vec<Choice>, usage: Usage) -> Self {
        Self {
            id,
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model,
            choices,
            usage,
            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
            service_tier: "default".to_string(),
        }
    }

    /// Create from Ollama response
    pub fn from_ollama(ollama: OllamaChatResponse) -> Self {
        // Carry over the real prompt/eval counts from the source
        // OllamaChatResponse rather than hardcoding `prompt_tokens: 0`
        // - clients consuming the converted response would otherwise
        // see usage that doesn't add up.
        let prompt_tokens = ollama.prompt_eval_count.unwrap_or(0) as i32;
        let completion_tokens = ollama.eval_count.unwrap_or(0) as i32;
        // Translate `done_reason` if set - "length" maps to OpenAI's
        // length termination, anything else falls back to "stop"
        // (matches the OpenAI default for end-of-turn).
        let finish_reason = match ollama.done_reason.as_deref() {
            Some("length") => "length",
            _ => "stop",
        };
        Self {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
            object: "chat.completion".to_string(),
            created: chrono::Utc::now().timestamp(),
            model: ollama.model,
            choices: vec![Choice::new(0, ollama.message, finish_reason.to_string())],
            usage: Usage::new(prompt_tokens, completion_tokens),
            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
            service_tier: "default".to_string(),
        }
    }
}

/// Stable identifier surfaced as OpenAI's `system_fingerprint` field.
/// Tied to the binary version so clients can detect backend updates
/// between calls (helpful when comparing deterministic seed runs).
const SYSTEM_FINGERPRINT: &str = concat!("loken-", env!("CARGO_PKG_VERSION"));

/// Chat completion choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Choice {
    pub index: usize,
    pub message: Message,
    pub finish_reason: String,
    /// OpenAI returns `logprobs: null` on every choice when the
    /// caller didn't request logprobs. Some SDK tests do equality
    /// comparison against a reference object that includes this
    /// field; emitting null keeps parity. `None` here serializes as
    /// `null` (no `skip_serializing_if`).
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
}

// ============================================================================
// OpenAI-compatible Streaming Types
// ============================================================================

/// Streaming chat completion chunk (OpenAI-compatible SSE format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// Only present in the final chunk with usage info
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    pub system_fingerprint: String,
    /// OpenAI's documented stream-chunk `service_tier` - matches the
    /// non-stream `ChatCompletionResponse.service_tier`. Default
    /// "default"; surfaced on every chunk for spec-parity (clients
    /// reading it across both surfaces see consistent values).
    pub service_tier: String,
}

impl ChatCompletionChunk {
    /// Create a content delta chunk. `created` is supplied by the
    /// handler so every chunk in a single SSE stream carries the same
    /// timestamp (OpenAI's spec - clients use `created` as a request
    /// id; a moving timestamp breaks that contract).
    pub fn new_delta_at(
        id: &str,
        model: &str,
        created: i64,
        index: usize,
        delta: ChunkDelta,
        finish_reason: Option<String>,
    ) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: vec![ChunkChoice {
                index,
                delta,
                finish_reason,
                logprobs: None,
            }],
            usage: None,
            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
            service_tier: "default".to_string(),
        }
    }

    /// Create the final chunk with usage statistics. Same `created`
    /// rule as `new_delta_at`.
    pub fn new_final_at(id: &str, model: &str, created: i64, usage: Usage) -> Self {
        Self {
            id: id.to_string(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.to_string(),
            choices: vec![],
            usage: Some(usage),
            system_fingerprint: SYSTEM_FINGERPRINT.to_string(),
            service_tier: "default".to_string(),
        }
    }
}

/// A choice within a streaming chunk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: usize,
    pub delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// OpenAI emits `"logprobs": null` on every streaming chunk
    /// choice when logprobs aren't requested. Same parity rationale
    /// as Choice.logprobs.
    #[serde(default)]
    pub logprobs: Option<serde_json::Value>,
}

/// Delta content within a streaming chunk choice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkDelta {
    /// Present in the first chunk to indicate the role
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Token content (may be empty string, absent only if role-only chunk)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// The model's reasoning, streamed apart from the answer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Streaming tool-call deltas. We buffer each tool region and emit one
    /// complete delta per call (id + name + full arguments), which is a
    /// valid degenerate case of OpenAI's incremental tool-call streaming  -
    /// clients accumulate deltas keyed by `index`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A tool-call entry inside a streaming `delta`. Mirrors OpenAI's chunked
/// tool-call shape; `index` keys the call across chunks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallDelta {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

/// Function name/arguments fragment inside a streaming tool-call delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

impl ChunkDelta {
    /// Role-only opening delta (`{"role":"assistant"}`).
    pub fn role(role: &str) -> Self {
        Self {
            role: Some(role.to_string()),
            content: None,
            tool_calls: None,
            reasoning_content: None,
        }
    }
    /// Content-only delta.
    pub fn content(text: String) -> Self {
        Self {
            role: None,
            content: Some(text),
            tool_calls: None,
            reasoning_content: None,
        }
    }
    /// Empty delta (used by finish-reason-only chunks).
    pub fn reasoning(text: String) -> Self {
        Self {
            role: None,
            content: None,
            tool_calls: None,
            reasoning_content: Some(text),
        }
    }
    pub fn empty() -> Self {
        Self {
            role: None,
            content: None,
            tool_calls: None,
            reasoning_content: None,
        }
    }
    /// Tool-call-only delta.
    pub fn tool_calls(calls: Vec<ToolCallDelta>) -> Self {
        Self {
            role: None,
            content: None,
            tool_calls: Some(calls),
            reasoning_content: None,
        }
    }
}

impl ToolCallDelta {
    /// Build a complete single-shot delta from a finished `ToolCall`.
    pub fn from_tool_call(index: usize, call: &ToolCall) -> Self {
        Self {
            index,
            id: Some(call.id.clone()),
            r#type: Some(call.r#type.clone()),
            function: call.function.as_ref().map(|f| ToolCallFunctionDelta {
                name: Some(f.name.clone()),
                arguments: Some(f.arguments.clone().unwrap_or_else(|| "{}".to_string())),
            }),
        }
    }
}

impl Choice {
    pub fn new(index: usize, message: Message, finish_reason: String) -> Self {
        Self {
            index,
            message,
            finish_reason,
            logprobs: None,
        }
    }
}

/// Token usage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    /// OpenAI's nested breakdowns (gpt-4o / o1 / etc.). We don't run
    /// reasoning models or prompt caching natively, so the inner
    /// fields all default to zero. Emitting the structures keeps
    /// SDK shape tests + UI count widgets happy.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: i32,
    #[serde(default)]
    pub audio_tokens: i32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    #[serde(default)]
    pub reasoning_tokens: i32,
    #[serde(default)]
    pub audio_tokens: i32,
    #[serde(default)]
    pub accepted_prediction_tokens: i32,
    #[serde(default)]
    pub rejected_prediction_tokens: i32,
}

impl Usage {
    pub fn new(prompt_tokens: i32, completion_tokens: i32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: Some(PromptTokensDetails::default()),
            completion_tokens_details: Some(CompletionTokensDetails::default()),
        }
    }
}

/// Model information (unified format)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub size: String,
    pub size_bytes: u64,
    pub modified_at: String,
    /// Model source
    #[serde(default = "default_model_source_str")]
    pub source: String,
}

impl ModelInfo {
    pub fn new(name: String, size: String, size_bytes: u64, modified_at: String) -> Self {
        Self {
            name,
            size,
            size_bytes,
            modified_at,
            source: "ollama".to_string(),
        }
    }

    pub fn with_source(mut self, source: ModelSource) -> Self {
        self.source = source.to_string();
        self
    }

    /// Create from Ollama model
    pub fn from_ollama(model: OllamaModel) -> Self {
        Self {
            name: model.name,
            size: format!("{} MB", model.size / (1024 * 1024)),
            size_bytes: model.size,
            modified_at: model.modified_at,
            source: model.source,
        }
    }
}

/// List models response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListModelsResponse {
    pub models: Vec<ModelInfo>,
}

impl ListModelsResponse {
    pub fn new(models: Vec<ModelInfo>) -> Self {
        Self { models }
    }

    /// Create from Ollama response
    pub fn from_ollama(resp: OllamaListModelsResponse) -> Self {
        Self {
            models: resp
                .models
                .into_iter()
                .map(ModelInfo::from_ollama)
                .collect(),
        }
    }
}

/// Get model request
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct GetModelRequest {
    #[validate(length(min = 1))]
    pub name: String,
}

impl GetModelRequest {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

/// Get model response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetModelResponse {
    pub message: String,
}

impl GetModelResponse {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

/// Delete model request
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DeleteModelRequest {
    #[validate(length(min = 1))]
    pub name: String,
}

impl DeleteModelRequest {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

/// Delete model response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteModelResponse {
    pub message: String,
}

impl DeleteModelResponse {
    pub fn new(message: String) -> Self {
        Self { message }
    }
}

/// Load model response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelResponse {
    pub model: String,
    pub status: String,
    pub message: String,
}

/// Response for listing loaded models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListLoadedModelsResponse {
    pub models: Vec<LoadedModelInfo>,
}

/// Layer distribution across devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDistribution {
    /// Device type (e.g., "CUDA", "CPU")
    pub device_type: String,
    /// Device index
    pub device_id: usize,
    /// First layer index
    pub layer_start: u32,
    /// Last layer index (inclusive)
    pub layer_end: u32,
    /// Memory used in bytes
    pub memory_bytes: u64,
}

impl LayerDistribution {
    pub fn new(
        device_type: String,
        device_id: usize,
        layer_start: u32,
        layer_end: u32,
        memory_bytes: u64,
    ) -> Self {
        Self {
            device_type,
            device_id,
            layer_start,
            layer_end,
            memory_bytes,
        }
    }

    /// Number of layers in this distribution
    pub fn layer_count(&self) -> u32 {
        self.layer_end.saturating_sub(self.layer_start) + 1
    }
}

/// Information about a loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModelInfo {
    pub model: String,
    pub status: String,
    /// Device the model is loaded on (e.g., "CUDA", "CPU") - primary device
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// Model size in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    /// Total number of layers in the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_layers: Option<u32>,
    /// Layer distribution across devices (for multi-device models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_distribution: Option<Vec<LayerDistribution>>,
}

impl ListLoadedModelsResponse {
    pub fn new(models: Vec<LoadedModelInfo>) -> Self {
        Self { models }
    }
}

/// Generate request (simple API)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub temperature: f64,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

fn default_max_tokens() -> usize {
    128
}

/// Generate response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub model: String,
    pub text: String,
    pub tokens_used: usize,
}

/// Model swap request (POST /api/swap)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapModelRequest {
    /// Current model ID to replace
    pub current_model: String,
    /// New model ID to load
    pub new_model: String,
    /// Optional keep_alive for the new model
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_keep_alive"
    )]
    pub keep_alive: Option<String>,
}

/// Model swap response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapModelResponse {
    /// Status of the swap operation
    pub status: String,
    /// Message describing the result
    pub message: String,
    /// Previous model that was replaced
    pub previous_model: String,
    /// New model that is now loaded
    pub current_model: String,
}

/// Layer swap request (POST /api/layers/swap)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapLayerRequest {
    /// Model to change. Must already be loaded: this replaces weights in place rather than
    /// arranging for a load.
    pub model: String,
    /// The adapter set the model should carry after the call. An empty list detaches
    /// everything and returns every projection to the checkpoint.
    ///
    /// It is a set, not a delta: the same request always leaves the same model, whatever was
    /// attached before, so a client does not have to track what it asked for last time.
    #[serde(default)]
    pub adapters: Vec<SwapAdapter>,
}

/// One adapter and how strongly to apply it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SwapAdapter {
    /// Name resolved inside the configured adapter directory.
    pub name: String,
    /// Multiplies the adapter's own alpha/rank scale. 1.0 applies it as trained; 0 is a
    /// request to disable it, and is refused rather than attached as a block of zeros.
    #[serde(default = "one")]
    pub strength: f32,
}

fn one() -> f32 {
    1.0
}

// ============================================================================
// Ollama API Compatibility Types (additional endpoints)
// ============================================================================

/// Ollama copy model request (POST /api/copy)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaCopyRequest {
    pub source: String,
    pub destination: String,
}

/// Ollama create model request (POST /api/create).
///
/// String fields carry length caps that match what the handler enforces
/// imperatively (b9d05fa, 0ce73c8). Documenting them on the type makes
/// the caps discoverable and lets integration tests pin them via the
/// validator crate without spinning up an axum router.
#[derive(Debug, Clone, Serialize, Deserialize, validator::Validate)]
pub struct OllamaCreateRequest {
    /// `model` is what current clients send; `name` what earlier ones did.
    #[serde(alias = "model")]
    pub name: String,
    /// Blobs uploaded through `/api/blobs`, by file name and digest: the model's
    /// weights and, optionally, a projector.
    #[serde(default)]
    pub files: Option<std::collections::HashMap<String, String>>,
    /// Adapter blobs, by file name and digest.
    #[serde(default)]
    pub adapters: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    /// Conversation turns to build in; accepted and stored with the model's config.
    #[serde(default)]
    pub messages: Option<Vec<Message>>,
    /// Base model to derive from
    #[serde(default)]
    pub from: Option<String>,
    /// System prompt. Gets persisted into loken_config.json on disk
    /// when a create call sets it; cap matches handler-side guard at
    /// 32 KiB - 4x the prompt-context most chat models can attend to.
    #[validate(length(max = 32768))]
    #[serde(default)]
    pub system: Option<String>,
    /// Model parameters (temperature, top_p, etc.). The handler caps the
    /// serialized JSON form at 64 KiB; validator macros can't measure
    /// JSON-serialized size so that gate lives only in the handler.
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
    /// Modelfile content (alternative to structured fields). 64 KiB cap
    /// matches handler-side guard - modelfiles are short DSL configs, not
    /// payloads.
    #[validate(length(max = 65536))]
    #[serde(default)]
    pub modelfile: Option<String>,
    #[serde(default)]
    pub stream: bool,
    /// Quantization level
    #[serde(default)]
    pub quantize: Option<String>,
}

/// Ollama push model request (POST /api/push)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPushRequest {
    pub name: String,
    #[serde(default)]
    pub insecure: bool,
    #[serde(default)]
    pub stream: bool,
}

/// Ollama embed request (POST /api/embed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbedRequest {
    pub model: String,
    pub input: serde_json::Value, // Can be string or array of strings
    /// Cut each input to the model's context (the default) or refuse one that exceeds it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate: Option<bool>,
    /// Keep only the leading dimensions of each vector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<serde_json::Value>,
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_keep_alive"
    )]
    pub keep_alive: Option<String>,
}

/// Ollama embed response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaEmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
    // Same null-vs-omit cleanup as the chat/generate response types
    // in ebf6bef. Real Ollama omits these when missing rather than
    // emitting explicit null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub load_duration: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_eval_count: Option<u64>,
}

/// Ollama running model info (GET /api/ps)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPsModel {
    pub name: String,
    pub model: String,
    pub size: u64,
    pub digest: String,
    pub details: OllamaModelDetails,
    pub expires_at: String,
    pub size_vram: u64,
    /// Context length from config (not from GGUF metadata)
    pub context_length: i32,
}

/// Ollama ps response (GET /api/ps)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaPsResponse {
    pub models: Vec<OllamaPsModel>,
}

#[cfg(test)]
mod deserialize_helper_tests;

#[cfg(test)]
mod thinking_field_tests {
    use super::*;

    /// A client may send `think`, `thinking`, or both. Both was a duplicate-field
    /// rejection when the two names shared one field through an alias - every request
    /// from a caller that hedges was refused outright.
    #[test]
    fn either_field_is_accepted_and_both_together_still_parse() {
        for body in [
            r#"{"model":"m","prompt":"p","think":false}"#,
            r#"{"model":"m","prompt":"p","thinking":false}"#,
            r#"{"model":"m","prompt":"p","think":false,"thinking":false}"#,
        ] {
            let r: OllamaGenerateRequest = serde_json::from_str(body).expect(body);
            assert_eq!(r.thinking_preference(), Some("disabled"), "{body}");
        }
        // The Ollama name decides when the two disagree.
        let r: OllamaGenerateRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"p","think":true,"thinking":false}"#)
                .unwrap();
        assert_eq!(r.thinking_preference(), Some("enabled"));
        let r: OllamaGenerateRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"p"}"#).unwrap();
        assert_eq!(r.thinking_preference(), None);
    }
}
