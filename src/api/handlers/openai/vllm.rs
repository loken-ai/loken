//! vLLM's extensions to the OpenAI request bodies, and the routes of its server that a vLLM
//! client calls besides the OpenAI ones.
//!
//! The fields are read once into [`VllmFields`], flattened into the chat request and read from
//! the raw body of a completion, then bounded and turned into the engine's
//! [`RequestRules`] and grammar: one reading for both routes.

use super::*;
use crate::api::types::vllm::{StructuredOutputs, VllmFields};
use crate::inference::engine::llm_engine::{
    FinishReason, PromptTruncation, RequestRules, TokenLogprob, TruncationSide,
};
use axum::response::IntoResponse;

/// Banned words one request may carry: each is tokenized once and checked at every step.
const BAD_WORDS_MAX: usize = 256;
/// Characters of a `cache_salt`, as vLLM bounds it.
const CACHE_SALT_MAX_CHARS: usize = 1024;
/// Upper bound of `repetition_penalty`, the same as the Ollama `repeat_penalty` it becomes.
const REPETITION_PENALTY_MAX: f64 = 4.0;
/// Token ids one list (`allowed_token_ids`, `stop_token_ids`) may name: past any vocabulary a
/// served model has.
const TOKEN_LIST_MAX: usize = 1_048_576;
/// The `truncate_prompt_tokens` value that asks for the model's whole window.
const TRUNCATE_TO_WINDOW: i64 = -1;

fn invalid(msg: impl Into<String>) -> ApiError {
    ApiError::Validation(msg.into())
}

impl VllmFields {
    /// The fields of a raw request body; a field of the wrong type is refused.
    pub(crate) fn of_body(body: &serde_json::Value) -> Result<Self, ApiError> {
        serde_json::from_value(body.clone()).map_err(|e| invalid(e.to_string()))
    }

    /// The engine's rules, each field bounded, with `max_tokens` the request's completion cap.
    pub(crate) fn rules(&self, max_tokens: Option<i64>) -> Result<RequestRules, ApiError> {
        if self.use_beam_search == Some(true) {
            return Err(invalid("`use_beam_search` is not served"));
        }
        if self.prompt_logprobs.is_some_and(|n| n > 0) {
            return Err(invalid("`prompt_logprobs` is not served"));
        }
        if let Some(p) = self.min_p {
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(invalid(format!("`min_p` must be in [0, 1]; got {p}")));
            }
        }
        let min_tokens = self.min_tokens.unwrap_or(0);
        if let Some(cap) = max_tokens.filter(|&m| m >= 0 && min_tokens > m as u64) {
            return Err(invalid(format!(
                "`min_tokens` = {min_tokens} must not exceed `max_tokens` = {cap}"
            )));
        }
        if min_tokens > super::super::sampling::MAX_TOKENS_CAP as u64 {
            return Err(invalid(format!(
                "`min_tokens` = {min_tokens}; the cap is {}",
                super::super::sampling::MAX_TOKENS_CAP
            )));
        }
        let bad_words = self.bad_words.clone().unwrap_or_default();
        if bad_words.len() > BAD_WORDS_MAX {
            return Err(invalid(format!(
                "`bad_words` has {} entries; cap at {BAD_WORDS_MAX}",
                bad_words.len()
            )));
        }
        let stop_max = super::super::sampling::STOP_ENTRY_MAX_CHARS;
        if let Some(i) = bad_words.iter().position(|w| w.chars().count() > stop_max) {
            return Err(invalid(format!(
                "`bad_words[{i}]` is longer than {stop_max} chars"
            )));
        }
        if let Some(ids) = &self.allowed_token_ids {
            if ids.is_empty() {
                return Err(invalid("`allowed_token_ids` must not be empty"));
            }
        }
        for (name, list) in [
            ("allowed_token_ids", &self.allowed_token_ids),
            ("stop_token_ids", &self.stop_token_ids),
        ] {
            if list.as_ref().is_some_and(|l| l.len() > TOKEN_LIST_MAX) {
                return Err(invalid(format!(
                    "`{name}` names more than {TOKEN_LIST_MAX} tokens"
                )));
            }
        }
        let side = match self.truncation_side.as_deref() {
            None | Some("left") => TruncationSide::Left,
            Some("right") => TruncationSide::Right,
            Some(other) => {
                return Err(invalid(format!(
                    "`truncation_side` '{other}' not supported; use 'left' or 'right'"
                )))
            }
        };
        let truncate_prompt = match self.truncate_prompt_tokens {
            None | Some(TRUNCATE_TO_WINDOW) => None,
            Some(n) if n >= 1 => Some(PromptTruncation {
                keep: n as usize,
                side,
            }),
            Some(n) => {
                return Err(invalid(format!(
                    "`truncate_prompt_tokens` must be -1 or at least 1; got {n}"
                )))
            }
        };
        if let Some(salt) = &self.cache_salt {
            let n = salt.chars().count();
            if n == 0 || n > CACHE_SALT_MAX_CHARS {
                return Err(invalid(format!(
                    "`cache_salt` must hold 1 to {CACHE_SALT_MAX_CHARS} chars"
                )));
            }
        }
        Ok(RequestRules {
            min_p: self.min_p.map(|p| p as f32),
            allowed_token_ids: self.allowed_token_ids.clone(),
            bad_words,
            min_tokens: min_tokens as usize,
            ignore_eos: self.ignore_eos.unwrap_or(false),
            stop_token_ids: self.stop_token_ids.clone().unwrap_or_default(),
            keep_special_tokens: self.skip_special_tokens == Some(false),
            truncate_prompt,
            cache_salt: self.cache_salt.clone(),
        })
    }

    /// `repetition_penalty`, bounded: above zero, 1 for none.
    pub(crate) fn repetition_penalty(&self) -> Result<Option<f64>, ApiError> {
        match self.repetition_penalty {
            Some(p) if !p.is_finite() || p <= 0.0 || p > REPETITION_PENALTY_MAX => Err(invalid(
                format!("`repetition_penalty` must be in (0, {REPETITION_PENALTY_MAX}]; got {p}"),
            )),
            p => Ok(p),
        }
    }

    /// The grammar spec of the request's structured output, `None` when it asks for none.
    /// `structured_outputs` wins over the older `guided_*` fields.
    pub(crate) fn grammar(&self) -> Result<Option<String>, ApiError> {
        let legacy = StructuredOutputs {
            json: self.guided_json.clone(),
            regex: self.guided_regex.clone(),
            choice: self.guided_choice.clone(),
            grammar: self.guided_grammar.clone(),
            whitespace_pattern: self.guided_whitespace_pattern.clone(),
            ..StructuredOutputs::default()
        };
        match &self.structured_outputs {
            Some(so) => so.grammar().map(Some),
            None if legacy.constraints() > 0 => legacy.grammar().map(Some),
            None => Ok(None),
        }
    }

    /// Whether the text keeps the stop sequence it ended on.
    pub(crate) fn keeps_stop(&self) -> bool {
        self.include_stop_str_in_output == Some(true)
    }

    /// The chat template's options and variables, refused as vLLM refuses them.
    pub(crate) fn template_options(
        &self,
    ) -> Result<super::super::chat_prompt::TemplateOptions, ApiError> {
        let kwargs_template = self
            .chat_template_kwargs
            .as_ref()
            .is_some_and(|k| k.contains_key("chat_template"));
        if self.chat_template.is_some() || kwargs_template {
            return Err(invalid(
                "a chat template sent with the request is refused; the model's own is used",
            ));
        }
        let continue_final = self.continue_final_message == Some(true);
        if continue_final && self.add_generation_prompt == Some(true) {
            return Err(invalid(
                "`continue_final_message` and `add_generation_prompt` cannot both be true",
            ));
        }
        let mut variables = self.chat_template_kwargs.clone().unwrap_or_default();
        if let Some(docs) = &self.documents {
            variables.insert("documents".into(), serde_json::json!(docs));
        }
        Ok(super::super::chat_prompt::TemplateOptions {
            add_generation_prompt: !continue_final && self.add_generation_prompt != Some(false),
            continue_final_message: continue_final,
            variables,
        })
    }
}

impl StructuredOutputs {
    fn constraints(&self) -> usize {
        [
            self.json.is_some(),
            self.regex.is_some(),
            self.choice.is_some(),
            self.grammar.is_some(),
            self.json_object.is_some(),
            self.structural_tag.is_some(),
        ]
        .into_iter()
        .filter(|&b| b)
        .count()
    }

    /// The grammar spec of the one constraint.
    fn grammar(&self) -> Result<String, ApiError> {
        if self.constraints() != 1 {
            return Err(invalid(
                "`structured_outputs` must name exactly one of json, regex, choice, grammar, \
                 json_object, structural_tag",
            ));
        }
        if self.structural_tag.is_some() {
            return Err(invalid("`structured_outputs.structural_tag` is not served"));
        }
        if let Some(rx) = &self.regex {
            return Ok(format!("regex:{rx}"));
        }
        if let Some(choices) = &self.choice {
            if choices.is_empty() {
                return Err(invalid("`structured_outputs.choice` must not be empty"));
            }
            let alts: Vec<String> = choices
                .iter()
                .map(|c| serde_json::to_string(c).unwrap_or_default())
                .collect();
            return Ok(format!("lark:start: {}", alts.join(" | ")));
        }
        if let Some(g) = &self.grammar {
            return if g.contains("::=") {
                gbnf_to_lark(g).map(|l| format!("lark:{l}"))
            } else {
                Ok(format!("lark:{g}"))
            };
        }
        let mut schema = match &self.json {
            Some(serde_json::Value::String(s)) => serde_json::from_str(s)
                .map_err(|e| invalid(format!("`structured_outputs.json` is not JSON: {e}")))?,
            Some(v) => v.clone(),
            None => serde_json::json!({"type": "object"}),
        };
        if !schema.is_object() {
            return Err(invalid(
                "`structured_outputs.json` must be a JSON schema object",
            ));
        }
        if self.disable_additional_properties == Some(true) {
            close_objects(&mut schema);
        }
        let mut options = serde_json::Map::new();
        if self.disable_any_whitespace == Some(true) {
            options.insert("whitespace_flexible".into(), serde_json::json!(false));
        }
        if let Some(p) = &self.whitespace_pattern {
            options.insert("whitespace_pattern".into(), serde_json::json!(p));
        }
        if !options.is_empty() {
            schema["x-guidance"] = serde_json::Value::Object(options);
        }
        json_schema_grammar_str(&schema)
            .ok_or_else(|| invalid("the JSON schema is larger than the grammar engine takes"))
    }
}

/// Every object of a schema that names its properties and says nothing of others is closed
/// to others.
fn close_objects(schema: &mut serde_json::Value) {
    match schema {
        serde_json::Value::Object(o) => {
            if o.contains_key("properties") && !o.contains_key("additionalProperties") {
                o.insert("additionalProperties".into(), serde_json::json!(false));
            }
            for v in o.values_mut() {
                close_objects(v);
            }
        }
        serde_json::Value::Array(a) => a.iter_mut().for_each(close_objects),
        _ => {}
    }
}

/// One piece of a GBNF grammar.
#[derive(Debug, PartialEq)]
enum GbnfToken {
    Name(String),
    Defines,
    Literal(String),
    Class(String),
    Any,
    Other(char),
    /// A `{m,n}` repetition, kept as written.
    Repeat(String),
}

/// The text of a GBNF string literal's escape, as a JSON string holds it.
fn gbnf_escape(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Result<char, ApiError> {
    let hex = |chars: &mut std::iter::Peekable<std::str::Chars<'_>>, n: usize| {
        let digits: String = chars.by_ref().take(n).collect();
        u32::from_str_radix(&digits, 16)
            .ok()
            .and_then(char::from_u32)
            .ok_or_else(|| invalid(format!("bad escape `{digits}` in the grammar")))
    };
    match chars.next() {
        Some('n') => Ok('\n'),
        Some('r') => Ok('\r'),
        Some('t') => Ok('\t'),
        Some('x') => hex(chars, 2),
        Some('u') => hex(chars, 4),
        Some('U') => hex(chars, 8),
        Some(c) => Ok(c),
        None => Err(invalid("the grammar ends inside an escape")),
    }
}

fn gbnf_tokens(src: &str) -> Result<Vec<GbnfToken>, ApiError> {
    let mut out = Vec::new();
    let mut chars = src.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {}
            '#' => {
                for c in chars.by_ref() {
                    if c == '\n' {
                        break;
                    }
                }
            }
            '"' => {
                let mut s = String::new();
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => s.push(gbnf_escape(&mut chars)?),
                        Some(c) => s.push(c),
                        None => return Err(invalid("the grammar ends inside a string")),
                    }
                }
                out.push(GbnfToken::Literal(s));
            }
            '[' => {
                let mut s = String::from("[");
                loop {
                    match chars.next() {
                        Some(']') => break,
                        Some('\\') => {
                            s.push('\\');
                            s.extend(chars.next());
                        }
                        Some('/') => s.push_str("\\/"),
                        Some(c) => s.push(c),
                        None => return Err(invalid("the grammar ends inside a character class")),
                    }
                }
                s.push(']');
                out.push(GbnfToken::Class(s));
            }
            '{' => {
                let mut s = String::from("{");
                for c in chars.by_ref() {
                    s.push(c);
                    if c == '}' {
                        break;
                    }
                }
                out.push(GbnfToken::Repeat(s));
            }
            ':' if chars.peek() == Some(&':') => {
                chars.next();
                if chars.next() != Some('=') {
                    return Err(invalid("`::` must be followed by `=` in the grammar"));
                }
                out.push(GbnfToken::Defines);
            }
            '.' => out.push(GbnfToken::Any),
            c if c.is_ascii_alphanumeric() || c == '_' || c == '-' => {
                let mut s = String::from(c);
                while let Some(&n) = chars.peek() {
                    if n.is_ascii_alphanumeric() || n == '_' || n == '-' {
                        s.push(n);
                        chars.next();
                    } else {
                        break;
                    }
                }
                out.push(GbnfToken::Name(s));
            }
            c => out.push(GbnfToken::Other(c)),
        }
    }
    Ok(out)
}

/// A GBNF rule name as a Lark rule name: `root` is the start, the others in lower case with
/// underscores.
fn lark_rule_name(name: &str) -> String {
    match name {
        "root" => "start".into(),
        _ => {
            let n = name.to_ascii_lowercase().replace('-', "_");
            if n == "start" {
                "start_rule".into()
            } else {
                n
            }
        }
    }
}

/// A grammar in llama.cpp's GBNF, the form vLLM's default backend reads, as the Lark the
/// grammar engine reads: one rule per line, strings as JSON strings, character classes as
/// regular expressions.
pub(crate) fn gbnf_to_lark(src: &str) -> Result<String, ApiError> {
    let tokens = gbnf_tokens(src)?;
    let mut rules: Vec<(String, Vec<String>)> = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        if let (GbnfToken::Name(n), Some(GbnfToken::Defines)) = (&tokens[i], tokens.get(i + 1)) {
            rules.push((lark_rule_name(n), Vec::new()));
            i += 2;
            continue;
        }
        let Some((_, body)) = rules.last_mut() else {
            return Err(invalid("the grammar must open with a rule `name ::= ...`"));
        };
        body.push(match &tokens[i] {
            GbnfToken::Name(n) => lark_rule_name(n),
            GbnfToken::Literal(s) => serde_json::to_string(s).unwrap_or_default(),
            GbnfToken::Class(c) => format!("/{c}/"),
            GbnfToken::Any => "/(?s:.)/".into(),
            GbnfToken::Repeat(r) => r.clone(),
            GbnfToken::Other(c) => c.to_string(),
            GbnfToken::Defines => unreachable!("a definition opens a rule"),
        });
        i += 1;
    }
    if !rules.iter().any(|(n, _)| n == "start") {
        return Err(invalid("the grammar has no `root` rule"));
    }
    Ok(rules
        .iter()
        .map(|(n, body)| format!("{n}: {}", body.join(" ")))
        .collect::<Vec<_>>()
        .join("\n"))
}

/// The text with the stop sequence it ended on taken off its end, unless the request keeps it.
pub(crate) fn strip_stop(mut text: String, finish: &FinishReason, keep: bool) -> String {
    if let (FinishReason::StopSequence(s), false) = (finish, keep) {
        if !s.is_empty() && text.ends_with(s.as_str()) {
            text.truncate(text.len() - s.len());
        }
    }
    text
}

/// Holds back the tail of a streamed text that a stop sequence could still end, so the stop
/// the generation ends on is never sent unless the request keeps it.
pub(crate) struct StopHoldback {
    /// Bytes held: the longest stop sequence, 0 when nothing is held.
    hold: usize,
    pending: String,
}

impl StopHoldback {
    pub(crate) fn new(stops: &[String], keep: bool) -> Self {
        let hold = if keep {
            0
        } else {
            stops.iter().map(String::len).max().unwrap_or(0)
        };
        Self {
            hold,
            pending: String::new(),
        }
    }

    /// What of the text so far may be sent now.
    pub(crate) fn push(&mut self, text: &str) -> String {
        self.pending.push_str(text);
        if self.pending.len() <= self.hold {
            return String::new();
        }
        let mut cut = self.pending.len() - self.hold;
        while !self.pending.is_char_boundary(cut) {
            cut -= 1;
        }
        let rest = self.pending.split_off(cut);
        std::mem::replace(&mut self.pending, rest)
    }

    /// What is left to send once the generation ended for `finish`.
    pub(crate) fn finish(&mut self, finish: Option<&FinishReason>) -> String {
        let pending = std::mem::take(&mut self.pending);
        match finish {
            Some(f) => strip_stop(pending, f, self.hold == 0),
            None => pending,
        }
    }
}

/// The log-probabilities with every token written `token_id:<id>`, as
/// `return_tokens_as_token_ids` asks.
pub(crate) fn tokens_as_ids(lps: &mut [TokenLogprob]) {
    for lp in lps {
        lp.text = format!("token_id:{}", lp.token);
        for (id, text, _) in &mut lp.top {
            *text = format!("token_id:{id}");
        }
    }
}

/// The loaded engine of `model`, loaded on first use as the generation routes do.
async fn engine_for(state: &APIServer, model: &str) -> Result<Arc<LlmEngine>, ApiError> {
    validate_model_id(model)?;
    let name = normalize_model_id(model);
    let load = super::super::admission::LoadRequest::new(state.get_effective_keep_alive(None));
    state.load_for_request(&name, load).await
}

/// vLLM's `/tokenize`: a `prompt`, or chat `messages` rendered by the model's template, as
/// `{count, max_model_len, tokens, token_strs}`.
pub(crate) async fn tokenize(
    State(state): State<APIServer>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let model = text_field(&body, "model");
    let engine = engine_for(&state, &model).await?;
    let flag = |key: &str| body.get(key).and_then(serde_json::Value::as_bool);
    let (text, special) = match (body.get("messages"), body.get("prompt")) {
        (Some(messages), _) => {
            let messages: Vec<Message> = serde_json::from_value(messages.clone())
                .map_err(|e| invalid(format!("`messages`: {e}")))?;
            let fields = VllmFields::of_body(&body)?;
            let tools: Option<Vec<Tool>> = match body.get("tools") {
                Some(t) => Some(
                    serde_json::from_value(t.clone())
                        .map_err(|e| invalid(format!("`tools`: {e}")))?,
                ),
                None => None,
            };
            let options = fields.template_options()?;
            let thinking = super::super::chat_prompt::thinking_from_variables(&options.variables);
            let prompt = state
                .prepare_prompt(
                    &normalize_model_id(&model),
                    Some(&engine),
                    &super::super::chat_prompt::PromptSpec {
                        messages: &messages,
                        tools: tools.as_ref(),
                        tool_choice: None,
                        thinking,
                        output_grammar: None,
                        template: options,
                    },
                )
                .await
                .prompt;
            // The template writes the special tokens a chat opens with.
            (prompt, flag("add_special_tokens").unwrap_or(false))
        }
        (None, Some(serde_json::Value::String(p))) => {
            (p.clone(), flag("add_special_tokens").unwrap_or(true))
        }
        _ => {
            return Err(invalid(
                "`/tokenize` takes a string `prompt` or a `messages` array",
            ))
        }
    };
    let (tokens, pieces) = engine
        .tokenize(&text, special)
        .await
        .ok_or_else(|| ApiError::Internal("the model is not loaded".into()))?
        .map_err(ApiError::Internal)?;
    let mut out = serde_json::json!({
        "count": tokens.len(),
        "max_model_len": engine.context_window().await.unwrap_or(0),
        "tokens": tokens,
        "token_strs": serde_json::Value::Null,
    });
    if flag("return_token_strs") == Some(true) {
        out["token_strs"] = serde_json::json!(pieces);
    }
    Ok(Json(out).into_response())
}

/// vLLM's `/detokenize`: `{model, tokens}` as `{prompt}`.
pub(crate) async fn detokenize(
    State(state): State<APIServer>,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let tokens: Vec<u32> = body
        .get("tokens")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| invalid(format!("`tokens`: {e}")))?
        .ok_or_else(|| invalid("`tokens` must be an array of token ids"))?;
    let engine = engine_for(&state, &text_field(&body, "model")).await?;
    let prompt = engine
        .detokenize(&tokens)
        .await
        .ok_or_else(|| ApiError::Internal("the model is not loaded".into()))?
        .map_err(ApiError::Internal)?;
    Ok(Json(serde_json::json!({ "prompt": prompt })).into_response())
}

/// vLLM's `/ping`: the health check SageMaker reads, an empty 200 while the server is up.
pub(crate) async fn ping() -> axum::http::StatusCode {
    axum::http::StatusCode::OK
}

/// vLLM's `/version`: this server's version.
pub(crate) async fn version() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "version": env!("CARGO_PKG_VERSION") }))
}

/// vLLM's `/load`: the requests running and waiting.
pub(crate) async fn load(State(state): State<APIServer>) -> Json<serde_json::Value> {
    let g = state.request_gate.snapshot().await;
    Json(serde_json::json!({ "server_load": g.in_flight + g.queue_depth }))
}

/// vLLM's `/reset_prefix_cache`: every loaded language model drops what it keeps of earlier
/// prompts, as `{success}`.
pub(crate) async fn reset_prefix_cache(State(state): State<APIServer>) -> Json<serde_json::Value> {
    let engines: Vec<Arc<LlmEngine>> = state
        .engines
        .read()
        .await
        .iter()
        .map(|e| e.engine.clone())
        .collect();
    for engine in engines {
        engine.reset_prefix_cache().await;
    }
    Json(serde_json::json!({ "success": true }))
}

/// vLLM's `/pooling`: the pooled output of an embedding model, `{data: [{index, object:
/// "pooling", data}]}`, from the embeddings route.
pub(crate) async fn pooling(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Response {
    let mut adapted = body.clone();
    adapted["encoding_format"] = serde_json::json!("float");
    let resp = openai_embeddings(State(state), headers, Json(adapted)).await;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap_or_default();
    let mut v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or_default();
    if !status.is_success() {
        return (status, Json(v)).into_response();
    }
    let data: Vec<serde_json::Value> = v["data"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|d| {
            serde_json::json!({
                "index": d["index"],
                "object": "pooling",
                "data": d["embedding"],
            })
        })
        .collect();
    v["data"] = serde_json::json!(data);
    v["id"] = serde_json::json!(format!("pool-{}", uuid::Uuid::new_v4()));
    v["created"] = serde_json::json!(chrono::Utc::now().timestamp());
    Json(v).into_response()
}

/// vLLM's `/v1/chat/completions/batch`: `messages` is a list of conversations, each answered
/// as its own chat completion, gathered as one response with a choice per conversation.
pub(crate) async fn chat_batch(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    let conversations = body
        .get("messages")
        .and_then(serde_json::Value::as_array)
        .filter(|c| !c.is_empty() && c.iter().all(serde_json::Value::is_array))
        .cloned()
        .ok_or_else(|| invalid("`messages` must be a non-empty list of conversations"))?;
    if body.get("stream").and_then(serde_json::Value::as_bool) == Some(true) {
        return Err(invalid(
            "`stream` is not supported on a batch of conversations",
        ));
    }
    if body.get("tools").is_some_and(|t| !t.is_null()) {
        return Err(invalid(
            "`tools` is not supported on a batch of conversations",
        ));
    }
    if body
        .get("n")
        .and_then(serde_json::Value::as_u64)
        .is_some_and(|n| n != 1)
    {
        return Err(invalid("`n` must be 1 on a batch of conversations"));
    }
    merge_runs(conversations.len(), true, None, |i| {
        let mut one = body.clone();
        one["messages"] = conversations[i].clone();
        let state = state.clone();
        let headers = headers.clone();
        Box::pin(async move {
            let request: ChatCompletionRequest =
                serde_json::from_value(one).map_err(|e| invalid(e.to_string()))?;
            chat_completion(State(state), headers, OpenAIJson(request)).await
        })
    })
    .await
}

/// SageMaker's `/invocations`, as vLLM serves it: the body goes to the route its shape names,
/// chat for `messages`, completions for `prompt`, embeddings for `input`.
pub(crate) async fn invocations(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, ApiError> {
    if body.get("messages").is_some() {
        let request: ChatCompletionRequest =
            serde_json::from_value(body).map_err(|e| invalid(e.to_string()))?;
        return chat_completion(State(state), headers, OpenAIJson(request)).await;
    }
    if body.get("prompt").is_some() {
        return text_completions(State(state), headers, Json(body)).await;
    }
    if body.get("input").is_some() {
        return Ok(openai_embeddings(State(state), headers, Json(body)).await);
    }
    Err(invalid(
        "`/invocations` takes a chat (`messages`), completion (`prompt`) or embedding (`input`) body",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fields(v: serde_json::Value) -> VllmFields {
        VllmFields::of_body(&v).unwrap()
    }

    fn refused(r: Result<impl std::fmt::Debug, ApiError>) -> String {
        match r {
            Err(ApiError::Validation(m)) => m,
            other => panic!("expected a validation error, got {other:?}"),
        }
    }

    /// Each field reaches its rule; an absent one is vLLM's default.
    #[test]
    fn the_fields_become_the_engines_rules() {
        let r = fields(serde_json::json!({
            "min_p": 0.1, "min_tokens": 3, "ignore_eos": true, "bad_words": ["x"],
            "allowed_token_ids": [1, 2], "stop_token_ids": [7], "skip_special_tokens": false,
            "truncate_prompt_tokens": 5, "truncation_side": "right", "cache_salt": "s",
        }))
        .rules(Some(10))
        .unwrap();
        assert_eq!(r.min_p, Some(0.1));
        assert_eq!(r.min_tokens, 3);
        assert!(r.ignore_eos && r.keep_special_tokens);
        assert_eq!(r.bad_words, vec!["x".to_string()]);
        assert_eq!(r.allowed_token_ids, Some(vec![1, 2]));
        assert_eq!(r.stop_token_ids, vec![7]);
        assert_eq!(
            r.truncate_prompt,
            Some(PromptTruncation {
                keep: 5,
                side: TruncationSide::Right
            })
        );
        assert_eq!(r.cache_salt.as_deref(), Some("s"));
        let r = fields(serde_json::json!({})).rules(None).unwrap();
        assert!(!r.decodes_sequentially() && !r.ignore_eos && r.truncate_prompt.is_none());
        let r = fields(serde_json::json!({"truncate_prompt_tokens": -1}))
            .rules(None)
            .unwrap();
        assert!(r.truncate_prompt.is_none());
    }

    /// Out-of-range values are refused naming the field.
    #[test]
    fn out_of_range_fields_are_refused() {
        let bad = |v: serde_json::Value| refused(fields(v).rules(Some(4)));
        assert!(bad(serde_json::json!({"min_p": 1.5})).contains("min_p"));
        assert!(bad(serde_json::json!({"min_tokens": 5})).contains("min_tokens"));
        assert!(bad(serde_json::json!({"allowed_token_ids": []})).contains("allowed_token_ids"));
        assert!(bad(serde_json::json!({"truncate_prompt_tokens": 0})).contains("truncate"));
        assert!(bad(serde_json::json!({"truncation_side": "middle"})).contains("truncation_side"));
        assert!(bad(serde_json::json!({"cache_salt": ""})).contains("cache_salt"));
        assert!(bad(serde_json::json!({"use_beam_search": true})).contains("beam"));
        assert!(bad(serde_json::json!({"prompt_logprobs": 1})).contains("prompt_logprobs"));
        let words = vec!["w"; BAD_WORDS_MAX + 1];
        assert!(bad(serde_json::json!({ "bad_words": words })).contains("bad_words"));
        let rp = |v| {
            refused(fields(serde_json::json!({ "repetition_penalty": v })).repetition_penalty())
        };
        assert!(rp(0.0).contains("repetition_penalty"));
        assert!(rp(REPETITION_PENALTY_MAX + 1.0).contains("repetition_penalty"));
        assert_eq!(
            fields(serde_json::json!({"repetition_penalty": 1.2}))
                .repetition_penalty()
                .unwrap(),
            Some(1.2)
        );
        assert!(VllmFields::of_body(&serde_json::json!({"min_tokens": "three"})).is_err());
    }

    /// Each structured output is the grammar spec the engine reads; the older `guided_*`
    /// fields are read the same way.
    #[test]
    fn structured_outputs_become_grammar_specs() {
        let g = |v: serde_json::Value| fields(v).grammar().unwrap().unwrap();
        assert_eq!(
            g(serde_json::json!({"structured_outputs": {"regex": "[0-9]+"}})),
            "regex:[0-9]+"
        );
        assert_eq!(
            g(serde_json::json!({"structured_outputs": {"choice": ["yes", "no \"x\""]}})),
            r#"lark:start: "yes" | "no \"x\"""#
        );
        assert_eq!(
            g(serde_json::json!({"structured_outputs": {"json_object": true}})),
            r#"json_schema:{"type":"object"}"#
        );
        let schema = g(serde_json::json!({"structured_outputs": {
            "json": {"type": "object", "properties": {"a": {"type": "integer"}}},
            "disable_any_whitespace": true, "disable_additional_properties": true,
        }}));
        let v: serde_json::Value =
            serde_json::from_str(schema.strip_prefix("json_schema:").unwrap()).unwrap();
        assert_eq!(v["additionalProperties"], false);
        assert_eq!(v["x-guidance"]["whitespace_flexible"], false);
        assert_eq!(
            g(serde_json::json!({"guided_json": "{\"type\":\"string\"}"})),
            r#"json_schema:{"type":"string"}"#
        );
        assert_eq!(g(serde_json::json!({"guided_regex": "a|b"})), "regex:a|b");
        assert_eq!(
            g(serde_json::json!({"structured_outputs": {"grammar": "start: \"a\""}})),
            "lark:start: \"a\""
        );
        assert!(fields(serde_json::json!({})).grammar().unwrap().is_none());
        let two = serde_json::json!({"structured_outputs": {"regex": "a", "choice": ["b"]}});
        assert!(refused(fields(two).grammar()).contains("exactly one"));
        let tag = serde_json::json!({"structured_outputs": {"structural_tag": {}}});
        assert!(refused(fields(tag).grammar()).contains("structural_tag"));
    }

    /// A GBNF grammar becomes Lark rule by rule: `root` the start, strings JSON-quoted,
    /// classes as regular expressions, names in Lark's spelling.
    #[test]
    fn gbnf_becomes_lark() {
        let lark = gbnf_to_lark(
            "root ::= greeting \" \" name # a comment\n\
             greeting ::= (\"hi\" | \"hello\")\n\
             name ::= [A-Z] [a-z/]{1,8} first-name?\n\
             first-name ::= \"\\x41\"",
        )
        .unwrap();
        assert_eq!(
            lark,
            "start: greeting \" \" name\n\
             greeting: ( \"hi\" | \"hello\" )\n\
             name: /[A-Z]/ /[a-z\\/]/ {1,8} first_name ?\n\
             first_name: \"A\""
        );
        assert!(parse_ok(&lark));
        assert!(refused(gbnf_to_lark("a ::= \"x\"")).contains("root"));
        assert!(refused(gbnf_to_lark("\"x\"")).contains("rule"));
    }

    fn parse_ok(lark: &str) -> bool {
        let g = llguidance::api::TopLevelGrammar::from_lark(lark.to_string());
        let tok = llguidance::toktrie::ApproximateTokEnv::single_byte_env();
        llguidance::ParserFactory::new_simple(&tok)
            .unwrap()
            .create_parser(g)
            .is_ok()
    }

    /// The stop sequence is taken off a finished text and held back in a stream, unless the
    /// request keeps it.
    #[test]
    fn the_stop_sequence_is_not_sent() {
        let stop = FinishReason::StopSequence("END".into());
        assert_eq!(strip_stop("abcEND".into(), &stop, false), "abc");
        assert_eq!(strip_stop("abcEND".into(), &stop, true), "abcEND");
        assert_eq!(strip_stop("abc".into(), &FinishReason::Eos, false), "abc");
        let mut h = StopHoldback::new(&["END".into()], false);
        let mut sent = String::new();
        for piece in ["ab", "c", "E", "ND"] {
            sent.push_str(&h.push(piece));
        }
        sent.push_str(&h.finish(Some(&stop)));
        assert_eq!(sent, "abc");
        let mut h = StopHoldback::new(&["END".into()], true);
        assert_eq!(h.push("abcEND"), "abcEND");
        // The cut falls on a character boundary, holding at least the stop's bytes.
        let mut h = StopHoldback::new(&["é".into()], false);
        assert_eq!(h.push("aéb"), "a");
        assert_eq!(h.finish(Some(&FinishReason::Eos)), "éb");
    }

    /// The template options follow vLLM: a request template is refused, a continued final
    /// message closes no turn and opens none.
    #[test]
    fn template_options_follow_vllm() {
        let o = fields(serde_json::json!({
            "continue_final_message": true,
            "chat_template_kwargs": {"enable_thinking": false},
            "documents": [{"title": "t", "text": "x"}],
        }))
        .template_options()
        .unwrap();
        assert!(o.continue_final_message && !o.add_generation_prompt);
        assert_eq!(o.variables["enable_thinking"], false);
        assert_eq!(o.variables["documents"][0]["text"], "x");
        assert!(
            refused(fields(serde_json::json!({"chat_template": "x"})).template_options())
                .contains("template")
        );
        let both =
            serde_json::json!({"continue_final_message": true, "add_generation_prompt": true});
        assert!(refused(fields(both).template_options()).contains("continue_final_message"));
        assert!(
            !fields(serde_json::json!({"add_generation_prompt": false}))
                .template_options()
                .unwrap()
                .add_generation_prompt
        );
    }

    /// `return_tokens_as_token_ids` writes every token by its id.
    #[test]
    fn tokens_are_written_by_id() {
        let mut lps = vec![TokenLogprob {
            token: 7,
            text: "a".into(),
            logprob: -0.1,
            top: vec![(9, "b".into(), -2.0)],
        }];
        tokens_as_ids(&mut lps);
        assert_eq!(lps[0].text, "token_id:7");
        assert_eq!(lps[0].top[0].1, "token_id:9");
    }
}
