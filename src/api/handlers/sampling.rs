//! The sampling knobs of a generation request, read from any dialect and bounded once.
//!
//! Each dialect names its fields its own way (`max_tokens`, `num_predict`,
//! `stop_sequences`), but the bounds a value must respect are the engine's, not the
//! dialect's: the same temperature, stop list or completion cap is accepted or refused
//! identically whichever route carried it.

use super::*;

/// Largest completion a request may ask for: the decode budget of the largest context the
/// served models open.
pub(crate) const MAX_TOKENS_CAP: i64 = 131_072;
/// Stop sequences a request may carry, and characters per sequence. The engine checks
/// every sequence against the generated tail at each step.
pub(crate) const STOP_MAX_ENTRIES: usize = 4;
pub(crate) const STOP_ENTRY_MAX_CHARS: usize = 256;
/// Alternatives a request may ask per generated token.
pub(crate) const TOP_LOGPROBS_MAX: u64 = 20;
/// Candidate pool bound for top-k: past any vocabulary a served model has.
const TOP_K_MAX: i64 = 1_048_576;
/// Largest context a request may ask the model to be loaded with.
const NUM_CTX_MAX: i64 = 1_048_576;
/// Rolling window scanned for the repetition penalty, bounded like the completion.
const REPEAT_LAST_N_MAX: i64 = MAX_TOKENS_CAP;

/// Completions one OpenAI request may ask for with `n`: each is a generation of its own,
/// so the bound is the one the image routes put on their `n`.
pub(crate) const N_MAX: u64 = 16;

/// The completions an OpenAI request asks for, 1 when `n` is absent.
pub(crate) fn completions_asked(n: Option<u64>) -> Result<usize, ApiError> {
    match n.unwrap_or(1) {
        0 => Err(ApiError::Validation("`n` must be >= 1".into())),
        n if n > N_MAX => Err(ApiError::Validation(format!(
            "`n` = {n}; the cap is {N_MAX}"
        ))),
        n => Ok(n as usize),
    }
}

/// How one dialect names the fields whose names differ, for the messages that cite them.
pub(crate) struct FieldNames {
    /// Put in front of every field name: `options.` for the Ollama bag.
    pub prefix: &'static str,
    pub max_tokens: &'static str,
    pub stop: &'static str,
    pub session_id: &'static str,
    /// The smallest completion cap the dialect accepts: Ollama's `-1` means "until EOS".
    pub max_tokens_floor: i64,
    /// Whether a request that names no penalty asks for none, as OpenAI's and Anthropic's
    /// APIs define it; otherwise the model's or the server's default applies, as Ollama's does.
    pub unnamed_penalty_is_none: bool,
}

pub(crate) const OPENAI_FIELDS: FieldNames = FieldNames {
    prefix: "",
    max_tokens: "max_tokens",
    stop: "stop",
    session_id: "user",
    max_tokens_floor: 1,
    unnamed_penalty_is_none: true,
};
pub(crate) const OLLAMA_FIELDS: FieldNames = FieldNames {
    prefix: "options.",
    max_tokens: "num_predict",
    stop: "stop",
    session_id: "session_id",
    max_tokens_floor: -1,
    unnamed_penalty_is_none: false,
};
pub(crate) const ANTHROPIC_FIELDS: FieldNames = FieldNames {
    prefix: "",
    max_tokens: "max_tokens",
    stop: "stop_sequences",
    session_id: "metadata.user_id",
    max_tokens_floor: 1,
    unnamed_penalty_is_none: true,
};

/// The knobs of one request, as the caller sent them.
pub(crate) struct SamplingRequest {
    pub fields: &'static FieldNames,
    pub max_tokens: Option<i64>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub repeat_last_n: Option<i64>,
    pub num_ctx: Option<i64>,
    pub session_id: Option<String>,
    /// `Some(n)` when per-token log-probabilities are asked, with `n` alternatives.
    pub top_logprobs: Option<u64>,
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    pub early_exit_threshold: Option<f64>,
    pub grammar: Option<String>,
}

fn f64_of(v: &serde_json::Value, key: &str) -> Option<f64> {
    v.get(key).and_then(serde_json::Value::as_f64)
}

fn i64_of(v: &serde_json::Value, key: &str) -> Option<i64> {
    v.get(key).and_then(serde_json::Value::as_i64)
}

/// A `stop` given as one string or a list; list entries that are not strings are dropped.
fn stop_of(v: Option<&serde_json::Value>) -> Vec<String> {
    match v {
        Some(serde_json::Value::String(one)) => vec![one.clone()],
        Some(serde_json::Value::Array(list)) => list
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

impl SamplingRequest {
    fn empty(fields: &'static FieldNames) -> Self {
        Self {
            fields,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            stop: Vec::new(),
            frequency_penalty: None,
            presence_penalty: None,
            repeat_penalty: None,
            repeat_last_n: None,
            num_ctx: None,
            session_id: None,
            top_logprobs: None,
            logit_bias: None,
            early_exit_threshold: None,
            grammar: None,
        }
    }

    /// `/v1/chat/completions`. `top_k` is read as an extension field.
    pub(crate) fn from_openai(req: &ChatCompletionRequest) -> Self {
        Self {
            max_tokens: req.completion_cap().map(|v| v as i64),
            temperature: req.temperature.map(f64::from),
            top_p: req.top_p.map(f64::from),
            top_k: req.top_k.map(i64::from),
            seed: req.seed,
            stop: req
                .stop
                .clone()
                .map(StopSequences::into_vec)
                .unwrap_or_default(),
            frequency_penalty: req.frequency_penalty.map(f64::from),
            presence_penalty: req.presence_penalty.map(f64::from),
            session_id: req.user.clone(),
            top_logprobs: (req.logprobs == Some(true))
                .then(|| u64::from(req.top_logprobs.unwrap_or(0))),
            logit_bias: req.logit_bias.as_ref().map(|m| {
                m.iter()
                    .filter_map(|(k, v)| k.parse::<u32>().ok().map(|id| (id, *v)))
                    .collect()
            }),
            ..Self::empty(&OPENAI_FIELDS)
        }
    }

    /// `/v1/completions`, whose body is read as raw JSON. `logprobs` is a count there.
    pub(crate) fn from_completions(body: &serde_json::Value) -> Self {
        Self {
            max_tokens: i64_of(body, "max_tokens"),
            temperature: f64_of(body, "temperature"),
            top_p: f64_of(body, "top_p"),
            top_k: i64_of(body, "top_k"),
            seed: body.get("seed").and_then(serde_json::Value::as_u64),
            stop: stop_of(body.get("stop")),
            frequency_penalty: f64_of(body, "frequency_penalty"),
            presence_penalty: f64_of(body, "presence_penalty"),
            session_id: body
                .get("user")
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            top_logprobs: match body.get("logprobs") {
                Some(serde_json::Value::Number(n)) => n.as_u64(),
                Some(serde_json::Value::Bool(true)) => Some(0),
                _ => None,
            },
            ..Self::empty(&OPENAI_FIELDS)
        }
    }

    /// The Ollama `options` bag, with the request's top-level `logprobs` fields. A value
    /// of the wrong type is read as absent, as Ollama does.
    pub(crate) fn from_ollama(
        options: Option<&serde_json::Value>,
        logprobs: Option<bool>,
        top_logprobs: Option<usize>,
    ) -> Self {
        let mut s = Self::empty(&OLLAMA_FIELDS);
        s.top_logprobs = (logprobs == Some(true)).then(|| top_logprobs.unwrap_or(0) as u64);
        let Some(o) = options else {
            return s;
        };
        s.max_tokens = i64_of(o, "num_predict");
        s.temperature = f64_of(o, "temperature");
        s.top_p = f64_of(o, "top_p");
        s.top_k = i64_of(o, "top_k");
        s.seed = o.get("seed").and_then(serde_json::Value::as_u64);
        s.stop = match o.get("stop") {
            Some(v @ serde_json::Value::Array(_)) => stop_of(Some(v)),
            _ => Vec::new(),
        };
        s.repeat_penalty = f64_of(o, "repeat_penalty");
        s.repeat_last_n = i64_of(o, "repeat_last_n");
        s.num_ctx = i64_of(o, "num_ctx");
        s.session_id = o
            .get("session_id")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        s.early_exit_threshold = f64_of(o, "early_exit_threshold");
        s.grammar = o
            .get("grammar")
            .and_then(serde_json::Value::as_str)
            .map(str::to_string);
        s
    }

    /// `/v1/messages`.
    pub(crate) fn from_anthropic(req: &crate::api::anthropic::AnthropicMessagesRequest) -> Self {
        Self {
            max_tokens: Some(req.effective_max_tokens() as i64),
            temperature: req.temperature.map(f64::from),
            top_p: req.top_p.map(f64::from),
            top_k: req.top_k.map(i64::from),
            seed: req.seed,
            stop: req.stop_sequences.clone().unwrap_or_default(),
            ..Self::empty(&ANTHROPIC_FIELDS)
        }
    }

    fn name(&self, field: &str) -> String {
        format!("{}{field}", self.fields.prefix)
    }

    fn in_range(&self, field: &str, v: Option<f64>, lo: f64, hi: f64) -> Result<(), ApiError> {
        match v {
            Some(x) if !x.is_finite() || !(lo..=hi).contains(&x) => Err(ApiError::Validation(
                format!("`{}` must be in [{lo}, {hi}]; got {x}", self.name(field)),
            )),
            _ => Ok(()),
        }
    }

    fn int_in_range(&self, field: &str, v: Option<i64>, lo: i64, hi: i64) -> Result<(), ApiError> {
        match v {
            Some(x) if !(lo..=hi).contains(&x) => Err(ApiError::Validation(format!(
                "`{}` must be in [{lo}, {hi}]; got {x}",
                self.name(field)
            ))),
            _ => Ok(()),
        }
    }

    /// The bounds every dialect shares.
    pub(crate) fn validate(&self) -> Result<(), ApiError> {
        self.in_range("temperature", self.temperature, 0.0, 2.0)?;
        self.in_range("top_p", self.top_p, 0.0, 1.0)?;
        self.in_range("frequency_penalty", self.frequency_penalty, -2.0, 2.0)?;
        self.in_range("presence_penalty", self.presence_penalty, -2.0, 2.0)?;
        self.in_range("repeat_penalty", self.repeat_penalty, 0.0, 4.0)?;
        self.int_in_range(
            self.fields.max_tokens,
            self.max_tokens,
            self.fields.max_tokens_floor,
            MAX_TOKENS_CAP,
        )?;
        self.int_in_range("top_k", self.top_k, 0, TOP_K_MAX)?;
        self.int_in_range("repeat_last_n", self.repeat_last_n, 0, REPEAT_LAST_N_MAX)?;
        self.int_in_range("num_ctx", self.num_ctx, 1, NUM_CTX_MAX)?;
        if self.stop.len() > STOP_MAX_ENTRIES {
            return Err(ApiError::Validation(format!(
                "`{}` has {} entries; cap at {STOP_MAX_ENTRIES}",
                self.name(self.fields.stop),
                self.stop.len()
            )));
        }
        for (i, s) in self.stop.iter().enumerate() {
            let n = s.chars().count();
            if n > STOP_ENTRY_MAX_CHARS {
                return Err(ApiError::Validation(format!(
                    "`{}[{i}]` is {n} chars; cap at {STOP_ENTRY_MAX_CHARS}",
                    self.name(self.fields.stop)
                )));
            }
        }
        if let Some(k) = self.top_logprobs.filter(|&k| k > TOP_LOGPROBS_MAX) {
            return Err(ApiError::Validation(format!(
                "`{}` = {k}; the cap is {TOP_LOGPROBS_MAX}",
                self.name("top_logprobs")
            )));
        }
        validate_session_key(
            &self.name(self.fields.session_id),
            self.session_id.as_deref(),
        )
    }

    /// The engine's parameters. An explicit `repeat_penalty` wins over the OpenAI
    /// penalties folded into one.
    pub(crate) fn into_params(self) -> GenerationParams {
        let named = self.frequency_penalty.is_some() || self.presence_penalty.is_some();
        let repeat_penalty = self.repeat_penalty.map(|v| v as f32).or_else(|| {
            (named || self.fields.unnamed_penalty_is_none).then(|| {
                crate::api::types::repeat_penalty_from_openai(
                    self.frequency_penalty.map(|v| v as f32),
                    self.presence_penalty.map(|v| v as f32),
                )
            })
        });
        GenerationParams {
            stable_prefix: None,
            prefix_tokens: None,
            // A negative cap is Ollama's "until EOS".
            max_tokens: self.max_tokens.and_then(|v| usize::try_from(v).ok()),
            temperature: self.temperature.map(|v| v as f32),
            top_p: self.top_p.map(|v| v as f32),
            top_k: self.top_k.and_then(|v| usize::try_from(v).ok()),
            seed: self.seed,
            stop_sequences: self.stop,
            early_exit_threshold: self.early_exit_threshold.map(|v| v as f32),
            repeat_penalty,
            repeat_last_n: self.repeat_last_n.and_then(|v| usize::try_from(v).ok()),
            context_length: self.num_ctx.and_then(|v| usize::try_from(v).ok()),
            session_id: self.session_id,
            grammar: self.grammar,
            logit_bias: self.logit_bias,
            top_logprobs: self.top_logprobs.map(|v| v as usize),
            cancel: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn err_of(s: SamplingRequest) -> String {
        match s.validate() {
            Err(ApiError::Validation(m)) => m,
            other => panic!("expected a validation error, got {other:?}"),
        }
    }

    /// One value, one verdict, whichever dialect carried it.
    #[test]
    fn every_dialect_applies_the_same_bounds() {
        let completions = |v: serde_json::Value| SamplingRequest::from_completions(&v);
        let ollama = |v: serde_json::Value| SamplingRequest::from_ollama(Some(&v), None, None);
        let anthropic = |v: serde_json::Value| {
            let mut body = serde_json::json!({"model": "m", "messages": []});
            body.as_object_mut()
                .unwrap()
                .extend(v.as_object().unwrap().clone());
            SamplingRequest::from_anthropic(&serde_json::from_value(body).unwrap())
        };
        let chat = |v: serde_json::Value| {
            let mut body = serde_json::json!({"model": "m", "messages": []});
            body.as_object_mut()
                .unwrap()
                .extend(v.as_object().unwrap().clone());
            SamplingRequest::from_openai(&serde_json::from_value(body).unwrap())
        };
        for build in [&completions as &dyn Fn(_) -> _, &ollama, &anthropic, &chat] {
            assert!(err_of(build(serde_json::json!({"temperature": 2.5}))).contains("temperature"));
            assert!(err_of(build(serde_json::json!({"top_p": 1.5}))).contains("top_p"));
            assert!(err_of(build(serde_json::json!({"top_k": TOP_K_MAX + 1}))).contains("top_k"));
            assert!(build(serde_json::json!({"temperature": 0.7, "top_k": 40}))
                .validate()
                .is_ok());
        }
        for build in [&completions as &dyn Fn(_) -> _, &chat] {
            let five = serde_json::json!({"stop": ["a", "b", "c", "d", "e"]});
            assert!(err_of(build(five)).contains("stop"));
            let long = serde_json::json!({"stop": ["x".repeat(STOP_ENTRY_MAX_CHARS + 1)]});
            assert!(err_of(build(long)).contains("stop[0]"));
            let big = serde_json::json!({"max_tokens": MAX_TOKENS_CAP + 1});
            assert!(err_of(build(big)).contains("max_tokens"));
        }
        let five = serde_json::json!({"stop_sequences": ["a", "b", "c", "d", "e"]});
        assert!(err_of(anthropic(five)).contains("stop_sequences"));
        let big = serde_json::json!({"max_tokens": MAX_TOKENS_CAP + 1});
        assert!(err_of(anthropic(big)).contains("max_tokens"));
        let five = serde_json::json!({"stop": ["a", "b", "c", "d", "e"]});
        assert!(err_of(ollama(five)).contains("options.stop"));
    }

    #[test]
    fn logprob_alternatives_are_capped_on_every_route() {
        let over = TOP_LOGPROBS_MAX as usize + 1;
        let s = SamplingRequest::from_ollama(None, Some(true), Some(over));
        assert!(err_of(s).contains("top_logprobs"));
        let s = SamplingRequest::from_completions(&serde_json::json!({"logprobs": over}));
        assert!(err_of(s).contains("top_logprobs"));
    }

    /// The OpenAI penalties become one repeat penalty, none when the request names none; an
    /// explicit one wins; an Ollama request naming none keeps the defaults.
    #[test]
    fn penalties_fold_into_the_repeat_penalty() {
        let s = SamplingRequest::from_completions(
            &serde_json::json!({"frequency_penalty": 2.0, "presence_penalty": 2.0}),
        );
        assert_eq!(s.into_params().repeat_penalty, Some(3.0));
        let s = SamplingRequest::from_completions(&serde_json::json!({}));
        assert_eq!(s.into_params().repeat_penalty, Some(1.0));
        let s = SamplingRequest::from_ollama(Some(&serde_json::json!({})), None, None);
        assert_eq!(s.into_params().repeat_penalty, None);
        let s = SamplingRequest::from_completions(
            &serde_json::json!({"frequency_penalty": 0.0, "presence_penalty": 0.0}),
        );
        assert_eq!(s.into_params().repeat_penalty, Some(1.0));
        let s = SamplingRequest::from_ollama(
            Some(&serde_json::json!({"repeat_penalty": 1.2})),
            None,
            None,
        );
        assert_eq!(s.into_params().repeat_penalty, Some(1.2));
    }

    /// OpenAI chat reads `top_k` when a client sends it as an extension.
    #[test]
    fn openai_chat_reads_top_k() {
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "m", "messages": [], "top_k": 40
        }))
        .unwrap();
        assert_eq!(
            SamplingRequest::from_openai(&req).into_params().top_k,
            Some(40)
        );
    }

    /// Ollama's `-1` asks for no cap.
    #[test]
    fn ollama_minus_one_is_uncapped() {
        let s =
            SamplingRequest::from_ollama(Some(&serde_json::json!({"num_predict": -1})), None, None);
        assert!(s.validate().is_ok());
        assert_eq!(s.into_params().max_tokens, None);
    }
}
