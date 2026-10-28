//! The fields vLLM's server reads on the OpenAI routes beyond OpenAI's own: the request
//! shapes alone, read into the chat request and from a completion's body. What they mean is
//! `api::handlers::openai::vllm`.

use serde::{Deserialize, Serialize};

/// `structured_outputs`: exactly one constraint, and the options of a JSON one.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StructuredOutputs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub choice: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_object: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structural_tag: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_any_whitespace: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub disable_additional_properties: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub whitespace_pattern: Option<String>,
}

/// The fields vLLM's server reads on `/v1/completions` and `/v1/chat/completions` beyond
/// OpenAI's. Every one is optional; an absent one is vLLM's default.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VllmFields {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bad_words: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allowed_token_ids: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_token_ids: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_stop_str_in_output: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skip_special_tokens: Option<bool>,
    /// Accepted; the tokenizer writes no spaces between special tokens either way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spaces_between_special_tokens: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncate_prompt_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncation_side: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_logprobs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_tokens_as_token_ids: Option<bool>,
    /// `n` beams answered, the beam width being `n`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_beam_search: Option<bool>,
    /// Read by beam search alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length_penalty: Option<f64>,
    /// Read by beam search alone: `true`, `false` or `"never"`, as transformers reads it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub early_stopping: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_outputs: Option<StructuredOutputs>,
    /// The constraints vLLM read before `structured_outputs`, served the same way.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guided_json: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guided_regex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guided_choice: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guided_grammar: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guided_whitespace_pattern: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_salt: Option<String>,
    /// Chat: whether the template opens the assistant's turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add_generation_prompt: Option<bool>,
    /// Chat: the last message, an assistant's, is continued rather than closed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continue_final_message: Option<bool>,
    /// Chat: a template sent with the request, refused as vLLM refuses it by default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    /// Chat: variables handed to the model's template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    /// Chat: documents handed to the model's template as `documents`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub documents: Option<Vec<serde_json::Map<String, serde_json::Value>>>,
    /// Chat: the final assistant message is written in front of the answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    /// Chat: `false` leaves the reasoning out of the answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_reasoning: Option<bool>,
}
