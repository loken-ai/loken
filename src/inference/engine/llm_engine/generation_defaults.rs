//! The sampling a request gets for every knob it leaves unset.
//!
//! A model ships its own recommended sampling: an Ollama model as the `params` layer of its
//! manifest (the Modelfile's `PARAMETER` lines), a Hugging Face checkpoint as
//! `generation_config.json`. Those win over any server-wide value, because they are the
//! values the model was tuned and evaluated with. What neither states falls back to the
//! defaults of the Ollama API, the one clients of this server already assume.

use std::path::Path;

/// Temperature the Ollama API applies when neither the request nor the model sets one.
pub const API_TEMPERATURE: f32 = 0.8;
/// Nucleus mass the Ollama API applies by default.
pub const API_TOP_P: f32 = 0.9;
/// Candidate count the Ollama API applies by default.
pub const API_TOP_K: usize = 40;
/// Repetition penalty the Ollama API applies by default.
pub const API_REPEAT_PENALTY: f32 = 1.1;
/// Window the repetition penalty looks back over, as the Ollama API defaults it.
pub const API_REPEAT_LAST_N: usize = 64;

/// What a model recommends; `None` where it says nothing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationDefaults {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    /// Sequences the model's Modelfile stops on, applied when the request names none.
    pub stop: Vec<String>,
}

impl GenerationDefaults {
    /// The `params` layer of an Ollama manifest: a JSON object keyed like the Modelfile.
    pub fn from_ollama_params(json: &str) -> Self {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
            return Self::default();
        };
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
        let u = |k: &str| {
            v.get(k)
                .and_then(|x| x.as_i64())
                .filter(|x| *x >= 0)
                .map(|x| x as usize)
        };
        let stop = match v.get("stop") {
            Some(serde_json::Value::Array(xs)) => xs
                .iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect(),
            Some(serde_json::Value::String(s)) => vec![s.clone()],
            _ => Vec::new(),
        };
        Self {
            temperature: f("temperature"),
            top_p: f("top_p"),
            top_k: u("top_k"),
            repeat_penalty: f("repeat_penalty"),
            repeat_last_n: u("repeat_last_n"),
            stop,
        }
    }

    /// A Hugging Face `generation_config.json`. Its sampling applies only when it asks to
    /// sample: with `do_sample` false the checkpoint is meant to be decoded greedily.
    pub fn from_hf_generation_config(json: &str) -> Self {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(json) else {
            return Self::default();
        };
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
        let greedy = v.get("do_sample").and_then(|x| x.as_bool()) == Some(false);
        Self {
            temperature: if greedy { Some(0.0) } else { f("temperature") },
            top_p: f("top_p"),
            top_k: v.get("top_k").and_then(|x| x.as_u64()).map(|x| x as usize),
            repeat_penalty: f("repetition_penalty"),
            repeat_last_n: None,
            stop: Vec::new(),
        }
    }

    /// `generation_config.json` beside a checkpoint, if there is one.
    pub fn from_hf_dir(dir: &Path) -> Self {
        std::fs::read_to_string(dir.join("generation_config.json"))
            .map(|s| Self::from_hf_generation_config(&s))
            .unwrap_or_default()
    }

    pub fn temperature(&self) -> f32 {
        self.temperature.unwrap_or(API_TEMPERATURE)
    }
    pub fn top_p(&self) -> f32 {
        self.top_p.unwrap_or(API_TOP_P)
    }
    pub fn top_k(&self) -> usize {
        self.top_k.unwrap_or(API_TOP_K)
    }
    pub fn repeat_penalty(&self) -> f32 {
        self.repeat_penalty.unwrap_or(API_REPEAT_PENALTY)
    }
    pub fn repeat_last_n(&self) -> usize {
        self.repeat_last_n.unwrap_or(API_REPEAT_LAST_N)
    }
}

/// A seed for a request that names none: every unseeded request samples its own stream, as
/// the Ollama API does, instead of all of them replaying one.
pub fn fresh_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0),
    );
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_modelfile_sets_what_it_names_and_leaves_the_rest_to_the_api() {
        let d = GenerationDefaults::from_ollama_params(
            r#"{"temperature":0.6,"top_k":20,"stop":["<|im_end|>","<|endoftext|>"]}"#,
        );
        assert_eq!(d.temperature(), 0.6);
        assert_eq!(d.top_k(), 20);
        assert_eq!(d.top_p(), API_TOP_P);
        assert_eq!(d.stop, vec!["<|im_end|>", "<|endoftext|>"]);
    }

    #[test]
    fn a_greedy_checkpoint_decodes_greedily() {
        let d = GenerationDefaults::from_hf_generation_config(
            r#"{"do_sample":false,"temperature":0.7,"repetition_penalty":1.05}"#,
        );
        assert_eq!(d.temperature(), 0.0);
        assert_eq!(d.repeat_penalty(), 1.05);
    }

    #[test]
    fn nothing_stated_is_the_api_defaults() {
        let d = GenerationDefaults::from_ollama_params("not json");
        assert_eq!(d, GenerationDefaults::default());
        assert_eq!(d.temperature(), API_TEMPERATURE);
        assert_eq!(d.repeat_last_n(), API_REPEAT_LAST_N);
    }

    #[test]
    fn unseeded_requests_draw_different_seeds() {
        assert_ne!(fresh_seed(), fresh_seed());
    }
}
