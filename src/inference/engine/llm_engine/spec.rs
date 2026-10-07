//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl LlmEngine {
    /// Cheap clone for spec decode. The LlmEngine struct holds Arc'd
    /// state so .clone() is just bumping reference counts.
    pub(super) fn clone_for_spec(&self) -> Self {
        Self {
            config: self.config.clone(),
            model_state: self.model_state.clone(),
            draft_engine: self.draft_engine.clone(),
            kv_disk: self.kv_disk.clone(),
            state_cache: self.state_cache.clone(),
            cancel: self.cancel.clone(),
            logprob_queue: self.logprob_queue.clone(),
            last_error: self.last_error.clone(),
            cached_model_size: self.cached_model_size.clone(),
            sessions: self.sessions.clone(),
            last_stream_stats: self.last_stream_stats.clone(),
            embedding_model: self.embedding_model.clone(),
            active: self.active.clone(),
            batching: self.batching.clone(),
            drafter_choice: self.drafter_choice.clone(),
            draft_length: self.draft_length.clone(),
            pld_cost: self.pld_cost.clone(),
        }
    }
}
