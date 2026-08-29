//! Inference engine implementations

/// Trait for inference engines
pub trait InferenceEngine {
    /// Create a new instance of the engine
    fn new() -> Self
    where
        Self: Sized;

    /// Execute inference on the given inputs
    fn execute(
        &self,
        inputs: Vec<crate::tensor::Tensor>,
    ) -> Result<Vec<crate::tensor::Tensor>, Box<dyn std::error::Error>>;
}

// Export concrete implementations
#[cfg(feature = "audio")]
pub mod audio_engine;
#[cfg(feature = "image")]
pub mod boogu_engine;
pub(crate) mod decode_step;
#[cfg(feature = "image")]
pub mod flux2_engine;
#[cfg(feature = "image")]
pub mod image_engine;
pub mod llm_engine;
pub(crate) mod model_backend;
pub(crate) mod prompt_cache;
#[cfg(feature = "image")]
pub mod qwen_image_engine;
#[cfg(feature = "audio")]
pub mod tts_engine;

#[cfg(feature = "audio")]
pub use audio_engine::{AudioEngine, AudioTranscribeParams};
#[cfg(feature = "image")]
pub use image_engine::ImageEngine;
pub use llm_engine::LlmEngine;
#[cfg(feature = "audio")]
pub use tts_engine::{TtsEngine, TtsSynthParams};

/// Centralised engine-failure logger. Every engine error path  - 
/// text-gen, image-gen, audio (ASR), TTS - funnels through here so
/// a 5xx-shaped response, an SSE error event, or a swallowed
/// channel error all surface in the server log with consistent
/// formatting and the full anyhow chain ({e:#}).
///
/// Long-standing user complaint: errors that fired through async
/// channels / SSE bodies never appeared in server logs because they
/// never hit a 5xx HTTP status (so TraceLayer's on_failure couldn't
/// see them). This is the modality-agnostic fix - every engine call
/// site invokes log_engine_error before returning the failure
/// upstream, regardless of how the error eventually reaches the
/// client.
pub(crate) fn log_engine_error<E: std::fmt::Display>(engine: &str, op: &str, err: E) {
    // Generic over Display so callers can pass an anyhow::Error
    // (which uses {:#} for the chain) OR a Box<dyn Error>
    // OR a plain String - all surface in the log with the same
    // formatting.
    tracing::error!("inference engine error: engine={engine} op={op}: {err:#}");
}
