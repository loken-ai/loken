//! Getting a generation onto an engine, the same way for every route that runs one.
//!
//! A request loads its model in the window and cache format it asked for, is told apart
//! a model that did not fit from one that does not exist, waits its turn at the gate
//! unless the model batches concurrent requests itself, is counted in flight while it
//! runs, and hands the vision encoder the images of its last user turn - or clears what
//! an earlier request left there.

use super::*;
use crate::inference::engine::llm_engine::KvQuant;

/// How a request wants its model loaded.
#[derive(Clone, Copy)]
pub(crate) struct LoadRequest {
    pub keep_alive_minutes: i64,
    /// The context window asked for, bounded by the checkpoint at load.
    pub window: Option<usize>,
    pub kv_quant: Option<KvQuant>,
}

impl LoadRequest {
    /// The window and cache format a request's headers ask for.
    pub(crate) fn from_headers(headers: &axum::http::HeaderMap, keep_alive_minutes: i64) -> Self {
        Self {
            keep_alive_minutes,
            window: crate::api::gate::Window::from_headers(headers),
            kv_quant: crate::api::gate::KvFormat::from_headers(headers),
        }
    }
}

/// The error a failed load answers with: a model that exists and did not fit is not a
/// model that does not exist, and a client that trusts 404 stops asking for it. The load
/// path escalates before giving up, so running out of memory here means the machine is
/// full right now.
pub(crate) fn load_error(model_name: &str, e: &str) -> ApiError {
    if crate::inference::place::ladder::is_out_of_memory(e) {
        return ApiError::Busy(format!(
            "Model '{model_name}' exists but the machine has no room for it right now: {e}. \
             Retry once the current work finishes."
        ));
    }
    ApiError::NotFound(format!("Model '{model_name}' could not be loaded: {e}"))
}

/// Refuses a prompt longer than the loaded model's context window. Served anyway, it would
/// be cut to fit and the answer written from what was left: a dropped system prompt or tool
/// instruction reads as a plausible wrong answer rather than as an error.
pub(crate) async fn ensure_prompt_fits(engine: &LlmEngine, prompt: &str) -> Result<(), ApiError> {
    if let (Some(tokens), Some(window)) = (
        engine.count_tokens(prompt).await,
        engine.context_window().await,
    ) {
        if tokens > window {
            return Err(ApiError::Validation(format!(
                "the prompt is {tokens} tokens and the model's context window is {window}; \
                 shorten the conversation or load the model with a larger num_ctx"
            )));
        }
    }
    Ok(())
}

/// A generation admitted to its engine. Dropping it releases the gate slot and the
/// in-flight count, so a stream holds it until its last chunk.
pub(crate) struct Admission {
    _gate: Option<crate::api::gate::RequestGuard>,
    _in_flight: crate::distributed::rate_meter::InFlight,
    _active: crate::inference::engine::llm_engine::ActiveRequest,
}

impl APIServer {
    /// The engine serving `model_name`, loaded as `load` asks: a cold load opens the
    /// requested window and cache format at once, a loaded model is reloaded only when
    /// the format differs or the window is wider than the one it holds.
    pub(crate) async fn load_for_request(
        &self,
        model_name: &str,
        load: LoadRequest,
    ) -> Result<Arc<LlmEngine>, ApiError> {
        self.ensure_loaded_with(
            model_name,
            load.keep_alive_minutes,
            load.window,
            load.kv_quant,
        )
        .await
        .map_err(|e| load_error(model_name, &e))?;
        // The format before the window: the format decides what a window costs, and
        // settling it second would size the cache twice.
        if let Some(want) = load.kv_quant {
            self.ensure_engine_kv_quant(model_name, want)
                .await
                .map_err(|e| {
                    ApiError::Internal(format!("kv_quant {want:?} for '{model_name}': {e}"))
                })?;
        }
        if let Some(want) = load.window {
            self.ensure_engine_context(model_name, want)
                .await
                .map_err(|e| {
                    ApiError::Internal(format!("window {want} for '{model_name}': {e}"))
                })?;
        }
        let engine = self.get_engine(model_name).await.map_err(|_| {
            ApiError::NotFound(format!(
                "Model '{model_name}' is not loaded after loading it"
            ))
        })?;
        self.reset_expiration(model_name).await;
        Ok(engine)
    }

    /// Admit one generation on `engine`: the gate slot for a model that serves one
    /// request at a time (a continuously batching model takes concurrent requests in its
    /// own worker), the in-flight count, and the images. After the gate, since setting
    /// images takes the model lock a running decode holds.
    pub(crate) async fn admit(
        &self,
        engine: Arc<LlmEngine>,
        model_name: &str,
        priority: crate::api::gate::Priority,
        endpoint: &'static str,
        images: &[String],
    ) -> Result<Admission, ApiError> {
        let in_flight = crate::distributed::rate_meter::InFlight::enter();
        let active = engine.admit_request().await;
        let gate = if engine.is_continuous().await {
            None
        } else {
            Some(acquire_gate(self, priority, model_name.to_string(), endpoint).await?)
        };
        if images.is_empty() {
            engine.clear_images().await;
        } else {
            // Kept as text: the encoder's error is not `Send`.
            let failure = engine.set_images(images).await.err().map(|e| e.to_string());
            if let Some(e) = failure {
                // What an earlier request encoded must not answer for these images.
                engine.clear_images().await;
                return Err(ApiError::Validation(format!("images: {e}")));
            }
        }
        Ok(Admission {
            _gate: gate,
            _in_flight: in_flight,
            _active: active,
        })
    }

    /// Hand the loaded model the clip of a chat request: `clip`, base64 in any container the
    /// server reads, decoded to mono at the tower's rate and encoded. Called after
    /// [`Self::admit`], which holds the gate and cleared what an earlier request held. A
    /// request without a clip does nothing; a clip for a model with no speech tower, a clip
    /// that does not decode or one longer than the context window holds is refused.
    #[cfg(feature = "audio")]
    pub(crate) async fn admit_audio(
        &self,
        engine: &LlmEngine,
        clip: Option<&str>,
    ) -> Result<(), ApiError> {
        use base64::Engine as _;
        let Some(clip) = clip else {
            return Ok(());
        };
        let input = engine.audio_input().await.ok_or_else(|| {
            ApiError::Validation(
                "this model takes no audio: its files carry no speech tower".into(),
            )
        })?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(crate::api::types::base64_payload(clip).trim())
            .map_err(|e| ApiError::Validation(format!("audio: not base64 ({e})")))?;
        let rate = u32::try_from(input.sample_rate).map_err(|_| {
            ApiError::Internal("the speech tower's sample rate is out of range".into())
        })?;
        let pcm = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<f32>> {
            let (planar, channels, source_rate) = super::audio::decode_audio_planar(&bytes)?;
            let frames = planar.len() / channels;
            // Mono: the mean of the channels.
            let mono: Vec<f32> = (0..frames)
                .map(|i| {
                    (0..channels).map(|c| planar[c * frames + i]).sum::<f32>() / channels as f32
                })
                .collect();
            crate::inference::media::audio_io::resample(&mono, source_rate, rate)
        })
        .await
        .map_err(|e| ApiError::Internal(format!("audio decode task: {e}")))?
        .map_err(|e| ApiError::Validation(format!("audio: {e}")))?;
        let embeds = engine.audio_embeds_for(pcm.len()).await;
        if let Some(window) = engine.context_window().await {
            if embeds >= window {
                return Err(ApiError::Validation(format!(
                    "the clip needs {embeds} positions and the context window holds {window}; \
                     send a shorter clip or raise num_ctx"
                )));
            }
        }
        let failure = engine.set_audio(pcm).await.err().map(|e| e.to_string());
        if let Some(e) = failure {
            engine.clear_images().await;
            return Err(ApiError::Validation(format!("audio: {e}")));
        }
        Ok(())
    }

    #[cfg(not(feature = "audio"))]
    pub(crate) async fn admit_audio(
        &self,
        _engine: &LlmEngine,
        clip: Option<&str>,
    ) -> Result<(), ApiError> {
        match clip {
            None => Ok(()),
            Some(_) => Err(ApiError::Validation(
                "this build takes no audio input (built without the `audio` feature)".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Running out of memory is a retryable refusal on every route, not a missing model.
    #[test]
    fn a_full_machine_is_not_a_missing_model() {
        assert!(matches!(
            load_error("m", "cuda OOM in QCudaStorage upload on GPU0"),
            ApiError::Busy(_)
        ));
        assert!(matches!(
            load_error("m", "CUDA error: out of memory"),
            ApiError::Busy(_)
        ));
        assert!(matches!(
            load_error("m", "no such model on disk"),
            ApiError::NotFound(_)
        ));
    }

    #[test]
    fn the_headers_carry_window_and_format() {
        let mut h = axum::http::HeaderMap::new();
        h.insert(crate::api::gate::Window::HEADER, "32768".parse().unwrap());
        h.insert(crate::api::gate::KvFormat::HEADER, "q8".parse().unwrap());
        let l = LoadRequest::from_headers(&h, 5);
        assert_eq!(l.window, Some(32768));
        assert!(matches!(l.kv_quant, Some(KvQuant::Q8)));
        assert_eq!(l.keep_alive_minutes, 5);
    }
}
