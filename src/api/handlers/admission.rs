//! Getting a generation onto an engine, the same way for every route that runs one.
//!
//! A request loads its model with a window that holds it, is told apart
//! a model that did not fit from one that does not exist, waits its turn at the gate
//! unless the model batches concurrent requests itself, is counted in flight while it
//! runs, and hands the vision encoder the images of its last user turn - or clears what
//! an earlier request left there.

use super::*;

/// How a request wants its model loaded.
#[derive(Clone, Copy)]
pub(crate) struct LoadRequest {
    pub keep_alive_minutes: i64,
    /// The context window asked for, bounded by the checkpoint at load: the Ollama
    /// `options.num_ctx`, or the window a prompt needs (see [`APIServer::fit_window`]).
    pub window: Option<usize>,
}

impl LoadRequest {
    /// A load in the window the model is configured with, kept for `keep_alive_minutes`.
    pub(crate) fn new(keep_alive_minutes: i64) -> Self {
        Self {
            keep_alive_minutes,
            window: None,
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
    /// requested window at once, a loaded model is reloaded only when the window is wider
    /// than the one it holds. The cache format is the placement's.
    pub(crate) async fn load_for_request(
        &self,
        model_name: &str,
        load: LoadRequest,
    ) -> Result<Arc<LlmEngine>, ApiError> {
        self.ensure_loaded_with_window(model_name, load.keep_alive_minutes, load.window)
            .await
            .map_err(|e| load_error(model_name, &e))?;
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

    /// The engine serving `model_name` with a window that holds `prompt` and the
    /// `max_tokens` the request may write: `engine` when its window already does, else the
    /// model reloaded with a wider window (see [`grown_window`]), as far as the checkpoint
    /// declares. A prompt longer than the widest window the model takes is refused.
    pub(crate) async fn fit_window(
        &self,
        engine: Arc<LlmEngine>,
        model_name: &str,
        prompt: &str,
        max_tokens: Option<usize>,
        load: LoadRequest,
    ) -> Result<Arc<LlmEngine>, ApiError> {
        let need = engine
            .count_tokens(prompt)
            .await
            .map(|tokens| tokens + max_tokens.unwrap_or(1));
        let engine = match (need, engine.context_window().await) {
            (Some(need), Some(window)) if need > window => {
                self.load_for_request(
                    model_name,
                    LoadRequest {
                        window: Some(grown_window(need, window)),
                        ..load
                    },
                )
                .await?
            }
            _ => engine,
        };
        ensure_prompt_fits(&engine, prompt).await?;
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
        self.default_inference_config
            .media_limits
            .check_images(images)
            .map_err(ApiError::Validation)?;
        let arrived = std::time::Instant::now();
        let in_flight = crate::distributed::rate_meter::InFlight::enter();
        let active = engine.admit_request().await;
        let mut gate = if engine.is_continuous().await {
            None
        } else {
            Some(acquire_gate(self, priority, model_name.to_string(), endpoint).await?)
        };
        // The move into the batch worker is made under the serial gate, which every serial
        // generation holds to its end; the gate is not needed past it. A request that waited
        // at the gate while the model moved into its batch worker gives the gate up too, and
        // the requests queued behind it reach the worker at once.
        if gate.is_some() {
            if active.wants_batched() {
                engine.serve_batched().await;
            }
            if engine.is_continuous().await {
                gate = None;
            }
        }
        crate::request_metrics::record_gate_wait(model_name, arrived.elapsed().as_secs_f64());
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

    /// Hand the loaded model the video clip of a chat request, after [`Self::admit`] set its
    /// images: the clip's temporal patches follow them, as the prompt states them. A request
    /// without a clip does nothing; a clip for a model that reads none, or one that does not
    /// decode, is refused.
    pub(crate) async fn admit_video(
        &self,
        engine: &LlmEngine,
        clip: Option<&str>,
    ) -> Result<(), ApiError> {
        let Some(clip) = clip else {
            return Ok(());
        };
        if let Err(e) = engine.set_video(clip).await {
            // What an earlier request encoded must not answer for this clip.
            engine.clear_images().await;
            return Err(ApiError::Validation(format!("video: {e}")));
        }
        Ok(())
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
        let limits = &self.default_inference_config.media_limits;
        if let Some(seconds) = super::audio::audio_header_seconds(&bytes) {
            limits
                .check_audio_seconds(seconds)
                .map_err(ApiError::Validation)?;
        }
        let rate = u32::try_from(input.sample_rate).map_err(|_| {
            ApiError::Internal("the speech tower's sample rate is out of range".into())
        })?;
        let pcm = tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<f32>, f64)> {
            let (planar, channels, source_rate) = super::audio::decode_audio_planar(&bytes)?;
            let frames = planar.len() / channels;
            let seconds = frames as f64 / f64::from(source_rate);
            // Mono: the mean of the channels.
            let mono: Vec<f32> = (0..frames)
                .map(|i| {
                    (0..channels).map(|c| planar[c * frames + i]).sum::<f32>() / channels as f32
                })
                .collect();
            Ok((
                crate::inference::media::audio_io::resample(&mono, source_rate, rate)?,
                seconds,
            ))
        })
        .await
        .map_err(|e| ApiError::Internal(format!("audio decode task: {e}")))?
        .map_err(|e| ApiError::Validation(format!("audio: {e}")))?;
        // The decoded length, for a container whose header does not state it.
        let (pcm, seconds) = pcm;
        limits
            .check_audio_seconds(seconds)
            .map_err(ApiError::Validation)?;
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

/// The window a model reloads with when a request needs `need` positions and its window holds
/// `window`: at least twice the window, so a conversation that grows a turn at a time reloads a
/// handful of times rather than at every turn, and an answer without a cap keeps room to run.
/// The load bounds it by what the checkpoint declares.
fn grown_window(need: usize, window: usize) -> usize {
    need.max(window.saturating_mul(2))
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

    /// A window that must grow at least doubles, and grows further when the request needs it.
    #[test]
    fn a_window_that_grows_at_least_doubles() {
        assert_eq!(grown_window(2049, 2048), 4096);
        assert_eq!(grown_window(3746, 2048), 4096);
        assert_eq!(grown_window(9000, 2048), 9000);
    }
}
