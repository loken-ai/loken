#[cfg(feature = "cuda")]
use crate::inference::engine::decode_step::{
    incremental_chunk_text, pld_verify_commit, pld_window_update, resolve_gen_params,
    sample_row_sync, spec_draft_lockstep, ResolvedGenParams, StopTracker,
};
#[cfg(feature = "cuda")]
use crate::inference::engine::model_backend::TpQwen2Backend;
use crate::inference::engine::model_backend::{
    BoxedModelBackend, ContinuousBackend, GenericBackend, GenericVisionBackend, GptOssBackend,
    Lfm2MoeBackend, MoondreamBackend, NemotronHBackend, Qwen35MoeBackend, QwenMoEMultiBackend,
    TakenBackend,
};
use crate::inference::engine::InferenceEngine;
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::inference::model::moondream::quantized as moondream;
use crate::inference::sample::token_sampling::{LogitsProcessor, Sampling};
use crate::tensor::quantized::gguf_file;
use crate::tensor::{Device, Tensor};
use anyhow::{anyhow, Result as AnyResult};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Once};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, error, info, trace, warn};

/// Prefill compute-batch size: long prompts are forwarded in chunks of this
/// many tokens so the per-forward activation peak stays bounded at
/// `chunk x ffn` rather than scaling with the whole prompt. This is the same
/// knob as llama.cpp/ollama's `num_batch` (default 512) - a COMPUTE batch,
/// not a memory reserve. Bounding the activation this way is what lets GPU
/// placement be sized from weights + KV alone, with no activation guess.
/// Short prompts (`seq <= 512`, the common case) take a single forward and are
/// byte-identical to the unchunked path.
pub(crate) const PREFILL_CHUNK_TOKENS: usize = 512;

/// Reserved sessions-table key for the single in-memory "resident KV" snapshot
/// (the token sequence currently in the model's KV cache). Used by the ollama-style
/// GLOBAL prompt cache: a request WITHOUT a session_id reuses the common prefix of
/// this resident sequence (in-memory only - never persisted to disk; see
/// project_prompt_cache_privacy). Updated after every request so it always mirrors
/// what is actually resident. Disable for multi-tenant privacy with
/// a vision request (the cross-tenant timing side-channel that
/// session-gating otherwise prevents).
pub(crate) const GLOBAL_PROMPT_CACHE_KEY: &str = "__loken_global_resident_kv__";

/// Smallest prefill compute-batch the OOM-adaptive prefill will fall back to
/// before giving up on the GPU path. At 32 tokens the per-forward activation
/// peak (`32 x widest_ffn x f32`) is ~16x smaller than the 512-token default  - 
/// enough to fit in the few hundred MB of headroom that a near-full 2-GPU TP
/// model (deepseek-r1:32b) or a co-resident MoE (gemma4:26b) leaves after the
/// weights + KV cache. Going below 32 buys little (the attention scores tensor
/// and KV-grow transients no longer dominate) while multiplying launch count,
/// so 32 is the floor.
pub(crate) const PREFILL_CHUNK_TOKENS_MIN: usize = 32;

/// Adaptive prefill compute-batch memory. After the OOM-adaptive prefill finds
/// a chunk size that fits the loaded model on this hardware, it is remembered
/// here so subsequent requests start at the known-good size instead of paying
/// the full 512->256->...->fit shrink ladder every time (each rung replays the whole
/// prefill - a real per-request cost on a near-full 2-GPU / co-resident model).
/// 0 = "no adaptation yet, start at the default". The engine serves one model
/// and serializes forwards through the model lock, so a single global is correct;
/// it is reset on model load so a freshly loaded (possibly larger-headroom) model
/// starts fresh. Worst case if stale: one extra successful prefill at a slightly
/// smaller-than-necessary chunk, never an OOM.
pub(crate) static ADAPTIVE_PREFILL_CHUNK: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Reset the adaptive prefill chunk memory (called on model load/unload).
pub(crate) fn reset_adaptive_prefill_chunk() {
    ADAPTIVE_PREFILL_CHUNK.store(0, std::sync::atomic::Ordering::Relaxed);
}
mod params;
pub use params::*;
mod load;
pub use load::*;

impl LoadedModelState {
    /// QuantizedMoondream-aware forward: uses captured CUDA graph for
    /// single-token decode when available, falls back to plain forward
    /// otherwise. For all other variants delegates to
    /// ModelBackend::forward unchanged.
    ///
    /// step/N: graph capture dispatch. Capture happens after
    /// `MOONDREAM_GRAPH_WARMUP` decode tokens to let stable buffers
    /// get exercised + warmed.
    pub fn forward(&mut self, x: &Tensor, pos: usize) -> crate::tensor::Result<Tensor> {
        // Plain forward for non-moondream variants OR for prefill (seq>1).
        let is_moondream = self.model.is_moondream();
        let seq = x.dim(1).unwrap_or(1);
        if !is_moondream || seq > 1 {
            return self.model.forward(x, pos);
        }
        #[cfg(feature = "cuda")]
        return self.forward_moondream_graph(x, pos);
        #[cfg(not(feature = "cuda"))]
        return self.model.forward(x, pos);
    }

    /// Moondream CUDA-graph capture/replay decode path (CUDA-only). On
    /// non-CUDA builds the caller falls back to the plain forward.
    #[cfg(feature = "cuda")]
    fn forward_moondream_graph(&mut self, x: &Tensor, pos: usize) -> crate::tensor::Result<Tensor> {
        // Only CUDA gets graph capture.
        let cuda_dev = match self.device.as_cuda_device() {
            Ok(d) => d,
            _ => return self.model.forward(x, pos),
        };
        const MOONDREAM_GRAPH_WARMUP: u32 = 3;
        let token: u32 = x.flatten_all()?.to_vec1::<u32>()?[0];

        // Step 1: ensure input_buf + set token
        {
            let m = self.model.moondream_mut().expect("moondream backend");
            m.text_model().ensure_input_buf(&self.device)?;
            m.text_model().set_input_id(token)?;
        }

        // Step 2: if captured graph exists, launch + read logits
        if self.moondream_graph.is_some() {
            let stream = cuda_dev.cuda_stream();
            let launch_res = self.moondream_graph.as_ref().unwrap().graph.launch();
            match launch_res {
                Ok(()) => {
                    let _ = stream.synchronize();
                    let m = self.model.moondream_mut().expect("moondream backend");
                    if let Some(logits) = m.text_model().logits_buf() {
                        return Ok(logits.clone());
                    }
                    warn!("moondream graph replay: logits_buf missing; clearing graph");
                    self.moondream_graph = None;
                }
                Err(e) => {
                    warn!("moondream graph replay failed: {e:?}; clearing graph");
                    self.moondream_graph = None;
                }
            }
        }

        // Step 3: pre-warmup forwards (use input_buf so all subsequent
        // ops bind to its device pointer).
        self.moondream_decode_count = self.moondream_decode_count.saturating_add(1);
        let input_buf = {
            let m = self.model.moondream_mut().expect("moondream backend");
            m.text_model().input_buf().unwrap().clone()
        };
        if self.moondream_decode_count < MOONDREAM_GRAPH_WARMUP {
            return self.model.forward(&input_buf, pos);
        }

        // Step 4: capture attempt.
        let stream = cuda_dev.cuda_stream();
        if let Err(e) = crate::tensor::cuda_ext::begin_capture(&stream) {
            warn!("moondream begin_capture failed: {e:?}");
            return self.model.forward(&input_buf, pos);
        }
        let fwd_res = self.model.forward(&input_buf, pos);
        let cap_res = crate::tensor::cuda_ext::end_capture(&stream);
        match (fwd_res, cap_res) {
            (Ok(logits), Ok(Some(graph))) => {
                let nc = graph.num_nodes().unwrap_or(0);
                // DEFENSE IN DEPTH: never store (and thus never replay) an
                // empty graph - a 0-node capture means the forward recorded no
                // work, so replaying it would freeze the logits buffer ->
                // constant-token output. Re-run the forward EAGERLY (capture
                // mode only records; the returned `logits` were never
                // computed) and stay on the plain decode path.
                if nc == 0 {
                    warn!("moondream CUDA graph captured 0 nodes; discarding and falling back to eager decode");
                    drop(graph);
                    return self.model.forward(&input_buf, pos);
                }
                tracing::info!("🟦 moondream CUDA graph captured ({} nodes)", nc);
                let _ = graph.upload();
                self.moondream_graph = Some(MoondreamGraphState { graph });
                Ok(logits)
            }
            (Ok(logits), _) => {
                warn!("moondream end_capture returned None; using fwd logits");
                Ok(logits)
            }
            (Err(e), _) => Err(e),
        }
    }
}

/// The language inference engine
/// Engine-pure timing of the most recent completed stream. Used by
/// `/api/generate` stream handler to emit accurate `eval_duration` in
/// the final NDJSON chunk so assay can compute decode tok/s
/// symmetrically with Ollama.
///
/// `eval_duration_ns` measures only the compute work (forward + sample)
/// inside the decode loop; it EXCLUDES time inside `stream_token` (which
/// covers tokenizer decode + `blocking_send` back-pressure). Without this
/// exclusion, eval_duration would be wall-clock-from-engine-thread,
/// which is asymmetric with Ollama's kernel-only reporting.
#[derive(Debug, Clone)]
pub struct StreamStats {
    pub eval_count: u64,
    pub eval_duration_ns: u64,
    pub prompt_eval_count: u64,
    pub prompt_eval_duration_ns: u64,
    pub total_duration_ns: u64,
}

pub struct LlmEngine {
    config: InferenceConfig,
    model_state: Arc<Mutex<Option<LoadedModelState>>>,
    /// Draft-model speculative decoding: an optional nested engine holding
    /// a small same-vocab drafter (e.g. qwen3:0.6b for qwen3:8b). Lazily loaded
    /// when a drafter is configured for the target. The
    /// drafter forwards K cheap tokens per cycle which the target batch-verifies
    /// in one forward (reads target weights ONCE) - amortizes the weight-GEMV
    /// DRAM wall (perf-confirmed 83% of CPU decode). Boxed to break the recursive
    /// type; `Arc<Mutex>` so the lazy load is interior-mutable behind `&self`.
    draft_engine: Arc<Mutex<Option<Box<LlmEngine>>>>,
    last_error: Arc<Mutex<Option<String>>>,
    cached_model_size: Arc<Mutex<u64>>,
    /// Session-persistent KV cache tracking. A single model has a single live
    /// KV cache, so only one session can be "resident" at a time - the map
    /// acts as a lookup (is the current cache yours?). When session_id
    /// changes the cache is effectively reset by the next `forward(_, 0)`.
    sessions: Arc<Mutex<HashMap<String, SessionState>>>,
    /// Slot for the most recent completed stream's pure-compute timing.
    /// Engine writes when the streaming task finishes; HTTP handler reads
    /// and includes in the stream final chunk.
    last_stream_stats: Arc<std::sync::Mutex<Option<StreamStats>>>,
    /// Lazily-built text-embedding model for `/v1/embeddings` & `/api/embed`,
    /// loaded from this engine's own GGUF on first request (cached behind `&self`).
    embedding_model: Arc<Mutex<Option<Arc<crate::inference::model::embedding::EmbeddingModel>>>>,
}

impl InferenceEngine for LlmEngine {
    fn new() -> Self {
        Self {
            config: InferenceConfig::default(),
            model_state: Arc::new(Mutex::new(None)),
            draft_engine: Arc::new(Mutex::new(None)),
            last_error: Arc::new(Mutex::new(None)),
            cached_model_size: Arc::new(Mutex::new(0)),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            last_stream_stats: Arc::new(std::sync::Mutex::new(None)),
            embedding_model: Arc::new(Mutex::new(None)),
        }
    }

    fn execute(&self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>, Box<dyn std::error::Error>> {
        // Basic pass-through implementation (to be expanded)
        Ok(inputs)
    }
}

/// Streaming-detok char-boundary guard: returns the suffix of `s` from byte `from`,
/// backed off to the nearest char boundary <= `from`. A multi-token commit (PLD /
/// spec-decode) can COMPLETE a multi-byte char (e.g. an emoji) whose bytes differ
/// from the previous step's partial decode, landing `from` INSIDE a char in the new
/// cumulative string - a raw `s[from..]` slice then panics. Used by every incremental
/// stream-decode site.
fn char_safe_suffix(s: &str, from: usize) -> String {
    let mut i = from.min(s.len());
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    s[i..].to_string()
}

impl LlmEngine {
    /// Create new engine with config
    pub fn with_config(config: InferenceConfig) -> Self {
        Self {
            config,
            model_state: Arc::new(Mutex::new(None)),
            draft_engine: Arc::new(Mutex::new(None)),
            last_error: Arc::new(Mutex::new(None)),
            cached_model_size: Arc::new(Mutex::new(0)),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            last_stream_stats: Arc::new(std::sync::Mutex::new(None)),
            embedding_model: Arc::new(Mutex::new(None)),
        }
    }

    /// Take the most recently completed stream's compute-only stats.
    /// Used by `/api/generate` stream handler to emit accurate
    /// `eval_duration` in the NDJSON final chunk. Returns `Some(..)`
    /// once per stream completion (atomic take); next stream overwrites
    /// the slot.
    pub fn take_last_stream_stats(&self) -> Option<StreamStats> {
        self.last_stream_stats
            .lock()
            .ok()
            .and_then(|mut g| g.take())
    }

    fn stream_stats_slot(&self) -> Arc<std::sync::Mutex<Option<StreamStats>>> {
        self.last_stream_stats.clone()
    }

    /// Which model drafts for this one, if any.
    ///
    /// None, measured: on the canonical binary (qwen3:8b drafted by qwen3:0.6b, CPU, greedy,
    /// short prose) the accept rate reaches only ~26%, which makes draft-and-verify
    /// speed-NEUTRAL against plain decode - and at speed parity it does strictly more work,
    /// so it costs more energy per token. It pays where acceptance is high, on structured or
    /// code continuations, and this returns a drafter again when a regime with a measured net
    /// win is identified. The earlier "+45%" was a sibling pure-CPU build, not a result.
    fn spec_draft_model_id(&self) -> Option<String> {
        None
    }

    /// Spec-decode: lazily load the nested drafter engine (once). The drafter
    /// reuses the full `load_model` machinery via a nested `LlmEngine` whose config
    /// is this engine's config with `model_id` swapped to the drafter. Returns true
    /// when a drafter is loaded and ready. No-op (false) when unset/already-failed.
    pub async fn ensure_draft_loaded(&self) -> bool {
        let drafter_id = match self.spec_draft_model_id() {
            Some(d) => d,
            None => return false,
        };
        {
            let g = self.draft_engine.lock().await;
            if g.is_some() {
                return true;
            }
        }
        let mut dcfg = self.config.clone();
        dcfg.model_id = drafter_id.clone();
        let nested = LlmEngine::with_config(dcfg);
        match nested.load_model().await {
            Ok(()) => {
                info!(
                    "🜂 spec-decode: drafter '{}' loaded for target '{}'",
                    drafter_id, self.config.model_id
                );
                *self.draft_engine.lock().await = Some(Box::new(nested));
                true
            }
            Err(e) => {
                warn!(
                    "spec-decode: drafter '{}' failed to load: {e}; spec-decode OFF",
                    drafter_id
                );
                false
            }
        }
    }

    /// Spec-decode: roll the drafter's KV cache back to `len` tokens (after a
    /// partial-accept reject), keeping it in lockstep with the verified target.
    pub fn draft_trim_kv(&self, len: usize) {
        if let Some(d) = self.draft_engine.blocking_lock().as_ref() {
            if let Some(s) = d.model_state.blocking_lock().as_mut() {
                let _ = s.model.trim_kv(len);
            }
        }
    }

    /// Spec-decode: prefill the drafter with the prompt tokens (forward at
    /// pos 0 resets + fills its KV -> length = tokens.len()). Returns false on error.
    pub fn draft_prefill(&self, tokens: &[u32]) -> bool {
        let g = self.draft_engine.blocking_lock();
        let d = match g.as_ref() {
            Some(d) => d,
            None => return false,
        };
        let mut sg = d.model_state.blocking_lock();
        let st = match sg.as_mut() {
            Some(s) => s,
            None => return false,
        };
        let x = match Tensor::new(tokens, &st.device).and_then(|t| t.unsqueeze(0)) {
            Ok(x) => x,
            Err(_) => return false,
        };
        st.model.forward(&x, 0).is_ok()
    }

    /// Spec-decode: greedily draft `k` tokens. Feeds `first_token` at
    /// `start_pos` (drafter's own KV position), then each greedy argmax back in.
    /// Drafter KV advances by exactly `k`. Returns the k drafted tokens (or fewer
    /// on error). Greedy is correct for verify: the target re-samples every
    /// position, so the drafter only needs to PROPOSE the likely continuation.
    ///
    /// Acceptance-rate alignment: the target verify samples each position AFTER a
    /// repeat-penalty over its recent-token window. A drafter that argmaxes the RAW
    /// logits proposes tokens the penalised target will reject whenever a recent
    /// token is the raw winner - depressing the accept rate (measured 22% without
    /// this). So mirror the EXACT same penalty here: drafter step i penalises over
    /// `recent_tokens` extended with the tokens drafted so far (draft[0..i]), which
    /// is precisely the window the target sees at verify position i (the spec cycle
    /// does not touch recent_tokens between draft and verify). penalty<=1.0 -> no-op.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_k(
        &self,
        first_token: u32,
        k: usize,
        start_pos: usize,
        recent_tokens: &[u32],
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Vec<u32> {
        let g = self.draft_engine.blocking_lock();
        let d = match g.as_ref() {
            Some(d) => d,
            None => return Vec::new(),
        };
        let mut sg = d.model_state.blocking_lock();
        let st = match sg.as_mut() {
            Some(s) => s,
            None => return Vec::new(),
        };
        let penalise = repeat_penalty > 1.0 && repeat_last_n > 0;
        // Growing penalty window: target's recent tokens + drafts committed so far.
        let mut window: Vec<u32> = if penalise {
            recent_tokens.to_vec()
        } else {
            Vec::new()
        };
        let mut out = Vec::with_capacity(k);
        let mut tok = first_token;
        let mut pos = start_pos;
        for _ in 0..k {
            let x = match Tensor::new(&[tok], &st.device).and_then(|t| t.unsqueeze(0)) {
                Ok(x) => x,
                Err(_) => break,
            };
            let logits = match st.model.forward(&x, pos) {
                Ok(l) => l,
                Err(_) => break,
            };
            let mut v: Vec<f32> = match logits
                .flatten_all()
                .and_then(|t| t.to_dtype(crate::tensor::DType::F32))
                .and_then(|t| t.to_vec1())
            {
                Ok(v) => v,
                Err(_) => break,
            };
            if penalise {
                let s = window.len().saturating_sub(repeat_last_n);
                for &t in &window[s..] {
                    let i = t as usize;
                    if i < v.len() {
                        if v[i] > 0.0 {
                            v[i] /= repeat_penalty;
                        } else {
                            v[i] *= repeat_penalty;
                        }
                    }
                }
            }
            let am = v
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv {
                        (i, x)
                    } else {
                        (bi, bv)
                    }
                })
                .0 as u32;
            out.push(am);
            if penalise {
                window.push(am);
            }
            tok = am;
            pos += 1;
        }
        out
    }

    /// Get the last error message from model loading
    /// Adapters attached to the loaded model, or nothing when none is loaded.
    pub async fn adapters(&self) -> Vec<String> {
        let guard = self.model_state.lock().await;
        guard.as_ref().map(|s| s.model.adapters()).unwrap_or_default()
    }

    /// Replace the attached adapter set on the loaded model. Empty detaches everything.
    ///
    /// Runs on a blocking thread and holds the model lock, which is the same lock a generate
    /// takes: a swap therefore waits for the token in flight instead of rewriting weights
    /// under it. Sessions are dropped, because a KV cache holds the answers of the weights
    /// that produced it and keeping it would mix two models in one conversation.
    pub async fn set_adapters(
        &self,
        wanted: &[(String, f32)],
    ) -> Result<crate::inference::load::lora::AdapterReport, String> {
        let model_state = self.model_state.clone();
        let wanted: Vec<(String, f32)> = wanted.to_vec();
        let report = tokio::task::spawn_blocking(move || {
            let mut guard = model_state.blocking_lock();
            let state = guard.as_mut().ok_or_else(|| "no model loaded".to_string())?;
            state.model.set_adapters(&wanted).map_err(|e| e.to_string())
        })
        .await
        .map_err(|e| e.to_string())??;
        self.sessions.lock().await.clear();
        Ok(report)
    }

    /// How many sessions hold a KV cache on this engine right now.
    ///
    /// What "busy" means for a node: a session is a conversation whose prefix is resident, so
    /// this is the state a router would lose by sending the next turn elsewhere.
    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }

    pub async fn get_last_error(&self) -> Option<String> {
        let err = self.last_error.lock().await;
        err.clone()
    }

    /// Clear the last error message
    pub async fn clear_last_error(&self) {
        let mut err = self.last_error.lock().await;
        *err = None;
    }

    /// Check if the loaded model is a vision model
    pub async fn is_vision_model(&self) -> bool {
        let guard = self.model_state.lock().await;
        guard.as_ref().is_some_and(|s| s.model.is_vision_model())
    }

    /// True iff the loaded model is qwen35moe WITH a vision ViT - the handler
    /// uses this to inject the `<|vision_start|><|image_pad|><|vision_end|>`
    /// marker into the prompt (qwen35 splices the ViT output at that token).
    pub async fn is_qwen35_vision(&self) -> bool {
        let guard = self.model_state.lock().await;
        guard.as_ref().is_some_and(|s| s.model.is_qwen35_vision())
    }

    /// Encode images and store embeddings for the next generation call.
    /// images: list of base64-encoded image strings
    pub async fn set_images(&self, images: &[String]) -> Result<(), Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let images_owned: Vec<String> = images.to_vec();

        let result = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            let mut guard = model_state.blocking_lock();
            let state = guard.as_mut().ok_or_else(|| anyhow!("Model not loaded"))?;

            if !state.model.is_vision_model() {
                return Err(anyhow!("Model does not support vision"));
            }

            // qwen35moe (Qwen3-VL): store the raw preprocessed patches + grid;
            // the ViT + splice run inside `forward_with_image` at prefill (NOT a
            // moondream-style prepend). The prompt carries one `image_token`
            // sentinel (handler-injected) that the generate path expands.
            if state.model.is_qwen35_vision() {
                if let Some(b64) = images_owned.first() {
                    let img = crate::inference::media::image_processor::decode_base64_image(b64)?;
                    let pp = crate::inference::media::image_processor::Qwen35VisionPreproc {
                        patch_size: 16,
                        merge_size: 2,
                        temporal: 2,
                        channels: 3,
                        shortest_edge: 65536,
                        longest_edge: 1_000_000,
                    };
                    let (px, grid) = crate::inference::media::image_processor::preprocess_qwen35vl(
                        &img,
                        pp,
                        &state.device,
                    )?;
                    state.qwen35_image = Some((px, grid));
                }
                return Ok(());
            }

            // Process the first image (Moondream supports single image).
            // Same-image cache hits skip the CLIP forward + the JPEG
            // preprocess - a ~100 ms saving per request on a 16-block
            // ViT, often the dominant TTFT cost in vision benches.
            if let Some(base64_str) = images_owned.first() {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                base64_str.hash(&mut hasher);
                let image_hash = hasher.finish();
                if let Some((cached_hash, cached_embeds)) = state.image_embed_cache.as_ref() {
                    if *cached_hash == image_hash {
                        debug!("Vision: cache HIT for image hash {:x}", image_hash);
                        state.image_embeds = Some(cached_embeds.clone());
                        return Ok(());
                    }
                }
                let image_embeds = if state.model.is_pixtral_vision() {
                    // Pixtral: variable-resolution - encode from the RAW image
                    // (HF-exact preprocess inside), NOT a fixed-size tensor.
                    let img =
                        crate::inference::media::image_processor::decode_base64_image(base64_str)?
                            .to_rgb8();
                    state.model.encode_raw_image(&img)?
                } else {
                    let image_tensor =
                        crate::inference::media::image_processor::prepare_moondream_image(
                            base64_str,
                            &state.device,
                        )?;
                    state.model.encode_image(&image_tensor)?
                };
                debug!(
                    "Vision: encoded image hash {:x} -> embeddings {:?}",
                    image_hash,
                    image_embeds.shape()
                );
                state.image_embed_cache = Some((image_hash, image_embeds.clone()));
                state.image_embeds = Some(image_embeds);
            }

            Ok(())
        })
        .await?;
        result.map_err(std::convert::Into::into)
    }

    /// Clear any stored image embeddings
    pub async fn clear_images(&self) {
        let mut guard = self.model_state.lock().await;
        if let Some(state) = guard.as_mut() {
            state.image_embeds = None;
        }
    }
}
mod gguf;
pub use gguf::*;
mod decode;
pub use decode::*;
/// The shape of the model currently loaded: what the checkpoint declares about its
/// own geometry, which the planner and the API both ask for.
#[derive(Debug, Clone)]
pub struct ModelGeometry {
    pub name: String,
    pub size: u64,
    pub parameters: u32,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub vocab_size: usize,
    pub context_length: usize,
}

/// Force-release every CUDA device's stream-ordered memory pool. Called
/// after a model unload so the freed blocks return to the OS instead of
/// remaining held by the cudarc allocator's per-device pool. Without
/// this, nvidia-smi continues to report the pool as used and our planner
/// over-pessimistically pushes layers to CPU on the next load.
///
/// Wrapped in a per-process call so we don't pay it per dropped tensor.
#[cfg(not(feature = "cuda"))]
pub fn trim_cuda_pools() {}

/// Warm one card, once per process: a tiny matmul allocates the cuBLAS handle and
/// workspace, which latches the primary context - the deliberate cost that makes every
/// later budget probe of this card read the same world.
#[cfg(feature = "cuda")]
pub(crate) fn warm_card(ordinal: usize) {
    use std::sync::{Mutex, OnceLock};
    static WARMED: OnceLock<Mutex<std::collections::HashSet<usize>>> = OnceLock::new();
    if !WARMED
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(ordinal)
    {
        return;
    }
    let Ok(dev) = crate::tensor::Device::new_cuda(ordinal) else {
        return;
    };
    let a = crate::tensor::Tensor::zeros_on((16usize, 16usize), crate::tensor::DType::F32, &dev);
    if let Ok(a) = a {
        let _ = a.matmul(&a);
    }
    let _ = dev.synchronize();
}

#[cfg(not(feature = "cuda"))]
pub fn release_cuda_pools() {}

#[cfg(feature = "cuda")]
/// MINIMAL pool trim, safe to call at ANY time - including while other models are actively
/// generating on other threads. Returns freed-but-pooled blocks to the driver so NVML probes
/// read TRUE free VRAM. Unlike [`release_cuda_pools`], it does NOT drop the MMVQ/MMQ
/// workspaces, per-model caches or the device-context cache (dropping live workspaces while a
/// decode kernel is in flight produced CUDA_ILLEGAL_ADDRESS at the next sync), does NOT
/// synchronize streams, and SKIPS devices whose pool release-threshold is raised (u64::MAX =
/// CUDA-graph capture mode; zeroing it mid-capture would let the pool return memory a captured
/// graph still addresses).
pub fn trim_cuda_pools() {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys::{
            cuDeviceGetDefaultMemPool, cuMemPoolGetAttribute, cuMemPoolTrimTo, cudaError_enum,
            CUmemPool_attribute, CUmemoryPool,
        };
        let device_count = match nvml_wrapper::Nvml::init() {
            Ok(nvml) => nvml.device_count().unwrap_or(0) as i32,
            Err(_) => 0,
        };
        for ordinal in 0..device_count {
            unsafe {
                let mut pool: CUmemoryPool = std::ptr::null_mut();
                if cuDeviceGetDefaultMemPool(&mut pool, ordinal) != cudaError_enum::CUDA_SUCCESS
                    || pool.is_null()
                {
                    continue;
                }
                let mut threshold: u64 = 0;
                let _ = cuMemPoolGetAttribute(
                    pool,
                    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                    &mut threshold as *mut u64 as *mut std::ffi::c_void,
                );
                if threshold == u64::MAX {
                    continue; // graph-capture mode: leave this device's pool alone
                }
                let _ = cuMemPoolTrimTo(pool, 0);
            }
        }
    }
}

#[cfg(feature = "cuda")]
pub fn release_cuda_pools() {
    // Drop big static workspaces from the upstream fast MMVQ/MMQ paths
    // BEFORE trimming the mempool - otherwise the held CudaSlices keep
    // their backing memory pinned in the pool and trim does nothing
    // visible at the nvidia-smi level. Each quantized-kernel module
    // exposes a `release_workspaces()` for exactly this teardown step.
    crate::tensor::quantized::fast_mmvq::release_workspaces();
    crate::tensor::quantized::release_mmq_workspaces();
    // Z-Image static caches (timestep_freqs + coordinate grids) hold
    // Tensors keeping VRAM alive. Without this clear, repeated load ->
    // gen -> unload cycles leak ~1-5 MB per shape variant. Real users
    // hit this when swapping between image models.
    #[cfg(feature = "image")]
    // Drop the global substrate CudaDevice cache. Each cache entry holds
    // Arc<CudaContext>; the primary context cannot be released (and its
    // pool reservation can't be returned to the OS) while ANY Arc clone
    // is alive. Tensors and workspaces above already dropped their
    // clones - clearing the cache typically removes the last one.
    crate::tensor::cuda_ext::clear_device_cache();
    use cudarc::driver::sys::{
        cuCtxPopCurrent_v2, cuCtxPushCurrent_v2, cuCtxSynchronize, cuDeviceGetDefaultMemPool,
        cuDevicePrimaryCtxRelease_v2, cuDevicePrimaryCtxRetain, cuMemPoolGetAttribute,
        cuMemPoolSetAttribute, cuMemPoolTrimTo, cudaError_enum, CUcontext, CUmemPool_attribute,
        CUmemoryPool,
    };
    // Iterate over physical CUDA devices. cuDeviceGetCount is safer than
    // querying via cudarc::driver::CudaContext (which would create a new
    // context just to count). We use the raw FFI directly.
    let device_count = match nvml_wrapper::Nvml::init() {
        Ok(nvml) => nvml.device_count().unwrap_or(0) as i32,
        Err(_) => 0,
    };
    let mut released = 0u64;
    for ordinal in 0..device_count {
        unsafe {
            // Drain THIS device's streams before trimming its pool. A single
            // current-context cuCtxSynchronize only drains GPU0 - GPU1's
            // stream-ordered frees stayed queued and its pool showed
            // "used 3 MB, reserved 15456 MB, trim CUDA_SUCCESS, nothing
            // released" (a campaign: every gemma4:31b retry then
            // OOM'd against the phantom-full GPU1 -> all-CPU at 1.2 tok/s).
            let mut ctx: CUcontext = std::ptr::null_mut();
            if cuDevicePrimaryCtxRetain(&mut ctx, ordinal) == cudaError_enum::CUDA_SUCCESS
                && !ctx.is_null()
            {
                let _ = cuCtxPushCurrent_v2(ctx);
                let _ = cuCtxSynchronize();
                let mut popped: CUcontext = std::ptr::null_mut();
                let _ = cuCtxPopCurrent_v2(&mut popped);
                let _ = cuDevicePrimaryCtxRelease_v2(ordinal);
            }
            let mut pool: CUmemoryPool = std::ptr::null_mut();
            let r = cuDeviceGetDefaultMemPool(&mut pool, ordinal);
            if r != cudaError_enum::CUDA_SUCCESS || pool.is_null() {
                continue;
            }
            // Read current release threshold for diagnostic.
            let mut threshold_before: u64 = 999;
            let _ = cuMemPoolGetAttribute(
                pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut threshold_before as *mut u64 as *mut std::ffi::c_void,
            );
            // Reset the release threshold to 0. `enable_graph_capture_mode`
            // (called when CUDA graphs are in use) sets this to u64::MAX so
            // the pool never returns memory to the OS - that's correct
            // during capture, but on unload we want VRAM back. Without this
            // reset, cuMemPoolTrimTo(0) is a no-op.
            let zero: u64 = 0;
            let r_set = cuMemPoolSetAttribute(
                pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &zero as *const u64 as *mut std::ffi::c_void,
            );
            tracing::info!(
                "  GPU{}: threshold was {} (set->0 returned {:?})",
                ordinal,
                threshold_before,
                r_set,
            );
            // Diagnostic: dump pool stats so we can tell whether the
            // memory is still in use (live CudaSlice somewhere) vs cached
            // and the trim is failing to release.
            let mut used_before: u64 = 0;
            let mut reserved_before: u64 = 0;
            let _ = cuMemPoolGetAttribute(
                pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                &mut used_before as *mut u64 as *mut std::ffi::c_void,
            );
            let _ = cuMemPoolGetAttribute(
                pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                &mut reserved_before as *mut u64 as *mut std::ffi::c_void,
            );
            // Trim the pool to keep at most 0 bytes - return everything to
            // the OS. CUDA may still retain a small accounting overhead.
            let r = cuMemPoolTrimTo(pool, 0);
            let mut used_after: u64 = 0;
            let mut reserved_after: u64 = 0;
            let _ = cuMemPoolGetAttribute(
                pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT,
                &mut used_after as *mut u64 as *mut std::ffi::c_void,
            );
            let _ = cuMemPoolGetAttribute(
                pool,
                CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                &mut reserved_after as *mut u64 as *mut std::ffi::c_void,
            );
            tracing::info!(
                "  GPU{}: pool used {}->{} MB, reserved {}->{} MB (trim {:?})",
                ordinal,
                used_before / (1024 * 1024),
                used_after / (1024 * 1024),
                reserved_before / (1024 * 1024),
                reserved_after / (1024 * 1024),
                r
            );
            if r == cudaError_enum::CUDA_SUCCESS {
                released += 1;
            }
        }
    }
    if released > 0 {
        tracing::info!("🧹 Trimmed CUDA memory pools on {} device(s)", released);
    }
}

/// Cumulative-decode helper for streaming paths: decodes the FULL token
/// sequence each call. Caller is responsible for the sent_text_len diff
/// arithmetic. Required for Metaspace/SentencePiece tokenizers where
/// per-token decode strips leading spaces.
async fn decode_all(engine: &LlmEngine, tokens: &[u32]) -> String {
    let guard = engine.model_state.lock().await;
    guard
        .as_ref()
        .and_then(|s| s.tokenizer.decode(tokens, true).ok())
        .unwrap_or_default()
}

/// Lazily get or build the llguidance ParserFactory for the loaded model.
/// Cached on `state.grammar_factory` (OnceLock). Builds by serializing the
/// engine's HF tokenizer to JSON and feeding the bytes to
/// `ByteTokenizer::from_json_bytes`, so it works regardless of whether the
/// tokenizer originally came from a tokenizer.json file or was synthesized
/// from GGUF metadata.
fn grammar_factory_for(
    state: &LoadedModelState,
) -> AnyResult<std::sync::Arc<llguidance::ParserFactory>> {
    if let Some(f) = state.grammar_factory.get() {
        return Ok(f.clone());
    }
    let json = state
        .tokenizer
        .to_string(false)
        .map_err(|e| anyhow!("tokenizer.to_string: {e}"))?;
    let bt = toktrie_hf_tokenizers::ByteTokenizer::from_json_bytes(json.as_bytes())
        .map_err(|e| anyhow!("ByteTokenizer::from_json_bytes: {e}"))?;
    let tok_env = bt
        .into_tok_env(Some(state.vocab_size))
        .map_err(|e| anyhow!("into_tok_env: {e}"))?;
    let factory = llguidance::ParserFactory::new(
        &tok_env,
        llguidance::toktrie::InferenceCapabilities::default(),
        &[],
    )
    .map_err(|e| anyhow!("ParserFactory::new: {e}"))?;
    let arc = std::sync::Arc::new(factory);
    let _ = state.grammar_factory.set(arc.clone());
    Ok(arc)
}

/// Parse a grammar spec string into a `TopLevelGrammar`. Accepted shapes:
///   - `"json_object"`              -> bare JSON object
///   - `"json_schema:{<schema>}"`   -> user-supplied JSON schema
///   - `"lark:<grammar text>"`      -> Lark grammar
fn parse_grammar_spec(spec: &str) -> AnyResult<llguidance::api::TopLevelGrammar> {
    let trimmed = spec.trim();
    if trimmed.eq_ignore_ascii_case("json_object") {
        return Ok(llguidance::api::TopLevelGrammar::from_json_schema(
            serde_json::json!({"type": "object"}),
        ));
    }
    if let Some(rest) = trimmed.strip_prefix("json_schema:") {
        let schema: serde_json::Value = serde_json::from_str(rest)
            .map_err(|e| anyhow!("invalid JSON schema in grammar spec: {e}"))?;
        return Ok(llguidance::api::TopLevelGrammar::from_json_schema(schema));
    }
    if let Some(rest) = trimmed.strip_prefix("lark:") {
        return Ok(llguidance::api::TopLevelGrammar::from_lark(
            rest.to_string(),
        ));
    }
    Err(anyhow!(
        "unrecognized grammar spec '{}': expected 'json_object', 'json_schema:...', or 'lark:...'",
        spec
    ))
}

#[cfg(test)]
mod grammar_spec_tests {
    use super::*;

    #[test]
    fn parse_grammar_spec_recognises_json_object_case_insensitively() {
        // Bare "json_object" -> unconstrained JSON-object grammar.
        // Case-insensitive so SDK clients sending "JSON_OBJECT" or
        // "Json_Object" resolve correctly.
        assert!(parse_grammar_spec("json_object").is_ok());
        assert!(parse_grammar_spec("JSON_OBJECT").is_ok());
        assert!(
            parse_grammar_spec("  Json_Object  ").is_ok(),
            "trim before discriminator match"
        );
    }

    #[test]
    fn parse_grammar_spec_accepts_well_formed_json_schema_prefix() {
        // "json_schema:<schema-json>" -> strict schema grammar.
        // The schema itself must be valid JSON.
        let spec = r#"json_schema:{"type":"string"}"#;
        assert!(parse_grammar_spec(spec).is_ok());

        let spec = r#"json_schema:{"type":"object","required":["x"]}"#;
        assert!(parse_grammar_spec(spec).is_ok());
    }

    #[test]
    fn parse_grammar_spec_rejects_malformed_json_schema_payload() {
        // Valid prefix, invalid JSON after the colon - must error
        // with a typed message naming "JSON schema", not silently
        // fall back to json_object (which would relax constraints
        // the caller explicitly asked for).
        let err = parse_grammar_spec("json_schema:not-valid-json").unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("schema"),
            "error should mention 'schema'; got: {err}"
        );
    }

    #[test]
    fn parse_grammar_spec_accepts_lark_grammar_text() {
        // "lark:<grammar>" -> user-supplied Lark grammar. The body
        // can be any string; llguidance validates the grammar shape
        // downstream. parse_grammar_spec just unwraps the prefix.
        let spec = "lark:start: \"hello\"";
        assert!(parse_grammar_spec(spec).is_ok());
        // Even empty body - Lark validates later; the prefix parse
        // must accept any text after the colon.
        assert!(parse_grammar_spec("lark:").is_ok());
    }

    #[test]
    fn parse_grammar_spec_rejects_unknown_discriminator_with_guidance() {
        // Unknown form -> error must name the THREE supported
        // discriminators so the caller can fix their request
        // without consulting the source.
        let err = parse_grammar_spec("regex:^.*$").unwrap_err();
        let msg = err.to_string().to_lowercase();
        assert!(
            msg.contains("json_object"),
            "must mention json_object: {msg}"
        );
        assert!(
            msg.contains("json_schema"),
            "must mention json_schema: {msg}"
        );
        assert!(msg.contains("lark"), "must mention lark: {msg}");
    }
}

/// The request parameters must stay `Send`: a retry around `generate` has to
/// hold them across an await, and the axum handlers that call it are rejected
/// the moment anything in that future is not. Checked here because the property
/// is invisible today - the parameters are consumed immediately, so a field
/// that broke it would only surface as a routing error in an unrelated file.
fn _assert_generation_params_send(p: GenerationParams) {
    fn assert_send<T: Send>(_: T) {}
    assert_send(p);
}
fn _assert_generate_future_send(e: &LlmEngine, prompt: &str, params: GenerationParams) {
    fn assert_send<T: Send>(_: T) {}
    assert_send(e.generate(prompt, params));
}

#[cfg(test)]
mod sampling_tests;
use std::io::Cursor;
use std::time::Instant;

mod discover;
mod embed;
mod generate;
mod loading;
// The pool must be sized from `main`, before anything can build rayon's global one.
pub use loading::configure_thread_pool;
mod query;
mod spec;
mod stream;
