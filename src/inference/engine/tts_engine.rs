//! TTS engine - Parler-TTS synthesis (mini-v1 / large-v1).
//!
//! End-to-end flow:
//!   1. `load_parler("parler-tts/parler-tts-mini-v1")` downloads model +
//!      tokenizer + config via hf-hub (cached after first call).
//!   2. `synthesize(text, voice_desc, opts)` tokenises text + the voice
//!      description, runs the decoder loop to produce audio codes, then
//!      uses the DAC audio encoder to decode codes -> PCM at the
//!      checkpoint's native rate.
//!   3. Result is `f32` mono PCM at `TtsResult.sample_rate` (mini-v1:
//!      44.1 kHz, large-v1: 24 kHz). The HTTP handler wraps that in a
//!      WAV header sized for that rate.
//!
//! Parler-TTS runs in F32 only - the model weights and the DAC vocoder
//! aren't tested under F16. Inference cost is dominated by the decoder
//! sample loop, similar in shape to a small autoregressive LLM.

use crate::inference::sample::token_sampling::LogitsProcessor;
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Tensor};
use anyhow::{anyhow, Context, Result as AnyResult};
// The Parler-TTS model, carried here with the decode path specialised for this stack (
// loken so the vendored fork stays pristine vs upstream).
use crate::inference::model::kyutai::cond::{load_voice as load_kyutai_voice, KyutaiConditioner};
use crate::inference::model::kyutai::depformer::KyutaiDepformer;
use crate::inference::model::kyutai::gen::{generate as kyutai_generate, prepare_entries};
use crate::inference::model::kyutai::lm::KyutaiLm;
use crate::inference::model::kyutai::mimi::KyutaiMimiDecoder;
use crate::inference::model::parler::{Config, Model as ParlerModel};
use crate::inference::model::piper::{PiperVoice, SynthOpts};
use crate::inference::model::pocket_tts::{
    load_wav_mono_24k, resolve_paths, MimiEncoder, PocketTts,
};
use crate::inference::token::sentencepiece::SentencePiece;
use hf_hub::api::sync::ApiBuilder;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::info;

use crate::inference::engine::prompt_cache::{PromptTextCache, TTS_DESCRIPTION_CACHE_CAP};

/// Default sample rate Parler models commonly emit at. Real value comes
/// from `config.audio_encoder.sampling_rate` per checkpoint and lives
/// on `TtsModelState.sample_rate`. Use that for header/format work
/// once the engine is loaded; this constant is for cold-path defaults.
pub const TTS_SAMPLE_RATE: u32 = 44_100;

/// Synthesis-time options. `voice_description` is the free-form prompt
/// Parler uses to control timbre / pitch / pace; `voice` is an
/// OpenAI-style preset name that the HTTP layer maps to a description.
#[derive(Debug, Clone)]
pub struct TtsSynthParams {
    pub voice_description: String,
    /// Top-p nucleus sampling probability; `None` = pure greedy.
    pub top_p: Option<f64>,
    /// Temperature for the logits processor (0.0 = greedy).
    pub temperature: f64,
    pub seed: u64,
    /// Hard cap on autoregressive decode steps. Default 1024 (~ 20 s of
    /// audio). Set higher for long-form synthesis.
    pub max_steps: usize,
}

impl Default for TtsSynthParams {
    fn default() -> Self {
        Self {
            voice_description: "A clear, expressive English speaker in a high-quality recording."
                .to_string(),
            // Match Parler-TTS's official inference recipe
            // (do_sample=True, top_p=0.95, temperature=1.0). Greedy
            // decoding (temperature=0.0) was loop-prone on short
            // inputs - the model gets stuck repeating the final
            // phoneme of the last spoken syllable because the
            // argmax keeps picking the highest-probability "stay
            // on the current sound" token and never triggers the
            // pad-all-codebooks EOS break.
            top_p: Some(0.95),
            temperature: 1.0,
            seed: 0,
            max_steps: 1024,
        }
    }
}

/// Output of a synthesis call: PCM samples + sample rate.
#[derive(Debug, Clone)]
pub struct TtsResult {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
}

/// What a caller wants to know about a load, and how it stops one.
///
/// Both halves are thread-local scopes (`progress::scoped`, `cancel::scoped`) and the
/// reading happens on a pooled blocking thread, so NEITHER can be inherited from the
/// request: published by the caller they would reach nobody. They are carried across the
/// fan-out explicitly, which is the contract both modules document.
///
/// Default is "nobody is watching": a CLI, a test and a parity harness load exactly as
/// they did before, with no reporter published and nothing that can cancel.
#[derive(Default, Clone)]
pub struct LoadWatch {
    /// Counts the weights as they are read. Every safetensors tensor in the process is
    /// read through one function, so this arrives without any TTS backend knowing.
    pub report: Option<crate::inference::serve::progress::SharedProgressFn>,
    /// Stops the read. Without one, a load whose client has gone finishes the checkpoint
    /// for nobody and the next request is told the card is busy on its behalf.
    pub cancel: Option<crate::inference::serve::cancel::CancelToken>,
}

/// Run a blocking TTS load with `watch` published for its duration, and only for it.
///
/// The guard lives in THIS async frame: when the caller's future is dropped - the client
/// left, the stream was abandoned - the frame drops with it, the token fires, and the load
/// bails at the next tensor instead of reading the rest of the checkpoint for nobody.
async fn watched_load<T: Send + 'static>(
    what: &'static str,
    watch: LoadWatch,
    f: impl FnOnce() -> AnyResult<T> + Send + 'static,
) -> AnyResult<T> {
    let token = watch.cancel.unwrap_or_default();
    let guard = crate::inference::serve::cancel::CancelGuard::new(token.clone());
    let report = watch.report;
    let joined = tokio::task::spawn_blocking(move || -> AnyResult<T> {
        let _stop = crate::inference::serve::cancel::scoped::publish(&token);
        let _counts = report.map(crate::inference::serve::progress::scoped::publish);
        f()
    })
    .await;
    guard.disarm();
    joined
        .map_err(|e| anyhow!("{what} load join: {e:#}"))?
        .inspect_err(|e| crate::inference::engine::log_engine_error("tts", "load", e))
}

/// Parler-TTS backend state (HF autoregressive codec model + DAC vocoder).
struct ParlerInner {
    model: ParlerModel,
    tokenizer: Tokenizer,
    /// LRU keyed by voice description. Holds the T5-encoded
    /// representation so repeated synth calls with the same
    /// description skip the ~5 ms encoder pass on mini-v1, and
    /// alternating between a few descriptions (A/B comparing
    /// voices) doesn't pay the encoder cost on every flip.
    /// Cleared by `unload`.
    description_cache: PromptTextCache<Tensor>,
}

/// Kyutai pocket-tts backend: flow-matching TTS conditioned on a catalogue voice.
/// The voice conditioning (`speaker_proj(mimi_encode(ref))`) is computed once at
/// load and reused; output is 24 kHz mono.
struct PocketTtsInner {
    model: PocketTts,
    tokenizer: SentencePiece,
    /// `[T_spk, 1024]` voice conditioning (native tensor).
    voice: crate::tensor::Tensor,
}

/// Kyutai `tts-1.6b-en_fr` backend: Moshi/Delayed-Streams TTS (Helium LM + Depformer +
/// Mimi codec), multilingual en/fr, conditioned on a precomputed tts-voices embedding.
struct KyutaiInner {
    lm: KyutaiLm,
    depformer: KyutaiDepformer,
    cond: KyutaiConditioner,
    mimi: KyutaiMimiDecoder,
    tokenizer: SentencePiece,
    /// `[512, T]` voice conditioning (native tensor).
    voice: crate::tensor::Tensor,
    ndevice: crate::tensor::Device,
    cfg_coef: f32,
}

/// Loaded TTS backend. Parler-TTS is the English HF codec model; Piper is the
/// full-Rust native VITS voice; PocketTts is the flow-matching TTS;
/// Kyutai is the multilingual en/fr Delayed-Streams TTS.
enum Backend {
    Parler(ParlerInner),
    Piper(PiperVoice),
    PocketTts(PocketTtsInner),
    Kyutai(KyutaiInner),
}

pub struct TtsModelState {
    backend: Backend,
    device: Device,
    sample_rate: u32,
}

/// Shared TTS engine - one loaded model at a time. Loads are cached so
/// repeated synthesis calls reuse the same `ParlerModel`.
pub struct TtsEngine {
    model_state: Arc<Mutex<Option<TtsModelState>>>,
    name: Arc<Mutex<Option<String>>>,
    /// GPU bytes the load took, measured. `/api/ps` published 0 here.
    resident_bytes: Arc<Mutex<u64>>,
}

impl TtsEngine {
    pub fn new() -> Self {
        Self {
            model_state: Arc::new(Mutex::new(None)),
            resident_bytes: Arc::new(Mutex::new(0)),
            name: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn is_loaded(&self) -> bool {
        self.model_state.lock().await.is_some()
    }

    /// GPU bytes the resident voice model took, measured across its load.
    pub async fn resident_bytes(&self) -> u64 {
        *self.resident_bytes.lock().await
    }

    pub async fn loaded_name(&self) -> Option<String> {
        self.name.lock().await.clone()
    }

    /// Sample rate of the loaded parler checkpoint, or `None` when no
    /// model is loaded. Used by the HTTP layer to size WAV headers and
    /// `audio/L16` rate parameters correctly per checkpoint.
    pub async fn loaded_sample_rate(&self) -> Option<u32> {
        self.model_state
            .lock()
            .await
            .as_ref()
            .map(|s| s.sample_rate)
    }

    /// Coarse device kind of the loaded TTS model: "CUDA", "CPU", or
    /// "Metal". Used by /api/models/loaded so the GUI Hardware tab
    /// shows the right device badge - earlier code hard-coded "CPU"
    /// because parler-tts was assumed CPU-only, but the loader
    /// actually picks_device() which prefers CUDA when available.
    pub async fn loaded_device(&self) -> Option<String> {
        self.model_state
            .lock()
            .await
            .as_ref()
            .map(|s| match s.device.location() {
                crate::tensor::DeviceLocation::Cuda { .. } => "CUDA".to_string(),
                crate::tensor::DeviceLocation::Cpu => "CPU".to_string(),
            })
    }

    pub async fn unload(&self) {
        let mut g = self.model_state.lock().await;
        if g.is_some() {
            *g = None;
            let mut n = self.name.lock().await;
            let label = n.take().unwrap_or_else(|| "<unknown>".to_string());
            info!("Unloaded TTS model: {label}");
        }
    }

    /// Load a Parler-TTS checkpoint from HF. Defaults to mini-v1 when no
    /// id is supplied; users routinely have ~700 MB free for that
    /// snapshot vs the multi-gigabyte large-v1 weights.
    pub async fn load_parler(&self, name: Option<&str>) -> AnyResult<()> {
        self.load_parler_reporting(name, LoadWatch::default()).await
    }

    /// [`Self::load_parler`], with the weights it reads counted into `watch`.
    pub async fn load_parler_reporting(
        &self,
        name: Option<&str>,
        watch: LoadWatch,
    ) -> AnyResult<()> {
        let free_before_load = crate::inference::place::vram_manager::free_total();
        let model_id = resolve_tts_model_id(name);
        info!("Loading TTS model: {model_id}");
        let model_id_clone = model_id.clone();
        let state =
            watched_load("tts", watch, move || load_parler_blocking(&model_id_clone)).await?;

        let resident =
            free_before_load.saturating_sub(crate::inference::place::vram_manager::free_total());
        let mut guard = self.model_state.lock().await;
        *guard = Some(state);
        *self.name.lock().await = Some(model_id);
        *self.resident_bytes.lock().await = resident;
        Ok(())
    }

    /// Load a native Piper (VITS) voice. `onnx_path` is resolved by the caller
    /// from the configured models dir; `canonical_name` is the name reported by
    /// `loaded_name()` (e.g. `piper/fr_FR-tom-medium`) so reload logic can tell
    /// voices apart.
    pub async fn load_piper(
        &self,
        onnx_path: std::path::PathBuf,
        canonical_name: String,
    ) -> AnyResult<()> {
        self.load_piper_reporting(onnx_path, canonical_name, LoadWatch::default())
            .await
    }

    /// [`Self::load_piper`], with the load watched by `watch`.
    ///
    /// Piper voices are ONNX, and the ONNX reader is not one of the counted loaders, so
    /// this reports the phase and nothing inside it. That is the honest state of it: the
    /// voice files are tens of megabytes and the load is seconds, which is the case a bar
    /// was never for.
    pub async fn load_piper_reporting(
        &self,
        onnx_path: std::path::PathBuf,
        canonical_name: String,
        watch: LoadWatch,
    ) -> AnyResult<()> {
        let free_before_load = crate::inference::place::vram_manager::free_total();
        info!("Loading TTS model: {canonical_name} ({onnx_path:?})");
        let state = watched_load("piper", watch, move || load_piper_blocking(&onnx_path)).await?;

        let resident =
            free_before_load.saturating_sub(crate::inference::place::vram_manager::free_total());
        let mut guard = self.model_state.lock().await;
        *guard = Some(state);
        *self.name.lock().await = Some(canonical_name);
        *self.resident_bytes.lock().await = resident;
        Ok(())
    }

    /// Load Kyutai pocket-tts with a reference voice clip. The checkpoint +
    /// SentencePiece tokenizer resolve via config; `voice_wav` is the target
    /// voice (any sample-rate 16-bit WAV, resampled to 24 kHz).
    pub async fn load_pocket_tts(
        &self,
        voice_wav: std::path::PathBuf,
        canonical_name: String,
    ) -> AnyResult<()> {
        self.load_pocket_tts_reporting(voice_wav, canonical_name, LoadWatch::default())
            .await
    }

    /// [`Self::load_pocket_tts`], with the weights it reads counted into `watch`.
    pub async fn load_pocket_tts_reporting(
        &self,
        voice_wav: std::path::PathBuf,
        canonical_name: String,
        watch: LoadWatch,
    ) -> AnyResult<()> {
        info!("Loading TTS model: {canonical_name} (voice {voice_wav:?})");
        let state = watched_load("pocket-tts", watch, move || {
            load_pocket_tts_blocking(&voice_wav)
        })
        .await?;
        let mut guard = self.model_state.lock().await;
        *guard = Some(state);
        *self.name.lock().await = Some(canonical_name);
        Ok(())
    }

    /// Load Kyutai `tts-1.6b-en_fr` with a precomputed tts-voices embedding.
    /// `voice_name` selects the voice (e.g. `alba-mackenna/a-moment-by`), resolved
    /// against the `kyutai/tts-voices` snapshot; `None` uses the first available.
    pub async fn load_kyutai(
        &self,
        voice_name: Option<String>,
        canonical_name: String,
    ) -> AnyResult<()> {
        self.load_kyutai_reporting(voice_name, canonical_name, LoadWatch::default())
            .await
    }

    /// [`Self::load_kyutai`], with the weights it reads counted into `watch`.
    pub async fn load_kyutai_reporting(
        &self,
        voice_name: Option<String>,
        canonical_name: String,
        watch: LoadWatch,
    ) -> AnyResult<()> {
        info!("Loading TTS model: {canonical_name} (voice {voice_name:?})");
        let state = watched_load("kyutai", watch, move || {
            load_kyutai_blocking(voice_name.as_deref())
        })
        .await?;
        let mut guard = self.model_state.lock().await;
        *guard = Some(state);
        *self.name.lock().await = Some(canonical_name);
        Ok(())
    }

    /// Synthesise `text` and return raw f32 mono PCM at the loaded
    /// checkpoint's native rate (see `TtsResult.sample_rate`).
    pub async fn synthesize(&self, text: String, params: TtsSynthParams) -> AnyResult<TtsResult> {
        if !self.is_loaded().await {
            self.load_parler(None).await?;
        }
        // Bump this engine to most-recently-used so the VRAM pressure protocol reclaims it last.
        crate::inference::place::vram_manager::touch("tts");
        let state = self.model_state.clone();
        // The guard lives in THIS async frame: when the caller's future is dropped
        // (client gone, timeout) the frame drops with it, the token fires, and the
        // decode loop bails at its next step. Nothing else can stop a blocking task.
        let token = crate::inference::serve::cancel::CancelToken::new();
        let guard = crate::inference::serve::cancel::CancelGuard::new(token.clone());
        let out = tokio::task::spawn_blocking(move || -> AnyResult<TtsResult> {
            crate::inference::serve::cancel::scoped::with(&token, || {
                let mut guard = state.blocking_lock();
                let s = guard
                    .as_mut()
                    .ok_or_else(|| anyhow!("tts unloaded mid-synthesize"))?;
                synthesize_blocking(s, &text, &params)
            })
        })
        .await
        .map_err(|e| anyhow!("tts synth join: {e}"))?
        .inspect_err(|e| crate::inference::engine::log_engine_error("tts", "synthesize", e));
        guard.disarm();
        out
    }
}

impl Default for TtsEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------------------------------------
// Internal: HF download + load
// ------------------------------------------------------------

fn load_parler_blocking(model_id: &str) -> AnyResult<TtsModelState> {
    let api = {
        let mut b = ApiBuilder::new();
        if let Ok(tok) = std::env::var("HF_TOKEN") {
            b = b.with_token(Some(tok));
        }
        b.build().context("hf-hub api build")?
    };
    let repo = api.model(model_id.to_string());

    let config_path = repo.get("config.json").context("download config.json")?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("download tokenizer.json")?;
    // mini-v1 ships a single safetensors; large-v1 is sharded.
    let weights_path = match repo.get("model.safetensors") {
        Ok(p) => vec![p],
        Err(_) => {
            // Sharded: index file lists the parts.
            let idx_path = repo
                .get("model.safetensors.index.json")
                .context("download model.safetensors.index.json")?;
            let idx_json: serde_json::Value =
                serde_json::from_reader(std::fs::File::open(&idx_path)?)?;
            let mut parts = std::collections::HashSet::new();
            if let Some(map) = idx_json.get("weight_map").and_then(|m| m.as_object()) {
                for v in map.values() {
                    if let Some(s) = v.as_str() {
                        parts.insert(s.to_string());
                    }
                }
            }
            let mut paths = Vec::with_capacity(parts.len());
            for p in parts {
                paths.push(repo.get(&p).with_context(|| format!("download {p}"))?);
            }
            paths
        }
    };

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("load parler tokenizer.json: {e}"))?;

    // Budget-aware placement (same policy as the other TTS backends): fastest GPU whose real
    // free VRAM fits the F32 weights, else CPU - never a fixed device index.
    let weights_bytes: u64 = weights_path
        .iter()
        .map(|p| std::fs::metadata(p).map(|m| m.len()).unwrap_or(0))
        .sum();
    let ndev = crate::inference::place::plan::place_whole(weights_bytes.max(512 << 20), 1 << 30);
    let device = ndev.clone();
    // Parler-TTS isn't validated under F16 (the audio encoder DAC has
    // strict numerical behaviour). Stay in F32 to avoid drift.
    let dtype = DType::F32;
    info!("TTS device: {device:?}, dtype: {dtype:?}");

    let vb = unsafe { VarBuilder::from_files(&weights_path, dtype, &device)? };
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
        .context("parse parler-tts config.json")?;
    let sample_rate = config.audio_encoder.sampling_rate as u32;
    // parler's T5 runs on the native substrate (F32, matching the model dtype).
    let t5_ndevice = &device.clone();
    let t5_vb = unsafe {
        crate::tensor::VarBuilder::from_files(&weights_path, crate::tensor::DType::F32, t5_ndevice)
    }
    .map_err(|e| anyhow!("parler t5 native weights: {e}"))?;
    let model = ParlerModel::new(&config, vb, t5_vb).context("load parler-tts weights")?;

    info!(
        "TTS loaded: parler-tts ({} Hz, vocab={})",
        sample_rate, config.vocab_size
    );

    Ok(TtsModelState {
        backend: Backend::Parler(ParlerInner {
            model,
            tokenizer,
            description_cache: PromptTextCache::new(TTS_DESCRIPTION_CACHE_CAP),
        }),
        device,
        sample_rate,
    })
}

/// Load a native Piper (VITS) voice from its `.onnx` path (the `.onnx.json`
/// sibling is read for the phoneme map + sample rate). Full-Rust inference on
/// the native tensor stack - no onnxruntime. The voice path is resolved by the
/// HTTP layer from the configured HF models dir (never hardcoded).
fn load_piper_blocking(onnx_path: &std::path::Path) -> AnyResult<TtsModelState> {
    // The HiFi-GAN decoder (the dominant cost) is all groups==1 convs, which
    // hit the cuBLAS im2col GPU path - ~60x faster than the single-threaded CPU
    // conv at audio resolution, bit-equivalent output. Budget-aware placement:
    // fastest GPU whose real free VRAM fits the voice, else CPU.
    let onnx_bytes = std::fs::metadata(onnx_path).map(|m| m.len()).unwrap_or(0);
    let ndevice = crate::inference::place::plan::place_whole(onnx_bytes.max(256 << 20), 512 << 20);
    let device = ndevice.clone();
    let voice = PiperVoice::load(onnx_path, &ndevice)
        .map_err(|e| anyhow!("load piper voice {onnx_path:?}: {e}"))?;
    let sample_rate = voice.sample_rate;
    info!(
        "TTS loaded: native Piper VITS ({} Hz, {:?}) from {onnx_path:?}",
        sample_rate,
        device.location()
    );
    Ok(TtsModelState {
        backend: Backend::Piper(voice),
        device,
        sample_rate,
    })
}

/// Load pocket-tts (model + encoder + tokenizer) and pre-compute the voice
/// conditioning from `voice_wav`. Runs on CPU (the native flow-matching loop is
/// CPU-validated); 24 kHz mono output.
fn load_pocket_tts_blocking(voice_wav: &std::path::Path) -> AnyResult<TtsModelState> {
    // Prefer GPU: the flow-matching synth is dominated by per-frame
    // transformer forwards - on CPU that's the /voice latency bottleneck.
    // Every op threads `device`, so a CUDA device runs the whole pipeline on
    // GPU. Falls back to CPU when CUDA is unavailable. (If a native op lacks a
    // CUDA path this surfaces at load/synth; the CPU path stays the fallback.)
    // Budget-aware placement (same policy as kyutai): fastest GPU that fits
    // the ~236 MB model within its real free VRAM, else CPU.
    let ndevice = crate::inference::place::plan::place_whole(512 << 20, 512 << 20);
    let on_gpu = ndevice.is_cuda();
    let device = ndevice.clone();
    info!(
        "pocket-tts device: {}",
        if on_gpu { "CUDA (plan-chosen)" } else { "CPU" }
    );
    let (ckpt, tok) = resolve_paths();
    let ckpt = ckpt
        .to_str()
        .ok_or_else(|| anyhow!("bad pocket-tts ckpt path"))?;
    if !std::path::Path::new(ckpt).is_file() {
        return Err(anyhow!(
            "pocket-tts checkpoint not found: {ckpt} (set huggingface_models_dir)"
        ));
    }
    if !voice_wav.is_file() {
        return Err(anyhow!(
            "pocket-tts reference voice not found: {voice_wav:?}"
        ));
    }
    let model = PocketTts::from_safetensors(ckpt, ndevice.clone())
        .map_err(|e| anyhow!("load pocket-tts: {e}"))?;
    let encoder = MimiEncoder::from_safetensors(ckpt, ndevice)
        .map_err(|e| anyhow!("load mimi encoder: {e}"))?;
    let tokenizer = SentencePiece::from_file(tok.to_str().unwrap())
        .map_err(|e| anyhow!("load sentencepiece: {e}"))?;
    let mut vsamp = load_wav_mono_24k(voice_wav.to_str().unwrap())
        .map_err(|e| anyhow!("read voice wav: {e}"))?;
    vsamp.truncate(144_000); // up to ~6 s of reference
    let voice = model
        .voice_from_audio(&encoder, &vsamp)
        .map_err(|e| anyhow!("encode voice: {e}"))?;
    info!(
        "TTS loaded: pocket-tts (24 kHz, voice {} samples)",
        vsamp.len()
    );
    Ok(TtsModelState {
        backend: Backend::PocketTts(PocketTtsInner {
            model,
            tokenizer,
            voice,
        }),
        device,
        sample_rate: 24_000,
    })
}

/// Resolve the Kyutai tts-1.6b-en_fr checkpoint files + a voice embedding from the
/// configured HF models dir. Returns (dsm_tts, mimi, spm, voice_safetensors).
fn resolve_kyutai_paths(
    voice_name: Option<&str>,
) -> AnyResult<(
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
    std::path::PathBuf,
)> {
    use crate::inference::cache::hf;
    let snap_dir = |repo: &str| -> AnyResult<std::path::PathBuf> {
        hf::snapshot(repo)
            .ok_or_else(|| anyhow!("no snapshot in {:?} (download {repo})", hf::snapshots(repo)))
    };
    let tts_snap = snap_dir("models--kyutai--tts-1.6b-en_fr")?;
    // The checkpoint is named after its own content hash, so it is found by prefix.
    let dsm = hf::file_starting_with(&tts_snap, "dsm_tts")
        .ok_or_else(|| anyhow!("dsm_tts safetensors not found in {tts_snap:?}"))?;
    let mimi = tts_snap.join("tokenizer-e351c8d8-checkpoint125.safetensors");
    let spm = tts_snap.join("tokenizer_spm_8k_en_fr_audio.model");
    // voice: pick by name (subdir/file prefix) or the first available .safetensors
    let voice_snap = snap_dir("models--kyutai--tts-voices")?;
    let voice = find_kyutai_voice(&voice_snap, voice_name)
        .ok_or_else(|| anyhow!("kyutai voice {voice_name:?} not found under {voice_snap:?}"))?;
    Ok((dsm, mimi, spm, voice))
}

/// Find a voice `.safetensors` under the tts-voices snapshot. `name` may be a full
/// relative path (`alba-mackenna/a-moment-by`) or a substring; `None` picks the first.
fn find_kyutai_voice(snap: &std::path::Path, name: Option<&str>) -> Option<std::path::PathBuf> {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "safetensors") {
                    out.push(p);
                }
            }
        }
    }
    let mut all = Vec::new();
    walk(snap, &mut all);
    all.sort();
    match name {
        None => all.into_iter().next(),
        Some(n) => all
            .into_iter()
            .find(|p| p.to_str().is_some_and(|s| s.contains(n))),
    }
}

fn load_kyutai_blocking(voice_name: Option<&str>) -> AnyResult<TtsModelState> {
    let (dsm, mimi, spm, voice_path) = resolve_kyutai_paths(voice_name)?;
    let dsm = dsm.to_str().ok_or_else(|| anyhow!("bad dsm path"))?;
    // The 1.6 B Helium LM is HeteroPlanned across real free VRAM (GPU->CPU, never OOM) inside
    // from_safetensors_hetero. The small aux towers (depformer / conditioner / mimi / voice)
    // load onto its primary device.
    let lm = KyutaiLm::from_safetensors_hetero(dsm).map_err(|e| anyhow!("load kyutai lm: {e}"))?;
    let ndevice = lm.device().clone();
    let on_gpu = ndevice.is_cuda();
    let depformer = KyutaiDepformer::from_safetensors(dsm, ndevice.clone())
        .map_err(|e| anyhow!("load depformer: {e}"))?;
    let cond = KyutaiConditioner::from_safetensors(dsm, ndevice.clone())
        .map_err(|e| anyhow!("load conditioner: {e}"))?;
    let mimidec = KyutaiMimiDecoder::from_safetensors(mimi.to_str().unwrap(), ndevice.clone())
        .map_err(|e| anyhow!("load mimi: {e}"))?;
    let tokenizer =
        SentencePiece::from_file(spm.to_str().unwrap()).map_err(|e| anyhow!("load spm: {e}"))?;
    let voice = load_kyutai_voice(voice_path.to_str().unwrap(), &ndevice)
        .map_err(|e| anyhow!("load voice: {e}"))?;
    // The streamed decode runs on the LM's primary from the HeteroPlan.
    let device = ndevice.clone();
    info!(
        "TTS loaded: kyutai tts-1.6b-en_fr (24 kHz, {}, voice {voice_path:?})",
        if on_gpu { "hetero" } else { "CPU" }
    );
    Ok(TtsModelState {
        backend: Backend::Kyutai(KyutaiInner {
            lm,
            depformer,
            cond,
            mimi: mimidec,
            tokenizer,
            voice,
            ndevice,
            cfg_coef: 2.0,
        }),
        device,
        sample_rate: 24_000,
    })
}

// ------------------------------------------------------------
// Internal: synthesis pipeline
// ------------------------------------------------------------

fn synthesize_blocking(
    s: &mut TtsModelState,
    text: &str,
    params: &TtsSynthParams,
) -> AnyResult<TtsResult> {
    let sample_rate = s.sample_rate;
    match &mut s.backend {
        Backend::Parler(inner) => synthesize_parler(inner, &s.device, sample_rate, text, params),
        Backend::Piper(voice) => {
            // Piper is deterministic VITS: ignore the Parler voice-description
            // prompt; honour `seed` for reproducible noise. Defaults mirror
            // Piper's own `infer` (noise 0.667, noise_w 0.8, length 1.0).
            let opts = SynthOpts {
                seed: params.seed,
                ..SynthOpts::default()
            };
            let pcm = voice
                .tts(text, opts)
                .map_err(|e| anyhow!("piper synth: {e}"))?;
            Ok(TtsResult { pcm, sample_rate })
        }
        Backend::PocketTts(inner) => {
            // Flow-matching TTS; honour `seed` for reproducible noise.
            let pcm = inner
                .model
                .synthesize_text(
                    &inner.tokenizer,
                    text,
                    Some(&inner.voice),
                    None,
                    params.seed,
                )
                .map_err(|e| anyhow!("pocket-tts synth: {e}"))?;
            Ok(TtsResult { pcm, sample_rate })
        }
        Backend::Kyutai(inner) => {
            // Multilingual Delayed-Streams TTS; temp 0.6 default, `seed` reproducible.
            let entries = prepare_entries(&inner.tokenizer, text);
            let pcm = kyutai_generate(
                &inner.lm,
                &inner.depformer,
                &inner.cond,
                &inner.mimi,
                entries,
                &inner.voice,
                inner.cfg_coef,
                0.6,
                params.seed,
                &inner.ndevice,
            )
            .map_err(|e| anyhow!("kyutai synth: {e}"))?;
            Ok(TtsResult { pcm, sample_rate })
        }
    }
}

fn synthesize_parler(
    s: &mut ParlerInner,
    device: &Device,
    sample_rate: u32,
    text: &str,
    params: &TtsSynthParams,
) -> AnyResult<TtsResult> {
    let device = device.clone();

    let prompt_ids = s
        .tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("tokenize prompt: {e}"))?
        .get_ids()
        .to_vec();
    let prompt = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;

    // Encode the voice description through the T5 encoder, but only
    // when it isn't already in the LRU. Repeated synth calls with one
    // voice hit; back-and-forth between a few alternatives stays warm
    // up to TTS_DESCRIPTION_CACHE_CAP. Saves the ~5 ms encoder pass
    // per call on mini-v1.
    let encoded = if let Some(enc) = s.description_cache.get(params.voice_description.as_str()) {
        enc
    } else {
        let description_ids = s
            .tokenizer
            .encode(params.voice_description.as_str(), true)
            .map_err(|e| anyhow!("tokenize voice description: {e}"))?
            .get_ids()
            .to_vec();
        let description = Tensor::new(description_ids.as_slice(), &device)?.unsqueeze(0)?;
        let enc = s.model.encode_description(&description)?;
        s.description_cache
            .insert(params.voice_description.clone(), enc.clone());
        enc
    };

    let lp = LogitsProcessor::new(params.seed, Some(params.temperature), params.top_p);
    let codes = s
        .model
        .generate_with_encoded(&prompt, &encoded, lp, params.max_steps)?;
    // The decode_codes path expects an [bsz, codebooks, T] tensor; the
    // generate result is [codebooks, T] on CPU.
    let codes = codes.to_dtype(DType::I64)?;
    let codes = codes.unsqueeze(0)?.to_device(&device)?;
    let pcm = s.model.audio_encoder.decode_codes(&codes)?;
    // Squeeze to a 1D mono buffer. pcm shape is [1, 1, T].
    let pcm = pcm
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_device(&Device::Cpu)?;
    let pcm = pcm.to_vec1::<f32>()?;
    Ok(TtsResult { pcm, sample_rate })
}

/// Resolve a caller-supplied TTS model name into a full HF model id.
///
/// Three input shapes accepted:
///
///   - `None` -> default to `parler-tts/parler-tts-mini-v1`
///     (works in ~700 MB free; large-v1 is multi-gigabyte).
///   - OpenAI documented ids (case-insensitive) -> mapped to the
///     closest local-hostable Parler checkpoint. SDK clients
///     passing `TTS-1` or `tts-1-HD` resolve correctly.
///   - Bare `parler-tts-...` name (no slash) -> prefixed with the
///     `parler-tts/` HF org. Anything else with a `/` is passed
///     through as a full HF model id.
///
/// Lifted out of `load_parler` so the mapping can be unit-tested
/// without spinning up an async runtime or hitting HF.
pub(crate) fn resolve_tts_model_id(name: Option<&str>) -> String {
    let raw = name.unwrap_or("parler-tts/parler-tts-mini-v1");
    let lower = raw.to_lowercase();
    let mapped = match lower.as_str() {
        "tts-1" | "tts-1-1106" => "parler-tts-mini-v1",
        "tts-1-hd" | "tts-1-hd-1106" => "parler-tts-large-v1",
        _ => raw,
    };
    if mapped.contains('/') {
        mapped.to_string()
    } else if mapped.to_lowercase().starts_with("parler") {
        format!("parler-tts/{}", mapped.to_lowercase())
    } else {
        mapped.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_tts_model_id_defaults_to_mini_v1() {
        // Most callers omit `model` - the default must resolve to
        // mini-v1 so the first-load experience fits in ~700 MB free.
        assert_eq!(resolve_tts_model_id(None), "parler-tts/parler-tts-mini-v1",);
    }

    #[test]
    fn resolve_tts_model_id_maps_openai_aliases_case_insensitively() {
        // OpenAI's documented TTS model ids -> local checkpoints.
        assert_eq!(
            resolve_tts_model_id(Some("tts-1")),
            "parler-tts/parler-tts-mini-v1",
        );
        assert_eq!(
            resolve_tts_model_id(Some("TTS-1")),
            "parler-tts/parler-tts-mini-v1",
        );
        assert_eq!(
            resolve_tts_model_id(Some("tts-1-1106")),
            "parler-tts/parler-tts-mini-v1",
        );
        assert_eq!(
            resolve_tts_model_id(Some("tts-1-hd")),
            "parler-tts/parler-tts-large-v1",
        );
        assert_eq!(
            resolve_tts_model_id(Some("TTS-1-HD")),
            "parler-tts/parler-tts-large-v1",
        );
        assert_eq!(
            resolve_tts_model_id(Some("tts-1-hd-1106")),
            "parler-tts/parler-tts-large-v1",
        );
    }

    #[test]
    fn resolve_tts_model_id_auto_prefixes_bare_parler_names() {
        // Bare `parler-tts-foo` -> `parler-tts/parler-tts-foo`.
        assert_eq!(
            resolve_tts_model_id(Some("parler-tts-mini-v1")),
            "parler-tts/parler-tts-mini-v1",
        );
        assert_eq!(
            resolve_tts_model_id(Some("Parler-TTS-Large-V1")),
            "parler-tts/parler-tts-large-v1",
            "case-insensitive prefix match; output lowercased",
        );
    }

    #[test]
    fn resolve_tts_model_id_passes_full_hf_paths_through() {
        // Custom forks or non-Parler vendors. Full slash-form ids are
        // threaded through verbatim - no lowercasing (HF model ids
        // are case-sensitive on disk).
        assert_eq!(
            resolve_tts_model_id(Some("ylacombe/parler-tts-mini-jenny-30H")),
            "ylacombe/parler-tts-mini-jenny-30H",
        );
        assert_eq!(resolve_tts_model_id(Some("OpenSci/MyTTS")), "OpenSci/MyTTS",);
    }

    #[test]
    fn resolve_tts_model_id_unknown_bare_name_passes_through() {
        // Bare name that doesn't start with `parler` and has no slash:
        // pass through unchanged (will fail at HF resolve, surfaced
        // as a proper 404 to the caller).
        assert_eq!(resolve_tts_model_id(Some("unknown-model")), "unknown-model");
    }

    #[test]
    fn tts_synth_params_default_matches_parler_recipe() {
        // Pin the documented Parler-TTS inference recipe
        // (do_sample=True, top_p=0.95, temperature=1.0).
        //
        // Greedy decoding (temperature=0.0) was loop-prone on short
        // inputs - the model got stuck repeating the final phoneme
        // of the last syllable (user-reported: "Hello !" looped on
        // the trailing "o"). Any future patch that flips back to
        // greedy or drops top_p must update this test deliberately,
        // not by accident.
        let p = TtsSynthParams::default();
        assert!(
            (p.temperature - 1.0).abs() < f64::EPSILON,
            "temperature must default to 1.0 (sampling), got {}",
            p.temperature
        );
        assert_eq!(
            p.top_p,
            Some(0.95),
            "top_p must default to 0.95 (Parler recipe)"
        );
        assert!(
            p.max_steps >= 1024,
            "max_steps default {} too small for ~20s clips",
            p.max_steps
        );
        assert!(
            p.voice_description.to_lowercase().contains("english"),
            "default voice_description should hint at English to avoid \
             multilingual codebook drift, got: {:?}",
            p.voice_description
        );
    }

    /// A safetensors bundle with one f32 per name, so a load has something real to read.
    fn write_bundle(path: &std::path::Path, names: &[&str]) {
        let data: Vec<[u8; 4]> = (0..names.len())
            .map(|i| (i as f32 + 1.0).to_le_bytes())
            .collect();
        let tensors: Vec<_> = names
            .iter()
            .zip(&data)
            .map(|(n, d)| {
                (
                    (*n).to_string(),
                    safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![1], d)
                        .unwrap(),
                )
            })
            .collect();
        safetensors::serialize_to_file(tensors, None, path).unwrap();
    }

    /// The reporter and the token are THREAD-LOCAL scopes and the reading happens on a
    /// pooled blocking thread, so neither can be inherited: published by the request they
    /// would reach nobody, and a streaming client would be back to watching a load in
    /// silence. This pins the carry across the fan-out - the one link in the chain that can
    /// break without any compiler noticing.
    #[tokio::test]
    async fn a_watched_load_reports_the_weights_it_reads() {
        use crate::inference::serve::progress::{phase, SharedProgressFn};
        use crate::tensor::safetensors_io::SafeTensorsLoader;

        let dir = std::env::temp_dir().join(format!("tts_watch_report_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.safetensors");
        let names = ["a", "b", "c", "d"];
        write_bundle(&path, &names);

        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen.clone();
        let report: SharedProgressFn = std::sync::Arc::new(move |p: &str, d: usize, t: usize| {
            sink.lock().unwrap().push((p.to_string(), d, t));
        });
        let read_path = path.clone();
        watched_load(
            "test",
            LoadWatch {
                report: Some(report),
                cancel: None,
            },
            move || {
                let loader = unsafe { SafeTensorsLoader::multi(&[&read_path]) }
                    .map_err(|e| anyhow!("{e}"))?;
                for n in names {
                    loader.load(n).map_err(|e| anyhow!("{e}"))?;
                }
                Ok(())
            },
        )
        .await
        .expect("a nominal load succeeds");

        let got = seen.lock().unwrap().clone();
        assert_eq!(
            got,
            vec![
                (phase::LOAD_MODEL.to_string(), 1, 4),
                (phase::LOAD_MODEL.to_string(), 2, 4),
                (phase::LOAD_MODEL.to_string(), 3, 4),
                (phase::LOAD_MODEL.to_string(), 4, 4),
            ],
            "the weights a TTS backend reads must be counted into the watch it was given"
        );
        // And the scopes are torn down: these threads are handed to the next request, and a
        // reporter left behind would send its counts into a channel that belongs to nobody.
        assert!(crate::inference::serve::progress::scoped::current().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A client that leaves during a load must not leave the checkpoint arriving for
    /// nobody - the card would then be claimed on behalf of a request that has gone, and
    /// the next one told it was busy.
    #[tokio::test]
    async fn a_watched_load_stops_when_its_token_fires() {
        use crate::inference::serve::cancel::CancelToken;
        use crate::tensor::safetensors_io::SafeTensorsLoader;

        let dir = std::env::temp_dir().join(format!("tts_watch_cancel_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("weights.safetensors");
        let names = ["a", "b", "c", "d"];
        write_bundle(&path, &names);

        let token = CancelToken::new();
        token.cancel(); // the stream was dropped before the load got a thread
        let read_path = path.clone();
        let err = watched_load(
            "test",
            LoadWatch {
                report: None,
                cancel: Some(token),
            },
            move || {
                let loader = unsafe { SafeTensorsLoader::multi(&[&read_path]) }
                    .map_err(|e| anyhow!("{e}"))?;
                for n in names {
                    loader.load(n).map_err(|e| anyhow!("{e}"))?;
                }
                Ok(())
            },
        )
        .await
        .expect_err("an abandoned load must refuse")
        .to_string();
        assert!(err.contains("cancelled"), "an error that says why: {err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
