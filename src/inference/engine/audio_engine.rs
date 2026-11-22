//! Audio engine - Whisper transcription.
//!
//! End-to-end flow:
//!   1. `load_whisper("openai/whisper-small")` downloads weights via hf-hub
//!      (cached after first call), loads model + tokenizer + mel filters.
//!   2. `transcribe(samples, sr)` takes f32 mono PCM, resamples conceptually
//!      (we accept 16 kHz only here; callers convert), runs the encoder
//!      followed by a greedy decoder loop, returns the final text.
//!
//! The decode is greedy and keeps 1 language per clip.
//!
//! A clip longer than 1 encoder window is read as the reference transcription reads it.
//! Each window decodes with timestamps.
//! The next window starts where the last segment the model left unfinished starts.
//! A window that does not reach the end of the clip ends inside a silence of the clip.
//!
//! A request for word times gets them from the cross attention of the alignment heads the
//! checkpoint names, and from its timestamp tokens when it names none.

use crate::inference::model::whisper::{self as m, align, audio, silence, Config};
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Tensor};
use anyhow::{anyhow, Context, Result as AnyResult};
use byteorder::{ByteOrder, LittleEndian};
// Whisper encoder/decoder on the NATIVE tensor substrate (inference/model/whisper/model.rs);
// audio front-end / Config / constants are shared from inference/model/whisper/mod.rs.
use crate::inference::model::whisper::model::Whisper as WhisperF32;
use hf_hub::api::sync::ApiBuilder;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

/// A word of a transcript and the seconds it spans in the recording.
pub use crate::inference::model::whisper::align::TimedWord as WhisperWord;

/// Sample rate whisper was trained on. Audio coming in at any other rate
/// must be resampled before `transcribe()`.
pub const WHISPER_SAMPLE_RATE: u32 = m::SAMPLE_RATE as u32;

/// Embedded mel filter banks (80- and 128-bin variants).
/// the reference whisper::audio::pcm_to_mel needs these as F32 weights.
const MEL_FILTERS_80: &[u8] = include_bytes!("../../../assets/whisper/melfilters80.bytes");
const MEL_FILTERS_128: &[u8] = include_bytes!("../../../assets/whisper/melfilters128.bytes");

/// Task the decoder should perform. Whisper exposes the choice via two
/// special prompt tokens: <|transcribe|> (same-language output) and
/// <|translate|> (English output regardless of source).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WhisperTask {
    #[default]
    Transcribe,
    /// Translate to English. OpenAI exposes this as `/v1/audio/translations`.
    Translate,
}

/// Parameters for an ASR (transcription) request - subset of OpenAI's
/// `/v1/audio/transcriptions` field set.
#[derive(Debug, Clone, Default)]
pub struct AudioTranscribeParams {
    /// BCP-47 language code (e.g. "en", "fr"). If `None`, language is
    /// auto-detected from the audio (multilingual model only).
    pub language: Option<String>,
    pub temperature: f32,
    pub task: WhisperTask,
    /// When true, drop `<|notimestamps|>` from the decoder prompt so the
    /// model emits timestamp tokens (`<|0.00|>`, `<|0.02|>`, ...). The
    /// transcribe pipeline then parses consecutive timestamp pairs into
    /// real segment boundaries for verbose_json / srt / vtt output.
    pub timestamps: bool,
    /// Collect per-token logprobs (for `avg_logprob`) and the iter-0
    /// no_speech_prob even when `timestamps` is false. The handler
    /// sets this true for verbose_json so the placeholder segment
    /// carries real confidence numbers; text/json formats leave it
    /// false to skip the per-token softmax + d2h round-trip
    /// (~7 ms / chunk on whisper-small).
    pub collect_metrics: bool,
    /// Return the time of each word.
    /// Word times come with segment timestamps.
    pub words: bool,
    /// Optional context prompt - text the model will see before the
    /// SOT prompt as `<|startofprev|>` + tokens. Whisper's documented
    /// way to bias vocabulary / continue a previous segment. Caller
    /// is responsible for length (we truncate at 220 tokens).
    pub initial_prompt: Option<String>,
    /// Set by the request that can be cancelled: the transcription stops at its next window
    /// and its next decoded token.
    pub cancel: Option<crate::inference::serve::cancel::CancelToken>,
}

/// One whisper segment - a contiguous span of audio with start/end
/// timestamps and the text decoded inside it. Maps to one item in
/// OpenAI's verbose_json `segments` array.
#[derive(Debug, Clone)]
pub struct WhisperSegment {
    pub start: f32,
    pub end: f32,
    pub text: String,
    pub tokens: Vec<u32>,
    /// Length of the text after zlib compression / original length.
    /// Whisper's paper treats values > 2.4 as a hallucination signal
    /// (the model emitted repetitive output the compressor crushes).
    pub compression_ratio: f32,
    /// Sampling temperature actually applied (mirrors the request's
    /// `temperature` field; greedy decode reports 0.0).
    pub temperature: f32,
    /// Mean log-probability of the sampled tokens in this segment.
    /// Negative values closer to 0 mean high model confidence.
    pub avg_logprob: f32,
    /// Probability the audio segment is silent / non-speech, computed
    /// from the `<|nospeech|>` token's softmax probability after the
    /// chunk's prompt prefix. Whisper's threshold for treating audio
    /// as silent is typically 0.6.
    pub no_speech_prob: f32,
}

/// Output of a transcription/translation request. Carries enough metadata
/// for the OpenAI-compatible `verbose_json` response format. When
/// `params.timestamps == true`, `segments` carries real per-utterance
/// boundaries; otherwise it falls back to a single segment spanning the
/// full audio duration (so SDKs expecting a `segments` array still work).
#[derive(Debug, Clone)]
pub struct TranscribeResult {
    pub text: String,
    /// BCP-47 language code that was actually used. For an English-only
    /// checkpoint this is always "en"; for multilingual it reflects the
    /// caller's `language` field, or the auto-detected code when omitted.
    pub language: String,
    /// Duration of the input audio in seconds.
    pub duration_s: f32,
    /// Raw output token IDs (post-prompt, pre-EOT). Some downstream
    /// clients consume these directly; the OpenAI verbose_json format
    /// passes them through.
    pub tokens: Vec<u32>,
    /// Real per-utterance segments when timestamps were requested;
    /// a single full-duration placeholder otherwise.
    pub segments: Vec<WhisperSegment>,
    /// The words and their times when the request asked for them, in order.
    pub words: Option<Vec<WhisperWord>>,
}

/// The control tokens a decode is steered with: what opens a transcript, which of the two
/// tasks it is, whether it carries timestamps, and what closes it.
///
/// They are resolved together, held together and read together, and a decode is impossible
/// without every one of them - so a tokenizer missing one fails at load, where the answer
/// is "this checkpoint cannot be served", rather than part-way through a transcript.
struct ControlTokens {
    sot: u32,
    transcribe: u32,
    translate: u32,
    no_timestamps: u32,
    eot: u32,
    /// `<|nospeech|>` on v2/v3, `<|nocaptions|>` on v1: what the silence probability is
    /// read off. None when the tokenizer exposes neither, which costs the caller that
    /// number and nothing else.
    no_speech: Option<u32>,
    /// `<|startofprev|>`, which opens the caller's context prompt. None on the rare
    /// tokenizer without it; both v2 and v3 have it.
    sot_prev: Option<u32>,
}

impl ControlTokens {
    fn resolve(tokenizer: &Tokenizer) -> AnyResult<Self> {
        Ok(Self {
            sot: token_id(tokenizer, m::SOT_TOKEN)?,
            transcribe: token_id(tokenizer, m::TRANSCRIBE_TOKEN)?,
            translate: token_id(tokenizer, m::TRANSLATE_TOKEN)?,
            no_timestamps: token_id(tokenizer, m::NO_TIMESTAMPS_TOKEN)?,
            eot: token_id(tokenizer, m::EOT_TOKEN)?,
            // The optional two: absent means the feature they carry is unavailable on this
            // checkpoint, not that the checkpoint cannot be served.
            no_speech: m::NO_SPEECH_TOKENS
                .iter()
                .find_map(|t| tokenizer.token_to_id(t)),
            sot_prev: tokenizer.token_to_id("<|startofprev|>"),
        })
    }
}

/// How this checkpoint's vocabulary constrains a decode.
///
/// The suppression mask and the language table are read together, and the second answers a
/// question that used to be stored beside it: English-only IS "there are no language
/// tokens to choose between". Held as two fields they can disagree, and the one that
/// decides whether to pick a language is then not the one that has the ids to pick from.
struct Vocabulary {
    /// `[vocab]` additive mask (0 or -inf), applied HOST-side to the downloaded
    /// per-token logits.
    suppress: Vec<f32>,
    /// Language code -> its token id. None on an English-only checkpoint.
    languages: Option<std::collections::HashMap<String, u32>>,
}

impl Vocabulary {
    /// True when the checkpoint is English-only (whisper-small.en, etc.).
    fn english_only(&self) -> bool {
        self.languages.is_none()
    }
}

pub struct WhisperModelState {
    model: WhisperF32,
    tokenizer: Tokenizer,
    config: Config,
    mel_filters: Vec<f32>,
    device: Device,
    control: ControlTokens,
    vocab: Vocabulary,
    /// One encoder pass's activation peak over one audio window: what each further window
    /// in the same pass adds.
    window_bytes: u64,
    /// The `(layer, head)` pairs of the decoder whose cross attention times the words.
    /// Empty when the checkpoint names none.
    alignment_heads: Vec<(usize, usize)>,
}

pub struct AudioModelInfo {
    pub name: String,
    pub model_type: String,
    /// GPU bytes the load actually took - see `ImageModelInfo::resident_bytes` for
    /// why this is measured rather than derived. `/api/ps` published 0 here, which a
    /// dashboard reads as "this model costs nothing".
    pub resident_bytes: u64,
    /// The formats the projections of the model are held in, joined with `+`.
    pub running_format: Option<String>,
}

pub struct AudioEngine {
    model_state: Arc<Mutex<Option<WhisperModelState>>>,
    name: Arc<Mutex<Option<String>>>,
    /// "CUDA" or "CPU", kept apart from the state so a listing never waits on a transcription.
    device_kind: Arc<Mutex<Option<String>>>,
    /// The formats the projections of the resident model are held in, joined with `+`.
    running_format: Arc<Mutex<Option<String>>>,
    resident_bytes: Arc<Mutex<u64>>,
    /// What one more audio window takes beside the others in an encoder pass, for the
    /// resident model.
    window_bytes: Arc<Mutex<u64>>,
    /// Transcriptions queued behind the one running on a model run as the next group:
    /// their audio windows go through the encoder as one batch.
    gather: TranscribeGather,
    /// Whether the fidelity level lets a card hold the projections in half precision.
    narrows: bool,
}

/// Transcriptions gathered by the model they run on; each job is the PCM and its
/// parameters.
type TranscribeGather = crate::inference::serve::gather::Gather<
    String,
    (Vec<f32>, AudioTranscribeParams),
    AnyResult<TranscribeResult>,
>;

impl AudioEngine {
    /// An engine whose loads follow the fidelity level `fidelity`.
    pub fn new(fidelity: crate::inference::engine::llm_engine::Fidelity) -> Self {
        Self {
            model_state: Arc::new(Mutex::new(None)),
            name: Arc::new(Mutex::new(None)),
            device_kind: Arc::new(Mutex::new(None)),
            running_format: Arc::new(Mutex::new(None)),
            resident_bytes: Arc::new(Mutex::new(0)),
            window_bytes: Arc::new(Mutex::new(0)),
            gather: TranscribeGather::default(),
            narrows: fidelity.narrows(),
        }
    }

    /// Whether a model is resident. A state under a lock is a model being loaded or at
    /// work, which counts as present.
    pub async fn is_loaded(&self) -> bool {
        match self.model_state.try_lock() {
            Ok(guard) => guard.is_some(),
            Err(_) => true,
        }
    }

    pub async fn get_loaded_model_info(&self) -> Option<AudioModelInfo> {
        let name = self.name.lock().await.clone()?;
        Some(AudioModelInfo {
            name,
            model_type: "whisper".to_string(),
            resident_bytes: *self.resident_bytes.lock().await,
            running_format: self.running_format.lock().await.clone(),
        })
    }

    /// Coarse device kind of the loaded whisper model: "CUDA" or "CPU".
    /// Used by /api/models/loaded so the GUI Hardware tab shows the
    /// right device badge, read from where the load actually placed it.
    pub async fn loaded_device(&self) -> Option<String> {
        self.device_kind.lock().await.clone()
    }

    /// Currently-loaded HF repo id, or `None` when no model is loaded.
    pub async fn loaded_name(&self) -> Option<String> {
        self.name.lock().await.clone()
    }

    /// The two towers of the resident model and where they sit; none when nothing is
    /// loaded or the model is at work under its lock.
    pub fn parts(&self) -> Option<crate::inference::serve::progress::placement::Parts> {
        use crate::inference::serve::progress::placement::whole;
        let guard = self.model_state.try_lock().ok()?;
        let state = guard.as_ref()?;
        Some(vec![
            (
                "encoder".to_string(),
                whole(&state.device, state.model.encoder.n_blocks()),
            ),
            (
                "decoder".to_string(),
                whole(&state.device, state.model.decoder.n_blocks()),
            ),
        ])
    }

    pub async fn unload(&self) {
        let mut guard = self.model_state.lock().await;
        if guard.take().is_some() {
            let mut name_guard = self.name.lock().await;
            let name = name_guard.take().unwrap_or_else(|| "<unknown>".to_string());
            *self.device_kind.lock().await = None;
            *self.running_format.lock().await = None;
            info!("Unloaded audio model: {name}");
        }
    }

    /// Load a whisper checkpoint from HF (`openai/whisper-small` if name is
    /// unspecified). Caches model state for the lifetime of this engine.
    pub async fn load_whisper(&self, name: Option<&str>) -> AnyResult<()> {
        let model_id = resolve_whisper_model_id(name);
        info!("Loading whisper model: {model_id}");
        let free_before_load = crate::inference::place::vram_manager::free_total();

        // hf-hub is sync. Do the I/O on a blocking thread so we don't stall
        // the tokio runtime.
        let model_id_clone = model_id.clone();
        let narrows = self.narrows;
        let (state, running_format) = tokio::task::spawn_blocking(move || {
            let held = crate::tensor::layer::held::collect();
            load_whisper_blocking(&model_id_clone, narrows).map(|state| (state, held.formats()))
        })
        .await
        .map_err(|e| anyhow!("whisper load join: {e}"))??;
        crate::inference::engine::tts_engine::log_running_format(
            "whisper",
            running_format.as_deref(),
        );

        let free_after_load = crate::inference::place::vram_manager::free_total();
        let kind = if state.device.is_cuda() {
            "CUDA"
        } else {
            "CPU"
        };
        *self.window_bytes.lock().await = state.window_bytes;
        let mut guard = self.model_state.lock().await;
        *guard = Some(state);
        *self.name.lock().await = Some(model_id);
        *self.device_kind.lock().await = Some(kind.to_string());
        *self.running_format.lock().await = running_format;
        *self.resident_bytes.lock().await = free_before_load.saturating_sub(free_after_load);
        Ok(())
    }

    /// Greedy transcription of 16-kHz mono f32 PCM. Returns the decoded text.
    pub async fn transcribe(
        &self,
        samples: Vec<f32>,
        sample_rate: u32,
        params: AudioTranscribeParams,
    ) -> AnyResult<TranscribeResult> {
        if sample_rate != m::SAMPLE_RATE as u32 {
            anyhow::bail!(
                "Whisper requires {} Hz PCM input; got {sample_rate} Hz. Resample before calling.",
                m::SAMPLE_RATE
            );
        }
        if !self.is_loaded().await {
            self.load_whisper(None).await?;
        }
        // A transcription starts at once when none runs on this model. The ones that
        // arrive meanwhile queue behind it and run as the next group, sharing the encoder:
        // their audio windows go through it as one batch, as many at once as the roomiest
        // card takes.
        let model = self.loaded_name().await.unwrap_or_default();
        // The rows a group holds are counted when the group is formed, once the one ahead has
        // ended and its pooled memory is back with the driver.
        let window_bytes = *self.window_bytes.lock().await;
        let cap = move || {
            crate::inference::engine::llm_engine::trim_cuda_pools();
            crate::inference::serve::gather::rows_that_fit(window_bytes, usize::MAX)
        };
        let state = self.model_state.clone();
        self.gather
            .submit(model, (samples, params), cap, move |jobs| async move {
                let count = jobs.len();
                tokio::task::spawn_blocking(move || -> Vec<AnyResult<TranscribeResult>> {
                    let mut guard = state.blocking_lock();
                    match guard.as_mut() {
                        Some(s) => transcribe_group_blocking(s, &jobs),
                        None => jobs
                            .iter()
                            .map(|_| Err(anyhow!("whisper unloaded mid-transcribe")))
                            .collect(),
                    }
                })
                .await
                .unwrap_or_else(|e| {
                    (0..count)
                        .map(|_| Err(anyhow!("transcribe join: {e}")))
                        .collect()
                })
            })
            .await
            .unwrap_or_else(|| Err(anyhow!("the batch this transcription joined was lost")))
            // Centralised engine-error logging (see engine::log_engine_error)
            // - covers both the synchronous Result path (which DOES surface
            // via 5xx but TraceLayer may not always elevate it) and ensures
            // consistent formatting across modalities.
            .inspect_err(|e| crate::inference::engine::log_engine_error("audio", "transcribe", e))
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new(crate::inference::engine::llm_engine::Fidelity::default())
    }
}

// ------------------------------------------------------------
// Internal: HF download + load
// ------------------------------------------------------------

/// `narrows` says whether the fidelity level lets a card hold the projections in half
/// precision.
fn load_whisper_blocking(model_id: &str, narrows: bool) -> AnyResult<WhisperModelState> {
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
    let weights_path = repo
        .get("model.safetensors")
        .context("download model.safetensors")?;

    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_path)?)
        .context("parse whisper config.json")?;
    let alignment_heads = alignment_heads_of(&repo, &config);
    let tokenizer =
        Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow!("load tokenizer.json: {e}"))?;

    // Mel filter table size depends on num_mel_bins (80 for most v2/v3 models,
    // 128 for v3-large variants).
    let mel_bytes: &[u8] = match config.num_mel_bins {
        80 => MEL_FILTERS_80,
        128 => MEL_FILTERS_128,
        n => anyhow::bail!("unsupported num_mel_bins: {n}"),
    };
    let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
    LittleEndian::read_f32_into(mel_bytes, &mut mel_filters);

    // Placed through the ladder in `place`: the weights as the card holds them (projections in
    // half precision, the rest F32, the position table on the host), and one encoder pass over
    // the full audio window beside them.
    let (weights_size, _) =
        crate::inference::place::runtime_demand::safetensors_card_split(&[&weights_path], |name| {
            !name.ends_with("embed_positions.weight")
        })
        .map_err(|e| anyhow!("size whisper weights: {e}"))?;
    let encoder_ffn =
        unsafe { crate::tensor::safetensors_io::SafeTensorsLoader::multi(&[&weights_path]) }
            .ok()
            .and_then(|st| st.shape_of("model.encoder.layers.0.fc1.weight"))
            .and_then(|d| d.first().copied())
            .ok_or_else(|| anyhow!("whisper: no encoder feed-forward in the checkpoint"))?;
    let reserve = crate::inference::place::runtime_demand::transformer_work_reserve(
        weights_size,
        config.max_source_positions,
        config.d_model,
        config.encoder_attention_heads,
        encoder_ffn,
    );
    let window_bytes = crate::inference::place::runtime_demand::dit_activation_bytes(
        config.max_source_positions,
        config.d_model,
        config.encoder_attention_heads,
        encoder_ffn as f64 / config.d_model.max(1) as f64,
    );
    let (device, model) =
        crate::inference::place::ladder::load_whole("Whisper", weights_size, reserve, |device| {
            let vb = unsafe { VarBuilder::from_files(&[&weights_path], DType::F32, device)? };
            let model = WhisperF32::load(&vb, config.clone(), narrows)?;
            Ok::<_, crate::tensor::Error>((device.clone(), model))
        })
        .map_err(|e| anyhow!("load whisper weights: {e}"))?;
    info!("Whisper device: {device:?}");

    let control = ControlTokens::resolve(&tokenizer)?;

    // O(vocab) build with O(1) suppress lookups via HashSet - was O(vocab x suppress_len)
    // which is ~50x cheaper for the typical 50-entry suppress list.
    let suppress_set: std::collections::HashSet<u32> = config
        .suppress_tokens
        .iter()
        .copied()
        .chain(std::iter::once(control.no_timestamps))
        .collect();
    let suppress_tokens: Vec<f32> = (0..config.vocab_size as u32)
        .map(|i| {
            if suppress_set.contains(&i) {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();

    // English-only checkpoints end in `.en` and have a smaller vocab/no
    // language tokens. Detect both ways.
    let english_only = model_id.ends_with(".en") || tokenizer.token_to_id("<|en|>").is_none();
    let lang_tokens = if english_only {
        None
    } else {
        let mut m = std::collections::HashMap::new();
        for lang in WHISPER_LANGS {
            let tok = format!("<|{lang}|>");
            if let Some(id) = tokenizer.token_to_id(&tok) {
                m.insert((*lang).to_string(), id);
            }
        }
        Some(m)
    };

    info!(
        "Whisper loaded: {} mel bins, {} vocab, english_only={}",
        config.num_mel_bins, config.vocab_size, english_only
    );

    Ok(WhisperModelState {
        model,
        tokenizer,
        config,
        mel_filters,
        device,
        control,
        vocab: Vocabulary {
            suppress: suppress_tokens,
            languages: lang_tokens,
        },
        window_bytes,
        alignment_heads,
    })
}

/// The `(layer, head)` pairs the checkpoint's generation config names as alignment heads.
///
/// A pair outside the decoder is dropped.
/// A checkpoint without the file or the list has none.
fn alignment_heads_of(repo: &hf_hub::api::sync::ApiRepo, config: &Config) -> Vec<(usize, usize)> {
    let generation = repo
        .get("generation_config.json")
        .ok()
        .and_then(|path| std::fs::read_to_string(path).ok())
        .and_then(|text| serde_json::from_str::<m::GenerationConfig>(&text).ok())
        .unwrap_or_default();
    let heads: Vec<(usize, usize)> = generation
        .alignment_heads
        .iter()
        .map(|&[layer, head]| (layer, head))
        .filter(|&(layer, head)| {
            layer < config.decoder_layers && head < config.decoder_attention_heads
        })
        .collect();
    info!("Whisper alignment heads: {}", heads.len());
    heads
}

/// The Hugging Face id a whisper load fetches: the caller's id, which the API has already
/// resolved from any alias, or the default checkpoint when the caller names none.
pub(crate) fn resolve_whisper_model_id(name: Option<&str>) -> String {
    name.unwrap_or(DEFAULT_WHISPER).to_string()
}

/// The checkpoint a transcription that names no model loads.
const DEFAULT_WHISPER: &str = "openai/whisper-small";

fn token_id(t: &Tokenizer, name: &str) -> AnyResult<u32> {
    t.token_to_id(name)
        .ok_or_else(|| anyhow!("whisper tokenizer missing {name}"))
}

// ------------------------------------------------------------
// Internal: transcription pipeline
// ------------------------------------------------------------

/// Several transcriptions on the resident model, one answer per job in job order.
///
/// The window of every clip that fits in 1 window runs through the encoder with the
/// others, in passes of as many as the roomiest card takes.
/// Each clip is then decoded on its own window's features, as it would be alone.
///
/// A longer clip brings no window to the shared passes.
/// It encodes each of its windows when its decode reaches the window's start.
fn transcribe_group_blocking(
    s: &mut WhisperModelState,
    jobs: &[(Vec<f32>, AudioTranscribeParams)],
) -> Vec<AnyResult<TranscribeResult>> {
    let cut: Vec<AnyResult<MelWindows>> = jobs
        .iter()
        .map(|(pcm, params)| {
            // A cancelled job skips its spectrum; the others of the group go on.
            if let Some(cancel) = &params.cancel {
                cancel.bail().map_err(|e| anyhow!("{e}"))?;
            }
            mel_windows(s, pcm)
        })
        .collect();
    let all: Vec<&Tensor> = cut.iter().flatten().flat_map(|w| w.mels.iter()).collect();
    let per_pass = crate::inference::serve::gather::rows_that_fit(s.window_bytes, usize::MAX);
    let features = encode_in_passes(&all, per_pass, |pass| encode_pass(s, pass));
    answer_each(
        jobs,
        cut,
        features,
        |w| w.mels.len(),
        |(pcm, params), w, own| transcribe_windows(s, pcm.len(), w, own, params),
    )
}

/// One answer per job, in job order, from the windows every job was cut into and one
/// feature answer per window of all of them, in the same order.
///
/// A job whose cut failed brought no window and is answered that error. Each other job
/// takes as many feature answers as it has windows, whatever they hold, so no job reads
/// another's. A job with a failed window is answered that failure.
fn answer_each<J, W, F, R>(
    jobs: &[J],
    cut: Vec<AnyResult<W>>,
    features: Vec<AnyResult<F>>,
    count: impl Fn(&W) -> usize,
    mut decode: impl FnMut(&J, &W, &[F]) -> AnyResult<R>,
) -> Vec<AnyResult<R>> {
    let mut features = features.into_iter();
    jobs.iter()
        .zip(cut)
        .map(|(job, w)| {
            let w = w?;
            let own: Vec<AnyResult<F>> = features.by_ref().take(count(&w)).collect();
            let own = own.into_iter().collect::<AnyResult<Vec<F>>>()?;
            decode(job, &w, &own)
        })
        .collect()
}

/// `encode` over `windows` in passes of at most `per_pass` windows: one answer per window,
/// in window order. A pass of several windows that fails is encoded again one window at a
/// time, so a failure of the batch never fails a window that runs alone.
fn encode_in_passes<T, F>(
    windows: &[T],
    per_pass: usize,
    mut encode: impl FnMut(&[T]) -> AnyResult<Vec<F>>,
) -> Vec<AnyResult<F>> {
    let mut features = Vec::with_capacity(windows.len());
    for pass in windows.chunks(per_pass.max(1)) {
        match encode(pass) {
            Err(e) if pass.len() > 1 => {
                warn!(
                    "whisper encoder: a pass of {} windows failed ({e:#}), encoding each alone",
                    pass.len()
                );
                for window in pass {
                    let one = std::slice::from_ref(window);
                    features.extend(pass_features(one.len(), encode(one)));
                }
            }
            answer => features.extend(pass_features(pass.len(), answer)),
        }
    }
    features
}

/// One answer per window of a pass of `n` windows: the pass's features, or its error for
/// each of them. An answer of another length than `n` counts as an error, so a later
/// window never takes an earlier window's feature.
fn pass_features<F>(n: usize, answer: AnyResult<Vec<F>>) -> Vec<AnyResult<F>> {
    let msg = match answer {
        Ok(encoded) if encoded.len() == n => return encoded.into_iter().map(Ok).collect(),
        Ok(encoded) => format!(
            "encoder answered {} features for {n} windows",
            encoded.len()
        ),
        Err(e) => format!("{e:#}"),
    };
    (0..n).map(|_| Err(anyhow!("{msg}"))).collect()
}

/// One clip's mel spectrogram, as the encoder reads it.
struct MelWindows {
    /// The window of a clip that fits in 1 window: `[1, n_mels, N_FRAMES]` on the host.
    /// Empty for a longer clip and for a clip without audio.
    mels: Vec<Tensor>,
    /// The mel frame each window of `mels` starts at.
    starts: Vec<usize>,
    /// The whole spectrogram of a clip longer than 1 window, read by its seek.
    long: Option<LongMel>,
    /// Wall time of the spectrogram, in milliseconds.
    mel_ms: f64,
}

/// The spectrogram of a clip longer than 1 encoder window.
struct LongMel {
    /// `[1, n_mels, frames]` on the host, the audio's frames followed by padding.
    mel: Tensor,
    /// The frames that hold audio.
    content: usize,
    /// The level of each frame that holds audio.
    levels: Vec<f32>,
    /// The level a frame is silent below, from the clip's own levels.
    silence: Option<f32>,
    /// The value of a frame of padding, which a window holds after the frames it keeps.
    quiet: f32,
}

/// The frames of `samples` that hold audio: a frame per hop, the last one partial.
fn content_frames(samples: usize) -> usize {
    samples.div_ceil(m::HOP_LENGTH)
}

/// The encoder window of `window` frames of `mel` (`[1, n_mels, frames]`) that starts at mel
/// frame `seek` and keeps `size` frames of it.
/// The frames after those it keeps hold `fill`.
fn window_at(
    mel: &Tensor,
    seek: usize,
    size: usize,
    window: usize,
    fill: f32,
) -> AnyResult<Tensor> {
    let (_, n_mels, n_frames) = mel.shape().dims3()?;
    let size = size.min(window).min(n_frames.saturating_sub(seek));
    let segment = mel.narrow(2, seek, size)?;
    Ok(if size < window {
        let pad = Tensor::from_vec_f32(
            vec![fill; n_mels * (window - size)],
            vec![1, n_mels, window - size],
        )?;
        Tensor::cat(&[&segment, &pad], 2)?
    } else {
        segment.contiguous()?
    })
}

/// Cut a clip into encoder windows.
/// Empty audio has none.
///
/// A clip whose audio fits in 1 window gives that window, which starts at frame 0.
/// A longer clip gives its whole spectrogram, and its decode picks each window's start.
fn mel_windows(s: &WhisperModelState, samples: &[f32]) -> AnyResult<MelWindows> {
    let mut windows = MelWindows {
        mels: Vec::new(),
        starts: Vec::new(),
        long: None,
        mel_ms: 0.0,
    };
    if samples.is_empty() {
        return Ok(windows);
    }
    let t_mel = std::time::Instant::now();
    let mel = audio::pcm_to_mel(&s.config, samples, &s.mel_filters);
    windows.mel_ms = t_mel.elapsed().as_secs_f64() * 1e3;
    let n_mels = s.config.num_mel_bins;
    let n_frames = mel.len() / n_mels;
    if n_frames == 0 {
        return Ok(windows);
    }
    let content = content_frames(samples.len());
    let window = s.config.window_frames();
    // The spectrogram carries padding after its audio, which every band holds at 1 value.
    let quiet = mel.get(content.min(n_frames - 1)).copied().unwrap_or(0.0);
    let levels = match content > window {
        true => silence::frame_levels(&mel, n_mels, n_frames, content),
        false => Vec::new(),
    };
    let mel = Tensor::from_vec_f32(mel, vec![1, n_mels, n_frames])?;
    if content > window {
        windows.long = Some(LongMel {
            mel,
            content,
            silence: silence::silence_level(&levels),
            levels,
            quiet,
        });
    } else {
        windows
            .mels
            .push(window_at(&mel, 0, n_frames, window, quiet)?);
        windows.starts.push(0);
    }
    Ok(windows)
}

/// Where the decode of a long clip goes on after 1 window.
#[derive(Debug, PartialEq, Eq)]
struct SeekStep {
    /// The tokens of the window's output that are kept: its complete segments.
    kept: usize,
    /// The mel frames from the window's start to the next window's start.
    advance: usize,
}

/// The next window of a long clip after a window of `segment_frames` audio frames whose
/// output is `body`.
///
/// A timestamp token is at least `timestamp_begin`.
/// Each timestamp step covers `frames_per_timestamp` mel frames.
///
/// 2 timestamps in a row close a segment and open the next one.
/// An output that ends inside a segment keeps its segments up to the last 2 timestamps in a row.
/// The next window then starts at the first of those 2, where the unfinished segment starts.
///
/// An output that ends on a closing timestamp keeps every token and moves by the whole window.
/// An output without 2 timestamps in a row does the same.
/// A last segment that closes at the window's start does the same, so the decode always moves.
fn seek_step(
    body: &[u32],
    timestamp_begin: u32,
    segment_frames: usize,
    frames_per_timestamp: usize,
) -> SeekStep {
    let whole = SeekStep {
        kept: body.len(),
        advance: segment_frames,
    };
    let is_timestamp = |t: u32| t >= timestamp_begin;
    let ends_closed = body.len() >= 2
        && !is_timestamp(body[body.len() - 2])
        && is_timestamp(body[body.len() - 1]);
    let last_pair = (1..body.len())
        .rev()
        .find(|&i| is_timestamp(body[i - 1]) && is_timestamp(body[i]));
    match last_pair {
        Some(i) if !ends_closed => {
            // A timestamp past the frames the window holds never skips audio after them.
            let advance = ((body[i - 1] - timestamp_begin) as usize * frames_per_timestamp)
                .min(segment_frames);
            match advance {
                0 => whole,
                advance => SeekStep { kept: i, advance },
            }
        }
        _ => whole,
    }
}

/// The rules of the model's reference decoding on timestamp tokens, applied to `logits`, the
/// next token's logits after the tokens `sampled` since the prompt.
///
/// The output opens with a timestamp.
/// A timestamp after text is followed by a second timestamp or by `eot`.
/// 2 timestamps in a row are followed by text.
/// A timestamp is never earlier than the last one.
/// A timestamp is taken whenever all timestamps together are likelier than the likeliest text
/// token.
fn apply_timestamp_rules(logits: &mut [f32], sampled: &[u32], timestamp_begin: u32, eot: u32) {
    let tb = (timestamp_begin as usize).min(logits.len());
    let is_timestamp = |t: u32| t >= timestamp_begin;
    let last = sampled.last().copied().filter(|&t| is_timestamp(t));
    let penultimate_is_timestamp = sampled.len() < 2 || is_timestamp(sampled[sampled.len() - 2]);
    if last.is_some() {
        match penultimate_is_timestamp {
            true => logits[tb..].fill(f32::NEG_INFINITY),
            false => logits[..(eot as usize).min(tb)].fill(f32::NEG_INFINITY),
        }
    }
    if let Some(&latest) = sampled.iter().rev().find(|&&t| is_timestamp(t)) {
        let floor = match last.is_some() && !penultimate_is_timestamp {
            true => latest,
            false => latest + 1,
        };
        let floor = (floor as usize).min(logits.len());
        logits[tb..floor.max(tb)].fill(f32::NEG_INFINITY);
    }
    if sampled.is_empty() {
        logits[..tb].fill(f32::NEG_INFINITY);
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let timestamps = logits[tb..].iter().map(|&l| (l - max).exp()).sum::<f32>();
    let best_text = logits[..tb]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if timestamps > 0.0 && timestamps.ln() + max > best_text {
        logits[..tb].fill(f32::NEG_INFINITY);
    }
}

/// One encoder pass over `pass` windows stacked along the batch axis: one
/// `[1, frames/2, n_state]` feature tensor per window, in order, brought back to the host
/// so the card holds one pass at a time however long the clips are.
fn encode_pass(s: &mut WhisperModelState, pass: &[&Tensor]) -> AnyResult<Vec<Tensor>> {
    let mel = match pass {
        [one] => (*one).clone(),
        _ => Tensor::cat(pass, 0)?,
    }
    .to_device(&s.device)?;
    let t_enc = std::time::Instant::now();
    let encoded = s.model.encoder.forward(&mel, true)?;
    s.device.synchronize()?;
    tracing::debug!(
        "whisper encoder: {} window(s) in {:.1} ms",
        pass.len(),
        t_enc.elapsed().as_secs_f64() * 1e3
    );
    let encoded = encoded.to_device(&Device::Cpu)?;
    if pass.len() == 1 {
        return Ok(vec![encoded]);
    }
    (0..pass.len())
        .map(|row| Ok(encoded.narrow(0, row, 1)?.contiguous()?))
        .collect()
}

/// One clip's transcript from its windows and their encoded features.
fn transcribe_windows(
    s: &mut WhisperModelState,
    n_samples: usize,
    windows: &MelWindows,
    features: &[Tensor],
    params: &AudioTranscribeParams,
) -> AnyResult<TranscribeResult> {
    let duration_s = n_samples as f32 / m::SAMPLE_RATE as f32;
    let mel_ms = windows.mel_ms;
    if windows.starts.is_empty() && windows.long.is_none() {
        return Ok(TranscribeResult {
            text: String::new(),
            language: params.language.clone().unwrap_or_else(|| "en".to_string()),
            duration_s,
            tokens: Vec::new(),
            segments: Vec::new(),
            words: params.words.then(Vec::new),
        });
    }

    // Whisper assumes a single language per file. Cache the detected
    // code after chunk 0 and pin params.language for the rest of the
    // file - skips a redundant per-chunk SOT decoder pass plus the
    // tokenizer language-id lookups.
    let mut all_text = String::new();
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut all_segments: Vec<WhisperSegment> = Vec::new();
    let mut all_words: Vec<WhisperWord> = Vec::new();
    let mut detected_lang: Option<String> = None;
    // Token-weighted logprob aggregate for the placeholder segment.
    let mut total_lp_sum: f64 = 0.0;
    let mut total_lp_count: usize = 0;
    let mut first_chunk_no_speech: Option<f32> = None;
    let mut effective = params.clone();
    // A clip of 1 window decodes the features of the shared pass.
    // A longer clip encodes the window at its seek, decodes it, and moves the seek to the
    // start of the last segment the window left unfinished.
    // Its window ends inside a silence when it does not reach the end of the clip.
    let window = s.config.window_frames();
    let frame_seconds = s.config.frame_seconds();
    let mut shared = windows.starts.iter().zip(features);
    let mut seek = 0usize;
    loop {
        if let Some(cancel) = &params.cancel {
            cancel.bail().map_err(|e| anyhow!("{e}"))?;
        }
        let (start, audio_features, segment_frames) = match &windows.long {
            Some(long) if seek < long.content => {
                let kept = silence::window_length(&long.levels, long.silence, seek, window);
                let mel = window_at(&long.mel, seek, kept, window, long.quiet)?;
                let encoded = encode_pass(s, &[&mel])?.remove(0);
                (seek, encoded, Some(kept))
            }
            Some(_) => break,
            None => match shared.next() {
                Some((start, features)) => (*start, features.clone(), None),
                None => break,
            },
        };
        let content = segment_frames.unwrap_or(content_frames(n_samples).min(window));
        let chunk_offset_s = start as f32 * frame_seconds;
        let audio_features = audio_features.to_device(&s.device)?;
        let dec = decode_segment(
            s,
            &audio_features,
            &effective,
            chunk_offset_s,
            content,
            segment_frames,
        )?;
        tracing::debug!(
            "whisper chunk at {chunk_offset_s:.1} s: mel {mel_ms:.1} ms, \
             prompt {:.1} ms, {} steps {:.1} ms",
            dec.timing.prompt_ms,
            dec.timing.steps,
            dec.timing.steps_ms,
        );
        seek = start + dec.advance;
        if detected_lang.is_none() {
            detected_lang = Some(dec.lang_code.clone());
            if effective.language.is_none() {
                effective.language = Some(dec.lang_code);
            }
        }
        if !dec.text.is_empty() {
            if !all_text.is_empty() && !all_text.ends_with(' ') {
                all_text.push(' ');
            }
            all_text.push_str(dec.text.trim());
        }
        all_tokens.extend_from_slice(&dec.body);
        all_segments.extend(dec.segments);
        let window_end = (chunk_offset_s + content as f32 * frame_seconds).min(duration_s);
        align::stitch_words(&mut all_words, dec.words, chunk_offset_s, window_end);
        total_lp_sum += dec.logprob_sum;
        total_lp_count += dec.logprob_count;
        if first_chunk_no_speech.is_none() && dec.chunk_no_speech_prob > 0.0 {
            first_chunk_no_speech = Some(dec.chunk_no_speech_prob);
        }
    }
    let language = detected_lang.unwrap_or_else(|| "en".to_string());
    let placeholder_avg_logprob = if total_lp_count > 0 {
        (total_lp_sum / total_lp_count as f64) as f32
    } else {
        0.0
    };
    let placeholder_no_speech_prob = first_chunk_no_speech.unwrap_or(0.0);
    // Clamp segment ends to the actual file duration.
    // An open segment ends where its window's audio ends.
    for seg in all_segments.iter_mut() {
        if seg.end > duration_s {
            seg.end = duration_s;
        }
        if seg.start > seg.end {
            seg.start = seg.end;
        }
    }
    // If timestamps weren't requested, emit a single placeholder segment
    // spanning the full audio so SDKs that key on `segments[0]` still
    // work. Real per-utterance segments are populated above when
    // `params.timestamps == true`.
    if all_segments.is_empty() {
        let compression_ratio = zlib_compression_ratio(&all_text);
        all_segments.push(WhisperSegment {
            start: 0.0,
            end: duration_s,
            text: all_text.clone(),
            tokens: all_tokens.clone(),
            compression_ratio,
            temperature: params.temperature.max(0.0),
            // Real chunk-aggregated metrics when collect_metrics is on,
            // 0.0 otherwise (text/json formats don't surface them).
            avg_logprob: placeholder_avg_logprob,
            no_speech_prob: placeholder_no_speech_prob,
        });
    }
    Ok(TranscribeResult {
        text: all_text,
        language,
        duration_s,
        tokens: all_tokens,
        segments: all_segments,
        words: params.words.then_some(all_words),
    })
}

/// One chunk's decode output. `segments` is non-empty only when
/// `params.timestamps` is true and the model actually emitted timestamp
/// tokens. `logprob_sum` + `logprob_count` let the caller compute a
/// **token-weighted** mean across chunks instead of averaging
/// chunk-level averages (which would over-count short chunks).
/// `chunk_no_speech_prob` is from the iter-0 pos-0 softmax.
struct ChunkDecode {
    text: String,
    body: Vec<u32>,
    lang_code: String,
    segments: Vec<WhisperSegment>,
    logprob_sum: f64,
    logprob_count: usize,
    chunk_no_speech_prob: f32,
    /// The mel frames from this window's start to the next window's start.
    advance: usize,
    /// The words of the kept segments, timed from the window's start, when the request asked
    /// for them.
    words: Vec<WhisperWord>,
    timing: ChunkTiming,
}

/// Wall time of one chunk's decode stages, in milliseconds: the language and prompt passes
/// up to the first sampled token, and the decode steps after it. The encoder is timed per
/// pass, where several windows share it.
#[derive(Default)]
struct ChunkTiming {
    prompt_ms: f64,
    steps: usize,
    steps_ms: f64,
}

/// Decode one window from its encoded `audio_features`.
///
/// `content` is the audio frames the window holds.
/// `segment_frames` is set for a window of a clip longer than 1 window, to `content`.
/// Such a window decodes with timestamps under the reference rules.
/// Its output keeps its complete segments, and `advance` says where the next window starts.
fn decode_segment(
    s: &mut WhisperModelState,
    audio_features: &Tensor,
    params: &AudioTranscribeParams,
    chunk_offset_s: f32,
    content: usize,
    segment_frames: Option<usize>,
) -> AnyResult<ChunkDecode> {
    let timestamps = params.timestamps || params.words || segment_frames.is_some();
    let timestamp_begin = s.control.no_timestamps + 1;
    let mut timing = ChunkTiming::default();
    let t_prompt = std::time::Instant::now();
    // New audio: the cross attentions project these features on their first call and keep
    // them for the language pass and the whole decode.
    s.model.decoder.reset_kv_cache();

    // Build the prompt tokens: [<sot>, <lang?>, <transcribe|translate>, <no_timestamps>]
    let (lang_token, lang_code) = if s.vocab.english_only() {
        (None, "en".to_string())
    } else if let Some(ref raw) = params.language {
        // Accept BCP-47 codes ("en") or common English language names
        // ("english", "French") - normalize before lookup so SDKs that
        // surface human names work without a 400.
        let code = normalize_language_code(raw.trim());
        // The tokenizer is keyed by the BARE language, so a region tag has to come off
        // before the lookup. Without this, `fr-FR` - which the endpoint documents as an
        // accepted BCP-47 code - missed, fell back to English, and transcribed French
        // audio as English with a 200 and a response claiming `"language": "en"`.
        // Confidently wrong output, and the only trace was a server-side warning.
        let bare = code.split('-').next().unwrap_or(&code).to_string();
        let id = s
            .vocab
            .languages
            .as_ref()
            .and_then(|m| m.get(bare.as_str()).copied())
            .or_else(|| {
                warn!(
                    "whisper language '{raw}' (-> '{bare}') is not one this model knows; \
                       falling back to <|en|>"
                );
                s.vocab
                    .languages
                    .as_ref()
                    .and_then(|m| m.get("en").copied())
            });
        let code = bare;
        let resolved = if id.is_some() && lang_id_to_code(s, id.unwrap()).is_some() {
            code
        } else {
            "en".to_string()
        };
        (id, resolved)
    } else {
        // Auto-detect language from the audio (multilingual model only).
        // Re-use the encoder output we just computed instead of running
        // another full encoder pass on the same mel segment.
        match detect_language_from_features(s, audio_features) {
            Ok(id) => {
                let code = lang_id_to_code(s, id).unwrap_or_else(|| "en".to_string());
                (Some(id), code)
            }
            Err(e) => {
                warn!("language detect failed: {e}; falling back to <|en|>");
                let id = s
                    .vocab
                    .languages
                    .as_ref()
                    .and_then(|m| m.get("en").copied());
                (id, "en".to_string())
            }
        }
    };
    let task_token = match params.task {
        WhisperTask::Transcribe => s.control.transcribe,
        WhisperTask::Translate => s.control.translate,
    };
    let mut tokens: Vec<u32> = Vec::new();
    // Optional context prompt: `<|startofprev|>` + tokenize(initial_prompt).
    // Whisper's documented hook for biasing vocabulary or continuing a
    // previous segment. Truncated to keep total prompt under the
    // decoder's context budget.
    if let (Some(prev_id), Some(prev_text)) = (s.control.sot_prev, params.initial_prompt.as_ref()) {
        let trimmed = prev_text.trim();
        if !trimmed.is_empty() {
            // Whisper takes up to ~220 prev-context tokens; rough proxy
            // is 4 chars/token, so cap input at 1200 chars to skip
            // pointless tokenization on huge prompts. Final hard cap
            // happens on the token slice below.
            let bounded = if trimmed.chars().count() > 1200 {
                trimmed
                    .chars()
                    .rev()
                    .take(1200)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect::<String>()
            } else {
                trimmed.to_string()
            };
            match s.tokenizer.encode(bounded.as_str(), false) {
                Ok(enc) => {
                    let ids = enc.get_ids();
                    // Cap at 220 prev-context tokens so the total prompt
                    // (prev + sot + lang + task + nots ≈ 224) leaves room
                    // for output within the decoder's max_target_positions.
                    let take = ids.len().min(220);
                    tokens.push(prev_id);
                    tokens.extend_from_slice(&ids[ids.len() - take..]);
                }
                Err(e) => warn!("initial_prompt encode failed, ignoring: {e}"),
            }
        }
    }
    tokens.push(s.control.sot);
    if let Some(lt) = lang_token {
        tokens.push(lt);
    }
    tokens.push(task_token);
    if !timestamps {
        tokens.push(s.control.no_timestamps);
    }
    // Track the prompt length so we know where to start stripping after
    // decode. With timestamps, the prompt is 1 token shorter.
    let prompt_len = tokens.len();

    // Greedy decoder loop. Cap output at the model's full context minus
    // the prompt prefix (and a small safety margin). Whisper-small's
    // max_target_positions is 448; the previous /2 = 224 cap silently
    // truncated dense 30-s utterances that emit ~3-5 tokens/sec.
    let sample_len = s.config.max_target_positions.saturating_sub(prompt_len + 2);
    let temperature = params.temperature.max(0.0) as f64;
    let mut rng = StdRng;

    // Per-step logprob of the sampled token, parallel to body tokens
    // (post-prompt). Used to compute per-segment avg_logprob for the
    // OpenAI verbose_json response.
    let mut logprobs: Vec<f32> = Vec::with_capacity(sample_len);
    // `no_speech_prob` for this chunk - populated from the iter-0
    // softmax over the post-prompt logits. Stays 0.0 when the
    // tokenizer doesn't expose `<|nospeech|>`.
    let mut no_speech_prob: f32 = 0.0;

    // First iteration: feed the entire prompt prefix into the self-attention
    // caches; the cross attentions keep what the first pass over these
    // features projected. Subsequent iterations feed just the most recently
    // sampled token - the cached K/V handle the prefix.
    s.model.decoder.restart_sequence();
    for i in 0..sample_len {
        if let Some(cancel) = &params.cancel {
            cancel.bail().map_err(|e| anyhow!("{e}"))?;
        }
        let step_tokens: &[u32] = if i == 0 {
            tokens.as_slice()
        } else {
            // Just the last token; the rest is already in the KV cache.
            &tokens[tokens.len() - 1..]
        };
        let t_step = std::time::Instant::now();
        // The ids stay on the host, where the decoder looks them up.
        let toks_t = Tensor::from_vec_u32(step_tokens.to_vec(), vec![1, step_tokens.len()])?;
        let ys = s.model.decoder.forward(&toks_t, audio_features, false)?;
        let (_, seq_len, _) = ys.shape().dims3()?;
        // On iter 0 with metrics collection + a `<|nospeech|>` token,
        // also project position 0 (right after SOT) for no_speech_prob.
        // Sampling always uses the last position. Logits land on HOST  -
        // suppress-mask, softmax and argmax all run on the Vec<f32>.
        let collect = params.timestamps || params.collect_metrics;
        let want_no_speech_at_i0 = i == 0 && collect && s.control.no_speech.is_some();
        if want_no_speech_at_i0 {
            // Raw logits: `<|nospeech|>` is normally in suppress_tokens.
            let pos0 = s
                .model
                .decoder
                .final_linear(&ys.narrow(1, 0, 1)?)?
                .to_vec_f32();
            let ns = s.control.no_speech.unwrap() as usize;
            no_speech_prob = host_softmax(&pos0).get(ns).copied().unwrap_or(0.0);
        }
        let last_h = if seq_len == 1 {
            ys.clone()
        } else {
            ys.narrow(1, seq_len - 1, 1)?
        };
        let mut logits = s.model.decoder.final_linear(&last_h)?.to_vec_f32();

        for (l, m) in logits.iter_mut().zip(&s.vocab.suppress) {
            *l += m;
        }
        if segment_frames.is_some() {
            let sampled = &tokens[prompt_len..];
            apply_timestamp_rules(&mut logits, sampled, timestamp_begin, s.control.eot);
        }

        let next_token = if temperature > 0.0 {
            let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature as f32).collect();
            rng.weighted_sample(&host_softmax(&scaled))
        } else {
            host_argmax(&logits)
        };
        // Log-probability of the sampled token. Only collected when
        // timestamps mode is on OR the caller flagged collect_metrics  -
        // text/json formats skip it.
        if collect {
            logprobs.push(host_log_softmax_at(&logits, next_token as usize));
        }
        if i == 0 {
            timing.prompt_ms = t_prompt.elapsed().as_secs_f64() * 1e3;
        } else {
            timing.steps += 1;
            timing.steps_ms += t_step.elapsed().as_secs_f64() * 1e3;
        }
        tokens.push(next_token);
        if next_token == s.control.eot || tokens.len() >= s.config.max_target_positions {
            break;
        }
    }

    // Strip the prompt prefix (and trailing EOT) before any further
    // processing. `prompt_len` was captured above before sampling.
    let mut body: Vec<u32> = tokens
        .into_iter()
        .skip(prompt_len)
        .take_while(|&t| t != s.control.eot)
        .collect();
    // A window of a long clip keeps its complete segments.
    // The unfinished one is decoded again from the next window, which starts where it starts.
    let advance = match segment_frames {
        Some(frames) => {
            let step = seek_step(&body, timestamp_begin, frames, m::FRAMES_PER_POSITION);
            body.truncate(step.kept);
            step.advance
        }
        None => s.config.window_frames(),
    };

    // Parse segments when timestamps were requested AND the model
    // actually emitted timestamp tokens (`token >= timestamp_begin`).
    // Each pair of consecutive timestamp tokens bounds a segment whose
    // text is the tokens between them.
    // `logprobs` has one entry per appended token (parallel to body
    // plus a trailing EOT if the loop closed on it). Truncate to body
    // length so per-segment averaging uses only the text/timestamp
    // tokens we exposed.
    if logprobs.len() > body.len() {
        logprobs.truncate(body.len());
    }
    let temperature_f32 = params.temperature.max(0.0);
    let window_s = content as f32 * s.config.frame_seconds();
    let segments = if params.timestamps || params.words {
        parse_timestamp_segments(
            s,
            &body,
            &logprobs,
            timestamp_begin,
            (chunk_offset_s, window_s),
            temperature_f32,
            no_speech_prob,
        )?
    } else {
        Vec::new()
    };
    let words = match params.words {
        true => {
            let sot_sequence: Vec<u32> = std::iter::once(s.control.sot)
                .chain(lang_token)
                .chain(std::iter::once(task_token))
                .collect();
            let spans: Vec<(f32, f32, Vec<u32>)> = match segments.is_empty() {
                true => vec![(0.0, window_s, body.clone())],
                false => segments
                    .iter()
                    .map(|seg| {
                        let start = seg.start - chunk_offset_s;
                        (start, seg.end - chunk_offset_s, seg.tokens.clone())
                    })
                    .collect(),
            };
            window_words(s, audio_features, &sot_sequence, &spans, content)?
        }
        false => Vec::new(),
    };

    // Full-segment text. When timestamps are on, build it by joining
    // per-segment texts (which were already decoded by the parser) so
    // we don't run the tokenizer a second time on the same body.
    // Without timestamps, decode the body directly with timestamp
    // tokens filtered out.
    let text = if !segments.is_empty() {
        let mut s = String::new();
        for seg in &segments {
            if !s.is_empty() && !s.ends_with(' ') {
                s.push(' ');
            }
            s.push_str(seg.text.trim());
        }
        s
    } else {
        let text_tokens: Vec<u32> = body
            .iter()
            .copied()
            .filter(|&t| t < timestamp_begin)
            .collect();
        s.tokenizer
            .decode(&text_tokens, true)
            .map_err(|e| anyhow!("tokenizer decode: {e}"))?
    };
    // Sum + count let `transcribe_windows` compute a token-weighted
    // mean across chunks instead of mean-of-means.
    let logprob_sum: f64 = logprobs.iter().map(|x| *x as f64).sum();
    let logprob_count = logprobs.len();
    // A caller that asked for no timestamps gets none in the tokens.
    if !params.timestamps {
        body.retain(|&t| t < timestamp_begin);
    }
    Ok(ChunkDecode {
        text,
        body,
        lang_code,
        segments,
        logprob_sum,
        logprob_count,
        chunk_no_speech_prob: no_speech_prob,
        advance,
        words,
        timing,
    })
}

/// The words of a window, timed from the window's start.
///
/// `spans` holds the start, the end and the tokens of each segment the window keeps.
/// `sot_sequence` is the prompt that opens a transcript in the window's language and task.
/// `content` is the audio frames the window holds.
///
/// With alignment heads, the words of every span are timed together through the cross
/// attention.
/// Without, or when the transcript does not fit the decoder, each span's time is shared among
/// its words.
fn window_words(
    s: &mut WhisperModelState,
    audio_features: &Tensor,
    sot_sequence: &[u32],
    spans: &[(f32, f32, Vec<u32>)],
    content: usize,
) -> AnyResult<Vec<WhisperWord>> {
    let eot = s.control.eot;
    let text_of = |ids: &[u32]| -> Vec<u32> { ids.iter().copied().filter(|&t| t < eot).collect() };
    let all: Vec<u32> = spans.iter().flat_map(|(_, _, ids)| text_of(ids)).collect();
    if all.is_empty() {
        return Ok(Vec::new());
    }
    let spaced = align::is_spaced(&text_decoder(&s.tokenizer)(&all));
    if !s.alignment_heads.is_empty() {
        let words = align::split_words(&all, text_decoder(&s.tokenizer), spaced);
        if let Some(timed) = aligned_words(s, audio_features, sot_sequence, &all, &words, content)?
        {
            return Ok(timed);
        }
    }
    let decode = text_decoder(&s.tokenizer);
    Ok(spans
        .iter()
        .flat_map(|(start, end, ids)| {
            let words = align::split_words(&text_of(ids), decode, spaced);
            align::timed_by_tokens(&words, *start, *end)
        })
        .collect())
}

/// The text of a run of tokens, special tokens included.
/// A run that ends inside a character ends on a replacement character.
fn text_decoder(tokenizer: &Tokenizer) -> impl Fn(&[u32]) -> String + Copy + '_ {
    move |ids: &[u32]| tokenizer.decode(ids, false).unwrap_or_default()
}

/// The times of `words`, the words of the text tokens `text`, from the cross attention of the
/// alignment heads over the window's `content` audio frames.
///
/// The decoder reads the transcript again from `sot_sequence`, without timestamps.
/// Each head's scores become weights over the audio positions that hold audio.
/// The path through their alignment matrix gives the position each token starts at.
/// None when the transcript does not fit the decoder.
fn aligned_words(
    s: &mut WhisperModelState,
    audio_features: &Tensor,
    sot_sequence: &[u32],
    text: &[u32],
    words: &[(String, Vec<u32>)],
    content: usize,
) -> AnyResult<Option<Vec<WhisperWord>>> {
    let sequence: Vec<u32> = sot_sequence
        .iter()
        .copied()
        .chain(std::iter::once(s.control.no_timestamps))
        .chain(text.iter().copied())
        .chain(std::iter::once(s.control.eot))
        .collect();
    let n = sequence.len();
    if n > s.config.max_target_positions {
        return Ok(None);
    }
    let mut layers: Vec<usize> = s.alignment_heads.iter().map(|&(l, _)| l).collect();
    layers.sort_unstable();
    layers.dedup();
    s.model.decoder.restart_sequence();
    let ids = Tensor::from_vec_u32(sequence, vec![1, n])?;
    let (_, scores) = s
        .model
        .decoder
        .forward_scored(&ids, audio_features, &layers)?;
    let stride = audio_features.dim(1)?;
    let positions = (content / m::FRAMES_PER_POSITION).clamp(1, stride);
    let mut heads = Vec::with_capacity(s.alignment_heads.len());
    for &(layer, head) in &s.alignment_heads {
        let Some(Some(scored)) = scores.get(layer) else {
            continue;
        };
        let rows = scored
            .narrow(1, head, 1)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        heads.push(align::softmax_rows(&rows, n, stride, positions));
    }
    if heads.is_empty() {
        return Ok(None);
    }
    let matrix = align::alignment_matrix(&heads, n, positions, align::MEDIAN_WIDTH);
    // The row of each position predicts the token after it: from the row that predicts the
    // first text token to the row that predicts the closing token.
    let first = sot_sequence.len() * positions;
    let cost: Vec<f32> = matrix[first..(n - 1) * positions]
        .iter()
        .map(|w| -w)
        .collect();
    let (path_rows, path_cols) = align::dtw(&cost, text.len() + 1, positions);
    let starts = align::row_starts(&path_rows, &path_cols);
    Ok(Some(align::timed_by_alignment(
        words,
        &starts,
        s.config.position_seconds(),
    )))
}

/// Walk `body` (the model's output tokens for this chunk) and split it
/// into (start_ts, end_ts, inner_text_tokens) tuples. Whisper's
/// convention: a segment starts at the first timestamp token after the
/// previous one closed, and ends at the next timestamp token.
/// A timestamp step lasts 1 audio position of the encoder.
/// The window starts `chunk_offset_s` into the clip and holds `window_s` seconds of audio.
fn parse_timestamp_segments(
    s: &WhisperModelState,
    body: &[u32],
    logprobs: &[f32],
    timestamp_begin: u32,
    (chunk_offset_s, window_s): (f32, f32),
    temperature: f32,
    no_speech_prob: f32,
) -> AnyResult<Vec<WhisperSegment>> {
    let step_s = s.config.position_seconds();
    // Helper: mean logprob across body[lo..hi] using the parallel
    // `logprobs` slice (one entry per body token). Returns 0.0 when
    // the range is empty.
    let avg_lp = |lo: usize, hi: usize| -> f32 {
        if hi <= lo || lo >= logprobs.len() {
            return 0.0;
        }
        let hi = hi.min(logprobs.len());
        let n = hi - lo;
        if n == 0 {
            return 0.0;
        }
        let sum: f32 = logprobs[lo..hi].iter().sum();
        sum / n as f32
    };

    // One segment, however the walk found its end. Decoding the tokens, trimming the text,
    // measuring how far it compresses and averaging its logprobs is the same work whether
    // a closing timestamp was there or the chunk simply ran out, and `what` names which of
    // the two failed when the tokenizer refuses the span.
    let build = |start: f32,
                 end: f32,
                 inner: Vec<u32>,
                 avg_logprob: f32,
                 what: &str|
     -> AnyResult<WhisperSegment> {
        let text = s
            .tokenizer
            .decode(&inner, true)
            .map_err(|e| anyhow!("tokenizer decode ({what}): {e}"))?;
        let text = text.trim().to_string();
        Ok(WhisperSegment {
            start,
            end,
            compression_ratio: zlib_compression_ratio(&text),
            text,
            tokens: inner,
            temperature,
            avg_logprob,
            no_speech_prob,
        })
    };

    let mut out = Vec::new();
    let mut i = 0;
    while i < body.len() {
        if body[i] < timestamp_begin {
            i += 1;
            continue;
        }
        let start_id = body[i];
        let start_s = (start_id - timestamp_begin) as f32 * step_s + chunk_offset_s;
        // Find the matching closing timestamp.
        let mut j = i + 1;
        while j < body.len() && body[j] < timestamp_begin {
            j += 1;
        }
        if j >= body.len() {
            // An open last segment ends where the window's audio ends.
            let inner: Vec<u32> = body[i + 1..]
                .iter()
                .copied()
                .filter(|&t| t < timestamp_begin)
                .collect();
            if !inner.is_empty() {
                let avg = avg_lp(i + 1, body.len());
                let end_s = (chunk_offset_s + window_s).max(start_s);
                out.push(build(start_s, end_s, inner, avg, "open")?);
            }
            break;
        }
        let end_id = body[j];
        let end_s = (end_id - timestamp_begin) as f32 * step_s + chunk_offset_s;
        let inner: Vec<u32> = body[i + 1..j].to_vec();
        // Sanity-check the timestamp pair. Whisper's logit rules
        // normally enforce monotonic non-decreasing timestamps, but
        // we don't apply them in greedy decode, so guard against a
        // pathological model emitting `end < start`. Drop the segment
        // and keep walking - better than surfacing a negative-duration
        // cue to the SDK.
        if end_s >= start_s && !inner.is_empty() {
            let avg = avg_lp(i + 1, j);
            out.push(build(start_s, end_s, inner, avg, "segment")?);
        }
        i = j; // Continue from this closing timestamp (may also be the next opener).
    }
    Ok(out)
}

/// Zlib compression ratio of a string: `len(text) / len(zlib(text))`.
/// Whisper's paper uses this to detect hallucination - when the model
/// emits repetitive text the compressor crushes the body and the ratio
/// climbs above ~2.4. Returns 0.0 for empty input.
fn zlib_compression_ratio(text: &str) -> f32 {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;
    if text.is_empty() {
        return 0.0;
    }
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
    if enc.write_all(text.as_bytes()).is_err() {
        return 0.0;
    }
    let compressed = match enc.finish() {
        Ok(b) => b,
        Err(_) => return 0.0,
    };
    if compressed.is_empty() {
        return 0.0;
    }
    text.len() as f32 / compressed.len() as f32
}

/// Normalize a user-supplied language string to whisper's BCP-47 code.
/// Accepts: BCP-47 ("en", "fr"), common names ("english", "French"),
/// case-insensitive. Returns the input lowercased when no mapping
/// exists so the lookup can fail downstream with a clear warning.
fn normalize_language_code(raw: &str) -> String {
    let lower = raw.to_lowercase();
    // BCP-47 short codes pass straight through.
    if lower.len() == 2 || lower.contains('-') {
        return lower;
    }
    // Unknown names fall through.
    for (name, code) in LANGUAGE_NAMES {
        if lower == *name {
            return (*code).to_string();
        }
    }
    lower
}

/// The English name of the language of the BCP-47 primary code `code`, lower case.
pub(crate) fn language_name(code: &str) -> Option<&'static str> {
    let code = code.to_lowercase();
    LANGUAGE_NAMES
        .iter()
        .find(|(_, c)| *c == code)
        .map(|(name, _)| *name)
}

/// The BCP-47 primary code of the language of the English name `name`, any case.
pub(crate) fn code_of(name: &str) -> Option<&'static str> {
    let name = name.to_lowercase();
    LANGUAGE_NAMES
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, code)| *code)
}

/// Common English language names, lower case, and their BCP-47 primary codes.
const LANGUAGE_NAMES: &[(&str, &str)] = &[
    ("english", "en"),
    ("chinese", "zh"),
    ("german", "de"),
    ("spanish", "es"),
    ("russian", "ru"),
    ("korean", "ko"),
    ("french", "fr"),
    ("japanese", "ja"),
    ("portuguese", "pt"),
    ("turkish", "tr"),
    ("polish", "pl"),
    ("catalan", "ca"),
    ("dutch", "nl"),
    ("arabic", "ar"),
    ("swedish", "sv"),
    ("italian", "it"),
    ("indonesian", "id"),
    ("hindi", "hi"),
    ("finnish", "fi"),
    ("vietnamese", "vi"),
    ("hebrew", "he"),
    ("ukrainian", "uk"),
    ("greek", "el"),
    ("malay", "ms"),
    ("czech", "cs"),
    ("romanian", "ro"),
    ("danish", "da"),
    ("hungarian", "hu"),
    ("tamil", "ta"),
    ("norwegian", "no"),
    ("thai", "th"),
    ("urdu", "ur"),
    ("croatian", "hr"),
    ("bulgarian", "bg"),
    ("lithuanian", "lt"),
    ("latin", "la"),
    ("welsh", "cy"),
    ("slovak", "sk"),
    ("telugu", "te"),
    ("persian", "fa"),
];

/// Reverse lookup: lang_token_id -> BCP-47 code (e.g. "en"). Returns None
/// for English-only models or unrecognized ids.
fn lang_id_to_code(s: &WhisperModelState, id: u32) -> Option<String> {
    s.vocab
        .languages
        .as_ref()
        .and_then(|m| m.iter().find(|(_, v)| **v == id).map(|(k, _)| k.clone()))
}

/// Whisper language auto-detect. Runs a single decoder step seeded with
/// just `<|startoftranscript|>`, then argmaxes over the subset of logits
/// at language-token positions. Returns the chosen language token id.
/// Language detection from a pre-computed encoder output. Run by
/// decode_segment after it has already encoded the chunk - keeps us
/// from running the encoder twice on the same mel.
fn detect_language_from_features(
    s: &mut WhisperModelState,
    audio_features: &Tensor,
) -> AnyResult<u32> {
    let lang_map = match s.vocab.languages.as_ref() {
        Some(m) => m,
        None => anyhow::bail!("language detect requires a multilingual model"),
    };
    let tokens_t = Tensor::from_vec_u32(vec![s.control.sot], vec![1, 1])?;
    // A scratch sequence: the main loop restarts the tokens before sampling and keeps the
    // cross attentions this pass projected from the same features.
    let ys = s.model.decoder.forward(&tokens_t, audio_features, false)?;
    let logits = s.model.decoder.final_linear(&ys)?.to_vec_f32();
    let mut codes_and_ids: Vec<(String, u32)> =
        lang_map.iter().map(|(c, id)| (c.clone(), *id)).collect();
    codes_and_ids.sort_by_key(|(_, id)| *id);
    let best = codes_and_ids
        .iter()
        .enumerate()
        .max_by(|(_, (_, a)), (_, (_, b))| {
            let la = logits
                .get(*a as usize)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            let lb = logits
                .get(*b as usize)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
            la.total_cmp(&lb)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);
    let chosen_id = codes_and_ids[best].1;
    info!("whisper language detect: <|{}|>", codes_and_ids[best].0);
    Ok(chosen_id)
}

/// Numerically-stable softmax over a host logits row.
fn host_softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn host_argmax(xs: &[f32]) -> u32 {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// `log_softmax(xs)[idx]` without materializing the full row.
fn host_log_softmax_at(xs: &[f32], idx: usize) -> f32 {
    let max = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse: f32 = xs.iter().map(|&x| (x - max).exp()).sum::<f32>().ln() + max;
    xs.get(idx).copied().unwrap_or(f32::NEG_INFINITY) - lse
}

/// Tiny non-crypto PRNG used only when temperature > 0 sampling is selected.
/// Avoids the rand dep just for this code path.
struct StdRng;
impl StdRng {
    fn weighted_sample(&mut self, weights: &[f32]) -> u32 {
        // Deterministic argmax fallback when temperature pretends to be > 0
        // but the caller hasn't supplied a real RNG. For ASR this is fine  -
        // greedy is the default, and the temperature path is only exercised
        // by the upstream confidence-based fallback logic which we don't
        // run in the v1 transcribe pipeline.
        weights
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i as u32)
            .unwrap_or(0)
    }
}

// Whisper supported language codes (subset - full list lives in the reference's
// example but isn't pub-exported). Matches the BCP-47 codes the tokenizer
// has language tokens for.
// ARCH-NAME-EXEMPT: language codes, among them `yi` for Yiddish.
const WHISPER_LANGS: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_language_code_passes_short_codes_through() {
        assert_eq!(normalize_language_code("en"), "en");
        assert_eq!(normalize_language_code("FR"), "fr");
        // BCP-47 with region tag (contains '-') - leave intact, just lowercased.
        assert_eq!(normalize_language_code("pt-BR"), "pt-br");
    }

    #[test]
    fn normalize_language_code_maps_full_names() {
        assert_eq!(normalize_language_code("English"), "en");
        assert_eq!(normalize_language_code("french"), "fr");
        assert_eq!(normalize_language_code("JAPANESE"), "ja");
    }

    #[test]
    fn normalize_language_code_falls_back_to_lowercased_input() {
        // Unknown name falls through to lowercased original so the downstream
        // language-token lookup can fail with a clear "no such token" error.
        assert_eq!(normalize_language_code("Klingon"), "klingon");
    }

    #[test]
    fn zlib_compression_ratio_empty_is_zero() {
        // Empty input returns 0.0 - the gibberish-detection rule treats 0 as
        // "no signal" rather than div-by-zero or NaN.
        assert_eq!(zlib_compression_ratio(""), 0.0);
    }

    #[test]
    fn zlib_compression_ratio_highly_repetitive_text_is_high() {
        // Long runs of the same character compress to ~nothing -> ratio is large.
        let ratio = zlib_compression_ratio(&"a".repeat(2048));
        assert!(
            ratio > 20.0,
            "repeated text should compress well, got {ratio}"
        );
    }

    #[test]
    fn zlib_compression_ratio_diverse_text_is_low() {
        // English prose compresses ~2-3x; this is the regime where whisper
        // decoded text normally lives. Any ratio under ~5 is considered
        // "not gibberish".
        let prose = "The quick brown fox jumps over the lazy dog. \
                     Sphinx of black quartz, judge my vow.";
        let ratio = zlib_compression_ratio(prose);
        assert!(
            ratio < 5.0,
            "diverse prose should not over-compress, got {ratio}"
        );
    }

    #[test]
    fn whisper_langs_list_contains_no_duplicates() {
        // Tiny invariant check - if anyone adds a duplicate entry, the
        // language-detection argmax will weight that code twice.
        let mut sorted: Vec<&&str> = WHISPER_LANGS.iter().collect();
        sorted.sort();
        let original_len = sorted.len();
        sorted.dedup();
        assert_eq!(sorted.len(), original_len, "WHISPER_LANGS has duplicates");
    }

    #[test]
    fn resolve_whisper_model_id_defaults_to_small() {
        // Default checkpoint - ~500 MB free is the most common ASR
        // request profile. Pin so a refactor doesn't silently
        // switch to large-v3 (multi-GB download on first use).
        assert_eq!(resolve_whisper_model_id(None), "openai/whisper-small",);
    }

    #[test]
    fn resolve_whisper_model_id_passes_full_hf_paths_through() {
        // Custom forks / non-openai vendors - full slash-form ids
        // threaded through verbatim (HF paths case-sensitive on disk).
        assert_eq!(
            resolve_whisper_model_id(Some("distil-whisper/distil-large-v3")),
            "distil-whisper/distil-large-v3",
        );
        assert_eq!(
            resolve_whisper_model_id(Some("Systran/faster-whisper-medium")),
            "Systran/faster-whisper-medium",
            "non-openai vendors must keep their case",
        );
    }

    #[test]
    fn resolve_whisper_model_id_unknown_bare_name_passes_through() {
        // Bare name without `whisper-` prefix and no slash -> pass
        // through unchanged (will fail at HF resolve, surfaced as a
        // proper 404 downstream).
        assert_eq!(resolve_whisper_model_id(Some("unknown-asr")), "unknown-asr");
    }

    /// A job is a clip id; its windows are `(clip, index)`; a feature is its window
    /// times ten.
    fn windows_of(clip: u32, n: u32) -> Vec<(u32, u32)> {
        (0..n).map(|i| (clip, i)).collect()
    }

    #[test]
    fn a_group_encodes_every_window_once_and_answers_each_job_from_its_own() {
        let jobs = [1u32, 2, 3];
        let cut: Vec<AnyResult<Vec<(u32, u32)>>> = vec![
            Ok(windows_of(1, 2)),
            Ok(windows_of(2, 1)),
            Ok(windows_of(3, 3)),
        ];
        let all: Vec<(u32, u32)> = cut.iter().flatten().flatten().copied().collect();
        let passes = std::cell::RefCell::new(Vec::new());
        let features = encode_in_passes(&all, 4, |pass| {
            passes.borrow_mut().push(pass.to_vec());
            Ok(pass.iter().map(|(c, i)| c * 10 + i).collect())
        });
        let answers = answer_each(&jobs, cut, features, Vec::len, |job, w, own| {
            let want: Vec<u32> = w.iter().map(|(c, i)| c * 10 + i).collect();
            assert_eq!(
                own,
                want.as_slice(),
                "clip {job} read another clip's features"
            );
            Ok(own.len())
        });
        let got: Vec<usize> = answers.into_iter().map(|a| a.unwrap()).collect();
        assert_eq!(got, [2, 1, 3]);
        let passes = passes.borrow();
        assert_eq!(passes.len(), 2, "six windows at four a pass");
        assert_eq!(passes[0].len(), 4);
        assert_eq!(passes[1], windows_of(3, 3)[1..]);
    }

    #[test]
    fn a_job_that_fails_alone_leaves_its_companions_answers_intact() {
        let jobs = [1u32, 2, 3];
        let cut: Vec<AnyResult<Vec<(u32, u32)>>> = vec![
            Ok(windows_of(1, 1)),
            Err(anyhow!("clip 2 is not audio")),
            Ok(windows_of(3, 2)),
        ];
        let all: Vec<(u32, u32)> = cut.iter().flatten().flatten().copied().collect();
        assert_eq!(all.len(), 3, "the failed clip brings no window");
        let features = encode_in_passes(&all, usize::MAX, |pass| Ok(pass.to_vec()));
        let answers = answer_each(&jobs, cut, features, Vec::len, |_, w, own| {
            assert_eq!(own, w.as_slice());
            Ok(own.len())
        });
        assert_eq!(answers[0].as_ref().unwrap(), &1);
        assert_eq!(
            answers[1].as_ref().unwrap_err().to_string(),
            "clip 2 is not audio"
        );
        assert_eq!(answers[2].as_ref().unwrap(), &2);
    }

    #[test]
    fn a_window_that_fails_fails_its_own_clip_alone() {
        let jobs = [1u32, 2, 3];
        let cut: Vec<AnyResult<Vec<(u32, u32)>>> = vec![
            Ok(windows_of(1, 1)),
            Ok(windows_of(2, 2)),
            Ok(windows_of(3, 1)),
        ];
        let all: Vec<(u32, u32)> = cut.iter().flatten().flatten().copied().collect();
        let calls = std::cell::Cell::new(0usize);
        // The pass of four fails, and so does the second window of clip 2 on its own.
        let features = encode_in_passes(&all, 4, |pass| {
            calls.set(calls.get() + 1);
            if pass.contains(&(2, 1)) {
                anyhow::bail!("window (2, 1) failed");
            }
            Ok(pass.to_vec())
        });
        assert_eq!(
            calls.get(),
            5,
            "the failed pass is encoded again one window at a time"
        );
        let answers = answer_each(&jobs, cut, features, Vec::len, |_, w, own| {
            assert_eq!(own, w.as_slice());
            Ok(own.len())
        });
        assert_eq!(answers[0].as_ref().unwrap(), &1);
        assert_eq!(
            answers[1].as_ref().unwrap_err().to_string(),
            "window (2, 1) failed"
        );
        assert_eq!(answers[2].as_ref().unwrap(), &1);
    }

    #[test]
    fn passes_keep_the_window_order_and_refuse_a_short_answer() {
        let all = windows_of(1, 5);
        let sizes = std::cell::RefCell::new(Vec::new());
        let features: Vec<(u32, u32)> = encode_in_passes(&all, 2, |pass| {
            sizes.borrow_mut().push(pass.len());
            Ok(pass.to_vec())
        })
        .into_iter()
        .map(|f| f.unwrap())
        .collect();
        assert_eq!(features, all);
        assert_eq!(*sizes.borrow(), [2, 2, 1]);
        let single: Vec<(u32, u32)> = encode_in_passes(&all, 0, |pass| Ok(pass.to_vec()))
            .into_iter()
            .map(|f| f.unwrap())
            .collect();
        assert_eq!(single, all, "a cap of zero still runs one window a pass");
        let short = encode_in_passes(&all, 2, |pass| Ok(pass[..1].to_vec()));
        assert!(
            short[..4].iter().all(Result::is_err),
            "a pass that drops a window is refused for each of its windows"
        );
        assert!(short[4].is_ok(), "the lone last window keeps its feature");
    }

    /// The first timestamp token of the toy vocabularies below.
    const TB: u32 = 1000;

    /// A timestamp token at `frames` mel frames from its window's start.
    fn ts(frames: usize) -> u32 {
        TB + (frames / 2) as u32
    }

    #[test]
    fn a_window_that_ends_inside_a_segment_hands_it_to_the_next_window() {
        let body = [ts(0), 1, 2, ts(500), ts(500), 3, 4, ts(2400), ts(2400), 5];
        assert_eq!(
            seek_step(&body, TB, 3000, 2),
            SeekStep {
                kept: 8,
                advance: 2400
            }
        );
    }

    #[test]
    fn a_window_whose_segments_all_close_moves_by_the_whole_window() {
        let whole = |body: &[u32], frames| SeekStep {
            kept: body.len(),
            advance: frames,
        };
        let closed = [ts(0), 1, ts(500), ts(500), 2, ts(900)];
        assert_eq!(seek_step(&closed, TB, 3000, 2), whole(&closed, 3000));
        let single = [ts(0), 1, 2, ts(700)];
        assert_eq!(seek_step(&single, TB, 1200, 2), whole(&single, 1200));
        let none = [1, 2, 3];
        assert_eq!(seek_step(&none, TB, 3000, 2), whole(&none, 3000));
        let at_start = [ts(0), ts(0), 1];
        assert_eq!(seek_step(&at_start, TB, 3000, 2), whole(&at_start, 3000));
        assert_eq!(seek_step(&[], TB, 3000, 2), whole(&[], 3000));
    }

    /// A toy recording: word `k` sounds from `k * 47 + 5` to `k * 47 + 45` mel frames, and
    /// every 7 words make a sentence.
    /// Word 63 straddles the end of the first window.
    fn toy_words() -> Vec<(u32, usize, usize)> {
        (0..160u32)
            .map(|k| (k, k as usize * 47 + 5, k as usize * 47 + 45))
            .collect()
    }

    /// The output of a toy model on the window of `frames` frames at `seek`: the sentences
    /// that start in the window, each opened by a timestamp, then their words heard whole.
    /// A sentence heard to its end is closed by a timestamp.
    fn toy_decode(words: &[(u32, usize, usize)], seek: usize, frames: usize) -> Vec<u32> {
        let end = seek + frames;
        let mut body = Vec::new();
        for sentence in words.chunks(7) {
            let (start, stop) = (sentence[0].1, sentence[sentence.len() - 1].2);
            if start < seek {
                continue;
            }
            if start >= end {
                break;
            }
            body.push(ts(start - seek));
            body.extend(sentence.iter().filter(|w| w.2 <= end).map(|w| w.0));
            if stop > end {
                break;
            }
            body.push(ts(stop - seek));
        }
        body
    }

    #[test]
    fn the_seek_keeps_every_word_of_a_recording_longer_than_a_window() {
        let words = toy_words();
        let content = words[words.len() - 1].2;
        let said: Vec<u32> = words.iter().map(|w| w.0).collect();
        let (mut seek, mut heard) = (0, Vec::<u32>::new());
        while seek < content {
            let frames = (content - seek).min(m::N_FRAMES);
            let body = toy_decode(&words, seek, frames);
            let step = seek_step(&body, TB, frames, 2);
            heard.extend(body[..step.kept].iter().copied().filter(|&t| t < TB));
            seek += step.advance;
        }
        assert_eq!(heard, said);
        let mut fixed = Vec::new();
        for seek in (0..content).step_by(m::N_FRAMES) {
            let frames = (content - seek).min(m::N_FRAMES);
            let heard_whole = words.iter().filter(|w| w.1 >= seek && w.2 <= seek + frames);
            fixed.extend(heard_whole.map(|w| w.0));
        }
        assert!(
            !fixed.contains(&63),
            "fixed windows lose the word across their edge"
        );
    }

    /// Logits of a toy vocabulary: text below `TB`, `eot` at 999, 1501 timestamps from `TB`.
    fn toy_logits() -> Vec<f32> {
        vec![0.0; TB as usize + 1501]
    }

    fn allowed(logits: &[f32]) -> (bool, bool, bool) {
        let text = logits[..999].iter().any(|l| l.is_finite());
        let eot = logits[999].is_finite();
        let stamp = logits[TB as usize..].iter().any(|l| l.is_finite());
        (text, eot, stamp)
    }

    #[test]
    fn the_timestamp_rules_open_pair_and_order_the_timestamps() {
        let mut l = toy_logits();
        apply_timestamp_rules(&mut l, &[], TB, 999);
        assert_eq!(
            allowed(&l),
            (false, false, true),
            "the output opens with a timestamp"
        );

        let mut l = toy_logits();
        apply_timestamp_rules(&mut l, &[TB + 5], TB, 999);
        assert!(
            !allowed(&l).2,
            "a lone opening timestamp is followed by text"
        );

        let mut l = toy_logits();
        apply_timestamp_rules(&mut l, &[TB, 7, TB + 40], TB, 999);
        assert!(
            !allowed(&l).0,
            "a timestamp after text is not followed by text"
        );
        assert!(
            l[TB as usize + 39].is_infinite(),
            "a timestamp never goes back"
        );
        assert!(
            l[TB as usize + 40].is_finite(),
            "the next segment may open at once"
        );

        let mut l = toy_logits();
        apply_timestamp_rules(&mut l, &[TB, 7, TB + 40, TB + 40], TB, 999);
        assert!(!allowed(&l).2, "2 timestamps in a row are followed by text");

        let mut l = toy_logits();
        apply_timestamp_rules(&mut l, &[TB, 7], TB, 999);
        assert!(
            l[TB as usize].is_infinite(),
            "a closing timestamp is later than the opening"
        );
        assert_eq!(
            allowed(&l),
            (false, false, true),
            "all timestamps together outweigh any single text token"
        );
        let mut l = toy_logits();
        l[7] = 10.0;
        apply_timestamp_rules(&mut l, &[TB, 7], TB, 999);
        assert!(
            l[7].is_finite(),
            "a text token likelier than every timestamp stays"
        );
    }

    #[test]
    fn a_clip_longer_than_a_window_is_read_by_its_seek() {
        assert_eq!(content_frames(m::N_SAMPLES), m::N_FRAMES);
        assert_eq!(content_frames(m::N_SAMPLES + 1), m::N_FRAMES + 1);
        assert_eq!(content_frames(1), 1);
        let mel = Tensor::from_vec_f32(vec![1.0; 2 * 4000], vec![1, 2, 4000]).unwrap();
        let tail = window_at(&mel, 3000, 3000, m::N_FRAMES, -0.5).unwrap();
        assert_eq!(tail.shape().dims3().unwrap(), (1, 2, m::N_FRAMES));
        let row = tail.narrow(1, 0, 1).unwrap().to_vec_f32();
        assert!(row[..1000].iter().all(|&v| v == 1.0));
        assert!(
            row[1000..].iter().all(|&v| v == -0.5),
            "past the last frame is the padding value"
        );
        let cut = window_at(&mel, 0, 1200, m::N_FRAMES, -0.5).unwrap();
        let row = cut
            .narrow(1, 1, 1)
            .unwrap()
            .contiguous()
            .unwrap()
            .to_vec_f32();
        assert!(row[..1200].iter().all(|&v| v == 1.0));
        assert!(
            row[1200..].iter().all(|&v| v == -0.5),
            "past the cut is the padding value"
        );
    }

    /// Windows cut at the silences between the words of the toy recording keep every word
    /// once, and each window but the last ends between 2 words.
    #[test]
    fn windows_cut_at_silences_keep_every_word_once() {
        let words = toy_words();
        let content = words[words.len() - 1].2;
        let mut levels = vec![0.0f32; content];
        for &(_, from, to) in &words {
            levels[from..to].iter_mut().for_each(|l| *l = 1.0);
        }
        let level = silence::silence_level(&levels);
        let said: Vec<u32> = words.iter().map(|w| w.0).collect();
        let (mut seek, mut heard) = (0, Vec::<u32>::new());
        while seek < content {
            let frames = silence::window_length(&levels, level, seek, m::N_FRAMES);
            let end = seek + frames;
            assert!(
                end == content || levels[end] == 0.0,
                "the window at {seek} ends inside a word at {end}"
            );
            let body = toy_decode(&words, seek, frames);
            let step = seek_step(&body, TB, frames, 2);
            heard.extend(body[..step.kept].iter().copied().filter(|&t| t < TB));
            seek += step.advance;
        }
        assert_eq!(heard, said);
    }
}
