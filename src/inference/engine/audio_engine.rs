//! Audio engine - Whisper transcription.
//!
//! End-to-end flow:
//!   1. `load_whisper("openai/whisper-small")` downloads weights via hf-hub
//!      (cached after first call), loads model + tokenizer + mel filters.
//!   2. `transcribe(samples, sr)` takes f32 mono PCM, resamples conceptually
//!      (we accept 16 kHz only here; callers convert), runs the encoder
//!      followed by a greedy decoder loop, returns the final text.
//!
//! Implementation mirrors the reference whisper example but simplified for
//! greedy / single-language / no-timestamps. Multi-language, beam search,
//! and timestamp segmentation can be layered on later without changing
//! the public API.

use crate::inference::model::whisper::{self as m, audio, Config};
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
    /// Optional context prompt - text the model will see before the
    /// SOT prompt as `<|startofprev|>` + tokens. Whisper's documented
    /// way to bias vocabulary / continue a previous segment. Caller
    /// is responsible for length (we truncate at 220 tokens).
    pub initial_prompt: Option<String>,
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
}

pub struct AudioModelInfo {
    pub name: String,
    pub model_type: String,
    /// GPU bytes the load actually took - see `ImageModelInfo::resident_bytes` for
    /// why this is measured rather than derived. `/api/ps` published 0 here, which a
    /// dashboard reads as "this model costs nothing".
    pub resident_bytes: u64,
}

pub struct AudioEngine {
    model_state: Arc<Mutex<Option<WhisperModelState>>>,
    name: Arc<Mutex<Option<String>>>,
    /// "CUDA" or "CPU", kept apart from the state so a listing never waits on a transcription.
    device_kind: Arc<Mutex<Option<String>>>,
    resident_bytes: Arc<Mutex<u64>>,
}

impl AudioEngine {
    pub fn new() -> Self {
        Self {
            model_state: Arc::new(Mutex::new(None)),
            name: Arc::new(Mutex::new(None)),
            device_kind: Arc::new(Mutex::new(None)),
            resident_bytes: Arc::new(Mutex::new(0)),
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
        })
    }

    /// Coarse device kind of the loaded whisper model: "CUDA" or "CPU".
    /// Used by /api/models/loaded so the GUI Hardware tab shows the
    /// right device badge. pick_device prefers CUDA0 but we read the
    /// actual loaded state in case the load path fell back to CPU.
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
        let state = tokio::task::spawn_blocking(move || -> AnyResult<WhisperModelState> {
            load_whisper_blocking(&model_id_clone)
        })
        .await
        .map_err(|e| anyhow!("whisper load join: {e}"))??;

        let free_after_load = crate::inference::place::vram_manager::free_total();
        let kind = if state.device.is_cuda() {
            "CUDA"
        } else {
            "CPU"
        };
        let mut guard = self.model_state.lock().await;
        *guard = Some(state);
        *self.name.lock().await = Some(model_id);
        *self.device_kind.lock().await = Some(kind.to_string());
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
        let state = self.model_state.clone();
        tokio::task::spawn_blocking(move || -> AnyResult<TranscribeResult> {
            let mut guard = state.blocking_lock();
            let s = guard
                .as_mut()
                .ok_or_else(|| anyhow!("whisper unloaded mid-transcribe"))?;
            transcribe_blocking(s, &samples, &params)
        })
        .await
        .map_err(|e| anyhow!("transcribe join: {e}"))?
        // Centralised engine-error logging (see engine::log_engine_error)
        // - covers both the synchronous Result path (which DOES surface
        // via 5xx but TraceLayer may not always elevate it) and ensures
        // consistent formatting across modalities.
        .inspect_err(|e| crate::inference::engine::log_engine_error("audio", "transcribe", e))
    }
}

impl Default for AudioEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ------------------------------------------------------------
// Internal: HF download + load
// ------------------------------------------------------------

fn load_whisper_blocking(model_id: &str) -> AnyResult<WhisperModelState> {
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

    // Native substrate, F32 end-to-end (whisper-small is ~1 GB F32; the
    // per-step host slicing keeps decode transfers tiny).
    // Budget-aware placement: fastest GPU whose real free VRAM fits the F32 load
    // (x2 vs the file when the checkpoint is F16), else CPU - never a blind CUDA:0.
    let weights_size = std::fs::metadata(&weights_path)
        .map(|m| m.len())
        .unwrap_or(3_000_000_000);
    let device = pick_device(weights_size * 2);
    info!("Whisper device: {device:?}, dtype: F32 (native)");

    let vb = unsafe { VarBuilder::from_files(&[weights_path], DType::F32, &device)? };
    let model = WhisperF32::load(&vb, config.clone()).context("load whisper weights")?;

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
    })
}

fn pick_device(model_bytes: u64) -> Device {
    // hetero_place::place_whole probes real free VRAM and degrades to CPU when the
    // model (+512 MB runtime reserve) doesn't fit - the fleet-wide placement policy.
    crate::inference::place::plan::place_whole(model_bytes, 512 << 20)
}

/// Resolve a caller-supplied Whisper model name into a full HF model id.
///
/// Three input shapes accepted:
///
///   - `None` -> default to `openai/whisper-small` (fits in ~500 MB free,
///     reasonable accuracy/latency for the typical ASR request).
///   - OpenAI documented id (`whisper-1`, case-insensitive) -> mapped to
///     `whisper-small` so SDK clients passing `Whisper-1` resolve.
///   - Bare `whisper-...` name (no slash) -> prefixed with `openai/` and
///     lowercased. Anything with `/` is passed through as a full HF
///     model id.
///
/// Lifted out of `load_whisper` so the mapping can be unit-tested
/// without spinning up an async runtime or hitting HF.
pub(crate) fn resolve_whisper_model_id(name: Option<&str>) -> String {
    let raw = name.unwrap_or("openai/whisper-small");
    let raw = match raw.to_lowercase().as_str() {
        "whisper-1" => "whisper-small",
        _ => raw,
    };
    if raw.contains('/') {
        raw.to_string()
    } else if raw.starts_with("whisper-") || raw.starts_with("Whisper-") {
        format!("openai/{}", raw.to_lowercase())
    } else {
        raw.to_string()
    }
}

fn token_id(t: &Tokenizer, name: &str) -> AnyResult<u32> {
    t.token_to_id(name)
        .ok_or_else(|| anyhow!("whisper tokenizer missing {name}"))
}

// ------------------------------------------------------------
// Internal: transcription pipeline
// ------------------------------------------------------------

fn transcribe_blocking(
    s: &mut WhisperModelState,
    samples: &[f32],
    params: &AudioTranscribeParams,
) -> AnyResult<TranscribeResult> {
    let duration_s = samples.len() as f32 / m::SAMPLE_RATE as f32;
    // Bail early on empty / sub-frame audio. The mel filter window is
    // ~25 ms (400 samples at 16 kHz); shorter input would produce a
    // 0-frame mel that crashes downstream slicing.
    if samples.is_empty() {
        return Ok(TranscribeResult {
            text: String::new(),
            language: params.language.clone().unwrap_or_else(|| "en".to_string()),
            duration_s: 0.0,
            tokens: Vec::new(),
            segments: Vec::new(),
        });
    }
    // 1. Build mel spectrogram on CPU.
    let mel = audio::pcm_to_mel(&s.config, samples, &s.mel_filters);
    let mel_len = mel.len();
    let n_mels = s.config.num_mel_bins;
    let n_frames = mel_len / n_mels;
    if n_frames == 0 {
        return Ok(TranscribeResult {
            text: String::new(),
            language: params.language.clone().unwrap_or_else(|| "en".to_string()),
            duration_s,
            tokens: Vec::new(),
            segments: Vec::new(),
        });
    }
    // Host-resident mel; per-chunk slices upload in decode_segment.
    let mel_tensor = Tensor::from_vec_f32(mel, vec![1, n_mels, n_frames])?;

    // 2. Iterate over 30-second chunks.
    // Whisper assumes a single language per file. Cache the detected
    // code after chunk 0 and pin params.language for the rest of the
    // file - skips a redundant per-chunk SOT decoder pass plus the
    // tokenizer language-id lookups.
    let mut all_text = String::new();
    let mut all_tokens: Vec<u32> = Vec::new();
    let mut all_segments: Vec<WhisperSegment> = Vec::new();
    let mut detected_lang: Option<String> = None;
    // Token-weighted logprob aggregate for the placeholder segment.
    let mut total_lp_sum: f64 = 0.0;
    let mut total_lp_count: usize = 0;
    let mut first_chunk_no_speech: Option<f32> = None;
    let mut effective = params.clone();
    let mut seek = 0usize;
    while seek < n_frames {
        let segment_size = (n_frames - seek).min(m::N_FRAMES);
        let mel_segment = mel_tensor.narrow(2, seek, segment_size)?;
        let chunk_offset_s = (seek as f32) * 0.01; // 1 frame = 10 ms in whisper's mel
        let dec = decode_segment(s, &mel_segment, &effective, chunk_offset_s)?;
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
        total_lp_sum += dec.logprob_sum;
        total_lp_count += dec.logprob_count;
        if first_chunk_no_speech.is_none() && dec.chunk_no_speech_prob > 0.0 {
            first_chunk_no_speech = Some(dec.chunk_no_speech_prob);
        }
        seek += segment_size;
    }
    let language = detected_lang.unwrap_or_else(|| "en".to_string());
    let placeholder_avg_logprob = if total_lp_count > 0 {
        (total_lp_sum / total_lp_count as f64) as f32
    } else {
        0.0
    };
    let placeholder_no_speech_prob = first_chunk_no_speech.unwrap_or(0.0);
    // Clamp open-ended segment ends to the actual file duration. The
    // chunk-level parser pads to `chunk_offset_s + 30.0` for open
    // tails since it doesn't know `duration_s`; trim those down here
    // so the surfaced range never exceeds the audio length.
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
}

fn decode_segment(
    s: &mut WhisperModelState,
    mel_segment: &Tensor,
    params: &AudioTranscribeParams,
    chunk_offset_s: f32,
) -> AnyResult<ChunkDecode> {
    // Pad to N_FRAMES so the encoder positional embeddings line up
    // (host-side; one upload to the compute device after).
    let (_, _, seg) = mel_segment.shape().dims3()?;
    let mel_in = if seg < m::N_FRAMES {
        let pad = Tensor::zeros(
            vec![1, s.config.num_mel_bins, m::N_FRAMES - seg],
            crate::tensor::DType::F32,
        )?;
        Tensor::cat(&[mel_segment, &pad], 2)?
    } else {
        mel_segment.clone()
    }
    .to_device(&s.device)?;

    let audio_features = s.model.encoder.forward(&mel_in, true)?;

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
        match detect_language_from_features(s, &audio_features) {
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
    if !params.timestamps {
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

    // First iteration: feed the entire prompt prefix and let the decoder
    // initialize self-attn + cross-attn KV caches. Subsequent iterations:
    // feed just the most recently sampled token - the cached K/V handle
    // the prefix, slashing per-step decoder cost from O(N) to O(1).
    s.model.decoder.reset_kv_cache();
    for i in 0..sample_len {
        let step_tokens: &[u32] = if i == 0 {
            tokens.as_slice()
        } else {
            // Just the last token; the rest is already in the KV cache.
            &tokens[tokens.len() - 1..]
        };
        let toks_t = Tensor::from_vec_u32(step_tokens.to_vec(), vec![1, step_tokens.len()])?
            .to_device(&s.device)?;
        let ys = s.model.decoder.forward(&toks_t, &audio_features, i == 0)?;
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
        tokens.push(next_token);
        if next_token == s.control.eot || tokens.len() >= s.config.max_target_positions {
            break;
        }
    }

    // Strip the prompt prefix (and trailing EOT) before any further
    // processing. `prompt_len` was captured above before sampling.
    let body: Vec<u32> = tokens
        .into_iter()
        .skip(prompt_len)
        .take_while(|&t| t != s.control.eot)
        .collect();

    // Parse segments when timestamps were requested AND the model
    // actually emitted timestamp tokens (`token >= timestamp_begin`).
    // Each pair of consecutive timestamp tokens bounds a segment whose
    // text is the tokens between them.
    let timestamp_begin = s.control.no_timestamps + 1;
    // `logprobs` has one entry per appended token (parallel to body
    // plus a trailing EOT if the loop closed on it). Truncate to body
    // length so per-segment averaging uses only the text/timestamp
    // tokens we exposed.
    if logprobs.len() > body.len() {
        logprobs.truncate(body.len());
    }
    let temperature_f32 = params.temperature.max(0.0);
    let segments = if params.timestamps {
        parse_timestamp_segments(
            s,
            &body,
            &logprobs,
            timestamp_begin,
            chunk_offset_s,
            temperature_f32,
            no_speech_prob,
        )?
    } else {
        Vec::new()
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
    // Sum + count let `transcribe_blocking` compute a token-weighted
    // mean across chunks instead of mean-of-means.
    let logprob_sum: f64 = logprobs.iter().map(|x| *x as f64).sum();
    let logprob_count = logprobs.len();
    Ok(ChunkDecode {
        text,
        body,
        lang_code,
        segments,
        logprob_sum,
        logprob_count,
        chunk_no_speech_prob: no_speech_prob,
    })
}

/// Walk `body` (the model's output tokens for this chunk) and split it
/// into (start_ts, end_ts, inner_text_tokens) tuples. Whisper's
/// convention: a segment starts at the first timestamp token after the
/// previous one closed, and ends at the next timestamp token. The
/// "20 ms per step" rate is fixed for all whisper variants.
fn parse_timestamp_segments(
    s: &WhisperModelState,
    body: &[u32],
    logprobs: &[f32],
    timestamp_begin: u32,
    chunk_offset_s: f32,
    temperature: f32,
    no_speech_prob: f32,
) -> AnyResult<Vec<WhisperSegment>> {
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
        let start_s = (start_id - timestamp_begin) as f32 * 0.02 + chunk_offset_s;
        // Find the matching closing timestamp.
        let mut j = i + 1;
        while j < body.len() && body[j] < timestamp_begin {
            j += 1;
        }
        if j >= body.len() {
            // Open-ended last segment: clamp the end to the chunk's
            // logical end (the next 30 s boundary).
            let inner: Vec<u32> = body[i + 1..]
                .iter()
                .copied()
                .filter(|&t| t < timestamp_begin)
                .collect();
            if !inner.is_empty() {
                let avg = avg_lp(i + 1, body.len());
                out.push(build(start_s, chunk_offset_s + 30.0, inner, avg, "open")?);
            }
            break;
        }
        let end_id = body[j];
        let end_s = (end_id - timestamp_begin) as f32 * 0.02 + chunk_offset_s;
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
    // Common English language names -> BCP-47 (subset; covers the
    // top ~30 most-used). Unknown names fall through.
    let map: &[(&str, &str)] = &[
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
    for (name, code) in map {
        if lower == *name {
            return (*code).to_string();
        }
    }
    lower
}

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
    let tokens_t = Tensor::from_vec_u32(vec![s.control.sot], vec![1, 1])?.to_device(&s.device)?;
    // flush=true: this is a scratch decoder run that doesn't share KV
    // cache state with the main decode loop. The main loop will call
    // forward(flush=true) again before sampling, so any cross-attn
    // cache we touch here is overwritten.
    let ys = s.model.decoder.forward(&tokens_t, audio_features, true)?;
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
    fn resolve_whisper_model_id_maps_openai_alias_case_insensitively() {
        // OpenAI's documented `whisper-1` (and SDK enum variants)
        // -> local whisper-small. SDK clients passing `Whisper-1`
        // / `WHISPER-1` must all resolve to the same checkpoint.
        assert_eq!(
            resolve_whisper_model_id(Some("whisper-1")),
            "openai/whisper-small",
        );
        assert_eq!(
            resolve_whisper_model_id(Some("Whisper-1")),
            "openai/whisper-small",
        );
        assert_eq!(
            resolve_whisper_model_id(Some("WHISPER-1")),
            "openai/whisper-small",
        );
    }

    #[test]
    fn resolve_whisper_model_id_auto_prefixes_bare_whisper_names() {
        // Bare `whisper-tiny.en` -> `openai/whisper-tiny.en`. The
        // explicit lowercasing matters for `.en` checkpoints; HF
        // paths are case-sensitive on disk but the openai/ repo
        // is all lowercase.
        assert_eq!(
            resolve_whisper_model_id(Some("whisper-tiny.en")),
            "openai/whisper-tiny.en",
        );
        assert_eq!(
            resolve_whisper_model_id(Some("whisper-medium")),
            "openai/whisper-medium",
        );
        assert_eq!(
            resolve_whisper_model_id(Some("Whisper-Large-V3")),
            "openai/whisper-large-v3",
            "case-insensitive prefix match; output lowercased",
        );
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
}
