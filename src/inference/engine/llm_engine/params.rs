//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// How much exactness a streamed model may give up for speed, as a policy the engine turns into
/// a measured divergence budget (see `deepseek_v41::calibrate`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum QualityLevel {
    /// The output of the full model, bit for bit.
    #[default]
    Exact,
    /// Differences no larger than the model's own run-to-run numerical noise.
    Balanced,
    /// More speed for small, measured differences: a fixed multiple of that noise.
    Fast,
}

/// Check whether an error indicates a CUDA out-of-memory condition. Used to
/// distinguish recoverable load failures (retry via split) from real errors,
/// and recoverable runtime allocation failures (retry with a smaller prefill
/// chunk) from genuine compute errors.
/// Truncate a tokenized prompt so it fits the model's KV window, reserving room
/// for `want_gen` generation tokens. A prompt longer than the window would overflow
/// the KV cache and break the prefill attention mask (scores [chunk, kv>ctx] vs
/// mask [chunk, ctx]) -> a crash. Keeps the first token (BOS/template start) and the
/// most-recent tail (the actual query), matching ollama's num_ctx behaviour.
/// `kv_window` = the effective KV capacity (config context ∧ model context ∧ request).
/// The effective KV window = min(model context, config/user cap, per-request num_ctx).
/// The KV cache is allocated with the config cap (`user_context_length`), NOT the model's
/// advertised GGUF context (e.g. mistral-nemo advertises 1024000 but is capped far lower),
/// so bound by both. `config_ctx == 0` means "no user cap" -> fall back to the model context.
pub(super) fn effective_kv_window(
    model_ctx: usize,
    config_ctx: usize,
    req_ctx: Option<usize>,
) -> usize {
    let base = if config_ctx > 0 {
        model_ctx.min(config_ctx)
    } else {
        model_ctx
    };
    req_ctx.map(|c| c.min(base)).unwrap_or(base)
}

/// Splice control tokens a pixtral vision prefill adds around the image
/// region: `[BOS][INST] <imgxN> [IMG_END] ...text... [/INST]`. Moondream only
/// prepends BOS (1); using the larger pixtral bound for both merely
/// over-reserves 3 tokens.
pub(super) const VISION_SPLICE_CONTROL_TOKENS: usize = 4;

/// KV positions a vision request occupies IN ADDITION to the tokenized text
/// prompt. Image embeddings are spliced into the prefill AFTER tokenization
/// (`forward_pixtral_spliced` / `forward_with_img`), so `clamp_prompt_to_window`
/// alone cannot bound the real prefill length - the text clamp must reserve
/// room for the embeds + splice control tokens. Without this reservation a
/// large image + long question overflows the allocated KV (`kv_ctx`): every
/// layer's Q8 KV append fails past `max_seq_len`, silently falls back to
/// F-dtype, and the subsequent CUDA-graph decode captures an empty graph ->
/// degenerate repeated-token output (historically: a mask-broadcast panic).
pub(super) fn vision_extra_kv(image_embeds: Option<&Tensor>) -> usize {
    image_embeds
        .and_then(|t| t.dim(1).ok())
        .map(|n| n + VISION_SPLICE_CONTROL_TOKENS)
        .unwrap_or(0)
}

/// Same accounting for the qwen35moe (Qwen3-VL) vision path: the prompt
/// carries ONE `image_token` sentinel that `forward_qwen35_image` expands in
/// place to `n_merged = (gh/2).(gw/2)` copies (spatial_merge_size=2) BEFORE
/// the ViT splice, so the real prefill length exceeds the tokenized prompt by
/// `n_merged - 1`. Without reserving these, a large image + long question
/// overflows the context window as a single monolithic prefill (`heads x
/// seq² x f32` attention scores -> VRAM OOM at long context).
pub(super) fn qwen35_vision_extra_kv(img: Option<&(Tensor, (usize, usize))>) -> usize {
    img.map(|&(_, (gh, gw))| ((gh / 2) * (gw / 2)).saturating_sub(1))
        .unwrap_or(0)
}

/// Prompt clamp for qwen35-VL: the image lives IN the token stream as a single
/// front-positioned sentinel, and the generic `clamp_prompt_to_window`
/// (BOS + tail) would drop it - turning an over-long request into a hard
/// "sentinel missing" error. Preserve the head THROUGH the sentinel (chat
/// preamble + vision markers), then fill the remaining budget with the prompt
/// tail (same keep-the-recent-text semantics as the generic clamp).
pub(super) fn clamp_qwen35_prompt_to_window(
    tokens: Vec<u32>,
    kv_window: usize,
    want_gen: usize,
    sentinel: u32,
) -> Vec<u32> {
    let Some(si) = tokens.iter().position(|&t| t == sentinel) else {
        return clamp_prompt_to_window(tokens, kv_window, want_gen);
    };
    if kv_window == 0 || tokens.len() <= 1 {
        return tokens;
    }
    let reserve = want_gen.max(1).min(kv_window / 2).max(16);
    let budget = kv_window.saturating_sub(reserve).max(1);
    if tokens.len() <= budget {
        return tokens;
    }
    let head_len = si + 1;
    if head_len >= budget {
        // Even the pre-image region overflows the budget; nothing sensible to
        // preserve - generic clamp (the request then fails with the clean
        // "sentinel missing" prefill error rather than a KV overflow).
        return clamp_prompt_to_window(tokens, kv_window, want_gen);
    }
    let orig = tokens.len();
    let tail_len = budget - head_len; // orig > budget ⇒ tail starts after the sentinel
    let mut out = Vec::with_capacity(budget);
    out.extend_from_slice(&tokens[..head_len]);
    out.extend_from_slice(&tokens[orig - tail_len..]);
    warn!("prompt {orig} tok > context window {kv_window}; truncated to {} (head incl. image sentinel + last {tail_len} tokens)",
        out.len());
    out
}

/// The request's `context` prefix ahead of the freshly tokenized prompt. Applied to the
/// raw ids, BEFORE the window clamp and any sentinel logic, so the replayed history is
/// exactly what the window trims and what the prompt cache can match.
pub(super) fn with_prefix_tokens(params: &GenerationParams, ids: &[u32]) -> Vec<u32> {
    match &params.prefix_tokens {
        Some(pre) if !pre.is_empty() => pre.iter().copied().chain(ids.iter().copied()).collect(),
        _ => ids.to_vec(),
    }
}

pub(super) fn clamp_prompt_to_window(
    tokens: Vec<u32>,
    kv_window: usize,
    want_gen: usize,
) -> Vec<u32> {
    if kv_window == 0 || tokens.len() <= 1 {
        return tokens;
    }
    // Reserve for generation, but never sacrifice more than half the window to it.
    let reserve = want_gen.max(1).min(kv_window / 2).max(16);
    let budget = kv_window.saturating_sub(reserve).max(1);
    if tokens.len() <= budget {
        return tokens;
    }
    let orig = tokens.len();
    let tail_len = budget.saturating_sub(1);
    let mut out = Vec::with_capacity(budget);
    out.push(tokens[0]); // preserve BOS / template start
    out.extend_from_slice(&tokens[orig - tail_len..]);
    warn!("prompt {orig} tok > context window {kv_window}; truncated to {} (BOS + last {tail_len} tokens)",
        out.len());
    out
}

pub(super) static RAYON_INIT: Once = Once::new();

/// Inference configuration for the language engine
#[derive(Clone)]
pub struct InferenceConfig {
    /// Model ID
    pub model_id: String,
    /// The window to load with; 0 is the model's own. A request's `num_ctx` sets it.
    pub context_length: usize,
    /// Data type for model weights
    pub dtype: crate::tensor::DType,
    /// The card a single-card load is pinned to, set by code that places a companion model
    /// (a drafter) on purpose. `None` loads on the fastest card, as placement decides.
    pub device_index: Option<usize>,
    /// Copies of the model to serve side by side, each whole on a card of its own: requests
    /// spread over them. One copy is the model once.
    pub replicas: usize,
    /// Directory of the disk tier under the snapshots: token blocks of KV at the cache's
    /// own dtype, shared between sequences by prefix, one manifest per sequence. `None`
    /// keeps snapshots in memory only.
    pub kv_disk_dir: Option<String>,
    /// The instance that reads long prompts for this one and hands back their KV cache.
    pub prefill_peer: Option<String>,
    /// Whether attention layers' caches are cut over every card.
    pub context_parallel: bool,
    /// Whether drafts are verified in recorded windows, the model's draft head guessing where
    /// they are.
    pub recorded_drafts: bool,
    /// Where the model's probes write the tensors they dump; none unless configured.
    pub probe_dump_dir: Option<String>,
    /// A routing profile that fills the expert caches at the load, before any prompt; a
    /// missing file is skipped.
    pub expert_profile: Option<String>,
    /// Where the routing counts, the loaded profile's included, are written after each
    /// request.
    pub expert_profile_save: Option<String>,
    /// Bytes the disk tier may hold, in GiB; 0 for no limit. Least recently used
    /// sequences go first.
    pub kv_disk_budget_gb: f64,
    /// Directory of the on-disk cache of DeepSeek V4.1 decode states (prompt-derived KV
    /// tensors, no text). `None` keeps the cache off.
    pub state_cache_dir: Option<String>,
    /// Bytes the decode-state cache may hold, in GiB; 0 derives it from the free space.
    pub state_cache_budget_gb: f64,
    /// Where `/v1/files` keeps uploads and batch outputs; `None` uses `files/` beside
    /// the model store.
    pub files_dir: Option<String>,
    /// The model `/v1/moderations` asks to judge; `None` answers 501.
    pub moderation_model: Option<String>,
    /// The origin clients reach this server at, for the URLs it hands out; `None` uses
    /// the request\'s `Host`.
    pub public_url: Option<String>,
    /// Per-request caps on images, pixels and audio seconds.
    pub media_limits: crate::api::media_limits::MediaLimits,
    /// How much exactness a streamed model may give up for speed.
    pub quality: QualityLevel,
    /// Progress callback
    pub progress_callback: Option<std::sync::Arc<dyn Fn(u64, u64) + Send + Sync>>,
    /// Models directory (for finding model files)
    pub models_dir: Option<PathBuf>,
    /// Disable Arc/OpenCL layers (test mode - filters devices before HeteroPlan)
    pub disable_arc_layers: bool,
    /// Disable CUDA (test mode - filters devices before HeteroPlan)
    pub disable_cuda: bool,
    /// Force all layers to CUDA, disabling heterogeneous distribution
    /// Eliminates device transfers, confirms bottleneck location
    pub force_cuda_only_layers: bool,
    /// KV-cache storage quantization. `Off` keeps K/V as the model dtype;
    /// `Q8` stores both as Q8_0 blocks (half VRAM, slight quality drift).
    /// Model-load-time only - changing this after load has no effect.
    pub kv_quant: KvQuant,
    /// The layers `[from, to)` this load holds when it is one segment of a stack that other
    /// nodes run the rest of; `None` loads the whole stack. Set by the cluster when it places a
    /// segment here, never read from the configuration.
    pub layer_share: Option<(usize, usize)>,
}

/// KV-cache storage mode for per-layer caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvQuant {
    /// Store K and V at the model's native dtype (existing behavior).
    #[default]
    Off,
    /// Store K and V as Q8_0 blocks. Halves KV memory footprint;
    /// introduces a small quant error (~0.55-0.78% vs F32 reference)
    /// that in practice is below the noise floor of coding outputs.
    Q8,
    /// Store K and V as Q4_0 blocks (1 fp16 scale + 16 bytes per 32
    /// elements). Quarters the native-dtype footprint - enables 128K
    /// contexts on 16 GB GPUs. Decode goes through a dequant-to-F16 path
    /// until fused Q4 attention kernels land, so
    /// expect slower single-token decode than `Q8` until then.
    Q4,
}

impl std::fmt::Debug for InferenceConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceConfig")
            .field("model_id", &self.model_id)
            .field("context_length", &self.context_length)
            .field("dtype", &self.dtype)
            .field("models_dir", &self.models_dir)
            .field("progress_callback", &"<closure>")
            .finish()
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_id: "default".to_string(),
            context_length: 0,
            dtype: crate::tensor::DType::F16,
            device_index: None,
            replicas: 1,
            kv_disk_dir: None,
            prefill_peer: None,
            context_parallel: false,
            recorded_drafts: false,
            probe_dump_dir: None,
            expert_profile: None,
            expert_profile_save: None,
            kv_disk_budget_gb: 0.0,
            state_cache_dir: None,
            state_cache_budget_gb: 0.0,
            files_dir: None,
            moderation_model: None,
            public_url: None,
            media_limits: crate::api::media_limits::MediaLimits::default(),
            quality: QualityLevel::Exact,
            progress_callback: None,
            models_dir: None,
            disable_arc_layers: false,
            disable_cuda: false,
            force_cuda_only_layers: false,
            kv_quant: KvQuant::Off,
            layer_share: None,
        }
    }
}

/// Why a generation ended, as the engine saw it end.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    /// The model emitted an end-of-sequence token.
    Eos,
    /// The token cap was reached.
    MaxTokens,
    /// One of the request's stop sequences was matched: this one.
    StopSequence(String),
    /// The grammar admitted no further token.
    Grammar,
    /// The client went away before the end.
    Disconnect,
}

impl FinishReason {
    /// Ollama's `done_reason`.
    pub fn ollama(&self) -> &'static str {
        match self {
            FinishReason::MaxTokens => "length",
            _ => "stop",
        }
    }
    /// OpenAI's `finish_reason`.
    pub fn openai(&self) -> &'static str {
        match self {
            FinishReason::MaxTokens => "length",
            _ => "stop",
        }
    }
    /// Anthropic's `stop_reason` and, for a stop sequence, the sequence.
    pub fn anthropic(&self) -> (&'static str, Option<&str>) {
        match self {
            FinishReason::MaxTokens => ("max_tokens", None),
            FinishReason::StopSequence(s) => ("stop_sequence", Some(s.as_str())),
            _ => ("end_turn", None),
        }
    }
}

/// One generated token's log-probability, with the alternatives asked for, each
/// decoded so a surface can print it.
#[derive(Debug, Clone)]
pub struct TokenLogprob {
    pub token: u32,
    pub text: String,
    pub logprob: f32,
    pub top: Vec<(u32, String, f32)>,
}

/// Result of a generation call, including timing and token count metrics.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The FULL token sequence: replayed context + prompt + completion.
    ///
    /// The ollama API returns this as `context` so a client can continue a
    /// conversation without resending its history. It carried the completion
    /// alone for a while, which is a valid-looking value that silently loses
    /// the client's history each turn - the ollama semantics is the whole
    /// conversation, and the next request replays it as the prefix.
    pub tokens: Vec<u32>,
    /// The generated text
    pub text: String,
    /// Number of prompt tokens actually processed by the tokenizer
    pub prompt_eval_count: u64,
    /// Time spent evaluating the prompt (nanoseconds)
    pub prompt_eval_duration: u64,
    /// Number of tokens actually generated
    pub eval_count: u64,
    /// Time spent generating tokens (nanoseconds), excludes prompt eval
    pub eval_duration: u64,
    /// Prompt tokens served from the resident KV rather than prefilled.
    pub cached_prompt_tokens: u64,
    pub finish_reason: FinishReason,
    /// Per generated token, when `top_logprobs` asked for them.
    pub logprobs: Vec<TokenLogprob>,
}

/// Per-request generation parameters that override InferenceConfig defaults
#[derive(Debug, Clone, Default)]
pub struct GenerationParams {
    /// Max tokens to generate (overrides config.max_tokens if set)
    pub max_tokens: Option<usize>,
    /// Conversation-so-far token prefix (ollama `context`), prepended before the
    /// window clamp so the history is what the window trims.
    pub prefix_tokens: Option<Vec<u32>>,
    /// Temperature for sampling (0.0 = argmax, 1.0+ = flat)
    pub temperature: Option<f32>,
    /// Top-P nucleus sampling threshold (0.0-1.0)
    pub top_p: Option<f32>,
    /// Top-K sampling (keep only top K tokens)
    pub top_k: Option<usize>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Stop sequences (generate until one is encountered)
    pub stop_sequences: Vec<String>,
    /// Early exit threshold: if Some(threshold), exit after CUDA if confidence > threshold
    pub early_exit_threshold: Option<f32>,
    /// Repetition penalty (1.0 = no penalty, >1.0 = penalize repeated tokens)
    /// Default: 1.1 (matches Ollama default)
    pub repeat_penalty: Option<f32>,
    /// How many recent tokens to consider for repetition penalty
    /// Default: 64 (matches Ollama default)
    pub repeat_last_n: Option<usize>,
    /// Per-request context window override (Ollama's `num_ctx`).
    /// If set, caps the effective context length for this request.
    /// Falls back to the model's loaded context_length when None.
    pub context_length: Option<usize>,
    /// Opaque session identifier. When two consecutive requests with the same
    /// session id share a token prefix, the KV cache is reused and only the
    /// new suffix gets prefilled. Set via Ollama option `session_id` or the
    /// OpenAI `user` field.
    pub session_id: Option<String>,
    /// Grammar-constrained decoding. Accepted shapes:
    ///   - `"json_object"`              -> bare JSON object output
    ///   - `"json_schema:<schema>"`     -> `<schema>` is a JSON-encoded JSON Schema
    ///   - `"lark:<grammar>"`           -> `<grammar>` is a Lark grammar
    ///
    /// When set, `generate()` dispatches to a dedicated sequential-decode
    /// path that masks logits with the parser's allowed-token bitset before
    /// every sample. Speculative/PLD paths are bypassed.
    pub grammar: Option<String>,
    /// Added to the named tokens' logits before every draw; `-inf` bans a token.
    pub logit_bias: Option<std::collections::HashMap<u32, f32>>,
    /// Report each drawn token's log-probability with this many alternatives; `Some(0)`
    /// reports the drawn token alone. Costs a host copy of the logits per token.
    pub top_logprobs: Option<usize>,
    /// Raised when the request's client goes away; the generation loops read it between tokens.
    /// Each request carries its own, so a request that ends never stops another.
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl GenerationParams {
    /// The model's stop sequences when the request names none of its own.
    pub fn with_default_stops(mut self, stops: &[String]) -> Self {
        if self.stop_sequences.is_empty() {
            self.stop_sequences = stops.to_vec();
        }
        self
    }
}

// The per-architecture dispatch enum (`ModelVariant`) was replaced by the
// `ModelBackend` trait - see `crate::inference::engine::model_backend`. Each arch wraps its
// concrete model in a backend struct; the engine stores a
// `Box<dyn ModelBackend>` and adding a model no longer touches ~27 match
// statements here.

/// The vision tower paired with a GenericHetero text model. `Clip` = the
/// moondream/phi2 Ollama dual-blob CLIP; `Pixtral` = the Pixtral-ViT +
/// projector (llama-arch decoder, mmproj with clip.projector_type=pixtral).
/// Pixtral is variable-resolution: it encodes from the RAW image (HF-exact
/// preprocessing + [IMG_BREAK] row interleave) rather than a fixed tensor.
pub enum VisionTower {
    Clip(crate::inference::model::moondream::vision::MoondreamVisionEncoder),
    Pixtral(crate::inference::model::pixtral::PixtralVision),
}

/// A prompt whose audio placeholder is one stand-in token: where it is, and the tokens.
pub(super) struct AudioPrompt {
    pub tokens: Vec<u32>,
    pub at: usize,
}

/// The prompt as tokens with its audio placeholder replaced by the single token `stand_in`
/// (its embedding is never read: the clip's embeddings take its place at prefill).
///
/// The text is split around the placeholder and each side tokenized alone, the leading side
/// with the model's special tokens: the placeholder need not be a vocabulary entry. Refused
/// when the prompt and the embeddings do not fit the KV window.
pub(super) fn tokenize_around_audio(
    tokenizer: &Tokenizer,
    prompt: &str,
    placeholder: &str,
    stand_in: u32,
    params: &GenerationParams,
    embeds: usize,
    kv_window: usize,
) -> Result<AudioPrompt, String> {
    let (before, after) = prompt
        .split_once(placeholder)
        .ok_or_else(|| format!("the prompt holds no audio placeholder {placeholder:?}"))?;
    let head = tokenizer
        .encode(before, true)
        .map_err(|e| format!("Tokenization error: {e}"))?;
    let tail = tokenizer
        .encode(after, false)
        .map_err(|e| format!("Tokenization error: {e}"))?;
    let mut tokens = with_prefix_tokens(params, head.get_ids());
    let at = tokens.len();
    tokens.push(stand_in);
    tokens.extend_from_slice(tail.get_ids());
    let positions = tokens.len() - 1 + embeds;
    if positions >= kv_window {
        return Err(format!(
            "the prompt and the audio occupy {positions} KV positions but the context window is \
             only {kv_window}; increase context_length / num_ctx or send a shorter clip"
        ));
    }
    Ok(AudioPrompt { tokens, at })
}

/// The split of `prompt` around the audio the loaded model holds for this request, `None`
/// when there is none.
pub(super) fn audio_split_for(
    state: &LoadedModelState,
    prompt: &str,
    params: &GenerationParams,
    kv_window: usize,
) -> Result<Option<AudioPrompt>, String> {
    let (Some(clip), Some(input)) = (state.audio.as_ref(), state.model.audio_input()) else {
        return Ok(None);
    };
    let embeds = clip.embeds.dim(1).map_err(|e| e.to_string())?;
    tokenize_around_audio(
        &state.tokenizer,
        prompt,
        &input.placeholder,
        state.eos_token_id,
        params,
        embeds,
        kv_window,
    )
    .map(Some)
}

/// The positions a prefill evaluated: the prompt's tokens less those the cache held, plus the
/// positions a clip's embeddings take beyond the one placeholder token they replace.
pub(super) fn prompt_eval_positions(tokens: usize, reused: usize, audio_extra: usize) -> u64 {
    (tokens.saturating_sub(reused) + audio_extra) as u64
}

#[cfg(test)]
mod audio_position_tests {
    use super::*;

    /// A clip of 375 embeddings in a prompt of 20 tokens (the placeholder one of them) is 394
    /// positions; a prompt with no clip is its tokens less the reused ones.
    #[test]
    fn a_clip_counts_as_its_embeddings_less_the_placeholder() {
        assert_eq!(prompt_eval_positions(20, 0, 375 - 1), 394);
        assert_eq!(prompt_eval_positions(20, 5, 0), 15);
    }

    /// Splitting around the placeholder keeps every other token and one stand-in.
    #[test]
    fn the_prompt_splits_around_its_placeholder() {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        let vocab: ahash::AHashMap<String, u32> = ["[UNK]", "a", "b", "c", "PH"]
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        let params = GenerationParams::default();
        let split = tokenize_around_audio(&tok, "a b PH c", "PH", 99, &params, 10, 100).unwrap();
        assert_eq!(split.tokens, vec![1, 2, 99, 3]);
        assert_eq!(split.at, 2);
        // 4 tokens, one replaced by 10 embeddings.
        assert_eq!(prompt_eval_positions(split.tokens.len(), 0, 10 - 1), 13);
        // A prompt that does not fit the window is refused, one with no placeholder too.
        assert!(tokenize_around_audio(&tok, "a b PH c", "PH", 99, &params, 10, 13).is_err());
        assert!(tokenize_around_audio(&tok, "a b c", "PH", 99, &params, 10, 100).is_err());
    }
}
