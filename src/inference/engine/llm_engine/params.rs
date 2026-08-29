//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

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

pub(crate) fn is_cuda_oom<E: std::fmt::Display>(err: &E) -> bool {
    let s = err.to_string().to_ascii_lowercase();
    s.contains("out of memory")
        || s.contains("cuda_error_out_of_memory")
        // cuBLAS workspace / handle allocation failures surface with a
        // cuBLAS status rather than the driver OOM string; treat the
        // alloc-class ones as recoverable too.
        || s.contains("cublas_status_alloc_failed")
        || s.contains("alloc_failed")
}

pub(super) static RAYON_INIT: Once = Once::new();

/// Inference configuration for the language engine
#[derive(Clone)]
pub struct InferenceConfig {
    /// Model ID
    pub model_id: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Context length
    pub context_length: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p for sampling
    pub top_p: f32,
    /// Top-k for sampling
    pub top_k: usize,
    /// Seed for reproducibility
    pub seed: u64,
    /// Data type for model weights
    pub dtype: crate::tensor::DType,
    /// GPU device index
    pub device_index: Option<usize>,
    /// Max GPU memory fraction (0.0-1.0)
    pub max_gpu_memory_fraction: f64,
    /// Force GPU layers count
    pub force_gpu_layers: Option<usize>,
    /// Use quantized GPU
    pub use_quantized_gpu: bool,
    /// CPU threads (0 = auto)
    pub cpu_threads: usize,
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
    /// Default repetition penalty (1.0 = off, >1.0 = penalize)
    pub repeat_penalty: f32,
    /// Default number of recent tokens to consider for repeat penalty
    pub repeat_last_n: usize,
    /// KV-cache storage quantization. `Off` keeps K/V as the model dtype;
    /// `Q8` stores both as Q8_0 blocks (half VRAM, slight quality drift).
    /// Model-load-time only - changing this after load has no effect.
    pub kv_quant: KvQuant,
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
            .field("max_tokens", &self.max_tokens)
            .field("context_length", &self.context_length)
            .field("temperature", &self.temperature)
            .field("top_p", &self.top_p)
            .field("top_k", &self.top_k)
            .field("seed", &self.seed)
            .field("dtype", &self.dtype)
            .field("device_index", &self.device_index)
            .field("max_gpu_memory_fraction", &self.max_gpu_memory_fraction)
            .field("force_gpu_layers", &self.force_gpu_layers)
            .field("use_quantized_gpu", &self.use_quantized_gpu)
            .field("cpu_threads", &self.cpu_threads)
            .field("models_dir", &self.models_dir)
            .field("progress_callback", &"<closure>")
            .finish()
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            model_id: "default".to_string(),
            max_tokens: 2048,
            context_length: 4096,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
            seed: 42,
            dtype: crate::tensor::DType::F16,
            device_index: None,
            max_gpu_memory_fraction: 0.9,
            force_gpu_layers: None,
            use_quantized_gpu: true,
            cpu_threads: 0,
            progress_callback: None,
            models_dir: None,
            disable_arc_layers: false,
            disable_cuda: false,
            force_cuda_only_layers: false,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            kv_quant: KvQuant::Off,
        }
    }
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
