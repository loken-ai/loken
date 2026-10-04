//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

/// A norm layer that is either RMSNorm (most modern LLMs) or full
/// LayerNorm-with-bias (phi2, gpt-neox-style). Detected at load time
/// from `blk.0.attn_norm.bias` presence.
#[derive(Debug, Clone)]
pub enum WeightedNorm {
    Rms(RmsNorm),
    Layer(LayerNorm),
    /// Passthrough for POST-NORM-ONLY archs (OLMo2): the sub-layer sees the raw
    /// residual (no pre-norm), the normalisation is applied AFTER via
    /// `post_attn_norm`/`post_ffn_norm`. `forward` returns x unchanged; the held
    /// tensor is a ones-weight kept only so `weight()`/device probes still work.
    Identity(Tensor),
}

impl WeightedNorm {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Rms(n) => n.forward(x),
            Self::Layer(n) => n.forward(x),
            Self::Identity(_) => Ok(x.clone()),
        }
    }

    /// Underlying weight tensor - used by fused kernels that perform
    /// the norm inline. Returns the gamma weight for both variants.
    pub fn weight(&self) -> &Tensor {
        match self {
            Self::Rms(n) => n.weight(),
            Self::Layer(n) => n.weight(),
            Self::Identity(w) => w,
        }
    }

    /// Epsilon - used by the zero-alloc decode executor.
    pub fn eps(&self) -> f64 {
        match self {
            Self::Rms(n) => n.eps(),
            Self::Layer(n) => n.eps(),
            Self::Identity(_) => 0.0,
        }
    }

    /// Bias tensor - only present for LayerNorm variants (phi2/gpt-neox).
    /// RmsNorm has no bias; returns None.
    pub fn bias(&self) -> Option<&Tensor> {
        match self {
            Self::Rms(_) => None,
            Self::Layer(n) => n.bias(),
            Self::Identity(_) => None,
        }
    }

    /// Epsilon for the norm. F64 for LayerNorm, treated as F32-cast for
    /// callers that pass it to fused kernels.
    pub fn eps_f32(&self) -> f32 {
        match self {
            Self::Rms(n) => n.eps() as f32,
            Self::Layer(n) => n.eps() as f32,
            Self::Identity(_) => 0.0,
        }
    }
}
// ------------------------------���----------------------------------------------
// Architecture flags detected from GGUF tensor key presence
// ------------------------------------------------------------

/// Per-model architectural variant flags, detected at load time from GGUF.
#[derive(Debug, Clone)]
pub struct GenericLayerFlags {
    /// Qwen2: `blk.0.attn_q.bias` exists
    pub has_qkv_bias: bool,
    /// Gemma3: `blk.0.attn_q_norm.weight` exists
    pub has_qk_norm: bool,
    /// Gemma3: `blk.0.post_attention_norm.weight` exists
    pub has_post_attn_norm: bool,
    /// Gemma3: `blk.0.post_ffw_norm.weight` exists
    pub has_post_ffn_norm: bool,
    /// Phi3: `blk.0.attn_qkv.weight` exists (fused Q/K/V projection)
    pub fused_qkv: bool,
    /// Phi3: `blk.0.ffn_gate.weight` missing -> gate fused into `ffn_up.weight`
    pub fused_ffn_gate_up: bool,
    /// Phi2: simple FFN with no gate, just `y = down(GELU(up(x)))`.
    /// Distinguished from fused_ffn_gate_up (Phi3) by ffn_up's output dim
    /// equalling intermediate_size (vs 2x for Phi3).
    pub is_phi2_simple_ffn: bool,
    /// The FFN gates through GELU instead of SiLU.
    pub use_gelu: bool,
    /// The attention values are RMS-normalised without a weight before caching.
    pub value_rms_norm: bool,
    /// Llama/Mistral: use interleaved RoPE (rope_i) vs non-interleaved (rope)
    pub use_rope_i: bool,
    /// Number of intermediate neurons (for Phi3 fused FFN split)
    /// For standard models: equals `ffn_dim`. For Phi3: `ffn_up.weight` rows / 2.
    pub intermediate_size: usize,
    /// Phi2: parallel attention. attn and ffn both run on the same
    /// post-attn_norm input; their outputs are summed into the residual.
    /// Detected by absence of a separate `blk.0.ffn_norm.weight` tensor.
    pub parallel_attn: bool,
    /// Phi2 / NeoX: input layer norm has both `.weight` and `.bias`  -
    /// it's a full LayerNorm (subtract mean, divide std, scale, shift),
    /// NOT an RMSNorm. Detected by `blk.0.attn_norm.bias` presence.
    pub layer_norm_with_bias: bool,
    /// Phi2: attn_qkv has `.bias`, attn_output has `.bias`, ffn_up has
    /// `.bias`, ffn_down has `.bias`. Existing `has_qkv_bias` is for
    /// Qwen2's per-projection biases (q/k/v) - phi2 has them on the
    /// FUSED qkv tensor.
    pub has_attn_output_bias: bool,
    pub has_ffn_bias: bool,
    /// A learned per-head logit joins each attention softmax denominator and absorbs
    /// probability mass without contributing a value: `blk.0.attn_sinks` exists.
    pub has_attn_sinks: bool,
    /// Gemma4-MoE (26B A4B): every layer has BOTH a shared dense FFN
    /// AND a 128-expert MoE block. Detected by
    /// `blk.0.ffn_gate_inp.weight`.
    pub is_moe: bool,
    /// MoE expert count from `<arch>.expert_count` metadata.
    pub n_experts: usize,
    /// Top-K experts used per token from `<arch>.expert_used_count`.
    pub n_experts_used: usize,
    /// Per-expert FFN width from `<arch>.expert_feed_forward_length`.
    pub expert_ffn_dim: usize,
    /// Gemma4 26B Global layers: `attn_v.weight` is absent - V is
    /// reused from K (AttentionKEqV pattern). Detected at load time
    /// per-layer.
    pub has_pre_ffw_norm_2: bool,
    /// Gemma4-MoE has a separate `post_ffw_norm_1` after the shared
    /// dense FFN and a `post_ffw_norm_2` after the MoE block.
    pub has_post_ffw_norm_split: bool,
    /// True when SOME layers have `attn_v.weight` and OTHERS don't
    /// (e.g., gemma4 26B mixes SWA layers with V and Global layers
    /// without V via AttentionKEqV). Disables the load-time QKV
    /// byte-concat fusion globally because the fusion path narrows
    /// a non-contiguous fused output that produced non-deterministic
    /// runtime behavior on this layer mix.
    pub mixed_attention_keqv: bool,
    /// (Phase B): fuse attn_norm + attn_q/k/v via the
    /// the fused rms_norm_then_qmatmul_bf16 kernel.
    /// Currently kept at `false` pending broader validation; the kernel
    /// is in tree but not on the auto-detect path.
    pub fused_qkv_norm_eligible: bool,
}

impl GenericLayerFlags {
    fn detect(content: &gguf_file::Content, ffn_dim: usize, arch: &str) -> Self {
        let has = |key: &str| content.tensor_infos.contains_key(key);
        let meta_u32 = |k: &str| -> Option<u32> {
            content
                .metadata
                .get(&format!("{arch}.{k}"))
                .and_then(|v| v.to_u32().ok())
        };
        let n_layer = meta_u32("block_count").unwrap_or(0) as usize;
        // The attention variant is read on the first block that attends: a hybrid's block 0
        // may run a recurrent mixer.
        let attn = super::mixer::attention_probe_layer(content, n_layer);
        let has_attn = |suffix: &str| has(&format!("blk.{attn}.{suffix}"));
        let ffn_norm = ffn_norm_stem(content);
        let facts = crate::inference::model_facts::facts_or_default(arch);
        Self {
            has_qkv_bias: has_attn("attn_q.bias"),
            has_qk_norm: has_attn("attn_q_norm.weight"),
            // A file whose feed-forward norm carries this name has no post-attention norm.
            has_post_attn_norm: ffn_norm != "post_attention_norm"
                && has("blk.0.post_attention_norm.weight"),
            has_post_ffn_norm: has("blk.0.post_ffw_norm.weight"),
            fused_qkv: has_attn("attn_qkv.weight"),
            fused_ffn_gate_up: !has("blk.0.ffn_gate.weight") && has("blk.0.ffn_up.weight"),
            // Phi2 simple FFN: ffn_up output dim equals ffn_dim (not 2x).
            // Detect by inspecting ffn_up's actual shape against ffn_dim.
            is_phi2_simple_ffn: {
                if has("blk.0.ffn_gate.weight") || !has("blk.0.ffn_up.weight") {
                    false
                } else {
                    content
                        .tensor_infos
                        .get("blk.0.ffn_up.weight")
                        .map(|info| {
                            // The substrate stores `[output, input]`. For Phi2's
                            // simple linear `up: hidden->ffn_dim`, output is
                            // ffn_dim. For Phi3's fused gate||up, output is
                            // 2*ffn_dim.
                            info.shape.dims().first().copied().unwrap_or(0) == ffn_dim
                        })
                        .unwrap_or(false)
                }
            },
            // A GGUF does not record the activation: it follows the architecture's facts, and
            // a per-layer output scale marks a GELU-gated build under any architecture key.
            use_gelu: facts.gelu_ffn || has("blk.0.layer_output_scale.weight"),
            // The same marker stands for the value norm of that build.
            value_rms_norm: facts.value_rms_norm || has("blk.0.layer_output_scale.weight"),
            // parallel-attn (phi2/gpt-neox): one input norm feeds both attn+ffn, so
            // there is no separate ffn_norm. OLMo2 ALSO lacks ffn_norm but is SERIAL
            // post-norm (it has post_ffw_norm) - exclude it so it isn't mis-routed.
            parallel_attn: !has(&format!("blk.0.{ffn_norm}.weight"))
                && has("blk.0.ffn_up.weight")
                && !has("blk.0.post_ffw_norm.weight"),
            layer_norm_with_bias: has("blk.0.attn_norm.bias"),
            has_attn_output_bias: has_attn(&format!("{}.bias", attn_output_name(content))),
            has_ffn_bias: has("blk.0.ffn_down.bias"),
            has_attn_sinks: attn_sinks_name(content, attn).is_some(),
            // The pairing is a fact of the architecture the file does not carry.
            use_rope_i: rope_uses_adjacent_pairs(arch),
            intermediate_size: ffn_dim,
            // A mixture may keep dense blocks ahead of its routed ones.
            is_moe: (0..n_layer.max(1)).any(|il| has(&format!("blk.{il}.ffn_gate_inp.weight"))),
            n_experts: meta_u32("expert_count").unwrap_or(0) as usize,
            n_experts_used: meta_u32("expert_used_count").unwrap_or(0) as usize,
            expert_ffn_dim: meta_u32("expert_feed_forward_length").unwrap_or(0) as usize,
            has_pre_ffw_norm_2: has("blk.0.pre_ffw_norm_2.weight"),
            has_post_ffw_norm_split: has("blk.0.post_ffw_norm_1.weight")
                && has("blk.0.post_ffw_norm_2.weight"),
            // Mixed AttentionKEqV: some layers have attn_v, some don't,
            // AND the model isn't using shared_kv_layers (which is a
            // different reuse mechanism - donor layer holds K/V).
            //
            // gemma4 26B: shared_kv_layers=0, mixed attn_v present
            //   -> AttentionKEqV (V=K within same layer for Global layers).
            //   The MoE+attention combination triggers a non-deterministic
            //   path; disable QKV byte-concat fusion + FA prefill for them.
            //
            // gemma4 8B: shared_kv_layers=18, also has missing attn_v on the
            //   shared layers - but that's a DIFFERENT mechanism handled by
            //   the donor-KV path. Don't flag it as mixed_attention_keqv.
            mixed_attention_keqv: {
                let shared_kv_layers = meta_u32("attention.shared_kv_layers").unwrap_or(0) as usize;
                if shared_kv_layers > 0 {
                    false
                } else if has("blk.0.attn_v.weight") {
                    (1..n_layer).any(|i| !has(&format!("blk.{}.attn_v.weight", i)))
                } else {
                    false
                }
            },
            // Phase B: qwen2-only, separate Q/K/V, no QK-norm,
            // no fused QKV, no LayerNorm-with-bias (kernel is RMS-only).
            // Default-off pending broader validation across qwen2 sizes;
            // the kernel exists but isn't dispatched on the auto-path.
            fused_qkv_norm_eligible: false,
        }
    }
}

/// Stem of the attention output projection: `attn_output` in the reference naming,
/// `attn_out` where the converter wrote the shorter form. Read from the file, never
/// assumed from the architecture.
pub(super) fn attn_output_name(content: &gguf_file::Content) -> &'static str {
    let il = content
        .metadata
        .get("general.architecture")
        .and_then(|a| a.to_string().ok())
        .and_then(|a| meta_u32(content, a, "block_count"))
        .map_or(0, |n| {
            super::mixer::attention_probe_layer(content, n as usize)
        });
    if content
        .tensor_infos
        .contains_key(&format!("blk.{il}.attn_output.weight"))
    {
        "attn_output"
    } else if content
        .tensor_infos
        .contains_key(&format!("blk.{il}.attn_out.weight"))
    {
        "attn_out"
    } else {
        "attn_output"
    }
}

/// Name of layer `il`'s attention-sink tensor, `None` when the layer has none. The
/// reference stores it without a suffix; a `.weight` suffix is accepted as the same tensor.
pub(super) fn attn_sinks_name(content: &gguf_file::Content, il: usize) -> Option<String> {
    [
        format!("blk.{il}.attn_sinks"),
        format!("blk.{il}.attn_sinks.weight"),
    ]
    .into_iter()
    .find(|n| content.tensor_infos.contains_key(n))
}

/// Name of the norm ahead of the feed-forward. A block with neither `ffn_norm` nor a norm
/// after its feed-forward, but with one after attention, normalises the feed-forward's input
/// with that `post_attention_norm`.
pub(super) fn ffn_norm_stem(content: &gguf_file::Content) -> &'static str {
    let has = |key: &str| content.tensor_infos.contains_key(key);
    if !has("blk.0.ffn_norm.weight")
        && !has("blk.0.post_ffw_norm.weight")
        && has("blk.0.post_attention_norm.weight")
    {
        "post_attention_norm"
    } else {
        "ffn_norm"
    }
}

/// Whether the attention Q projection interleaves Q and an output gate per head: on the
/// first attending block it is twice as wide as the heads it projects and twice as wide as
/// what the output projection reads.
fn q_carries_output_gate(
    content: &gguf_file::Content,
    n_layers: usize,
    n_head: usize,
    head_dim: usize,
) -> bool {
    let il = super::mixer::attention_probe_layer(content, n_layers);
    let dim = |name: &str, axis: usize| {
        content
            .tensor_infos
            .get(&format!("blk.{il}.{name}.weight"))
            .and_then(|t| t.shape.dims().get(axis).copied())
    };
    match (dim("attn_q", 0), dim(attn_output_name(content), 1)) {
        (Some(q), Some(o)) => q == 2 * o && q == 2 * n_head * head_dim,
        _ => false,
    }
}

/// Whether the file's `v.*` tower splices its embeddings into the text stream: a patch
/// embedding and rotary sections over several position axes.
pub fn splices_vision_tower(content: &gguf_file::Content, arch: &str) -> bool {
    let key = |k: &str| content.metadata.contains_key(&format!("{arch}.{k}"));
    content.tensor_infos.contains_key("v.patch_embed.weight")
        && (key("rope.dimension_sections") || key("rope.mrope_section") || key("mrope_sections"))
}

/// Whether layer `il` is a sliding-window layer under a period of `period`.
pub fn windowed_by_period(period: usize, il: usize) -> bool {
    period == 0 || il % period < period - 1
}

/// Whether the file is one part of a set of two or more: `split.count` names the set and
/// `split.no` the part. A whole file carries neither.
pub fn is_a_split_part(content: &gguf_file::Content) -> bool {
    content
        .metadata
        .get("split.count")
        .and_then(|v| v.to_u32().ok())
        .is_some_and(|count| count > 1)
}

/// Refuses a GGUF that is one part of a split set on a route that reads a single file.
///
/// A part lists only the tensors it carries, and a block with no mixer and no feed-forward is
/// built as a pass-through, so the load would succeed with part of the stack missing. A set is
/// read as one source by `GenericHeteroTransformer::from_split_gguf`.
pub fn refuse_a_split_part(content: &gguf_file::Content) -> crate::tensor::Result<()> {
    if !is_a_split_part(content) {
        return Ok(());
    }
    let u = |k: &str| content.metadata.get(k).and_then(|v| v.to_u32().ok());
    let count = u("split.count").unwrap_or(1);
    let part = u("split.no").map_or(1, |n| n + 1);
    Err(crate::tensor::Error::msg(format!(
        "this file is part {part} of a {count}-part split GGUF, and this route reads one file: \
         the blocks the other parts carry would be built as pass-through placeholders. Open the \
         set by its first part, or merge the parts into one file."
    )))
}

// ------------------------------------------------------------
// Configuration read from GGUF metadata
// ------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GenericTransformerConfig {
    pub arch: String,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub embedding_length: usize,
    pub ffn_dim: usize,
    pub rope_freq_base: f32,
    /// Gemma4: SWA layers use a different (lower) RoPE frequency base
    pub rope_freq_base_swa: Option<f32>,
    /// Number of dimensions RoPE applies to (partial RoPE: StableLM uses 16 of 64).
    /// None or 0 = full head_dim.
    pub rope_dim: Option<usize>,
    /// YaRN rope context-extension scaling (mistral3 devstral-small-2, deepseek).
    /// When Some, the rope tables interpolate inverse-freqs per the YaRN ramp.
    pub yarn: Option<crate::inference::generic_transformer::rope::YarnParams>,
    pub rms_norm_eps: f64,
    pub sliding_window: Option<usize>,
    /// Number of layers at the end that share KV cache from earlier layers (Gemma4: 20)
    pub shared_kv_layers: usize,
    /// Multiply embeddings after lookup (Gemma: √embed_len, Granite: explicit value)
    pub embed_scale: Option<f64>,
    /// Divide final logits by this value (Granite: 8.0)
    pub logit_scale: Option<f64>,
    /// Custom attention scale replacing 1/√head_dim (Granite: 0.015625)
    pub attention_scale: Option<f64>,
    /// Residual connection multiplier (Granite: 0.22)
    pub residual_scale: Option<f64>,
    /// Gemma4: final logit softcapping -> tanh(logits/cap)*cap
    pub final_logit_softcapping: Option<f64>,
    /// Gemma2: attention score softcapping -> tanh(scores/cap)*cap, before the mask.
    pub attn_logit_softcapping: Option<f64>,
    /// Per-layer sliding-window pattern (gemma4 `attention.sliding_window_pattern`,
    /// 1=SWA / 0=global). Authoritative SWA classification - preferred over the
    /// head_dim heuristic, which mis-classifies KV-shared layers whose Q dim does
    /// not encode their head size. None ⟹ fall back to the head_dim heuristic.
    pub swa_pattern: Option<Vec<bool>>,
    /// Sliding-window layers classified by the architecture's period when the file
    /// carries no per-layer pattern. Such a model keeps every position in each layer's
    /// cache and gives each layer the prefill mask of its own window: a global layer
    /// sees the whole causal history.
    pub swa_period: Option<usize>,
    /// Routed experts gate through the clamped SwiGLU, `(alpha, limit)`.
    pub expert_swiglu_clamp: Option<(f64, f64)>,
    /// The router scores experts by sigmoid: `expert_gating_func` 2 in the file, or the
    /// architecture's convention.
    pub expert_sigmoid_gating: bool,
    /// Sigmoid-routed weights are renormalised over the chosen experts.
    pub expert_weights_norm: bool,
    /// Softmax-routed weights are renormalised over the chosen experts: the file's
    /// `norm_top_k_prob` or `expert_weights_norm`, else the architecture's fact.
    pub expert_topk_norm: bool,
    /// Multiplies sigmoid-routed weights; 1 when the file gives none or zero.
    pub expert_weights_scale: f64,
    /// The recurrent mixers' dimensions, when the file gives `ssm.inner_size`.
    pub ssm: Option<SsmMeta>,
    /// The delta rule gates its output norm by `sigmoid(z)` rather than `silu(z)`.
    pub delta_output_gate_sigmoid: bool,
    /// The wide residual, when the file declares parallel streams.
    pub hyper: Option<super::hyper::HyperMeta>,
    /// The block-sparse attention indexer, when the file declares one.
    pub indexer: Option<super::qsa::IndexerMeta>,
    /// The hashed n-gram embedding, when the file declares one.
    pub ngram: Option<super::ngram::NgramMeta>,
    /// The attention Q projection interleaves Q and an output gate per head.
    pub attn_output_gate: bool,
    /// Name of the norm ahead of the feed-forward.
    pub ffn_norm_stem: &'static str,
    pub flags: GenericLayerFlags,
    /// Per-layer KV cache storage mode. Off -> F-dtype SpecKvCache only;
    /// Q8 -> also maintain a Q8KvCache alongside so the decode-path (seq=1)
    /// attention can run against Q8 K/V. Prefill stays on the F16 cache.
    pub kv_quant: crate::inference::engine::llm_engine::KvQuant,
    /// Preallocation size for the Q8 KV cache (K+V per layer per kv-head
    /// at max_q8_seq_len x head_dim bytes). Defaults to 8192 - call
    /// `with_max_q8_seq_len` at config construction to raise for long-ctx.
    pub max_q8_seq_len: usize,
}

/// The `ssm.*` metadata of a model with state-space or delta-rule mixers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmMeta {
    /// Taps of the causal convolution ahead of the recurrence.
    pub conv_kernel: usize,
    /// Width of the recurrent state per head (Mamba2 `d_state`, the delta rule's key
    /// head width).
    pub state_size: usize,
    /// Groups sharing `B`/`C` (Mamba2), key heads (delta rule).
    pub group_count: usize,
    /// Heads (Mamba2 `dt` rank), value heads (delta rule).
    pub time_step_rank: usize,
    /// Channels the recurrence runs over.
    pub inner_size: usize,
    /// The delta rule's value head `h` reads key head `h % group_count` rather than
    /// `h / (time_step_rank / group_count)`.
    pub v_heads_tiled: bool,
}

/// Panicking wrapper for the config builder, which has no error path here. The generic
/// route refuses an architecture without a pairing before it builds a config, so this only
/// fires for a caller that skipped that check.
fn rope_uses_adjacent_pairs(arch: &str) -> bool {
    match crate::inference::model_facts::rope_pairing(arch) {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    }
}

impl GenericTransformerConfig {
    /// Max seq_len used to pre-allocate the Q8 KV cache. Capped to a
    /// reasonable ceiling so very long-ctx models don't blow VRAM at
    /// load time - see the warning in `build_generic_layer`.
    pub fn effective_max_context(&self) -> usize {
        self.max_q8_seq_len
    }
}

/// Read a u32 from GGUF metadata, trying arch-prefixed key then bare key.
pub(super) fn meta_u32(ct: &gguf_file::Content, arch: &str, key: &str) -> Option<u32> {
    ct.metadata
        .get(&format!("{arch}.{key}"))
        .or_else(|| ct.metadata.get(key))
        .and_then(|v| v.to_u32().ok())
}

fn meta_f32(ct: &gguf_file::Content, arch: &str, key: &str) -> Option<f32> {
    ct.metadata
        .get(&format!("{arch}.{key}"))
        .or_else(|| ct.metadata.get(key))
        .and_then(|v| v.to_f32().ok())
}

/// Read a GGUF array-of-int metadata key as `Vec<u64>`; `None` if absent, not an array, or
/// holding a value that is not a non-negative whole number.
pub(super) fn meta_u64_array(ct: &gguf_file::Content, arch: &str, key: &str) -> Option<Vec<u64>> {
    let v = ct
        .metadata
        .get(&format!("{arch}.{key}"))
        .or_else(|| ct.metadata.get(key))?;
    v.to_vec().ok()?.iter().map(|e| e.to_u64().ok()).collect()
}

/// Read a GGUF array-of-int metadata key as `Vec<u32>` (e.g. gemma4's
/// `attention.sliding_window_pattern` = per-layer 1=SWA / 0=global). Returns
/// None if absent or not an array.
pub(super) fn meta_u32_array(ct: &gguf_file::Content, arch: &str, key: &str) -> Option<Vec<u32>> {
    use gguf_file::Value as V;
    let elem_u32 = |e: &V| -> Option<u32> {
        match e {
            V::U8(x) => Some(*x as u32),
            V::U16(x) => Some(*x as u32),
            V::U32(x) => Some(*x),
            V::U64(x) => Some(*x as u32),
            V::I8(x) => Some(*x as u32),
            V::I16(x) => Some(*x as u32),
            V::I32(x) => Some(*x as u32),
            V::I64(x) => Some(*x as u32),
            V::Bool(b) => Some(*b as u32),
            _ => None,
        }
    };
    let v = ct
        .metadata
        .get(&format!("{arch}.{key}"))
        .or_else(|| ct.metadata.get(key))?;
    match v {
        V::Array(items) => Some(items.iter().filter_map(elem_u32).collect()),
        _ => None,
    }
}

impl GenericTransformerConfig {
    pub fn from_gguf(content: &gguf_file::Content, arch: &str) -> crate::tensor::Result<Self> {
        let get_u32 = |key: &str| -> crate::tensor::Result<usize> {
            meta_u32(content, arch, key)
                .map(|v| v as usize)
                .ok_or_else(|| crate::tensor::Error::msg(format!("missing GGUF key {arch}.{key}")))
        };

        let n_head = get_u32("attention.head_count")?;
        // GQA: n_kv_head < n_head. Models without GQA (StableLM) omit this key.
        let n_kv_head = meta_u32(content, arch, "attention.head_count_kv")
            .map(|v| v as usize)
            .unwrap_or(n_head);
        let n_layers = get_u32("block_count")?;

        // head_dim: try explicit key first (Gemma3/Mistral3), else compute
        let head_dim = meta_u32(content, arch, "attention.key_length")
            .map(|v| v as usize)
            .or_else(|| meta_u32(content, arch, "embedding_length").map(|e| e as usize / n_head))
            .unwrap_or_else(|| {
                warn!("Could not determine head_dim for {arch}, using 64 fallback");
                64
            });

        let embedding_length = meta_u32(content, arch, "embedding_length")
            .unwrap_or((n_head * head_dim) as u32) as usize;

        // The width the dense feed-forward tensors have. The declared value stands when some
        // layer has that width; a file whose converter rounds the width after writing it
        // declares one no layer has, and the tensors are then what says.
        let declared_ffn = meta_u32(content, arch, "feed_forward_length").map(|v| v as usize);
        let mut widths = std::collections::BTreeMap::<usize, usize>::new();
        for i in 0..n_layers {
            if let Some(w) = content
                .tensor_infos
                .get(&format!("blk.{i}.ffn_gate.weight"))
                .and_then(|t| t.shape.dims().first().copied())
            {
                *widths.entry(w).or_default() += 1;
            }
        }
        let ffn_dim = match declared_ffn {
            Some(d) if widths.is_empty() || widths.contains_key(&d) => d,
            _ => widths
                .iter()
                .max_by_key(|(_, n)| **n)
                .map(|(w, _)| *w)
                .or(declared_ffn)
                .unwrap_or(4 * embedding_length),
        };

        let rope_freq_base = meta_f32(content, arch, "rope.freq_base").unwrap_or(10000.0);

        // Gemma4: separate RoPE freq for SWA layers (lower base -> shorter context)
        let rope_freq_base_swa = meta_f32(content, arch, "rope.freq_base_swa")
            .or_else(|| meta_f32(content, arch, "rope.freq_base_local"));

        // Partial RoPE: some models (StableLM) only apply RoPE to a subset of head dims
        let rope_dim = meta_u32(content, arch, "rope.dimension_count").map(|v| v as usize);

        // YaRN rope scaling (devstral-small-2: type=yarn, factor=48, orig_ctx=8192).
        // Without it the rope frequencies are wrong past the original context and the
        // model emits garbage. The parameters go to `rope::precomput_freqs_cis_yarn`, which
        // is the only place these tables are built.
        let facts = crate::inference::model_facts::facts_or_default(arch);
        let yarn = {
            let scaling_type = content
                .metadata
                .get(&format!("{arch}.rope.scaling.type"))
                .and_then(|v| v.to_string().ok().map(|s| s.to_string()))
                .or_else(|| facts.yarn_by_default.then(|| "yarn".to_string()));
            match scaling_type.as_deref() {
                Some("yarn") => {
                    let factor = meta_f32(content, arch, "rope.scaling.factor").unwrap_or(1.0);
                    let orig_ctx = meta_u32(content, arch, "rope.scaling.original_context_length")
                        .map(|v| v as f32)
                        .unwrap_or(8192.0);
                    if factor > 1.0 {
                        info!("📐 {arch} YaRN rope: factor={factor}, orig_ctx={orig_ctx}");
                        Some(crate::inference::generic_transformer::rope::YarnParams {
                            factor,
                            orig_ctx,
                            beta_fast: meta_f32(content, arch, "rope.scaling.beta_fast")
                                .unwrap_or(32.0),
                            beta_slow: meta_f32(content, arch, "rope.scaling.beta_slow")
                                .unwrap_or(1.0),
                            attn_factor: if facts.yarn_attention_factor {
                                1.0 + 0.1 * factor.ln()
                            } else {
                                1.0
                            },
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            }
        };

        let rms_norm_eps = meta_f32(content, arch, "attention.layer_norm_rms_epsilon")
            .or_else(|| meta_f32(content, arch, "attention.layer_norm_epsilon"))
            .unwrap_or(1e-6) as f64;

        let sliding_window =
            meta_u32(content, arch, "attention.sliding_window").map(|v| v as usize);

        // gemma4 ships an explicit per-layer SWA mask (1=SWA, 0=global). This is
        // the authoritative classification (llama.cpp `is_swa_impl`); the head_dim
        // heuristic mis-classifies the KV-shared tail layers.
        let swa_pattern = meta_u32_array(content, arch, "attention.sliding_window_pattern")
            .map(|arr| arr.iter().map(|&x| x != 0).collect::<Vec<bool>>())
            .filter(|v| !v.is_empty());
        let swa_period = facts
            .swa_period
            .filter(|_| swa_pattern.is_none() && sliding_window.is_some());

        let shared_kv_layers =
            meta_u32(content, arch, "attention.shared_kv_layers").unwrap_or(0) as usize;

        // llama.cpp's gating function numbering: 1 softmax, 2 sigmoid.
        const GATING_SIGMOID: u32 = 2;
        let expert_sigmoid_gating = match meta_u32(content, arch, "expert_gating_func") {
            Some(f) => f == GATING_SIGMOID,
            None => facts.sigmoid_expert_gating,
        };
        let expert_weights_norm = facts.sigmoid_weights_norm.unwrap_or_else(|| {
            content
                .metadata
                .get(&format!("{arch}.expert_weights_norm"))
                .and_then(|v| v.to_bool().ok())
                .unwrap_or(false)
        });
        let meta_bool = |key: &str| {
            content
                .metadata
                .get(&format!("{arch}.{key}"))
                .and_then(|v| v.to_bool().ok())
        };
        let expert_topk_norm = meta_bool("norm_top_k_prob")
            .or_else(|| meta_bool("expert_weights_norm"))
            .unwrap_or(facts.softmax_topk_renormalised);
        let ssm = meta_u32(content, arch, "ssm.inner_size").map(|inner| SsmMeta {
            conv_kernel: meta_u32(content, arch, "ssm.conv_kernel").unwrap_or(0) as usize,
            state_size: meta_u32(content, arch, "ssm.state_size").unwrap_or(0) as usize,
            group_count: meta_u32(content, arch, "ssm.group_count").unwrap_or(1) as usize,
            time_step_rank: meta_u32(content, arch, "ssm.time_step_rank").unwrap_or(0) as usize,
            inner_size: inner as usize,
            v_heads_tiled: content
                .metadata
                .get(&format!("{arch}.ssm.v_head_reordered"))
                .and_then(|v| v.to_bool().ok())
                .unwrap_or(facts.delta_v_heads_tiled),
        });
        let expert_weights_scale = match meta_f32(content, arch, "expert_weights_scale") {
            Some(s) if s != 0.0 => s as f64,
            _ => 1.0,
        };

        // vocab_size: read from embedding tensor shape (most reliable)
        let vocab_size = content
            .tensor_infos
            .get("token_embd.weight")
            .map(|ti| ti.shape.dims()[0])
            .unwrap_or_else(|| meta_u32(content, arch, "vocab_size").unwrap_or(32000) as usize);

        let flags = GenericLayerFlags::detect(content, ffn_dim, arch);

        // Embedding scaling: the file's value, else the square root of the width when the
        // architecture scales by it.
        let embed_scale = meta_f32(content, arch, "embedding_scale")
            .map(|v| v as f64)
            .or_else(|| {
                facts
                    .embed_scale_sqrt_width
                    .then(|| (embedding_length as f64).sqrt())
            });

        // Logit scaling: Granite divides logits by this value
        let logit_scale = meta_f32(content, arch, "logit_scale").map(|v| v as f64);

        // Attention scale replacing 1/sqrt(head_dim): the file's value, else the
        // architecture's.
        let attention_scale = meta_f32(content, arch, "attention.scale")
            .map(|v| v as f64)
            .or(facts.attention_scale);

        // Residual connection multiplier (Granite: 0.22)
        let residual_scale = meta_f32(content, arch, "residual_scale").map(|v| v as f64);

        // Attention score softcapping (Gemma2: 50.0): tanh(scores/cap)*cap before the mask.
        let attn_logit_softcapping =
            meta_f32(content, arch, "attn_logit_softcapping").map(|v| v as f64);

        // Logit softcapping (Gemma4: 30.0)
        let final_logit_softcapping =
            meta_f32(content, arch, "final_logit_softcapping").map(|v| v as f64);

        info!("GenericTransformerConfig [{arch}]: n_head={n_head}, n_kv_head={n_kv_head}, \
               head_dim={head_dim}, n_layers={n_layers}, embedding={embedding_length}, \
               ffn_dim={ffn_dim}, rope_base={rope_freq_base}, embed_scale={:?}, \
               logit_scale={:?}, attention_scale={:?}, residual_scale={:?}, qkv_bias={}, qk_norm={}, post_norms={}, fused_qkv={}, fused_ffn={}, \
               parallel_attn={}, is_phi2_simple_ffn={}, has_attn_output_bias={}, has_ffn_bias={}",
            embed_scale, logit_scale, attention_scale, residual_scale,
            flags.has_qkv_bias, flags.has_qk_norm, flags.has_post_attn_norm,
            flags.fused_qkv, flags.fused_ffn_gate_up,
            flags.parallel_attn, flags.is_phi2_simple_ffn,
            flags.has_attn_output_bias, flags.has_ffn_bias);

        Ok(Self {
            arch: arch.to_string(),
            n_head,
            n_kv_head,
            head_dim,
            n_layers,
            vocab_size,
            embedding_length,
            ffn_dim,
            rope_freq_base,
            rope_freq_base_swa,
            rope_dim,
            yarn,
            rms_norm_eps,
            sliding_window,
            swa_pattern,
            swa_period,
            expert_swiglu_clamp: facts.expert_swiglu_clamp,
            expert_sigmoid_gating,
            expert_weights_norm,
            expert_topk_norm,
            expert_weights_scale,
            ssm,
            delta_output_gate_sigmoid: facts.delta_output_gate_sigmoid,
            hyper: super::hyper::HyperMeta::from_gguf(content, arch),
            indexer: super::qsa::IndexerMeta::from_gguf(content, arch),
            ngram: super::ngram::NgramMeta::from_gguf(content, arch),
            attn_output_gate: q_carries_output_gate(content, n_layers, n_head, head_dim),
            ffn_norm_stem: ffn_norm_stem(content),
            shared_kv_layers,
            embed_scale,
            logit_scale,
            attention_scale,
            residual_scale,
            final_logit_softcapping,
            attn_logit_softcapping,
            flags,
            kv_quant: crate::inference::engine::llm_engine::KvQuant::Off,
            max_q8_seq_len: 8192,
        })
    }
}

#[cfg(test)]
mod arch_facts_tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::{Content, TensorInfo, Value, VersionedMagic};
    use crate::tensor::quantized::GgmlDType;

    /// A header-only file: the metadata `arch` writes and the tensor directory the
    /// detection reads, with no tensor data.
    fn synth_content(
        arch: &str,
        meta: &[(&str, Value)],
        tensors: &[(&str, Vec<usize>)],
    ) -> Content {
        let mut metadata: HashMap<String, Value> = HashMap::new();
        metadata.insert(
            "general.architecture".into(),
            Value::String(arch.to_string()),
        );
        for (k, v) in meta {
            metadata.insert(format!("{arch}.{k}"), v.clone());
        }
        let tensor_infos = tensors
            .iter()
            .map(|(name, dims)| {
                (
                    name.to_string(),
                    TensorInfo {
                        ggml_dtype: GgmlDType::F32,
                        shape: crate::tensor::Shape::from(dims.clone()),
                        offset: 0,
                    },
                )
            })
            .collect();
        Content {
            magic: VersionedMagic::GgufV3,
            metadata,
            tensor_infos,
            tensor_data_offset: 0,
            mmap_owner: None,
        }
    }

    /// The dimensions of a small gpt-oss-shaped model: 4 query heads over 2 key/value
    /// heads of 8, 4 layers, 4 experts of which 2 are used, a window of 4.
    fn gptoss_like(arch: &str, rope_type: Option<&str>) -> Content {
        let (hidden, heads, kv_heads, hd, experts) = (32usize, 4usize, 2usize, 8usize, 4usize);
        let mut meta = vec![
            ("attention.head_count", Value::U32(heads as u32)),
            ("attention.head_count_kv", Value::U32(kv_heads as u32)),
            ("attention.key_length", Value::U32(hd as u32)),
            ("attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("attention.sliding_window", Value::U32(4)),
            ("block_count", Value::U32(4)),
            ("embedding_length", Value::U32(hidden as u32)),
            ("feed_forward_length", Value::U32(hidden as u32)),
            ("expert_count", Value::U32(experts as u32)),
            ("expert_used_count", Value::U32(2)),
            ("rope.freq_base", Value::F32(150000.0)),
            ("rope.scaling.factor", Value::F32(32.0)),
            ("rope.scaling.original_context_length", Value::U32(4096)),
        ];
        if let Some(t) = rope_type {
            meta.push(("rope.scaling.type", Value::String(t.to_string())));
        }
        let tensors = vec![
            ("token_embd.weight", vec![100, hidden]),
            ("blk.0.attn_norm.weight", vec![hidden]),
            ("blk.0.attn_q.weight", vec![heads * hd, hidden]),
            ("blk.0.attn_q.bias", vec![heads * hd]),
            ("blk.0.attn_k.weight", vec![kv_heads * hd, hidden]),
            ("blk.0.attn_k.bias", vec![kv_heads * hd]),
            ("blk.0.attn_v.weight", vec![kv_heads * hd, hidden]),
            ("blk.0.attn_v.bias", vec![kv_heads * hd]),
            ("blk.0.attn_out.weight", vec![hidden, heads * hd]),
            ("blk.0.attn_out.bias", vec![hidden]),
            ("blk.0.attn_sinks", vec![heads]),
            ("blk.0.ffn_norm.weight", vec![hidden]),
            ("blk.0.ffn_gate_inp.weight", vec![experts, hidden]),
            ("blk.0.ffn_gate_inp.bias", vec![experts]),
            ("blk.0.ffn_gate_exps.weight", vec![experts, hidden, hidden]),
            ("blk.0.ffn_up_exps.weight", vec![experts, hidden, hidden]),
            ("blk.0.ffn_down_exps.weight", vec![experts, hidden, hidden]),
        ];
        synth_content(arch, &meta, &tensors)
    }

    #[test]
    fn a_gptoss_file_configures_sinks_windows_yarn_and_the_clamped_experts() {
        let content = gptoss_like("gptoss", None);
        let cfg = GenericTransformerConfig::from_gguf(&content, "gptoss").unwrap();
        assert!(cfg.flags.has_attn_sinks);
        assert!(cfg.flags.has_qkv_bias);
        assert!(
            cfg.flags.has_attn_output_bias,
            "attn_out.bias names the output bias"
        );
        assert!(cfg.flags.is_moe);
        assert_eq!(attn_output_name(&content), "attn_out");
        assert_eq!(
            attn_sinks_name(&content, 0).as_deref(),
            Some("blk.0.attn_sinks")
        );
        assert_eq!(cfg.sliding_window, Some(4));
        assert_eq!(cfg.swa_pattern, None);
        assert_eq!(cfg.swa_period, Some(2));
        let windowed: Vec<bool> = (0..cfg.n_layers)
            .map(|il| windowed_by_period(2, il))
            .collect();
        assert_eq!(windowed, vec![true, false, true, false]);
        let yarn = cfg.yarn.expect("the architecture rotates with YaRN");
        assert_eq!(yarn.factor, 32.0);
        assert_eq!(yarn.orig_ctx, 4096.0);
        assert!((yarn.attn_factor - (1.0 + 0.1 * 32f32.ln())).abs() < 1e-6);
        assert_eq!(cfg.expert_swiglu_clamp, Some((1.702, 7.0)));
        assert!(
            !cfg.flags.use_rope_i,
            "gpt-oss pairs a lane with the one half a head away"
        );
    }

    #[test]
    fn a_scaling_type_in_the_file_wins_over_the_convention() {
        let content = gptoss_like("gptoss", Some("linear"));
        let cfg = GenericTransformerConfig::from_gguf(&content, "gptoss").unwrap();
        assert!(cfg.yarn.is_none());
    }

    #[test]
    fn the_same_tensors_under_another_architecture_keep_its_defaults() {
        let content = gptoss_like("qwen3", None);
        let cfg = GenericTransformerConfig::from_gguf(&content, "qwen3").unwrap();
        assert_eq!(cfg.swa_period, None);
        assert_eq!(cfg.expert_swiglu_clamp, None);
        assert!(
            cfg.yarn.is_none(),
            "no scaling type and no convention: no YaRN"
        );
        assert_eq!(
            crate::inference::model_facts::facts("qwen3").and_then(|f| f.swa_period),
            None
        );
    }

    /// The dimensions of a small lfm2moe-shaped model: block 0 a short convolution, block 1
    /// attention with per-head QK norms, the leading block dense and the others routed.
    fn lfm2moe_like() -> Content {
        let (hidden, heads, kv_heads, hd, experts) = (32usize, 4usize, 2usize, 8usize, 4usize);
        let meta = vec![
            ("attention.head_count", Value::U32(heads as u32)),
            (
                "attention.head_count_kv",
                Value::Array(vec![
                    Value::U32(0),
                    Value::U32(kv_heads as u32),
                    Value::U32(0),
                ]),
            ),
            ("attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("block_count", Value::U32(3)),
            ("embedding_length", Value::U32(hidden as u32)),
            ("feed_forward_length", Value::U32(2 * hidden as u32)),
            ("expert_count", Value::U32(experts as u32)),
            ("expert_used_count", Value::U32(2)),
            ("expert_feed_forward_length", Value::U32(hidden as u32)),
            ("expert_gating_func", Value::U32(2)),
            ("expert_weights_scale", Value::F32(0.0)),
            ("leading_dense_block_count", Value::U32(1)),
            ("shortconv.l_cache", Value::U32(3)),
        ];
        let mut tensors: Vec<(String, Vec<usize>)> = [
            ("token_embd.weight", vec![100, hidden]),
            ("token_embd_norm.weight", vec![hidden]),
            ("blk.0.attn_norm.weight", vec![hidden]),
            ("blk.0.shortconv.in_proj.weight", vec![3 * hidden, hidden]),
            ("blk.0.shortconv.conv.weight", vec![hidden, 3]),
            ("blk.0.shortconv.out_proj.weight", vec![hidden, hidden]),
            ("blk.0.ffn_norm.weight", vec![hidden]),
            ("blk.0.ffn_gate.weight", vec![2 * hidden, hidden]),
            ("blk.0.ffn_up.weight", vec![2 * hidden, hidden]),
            ("blk.0.ffn_down.weight", vec![hidden, 2 * hidden]),
            ("blk.1.attn_norm.weight", vec![hidden]),
            ("blk.1.attn_q.weight", vec![heads * hd, hidden]),
            ("blk.1.attn_k.weight", vec![kv_heads * hd, hidden]),
            ("blk.1.attn_v.weight", vec![kv_heads * hd, hidden]),
            ("blk.1.attn_q_norm.weight", vec![hd]),
            ("blk.1.attn_k_norm.weight", vec![hd]),
            ("blk.1.attn_output.weight", vec![hidden, heads * hd]),
            ("blk.2.attn_norm.weight", vec![hidden]),
            ("blk.2.shortconv.in_proj.weight", vec![3 * hidden, hidden]),
            ("blk.2.shortconv.conv.weight", vec![hidden, 3]),
            ("blk.2.shortconv.out_proj.weight", vec![hidden, hidden]),
        ]
        .into_iter()
        .map(|(n, d)| (n.to_string(), d))
        .collect();
        for il in 1..3 {
            tensors.extend([
                (format!("blk.{il}.ffn_norm.weight"), vec![hidden]),
                (
                    format!("blk.{il}.ffn_gate_inp.weight"),
                    vec![experts, hidden],
                ),
                (format!("blk.{il}.exp_probs_b.bias"), vec![experts]),
                (
                    format!("blk.{il}.ffn_gate_exps.weight"),
                    vec![experts, hidden, hidden],
                ),
                (
                    format!("blk.{il}.ffn_up_exps.weight"),
                    vec![experts, hidden, hidden],
                ),
                (
                    format!("blk.{il}.ffn_down_exps.weight"),
                    vec![experts, hidden, hidden],
                ),
            ]);
        }
        let tensors: Vec<(&str, Vec<usize>)> = tensors
            .iter()
            .map(|(n, d)| (n.as_str(), d.clone()))
            .collect();
        synth_content("lfm2moe", &meta, &tensors)
    }

    #[test]
    fn an_lfm2moe_file_configures_its_mixers_and_the_sigmoid_router() {
        use super::super::mixer::{
            attention_layer_count, attention_probe_layer, layer_mixer_kind, MixerKind,
        };
        let content = lfm2moe_like();
        let cfg = GenericTransformerConfig::from_gguf(&content, "lfm2moe").unwrap();
        assert_eq!(layer_mixer_kind(&content, 0), MixerKind::ShortConv);
        assert_eq!(layer_mixer_kind(&content, 1), MixerKind::Attention);
        assert_eq!(layer_mixer_kind(&content, 2), MixerKind::ShortConv);
        assert_eq!(attention_probe_layer(&content, 3), 1);
        assert_eq!(attention_layer_count(&content, 3), 1);
        assert!(
            cfg.flags.has_qk_norm,
            "read on the attending block, not block 0"
        );
        assert!(!cfg.flags.fused_qkv);
        assert!(cfg.flags.is_moe, "block 0 is dense, the later ones route");
        assert!(cfg.expert_sigmoid_gating);
        assert!(cfg.expert_weights_norm, "the architecture renormalises");
        assert_eq!(cfg.expert_weights_scale, 1.0, "a zero scale is no scale");
        assert_eq!(cfg.head_dim, 8, "embedding length over heads");
        assert!(!cfg.flags.use_rope_i);
        assert!(!cfg.flags.parallel_attn);
    }

    #[test]
    fn a_nemotron_h_file_configures_its_blocks_without_rope() {
        use super::super::mixer::{
            attention_probe_layer, layer_has_ffn, layer_mixer_kind, MixerKind,
        };
        let (hidden, heads, hd, experts, inner, state, groups) =
            (32usize, 4usize, 8usize, 4usize, 64usize, 8usize, 2usize);
        let nheads = 8usize;
        let meta = vec![
            ("attention.head_count", Value::U32(heads as u32)),
            (
                "attention.head_count_kv",
                Value::Array(vec![Value::U32(0), Value::U32(0), Value::U32(2)]),
            ),
            ("attention.key_length", Value::U32(hd as u32)),
            ("attention.layer_norm_rms_epsilon", Value::F32(1e-5)),
            ("block_count", Value::U32(3)),
            ("embedding_length", Value::U32(hidden as u32)),
            ("expert_count", Value::U32(experts as u32)),
            ("expert_used_count", Value::U32(2)),
            ("expert_weights_norm", Value::Bool(true)),
            ("expert_weights_scale", Value::F32(2.5)),
            ("ssm.conv_kernel", Value::U32(4)),
            ("ssm.inner_size", Value::U32(inner as u32)),
            ("ssm.state_size", Value::U32(state as u32)),
            ("ssm.group_count", Value::U32(groups as u32)),
            ("ssm.time_step_rank", Value::U32(nheads as u32)),
        ];
        let xbc = inner + 2 * groups * state;
        let tensors: Vec<(&str, Vec<usize>)> = vec![
            ("token_embd.weight", vec![100, hidden]),
            ("output_norm.weight", vec![hidden]),
            ("blk.0.attn_norm.weight", vec![hidden]),
            ("blk.0.ssm_in.weight", vec![inner + xbc + nheads, hidden]),
            ("blk.0.ssm_conv1d.weight", vec![xbc, 4]),
            ("blk.0.ssm_a", vec![nheads, 1]),
            ("blk.0.ssm_d", vec![nheads, 1]),
            ("blk.0.ssm_dt.bias", vec![nheads]),
            ("blk.0.ssm_norm.weight", vec![groups, inner / groups]),
            ("blk.0.ssm_out.weight", vec![hidden, inner]),
            ("blk.1.attn_norm.weight", vec![hidden]),
            ("blk.1.ffn_gate_inp.weight", vec![experts, hidden]),
            ("blk.1.exp_probs_b.bias", vec![experts]),
            ("blk.1.ffn_up_exps.weight", vec![experts, hidden, hidden]),
            ("blk.1.ffn_down_exps.weight", vec![experts, hidden, hidden]),
            ("blk.1.ffn_up_shexp.weight", vec![hidden, hidden]),
            ("blk.1.ffn_down_shexp.weight", vec![hidden, hidden]),
            ("blk.2.attn_norm.weight", vec![hidden]),
            ("blk.2.attn_q.weight", vec![heads * hd, hidden]),
            ("blk.2.attn_k.weight", vec![2 * hd, hidden]),
            ("blk.2.attn_v.weight", vec![2 * hd, hidden]),
            ("blk.2.attn_output.weight", vec![hidden, heads * hd]),
        ];
        let content = synth_content("nemotron_h_moe", &meta, &tensors);
        let cfg = GenericTransformerConfig::from_gguf(&content, "nemotron_h_moe").unwrap();
        assert_eq!(layer_mixer_kind(&content, 0), MixerKind::Ssm);
        assert_eq!(layer_mixer_kind(&content, 1), MixerKind::Absent);
        assert_eq!(layer_mixer_kind(&content, 2), MixerKind::Attention);
        assert_eq!(
            (0..3)
                .map(|il| layer_has_ffn(&content, il))
                .collect::<Vec<_>>(),
            vec![false, true, false]
        );
        assert_eq!(attention_probe_layer(&content, 3), 2);
        assert!(cfg.flags.is_moe);
        assert!(!cfg.flags.parallel_attn);
        assert!(
            cfg.expert_sigmoid_gating,
            "the architecture routes by sigmoid"
        );
        assert!(cfg.expert_weights_norm, "read from the file");
        assert_eq!(cfg.expert_weights_scale, 2.5);
        let ssm = cfg.ssm.expect("ssm metadata");
        assert_eq!(
            (
                ssm.conv_kernel,
                ssm.inner_size,
                ssm.state_size,
                ssm.group_count,
                ssm.time_step_rank
            ),
            (4, inner, state, groups, nheads)
        );
        assert_eq!(
            crate::inference::model_facts::no_rope_layer_period("nemotron_h_moe"),
            Some(1),
            "no layer rotates"
        );
    }

    #[test]
    fn a_qwen35moe_file_configures_its_delta_rule_and_gated_attention() {
        use super::super::mixer::{attention_probe_layer, layer_mixer_kind, MixerKind};
        let (hidden, heads, hd, experts) = (32usize, 4usize, 8usize, 4usize);
        let meta = vec![
            ("attention.head_count", Value::U32(heads as u32)),
            ("attention.head_count_kv", Value::U32(2)),
            ("attention.key_length", Value::U32(hd as u32)),
            ("attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("block_count", Value::U32(2)),
            ("embedding_length", Value::U32(hidden as u32)),
            ("expert_count", Value::U32(experts as u32)),
            ("expert_used_count", Value::U32(2)),
            ("rope.dimension_count", Value::U32(4)),
            ("ssm.conv_kernel", Value::U32(4)),
            ("ssm.inner_size", Value::U32(16)),
            ("ssm.state_size", Value::U32(8)),
            ("ssm.group_count", Value::U32(2)),
            ("ssm.time_step_rank", Value::U32(4)),
            ("ssm.v_head_reordered", Value::Bool(true)),
        ];
        let mut tensors: Vec<(String, Vec<usize>)> = [
            ("token_embd.weight", vec![100, hidden]),
            ("output_norm.weight", vec![hidden]),
            ("blk.0.attn_qkv.weight", vec![48, hidden]),
            ("blk.0.attn_gate.weight", vec![16, hidden]),
            ("blk.0.ssm_alpha.weight", vec![4, hidden]),
            ("blk.0.ssm_beta.weight", vec![4, hidden]),
            ("blk.0.ssm_conv1d.weight", vec![48, 4]),
            ("blk.0.ssm_a", vec![4]),
            ("blk.0.ssm_dt", vec![4]),
            ("blk.0.ssm_norm.weight", vec![4]),
            ("blk.0.ssm_out.weight", vec![hidden, 16]),
            ("blk.1.attn_q.weight", vec![2 * heads * hd, hidden]),
            ("blk.1.attn_k.weight", vec![2 * hd, hidden]),
            ("blk.1.attn_v.weight", vec![2 * hd, hidden]),
            ("blk.1.attn_q_norm.weight", vec![hd]),
            ("blk.1.attn_k_norm.weight", vec![hd]),
            ("blk.1.attn_output.weight", vec![hidden, heads * hd]),
        ]
        .into_iter()
        .map(|(n, d)| (n.to_string(), d))
        .collect();
        for il in 0..2 {
            tensors.extend([
                (format!("blk.{il}.attn_norm.weight"), vec![hidden]),
                (format!("blk.{il}.post_attention_norm.weight"), vec![hidden]),
                (
                    format!("blk.{il}.ffn_gate_inp.weight"),
                    vec![experts, hidden],
                ),
                (
                    format!("blk.{il}.ffn_gate_exps.weight"),
                    vec![experts, hidden, hidden],
                ),
                (
                    format!("blk.{il}.ffn_up_exps.weight"),
                    vec![experts, hidden, hidden],
                ),
                (
                    format!("blk.{il}.ffn_down_exps.weight"),
                    vec![experts, hidden, hidden],
                ),
                (format!("blk.{il}.ffn_gate_inp_shexp.weight"), vec![hidden]),
                (
                    format!("blk.{il}.ffn_gate_shexp.weight"),
                    vec![hidden, hidden],
                ),
                (
                    format!("blk.{il}.ffn_up_shexp.weight"),
                    vec![hidden, hidden],
                ),
                (
                    format!("blk.{il}.ffn_down_shexp.weight"),
                    vec![hidden, hidden],
                ),
            ]);
        }
        let tensors: Vec<(&str, Vec<usize>)> = tensors
            .iter()
            .map(|(n, d)| (n.as_str(), d.clone()))
            .collect();
        let content = synth_content("qwen35moe", &meta, &tensors);
        let cfg = GenericTransformerConfig::from_gguf(&content, "qwen35moe").unwrap();
        assert_eq!(layer_mixer_kind(&content, 0), MixerKind::DeltaNet);
        assert_eq!(layer_mixer_kind(&content, 1), MixerKind::Attention);
        assert_eq!(attention_probe_layer(&content, 2), 1);
        assert!(
            !cfg.flags.fused_qkv,
            "the delta rule's attn_qkv is not attention's"
        );
        assert!(cfg.flags.has_qk_norm);
        assert!(
            !cfg.flags.has_post_attn_norm,
            "post_attention_norm is the feed-forward's"
        );
        assert!(!cfg.flags.parallel_attn);
        assert!(cfg.flags.is_moe);
        assert!(!cfg.expert_sigmoid_gating);
        assert!(cfg.attn_output_gate);
        assert_eq!(cfg.ffn_norm_stem, "post_attention_norm");
        assert_eq!(cfg.head_dim, hd);
        assert_eq!(cfg.rope_dim, Some(4));
        let ssm = cfg.ssm.expect("ssm metadata");
        assert!(ssm.v_heads_tiled);
        assert_eq!(
            (ssm.group_count, ssm.time_step_rank, ssm.state_size),
            (2, 4, 8)
        );
    }

    /// A part of a split set carries only its own blocks, and a single-file route reads one
    /// file: the part is refused rather than built with its missing blocks as placeholders.
    #[test]
    fn a_split_part_is_refused_rather_than_built_incomplete() {
        let mut content = synth_content(
            "llama",
            &[("block_count", Value::U32(2))],
            &[("blk.0.attn_norm.weight", vec![4])],
        );
        assert!(refuse_a_split_part(&content).is_ok(), "a whole file");
        content
            .metadata
            .insert("split.count".to_string(), Value::U32(1));
        assert!(
            refuse_a_split_part(&content).is_ok(),
            "one part is a whole file"
        );
        content
            .metadata
            .insert("split.count".to_string(), Value::U32(2));
        let e = refuse_a_split_part(&content).unwrap_err().to_string();
        assert!(e.contains("part 1 of a 2-part split"), "{e}");
        content
            .metadata
            .insert("split.no".to_string(), Value::U32(1));
        let e = refuse_a_split_part(&content).unwrap_err().to_string();
        assert!(e.contains("part 2 of a 2-part split"), "{e}");
    }

    #[test]
    fn softmax_renormalisation_follows_the_file_then_the_facts() {
        let olmoe =
            GenericTransformerConfig::from_gguf(&gptoss_like("olmoe", None), "olmoe").unwrap();
        assert!(
            !olmoe.expert_topk_norm,
            "the architecture weights by the full softmax"
        );
        let qwen3 =
            GenericTransformerConfig::from_gguf(&gptoss_like("qwen3", None), "qwen3").unwrap();
        assert!(qwen3.expert_topk_norm);
        let mut said = gptoss_like("olmoe", None);
        said.metadata
            .insert("olmoe.norm_top_k_prob".into(), Value::Bool(true));
        let said = GenericTransformerConfig::from_gguf(&said, "olmoe").unwrap();
        assert!(said.expert_topk_norm, "the file's key wins");
    }

    #[test]
    fn an_ungated_q_projection_and_a_plain_ffn_norm_are_read_as_such() {
        let content = gptoss_like("gptoss", None);
        let cfg = GenericTransformerConfig::from_gguf(&content, "gptoss").unwrap();
        assert!(!cfg.attn_output_gate);
        assert_eq!(cfg.ffn_norm_stem, "ffn_norm");
        assert!(!splices_vision_tower(&content, "gptoss"));
    }

    #[test]
    fn a_softmax_mixture_keeps_its_router() {
        let content = gptoss_like("gptoss", None);
        let cfg = GenericTransformerConfig::from_gguf(&content, "gptoss").unwrap();
        assert!(!cfg.expert_sigmoid_gating);
        assert_eq!(super::super::mixer::attention_probe_layer(&content, 4), 0);
    }

    #[test]
    fn the_period_classifies_windowed_and_global_layers() {
        assert!(windowed_by_period(0, 5), "period 0 windows every layer");
        assert!(windowed_by_period(3, 0));
        assert!(windowed_by_period(3, 1));
        assert!(!windowed_by_period(3, 2));
        assert!(windowed_by_period(3, 3));
    }
}
