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
    /// Gemma4: use GELU activation instead of SiLU in FFN
    pub use_gelu: bool,
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
        Self {
            has_qkv_bias: has_attn("attn_q.bias"),
            has_qk_norm: has_attn("attn_q_norm.weight"),
            has_post_attn_norm: has("blk.0.post_attention_norm.weight"),
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
            // The Gemma family gates its FFN with GELU. A GGUF does not record the activation,
            // so it follows the architecture, as the reference does; layer_output_scale marks
            // the Gemma4/3n builds that name themselves otherwise.
            use_gelu: arch.starts_with("gemma") || has("blk.0.layer_output_scale.weight"),
            // parallel-attn (phi2/gpt-neox): one input norm feeds both attn+ffn, so
            // there is no separate ffn_norm. OLMo2 ALSO lacks ffn_norm but is SERIAL
            // post-norm (it has post_ffw_norm) - exclude it so it isn't mis-routed.
            parallel_attn: !has("blk.0.ffn_norm.weight")
                && has("blk.0.ffn_up.weight")
                && !has("blk.0.post_ffw_norm.weight"),
            layer_norm_with_bias: has("blk.0.attn_norm.bias"),
            has_attn_output_bias: has_attn(&format!("{}.bias", attn_output_name(content))),
            has_ffn_bias: has("blk.0.ffn_down.bias"),
            has_attn_sinks: attn_sinks_name(content, attn).is_some(),
            // Llama, Mistral and Mistral3 rotate ADJACENT pairs; every other architecture
            // pairs a dimension with the one half a head away. Getting it wrong does not
            // fail - it scrambles attention and the model answers nonsense.
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

/// Facts an architecture fixes that its GGUF files do not carry. Every entry names the
/// reference behaviour it reproduces; a value present in the file's metadata takes
/// precedence over the one here.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArchConventions {
    /// Period of the sliding-window layers: layer `i` is windowed when
    /// `i % period < period - 1`, global otherwise, and every layer is windowed at
    /// period 0. Applies when the file carries no per-layer pattern.
    pub swa_period: Option<usize>,
    /// The rotary scaling applies YaRN when the file gives a scaling factor but no
    /// scaling type.
    pub yarn_by_default: bool,
    /// YaRN multiplies the rotary table by `1 + 0.1 ln(factor)`: the attention factor the
    /// reference rope kernel applies whenever its extrapolation mix is on.
    pub yarn_attention_factor: bool,
    /// Routed experts combine gate and up through the clamped SwiGLU
    /// `x.sigmoid(alpha.x).(1 + up)` with `x = min(gate, limit)` and `up` clamped to
    /// `[-limit, limit]`, as `(alpha, limit)`.
    pub expert_swiglu_clamp: Option<(f64, f64)>,
    /// The router scores experts by sigmoid when the file names no gating function.
    pub sigmoid_expert_gating: bool,
    /// Whether sigmoid-routed expert weights are renormalised over the chosen experts,
    /// over what the file says.
    pub expert_weights_norm: Option<bool>,
}

/// Whether layer `il` is a sliding-window layer under a period of `period`.
pub fn windowed_by_period(period: usize, il: usize) -> bool {
    period == 0 || il % period < period - 1
}

/// The conventions `arch` fixes beyond its files.
///
/// gptoss: the reference sets a sliding-window period of 2 (even layers windowed, odd layers
/// global), rotates with YaRN from the file's `rope.scaling.factor` and
/// `rope.scaling.original_context_length` with the rope kernel's implicit attention factor,
/// and gates its experts with the clamped SwiGLU at alpha 1.702 and limit 7.0, constants of
/// its expert graph that no converter writes.
pub fn arch_conventions(arch: &str) -> ArchConventions {
    match arch {
        "gptoss" => ArchConventions {
            swa_period: Some(2),
            yarn_by_default: true,
            yarn_attention_factor: true,
            expert_swiglu_clamp: Some((1.702, 7.0)),
            ..ArchConventions::default()
        },
        // lfm2moe: the reference renormalises the routed weights of its sigmoid router
        // whatever the file says.
        "lfm2moe" => ArchConventions {
            expert_weights_norm: Some(true),
            ..ArchConventions::default()
        },
        _ => ArchConventions::default(),
    }
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
    /// Multiplies sigmoid-routed weights; 1 when the file gives none or zero.
    pub expert_weights_scale: f64,
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

/// Which pairing RoPE rotates a head with: adjacent lanes `(2i, 2i+1)` (true) or
/// the half-split `(i, i + head_dim/2)` (false).
///
/// THIS CANNOT BE READ FROM THE FILE. GGUF carries no rope-type key, and
/// `rope.dimension_count` equals head_dim under both conventions, so it does not
/// discriminate. Both reference implementations derive it from the architecture
/// with a hardcoded table for the same reason (ollama delegates GGUF inference to
/// llama.cpp, whose `llama_model_rope_type` is such a table).
///
/// So a table is unavoidable - but SILENCE IS NOT. An architecture nobody
/// classified used to fall through to the half-split pairing without a word, and
/// that is exactly how the granite families spent months answering fluent English
/// unrelated to the prompt, with neither the loader, the benchmark nor the
/// determinism check noticing. Getting this wrong does not crash and does not
/// look like a bug; it just makes the model answer something else. Every
/// architecture is therefore named here, and a new one must be classified before
/// it can load.
pub fn rope_pairing(arch: &str) -> std::result::Result<bool, String> {
    // Adjacent pairs: the reference's "norm" rope type.
    const ADJACENT: &[&str] = &[
        "llama",
        "mistral",
        "mistral3",
        "smollm3",
        "ernie4_5",
        "granite",
        "granitemoe",
        "granite3",
        "yi",
        "internlm2",
        "starcoder",
    ];
    // Half-split: the reference's "neox" rope type.
    const HALF_SPLIT: &[&str] = &[
        "qwen2",
        "qwen3",
        "qwen3moe",
        "qwen35",
        "qwen35moe",
        "gptoss",
        "nemotron_h_moe",
        "lfm2",
        "lfm2moe",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma4",
        "phi",
        "phi2",
        "phi3",
        "phi4",
        "chatglm",
        "glm4",
        "stablelm",
        "starcoder2",
        "olmo2",
        "olmoe",
        "moondream",
        "falcon",
        "falcon3",
        "dbrx",
        "orion",
    ];
    // Positions come from somewhere other than a rotation - an attention bias, or
    // learned embeddings. The pairing is never consulted, so the value below is
    // inert; these are listed to keep the gate exhaustive rather than to choose.
    const NO_ROPE: &[&str] = &["mpt"];
    if ADJACENT.contains(&arch) {
        Ok(true)
    } else if HALF_SPLIT.contains(&arch) || NO_ROPE.contains(&arch) {
        Ok(false)
    } else {
        Err(format!(
            "architecture {arch:?} is not classified for RoPE pairing. GGUF does not carry \
             this, so it must be decided from the architecture: add it to ADJACENT (rotating \
             lanes 2i and 2i+1, what the reference calls the norm type) or to HALF_SPLIT \
             (lanes i and i + head_dim/2, the neox type) in generic_transformer/config.rs. \
             Check the reference's llama_model_rope_type for this architecture rather than \
             guessing: the wrong choice does not fail, it makes the model answer something \
             else."
        ))
    }
}

/// Period of the layers that skip RoPE (NoPE): layer `i` skips it when `(i + 1)` is a
/// multiple of the period. None when every layer rotates. GGUF does not carry this, so it
/// follows the architecture, as the reference does.
pub fn no_rope_layer_period(arch: &str) -> Option<usize> {
    match arch {
        "smollm3" => Some(4),
        _ => None,
    }
}

/// Panicking wrapper for the config builder, which has no error path here. The
/// message names the file and the choice, so an unclassified architecture stops
/// at load with an instruction rather than serving wrong answers.
fn rope_uses_adjacent_pairs(arch: &str) -> bool {
    match rope_pairing(arch) {
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

/// Read a GGUF array-of-int metadata key as `Vec<u32>` (e.g. gemma4's
/// `attention.sliding_window_pattern` = per-layer 1=SWA / 0=global). Returns
/// None if absent or not an array.
fn meta_u32_array(ct: &gguf_file::Content, arch: &str, key: &str) -> Option<Vec<u32>> {
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

        let ffn_dim = meta_u32(content, arch, "feed_forward_length")
            .unwrap_or(4 * embedding_length as u32) as usize;

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
        let conventions = arch_conventions(arch);
        let yarn = {
            let scaling_type = content
                .metadata
                .get(&format!("{arch}.rope.scaling.type"))
                .and_then(|v| v.to_string().ok().map(|s| s.to_string()))
                .or_else(|| conventions.yarn_by_default.then(|| "yarn".to_string()));
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
                            attn_factor: if conventions.yarn_attention_factor {
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
        let swa_period = conventions
            .swa_period
            .filter(|_| swa_pattern.is_none() && sliding_window.is_some());

        let shared_kv_layers =
            meta_u32(content, arch, "attention.shared_kv_layers").unwrap_or(0) as usize;

        // llama.cpp's gating function numbering: 1 softmax, 2 sigmoid.
        const GATING_SIGMOID: u32 = 2;
        let expert_sigmoid_gating = match meta_u32(content, arch, "expert_gating_func") {
            Some(f) => f == GATING_SIGMOID,
            None => conventions.sigmoid_expert_gating,
        };
        let expert_weights_norm = conventions.expert_weights_norm.unwrap_or_else(|| {
            content
                .metadata
                .get(&format!("{arch}.expert_weights_norm"))
                .and_then(|v| v.to_bool().ok())
                .unwrap_or(false)
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

        // Embedding scaling: Gemma uses √embedding_length, Granite uses explicit value
        let embed_scale = meta_f32(content, arch, "embedding_scale")
            .map(|v| v as f64)
            .or_else(|| {
                if arch.starts_with("gemma") {
                    Some((embedding_length as f64).sqrt())
                } else {
                    None
                }
            });

        // Logit scaling: Granite divides logits by this value
        let logit_scale = meta_f32(content, arch, "logit_scale").map(|v| v as f64);

        // Custom attention scale replacing 1/√head_dim (Granite: 0.015625 = 1/64)
        // Per-arch attention scale override. Gemma4 uses self.scaling=1.0
        // (NOT the default 1/sqrt(head_dim)) - see llama.cpp's
        // llama-model.cpp LLM_ARCH_GEMMA4 case where
        // `hparams.f_attention_scale = 1.0f` is hardcoded.
        let attention_scale = meta_f32(content, arch, "attention.scale")
            .map(|v| v as f64)
            .or_else(|| if arch == "gemma4" { Some(1.0) } else { None });

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
            expert_swiglu_clamp: conventions.expert_swiglu_clamp,
            expert_sigmoid_gating,
            expert_weights_norm,
            expert_weights_scale,
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
mod rope_pairing_tests {
    use super::rope_pairing;

    #[test]
    fn every_served_architecture_is_classified() {
        // The architectures the engine advertises as supported. A new one added
        // to the dispatch without a pairing must fail here, not in production.
        for arch in [
            "llama",
            "mistral",
            "mistral3",
            "qwen2",
            "qwen3",
            "qwen3moe",
            "qwen35",
            "qwen35moe",
            "gptoss",
            "nemotron_h_moe",
            "lfm2moe",
            "gemma",
            "gemma2",
            "gemma3",
            "gemma4",
            "phi",
            "phi2",
            "phi3",
            "phi4",
            "glm4",
            "stablelm",
            "starcoder",
            "starcoder2",
            "granite",
            "granite3",
            "granitemoe",
            "dbrx",
            "internlm2",
            "yi",
            "orion",
            "mpt",
            "moondream",
            "smollm3",
            "ernie4_5",
            "olmo2",
            "olmoe",
        ] {
            assert!(rope_pairing(arch).is_ok(), "{arch} has no RoPE pairing");
        }
    }

    #[test]
    fn granite_rotates_on_adjacent_pairs() {
        // The regression this gate exists for: both granite families answered
        // fluent but unrelated text under the half-split pairing.
        assert_eq!(rope_pairing("granite"), Ok(true));
        assert_eq!(rope_pairing("granitemoe"), Ok(true));
        assert_eq!(rope_pairing("qwen3"), Ok(false));
    }

    #[test]
    fn the_pairing_matches_the_reference_where_intuition_gets_it_wrong() {
        // Four architectures whose pairing does not follow from the family name,
        // and which a first pass classified from memory rather than from the
        // reference. They are pinned so a future edit has to consult it too.
        assert_eq!(rope_pairing("starcoder"), Ok(true)); // adjacent, unlike starcoder2
        assert_eq!(rope_pairing("starcoder2"), Ok(false));
        assert_eq!(rope_pairing("dbrx"), Ok(false)); // routed, but not a llama pairing
        assert_eq!(rope_pairing("orion"), Ok(false));
        assert!(rope_pairing("mpt").is_ok()); // no rotation at all: the value is inert
    }

    #[test]
    fn an_unclassified_architecture_is_refused_with_instructions() {
        let e = rope_pairing("brand_new_arch").unwrap_err();
        assert!(e.contains("not classified"), "{e}");
        assert!(e.contains("ADJACENT") && e.contains("HALF_SPLIT"), "{e}");
    }
}

#[cfg(test)]
mod arch_conventions_tests {
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
        assert_eq!(arch_conventions("qwen3"), ArchConventions::default());
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
