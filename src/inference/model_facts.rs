//! What an architecture fixes that its files do not say.
//!
//! Loading decides everything it can from the file: which tensors a block carries, their
//! shapes, and the metadata keys. A few facts are nowhere in a GGUF file, yet a model
//! computes something else without them: the reference implementations hardcode them per
//! architecture for the same reason. They live here, in one table keyed by the file's
//! `general.architecture` value, and nowhere else in the code. Each entry says why the file
//! cannot tell.
//!
//! An architecture absent from the table is not refused for being unknown; it is refused
//! when a fact it needs is missing. Every attending architecture needs its RoPE pairing, so
//! in practice each architecture served through the generic transformer has an entry.

/// How RoPE pairs a head's lanes.
///
/// GGUF carries no rope-type key, and `rope.dimension_count` equals the head size under both
/// conventions. The wrong choice does not fail: attention is scrambled and the model answers
/// fluent text unrelated to the prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopePairing {
    /// Lanes `2i` and `2i+1`: the reference's "norm" rope type.
    Adjacent,
    /// Lanes `i` and `i + head_dim/2`: the reference's "neox" rope type.
    HalfSplit,
    /// No layer rotates: positions come from an attention bias, learned embeddings or
    /// recurrent layers. The pairing is never consulted.
    Unused,
}

impl RopePairing {
    /// Whether the rotation pairs adjacent lanes.
    pub fn adjacent(self) -> bool {
        matches!(self, Self::Adjacent)
    }
}

/// The facts one architecture fixes beyond its files. A value the file carries takes
/// precedence wherever the loader reads one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchFacts {
    /// How RoPE pairs a head's lanes.
    pub rope: RopePairing,
    /// Period of the layers that skip RoPE: layer `i` skips it when `i + 1` is a multiple of
    /// the period.
    pub no_rope_period: Option<usize>,
    /// Period of the sliding-window layers when the file gives a window but no per-layer
    /// pattern: layer `i` is windowed when `i % period < period - 1`.
    pub swa_period: Option<usize>,
    /// The rotary scaling is YaRN when the file gives a scaling factor but no type.
    pub yarn_by_default: bool,
    /// YaRN scales the rotary table by `1 + 0.1 ln(factor)`.
    pub yarn_attention_factor: bool,
    /// Routed experts gate through the clamped SwiGLU `x.sigmoid(alpha.x).(1 + up)` with
    /// `x = min(gate, limit)` and `up` clamped to `[-limit, limit]`, as `(alpha, limit)`.
    pub expert_swiglu_clamp: Option<(f64, f64)>,
    /// The router scores experts by sigmoid when the file names no gating function.
    pub sigmoid_expert_gating: bool,
    /// Sigmoid-routed weights are renormalised over the chosen experts, whatever the file says.
    pub sigmoid_weights_norm: Option<bool>,
    /// Softmax-routed weights are renormalised over the chosen experts when the file does not
    /// say.
    pub softmax_topk_renormalised: bool,
    /// The feed-forward gate is GELU rather than SiLU.
    pub gelu_ffn: bool,
    /// The attention values are RMS-normalised, without a weight, before they are cached.
    pub value_rms_norm: bool,
    /// Embeddings are multiplied by the square root of the model width when the file gives
    /// no `embedding_scale`.
    pub embed_scale_sqrt_width: bool,
    /// Attention scores are scaled by this instead of `1/sqrt(head_dim)` when the file gives
    /// no `attention.scale`.
    pub attention_scale: Option<f64>,
    /// A quantised KV cache degrades the output; the cache stays unquantised.
    pub quantized_kv_degrades: bool,
    /// A BOS leads every sequence even when the file declares `add_bos_token = false`.
    pub forces_bos: bool,
    /// Vocabulary pieces that end the model's turn when the file lists no
    /// `tokenizer.ggml.eos_token_ids`.
    pub turn_stop_pieces: &'static [&'static str],
    /// The pre-tokenizer a file naming `default` was trained with.
    pub default_pre_tokenizer: Option<&'static str>,
}

impl ArchFacts {
    /// No fact beyond the pairing: what the file says is all there is.
    const fn rotating(rope: RopePairing) -> Self {
        Self {
            rope,
            no_rope_period: None,
            swa_period: None,
            yarn_by_default: false,
            yarn_attention_factor: false,
            expert_swiglu_clamp: None,
            sigmoid_expert_gating: false,
            sigmoid_weights_norm: None,
            softmax_topk_renormalised: true,
            gelu_ffn: false,
            value_rms_norm: false,
            embed_scale_sqrt_width: false,
            attention_scale: None,
            quantized_kv_degrades: false,
            forces_bos: false,
            turn_stop_pieces: &[],
            default_pre_tokenizer: None,
        }
    }
}

const ADJACENT: ArchFacts = ArchFacts::rotating(RopePairing::Adjacent);
const HALF_SPLIT: ArchFacts = ArchFacts::rotating(RopePairing::HalfSplit);
const UNUSED: ArchFacts = ArchFacts::rotating(RopePairing::Unused);

/// The Gemma feed-forward and embedding: GELU-gated, embeddings scaled by the square root of
/// the width. Neither the activation nor the scale is written by the converters.
const GEMMA: ArchFacts = ArchFacts {
    gelu_ffn: true,
    embed_scale_sqrt_width: true,
    ..HALF_SPLIT
};

/// Every architecture with a fact its files do not carry. The pairings follow the
/// reference's `llama_model_rope_type`; check it for a new entry rather than guessing.
const FACTS: &[(&str, ArchFacts)] = &[
    ("llama", ADJACENT),
    ("mistral", ADJACENT),
    ("mistral3", ADJACENT),
    // Every fourth layer skips RoPE; no converter writes the period.
    (
        "smollm3",
        ArchFacts {
            no_rope_period: Some(4),
            ..ADJACENT
        },
    ),
    ("ernie4_5", ADJACENT),
    ("granite", ADJACENT),
    ("granitemoe", ADJACENT),
    ("granite3", ADJACENT),
    ("yi", ADJACENT),
    ("internlm2", ADJACENT),
    // Adjacent, unlike starcoder2.
    ("starcoder", ADJACENT),
    ("qwen2", HALF_SPLIT),
    ("qwen3", HALF_SPLIT),
    ("qwen3moe", HALF_SPLIT),
    ("qwen35", HALF_SPLIT),
    ("qwen35moe", HALF_SPLIT),
    ("qwen3next", HALF_SPLIT),
    // The window alternates on even and odd layers, YaRN applies from the file's factor with
    // the rope kernel's implicit attention factor, and the experts gate through the clamped
    // SwiGLU at alpha 1.702 and limit 7.0: constants of the reference graph no converter
    // writes. A file naming the `default` pre-tokenizer is split as o200k, the vocabulary the
    // architecture was trained with.
    (
        "gptoss",
        ArchFacts {
            swa_period: Some(2),
            yarn_by_default: true,
            yarn_attention_factor: true,
            expert_swiglu_clamp: Some((1.702, 7.0)),
            default_pre_tokenizer: Some("gpt-4o"),
            ..HALF_SPLIT
        },
    ),
    ("lfm2", HALF_SPLIT),
    // The reference renormalises the routed weights of the sigmoid router; the files carry
    // no `expert_weights_norm`.
    (
        "lfm2moe",
        ArchFacts {
            sigmoid_weights_norm: Some(true),
            ..HALF_SPLIT
        },
    ),
    ("gemma", GEMMA),
    ("gemma2", GEMMA),
    ("gemma3", GEMMA),
    // Beyond the Gemma feed-forward: the reference hardcodes an attention scale of 1 and
    // normalises the values without a weight, which Gemma 2 and 3 do not. The
    // K-norm weights are small and uniform, and a quantised cache amplifies their rounding
    // until the output repeats. The converters write `add_bos_token = false` though the model
    // needs a BOS, which the reference overrides for this architecture too. Turns end on the
    // chat template's turn and tool-response markers, while the file names one end token.
    (
        "gemma4",
        ArchFacts {
            attention_scale: Some(1.0),
            value_rms_norm: true,
            quantized_kv_degrades: true,
            forces_bos: true,
            turn_stop_pieces: &["<|turn>", "<turn|>", "<|tool_response>"],
            ..GEMMA
        },
    ),
    ("phi", HALF_SPLIT),
    ("phi2", HALF_SPLIT),
    ("phi3", HALF_SPLIT),
    ("phi4", HALF_SPLIT),
    ("chatglm", HALF_SPLIT),
    ("glm4", HALF_SPLIT),
    ("stablelm", HALF_SPLIT),
    ("starcoder2", HALF_SPLIT),
    ("olmo2", HALF_SPLIT),
    // The chosen experts are weighted by their softmax over all experts, not renormalised;
    // the files carry no `expert_weights_norm`.
    (
        "olmoe",
        ArchFacts {
            softmax_topk_renormalised: false,
            ..HALF_SPLIT
        },
    ),
    ("moondream", HALF_SPLIT),
    ("falcon", HALF_SPLIT),
    ("falcon3", HALF_SPLIT),
    // Routed, but half-split.
    ("dbrx", HALF_SPLIT),
    ("orion", HALF_SPLIT),
    // Positions come from an attention bias.
    ("mpt", UNUSED),
    // No attention layer rotates, and the router scores by sigmoid; the files carry neither.
    (
        "nemotron_h_moe",
        ArchFacts {
            no_rope_period: Some(1),
            sigmoid_expert_gating: true,
            ..UNUSED
        },
    ),
];

/// The architecture keys the table holds.
pub fn architectures() -> impl Iterator<Item = &'static str> {
    FACTS.iter().map(|(arch, _)| *arch)
}

/// The facts `arch` fixes, `None` when the table has no entry for it.
pub fn facts(arch: &str) -> Option<&'static ArchFacts> {
    FACTS.iter().find(|(a, _)| *a == arch).map(|(_, f)| f)
}

/// The facts `arch` fixes, or the default of an architecture that fixes none beyond its RoPE
/// pairing. For readers of a fact that has a neutral value; the pairing itself is read
/// through [`rope_pairing`].
pub fn facts_or_default(arch: &str) -> ArchFacts {
    facts(arch).copied().unwrap_or(HALF_SPLIT)
}

/// Whether `arch` rotates adjacent lanes (true) or lanes half a head apart (false), or why it
/// cannot be built when the table does not say.
pub fn rope_pairing(arch: &str) -> Result<bool, String> {
    match facts(arch) {
        Some(f) => Ok(f.rope.adjacent()),
        None if arch.is_empty() => Err(
            "the file names no general.architecture, so how its RoPE pairs lanes cannot be \
             looked up"
                .to_string(),
        ),
        None => Err(format!(
            "architecture {arch:?} has no entry in inference/model_facts.rs. Its file does not \
             say whether RoPE rotates adjacent lanes (the reference's norm type) or lanes half a \
             head apart (the neox type), and the wrong choice does not fail: the model answers \
             something else. Add it with the pairing the reference's llama_model_rope_type gives."
        )),
    }
}

/// Period of the layers of `arch` that skip RoPE, `None` when every layer rotates.
pub fn no_rope_layer_period(arch: &str) -> Option<usize> {
    facts(arch).and_then(|f| f.no_rope_period)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_key_appears_once() {
        let mut keys: Vec<&str> = architectures().collect();
        let n = keys.len();
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), n, "an architecture is listed twice");
    }

    #[test]
    fn granite_rotates_on_adjacent_pairs() {
        // Both granite families answered fluent but unrelated text under the half-split
        // pairing.
        assert_eq!(rope_pairing("granite"), Ok(true));
        assert_eq!(rope_pairing("granitemoe"), Ok(true));
        assert_eq!(rope_pairing("qwen3"), Ok(false));
    }

    #[test]
    fn the_pairing_matches_the_reference_where_intuition_gets_it_wrong() {
        assert_eq!(rope_pairing("starcoder"), Ok(true));
        assert_eq!(rope_pairing("starcoder2"), Ok(false));
        assert_eq!(rope_pairing("dbrx"), Ok(false));
        assert_eq!(rope_pairing("orion"), Ok(false));
        assert!(rope_pairing("mpt").is_ok());
    }

    #[test]
    fn an_architecture_without_an_entry_is_refused_with_instructions() {
        let e = rope_pairing("brand_new_arch").unwrap_err();
        assert!(
            e.contains("model_facts.rs") && e.contains("llama_model_rope_type"),
            "{e}"
        );
        assert!(rope_pairing("")
            .unwrap_err()
            .contains("general.architecture"));
    }

    #[test]
    fn a_hybrid_without_rotation_is_classified() {
        assert_eq!(no_rope_layer_period("nemotron_h_moe"), Some(1));
        assert!(rope_pairing("nemotron_h_moe").is_ok());
        assert_eq!(no_rope_layer_period("qwen3"), None);
    }
}
