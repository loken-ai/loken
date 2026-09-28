//! Which GGUF files the tensor-parallel decoder (`tp_model::TpQwen2`) can run.
//!
//! The decoder implements one layer shape: RMSNorm before attention and before the FFN,
//! separate q/k/v projections with an optional q/k/v bias, full-width RoPE with no
//! scaling of embeddings, attention, residuals or logits, full causal attention, and a
//! dense SiLU-gated FFN. The decision reads the file's tensors and metadata; only the
//! facts GGUF does not record (RoPE pairing, NoPE layers, FFN activation) follow the
//! architecture, through the facts table the generic transformer uses.

use crate::inference::generic_transformer::GenericTransformerConfig;
use crate::inference::model_facts::{no_rope_layer_period, rope_pairing};
use crate::tensor::quantized::gguf_file;

/// Number of GPUs the decoder shards every layer across.
pub const TP_RANKS: usize = 2;

/// Tensors every file needs outside the layers.
const MODEL_TENSORS: &[&str] = &["token_embd.weight", "output_norm.weight"];

/// Tensors every layer `blk.{i}` needs.
const LAYER_TENSORS: &[&str] = &[
    "attn_norm.weight",
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_norm.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
];

/// `Ok(())` when the tensor-parallel decoder computes the same model as the file
/// describes, else the first feature it lacks.
pub fn tensor_parallel_support(ct: &gguf_file::Content, arch: &str) -> Result<(), String> {
    rope_pairing(arch)?;
    if no_rope_layer_period(arch).is_some() {
        return Err("some layers skip RoPE".into());
    }
    for name in MODEL_TENSORS {
        if !ct.tensor_infos.contains_key(*name) {
            return Err(format!("no {name}"));
        }
    }
    // The decoder rotates with the plain base frequencies: a file that scales them
    // (llama3 frequency factors, YaRN or any other scaling) would drift at long context.
    if ct.tensor_infos.contains_key("rope_freqs.weight") {
        return Err("scaled RoPE frequencies".into());
    }
    let scaling = ct
        .metadata
        .get(&format!("{arch}.rope.scaling.type"))
        .and_then(|v| v.to_string().ok().cloned());
    if scaling.as_deref().is_some_and(|t| t != "none") {
        return Err("RoPE scaling".into());
    }
    let cfg = GenericTransformerConfig::from_gguf(ct, arch).map_err(|e| e.to_string())?;
    for i in 0..cfg.n_layers {
        for t in LAYER_TENSORS {
            let name = format!("blk.{i}.{t}");
            if !ct.tensor_infos.contains_key(&name) {
                return Err(format!("no {name}"));
            }
        }
    }
    let value_length = ct
        .metadata
        .get(&format!("{arch}.attention.value_length"))
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize);
    let f = &cfg.flags;
    let unsupported = [
        (f.has_qk_norm, "q/k norm"),
        (
            f.has_post_attn_norm || f.has_post_ffn_norm || f.has_post_ffw_norm_split,
            "post-norms",
        ),
        (f.has_pre_ffw_norm_2, "a second pre-FFN norm"),
        (f.layer_norm_with_bias, "LayerNorm with bias"),
        (f.has_attn_output_bias, "an attention output bias"),
        (f.has_ffn_bias, "an FFN bias"),
        (f.use_gelu, "a GELU-gated FFN"),
        (f.is_moe || f.n_experts > 0, "routed experts"),
        (cfg.embed_scale.is_some(), "embedding scaling"),
        (cfg.attention_scale.is_some(), "a custom attention scale"),
        (cfg.residual_scale.is_some(), "residual scaling"),
        (cfg.logit_scale.is_some(), "logit scaling"),
        (
            cfg.attn_logit_softcapping.is_some() || cfg.final_logit_softcapping.is_some(),
            "logit softcapping",
        ),
        (
            cfg.sliding_window.is_some()
                || cfg.swa_pattern.is_some()
                || cfg.rope_freq_base_swa.is_some(),
            "sliding-window attention",
        ),
        (cfg.shared_kv_layers > 0, "shared KV layers"),
        (
            cfg.rope_dim.is_some_and(|d| d != 0 && d != cfg.head_dim),
            "partial RoPE",
        ),
        (
            value_length.is_some_and(|v| v != cfg.head_dim),
            "a value head size unlike the key head size",
        ),
        (
            cfg.n_head % TP_RANKS != 0 || cfg.n_kv_head % TP_RANKS != 0,
            "a head count the ranks cannot split evenly",
        ),
    ];
    match unsupported.iter().find(|(hit, _)| *hit) {
        Some((_, what)) => Err(what.to_string()),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::quantized::gguf_file::{Content, TensorInfo, Value, VersionedMagic};
    use crate::tensor::quantized::GgmlDType;
    use std::collections::HashMap;

    const HIDDEN: usize = 64;
    const HEADS: u32 = 4;
    const KV_HEADS: u32 = 2;
    const LAYERS: usize = 2;
    const FFN: usize = 128;
    const VOCAB: usize = 32;

    /// A dense file of `arch` with every tensor the decoder reads, no data.
    fn dense(arch: &str) -> Content {
        let mut c = Content {
            magic: VersionedMagic::GgufV3,
            metadata: HashMap::new(),
            tensor_infos: HashMap::new(),
            tensor_data_offset: 0,
            mmap_owner: None,
        };
        meta(&mut c, "general.architecture", Value::String(arch.into()));
        meta(
            &mut c,
            &format!("{arch}.attention.head_count"),
            Value::U32(HEADS),
        );
        meta(
            &mut c,
            &format!("{arch}.attention.head_count_kv"),
            Value::U32(KV_HEADS),
        );
        meta(
            &mut c,
            &format!("{arch}.block_count"),
            Value::U32(LAYERS as u32),
        );
        meta(
            &mut c,
            &format!("{arch}.embedding_length"),
            Value::U32(HIDDEN as u32),
        );
        meta(
            &mut c,
            &format!("{arch}.feed_forward_length"),
            Value::U32(FFN as u32),
        );
        tensor(&mut c, "token_embd.weight", &[VOCAB, HIDDEN]);
        tensor(&mut c, "output_norm.weight", &[HIDDEN]);
        for i in 0..LAYERS {
            for t in LAYER_TENSORS {
                tensor(&mut c, &format!("blk.{i}.{t}"), &[HIDDEN, HIDDEN]);
            }
        }
        c
    }

    fn meta(c: &mut Content, key: &str, v: Value) {
        c.metadata.insert(key.into(), v);
    }

    fn tensor(c: &mut Content, name: &str, dims: &[usize]) {
        c.tensor_infos.insert(
            name.into(),
            TensorInfo {
                ggml_dtype: GgmlDType::F32,
                shape: crate::tensor::Shape::from(dims.to_vec()),
                offset: 0,
            },
        );
    }

    fn per_layer(c: &mut Content, t: &str) {
        for i in 0..LAYERS {
            tensor(c, &format!("blk.{i}.{t}"), &[HIDDEN]);
        }
    }

    fn refusal(c: &Content, arch: &str) -> String {
        tensor_parallel_support(c, arch).expect_err("the decoder must refuse this file")
    }

    #[test]
    fn qwen2_with_qkv_bias_is_admitted() {
        let mut c = dense("qwen2");
        for t in ["attn_q.bias", "attn_k.bias", "attn_v.bias"] {
            per_layer(&mut c, t);
        }
        assert_eq!(tensor_parallel_support(&c, "qwen2"), Ok(()));
    }

    #[test]
    fn dense_llama_and_mistral_files_are_admitted() {
        for arch in ["llama", "mistral", "mistral3"] {
            assert_eq!(
                tensor_parallel_support(&dense(arch), arch),
                Ok(()),
                "{arch}"
            );
        }
    }

    #[test]
    fn a_vision_tower_beside_the_text_layers_does_not_refuse_a_file() {
        let mut c = dense("mistral3");
        tensor(&mut c, "v.blk.0.attn_q.weight", &[HIDDEN, HIDDEN]);
        meta(
            &mut c,
            "mistral3.rope.dimension_count",
            Value::U32(HIDDEN as u32 / HEADS),
        );
        assert_eq!(tensor_parallel_support(&c, "mistral3"), Ok(()));
    }

    #[test]
    fn scaled_rope_frequencies_are_refused() {
        let mut c = dense("llama");
        tensor(&mut c, "rope_freqs.weight", &[HIDDEN / HEADS as usize / 2]);
        assert_eq!(refusal(&c, "llama"), "scaled RoPE frequencies");
        let mut c = dense("mistral3");
        meta(
            &mut c,
            "mistral3.rope.scaling.type",
            Value::String("yarn".into()),
        );
        assert_eq!(refusal(&c, "mistral3"), "RoPE scaling");
    }

    #[test]
    fn a_routed_ffn_is_refused() {
        let mut c = dense("llama");
        per_layer(&mut c, "ffn_gate_inp.weight");
        meta(&mut c, "llama.expert_count", Value::U32(8));
        assert_eq!(refusal(&c, "llama"), "routed experts");
    }

    #[test]
    fn an_expert_count_alone_is_refused() {
        let mut c = dense("llama");
        meta(&mut c, "llama.expert_count", Value::U32(8));
        assert_eq!(refusal(&c, "llama"), "routed experts");
    }

    #[test]
    fn qk_norm_is_refused() {
        let mut c = dense("qwen3");
        per_layer(&mut c, "attn_q_norm.weight");
        per_layer(&mut c, "attn_k_norm.weight");
        assert_eq!(refusal(&c, "qwen3"), "q/k norm");
    }

    #[test]
    fn post_norms_and_softcapping_are_refused() {
        let mut c = dense("llama");
        per_layer(&mut c, "post_attention_norm.weight");
        assert_eq!(refusal(&c, "llama"), "post-norms");
        let mut c = dense("llama");
        meta(&mut c, "llama.attn_logit_softcapping", Value::F32(50.0));
        assert_eq!(refusal(&c, "llama"), "logit softcapping");
    }

    #[test]
    fn a_gemma_file_is_refused_for_its_activation() {
        // Tensor for tensor a llama layer; GELU is what the file cannot say.
        assert_eq!(refusal(&dense("gemma"), "gemma"), "a GELU-gated FFN");
    }

    #[test]
    fn granite_scaling_is_refused() {
        let mut c = dense("granite");
        meta(&mut c, "granite.embedding_scale", Value::F32(12.0));
        assert_eq!(refusal(&c, "granite"), "embedding scaling");
        let mut c = dense("granite");
        meta(&mut c, "granite.residual_scale", Value::F32(0.22));
        assert_eq!(refusal(&c, "granite"), "residual scaling");
    }

    #[test]
    fn sliding_window_is_refused() {
        let mut c = dense("mistral");
        meta(&mut c, "mistral.attention.sliding_window", Value::U32(4096));
        assert_eq!(refusal(&c, "mistral"), "sliding-window attention");
    }

    #[test]
    fn nope_layers_are_refused() {
        assert_eq!(
            refusal(&dense("smollm3"), "smollm3"),
            "some layers skip RoPE"
        );
    }

    #[test]
    fn partial_rope_is_refused() {
        let mut c = dense("llama");
        meta(&mut c, "llama.rope.dimension_count", Value::U32(4));
        assert_eq!(refusal(&c, "llama"), "partial RoPE");
    }

    #[test]
    fn layernorm_and_projection_biases_are_refused() {
        let mut c = dense("llama");
        per_layer(&mut c, "attn_norm.bias");
        assert_eq!(refusal(&c, "llama"), "LayerNorm with bias");
        let mut c = dense("llama");
        per_layer(&mut c, "attn_output.bias");
        assert_eq!(refusal(&c, "llama"), "an attention output bias");
        let mut c = dense("llama");
        per_layer(&mut c, "ffn_down.bias");
        assert_eq!(refusal(&c, "llama"), "an FFN bias");
    }

    #[test]
    fn a_missing_layer_tensor_is_refused() {
        let mut c = dense("llama");
        c.tensor_infos
            .remove(&format!("blk.{}.attn_v.weight", LAYERS - 1));
        assert!(refusal(&c, "llama").contains("attn_v.weight"));
        let mut c = dense("phi3");
        c.tensor_infos.remove("blk.0.ffn_gate.weight");
        assert!(refusal(&c, "phi3").contains("ffn_gate.weight"));
    }

    #[test]
    fn heads_the_ranks_cannot_split_are_refused() {
        let mut c = dense("llama");
        meta(&mut c, "llama.attention.head_count_kv", Value::U32(1));
        assert_eq!(
            refusal(&c, "llama"),
            "a head count the ranks cannot split evenly"
        );
    }

    #[test]
    fn a_value_head_unlike_the_key_head_is_refused() {
        let mut c = dense("llama");
        meta(&mut c, "llama.attention.value_length", Value::U32(8));
        assert_eq!(
            refusal(&c, "llama"),
            "a value head size unlike the key head size"
        );
    }

    #[test]
    fn an_unclassified_architecture_is_refused() {
        assert!(refusal(&dense("brand_new_arch"), "brand_new_arch").contains("has no entry"));
    }
}
