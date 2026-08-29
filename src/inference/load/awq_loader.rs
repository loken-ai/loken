//! AWQ model loader - builds a `GenericHeteroTransformer` from an
//! HF AutoAWQ checkpoint so the generic decode/attention/KV path serves AWQ
//! weights unchanged (the projection `QMatMul`s become `QMatMul::Awq`).
//!
//! ## Strategy: reuse the canonical config/flag logic
//! The trickiest correctness surface is `GenericTransformerConfig` +
//! `GenericLayerFlags` (≈25 per-arch flags). Rather than re-derive them, we
//! synthesize a **metadata-only** `gguf_file::Content` from `config.json` (the
//! GGUF metadata keys + the handful of tensor *names* `GenericLayerFlags::detect`
//! probes for) and feed it to `GenericTransformerConfig::from_gguf`. The weights
//! themselves load separately from the safetensors shards (see
//! `GenericHeteroTransformer::from_awq_safetensors`).
//!
//! Target arches: dense Qwen2 / Qwen3 / Mistral (deepcoder, qwen3-8b, deepseek-r1,
//! devstral). MoE AWQ (qwen3-coder) is out of scope for the first cut.

use crate::tensor::quantized::gguf_file::{Content, TensorInfo, Value};
use crate::tensor::quantized::GgmlDType;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

/// Parsed `config.json` essentials for an AWQ checkpoint.
#[derive(Debug, Clone)]
pub struct AwqConfigJson {
    pub hf_arch: String,
    pub gguf_arch: String,
    pub hidden_size: usize,
    pub n_layers: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: Option<usize>,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub sliding_window: Option<usize>,
    pub group_size: usize,
    pub awq_bits: usize,
    /// rope_scaling block (yarn) raw fields if present.
    pub yarn: Option<(f32, f32)>, // (factor, original_max_position)
    pub tie_word_embeddings: bool,
}

/// Map an HF `architectures[0]` to the engine's GGUF arch tag.
pub fn map_hf_arch(hf: &str) -> Result<&'static str> {
    Ok(match hf {
        "Qwen2ForCausalLM" => "qwen2",
        "Qwen3ForCausalLM" => "qwen3",
        "MistralForCausalLM" | "Mistral3ForCausalLM" | "MistralForConditionalGeneration" => "llama",
        other => {
            return Err(anyhow!(
                "AWQ: unsupported architecture {other} (dense qwen2/qwen3/mistral only)"
            ))
        }
    })
}

/// Parse the essentials out of a HF `config.json`.
pub fn parse_config_json(dir: &str) -> Result<AwqConfigJson> {
    let path = format!("{dir}/config.json");
    let txt = std::fs::read_to_string(&path).map_err(|e| anyhow!("read {path}: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&txt)?;
    let g = |k: &str| v.get(k);
    let u = |k: &str| g(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let f = |k: &str| g(k).and_then(|x| x.as_f64()).map(|x| x as f32);

    let hf_arch = v
        .get("architectures")
        .and_then(|a| a.get(0))
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow!("config.json: missing architectures[0]"))?
        .to_string();
    let gguf_arch = map_hf_arch(&hf_arch)?.to_string();

    let hidden_size = u("hidden_size").ok_or_else(|| anyhow!("missing hidden_size"))?;
    let n_head = u("num_attention_heads").ok_or_else(|| anyhow!("missing num_attention_heads"))?;
    let n_kv_head = u("num_key_value_heads").unwrap_or(n_head);

    let qc = g("quantization_config").ok_or_else(|| anyhow!("missing quantization_config"))?;
    let group_size = qc.get("group_size").and_then(|x| x.as_u64()).unwrap_or(128) as usize;
    let awq_bits = qc.get("bits").and_then(|x| x.as_u64()).unwrap_or(4) as usize;

    let yarn = g("rope_scaling").and_then(|rs| {
        let t = rs
            .get("rope_type")
            .or_else(|| rs.get("type"))
            .and_then(|x| x.as_str());
        if t == Some("yarn") {
            let factor = rs.get("factor").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
            let orig = rs
                .get("original_max_position_embeddings")
                .and_then(|x| x.as_f64())
                .unwrap_or(8192.0) as f32;
            Some((factor, orig))
        } else {
            None
        }
    });

    let use_swa = g("use_sliding_window")
        .and_then(|x| x.as_bool())
        .unwrap_or(false);
    let sliding_window = if use_swa { u("sliding_window") } else { None };

    Ok(AwqConfigJson {
        hf_arch,
        gguf_arch,
        hidden_size,
        n_layers: u("num_hidden_layers").ok_or_else(|| anyhow!("missing num_hidden_layers"))?,
        n_head,
        n_kv_head,
        head_dim: u("head_dim"),
        intermediate_size: u("intermediate_size")
            .ok_or_else(|| anyhow!("missing intermediate_size"))?,
        vocab_size: u("vocab_size").ok_or_else(|| anyhow!("missing vocab_size"))?,
        rope_theta: f("rope_theta").unwrap_or(10000.0),
        rms_norm_eps: f("rms_norm_eps").unwrap_or(1e-6),
        sliding_window,
        group_size,
        awq_bits,
        yarn,
        tie_word_embeddings: g("tie_word_embeddings")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
    })
}

/// Which optional per-layer tensors exist (drives `GenericLayerFlags::detect`).
/// Derived from the actual safetensors so flag detection matches the real model.
#[derive(Debug, Clone, Copy, Default)]
pub struct AwqTensorPresence {
    pub has_q_bias: bool,  // qwen2: q/k/v_proj.bias -> has_qkv_bias
    pub has_qk_norm: bool, // qwen3: q_norm/k_norm -> has_qk_norm
}

/// Synthesize a metadata-only `gguf_file::Content` so the canonical
/// `GenericTransformerConfig::from_gguf` + `GenericLayerFlags::detect` produce the
/// right config + flags for this AWQ model. No tensor DATA is attached - only
/// metadata values and the tensor *names* that `detect`/vocab-size probe for.
pub fn synth_gguf_content(cfg: &AwqConfigJson, present: AwqTensorPresence) -> Content {
    let a = &cfg.gguf_arch;
    let mut metadata: HashMap<String, Value> = HashMap::new();
    let mut put = |k: String, v: Value| {
        metadata.insert(k, v);
    };
    put("general.architecture".into(), Value::String(a.clone()));
    put(
        format!("{a}.attention.head_count"),
        Value::U32(cfg.n_head as u32),
    );
    put(
        format!("{a}.attention.head_count_kv"),
        Value::U32(cfg.n_kv_head as u32),
    );
    put(format!("{a}.block_count"), Value::U32(cfg.n_layers as u32));
    put(
        format!("{a}.embedding_length"),
        Value::U32(cfg.hidden_size as u32),
    );
    put(
        format!("{a}.feed_forward_length"),
        Value::U32(cfg.intermediate_size as u32),
    );
    if let Some(hd) = cfg.head_dim {
        put(format!("{a}.attention.key_length"), Value::U32(hd as u32));
        put(format!("{a}.attention.value_length"), Value::U32(hd as u32));
    }
    put(format!("{a}.rope.freq_base"), Value::F32(cfg.rope_theta));
    put(
        format!("{a}.attention.layer_norm_rms_epsilon"),
        Value::F32(cfg.rms_norm_eps),
    );
    if let Some(sw) = cfg.sliding_window {
        put(
            format!("{a}.attention.sliding_window"),
            Value::U32(sw as u32),
        );
    }
    if let Some((factor, orig)) = cfg.yarn {
        put(
            format!("{a}.rope.scaling.type"),
            Value::String("yarn".into()),
        );
        put(format!("{a}.rope.scaling.factor"), Value::F32(factor));
        put(
            format!("{a}.rope.scaling.original_context_length"),
            Value::U32(orig as u32),
        );
    }

    // Tensor names probed by detect()/vocab. shape/dtype only - no data.
    let mut tensor_infos: HashMap<String, TensorInfo> = HashMap::new();
    let mut ti = |name: String, dims: Vec<usize>| {
        tensor_infos.insert(
            name,
            TensorInfo {
                ggml_dtype: GgmlDType::F16,
                shape: crate::tensor::Shape::from(dims),
                offset: 0,
            },
        );
    };
    ti(
        "token_embd.weight".into(),
        vec![cfg.vocab_size, cfg.hidden_size],
    ); // vocab_size probe
    ti("blk.0.attn_norm.weight".into(), vec![cfg.hidden_size]);
    ti("blk.0.ffn_norm.weight".into(), vec![cfg.hidden_size]);
    ti(
        "blk.0.ffn_gate.weight".into(),
        vec![cfg.intermediate_size, cfg.hidden_size],
    );
    ti(
        "blk.0.ffn_up.weight".into(),
        vec![cfg.intermediate_size, cfg.hidden_size],
    );
    ti(
        "blk.0.ffn_down.weight".into(),
        vec![cfg.hidden_size, cfg.intermediate_size],
    );
    if present.has_q_bias {
        ti(
            "blk.0.attn_q.bias".into(),
            vec![cfg.n_head * cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.n_head)],
        );
    }
    if present.has_qk_norm {
        let hd = cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.n_head);
        ti("blk.0.attn_q_norm.weight".into(), vec![hd]);
        ti("blk.0.attn_k_norm.weight".into(), vec![hd]);
    }

    Content {
        magic: crate::tensor::quantized::gguf_file::VersionedMagic::GgufV3,
        metadata,
        tensor_infos,
        tensor_data_offset: 0,
        mmap_owner: None,
    }
}

/// Detect a model directory as AWQ: a `config.json` with
/// `quantization_config.quant_method == "awq"`. Returns the resolved dir.
pub fn find_awq_model_dir(candidate: &str) -> Option<String> {
    let cfg = format!("{candidate}/config.json");
    let txt = std::fs::read_to_string(&cfg).ok()?;
    let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
    let qm = v
        .get("quantization_config")?
        .get("quant_method")?
        .as_str()?;
    if qm.eq_ignore_ascii_case("awq") {
        Some(candidate.to_string())
    } else {
        None
    }
}
