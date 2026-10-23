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

use super::hf_config::{hf_class, lookup, lookup_b, lookup_f, lookup_u, rope_block};
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
    /// The frequency-dependent rope scaling of a long-context checkpoint:
    /// `(factor, low_freq_factor, high_freq_factor, original_max_position)`.
    pub frequency_scaling: Option<(f32, f32, f32, f32)>,
    /// Whether the checkpoint carries a `quantization_config`; a dense one does not.
    pub quantized: bool,
    pub tie_word_embeddings: bool,
    /// The weights are in the layout a vendor's own release writes (`params.json` and a
    /// `consolidated` shard) and not in the framework's (`config.json` and numbered shards).
    pub native: bool,
    /// The parsed `config.json` (or `params.json`), for the keys only some classes carry.
    pub root: serde_json::Value,
}

/// Map an HF `architectures[0]` to the engine's GGUF arch tag.
pub fn map_hf_arch(hf: &str) -> Result<&'static str> {
    match hf_class(hf) {
        Some(c) => Ok(c.arch),
        None => Err(anyhow!(
            "unsupported checkpoint architecture {hf}: no row in the class table (hf_config.rs)"
        )),
    }
}

/// Parse the essentials out of a HF `config.json`, or out of the `params.json` of a native
/// layout when the directory has no `config.json`.
pub fn parse_config_json(dir: &str) -> Result<AwqConfigJson> {
    let path = format!("{dir}/config.json");
    let txt = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            return match std::fs::read_to_string(format!("{dir}/params.json")) {
                Ok(p) => parse_params_json(&p),
                Err(_) => Err(anyhow!("read {path}: {e}")),
            }
        }
    };
    let v: serde_json::Value = serde_json::from_str(&txt)?;
    let u = |keys: &[&str]| lookup_u(&v, keys);
    let f = |keys: &[&str]| lookup_f(&v, keys);

    let hf_arch = v
        .get("architectures")
        .and_then(|a| a.get(0))
        .and_then(|s| s.as_str())
        .ok_or_else(|| anyhow!("config.json: missing architectures[0]"))?
        .to_string();
    let gguf_arch = map_hf_arch(&hf_arch)?.to_string();

    let hidden_size =
        u(&["hidden_size", "d_model", "n_embd"]).ok_or_else(|| anyhow!("missing hidden_size"))?;
    let n_head = u(&["num_attention_heads", "n_heads", "n_head"])
        .ok_or_else(|| anyhow!("missing num_attention_heads"))?;
    let n_kv_head = u(&[
        "num_key_value_heads",
        "attn_config.kv_n_heads",
        "multi_query_group_num",
        "num_kv_heads",
        "n_head_kv",
    ])
    .unwrap_or_else(|| match lookup_b(&v, &["multi_query"]) {
        Some(true) => 1,
        _ => n_head,
    });

    // A dense checkpoint carries none; its group size and width are never read.
    let qc = lookup(&v, &["quantization_config"]);
    let group_size = qc
        .and_then(|q| q.get("group_size"))
        .and_then(|x| x.as_u64())
        .unwrap_or(128) as usize;
    let awq_bits = qc
        .and_then(|q| q.get("bits"))
        .and_then(|x| x.as_u64())
        .unwrap_or(4) as usize;
    let rope = rope_block(&v);
    let rope_type = rope
        .and_then(|rs| rs.get("rope_type").or_else(|| rs.get("type")))
        .and_then(|x| x.as_str());
    let frequency_scaling = rope.and_then(|rs| {
        // ARCH-NAME-EXEMPT: the `rope_type` a checkpoint's configuration declares.
        if rope_type != Some("llama3") {
            return None;
        }
        let f = |k: &str| rs.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
        Some((
            f("factor")?,
            f("low_freq_factor")?,
            f("high_freq_factor")?,
            f("original_max_position_embeddings")?,
        ))
    });

    let yarn = rope.and_then(|rs| {
        if rope_type == Some("yarn") {
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

    let use_swa = lookup_b(&v, &["use_sliding_window"]).unwrap_or(false);
    let sliding_window = if use_swa {
        u(&["sliding_window"])
    } else {
        None
    };
    let expansion = u(&["expansion_ratio"]).map(|r| r * hidden_size);

    Ok(AwqConfigJson {
        hf_arch,
        gguf_arch,
        hidden_size,
        n_layers: u(&["num_hidden_layers", "n_layers", "num_layers", "n_layer"])
            .ok_or_else(|| anyhow!("missing num_hidden_layers"))?,
        n_head,
        n_kv_head,
        head_dim: u(&["head_dim", "kv_channels"]),
        // The configuration's own default, four times the width, when it has none.
        intermediate_size: u(&[
            "intermediate_size",
            "ffn_hidden_size",
            "ffn_config.ffn_hidden_size",
            "n_inner",
        ])
        .or(expansion)
        .unwrap_or(4 * hidden_size),
        vocab_size: u(&["vocab_size", "padded_vocab_size"]).unwrap_or(0),
        rope_theta: f(&[
            "rope_theta",
            "rope_parameters.rope_theta",
            "attn_config.rope_theta",
            "rotary_emb_base",
        ])
        .or_else(|| f(&["rope_ratio"]).map(|r| 10000.0 * r))
        .unwrap_or(10000.0),
        rms_norm_eps: f(&[
            "rms_norm_eps",
            "layer_norm_epsilon",
            "layer_norm_eps",
            "layernorm_epsilon",
            "norm_eps",
            "norm_epsilon",
        ])
        .unwrap_or(1e-5),
        sliding_window,
        group_size,
        awq_bits,
        yarn,
        frequency_scaling,
        quantized: qc.is_some(),
        tie_word_embeddings: lookup_b(&v, &["tie_word_embeddings"]).unwrap_or(false),
        native: false,
        root: v,
    })
}

/// The essentials of a native layout's `params.json`. The text model is the llama family's:
/// that is what the vendor's own release of a decoder-only model is.
fn parse_params_json(text: &str) -> Result<AwqConfigJson> {
    let v: serde_json::Value = serde_json::from_str(text)?;
    let u = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let need = |k: &str| u(k).ok_or_else(|| anyhow!("params.json: missing {k}"));
    let n_head = need("n_heads")?;
    Ok(AwqConfigJson {
        hf_arch: String::new(),
        gguf_arch: "llama".to_string(),
        hidden_size: need("dim")?,
        n_layers: need("n_layers")?,
        n_head,
        n_kv_head: u("n_kv_heads").unwrap_or(n_head),
        head_dim: u("head_dim"),
        intermediate_size: need("hidden_dim")?,
        vocab_size: need("vocab_size")?,
        rope_theta: v
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .unwrap_or(10000.0) as f32,
        rms_norm_eps: v.get("norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-5) as f32,
        sliding_window: None,
        group_size: 128,
        awq_bits: 4,
        yarn: None,
        frequency_scaling: None,
        quantized: v.get("quantization").is_some(),
        tie_word_embeddings: v
            .get("tied_embeddings")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        native: true,
        root: v.clone(),
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
    if cfg.intermediate_size > 0 {
        put(
            format!("{a}.feed_forward_length"),
            Value::U32(cfg.intermediate_size as u32),
        );
    }
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

    for (k, v) in super::hf_config::hf_metadata(cfg) {
        put(k, v);
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

/// Per-pair frequency divisors of a long-context rope: pairs whose wavelength is short keep
/// their frequency, pairs whose wavelength is long are divided by `factor`, and those between
/// are interpolated. `[head_dim / 2]`.
pub fn frequency_factors(head_dim: usize, base: f32, scaling: (f32, f32, f32, f32)) -> Vec<f32> {
    let (factor, low, high, original) = scaling;
    let (low_wavelength, high_wavelength) = (original / low, original / high);
    (0..head_dim / 2)
        .map(|i| {
            let freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let wavelength = 2.0 * std::f32::consts::PI / freq;
            if wavelength < high_wavelength {
                1.0
            } else if wavelength > low_wavelength {
                factor
            } else {
                let smooth = (original / wavelength - low) / (high - low);
                1.0 / ((1.0 - smooth) / factor + smooth)
            }
        })
        .collect()
}

/// The weight files of a checkpoint directory, by layout. The framework's layout is a
/// `config.json` and the shards its index names (every `*.safetensors` but a native one when
/// there is no index); a vendor's native layout is a `params.json` and `consolidated*.safetensors`.
/// A directory holding both is read in the framework's layout. The error names what the
/// directory lacks of each.
pub fn checkpoint_shards(dir: &std::path::Path) -> Result<(bool, Vec<String>)> {
    let safetensors: Vec<String> = std::fs::read_dir(dir)
        .map_err(|e| anyhow!("{}: {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    let is_native_shard = |p: &String| {
        std::path::Path::new(p)
            .file_name()
            .is_some_and(|n| n.to_string_lossy().starts_with("consolidated"))
    };
    let has_config = dir.join("config.json").is_file();
    let mut framework: Vec<String> = safetensors
        .iter()
        .filter(|p| !is_native_shard(p))
        .cloned()
        .collect();
    let mut native: Vec<String> = safetensors.into_iter().filter(is_native_shard).collect();
    let has_params = dir.join("params.json").is_file();
    if has_config && !framework.is_empty() {
        if let Some(listed) = indexed_shards(dir) {
            framework.retain(|p| listed.iter().any(|l| p.ends_with(l.as_str())));
        }
        framework.sort();
        return Ok((false, framework));
    }
    if has_params && !native.is_empty() {
        native.sort();
        return Ok((true, native));
    }
    let lacks = |what: &[(bool, &str)]| {
        what.iter()
            .filter(|(present, _)| !present)
            .map(|(_, n)| *n)
            .collect::<Vec<_>>()
            .join(", ")
    };
    Err(anyhow!(
        "{}: no loadable checkpoint layout. Framework layout lacks: {}. Native layout lacks: {}",
        dir.display(),
        lacks(&[
            (has_config, "config.json"),
            (!framework.is_empty(), "numbered *.safetensors shards"),
        ]),
        lacks(&[
            (has_params, "params.json"),
            (!native.is_empty(), "consolidated*.safetensors"),
        ]),
    ))
}

/// The file names a framework layout's `model.safetensors.index.json` lists, `None` when there
/// is no index.
fn indexed_shards(dir: &std::path::Path) -> Option<Vec<String>> {
    let text = std::fs::read_to_string(dir.join("model.safetensors.index.json")).ok()?;
    let index: serde_json::Value = serde_json::from_str(&text).ok()?;
    let mut files: Vec<String> = index
        .get("weight_map")?
        .as_object()?
        .values()
        .filter_map(|f| f.as_str().map(str::to_string))
        .collect();
    files.sort();
    files.dedup();
    Some(files)
}

/// A checkpoint directory of dense floating-point weights this loader reads: no
/// quantisation, a supported text architecture (a `config.json`'s `architectures`, or a
/// native layout's `params.json`), and weight files for one of the two layouts.
pub fn find_dense_model_dir(candidate: &str) -> Option<String> {
    let dir = std::path::Path::new(candidate);
    let (native, _) = checkpoint_shards(dir).ok()?;
    if native {
        return parse_config_json(candidate)
            .ok()
            .filter(|c| !c.quantized)
            .map(|_| candidate.to_string());
    }
    let txt = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&txt).ok()?;
    if v.get("quantization_config").is_some() {
        return None;
    }
    let arch = v.get("architectures")?.get(0)?.as_str()?;
    map_hf_arch(arch).ok()?;
    Some(candidate.to_string())
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    const COMPOSITE: &str = r#"{"architectures": ["VoxtralForConditionalGeneration"],
        "hidden_size": 24, "vocab_size": 100,
        "text_config": {"hidden_size": 24, "intermediate_size": 48, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
            "rope_theta": 1000000.0, "rms_norm_eps": 1e-5, "vocab_size": 100}}"#;
    const NATIVE: &str = r#"{"dim": 24, "n_layers": 2, "head_dim": 8, "hidden_dim": 48,
        "n_heads": 4, "n_kv_heads": 2, "rope_theta": 1000000.0, "norm_eps": 1e-5,
        "vocab_size": 100}"#;

    fn directory(name: &str, files: &[(&str, &str)]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("loken-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for (file, text) in files {
            std::fs::write(dir.join(file), text).unwrap();
        }
        dir
    }

    /// The composite model's text fields are read from its `text_config`, the native
    /// layout's from `params.json`; both describe the same text model.
    #[test]
    fn both_layouts_describe_the_same_text_model() {
        let framework = directory("fw-config", &[("config.json", COMPOSITE)]);
        let native = directory("native-config", &[("params.json", NATIVE)]);
        let a = parse_config_json(framework.to_str().unwrap()).unwrap();
        let b = parse_config_json(native.to_str().unwrap()).unwrap();
        for d in [&framework, &native] {
            let _ = std::fs::remove_dir_all(d);
        }
        assert!(!a.native && b.native);
        assert_eq!(
            (a.hidden_size, a.n_layers, a.n_head, a.n_kv_head, a.head_dim),
            (b.hidden_size, b.n_layers, b.n_head, b.n_kv_head, b.head_dim)
        );
        assert_eq!(
            (a.intermediate_size, a.vocab_size, a.rope_theta, a.gguf_arch),
            (b.intermediate_size, b.vocab_size, b.rope_theta, b.gguf_arch)
        );
    }

    /// Both layouts in one directory are read in the framework's, and only the shards its
    /// index names; a directory of one layout is read in that one.
    #[test]
    fn the_layout_is_told_by_the_files() {
        let both = directory(
            "both-layouts",
            &[
                ("config.json", COMPOSITE),
                ("params.json", NATIVE),
                ("consolidated.safetensors", ""),
                ("model-00001-of-00002.safetensors", ""),
                ("model-00002-of-00002.safetensors", ""),
                (
                    "model.safetensors.index.json",
                    r#"{"weight_map": {"a": "model-00001-of-00002.safetensors",
                        "b": "model-00002-of-00002.safetensors"}}"#,
                ),
            ],
        );
        let (native, shards) = checkpoint_shards(&both).unwrap();
        assert!(!native);
        assert_eq!(shards.len(), 2);
        assert!(shards.iter().all(|s| s.contains("model-0000")));
        std::fs::remove_file(both.join("config.json")).unwrap();
        let (native, shards) = checkpoint_shards(&both).unwrap();
        assert!(native && shards.len() == 1);
        let _ = std::fs::remove_dir_all(&both);

        let only = directory(
            "native-only",
            &[("params.json", NATIVE), ("consolidated.safetensors", "")],
        );
        let (native, shards) = checkpoint_shards(&only).unwrap();
        assert!(native && shards.len() == 1);
        assert!(find_dense_model_dir(only.to_str().unwrap()).is_some());
        let _ = std::fs::remove_dir_all(&only);
    }

    /// A directory that is neither layout is refused with what each layout lacks.
    #[test]
    fn a_refusal_names_what_is_missing() {
        let dir = directory("no-layout", &[("params.json", NATIVE)]);
        let message = checkpoint_shards(&dir).unwrap_err().to_string();
        let _ = std::fs::remove_dir_all(&dir);
        assert!(message.contains("config.json"), "{message}");
        assert!(
            message.contains("numbered *.safetensors shards"),
            "{message}"
        );
        assert!(message.contains("consolidated*.safetensors"), "{message}");
        assert!(!message.contains("lacks: params.json"), "{message}");
    }
}
