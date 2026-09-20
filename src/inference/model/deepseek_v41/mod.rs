//! DeepSeek V4.1 Flash - configuration and shapes (bet phase 1).
//!
//! The architecture is decoder-only over forty blocks whose attention mode varies by layer, with
//! hyper-connections carrying the residual as `hc_mult` parallel copies, latent MQA with a grouped
//! low-rank output projection, CSA2 (a sliding window plus a compressor and a two-level indexer),
//! a shared expert beside routed ones under a sqrt-softplus router, and block-scaled fp8/fp4
//! weights. None of that fits `GenericTransformerConfig`, so this is its own model module.
//!
//! Phase 1 is shapes only: parse the whole configuration from GGUF metadata and prove it matches
//! the reference. The forward - attention, MoE, hyper-connections, CSA2 - is phases 2 to 6, each
//! gated against `notes/deepseek-oracle`. No real GGUF carries this architecture yet, so the keys
//! below are what a converter would emit, following llama.cpp's deepseek naming where it exists.

pub mod attention;
pub mod band;
pub mod block;
pub mod cache;
pub mod candidates;
pub mod convert;
pub mod derived;
pub mod engram;
pub mod engram_hash;
pub mod hyper_connections;
pub mod load;
pub mod model;
pub mod moe;
pub mod safetensors_source;
pub mod source;
pub mod token_map;

use crate::tensor::quantized::gguf_file::{self, Value};
use crate::tensor::quantized::gguf_source::GgufMeta;
use crate::tensor::{Error, Result};

/// The GGUF metadata prefix for this architecture's keys.
pub const ARCH: &str = "deepseek_v41";

/// Every shape the port needs, read from GGUF metadata. Scalars follow llama.cpp's deepseek keys
/// (`attention.head_count`, `attention.key_length`, ...); the novel per-layer and CSA2 fields take
/// keys under the same prefix.
#[derive(Debug, Clone)]
pub struct DeepseekV41Config {
    // Backbone.
    pub n_layers: usize,
    pub n_mtp_layers: usize,
    pub d_model: usize,
    pub vocab_size: usize,
    pub rms_eps: f64,

    // Attention: latent Q/KV, grouped low-rank output, one KV head (MQA).
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub window_size: usize,

    // MoE: routed experts plus one shared, sqrt-softplus scoring.
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub n_activated_experts: usize,
    pub moe_inter_dim: usize,
    pub score_func: String,
    pub route_scale: f32,
    pub gate_temp: f32,
    /// The SwiGLU clamp of layer 0; `swiglu_limits` carries every layer's.
    pub swiglu_limit: f32,
    pub swiglu_limits: Vec<f32>,
    pub norm_topk: bool,
    /// Routed experts kept resident in the hot cache when they are streamed; 0 keeps every expert
    /// once built (resident-equivalent). A low-memory deployment sets it below the expert count.
    pub expert_cache_count: usize,

    // CSA2: per-layer compression, the layers that source shared KV and the indexer.
    pub compress_ratios: Vec<usize>,
    /// Per layer, experts from the most routed-to on the calibration corpus downwards; empty
    /// for a file converted without the prior.
    pub hot_experts: Vec<Vec<usize>>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    // Two-level candidate pre-filter. `candidate_source_layer` is -1 when off.
    pub candidate_source_layer: i64,
    pub candidate_topk_blocks: usize,
    pub candidate_block_size: usize,

    // Hyper-connections.
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,

    // RoPE: base theta, YaRN, and the separate theta the compressed KV rotates at.
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub original_seq_len: usize,
    pub compress_rope_theta: f32,

    /// The n-gram hash layers and everything their hash needs, when the file carries it all.
    pub engram: Option<engram::EngramConfig>,
}

/// The architecture tag llama.cpp's converter writes.
pub const ARCH_LLAMACPP: &str = "deepseek4";

/// Values llama.cpp's `deepseek4` converter does not write into the file. They are the
/// V4.1-Flash release's, from the model's own inference config at the revision the converter
/// names, and are read only when the key is absent from one of that converter's files.
mod release {
    pub const ROPE_HEAD_DIM: usize = 64;
    pub const ROPE_FACTOR: f32 = 16.0;
    pub const ORIGINAL_SEQ_LEN: usize = 65536;
    pub const CANDIDATE_SOURCE_LAYER: i64 = 20;
    pub const CANDIDATE_TOPK_BLOCKS: usize = 2048;
    pub const CANDIDATE_BLOCK_SIZE: usize = 8;
}

impl DeepseekV41Config {
    pub fn from_gguf(ct: &gguf_file::Content) -> Result<Self> {
        Self::from_meta(ct)
    }

    /// The reference implementation's arguments (`inference/config.json` of the released
    /// checkpoint), read under their own names. `engram` is what the hash needs beyond those
    /// arguments - the compressed token map - when the caller could derive it; `None` runs the
    /// model without its engram layers.
    pub fn from_reference_config(
        v: &serde_json::Value,
        engram: Option<engram::EngramConfig>,
    ) -> Result<Self> {
        let req_u = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| Error::msg(format!("inference config: missing {k}")))
        };
        let opt_u = |k: &str, d: usize| {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .unwrap_or(d)
        };
        let opt_f = |k: &str, d: f32| {
            v.get(k)
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .unwrap_or(d)
        };
        let opt_i = |k: &str, d: i64| v.get(k).and_then(|x| x.as_i64()).unwrap_or(d);
        let opt_b = |k: &str, d: bool| v.get(k).and_then(|x| x.as_bool()).unwrap_or(d);
        let opt_s = |k: &str, d: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or(d).to_string();
        let arr_u = |k: &str| -> Vec<usize> {
            v.get(k)
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as usize)
                        .collect()
                })
                .unwrap_or_default()
        };
        let n_layers = req_u("n_layers");
        let n_layers = n_layers?;
        let swiglu_limit = opt_f("swiglu_limit", 0.0);
        let cfg = Self {
            n_layers,
            n_mtp_layers: opt_u("n_mtp_layers", 0),
            d_model: req_u("dim")?,
            vocab_size: req_u("vocab_size")?,
            rms_eps: opt_f("norm_eps", 1e-20) as f64,
            n_head: req_u("n_heads")?,
            n_kv_head: 1,
            head_dim: req_u("head_dim")?,
            rope_head_dim: req_u("rope_head_dim")?,
            q_lora_rank: req_u("q_lora_rank")?,
            o_lora_rank: req_u("o_lora_rank")?,
            o_groups: req_u("o_groups")?,
            window_size: opt_u("window_size", 0),
            n_routed_experts: req_u("n_routed_experts")?,
            n_shared_experts: opt_u("n_shared_experts", 1),
            n_activated_experts: req_u("n_activated_experts")?,
            moe_inter_dim: req_u("moe_inter_dim")?,
            score_func: opt_s("score_func", "sqrtsoftplus"),
            route_scale: opt_f("route_scale", 1.0),
            gate_temp: opt_f("gate_temp", 1.0),
            swiglu_limit,
            swiglu_limits: vec![swiglu_limit; n_layers],
            norm_topk: opt_b("norm_topk_prob", true),
            expert_cache_count: 0,
            compress_ratios: arr_u("compress_ratios"),
            hot_experts: Vec::new(),
            kv_source_layers: arr_u("kv_source_layers"),
            index_source_layers: arr_u("index_source_layers"),
            index_n_heads: opt_u("index_n_heads", 0),
            index_head_dim: opt_u("index_head_dim", 0),
            index_topk: opt_u("index_topk", 0),
            candidate_source_layer: opt_i("candidate_source_layer", -1),
            candidate_topk_blocks: opt_u("candidate_topk_blocks", 0),
            candidate_block_size: opt_u("candidate_block_size", 0),
            hc_mult: opt_u("hc_mult", 1),
            hc_sinkhorn_iters: opt_u("hc_sinkhorn_iters", 0),
            hc_eps: opt_f("hc_eps", 1e-6),
            rope_theta: opt_f("rope_theta", 10000.0),
            rope_factor: opt_f("rope_factor", 1.0),
            original_seq_len: opt_u("original_seq_len", 0),
            compress_rope_theta: opt_f("compress_rope_theta", 10000.0),
            engram,
        };
        if cfg.n_routed_experts == 0 {
            return Err(Error::msg(format!(
                "{ARCH}: no routed experts - not a deepseek_v41 MoE"
            )));
        }
        Ok(cfg)
    }

    /// Read the config from any GGUF source. The keys are looked up under the file's own
    /// architecture tag - `deepseek4.*` or `deepseek41.*` from a conversion, `deepseek_v41.*` from
    /// the synthetic gates - and by each spelling a key has had. What a file leaves out is
    /// derived - band roles from the tensors a block carries, low-rank widths from the projection
    /// shapes - or, for the release constants no converter writes, taken from the model's own
    /// config; only the synthetic scheme writes everything and takes no such default.
    pub fn from_meta(src: &impl GgufMeta) -> Result<Self> {
        let md = src.metadata();
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| ARCH.to_string());
        let llamacpp = arch != ARCH;
        let g = |keys: &[&str]| keys.iter().find_map(|k| md.get(&format!("{arch}.{k}")));
        let req_u = |keys: &[&str]| -> Result<usize> {
            g(keys)
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .ok_or_else(|| Error::msg(format!("{arch}: missing {arch}.{}", keys[0])))
        };
        let opt_u = |keys: &[&str], d: usize| {
            g(keys)
                .and_then(|v| v.to_u32().ok())
                .map(|v| v as usize)
                .unwrap_or(d)
        };
        let opt_f = |keys: &[&str], d: f32| g(keys).and_then(|v| v.to_f32().ok()).unwrap_or(d);
        let opt_s = |keys: &[&str], d: &str| {
            g(keys)
                .and_then(|v| v.to_string().ok().cloned())
                .unwrap_or_else(|| d.to_string())
        };
        let opt_b = |keys: &[&str], d: bool| match g(keys) {
            Some(Value::Bool(b)) => *b,
            Some(v) => v.to_u32().map(|x| x != 0).unwrap_or(d),
            None => d,
        };
        let arr_u = |keys: &[&str]| -> Vec<usize> {
            match g(keys) {
                Some(Value::Array(a)) => a
                    .iter()
                    .filter_map(|v| v.to_u32().ok().map(|x| x as usize))
                    .collect(),
                _ => Vec::new(),
            }
        };
        let arr_f = |keys: &[&str]| -> Vec<f32> {
            match g(keys) {
                Some(Value::Array(a)) => a.iter().filter_map(|v| v.to_f32().ok()).collect(),
                _ => Vec::new(),
            }
        };
        // A tensor's shape by whichever spelling the file uses.
        let dims = |names: &[&str]| {
            names
                .iter()
                .find_map(|n| src.info(n))
                .map(|i| i.shape.dims().to_vec())
        };

        let n_layers = req_u(&["block_count"])?;
        let d_model = req_u(&["embedding_length"])?;
        let n_head = req_u(&["attention.head_count"])?;
        let head_dim = req_u(&["attention.key_length"])?;

        // Low-rank widths: the keys when written, else the projections' shapes. `attn_output_a`
        // is [o_groups * o_lora, n_heads * head_dim / o_groups], so both follow from it.
        let q_lora_rank = match opt_u(&["attention.q_lora_rank"], 0) {
            0 => dims(&["blk.0.attn_q_a", "blk.0.attn_q_a.weight"])
                .map(|d| d[0])
                .unwrap_or(0),
            v => v,
        };
        let (o_groups, o_lora_rank) = match (
            opt_u(&["attention.o_groups", "attention.output_group_count"], 0),
            opt_u(&["attention.o_lora_rank", "attention.output_lora_rank"], 0),
        ) {
            (0, _) | (_, 0) => match dims(&[
                "blk.0.attn_output_a",
                "blk.0.attn_output_a.weight",
                "blk.0.attn_o_a.weight",
            ]) {
                Some(d) if d.len() == 2 && d[1] > 0 && (n_head * head_dim) % d[1] == 0 => {
                    let groups = n_head * head_dim / d[1];
                    (groups, d[0] / groups)
                }
                _ => (1, 0),
            },
            (groups, rank) => (groups, rank),
        };

        // Band roles: the layer lists when written, else the tensors a block carries - a
        // kv-source has a compressor, an index-source an index query. The index keys are not
        // the mark: an index-source that reads another layer's keys carries none of its own.
        let has = |l: usize, real: &str, legacy: &str| {
            src.info(&format!("blk.{l}.{real}")).is_some()
                || src.info(&format!("blk.{l}.{real}.weight")).is_some()
                || src.info(&format!("blk.{l}.{legacy}")).is_some()
        };
        let mut kv_source_layers = arr_u(&["attention.kv_source_layers"]);
        if kv_source_layers.is_empty() {
            kv_source_layers = (0..n_layers)
                .filter(|&l| has(l, "attn_compressor_kv", "attn_compressor_kv.weight"))
                .collect();
        }
        let mut index_source_layers = arr_u(&["attention.index_source_layers"]);
        if index_source_layers.is_empty() {
            index_source_layers = (0..n_layers)
                .filter(|&l| has(l, "indexer.attn_q_b", "attn_indexer_q_b.weight"))
                .collect();
        }

        // The SwiGLU clamp is per layer in llama.cpp's file and one value in the synthetic ones.
        let swiglu_limits = {
            let per_layer = arr_f(&["swiglu_clamp_exp"]);
            if per_layer.len() >= n_layers {
                per_layer[..n_layers].to_vec()
            } else {
                vec![opt_f(&["expert_swiglu_limit"], 0.0); n_layers]
            }
        };
        let swiglu_limit = swiglu_limits.first().copied().unwrap_or(0.0);

        let candidate_source_layer =
            match g(&["attention.candidate_source_layer"]).and_then(|v| v.to_u32().ok()) {
                Some(v) => v as i64,
                None if llamacpp => release::CANDIDATE_SOURCE_LAYER,
                None => -1,
            };
        let vocab_size = match opt_u(&["vocab_size"], 0) {
            0 => dims(&["token_embd", "token_embd.weight"])
                .map(|d| d[0])
                .unwrap_or(0),
            v => v,
        };

        let cfg = Self {
            n_layers,
            n_mtp_layers: opt_u(&["nextn_predict_layers"], 0),
            d_model,
            vocab_size,
            rms_eps: opt_f(&["attention.layer_norm_rms_epsilon"], 1e-5) as f64,
            n_head,
            n_kv_head: opt_u(&["attention.head_count_kv"], 1),
            head_dim,
            rope_head_dim: opt_u(&["rope.dimension_count"], release::ROPE_HEAD_DIM),
            q_lora_rank,
            o_lora_rank,
            o_groups,
            window_size: opt_u(&["attention.sliding_window"], 0),
            n_routed_experts: opt_u(&["expert_count"], 0),
            n_shared_experts: opt_u(&["expert_shared_count"], 0),
            n_activated_experts: opt_u(&["expert_used_count"], 0),
            moe_inter_dim: opt_u(&["feed_forward_length", "expert_feed_forward_length"], 0),
            score_func: opt_s(&["expert_scoring_func"], "sqrtsoftplus"),
            route_scale: opt_f(&["expert_weights_scale"], 1.0),
            gate_temp: opt_f(&["expert_gate_temp"], 1.0),
            swiglu_limit,
            swiglu_limits,
            norm_topk: opt_b(&["expert_weights_norm", "expert_norm_topk_prob"], true),
            expert_cache_count: opt_u(&["expert_cache_count"], 0),
            compress_ratios: arr_u(&["attention.compress_ratios"]),
            hot_experts: (0..n_layers)
                .map(|l| arr_u(&[&format!("hot_experts.{l}")]))
                .collect(),
            kv_source_layers,
            index_source_layers,
            index_n_heads: opt_u(
                &[
                    "indexer.head_count",
                    "attention.indexer.head_count",
                    "attention.index_head_count",
                ],
                0,
            ),
            index_head_dim: opt_u(
                &[
                    "indexer.key_length",
                    "attention.indexer.key_length",
                    "attention.index_key_length",
                ],
                0,
            ),
            index_topk: opt_u(
                &[
                    "indexer.top_k",
                    "attention.indexer.top_k",
                    "attention.index_topk",
                ],
                0,
            ),
            candidate_source_layer,
            candidate_topk_blocks: opt_u(
                &["attention.candidate_topk_blocks"],
                if llamacpp {
                    release::CANDIDATE_TOPK_BLOCKS
                } else {
                    0
                },
            ),
            candidate_block_size: opt_u(
                &["attention.candidate_block_size"],
                if llamacpp {
                    release::CANDIDATE_BLOCK_SIZE
                } else {
                    0
                },
            ),
            hc_mult: opt_u(&["hyper_connection.count", "hyper_connection_mult"], 1),
            hc_sinkhorn_iters: opt_u(
                &[
                    "hyper_connection.sinkhorn_iterations",
                    "hyper_connection_sinkhorn_iters",
                ],
                0,
            ),
            hc_eps: opt_f(&["hyper_connection.epsilon"], 1e-6),
            rope_theta: opt_f(&["rope.freq_base"], 10000.0),
            rope_factor: opt_f(
                &["rope.scaling.factor"],
                if llamacpp { release::ROPE_FACTOR } else { 1.0 },
            ),
            original_seq_len: opt_u(
                &["rope.scaling.original_context_length"],
                if llamacpp {
                    release::ORIGINAL_SEQ_LEN
                } else {
                    0
                },
            ),
            compress_rope_theta: opt_f(
                &[
                    "attention.compress_rope_freq_base",
                    "attention.compress_rope_theta",
                ],
                10000.0,
            ),
            engram: engram::EngramConfig::read(md, &arch),
        };
        // The one shape that must never be guessed: a shared expert is what makes this a
        // deepseek MoE rather than a plain one, and a config that read zero routed experts is
        // not this architecture at all.
        if cfg.n_routed_experts == 0 {
            return Err(Error::msg(format!(
                "{ARCH}: no routed experts - not a deepseek_v41 MoE"
            )));
        }
        Ok(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::Content;
    use crate::tensor::quantized::GgmlDType;

    /// A synthetic GGUF carrying the released shapes, parsed back into the config: the phase-1
    /// gate. The values are `inference/config.json` from the reference; a converter would write
    /// exactly these. Built with the metadata writer, so no real weights are needed to prove the
    /// shapes are read correctly.
    #[test]
    fn the_released_shapes_parse_from_a_synthetic_gguf() {
        let u = |k: &str, v: u32| (format!("{ARCH}.{k}"), Value::U32(v));
        let f = |k: &str, v: f32| (format!("{ARCH}.{k}"), Value::F32(v));
        let arr = |k: &str, xs: &[u32]| {
            (
                format!("{ARCH}.{k}"),
                Value::Array(xs.iter().map(|x| Value::U32(*x)).collect()),
            )
        };
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String(ARCH.into()),
            ),
            u("block_count", 40),
            u("nextn_predict_layers", 3),
            u("embedding_length", 5120),
            u("vocab_size", 129280),
            f("attention.layer_norm_rms_epsilon", 1e-20),
            u("attention.head_count", 64),
            u("attention.head_count_kv", 1),
            u("attention.key_length", 512),
            u("rope.dimension_count", 64),
            u("attention.q_lora_rank", 1280),
            u("attention.o_lora_rank", 1024),
            u("attention.o_groups", 8),
            u("attention.sliding_window", 128),
            u("expert_count", 384),
            u("expert_shared_count", 1),
            u("expert_used_count", 6),
            u("expert_feed_forward_length", 2304),
            (
                format!("{ARCH}.expert_scoring_func"),
                Value::String("sqrtsoftplus".into()),
            ),
            f("expert_weights_scale", 1.5),
            arr("attention.kv_source_layers", &[2, 8, 14, 20]),
            arr(
                "attention.index_source_layers",
                &[2, 8, 14, 20, 24, 28, 32, 36],
            ),
            u("attention.index_head_count", 32),
            u("attention.index_key_length", 128),
            u("attention.index_topk", 512),
            u("hyper_connection_mult", 4),
            u("hyper_connection_sinkhorn_iters", 20),
            f("rope.freq_base", 10000.0),
            f("rope.scaling.factor", 16.0),
            u("rope.scaling.original_context_length", 65536),
            f("attention.compress_rope_theta", 160000.0),
        ];
        let path = std::env::temp_dir().join(format!("loken-dsv41-{}.gguf", std::process::id()));
        let entries = vec![GgufEntry {
            name: "token_embd.weight".to_string(),
            dims: vec![1],
            dtype: GgmlDType::F32,
            data: 0f32.to_le_bytes().to_vec(),
        }];
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let mut file = std::fs::File::open(&path).unwrap();
        let content = Content::read(&mut file).unwrap();
        let _ = std::fs::remove_file(&path);

        let c = DeepseekV41Config::from_gguf(&content).unwrap();
        assert_eq!(c.n_layers, 40);
        assert_eq!(c.n_mtp_layers, 3);
        assert_eq!(c.d_model, 5120);
        assert_eq!(c.n_head, 64);
        assert_eq!(c.n_kv_head, 1);
        assert_eq!(c.head_dim, 512);
        assert_eq!(c.rope_head_dim, 64);
        assert_eq!(c.q_lora_rank, 1280);
        assert_eq!(c.o_lora_rank, 1024);
        assert_eq!(c.o_groups, 8);
        assert_eq!(c.window_size, 128);
        assert_eq!(c.n_routed_experts, 384);
        assert_eq!(c.n_shared_experts, 1);
        assert_eq!(c.n_activated_experts, 6);
        assert_eq!(c.moe_inter_dim, 2304);
        assert_eq!(c.score_func, "sqrtsoftplus");
        assert_eq!(c.route_scale, 1.5);
        assert_eq!(c.kv_source_layers, vec![2, 8, 14, 20]);
        assert_eq!(c.index_source_layers, vec![2, 8, 14, 20, 24, 28, 32, 36]);
        assert_eq!(c.index_n_heads, 32);
        assert_eq!(c.index_head_dim, 128);
        assert_eq!(c.index_topk, 512);
        assert_eq!(c.hc_mult, 4);
        assert_eq!(c.hc_sinkhorn_iters, 20);
        assert_eq!(c.original_seq_len, 65536);
        assert_eq!(c.compress_rope_theta, 160000.0);
        // rms_eps is stored as f32 in the file, so compare relative, not to f64 precision.
        assert!(
            (c.rms_eps / 1e-20 - 1.0).abs() < 1e-3,
            "rms_eps was {}",
            c.rms_eps
        );
    }

    /// A config with no routed experts is not this architecture; the parse says so rather than
    /// building a silently wrong model.
    #[test]
    fn a_config_without_routed_experts_is_refused() {
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String(ARCH.into()),
            ),
            (format!("{ARCH}.block_count"), Value::U32(40)),
            (format!("{ARCH}.embedding_length"), Value::U32(5120)),
            (format!("{ARCH}.attention.head_count"), Value::U32(64)),
            (format!("{ARCH}.attention.key_length"), Value::U32(512)),
        ];
        let path =
            std::env::temp_dir().join(format!("loken-dsv41-bad-{}.gguf", std::process::id()));
        let entries = vec![GgufEntry {
            name: "token_embd.weight".to_string(),
            dims: vec![1],
            dtype: GgmlDType::F32,
            data: 0f32.to_le_bytes().to_vec(),
        }];
        write_gguf_with_metadata(&path, &md, &entries).unwrap();
        let mut file = std::fs::File::open(&path).unwrap();
        let content = Content::read(&mut file).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(DeepseekV41Config::from_gguf(&content).is_err());
    }
}
