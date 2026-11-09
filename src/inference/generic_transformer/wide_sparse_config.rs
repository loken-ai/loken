//! The description of a wide-residual mixture model with block-sparse attention, read from a
//! file's header.
//!
//! Such a model keeps its residual as several parallel streams mixed by a Sinkhorn projection.
//! Its attention is latent and shares 1 key row among all heads. A layer attends over a sliding
//! window alone, or also over a compressed copy of the past that a sparse indexer selects from.
//! Its feed-forward block is a shared expert beside routed ones under a sqrt-softplus router.
//! Some layers also hash the last tokens into an n-gram table.
//!
//! The description holds the layer kinds, the widths, the rotary parameters, the mixture, the
//! indexer and compressor fields, the stream mixing and the n-gram hash. Nothing here loads a
//! weight.

use super::ngram::engram::EngramMeta;
use super::wide_sparse_llamacpp::scoring_func;
use super::wide_sparse_names::{ARCH, HYPER_CONNECTION_MARKER};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::gguf_source::GgufMeta;
use crate::tensor::{Error, Result};

/// What a layer attends over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// The sliding window alone.
    Window,
    /// The window and a compressed copy of the past, 1 row per `ratio` tokens.
    Compressed {
        ratio: usize,
        /// The layer builds the compressed rows that later layers read.
        kv_source: bool,
        /// The layer builds the index keys that later layers read.
        index_source: bool,
    },
}

/// The sparse indexer and the candidate pre-filter beside it.
#[derive(Debug, Clone, PartialEq)]
pub struct IndexerFields {
    /// Query heads of the indexer.
    pub heads: usize,
    /// Width of an indexer head.
    pub head_dim: usize,
    /// Compressed rows a query attends to at most.
    pub top_k: usize,
    /// The layer whose index feeds the candidate pre-filter, `-1` when the pre-filter is off.
    pub candidate_source_layer: i64,
    /// Blocks the pre-filter keeps.
    pub candidate_top_k_blocks: usize,
    /// Rows in a block of the pre-filter.
    pub candidate_block_size: usize,
}

/// The routed and shared experts of the feed-forward block.
#[derive(Debug, Clone, PartialEq)]
pub struct MixtureFields {
    pub routed: usize,
    pub shared: usize,
    pub used: usize,
    pub inter_dim: usize,
    pub score_func: String,
    pub route_scale: f32,
    pub gate_temp: f32,
    /// Per layer, the clamp of the SwiGLU activation.
    pub swiglu_limits: Vec<f32>,
    /// Whether the selected weights are renormalised to sum to 1.
    pub norm_topk: bool,
    /// Routed experts the file asks to keep resident, `0` for all of them.
    pub cache_count: usize,
    /// Per layer, the experts from the most to the least routed to; empty without the prior.
    pub hot: Vec<Vec<usize>>,
}

/// The rotary parameters of the window rows and of the compressed rows.
#[derive(Debug, Clone, PartialEq)]
pub struct RopeFields {
    pub theta: f32,
    pub scaling_factor: f32,
    pub original_context: usize,
    /// The base the compressed rows rotate at.
    pub compress_theta: f32,
}

/// The stream mixing of the residual.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamFields {
    /// Parallel copies of the residual.
    pub streams: usize,
    pub sinkhorn_iters: usize,
    pub eps: f32,
}

/// The n-gram hash of the layers that read the last tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct NgramHashFields {
    pub layer_ids: Vec<usize>,
    pub heads: usize,
    pub head_dim: usize,
    pub max_ngram: usize,
    /// Token id to compressed id, over the whole vocabulary.
    pub token_map: Vec<u32>,
    /// The compressed id an n-gram slot takes before the sequence starts.
    pub pad_id: u32,
    pub multipliers: Vec<Vec<i64>>,
    pub primes: Vec<Vec<u64>>,
    pub offsets: Vec<Vec<u64>>,
}

impl From<&EngramMeta> for NgramHashFields {
    fn from(m: &EngramMeta) -> Self {
        Self {
            layer_ids: m.layer_ids.clone(),
            heads: m.heads,
            head_dim: m.head_dim,
            max_ngram: m.max_ngram,
            token_map: m.token_map.clone(),
            pad_id: m.pad_id,
            multipliers: m.multipliers.clone(),
            primes: m.primes.clone(),
            offsets: m.offsets.clone(),
        }
    }
}

/// Values llama.cpp's `deepseek4` converter does not write into the file.
///
/// They are the V4.1-Flash release's, from the model's own inference config.
/// They are read only when the key is absent from one of that converter's files.
pub mod release {
    pub const ROPE_HEAD_DIM: usize = 64;
    pub const ROPE_FACTOR: f32 = 16.0;
    pub const ORIGINAL_SEQ_LEN: usize = 65536;
    pub const CANDIDATE_SOURCE_LAYER: i64 = 20;
    pub const CANDIDATE_TOPK_BLOCKS: usize = 2048;
    pub const CANDIDATE_BLOCK_SIZE: usize = 8;
}

/// The values a converter does not write, taken from the release it targets. `rope_dim` fills
/// any file that omits it. The others fill only a converted file, as a file that writes every
/// key takes no default for them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReleaseValues {
    pub rope_dim: usize,
    pub rope_scaling_factor: f32,
    pub original_context: usize,
    pub candidate_source_layer: i64,
    pub candidate_top_k_blocks: usize,
    pub candidate_block_size: usize,
}

/// The whole description.
#[derive(Debug, Clone, PartialEq)]
pub struct WideSparseMeta {
    pub n_layers: usize,
    /// Layers that predict further tokens, beyond the ones above.
    pub n_extra_layers: usize,
    pub d_model: usize,
    pub vocab: usize,
    pub rms_eps: f64,
    pub heads: usize,
    pub head_dim: usize,
    /// The rotated lanes of a head.
    pub rope_dim: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub o_groups: usize,
    pub window: usize,
    /// Per layer, tokens per compressed row, `0` for a window layer.
    pub compress_ratios: Vec<usize>,
    pub kv_source_layers: Vec<usize>,
    pub index_source_layers: Vec<usize>,
    pub indexer: IndexerFields,
    pub mixture: MixtureFields,
    pub rope: RopeFields,
    pub streams: StreamFields,
    pub ngram: Option<NgramHashFields>,
}

impl WideSparseMeta {
    /// What layer `il` attends over.
    pub fn layer_kind(&self, il: usize) -> LayerKind {
        match self.compress_ratios.get(il).copied().unwrap_or(0) {
            0 => LayerKind::Window,
            ratio => LayerKind::Compressed {
                ratio,
                kv_source: self.kv_source_layers.contains(&il),
                index_source: self.index_source_layers.contains(&il),
            },
        }
    }

    /// Whether a file is a wide-residual model of this family: block 0 carries the stream mix
    /// under one of its spellings, or the metadata counts streams without a low-rank gate.
    pub fn claims(src: &impl GgufMeta) -> bool {
        let md = src.metadata();
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok());
        let has = |key: &str| arch.is_some_and(|a| md.contains_key(&format!("{a}.{key}")));
        let mix = ["blk.0.hc_attn_fn", "blk.0.hc_attn_fn.weight"]
            .iter()
            .any(|n| src.info(n).is_some());
        mix || ((has("hyper_connection.count") || has("hyper_connection_mult"))
            && !has("hyper_connection.low_rank"))
    }

    /// Whether a checkpoint directory holds released weights of this family.
    ///
    /// Its shard index names block 0's stream mix.
    /// Without an index, the header of one of its shards names it.
    pub fn claims_checkpoint(dir: &std::path::Path) -> bool {
        if !dir.is_dir() {
            return false;
        }
        if let Ok(txt) = std::fs::read_to_string(dir.join("model.safetensors.index.json")) {
            return serde_json::from_str::<serde_json::Value>(&txt)
                .ok()
                .and_then(|v| {
                    v.get("weight_map")
                        .map(|m| m.get(HYPER_CONNECTION_MARKER).is_some())
                })
                .unwrap_or(false);
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
        entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .any(|p| {
                super::dense::header_names(&p)
                    .is_some_and(|n| n.iter().any(|t| t == HYPER_CONNECTION_MARKER))
            })
    }

    /// Read the description from a file's header and tensor directory, as the family's own reader
    /// reads it.
    ///
    /// A file whose architecture tag is not the in-tree scheme's is a converted one.
    /// It takes the release values for the keys its converter does not write.
    pub fn from_file(src: &impl GgufMeta) -> Result<Self> {
        let converted = src
            .metadata()
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .is_some_and(|a| a != ARCH);
        let values = ReleaseValues {
            rope_dim: release::ROPE_HEAD_DIM,
            rope_scaling_factor: release::ROPE_FACTOR,
            original_context: release::ORIGINAL_SEQ_LEN,
            candidate_source_layer: release::CANDIDATE_SOURCE_LAYER,
            candidate_top_k_blocks: release::CANDIDATE_TOPK_BLOCKS,
            candidate_block_size: release::CANDIDATE_BLOCK_SIZE,
        };
        Self::from_meta(src, ARCH, &values, converted)
    }

    /// Read the description from a file's header and tensor directory, keys looked up under the
    /// file's own architecture tag and by each spelling a key has had. What the file leaves out
    /// is derived from the tensors a layer carries and from the projection shapes. What no
    /// converter writes comes from `release`, for the fields `converted` covers.
    pub fn from_meta(
        src: &impl GgufMeta,
        arch_default: &str,
        release: &ReleaseValues,
        converted: bool,
    ) -> Result<Self> {
        let omitted = converted.then_some(release);
        let md = src.metadata();
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| arch_default.to_string());
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
        let dims = |names: &[&str]| {
            names
                .iter()
                .find_map(|n| src.info(n))
                .map(|i| i.shape.dims().to_vec())
        };

        let n_layers = req_u(&["block_count"])?;
        let d_model = req_u(&["embedding_length"])?;
        let heads = req_u(&["attention.head_count"])?;
        let head_dim = req_u(&["attention.key_length"])?;

        // The low-rank widths follow from the keys when written, else from the projections.
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
                Some(d) if d.len() == 2 && d[1] > 0 && (heads * head_dim) % d[1] == 0 => {
                    let groups = heads * head_dim / d[1];
                    (groups, d[0] / groups)
                }
                _ => (1, 0),
            },
            (groups, rank) => (groups, rank),
        };

        // A source layer is listed, or else carries the tensors that build what it shares.
        let has = |l: usize, name: &str, legacy: &str| {
            src.info(&format!("blk.{l}.{name}")).is_some()
                || src.info(&format!("blk.{l}.{name}.weight")).is_some()
                || src.info(&format!("blk.{l}.{legacy}")).is_some()
        };
        let mut kv_source_layers = arr_u(&["attention.kv_source_layers"]);
        if kv_source_layers.is_empty() {
            kv_source_layers = (0..n_layers)
                .filter(|&l| has(l, "attn_compressor_kv", "attn_compressor_kv.weight"))
                .collect();
        }
        let mut index_source_layers = arr_u(&[
            "attention.index_source_layers",
            "attention.indexer.source_layers",
        ]);
        if index_source_layers.is_empty() {
            index_source_layers = (0..n_layers)
                .filter(|&l| has(l, "indexer.attn_q_b", "attn_indexer_q_b.weight"))
                .collect();
        }

        let swiglu_limits = {
            let per_layer = arr_f(&["swiglu_clamp_exp"]);
            if per_layer.len() >= n_layers {
                per_layer[..n_layers].to_vec()
            } else {
                vec![opt_f(&["expert_swiglu_limit"], 0.0); n_layers]
            }
        };
        let candidate_source_layer = match g(&[
            "attention.candidate_source_layer",
            "attention.indexer.candidate_source_layer",
        ])
        .and_then(|v| v.to_i64().ok())
        {
            Some(v) => v,
            None => omitted.map_or(-1, |o| o.candidate_source_layer),
        };
        let score_func = match g(&["expert_scoring_func"]) {
            Some(v) => v.to_string().ok().cloned(),
            None => g(&["expert_gating_func"])
                .and_then(|v| v.to_u32().ok())
                .and_then(scoring_func)
                .map(str::to_string),
        }
        .unwrap_or_else(|| "sqrtsoftplus".to_string());
        let vocab = match opt_u(&["vocab_size"], 0) {
            0 => dims(&["token_embd", "token_embd.weight"])
                .map(|d| d[0])
                .unwrap_or(0),
            v => v,
        };

        let meta = Self {
            n_layers,
            n_extra_layers: opt_u(&["nextn_predict_layers"], 0),
            d_model,
            vocab,
            rms_eps: opt_f(&["attention.layer_norm_rms_epsilon"], 1e-5) as f64,
            heads,
            head_dim,
            rope_dim: opt_u(&["rope.dimension_count"], release.rope_dim),
            q_lora_rank,
            o_lora_rank,
            o_groups,
            window: opt_u(&["attention.sliding_window"], 0),
            compress_ratios: arr_u(&["attention.compress_ratios"]),
            kv_source_layers,
            index_source_layers,
            indexer: IndexerFields {
                heads: opt_u(
                    &[
                        "indexer.head_count",
                        "attention.indexer.head_count",
                        "attention.index_head_count",
                    ],
                    0,
                ),
                head_dim: opt_u(
                    &[
                        "indexer.key_length",
                        "attention.indexer.key_length",
                        "attention.index_key_length",
                    ],
                    0,
                ),
                top_k: opt_u(
                    &[
                        "indexer.top_k",
                        "attention.indexer.top_k",
                        "attention.index_topk",
                    ],
                    0,
                ),
                candidate_source_layer,
                candidate_top_k_blocks: opt_u(
                    &[
                        "attention.candidate_topk_blocks",
                        "attention.indexer.candidate_top_k_blocks",
                    ],
                    omitted.map_or(0, |o| o.candidate_top_k_blocks),
                ),
                candidate_block_size: opt_u(
                    &[
                        "attention.candidate_block_size",
                        "attention.indexer.candidate_block_size",
                    ],
                    omitted.map_or(0, |o| o.candidate_block_size),
                ),
            },
            mixture: MixtureFields {
                routed: opt_u(&["expert_count"], 0),
                shared: opt_u(&["expert_shared_count"], 0),
                used: opt_u(&["expert_used_count"], 0),
                inter_dim: opt_u(&["feed_forward_length", "expert_feed_forward_length"], 0),
                score_func,
                route_scale: opt_f(&["expert_weights_scale"], 1.0),
                gate_temp: opt_f(&["expert_gate_temp"], 1.0),
                swiglu_limits,
                norm_topk: opt_b(&["expert_weights_norm", "expert_norm_topk_prob"], true),
                cache_count: opt_u(&["expert_cache_count"], 0),
                hot: (0..n_layers)
                    .map(|l| arr_u(&[&format!("hot_experts.{l}")]))
                    .collect(),
            },
            rope: RopeFields {
                theta: opt_f(&["rope.freq_base"], 10000.0),
                scaling_factor: opt_f(
                    &["rope.scaling.factor"],
                    omitted.map_or(1.0, |o| o.rope_scaling_factor),
                ),
                original_context: opt_u(
                    &["rope.scaling.original_context_length"],
                    omitted.map_or(0, |o| o.original_context),
                ),
                compress_theta: opt_f(
                    &[
                        "attention.compress_rope_freq_base",
                        "attention.compress_rope_theta",
                    ],
                    10000.0,
                ),
            },
            streams: StreamFields {
                streams: opt_u(&["hyper_connection.count", "hyper_connection_mult"], 1),
                sinkhorn_iters: opt_u(
                    &[
                        "hyper_connection.sinkhorn_iterations",
                        "hyper_connection_sinkhorn_iters",
                    ],
                    0,
                ),
                eps: opt_f(&["hyper_connection.epsilon"], 1e-6),
            },
            ngram: EngramMeta::from_metadata(md, &arch)
                .as_ref()
                .map(NgramHashFields::from),
        };
        meta.check()?;
        Ok(meta)
    }

    /// Read the description from the reference arguments of a released checkpoint
    /// (`inference/config.json`), under their own names. `ngram` is the hash layout the
    /// caller derived from the checkpoint, `None` for a model run without its hash layers.
    pub fn from_reference(v: &serde_json::Value, ngram: Option<NgramHashFields>) -> Result<Self> {
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
        let n_layers = req_u("n_layers")?;
        let swiglu_limit = opt_f("swiglu_limit", 0.0);
        let meta = Self {
            n_layers,
            n_extra_layers: opt_u("n_mtp_layers", 0),
            d_model: req_u("dim")?,
            vocab: req_u("vocab_size")?,
            rms_eps: opt_f("norm_eps", 1e-20) as f64,
            heads: req_u("n_heads")?,
            head_dim: req_u("head_dim")?,
            rope_dim: req_u("rope_head_dim")?,
            q_lora_rank: req_u("q_lora_rank")?,
            o_lora_rank: req_u("o_lora_rank")?,
            o_groups: req_u("o_groups")?,
            window: opt_u("window_size", 0),
            compress_ratios: arr_u("compress_ratios"),
            kv_source_layers: arr_u("kv_source_layers"),
            index_source_layers: arr_u("index_source_layers"),
            indexer: IndexerFields {
                heads: opt_u("index_n_heads", 0),
                head_dim: opt_u("index_head_dim", 0),
                top_k: opt_u("index_topk", 0),
                candidate_source_layer: opt_i("candidate_source_layer", -1),
                candidate_top_k_blocks: opt_u("candidate_topk_blocks", 0),
                candidate_block_size: opt_u("candidate_block_size", 0),
            },
            mixture: MixtureFields {
                routed: req_u("n_routed_experts")?,
                shared: opt_u("n_shared_experts", 1),
                used: req_u("n_activated_experts")?,
                inter_dim: req_u("moe_inter_dim")?,
                score_func: opt_s("score_func", "sqrtsoftplus"),
                route_scale: opt_f("route_scale", 1.0),
                gate_temp: opt_f("gate_temp", 1.0),
                swiglu_limits: vec![swiglu_limit; n_layers],
                norm_topk: opt_b("norm_topk_prob", true),
                cache_count: 0,
                hot: Vec::new(),
            },
            rope: RopeFields {
                theta: opt_f("rope_theta", 10000.0),
                scaling_factor: opt_f("rope_factor", 1.0),
                original_context: opt_u("original_seq_len", 0),
                compress_theta: opt_f("compress_rope_theta", 10000.0),
            },
            streams: StreamFields {
                streams: opt_u("hc_mult", 1),
                sinkhorn_iters: opt_u("hc_sinkhorn_iters", 0),
                eps: opt_f("hc_eps", 1e-6),
            },
            ngram,
        };
        meta.check()?;
        Ok(meta)
    }

    /// Bytes layer `il` holds of what `tokens` tokens leave, every value f32.
    ///
    /// Every layer keeps its window of key rows, `head_dim` wide.
    /// A kv-source also keeps 1 compressed row per `ratio` tokens, `head_dim` wide.
    /// A kv-source that is an index-source keeps 1 index key per compressed row,
    /// `index_head_dim` wide.
    /// A kv-source keeps the input rows of the group still filling, `d_model` wide.
    pub fn cache_bytes(&self, il: usize, tokens: usize) -> u64 {
        let window = tokens.min(self.window) * self.head_dim;
        let band = match self.layer_kind(il) {
            LayerKind::Compressed {
                ratio,
                kv_source: true,
                index_source,
            } => {
                let groups = tokens / ratio;
                let keys = if index_source {
                    groups * self.indexer.head_dim
                } else {
                    0
                };
                groups * self.head_dim + keys + tokens % ratio * self.d_model
            }
            _ => 0,
        };
        ((window + band) * std::mem::size_of::<f32>()) as u64
    }

    /// The cache bytes a plan reserves for each layer at `tokens`: the largest layer's.
    pub fn kv_bytes_per_layer(&self, tokens: usize) -> u64 {
        (0..self.n_layers)
            .map(|il| self.cache_bytes(il, tokens))
            .max()
            .unwrap_or(0)
    }

    /// Bytes a token adds to a stored state's blocks, at least 1.
    ///
    /// Per layer that compresses, 1 compressed row and 1 index key per `ratio` tokens, in f32.
    pub fn state_bytes_per_token(&self) -> usize {
        self.compress_ratios
            .iter()
            .take(self.n_layers)
            .filter(|&&r| r > 0)
            .map(|&r| (self.head_dim + self.indexer.head_dim) * std::mem::size_of::<f32>() / r)
            .sum::<usize>()
            .max(1)
    }

    /// A description of a model without routed experts is not of this family.
    fn check(&self) -> Result<()> {
        if self.mixture.routed == 0 {
            return Err(Error::msg(
                "wide-residual model without routed experts: not a mixture of experts",
            ));
        }
        Ok(())
    }
}
