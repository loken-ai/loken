//! The generic description of a wide-residual mixture model against the configuration the
//! dedicated module reads from the same header, field by field.
//!
//! The headers are the synthetic ones of the dedicated module's tests, in each spelling a file
//! has had: the converter's, the mainline conversion's and the in-tree scheme's.

use std::collections::HashMap;

use super::wide_sparse_config::{LayerKind, ReleaseValues, WideSparseMeta};
use crate::inference::model::deepseek_v41::model::tests::{real_entries_for, real_md_layout};
use crate::inference::model::deepseek_v41::{release, DeepseekV41Config, ARCH};
use crate::tensor::gguf_write::GgufEntry;
use crate::tensor::quantized::gguf_file::{TensorInfo, Value};
use crate::tensor::quantized::gguf_source::GgufMeta;
use crate::tensor::quantized::GgmlDType;

/// A header and a tensor directory, nothing behind them.
struct Header {
    md: HashMap<String, Value>,
    infos: HashMap<String, TensorInfo>,
}

impl Header {
    fn new(md: Vec<(String, Value)>, entries: &[GgufEntry]) -> Self {
        Self {
            md: md.into_iter().collect(),
            infos: entries
                .iter()
                .map(|e| {
                    let info = TensorInfo {
                        ggml_dtype: GgmlDType::F32,
                        shape: e.dims.clone().into(),
                        offset: 0,
                    };
                    (e.name.clone(), info)
                })
                .collect(),
        }
    }
}

impl GgufMeta for Header {
    fn metadata(&self) -> &HashMap<String, Value> {
        &self.md
    }
    fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.infos.get(name)
    }
    fn tensor_names(&self) -> Vec<String> {
        self.infos.keys().cloned().collect()
    }
}

fn release_values() -> ReleaseValues {
    ReleaseValues {
        rope_dim: release::ROPE_HEAD_DIM,
        rope_scaling_factor: release::ROPE_FACTOR,
        original_context: release::ORIGINAL_SEQ_LEN,
        candidate_source_layer: release::CANDIDATE_SOURCE_LAYER,
        candidate_top_k_blocks: release::CANDIDATE_TOPK_BLOCKS,
        candidate_block_size: release::CANDIDATE_BLOCK_SIZE,
    }
}

/// The generic description of `h`, read the way the dedicated module's callers read it: a file
/// whose tag is not the in-tree one is a converted file.
fn generic(h: &Header) -> WideSparseMeta {
    let arch =
        h.md.get("general.architecture")
            .and_then(|v| v.to_string().ok());
    let converted = arch.is_some_and(|a| a != ARCH);
    WideSparseMeta::from_meta(h, ARCH, &release_values(), converted).unwrap()
}

/// Every field of `want` and the one of `got` that holds it.
fn assert_same(got: &WideSparseMeta, want: &DeepseekV41Config) {
    assert_eq!(got.n_layers, want.n_layers);
    assert_eq!(got.n_extra_layers, want.n_mtp_layers);
    assert_eq!(got.d_model, want.d_model);
    assert_eq!(got.vocab, want.vocab_size);
    assert_eq!(got.rms_eps, want.rms_eps);
    assert_eq!(got.heads, want.n_head);
    assert_eq!(got.head_dim, want.head_dim);
    assert_eq!(got.rope_dim, want.rope_head_dim);
    assert_eq!(got.q_lora_rank, want.q_lora_rank);
    assert_eq!(got.o_lora_rank, want.o_lora_rank);
    assert_eq!(got.o_groups, want.o_groups);
    assert_eq!(got.window, want.window_size);
    assert_eq!(got.compress_ratios, want.compress_ratios);
    assert_eq!(got.kv_source_layers, want.kv_source_layers);
    assert_eq!(got.index_source_layers, want.index_source_layers);

    assert_eq!(got.indexer.heads, want.index_n_heads);
    assert_eq!(got.indexer.head_dim, want.index_head_dim);
    assert_eq!(got.indexer.top_k, want.index_topk);
    assert_eq!(
        got.indexer.candidate_source_layer,
        want.candidate_source_layer
    );
    assert_eq!(
        got.indexer.candidate_top_k_blocks,
        want.candidate_topk_blocks
    );
    assert_eq!(got.indexer.candidate_block_size, want.candidate_block_size);

    let m = &got.mixture;
    assert_eq!(m.routed, want.n_routed_experts);
    assert_eq!(m.shared, want.n_shared_experts);
    assert_eq!(m.used, want.n_activated_experts);
    assert_eq!(m.inter_dim, want.moe_inter_dim);
    assert_eq!(m.score_func, want.score_func);
    assert_eq!(m.route_scale, want.route_scale);
    assert_eq!(m.gate_temp, want.gate_temp);
    assert_eq!(m.swiglu_limits, want.swiglu_limits);
    assert_eq!(m.norm_topk, want.norm_topk);
    assert_eq!(m.cache_count, want.expert_cache_count);
    assert_eq!(m.hot, want.hot_experts);

    assert_eq!(got.rope.theta, want.rope_theta);
    assert_eq!(got.rope.scaling_factor, want.rope_factor);
    assert_eq!(got.rope.original_context, want.original_seq_len);
    assert_eq!(got.rope.compress_theta, want.compress_rope_theta);

    assert_eq!(got.streams.streams, want.hc_mult);
    assert_eq!(got.streams.sinkhorn_iters, want.hc_sinkhorn_iters);
    assert_eq!(got.streams.eps, want.hc_eps);

    match (&got.ngram, &want.engram) {
        (None, None) => {}
        (Some(g), Some(w)) => {
            assert_eq!(g.layer_ids, w.layer_ids);
            assert_eq!(g.heads, w.n_heads);
            assert_eq!(g.head_dim, w.head_dim);
            assert_eq!(g.max_ngram, w.max_ngram);
            assert_eq!(g.token_map, w.token_map);
            assert_eq!(g.pad_id, w.pad_id);
            assert_eq!(g.multipliers, w.multipliers);
            assert_eq!(g.primes, w.primes);
            assert_eq!(g.offsets, w.offsets);
        }
        (g, w) => panic!("n-gram hash {g:?} against {w:?}"),
    }
}

fn check(h: &Header) -> WideSparseMeta {
    let want = DeepseekV41Config::from_meta(h).unwrap();
    let got = generic(h);
    assert_same(&got, &want);
    got
}

fn bands(ratios: &[u32]) -> Vec<bool> {
    ratios.iter().map(|&r| r != 0).collect()
}

/// The converter's spelling, nothing but the keys it writes: the low-rank widths and the layer
/// roles come from the tensors.
#[test]
fn the_converters_header_gives_the_same_description() {
    let ratios = [0, 2, 0];
    let h = Header::new(
        real_md_layout(&ratios, &[], &[]),
        &real_entries_for(&bands(&ratios)),
    );
    let got = check(&h);
    assert_eq!(
        (got.q_lora_rank, got.o_groups, got.o_lora_rank),
        (64, 4, 64)
    );
    assert_eq!(got.kv_source_layers, vec![1]);
    assert_eq!(got.index_source_layers, vec![1]);
    assert_eq!(got.rope.original_context, release::ORIGINAL_SEQ_LEN);
    assert_eq!(
        got.indexer.candidate_source_layer,
        release::CANDIDATE_SOURCE_LAYER
    );
    assert!(WideSparseMeta::claims(&h));
}

/// A layout that lists its source layers, with a reader between 2 sources.
#[test]
fn listed_source_layers_give_the_same_description() {
    let ratios = [0, 2, 2, 0];
    let h = Header::new(
        real_md_layout(&ratios, &[1], &[1, 3]),
        &real_entries_for(&bands(&ratios)),
    );
    let got = check(&h);
    assert_eq!(got.layer_kind(0), LayerKind::Window);
    assert_eq!(
        got.layer_kind(1),
        LayerKind::Compressed {
            ratio: 2,
            kv_source: true,
            index_source: true
        }
    );
    assert_eq!(
        got.layer_kind(2),
        LayerKind::Compressed {
            ratio: 2,
            kv_source: false,
            index_source: false
        }
    );
    assert_eq!(got.layer_kind(3), LayerKind::Window);
}

/// The mainline conversion: suffixed tensor names, the tag with the model's version, widths
/// written as keys, integer arrays signed.
#[test]
fn the_mainline_header_gives_the_same_description() {
    let ratios = [0, 2, 0];
    let entries: Vec<GgufEntry> = real_entries_for(&bands(&ratios))
        .into_iter()
        .map(|mut e| {
            e.name = if e.name.ends_with("exp_probs_b") {
                format!("{}.bias", e.name)
            } else {
                format!("{}.weight", e.name)
            };
            e
        })
        .collect();
    let mut md: Vec<(String, Value)> = real_md_layout(&ratios, &[], &[])
        .into_iter()
        .map(|(k, v)| match (k.as_str(), v) {
            ("general.architecture", _) => (k, Value::String("deepseek41".into())),
            ("deepseek4.attention.compress_ratios", Value::Array(a)) => (
                "deepseek41.attention.compress_ratios".into(),
                Value::Array(
                    a.iter()
                        .map(|v| Value::I32(v.to_u32().unwrap() as i32))
                        .collect(),
                ),
            ),
            (_, v) => (
                k.replacen("deepseek4.indexer.", "deepseek41.attention.indexer.", 1)
                    .replacen("deepseek4.", "deepseek41.", 1),
                v,
            ),
        })
        .collect();
    for (k, v) in [
        ("attention.q_lora_rank", 64),
        ("attention.output_group_count", 4),
        ("attention.output_lora_rank", 64),
    ] {
        md.push((format!("deepseek41.{k}"), Value::U32(v)));
    }
    let h = Header::new(md, &entries);
    let got = check(&h);
    assert_eq!(
        (got.q_lora_rank, got.o_groups, got.o_lora_rank),
        (64, 4, 64)
    );
}

/// The in-tree scheme: every key written, the clamp and the stream count under their own names,
/// and no default taken for what is absent.
#[test]
fn the_in_tree_header_gives_the_same_description() {
    let u = |k: &str, v: u32| (format!("{ARCH}.{k}"), Value::U32(v));
    let f = |k: &str, v: f32| (format!("{ARCH}.{k}"), Value::F32(v));
    let arr = |k: &str, xs: &[u32]| {
        (
            format!("{ARCH}.{k}"),
            Value::Array(xs.iter().map(|&x| Value::U32(x)).collect()),
        )
    };
    let md = vec![
        (
            "general.architecture".to_string(),
            Value::String(ARCH.into()),
        ),
        u("block_count", 3),
        u("embedding_length", 256),
        u("attention.head_count", 4),
        u("attention.key_length", 128),
        u("expert_count", 8),
        u("expert_used_count", 2),
        u("expert_shared_count", 1),
        u("expert_feed_forward_length", 256),
        f("expert_weights_scale", 1.5),
        f("expert_swiglu_limit", 10.0),
        u("hyper_connection_mult", 4),
        u("hyper_connection_sinkhorn_iters", 20),
        arr("attention.compress_ratios", &[0, 2, 0]),
        arr("attention.kv_source_layers", &[1]),
        arr("attention.index_source_layers", &[1]),
        u("attention.index_head_count", 4),
        u("attention.index_key_length", 64),
        u("attention.index_topk", 16),
        f("rope.freq_base", 10000.0),
        f("attention.compress_rope_theta", 160000.0),
        f("rope.scaling.factor", 16.0),
        u("rope.scaling.original_context_length", 128),
    ];
    let h = Header::new(md, &real_entries_for(&[false, true, false]));
    let got = check(&h);
    assert_eq!(got.indexer.candidate_source_layer, -1);
    assert_eq!(got.mixture.swiglu_limits, vec![10.0; 3]);
    assert_eq!(got.streams.streams, 4);
}

/// A header with the n-gram hash keys and the per-layer expert prior.
#[test]
fn the_hash_and_the_expert_prior_are_read_alike() {
    let ratios = [0, 2, 0];
    let mut md = real_md_layout(&ratios, &[], &[]);
    let u64s = |k: &str, xs: &[u64]| {
        (
            format!("deepseek4.{k}"),
            Value::Array(xs.iter().map(|&x| Value::U64(x)).collect()),
        )
    };
    let n = |k: &str, v: u32| (format!("deepseek4.{k}"), Value::U32(v));
    md.extend([
        u64s("engram.layer_ids", &[1]),
        n("engram.head_count", 2),
        n("engram.key_length", 4),
        n("engram.max_ngram_size", 3),
        u64s("engram.multipliers", &[3, 5, 7]),
        u64s("engram.primes", &[11, 13, 17, 19]),
        u64s("engram.offsets", &[0, 11, 24, 41]),
        u64s("engram.token_map", &[0, 1, 0, 1, 2, 2, 1, 0]),
        n("engram.pad_id", 4),
        u64s("hot_experts.1", &[3, 1, 0]),
    ]);
    let h = Header::new(md, &real_entries_for(&bands(&ratios)));
    let got = check(&h);
    let hash = got.ngram.expect("the hash layout");
    assert_eq!(hash.layer_ids, vec![1]);
    assert_eq!(hash.pad_id, 2);
    assert_eq!(got.mixture.hot[1], vec![3, 1, 0]);
}

/// The reference arguments of a released checkpoint.
#[test]
fn the_reference_arguments_give_the_same_description() {
    let v = serde_json::json!({
        "n_layers": 4, "n_mtp_layers": 1, "dim": 256, "vocab_size": 48, "norm_eps": 1e-20,
        "n_heads": 4, "head_dim": 128, "rope_head_dim": 32, "q_lora_rank": 64,
        "o_lora_rank": 64, "o_groups": 4, "window_size": 32,
        "n_routed_experts": 8, "n_shared_experts": 1, "n_activated_experts": 2,
        "moe_inter_dim": 256, "score_func": "sqrtsoftplus", "route_scale": 1.5,
        "swiglu_limit": 10.0, "norm_topk_prob": true,
        "compress_ratios": [0, 2, 2, 0], "kv_source_layers": [1],
        "index_source_layers": [1, 3], "index_n_heads": 4, "index_head_dim": 64,
        "index_topk": 16, "candidate_source_layer": 1, "candidate_topk_blocks": 8,
        "candidate_block_size": 4, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-6,
        "rope_theta": 10000.0, "rope_factor": 16.0, "original_seq_len": 128,
        "compress_rope_theta": 160000.0
    });
    let want = DeepseekV41Config::from_reference_config(&v, None).unwrap();
    let got = WideSparseMeta::from_reference(&v, None).unwrap();
    assert_same(&got, &want);
    assert_eq!(got.mixture.swiglu_limits, vec![10.0; 4]);
}

/// A header without routed experts, or without a width, is refused as the dedicated module
/// refuses it; a low-rank gate is not this family.
#[test]
fn a_header_of_another_family_is_refused() {
    let ratios = [0, 2, 0];
    let entries = real_entries_for(&bands(&ratios));
    let md: Vec<_> = real_md_layout(&ratios, &[], &[])
        .into_iter()
        .filter(|(k, _)| !k.ends_with(".expert_count"))
        .collect();
    let h = Header::new(md, &entries);
    assert!(DeepseekV41Config::from_meta(&h).is_err());
    assert!(WideSparseMeta::from_meta(&h, ARCH, &release_values(), true).is_err());

    let md: Vec<_> = real_md_layout(&ratios, &[], &[])
        .into_iter()
        .filter(|(k, _)| !k.ends_with(".block_count"))
        .collect();
    let h = Header::new(md, &entries);
    assert!(WideSparseMeta::from_meta(&h, ARCH, &release_values(), true).is_err());

    let mut md = real_md_layout(&ratios, &[], &[]);
    md.push(("deepseek4.hyper_connection.low_rank".into(), Value::U32(8)));
    let plain: Vec<_> = real_entries_for(&bands(&ratios))
        .into_iter()
        .filter(|e| !e.name.contains(".hc_"))
        .collect();
    assert!(!WideSparseMeta::claims(&Header::new(md, &plain)));
}
