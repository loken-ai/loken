//! Toy files of the DeepSeek V4.1 family for the tests: llama.cpp's naming and keys over tiny
//! shapes.
//!
//! A file carries the keys llama.cpp's converter writes and only those.
//! The low-rank widths, the band roles and the release values are left to the reader.
//! Every weight is an F32 ramp, so a model built from a file is deterministic.

use crate::tensor::gguf_write::GgufEntry;
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;

pub(crate) const DIM: usize = 256;
pub(crate) const NH: usize = 4;
pub(crate) const HD: usize = 128;
pub(crate) const QLORA: usize = 64;
pub(crate) const OLORA: usize = 64;
pub(crate) const OG: usize = 4;
pub(crate) const ER: usize = 8;
pub(crate) const INTER: usize = 256;
pub(crate) const HC: usize = 4;
pub(crate) const MIX: usize = 24;
pub(crate) const INH: usize = 4;
pub(crate) const IHD: usize = 64;
pub(crate) const VOCAB: usize = 48;

/// An F32 tensor of a fixed ramp of values.
pub(crate) fn f32e(name: String, dims: Vec<usize>) -> GgufEntry {
    let n: usize = dims.iter().product();
    let data: Vec<u8> = (0..n)
        .flat_map(|i| (i as f32 * 0.0007 - 0.05).to_le_bytes())
        .collect();
    GgufEntry {
        name,
        dims,
        dtype: GgmlDType::F32,
        data,
    }
}

/// One block's tensors under llama.cpp's names: bare (no `.weight`), `attn_kv_a_norm`,
/// `attn_sinks`, `attn_output_a/b`, the gate bias as `exp_probs_b`, and the CSA2 extras as
/// `attn_compressor_*` / `indexer.*`. A band block carries the extras; a reader would not.
pub(crate) fn real_layer_tensors(l: usize, band: bool) -> Vec<GgufEntry> {
    let p = format!("blk.{l}");
    let mut e = vec![
        f32e(format!("{p}.attn_q_a"), vec![QLORA, DIM]),
        f32e(format!("{p}.attn_q_a_norm"), vec![QLORA]),
        f32e(format!("{p}.attn_q_b"), vec![NH * HD, QLORA]),
        f32e(format!("{p}.attn_kv"), vec![HD, DIM]),
        f32e(format!("{p}.attn_kv_a_norm"), vec![HD]),
        f32e(format!("{p}.attn_sinks"), vec![NH]),
        f32e(format!("{p}.attn_output_a"), vec![OG * OLORA, NH * HD / OG]),
        f32e(format!("{p}.attn_output_b"), vec![DIM, OG * OLORA]),
        f32e(format!("{p}.attn_norm"), vec![DIM]),
        f32e(format!("{p}.ffn_norm"), vec![DIM]),
        f32e(format!("{p}.ffn_gate_inp"), vec![ER, DIM]),
        f32e(format!("{p}.exp_probs_b"), vec![ER]),
        f32e(format!("{p}.ffn_gate_shexp"), vec![INTER, DIM]),
        f32e(format!("{p}.ffn_up_shexp"), vec![INTER, DIM]),
        f32e(format!("{p}.ffn_down_shexp"), vec![DIM, INTER]),
        f32e(format!("{p}.ffn_gate_exps"), vec![ER, INTER, DIM]),
        f32e(format!("{p}.ffn_up_exps"), vec![ER, INTER, DIM]),
        f32e(format!("{p}.ffn_down_exps"), vec![ER, DIM, INTER]),
        f32e(format!("{p}.hc_attn_fn"), vec![MIX, HC * DIM]),
        f32e(format!("{p}.hc_attn_scale"), vec![3]),
        f32e(format!("{p}.hc_attn_base"), vec![MIX]),
        f32e(format!("{p}.hc_ffn_fn"), vec![MIX, HC * DIM]),
        f32e(format!("{p}.hc_ffn_scale"), vec![3]),
        f32e(format!("{p}.hc_ffn_base"), vec![MIX]),
    ];
    if band {
        e.extend([
            f32e(format!("{p}.attn_compressor_norm"), vec![HD]),
            f32e(format!("{p}.attn_compressor_kv"), vec![HD, DIM]),
            f32e(format!("{p}.attn_compressor_gate"), vec![HD, DIM]),
            f32e(format!("{p}.indexer.attn_q_b"), vec![INH * IHD, QLORA]),
            f32e(format!("{p}.indexer.proj"), vec![INH, DIM]),
            f32e(format!("{p}.indexer.attn_k"), vec![IHD, HD]),
            f32e(format!("{p}.indexer.k_norm"), vec![IHD]),
        ]);
    }
    e
}

/// The same metadata for a chosen band layout: the compress ratios, and the kv and index
/// source layers when the layout names them (an empty list leaves them to be derived).
pub(crate) fn real_md_layout(
    ratios: &[u32],
    kv_src: &[u32],
    index_src: &[u32],
) -> Vec<(String, Value)> {
    let n_layers = ratios.len() as u32;
    let arr = |k: &str, xs: &[u32]| {
        (
            format!("deepseek4.{k}"),
            Value::Array(xs.iter().map(|&x| Value::U32(x)).collect()),
        )
    };
    let mut md = real_md_base(n_layers, ratios);
    if !kv_src.is_empty() {
        md.push(arr("attention.kv_source_layers", kv_src));
    }
    if !index_src.is_empty() {
        md.push(arr("attention.index_source_layers", index_src));
    }
    md
}

fn real_md_base(n_layers: u32, ratios: &[u32]) -> Vec<(String, Value)> {
    let u = |k: &str, v: u32| (format!("deepseek4.{k}"), Value::U32(v));
    let f = |k: &str, v: f32| (format!("deepseek4.{k}"), Value::F32(v));
    vec![
        (
            "general.architecture".to_string(),
            Value::String("deepseek4".into()),
        ),
        u("block_count", n_layers),
        u("embedding_length", DIM as u32),
        u("attention.head_count", NH as u32),
        u("attention.key_length", HD as u32),
        u("attention.sliding_window", 32),
        f("attention.layer_norm_rms_epsilon", 1e-20),
        u("expert_count", ER as u32),
        u("expert_used_count", 2),
        u("expert_shared_count", 1),
        u("feed_forward_length", INTER as u32),
        f("expert_weights_scale", 1.5),
        (
            "deepseek4.expert_weights_norm".to_string(),
            Value::Bool(true),
        ),
        (
            "deepseek4.swiglu_clamp_exp".to_string(),
            Value::Array((0..n_layers).map(|_| Value::F32(10.0)).collect()),
        ),
        u("hyper_connection.count", HC as u32),
        u("hyper_connection.sinkhorn_iterations", 20),
        f("hyper_connection.epsilon", 1e-6),
        (
            "deepseek4.attention.compress_ratios".to_string(),
            Value::Array(ratios.iter().map(|&r| Value::U32(r)).collect()),
        ),
        f("attention.compress_rope_freq_base", 160000.0),
        u("indexer.head_count", INH as u32),
        u("indexer.key_length", IHD as u32),
        u("indexer.top_k", 16),
        f("rope.freq_base", 10000.0),
        u("rope.dimension_count", 32),
    ]
}

/// The tensors of a model whose layers are band layers where `bands` says.
pub(crate) fn real_entries_for(bands: &[bool]) -> Vec<GgufEntry> {
    let mut entries = vec![
        f32e("token_embd".into(), vec![VOCAB, DIM]),
        f32e("output_norm".into(), vec![DIM]),
        f32e("output".into(), vec![VOCAB, DIM]),
    ];
    for (l, &band) in bands.iter().enumerate() {
        entries.extend(real_layer_tensors(l, band));
    }
    entries
}

/// A prompt of 24 tokens over the toy vocabulary.
pub(crate) fn prompt() -> Vec<u32> {
    (0..24).map(|i| (i * 7 + 3) % VOCAB as u32).collect()
}
