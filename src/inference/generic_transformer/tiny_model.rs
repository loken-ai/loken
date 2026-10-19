//! A tiny dense model, and a tiny mixture, written as GGUF files, for tests that need a whole
//! model.

use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;

const EMBD: usize = 8;
const FFN: usize = 16;
pub(crate) const VOCAB: usize = 11;

/// The tensors of a model of `layers` blocks: `[out, in]` weights of a deterministic pattern.
pub(crate) fn entries(layers: usize) -> Vec<GgufEntry> {
    let mut seed = 7u32;
    let mut entry = |name: String, dims: Vec<usize>| {
        let n: usize = dims.iter().product();
        let data = (0..n)
            .flat_map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 16) as f32 / 65536.0 - 0.5) * 0.4).to_le_bytes()
            })
            .collect();
        GgufEntry {
            name,
            dims,
            dtype: GgmlDType::F32,
            data,
        }
    };
    let mut out = vec![
        entry("token_embd.weight".into(), vec![VOCAB, EMBD]),
        entry("output.weight".into(), vec![VOCAB, EMBD]),
    ];
    let norm = |name: String| GgufEntry {
        name,
        dims: vec![EMBD],
        dtype: GgmlDType::F32,
        data: [1.0f32.to_le_bytes(); EMBD].concat(),
    };
    out.push(norm("output_norm.weight".into()));
    for il in 0..layers {
        out.push(norm(format!("blk.{il}.attn_norm.weight")));
        out.push(norm(format!("blk.{il}.ffn_norm.weight")));
        for (n, d) in [
            ("attn_q", [EMBD, EMBD]),
            ("attn_k", [EMBD, EMBD]),
            ("attn_v", [EMBD, EMBD]),
            ("attn_output", [EMBD, EMBD]),
            ("ffn_gate", [FFN, EMBD]),
            ("ffn_up", [FFN, EMBD]),
            ("ffn_down", [EMBD, FFN]),
        ] {
            out.push(entry(format!("blk.{il}.{n}.weight"), d.to_vec()));
        }
    }
    out
}

/// The header of a model of `layers` blocks.
pub(crate) fn metadata(layers: usize) -> Vec<(String, Value)> {
    let u = |k: &str, v: u32| (format!("llama.{k}"), Value::U32(v));
    vec![
        ("general.architecture".into(), Value::String("llama".into())),
        u("block_count", layers as u32),
        u("embedding_length", EMBD as u32),
        u("feed_forward_length", FFN as u32),
        u("attention.head_count", 2),
        u("attention.head_count_kv", 2),
        u("context_length", 64),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-5),
        ),
        ("llama.rope.freq_base".into(), Value::F32(10000.0)),
    ]
}

/// The model written under a directory of its own in the temp dir, and the path of the file.
/// Removed by `remove` when the test is done.
pub(crate) fn write(tag: &str, layers: usize) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("loken-tiny-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.gguf");
    write_gguf_with_metadata(&path, &metadata(layers), &entries(layers)).unwrap();
    path
}

/// Width, expert width, experts and experts a row routes to of the tiny mixture: widths the
/// expert kernels take in whole blocks.
const MOE_EMBD: usize = 64;
const MOE_FFN: usize = 64;
const MOE_EXPERTS: usize = 8;
const MOE_USED: usize = 2;

/// A mixture of `layers` blocks, each routing every row to two of its eight experts, written as
/// [`write`] writes the dense model.
pub(crate) fn write_moe(tag: &str, layers: usize) -> std::path::PathBuf {
    write_wide(tag, layers, true)
}

/// The tiny mixture's shape with its experts, or with a dense feed-forward of their width.
pub(crate) fn write_wide(tag: &str, layers: usize, routed: bool) -> std::path::PathBuf {
    let mut seed = 11u32;
    let mut entry = |name: String, dims: Vec<usize>| {
        let n: usize = dims.iter().product();
        let data = (0..n)
            .flat_map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                (((seed >> 16) as f32 / 65536.0 - 0.5) * 0.4).to_le_bytes()
            })
            .collect();
        GgufEntry {
            name,
            dims,
            dtype: GgmlDType::F32,
            data,
        }
    };
    let norm = |name: String| GgufEntry {
        name,
        dims: vec![MOE_EMBD],
        dtype: GgmlDType::F32,
        data: [1.0f32.to_le_bytes(); MOE_EMBD].concat(),
    };
    let mut entries = vec![
        entry("token_embd.weight".into(), vec![VOCAB, MOE_EMBD]),
        entry("output.weight".into(), vec![VOCAB, MOE_EMBD]),
        norm("output_norm.weight".into()),
    ];
    for il in 0..layers {
        entries.push(norm(format!("blk.{il}.attn_norm.weight")));
        entries.push(norm(format!("blk.{il}.ffn_norm.weight")));
        let ffn = if routed {
            vec![
                ("ffn_gate_inp", vec![MOE_EXPERTS, MOE_EMBD]),
                ("ffn_gate_exps", vec![MOE_EXPERTS, MOE_FFN, MOE_EMBD]),
                ("ffn_up_exps", vec![MOE_EXPERTS, MOE_FFN, MOE_EMBD]),
                ("ffn_down_exps", vec![MOE_EXPERTS, MOE_EMBD, MOE_FFN]),
            ]
        } else {
            vec![
                ("ffn_gate", vec![MOE_FFN, MOE_EMBD]),
                ("ffn_up", vec![MOE_FFN, MOE_EMBD]),
                ("ffn_down", vec![MOE_EMBD, MOE_FFN]),
            ]
        };
        for (n, d) in [
            ("attn_q", vec![MOE_EMBD, MOE_EMBD]),
            ("attn_k", vec![MOE_EMBD, MOE_EMBD]),
            ("attn_v", vec![MOE_EMBD, MOE_EMBD]),
            ("attn_output", vec![MOE_EMBD, MOE_EMBD]),
        ]
        .into_iter()
        .chain(ffn)
        {
            entries.push(entry(format!("blk.{il}.{n}.weight"), d));
        }
    }
    let u = |k: &str, v: u32| (format!("llama.{k}"), Value::U32(v));
    let metadata = vec![
        ("general.architecture".into(), Value::String("llama".into())),
        u("block_count", layers as u32),
        u("embedding_length", MOE_EMBD as u32),
        u("feed_forward_length", MOE_FFN as u32),
        u("attention.head_count", 2),
        u("attention.head_count_kv", 2),
        u("context_length", 64),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-5),
        ),
        ("llama.rope.freq_base".into(), Value::F32(10000.0)),
    ];
    let metadata: Vec<_> = metadata
        .into_iter()
        .chain(
            routed
                .then(|| {
                    [
                        u("expert_feed_forward_length", MOE_FFN as u32),
                        u("expert_count", MOE_EXPERTS as u32),
                        u("expert_used_count", MOE_USED as u32),
                    ]
                })
                .into_iter()
                .flatten(),
        )
        .collect();
    let dir = std::env::temp_dir().join(format!("loken-tiny-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.gguf");
    write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
    path
}

pub(crate) fn remove(path: &std::path::Path) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// The model at `path` loaded whole on the host.
pub(crate) fn load(path: &std::path::Path, layers: usize) -> super::GenericHeteroTransformer {
    let plan = crate::inference::place::layer_executor::HeteroPlan::calculate_with_kv(
        layers,
        1 << 20,
        &[],
        &[],
        1.0,
        0,
    );
    let g = crate::tensor::quantized::gguf_file::open_mapped(path).unwrap();
    let bytes = g.mmap.clone();
    super::GenericHeteroTransformer::from_gguf(
        g.content,
        &bytes[..],
        &std::collections::HashMap::new(),
        &plan,
    )
    .unwrap()
}

/// Width, feed-forward width, heads, key-value heads and vocabulary of the quantised model:
/// whole K-quant blocks along every reduction, grouped-query attention.
pub(crate) const QUANT_EMBD: usize = 256;
const QUANT_FFN: usize = 512;
const QUANT_HEADS: usize = 4;
const QUANT_KV_HEADS: usize = 2;
pub(crate) const QUANT_VOCAB: usize = 64;

/// A qwen3 model of `layers` blocks with the weights of a K-quant file: Q4_K projections,
/// Q6_K value, down and head, f32 norms and embedding. Written as [`write`] writes the dense
/// model.
pub(crate) fn write_quant(tag: &str, layers: usize) -> std::path::PathBuf {
    let mut seed = 23u32;
    let mut values = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 16) as f32 / 65536.0 - 0.5) * 0.4
            })
            .collect()
    };
    let mut entry = |name: String, dims: Vec<usize>, dtype: GgmlDType| {
        let xs = values(dims.iter().product());
        let data = if dtype == GgmlDType::F32 {
            xs.iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            crate::tensor::quant_cpu::from_float_bytes(dtype, &xs).unwrap()
        };
        GgufEntry {
            name,
            dims,
            dtype,
            data,
        }
    };
    let ones = |name: String, n: usize| GgufEntry {
        name,
        dims: vec![n],
        dtype: GgmlDType::F32,
        data: [1.0f32.to_le_bytes(); 1].repeat(n).concat(),
    };
    let head_dim = QUANT_EMBD / QUANT_HEADS;
    let kv = QUANT_KV_HEADS * head_dim;
    let mut entries = vec![
        entry(
            "token_embd.weight".into(),
            vec![QUANT_VOCAB, QUANT_EMBD],
            GgmlDType::F32,
        ),
        entry(
            "output.weight".into(),
            vec![QUANT_VOCAB, QUANT_EMBD],
            GgmlDType::Q6K,
        ),
        ones("output_norm.weight".into(), QUANT_EMBD),
    ];
    for il in 0..layers {
        entries.push(ones(format!("blk.{il}.attn_norm.weight"), QUANT_EMBD));
        entries.push(ones(format!("blk.{il}.ffn_norm.weight"), QUANT_EMBD));
        entries.push(ones(format!("blk.{il}.attn_q_norm.weight"), head_dim));
        entries.push(ones(format!("blk.{il}.attn_k_norm.weight"), head_dim));
        for (n, d, t) in [
            ("attn_q", [QUANT_EMBD, QUANT_EMBD], GgmlDType::Q4K),
            ("attn_k", [kv, QUANT_EMBD], GgmlDType::Q4K),
            ("attn_v", [kv, QUANT_EMBD], GgmlDType::Q6K),
            ("attn_output", [QUANT_EMBD, QUANT_EMBD], GgmlDType::Q4K),
            ("ffn_gate", [QUANT_FFN, QUANT_EMBD], GgmlDType::Q4K),
            ("ffn_up", [QUANT_FFN, QUANT_EMBD], GgmlDType::Q4K),
            ("ffn_down", [QUANT_EMBD, QUANT_FFN], GgmlDType::Q6K),
        ] {
            entries.push(entry(format!("blk.{il}.{n}.weight"), d.to_vec(), t));
        }
    }
    let u = |k: &str, v: u32| (format!("qwen3.{k}"), Value::U32(v));
    let metadata = vec![
        ("general.architecture".into(), Value::String("qwen3".into())),
        u("block_count", layers as u32),
        u("embedding_length", QUANT_EMBD as u32),
        u("feed_forward_length", QUANT_FFN as u32),
        u("attention.head_count", QUANT_HEADS as u32),
        u("attention.head_count_kv", QUANT_KV_HEADS as u32),
        u("attention.key_length", head_dim as u32),
        u("attention.value_length", head_dim as u32),
        u("context_length", 64),
        (
            "qwen3.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-6),
        ),
        ("qwen3.rope.freq_base".into(), Value::F32(1_000_000.0)),
    ];
    let dir = std::env::temp_dir().join(format!("loken-tiny-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.gguf");
    write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
    path
}
