//! A tiny dense model written as a GGUF file, for tests that need a whole model on the host.

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
