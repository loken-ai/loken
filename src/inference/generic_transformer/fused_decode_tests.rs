//! The fused decode against the unfused chain on one layer of synthetic weights: frequency
//! factors, attention sinks and the no-norm layer over an F-dtype cache.

use super::super::GenericHeteroTransformer;
use crate::inference::engine::llm_engine::KvQuant;
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
use crate::tensor::quantized::gguf_file::{self, Value};
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Device, Tensor};
use std::collections::HashMap;

const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const EMBD: usize = HEADS * HEAD_DIM;
const FFN: usize = 64;
const VOCAB: usize = 16;
const CONTEXT: usize = 64;
/// Decode steps compared, from an empty cache.
const STEPS: usize = 48;
/// Largest difference of the attention output allowed, relative to its largest magnitude.
const REL_TOL: f32 = 1e-3;

/// What the synthetic layer carries besides its fused projection.
#[derive(Clone, Copy)]
struct Shape {
    qk_norm: bool,
    freq_factors: bool,
    sinks: bool,
}

/// Values of a fixed generator in [-scale, scale).
fn pattern(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * scale
        })
        .collect()
}

fn entry(name: &str, dims: Vec<usize>, data: Vec<f32>) -> GgufEntry {
    GgufEntry {
        name: name.into(),
        dims,
        dtype: GgmlDType::F32,
        data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

/// A one-layer model of `shape` written under a directory of its own; the file's path.
fn write(tag: &str, shape: Shape) -> std::path::PathBuf {
    let qkv_rows = (HEADS + 2 * KV_HEADS) * HEAD_DIM;
    let mut entries = vec![
        entry(
            "token_embd.weight",
            vec![VOCAB, EMBD],
            pattern(VOCAB * EMBD, 1, 0.5),
        ),
        entry(
            "output.weight",
            vec![VOCAB, EMBD],
            pattern(VOCAB * EMBD, 2, 0.5),
        ),
        entry("output_norm.weight", vec![EMBD], vec![1.0; EMBD]),
        entry("blk.0.attn_norm.weight", vec![EMBD], vec![1.0; EMBD]),
        entry("blk.0.ffn_norm.weight", vec![EMBD], vec![1.0; EMBD]),
        entry(
            "blk.0.attn_qkv.weight",
            vec![qkv_rows, EMBD],
            pattern(qkv_rows * EMBD, 3, 0.25),
        ),
        entry(
            "blk.0.attn_output.weight",
            vec![EMBD, EMBD],
            pattern(EMBD * EMBD, 4, 0.1),
        ),
        entry(
            "blk.0.ffn_gate.weight",
            vec![FFN, EMBD],
            pattern(FFN * EMBD, 5, 0.1),
        ),
        entry(
            "blk.0.ffn_up.weight",
            vec![FFN, EMBD],
            pattern(FFN * EMBD, 6, 0.1),
        ),
        entry(
            "blk.0.ffn_down.weight",
            vec![EMBD, FFN],
            pattern(EMBD * FFN, 7, 0.1),
        ),
    ];
    if shape.qk_norm {
        let norm = |seed| {
            pattern(HEAD_DIM, seed, 0.5)
                .iter()
                .map(|v| 1.0 + v)
                .collect()
        };
        entries.push(entry("blk.0.attn_q_norm.weight", vec![HEAD_DIM], norm(8)));
        entries.push(entry("blk.0.attn_k_norm.weight", vec![HEAD_DIM], norm(9)));
    }
    if shape.freq_factors {
        // Factors from one to a few dozen across the pairs, as a long-context scaling sets.
        let factors = (0..HEAD_DIM / 2)
            .map(|i| 1.0 + (i * i) as f32 / 16.0)
            .collect();
        entries.push(entry("rope_freqs.weight", vec![HEAD_DIM / 2], factors));
    }
    if shape.sinks {
        entries.push(entry(
            "blk.0.attn_sinks",
            vec![HEADS],
            pattern(HEADS, 10, 2.0),
        ));
    }
    let u = |k: &str, v: u32| (format!("llama.{k}"), Value::U32(v));
    let metadata = vec![
        ("general.architecture".into(), Value::String("llama".into())),
        u("block_count", 1),
        u("embedding_length", EMBD as u32),
        u("feed_forward_length", FFN as u32),
        u("attention.head_count", HEADS as u32),
        u("attention.head_count_kv", KV_HEADS as u32),
        u("context_length", CONTEXT as u32),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-5),
        ),
        ("llama.rope.freq_base".into(), Value::F32(10000.0)),
    ];
    let dir = std::env::temp_dir().join(format!("loken-fused-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("m.gguf");
    write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
    path
}

/// The model at `path` on `card` with its cache in `kv`.
fn load(path: &std::path::Path, card: &Device, kv: KvQuant) -> GenericHeteroTransformer {
    let g = gguf_file::open_mapped(path).unwrap();
    let bytes = g.mmap.clone();
    let cards: HashMap<usize, Device> = [(0usize, card.clone())].into_iter().collect();
    GenericHeteroTransformer::from_gguf_with_kv_quant(
        g.content,
        &bytes[..],
        &cards,
        &HeteroPlan::forced_gpu(1, 1, 0),
        kv,
        None,
    )
    .unwrap()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(crate::tensor::DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

/// Decodes `STEPS` rows through the fused path of one copy of the layer and the unfused chain
/// of another, and returns the worst difference of their outputs relative to the output's
/// magnitude. `f16_cache`, when given, sets whether the F-dtype cache stores F16.
fn worst_difference(tag: &str, shape: Shape, kv: KvQuant, f16_cache: Option<bool>) -> f32 {
    let Ok(card) = Device::new_cuda(0) else {
        eprintln!("no CUDA device");
        return 0.0;
    };
    let path = write(tag, shape);
    let mut fused_model = load(&path, &card, kv);
    let mut chain_model = load(&path, &card, kv);
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
    let fused = &mut fused_model.layers[0];
    let chain = &mut chain_model.layers[0];
    if let Some(f16) = f16_cache {
        fused.kv_f16 = f16;
        chain.kv_f16 = f16;
    }
    assert_eq!(
        fused.attn_sinks.is_some(),
        shape.sinks,
        "{tag}: sinks not loaded"
    );
    assert_eq!(
        fused.rope_freq_factors.is_some(),
        shape.freq_factors,
        "{tag}: frequency factors not loaded"
    );
    let rows = pattern(STEPS * EMBD, 11, 1.0);
    let mut worst = 0f32;
    for pos in 0..STEPS {
        let x = Tensor::from_vec(
            rows[pos * EMBD..(pos + 1) * EMBD].to_vec(),
            (1, 1, EMBD),
            &card,
        )
        .unwrap();
        assert_eq!(
            fused.fused_decode_route(&x, true).unwrap(),
            Some(shape.qk_norm),
            "{tag}: the fused decode does not serve the layer"
        );
        let (want, _) = {
            // The rotation rows the chain computes itself, given ahead: the fused path stands
            // aside and nothing else changes.
            chain.rope_rows = Some(chain.rope_table().rows(pos, 1, &card).unwrap());
            assert_eq!(chain.fused_decode_route(&x, true).unwrap(), None);
            chain.attend(&x, None, pos, None, None, None).unwrap()
        };
        let (got, _) = fused.attend(&x, None, pos, None, None, None).unwrap();
        let (want, got) = (host(&want), host(&got));
        assert_eq!(want.len(), got.len());
        let scale = want
            .iter()
            .fold(0f32, |m, v| m.max(v.abs()))
            .max(f32::MIN_POSITIVE);
        let diff = want
            .iter()
            .zip(&got)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff.is_finite(), "{tag}: step {pos} not finite");
        worst = worst.max(diff / scale);
    }
    eprintln!("{tag}: worst relative difference {worst:e} over {STEPS} steps");
    worst
}

#[test]
fn frequency_factors_take_the_fused_decode() {
    for (tag, qk_norm, kv, f16) in [
        ("ff-norm-q8", true, KvQuant::Q8, None),
        ("ff-norm-f16", true, KvQuant::Off, Some(true)),
        ("ff-plain-q8", false, KvQuant::Q8, None),
        ("ff-plain-f16", false, KvQuant::Off, Some(true)),
    ] {
        let shape = Shape {
            qk_norm,
            freq_factors: true,
            sinks: false,
        };
        let worst = worst_difference(tag, shape, kv, f16);
        assert!(worst < REL_TOL, "{tag}: {worst:e}");
    }
}

#[test]
fn sink_layers_take_the_fused_decode() {
    for (tag, f16) in [("sinks-f16", true), ("sinks-f32", false)] {
        let shape = Shape {
            qk_norm: false,
            freq_factors: false,
            sinks: true,
        };
        let worst = worst_difference(tag, shape, KvQuant::Off, Some(f16));
        assert!(worst < REL_TOL, "{tag}: {worst:e}");
    }
}

#[test]
fn no_norm_layers_take_the_fused_decode_over_an_f_dtype_cache() {
    for (tag, f16) in [("plain-f16", true), ("plain-f32", false)] {
        let shape = Shape {
            qk_norm: false,
            freq_factors: false,
            sinks: false,
        };
        let worst = worst_difference(tag, shape, KvQuant::Off, Some(f16));
        assert!(worst < REL_TOL, "{tag}: {worst:e}");
    }
}
