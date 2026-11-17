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
    write_wide_with(tag, layers, routed, Vec::new())
}

/// [`write_wide_with`] with a sink logit for each head of every layer.
pub(crate) fn write_wide_sinks(
    tag: &str,
    layers: usize,
    routed: bool,
    extra: Vec<(String, Value)>,
) -> std::path::PathBuf {
    write_wide_full(tag, layers, routed, true, extra)
}

/// Heads of the tiny mixture.
const MOE_HEADS: usize = 2;

/// Width of a head of the tiny mixture: its width over its heads.
pub(crate) const MOE_HEAD_DIM: usize = MOE_EMBD / MOE_HEADS;

/// [`write_wide`] with `extra` keys added to the header.
pub(crate) fn write_wide_with(
    tag: &str,
    layers: usize,
    routed: bool,
    extra: Vec<(String, Value)>,
) -> std::path::PathBuf {
    write_wide_full(tag, layers, routed, false, extra)
}

/// [`write_wide_with`], each layer holding a sink logit per head when `sinks` is set.
fn write_wide_full(
    tag: &str,
    layers: usize,
    routed: bool,
    sinks: bool,
    extra: Vec<(String, Value)>,
) -> std::path::PathBuf {
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
        if sinks {
            entries.push(entry(format!("blk.{il}.attn_sinks"), vec![MOE_HEADS]));
        }
    }
    let u = |k: &str, v: u32| (format!("llama.{k}"), Value::U32(v));
    let metadata = vec![
        ("general.architecture".into(), Value::String("llama".into())),
        u("block_count", layers as u32),
        u("embedding_length", MOE_EMBD as u32),
        u("feed_forward_length", MOE_FFN as u32),
        u("attention.head_count", MOE_HEADS as u32),
        u("attention.head_count_kv", MOE_HEADS as u32),
        u("context_length", 64),
        (
            "llama.attention.layer_norm_rms_epsilon".into(),
            Value::F32(1e-5),
        ),
        ("llama.rope.freq_base".into(), Value::F32(10000.0)),
    ];
    // A key of `extra` replaces the base key of the same name.
    let replaced: Vec<String> = extra.iter().map(|(k, _)| k.clone()).collect();
    let metadata: Vec<_> = metadata
        .into_iter()
        .filter(|(k, _)| !replaced.contains(k))
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
        .chain(extra)
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

/// The model at `path` loaded whole on the host, its cache unquantised.
pub(crate) fn load(path: &std::path::Path, layers: usize) -> super::GenericHeteroTransformer {
    load_with_kv(
        path,
        layers,
        crate::inference::engine::llm_engine::KvQuant::Off,
    )
}

/// The model at `path` loaded whole on card 0, its cache unquantised.
#[cfg(feature = "gpu")]
pub(crate) fn load_on_card(
    path: &std::path::Path,
    layers: usize,
) -> super::GenericHeteroTransformer {
    load_on_cards(path, layers, &[(0, 0)])
}

/// The model at `path` loaded on the cards of `runs`, its cache unquantised.
///
/// Each run is a card ordinal and the first layer it holds.
/// A run holds the layers up to the first layer of the next run.
#[cfg(feature = "gpu")]
pub(crate) fn load_on_cards(
    path: &std::path::Path,
    layers: usize,
    runs: &[(usize, usize)],
) -> super::GenericHeteroTransformer {
    use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};
    let ends = runs.iter().skip(1).map(|r| r.1).chain([layers]);
    let segments = runs
        .iter()
        .zip(ends)
        .map(|(&(card, start), end)| HeteroSegment {
            kind: DeviceKind::Cuda(card),
            layer_start: start,
            layer_end: end,
            free_memory_bytes: 1 << 30,
        })
        .collect();
    let plan = HeteroPlan {
        segments,
        tp_cards: None,
        total_layers: layers,
        host_experts_from: None,
    };
    let g = crate::tensor::quantized::gguf_file::open_mapped(path).unwrap();
    let bytes = g.mmap.clone();
    let cards = runs
        .iter()
        .map(|&(card, _)| (card, crate::tensor::Device::new_cuda(card).unwrap()))
        .collect::<std::collections::HashMap<_, _>>();
    super::GenericHeteroTransformer::from_gguf_with_kv_quant(
        g.content,
        &bytes[..],
        &cards,
        &plan,
        crate::inference::engine::llm_engine::KvQuant::Off,
        None,
    )
    .unwrap()
}

/// The model at `path` loaded whole on the host with the Q8 cache the server gives a model whose
/// head shape the quantised kernels serve: the host decode attends over its Q8 store.
pub(crate) fn load_q8_kv(path: &std::path::Path, layers: usize) -> super::GenericHeteroTransformer {
    load_with_kv(
        path,
        layers,
        crate::inference::engine::llm_engine::KvQuant::Q8,
    )
}

fn load_with_kv(
    path: &std::path::Path,
    layers: usize,
    kv_quant: crate::inference::engine::llm_engine::KvQuant,
) -> super::GenericHeteroTransformer {
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
    super::GenericHeteroTransformer::from_gguf_with_kv_quant(
        g.content,
        &bytes[..],
        &std::collections::HashMap::new(),
        &plan,
        kv_quant,
        None,
    )
    .unwrap()
}

/// Width, feed-forward width, heads, key-value heads and vocabulary of the quantised model:
/// whole K-quant blocks along every reduction, grouped-query attention.
pub(crate) const QUANT_EMBD: usize = 256;
const QUANT_FFN: usize = 512;
pub(crate) const QUANT_HEADS: usize = 4;
const QUANT_KV_HEADS: usize = 2;
pub(crate) const QUANT_VOCAB: usize = 64;

/// A qwen3 model of `layers` blocks with the weights of a K-quant file: Q4_K projections,
/// Q6_K value, down and head, f32 norms and embedding. Written as [`write`] writes the dense
/// model.
pub(crate) fn write_quant(tag: &str, layers: usize) -> std::path::PathBuf {
    let (metadata, entries) = quant_model(layers);
    let path = scratch_dir(tag).join("m.gguf");
    write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
    path
}

/// What [`write_quant_with`] adds to the quantised model.
#[derive(Default)]
pub(crate) struct QuantExtras {
    /// Header keys, after the model's own.
    pub keys: Vec<(String, Value)>,
    /// A sink logit for each head of every layer.
    pub sinks: bool,
    /// Quantised experts, routed 2 of 8, in place of each feed-forward.
    pub routed: bool,
}

/// [`write_quant`] with `extras`.
pub(crate) fn write_quant_with(
    tag: &str,
    layers: usize,
    extras: QuantExtras,
) -> std::path::PathBuf {
    let (mut metadata, entries) = quant_model_with(layers, extras.sinks, extras.routed);
    metadata.extend(extras.keys);
    let path = scratch_dir(tag).join("m.gguf");
    write_gguf_with_metadata(&path, &metadata, &entries).unwrap();
    path
}

/// The model of [`write_quant`] stored as a split set of `parts` files, its tensors dealt to the
/// parts in order. Returns the path of the first part.
pub(crate) fn write_quant_split(tag: &str, layers: usize, parts: usize) -> std::path::PathBuf {
    let (mut metadata, entries) = quant_model(layers);
    let dir = scratch_dir(tag);
    let name = |i: usize| dir.join(format!("m-{:05}-of-{parts:05}.gguf", i + 1));
    let split = |i: usize| {
        vec![
            ("split.count".to_string(), Value::U32(parts as u32)),
            ("split.no".to_string(), Value::U32(i as u32)),
        ]
    };
    metadata.extend(split(0));
    let per = entries.len().div_ceil(parts);
    for (i, chunk) in entries.chunks(per).enumerate() {
        let md = if i == 0 { metadata.clone() } else { split(i) };
        write_gguf_with_metadata(&name(i), &md, chunk).unwrap();
    }
    name(0)
}

/// The model of [`write_quant`] with float weights, stored as a qwen3 checkpoint directory of
/// `shards` safetensors files, an index and its `config.json`. Returns the directory.
pub(crate) fn write_quant_checkpoint(
    tag: &str,
    layers: usize,
    shards: usize,
) -> std::path::PathBuf {
    let (_, entries) = quant_model(layers);
    let dir = scratch_dir(tag);
    let named = |n: &str| -> String {
        match n {
            "token_embd.weight" => "model.embed_tokens.weight".into(),
            "output.weight" => "lm_head.weight".into(),
            "output_norm.weight" => "model.norm.weight".into(),
            _ => {
                let (il, rest) = n
                    .strip_prefix("blk.")
                    .and_then(|r| r.split_once('.'))
                    .unwrap();
                let hf = match rest {
                    "attn_norm.weight" => "input_layernorm.weight",
                    "ffn_norm.weight" => "post_attention_layernorm.weight",
                    "attn_q_norm.weight" => "self_attn.q_norm.weight",
                    "attn_k_norm.weight" => "self_attn.k_norm.weight",
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    other => panic!("no checkpoint name for {other}"),
                };
                format!("model.layers.{il}.{hf}")
            }
        }
    };
    // Each tensor as f32 bytes, its dimensions outer first.
    let floats: Vec<(String, Vec<usize>, Vec<u8>)> = entries
        .iter()
        .map(|e| {
            let n: usize = e.dims.iter().product();
            let xs = if e.dtype == GgmlDType::F32 {
                e.data.clone()
            } else {
                let mut out = vec![0f32; n];
                crate::tensor::quant_cpu::to_float_bytes(e.dtype, &e.data, &mut out).unwrap();
                out.iter().flat_map(|v| v.to_le_bytes()).collect()
            };
            (named(&e.name), e.dims.clone(), xs)
        })
        .collect();
    let file = |i: usize| format!("model-{:05}-of-{shards:05}.safetensors", i + 1);
    let mut weight_map = serde_json::Map::new();
    let per = floats.len().div_ceil(shards);
    for (i, chunk) in floats.chunks(per).enumerate() {
        let views: Vec<_> = chunk
            .iter()
            .map(|(name, dims, bytes)| {
                weight_map.insert(name.clone(), serde_json::Value::String(file(i)));
                let view = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    dims.clone(),
                    bytes,
                )
                .unwrap();
                (name.clone(), view)
            })
            .collect();
        safetensors::serialize_to_file(views, None, &dir.join(file(i))).unwrap();
    }
    let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    let config = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": QUANT_EMBD,
        "intermediate_size": QUANT_FFN,
        "num_hidden_layers": layers,
        "num_attention_heads": QUANT_HEADS,
        "num_key_value_heads": QUANT_KV_HEADS,
        "head_dim": QUANT_EMBD / QUANT_HEADS,
        "vocab_size": QUANT_VOCAB,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1_000_000.0,
        "tie_word_embeddings": false,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "torch_dtype": "float32"
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    dir
}

/// An empty directory of its own for the model `tag` in the temp dir.
fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("loken-tiny-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The metadata and tensors of the model [`write_quant`] writes.
fn quant_model(layers: usize) -> (Vec<(String, Value)>, Vec<GgufEntry>) {
    quant_model_with(layers, false, false)
}

/// [`quant_model`], each layer with a sink logit per head when `sinks` is set, and with
/// quantised experts in place of its feed-forward when `routed` is set.
fn quant_model_with(
    layers: usize,
    sinks: bool,
    routed: bool,
) -> (Vec<(String, Value)>, Vec<GgufEntry>) {
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
        ] {
            entries.push(entry(format!("blk.{il}.{n}.weight"), d.to_vec(), t));
        }
        let ffn = if routed {
            vec![
                (
                    "ffn_gate_inp",
                    vec![MOE_EXPERTS, QUANT_EMBD],
                    GgmlDType::F32,
                ),
                (
                    "ffn_gate_exps",
                    vec![MOE_EXPERTS, QUANT_FFN, QUANT_EMBD],
                    GgmlDType::Q4K,
                ),
                (
                    "ffn_up_exps",
                    vec![MOE_EXPERTS, QUANT_FFN, QUANT_EMBD],
                    GgmlDType::Q4K,
                ),
                (
                    "ffn_down_exps",
                    vec![MOE_EXPERTS, QUANT_EMBD, QUANT_FFN],
                    GgmlDType::Q6K,
                ),
            ]
        } else {
            vec![
                ("ffn_gate", vec![QUANT_FFN, QUANT_EMBD], GgmlDType::Q4K),
                ("ffn_up", vec![QUANT_FFN, QUANT_EMBD], GgmlDType::Q4K),
                ("ffn_down", vec![QUANT_EMBD, QUANT_FFN], GgmlDType::Q6K),
            ]
        };
        for (n, d, t) in ffn {
            entries.push(entry(format!("blk.{il}.{n}.weight"), d, t));
        }
        if sinks {
            entries.push(entry(
                format!("blk.{il}.attn_sinks"),
                vec![QUANT_HEADS],
                GgmlDType::F32,
            ));
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
    let metadata = match routed {
        false => metadata,
        true => metadata
            .into_iter()
            .chain([
                u("expert_feed_forward_length", QUANT_FFN as u32),
                u("expert_count", MOE_EXPERTS as u32),
                u("expert_used_count", MOE_USED as u32),
            ])
            .collect(),
    };
    (metadata, entries)
}
