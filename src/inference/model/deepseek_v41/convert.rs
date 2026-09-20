//! A GGUF of this model from the released checkpoint, calibrated and written by loken itself.
//!
//! Two steps, each worth running on its own. `calibrate` serves the released weights over a text
//! and records, for every layer, what its MoE block took in and what each routed expert saw. From
//! that record `convert` writes the file: the routed experts in a low-bit format fitted to what
//! they saw, each block's error carried onto the columns after it; the path every token reads at a
//! higher precision; the engram tables exactly as released, beside the hash that indexes them.
//!
//! Every name and key is the one this module's own reader looks for, so the file loads here
//! without translation. The file is streamed tensor by tensor: nothing larger than one expert
//! stack is held.

use super::engram::EngramConfig;
use super::model::DeepseekV41Model;
use super::safetensors_source::SafeTensorsSource;
use super::source::{gguf_names, WeightSource};
use super::DeepseekV41Config;
use crate::inference::load::calibrated::{quantize_projection_on, Kernels, Recipe, Records};
use crate::inference::load::calibration::Calibration;
use crate::inference::load::compensate::{Calibration as Moment, Factor};
use crate::tensor::gguf_write::{GgufStreamWriter, PlannedEntry};
use crate::tensor::quantized::gguf_file::Value;
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use std::path::Path;
use std::sync::{Arc, Mutex};

/// The formats each class of tensor is written in.
#[derive(Clone, Copy, Debug)]
pub struct Formats {
    /// Routed experts' gate and up projections.
    pub gate_up: GgmlDType,
    /// Routed experts' down projection.
    pub down: GgmlDType,
    /// Every large weight a token reads whatever it is: attention, shared expert, embeddings, head.
    pub always_read: GgmlDType,
}

impl Default for Formats {
    fn default() -> Self {
        Self {
            gate_up: GgmlDType::Iq2Xxs,
            down: GgmlDType::Q2K,
            always_read: GgmlDType::Q8_0,
        }
    }
}

/// The record names a calibration writes and a conversion reads.
fn moe_input(layer: usize) -> String {
    format!("blk.{layer}.ffn_inp")
}
fn gate_up_seen(layer: usize, expert: usize) -> String {
    format!("blk.{layer}.ffn_gate_up_exps.{expert}")
}
fn down_seen(layer: usize, expert: usize) -> String {
    format!("blk.{layer}.ffn_down_exps.{expert}")
}
fn down_pooled(layer: usize) -> String {
    format!("blk.{layer}.ffn_down_exps")
}

/// The configuration the released checkpoint describes, engram hash included.
pub fn released_config(src: &SafeTensorsSource, dir: &Path) -> Result<DeepseekV41Config> {
    let (map, _) = super::token_map::build_from_file(&dir.join("tokenizer.json"))?;
    let engram = EngramConfig::derive(&src.inference_config, map);
    DeepseekV41Config::from_reference_config(&src.inference_config, engram)
}

/// Excerpts of `length` tokens for a calibration run, taken in turn from each text so that no one
/// kind of writing fills the budget, until `budget` tokens are chosen. Each text is cut from its
/// start, so the same texts always give the same excerpts.
pub fn excerpts(texts: &[Vec<u32>], length: usize, budget: usize) -> Vec<Vec<u32>> {
    let mut out = Vec::new();
    let (mut taken, mut round) = (0usize, 0usize);
    while taken + length <= budget {
        let mut any = false;
        for t in texts {
            let start = round * length;
            if start + length > t.len() {
                continue;
            }
            if taken + length > budget {
                break;
            }
            out.push(t[start..start + length].to_vec());
            taken += length;
            any = true;
        }
        if !any {
            break;
        }
        round += 1;
    }
    out
}

/// The most rows a record keeps. Every row a projection sees improves the correction's estimate, so
/// a run keeps them all when they fit: the rows a text produces are bounded by its tokens, and while
/// that bound fits in half the memory available now nothing is dropped. A longer text keeps a
/// sample, sized so the whole record still fits. `asked` overrides it.
pub fn rows_to_keep(
    cfg: &DeepseekV41Config,
    inter: usize,
    tokens: usize,
    asked: Option<usize>,
) -> usize {
    if let Some(rows) = asked {
        return rows;
    }
    // Per token and layer: the block's input once, and each chosen expert's input and down input.
    let per_token = cfg.n_layers * (cfg.d_model + cfg.n_activated_experts * (cfg.d_model + inter));
    let bytes_all = tokens
        .saturating_mul(per_token)
        .saturating_mul(std::mem::size_of::<f32>());
    let budget = {
        let mut sys = sysinfo::System::new();
        sys.refresh_memory();
        sys.available_memory() as usize / 2
    };
    if bytes_all <= budget {
        return usize::MAX;
    }
    // Scaled down with the budget; never below what the correction needs to be worth running.
    (tokens * budget / bytes_all.max(1)).max(cfg.n_activated_experts * 64)
}

/// Serve the released weights over each excerpt of `texts` and record what every MoE block took in
/// and what every routed expert saw, keeping at most `rows_kept` rows per record.
pub fn calibrate(
    src: &SafeTensorsSource,
    cfg: &DeepseekV41Config,
    texts: &[Vec<u32>],
    rows_kept: usize,
    progress: &dyn Fn(usize, usize),
) -> Result<Calibration> {
    let model = DeepseekV41Model::load(src, cfg)?;
    // Every excerpt crosses a layer before any goes on to the next, so each layer's experts are read
    // once for the whole run, and the layer's MoE runs once over every excerpt's tokens, its routed
    // experts on the cards when there are any. The layer being run keeps every expert its excerpts
    // ask for, and gives them back before the next layer is read: at most one layer's experts are
    // held at a time.
    for layer in 0..cfg.n_layers {
        model.set_layer_expert_cache(layer, 0);
    }
    let record = Arc::new(Mutex::new(Calibration::default()));
    let dim = cfg.d_model;
    let sink = record.clone();
    model.observe_experts(Some(Arc::new(
        move |layer: usize, expert: Option<usize>, rows: &[f32], _w: &[f32], down: &[f32]| {
            let mut c = sink.lock().unwrap();
            match expert {
                None => {
                    for row in rows.chunks(dim) {
                        c.observe(&moe_input(layer), row, rows_kept);
                    }
                }
                Some(e) => {
                    for row in rows.chunks(dim) {
                        c.observe(&gate_up_seen(layer, e), row, rows_kept);
                    }
                    let n = rows.len() / dim;
                    if n > 0 {
                        for row in down.chunks(down.len() / n) {
                            c.observe(&down_seen(layer, e), row, rows_kept);
                            // The pooled record only ever lends its importance: it keeps no rows.
                            c.observe(&down_pooled(layer), row, 0);
                        }
                    }
                }
            }
        },
    )));
    // The engine's streamed placement, as serving uses it: the routed experts over the lanes,
    // the attention and shared-expert products of this thread on a card, when there is one.
    let no_prior: [Vec<usize>; 0] = [];
    let streamed = crate::inference::offload::streamed::Streamed::open(
        &crate::inference::offload::streamed::Demand {
            always_read: model.resident_path_bytes(),
            transient: model.transient_bytes(texts.iter().map(|t| t.len()).max().unwrap_or(0)),
            concurrency: model.n_activated(),
            prior: &no_prior,
            fetch: &|_, _| None,
        },
    );
    model.offload_experts(streamed.as_ref().map(|s| s.lanes.clone()));
    let run = || -> Result<()> {
        let mut states = texts
            .iter()
            .map(|text| model.prefill_begin(text))
            .collect::<Result<Vec<_>>>()?;
        for layer in 0..cfg.n_layers {
            model.set_layer_expert_cache(layer, cfg.n_routed_experts);
            model.prefill_layer_batch(layer, &mut states)?;
            model.set_layer_expert_cache(layer, 0);
            progress(layer + 1, cfg.n_layers);
        }
        Ok(())
    };
    match &streamed {
        Some(s) => crate::inference::offload::with_offload(s.offload.clone(), run)?,
        None => run()?,
    }
    model.offload_experts(None);
    model.observe_experts(None);
    drop(model);
    Arc::try_unwrap(record)
        .map_err(|_| Error::msg("calibration: the record is still shared"))
        .map(|m| m.into_inner().unwrap())
}

/// A tensor's bytes, produced when its turn comes.
enum Source {
    /// Dequantised and written in `dtype`.
    Dense(String),
    /// The shard's own bytes, untouched.
    Verbatim(String),
    /// One layer's routed experts: `w1`, `w2` or `w3`.
    Experts(usize, &'static str),
}

/// The keys this module's reader looks for, from the checkpoint's own configuration.
/// The metadata key under which the file carries the checkpoint's `tokenizer.json`, whole.
pub const TOKENIZER_KEY: &str = "tokenizer.huggingface.json";

/// The checkpoint's chat format, as its `encoding/encoding.py` renders a conversation: the
/// sequence token first, a system turn when one is given, every turn behind its role token,
/// the assistant header closed by `</think>` for a plain answer and opened by `<think>` when
/// `enable_thinking` is set, in which case the encoder's default reasoning effort leads the
/// system turn. Tool turns are outside this template; the server flattens them.
pub const CHAT_TEMPLATE: &str = concat!(
    "{{ '<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>' }}",
    "{% set thinking = enable_thinking is defined and enable_thinking %}",
    "{% if thinking %}",
    "<\u{ff5c}System\u{ff5c}>Reasoning Effort: 75 (range 1-100, the higher the value, ",
    "the more thorough the reasoning)\n\n",
    "{% endif %}",
    "{% for m in messages %}",
    "{% if m.role == 'system' %}",
    "{% if not (thinking and loop.first) %}<\u{ff5c}System\u{ff5c}>{% endif %}",
    "{% if m.content %}{{ m.content }}{% endif %}",
    "{% elif m.role == 'user' %}",
    "<\u{ff5c}User\u{ff5c}>{% if m.content %}{{ m.content }}{% endif %}",
    "{% elif m.role == 'assistant' %}",
    "{% if m.content %}{{ m.content }}{% endif %}<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "{% endif %}",
    "{% if m.role != 'assistant' and ((loop.last and add_generation_prompt) ",
    "or (not loop.last and loop.nextitem.role == 'assistant')) %}",
    "<\u{ff5c}Assistant\u{ff5c}>",
    "{% if thinking and loop.last %}<think>{% else %}</think>{% endif %}",
    "{% endif %}",
    "{% endfor %}",
);

/// What the checkpoint says about its tokens: `tokenizer.json` whole, and the ids its
/// `tokenizer_config.json` names for the sequence and end tokens.
pub struct TokenizerFiles {
    pub json: String,
    pub bos: u32,
    pub eos: u32,
}

impl TokenizerFiles {
    pub fn read(dir: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(dir.join("tokenizer.json"))
            .map_err(|e| Error::msg(format!("tokenizer.json: {e}")))?;
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("tokenizer_config.json"))
                .map_err(|e| Error::msg(format!("tokenizer_config.json: {e}")))?,
        )
        .map_err(|e| Error::msg(format!("tokenizer_config.json: {e}")))?;
        let tok: serde_json::Value =
            serde_json::from_str(&json).map_err(|e| Error::msg(format!("tokenizer.json: {e}")))?;
        let id = |key: &str| -> Result<u32> {
            let name = match cfg.get(key) {
                Some(serde_json::Value::String(s)) => s.clone(),
                Some(v) => v
                    .get("content")
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| {
                        Error::msg(format!("tokenizer_config.json: {key} has no content"))
                    })?
                    .to_string(),
                None => return Err(Error::msg(format!("tokenizer_config.json: no {key}"))),
            };
            tok.get("added_tokens")
                .and_then(|a| a.as_array())
                .and_then(|a| {
                    a.iter()
                        .find(|t| t.get("content").and_then(|c| c.as_str()) == Some(&name))
                })
                .and_then(|t| t.get("id"))
                .and_then(|i| i.as_u64())
                .map(|i| i as u32)
                .ok_or_else(|| Error::msg(format!("tokenizer.json: {name} is not an added token")))
        };
        Ok(Self {
            bos: id("bos_token")?,
            eos: id("eos_token")?,
            json,
        })
    }
}

/// The tokenizer's metadata: the file whole, the two token ids the loader stops and starts
/// on, and the chat format. The sequence token is written by the template, not by the
/// tokenizer, as the checkpoint's configuration says.
/// The routing prior as metadata: for each layer, its experts from the most routed-to on the
/// calibration corpus downwards. A loader reads the head of each list to fill its cards before
/// the first request, where they would otherwise fill over that request's first tokens.
pub fn hot_experts_metadata(lists: &[Vec<u32>]) -> Vec<(String, Value)> {
    lists
        .iter()
        .enumerate()
        .map(|(l, ids)| {
            (
                format!("{}.hot_experts.{l}", super::ARCH),
                Value::Array(ids.iter().map(|&x| Value::U32(x)).collect()),
            )
        })
        .collect()
}

pub fn tokenizer_metadata(files: &TokenizerFiles) -> Vec<(String, Value)> {
    vec![
        (TOKENIZER_KEY.to_string(), Value::String(files.json.clone())),
        (
            "tokenizer.ggml.bos_token_id".to_string(),
            Value::U32(files.bos),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            Value::U32(files.eos),
        ),
        (
            "tokenizer.ggml.add_bos_token".to_string(),
            Value::Bool(false),
        ),
        (
            "tokenizer.chat_template".to_string(),
            Value::String(CHAT_TEMPLATE.to_string()),
        ),
    ]
}

/// The checkpoint's own configuration, as the file will carry it. The draft layers the
/// checkpoint holds under `mtp` are not among the keys written here: this converter does not
/// take them, and a file that announced them would promise a reader something it cannot serve.
fn metadata(
    cfg: &serde_json::Value,
    engram: Option<&EngramConfig>,
    tokenizer: Option<&TokenizerFiles>,
) -> Vec<(String, Value)> {
    let arch = super::ARCH;
    let mut md = vec![
        (
            "general.architecture".to_string(),
            Value::String(arch.to_string()),
        ),
        (
            "general.name".to_string(),
            Value::String("DeepSeek-V4.1-Flash".into()),
        ),
    ];
    let key = |k: &str| format!("{arch}.{k}");
    let u = |v: &str| cfg.get(v).and_then(|x| x.as_u64());
    let f = |v: &str| cfg.get(v).and_then(|x| x.as_f64());
    for (k, v) in [
        ("block_count", "n_layers"),
        ("embedding_length", "dim"),
        ("vocab_size", "vocab_size"),
        ("attention.head_count", "n_heads"),
        ("attention.key_length", "head_dim"),
        ("rope.dimension_count", "rope_head_dim"),
        ("attention.q_lora_rank", "q_lora_rank"),
        ("attention.o_lora_rank", "o_lora_rank"),
        ("attention.o_groups", "o_groups"),
        ("attention.sliding_window", "window_size"),
        ("expert_count", "n_routed_experts"),
        ("expert_shared_count", "n_shared_experts"),
        ("expert_used_count", "n_activated_experts"),
        ("expert_feed_forward_length", "moe_inter_dim"),
        ("indexer.head_count", "index_n_heads"),
        ("indexer.key_length", "index_head_dim"),
        ("indexer.top_k", "index_topk"),
        ("hyper_connection.count", "hc_mult"),
        ("hyper_connection.sinkhorn_iterations", "hc_sinkhorn_iters"),
        ("attention.candidate_source_layer", "candidate_source_layer"),
        ("attention.candidate_topk_blocks", "candidate_topk_blocks"),
        ("attention.candidate_block_size", "candidate_block_size"),
        ("rope.scaling.original_context_length", "original_seq_len"),
    ] {
        if let Some(x) = u(v) {
            md.push((key(k), Value::U32(x as u32)));
        }
    }
    md.push((key("attention.head_count_kv"), Value::U32(1)));
    for (k, v) in [
        ("attention.layer_norm_rms_epsilon", "norm_eps"),
        ("expert_weights_scale", "route_scale"),
        ("expert_swiglu_limit", "swiglu_limit"),
        ("hyper_connection.epsilon", "hc_eps"),
        ("rope.freq_base", "rope_theta"),
        ("rope.scaling.factor", "rope_factor"),
        ("attention.compress_rope_freq_base", "compress_rope_theta"),
    ] {
        if let Some(x) = f(v) {
            md.push((key(k), Value::F32(x as f32)));
        }
    }
    if let Some(s) = cfg.get("score_func").and_then(|x| x.as_str()) {
        md.push((key("expert_scoring_func"), Value::String(s.to_string())));
    }
    let arr = |xs: &[u64]| Value::Array(xs.iter().map(|&x| Value::U32(x as u32)).collect());
    let arr64 = |xs: &[u64]| Value::Array(xs.iter().map(|&x| Value::U64(x)).collect());
    let list = |v: &str| -> Option<Vec<u64>> {
        cfg.get(v)?.as_array()?.iter().map(|x| x.as_u64()).collect()
    };
    for (k, v) in [
        ("attention.compress_ratios", "compress_ratios"),
        ("attention.kv_source_layers", "kv_source_layers"),
        ("attention.index_source_layers", "index_source_layers"),
    ] {
        if let Some(xs) = list(v) {
            md.push((key(k), arr(&xs)));
        }
    }
    // A table is unusable without the hash that indexed it: the file carries that hash whole.
    if let Some(e) = engram {
        let pad = u("engram_pad_id").unwrap_or(0);
        md.push((key("engram.head_count"), Value::U32(e.n_heads as u32)));
        md.push((key("engram.key_length"), Value::U32(e.head_dim as u32)));
        md.push((key("engram.max_ngram_size"), Value::U32(e.max_ngram as u32)));
        md.push((key("engram.pad_id"), Value::U32(pad as u32)));
        md.push((
            key("engram.layer_ids"),
            arr64(&e.layer_ids.iter().map(|&x| x as u64).collect::<Vec<_>>()),
        ));
        md.push((
            key("engram.multipliers"),
            arr64(
                &e.multipliers
                    .iter()
                    .flatten()
                    .map(|&x| x as u64)
                    .collect::<Vec<_>>(),
            ),
        ));
        md.push((key("engram.primes"), arr64(&e.primes.concat())));
        md.push((key("engram.offsets"), arr64(&e.offsets.concat())));
        md.push((
            key("engram.token_map"),
            arr64(&e.token_map.iter().map(|&x| x as u64).collect::<Vec<_>>()),
        ));
    }
    if let Some(files) = tokenizer {
        md.extend(tokenizer_metadata(files));
    }
    md
}

fn bytes_per(dtype: GgmlDType, elems: usize) -> u64 {
    (elems / dtype.block_size() * dtype.type_size()) as u64
}

/// Workers that quantise experts at once, spread over the accelerators when there are any. Each
/// expert's CPU work already spreads over the cores, so a handful of experts at a time is what keeps
/// the cores and the cards fed without starving each other.
const WORKERS: usize = 4;

/// Streams opened on each accelerator: a worker holds one, so this many experts are on a card at once.
#[cfg(feature = "cuda")]
const STREAMS_PER_CARD: usize = 4;

/// The accelerators present, each opened `STREAMS_PER_CARD` times: every handle has its own stream
/// and its own cuBLAS handle, so the workers holding them do not wait on each other.
#[cfg(feature = "cuda")]
pub(super) fn accelerators() -> Vec<Arc<crate::tensor::cuda::CudaDevice>> {
    let cards = (0..)
        .take_while(|&i| crate::tensor::cuda::CudaDevice::new(i).is_ok())
        .count();
    (0..STREAMS_PER_CARD)
        .flat_map(|_| 0..cards)
        .filter_map(|i| crate::tensor::cuda::CudaDevice::new(i).ok())
        .collect()
}

/// One projection with the correction's products and the blocks it has kernels for on `dev`, the rest
/// on the CPU.
#[cfg(feature = "cuda")]
fn quantize_on_device(
    dev: &crate::tensor::cuda::CudaDevice,
    dtype: GgmlDType,
    values: &[f32],
    shape: (usize, usize),
    record: Option<&(Vec<f32>, Vec<f32>, usize)>,
    compensate: bool,
    recipe: Recipe,
    factor: Option<&Factor>,
) -> Result<Vec<u8>> {
    use crate::tensor::cuda::{gpu_matmul_host, gpu_quantize_iq2_xxs, gpu_quantize_q2_k_guided};
    let product = |a: &[f32], b: &[f32], out: &mut [f32], dims: (usize, usize, usize)| {
        gpu_matmul_host(dev, a, b, out, dims)
    };
    let encode = |dtype: GgmlDType, v: &[f32], width: usize, imp: Option<&[f32]>| match (dtype, imp)
    {
        (GgmlDType::Iq2Xxs, _) => Some(gpu_quantize_iq2_xxs(dev, v, imp, width)),
        (GgmlDType::Q2K, Some(imp)) => Some(gpu_quantize_q2_k_guided(dev, v, imp, width)),
        _ => None,
    };
    let kernels = Kernels {
        product: &product,
        encode: &encode,
    };
    quantize_projection_on(
        dtype,
        values,
        shape,
        record,
        compensate,
        recipe,
        factor,
        Some(&kernels),
    )
}

/// Write the file. `records` is a calibration read back with `calibration::read`. `layers` limits the
/// blocks written, for a partial file that checks a recipe on its first layers before the whole
/// model is spent on it. `tokenizer` is the checkpoint's `tokenizer.json`, carried whole so the file
/// serves without a sidecar. Returns the SHA-256 of the file, hashed as it was written.
#[allow(clippy::too_many_arguments)]
pub fn convert(
    src: &SafeTensorsSource,
    cfg: &DeepseekV41Config,
    records: &Records,
    out: &Path,
    formats: Formats,
    recipe: Recipe,
    layers: std::ops::Range<usize>,
    tokenizer: Option<&TokenizerFiles>,
    progress: &dyn Fn(usize, usize, &str),
) -> Result<String> {
    let mut plan: Vec<(PlannedEntry, Source)> = Vec::new();
    let dense =
        |name: &str, shape: &[usize], plan: &mut Vec<(PlannedEntry, Source)>| -> Result<()> {
            let Some(target) = gguf_names(name).into_iter().next().filter(|n| n != name) else {
                return Ok(());
            };
            let elems: usize = shape.iter().product();
            let large = shape.len() == 2 && elems >= 1 << 20;
            let dtype = if large && shape[1].is_multiple_of(formats.always_read.block_size()) {
                formats.always_read
            } else {
                GgmlDType::F32
            };
            plan.push((
                PlannedEntry {
                    name: target,
                    dims: shape.to_vec(),
                    dtype,
                    byte_len: bytes_per(dtype, elems),
                },
                Source::Dense(name.to_string()),
            ));
            Ok(())
        };

    for name in ["embed.weight", "norm.weight", "head.weight"] {
        let shape = src
            .shape(name)
            .ok_or_else(|| Error::msg(format!("no tensor {name}")))?;
        dense(name, &shape, &mut plan)?;
    }
    let mut names: Vec<String> = src.names().into_iter().map(str::to_string).collect();
    names.sort();
    for layer in layers.start..layers.end.min(cfg.n_layers) {
        let prefix = format!("layers.{layer}.");
        for name in names.iter().filter(|n| n.starts_with(&prefix)) {
            if name.contains(".ffn.experts.") || name.ends_with(".scale") {
                continue;
            }
            if name.ends_with("engram.embed.weight") {
                let target = gguf_names(name).remove(0);
                let (_, shape, _) = src.raw(name)?;
                let scale = format!("{}.scale", name.strip_suffix(".weight").unwrap_or(name));
                let (_, sshape, _) = src.raw(&scale)?;
                plan.push((
                    PlannedEntry {
                        name: target.clone(),
                        dims: shape.clone(),
                        dtype: GgmlDType::I8,
                        byte_len: shape.iter().product::<usize>() as u64,
                    },
                    Source::Verbatim(name.clone()),
                ));
                plan.push((
                    PlannedEntry {
                        name: format!("{target}.scale"),
                        dims: sshape.clone(),
                        dtype: GgmlDType::I8,
                        byte_len: sshape.iter().product::<usize>() as u64,
                    },
                    Source::Verbatim(scale),
                ));
                continue;
            }
            let shape = src
                .shape(name)
                .ok_or_else(|| Error::msg(format!("no tensor {name}")))?;
            dense(name, &shape, &mut plan)?;
        }
        for (w, stack, dtype) in [
            ("w1", "ffn_gate_exps", formats.gate_up),
            ("w3", "ffn_up_exps", formats.gate_up),
            ("w2", "ffn_down_exps", formats.down),
        ] {
            let one = src
                .shape(&format!("layers.{layer}.ffn.experts.0.{w}.weight"))
                .ok_or_else(|| Error::msg(format!("layer {layer}: no expert {w}")))?;
            let dims = vec![cfg.n_routed_experts, one[0], one[1]];
            let elems: usize = dims.iter().product();
            plan.push((
                PlannedEntry {
                    name: format!("blk.{layer}.{stack}"),
                    dims,
                    dtype,
                    byte_len: bytes_per(dtype, elems),
                },
                Source::Experts(layer, w),
            ));
        }
    }

    let entries: Vec<PlannedEntry> = plan
        .iter()
        .map(|(e, _)| PlannedEntry {
            name: e.name.clone(),
            dims: e.dims.clone(),
            dtype: e.dtype,
            byte_len: e.byte_len,
        })
        .collect();
    let mut writer = GgufStreamWriter::create(
        out,
        &metadata(&src.inference_config, cfg.engram.as_ref(), tokenizer),
        &entries,
    )?;
    let total = plan.len();
    // Opened once for the whole file: a context costs seconds and holds memory on the card.
    #[cfg(feature = "cuda")]
    let devices = accelerators();
    for (i, (entry, source)) in plan.iter().enumerate() {
        match source {
            Source::Dense(name) => {
                let values = src.dense_f32(name)?.flatten_all()?.to_vec1::<f32>()?;
                let bytes = if entry.dtype == GgmlDType::F32 {
                    values.iter().flat_map(|v| v.to_le_bytes()).collect()
                } else {
                    crate::tensor::quant_cpu::from_float_bytes(entry.dtype, &values)?
                };
                writer.append(&bytes)?;
            }
            Source::Verbatim(name) => {
                let (_, _, bytes) = src.raw(name)?;
                writer.append(bytes.as_slice())?;
            }
            Source::Experts(layer, w) => {
                let (layer, w) = (*layer, *w);
                let (rows, cols, dtype) = (entry.dims[1], entry.dims[2], entry.dtype);
                // Gate and up see the rows routed to the expert, or the whole block's input when it
                // saw too few; down sees its own rows, or takes the layer's pooled weights without
                // their rows, which describe other experts.
                let record_for = |e: usize| {
                    if w == "w2" {
                        match records
                            .get(&down_seen(layer, e))
                            .filter(|r| r.2 >= recipe.min_rows)
                        {
                            Some(r) => (Some(r), true),
                            None => (records.get(&down_pooled(layer)), false),
                        }
                    } else {
                        match records
                            .get(&gate_up_seen(layer, e))
                            .filter(|r| r.2 >= recipe.min_rows)
                        {
                            Some(r) => (Some(r), true),
                            None => (records.get(&moe_input(layer)), true),
                        }
                    }
                };
                let weights = |e: usize| -> Result<Vec<f32>> {
                    src.dense_f32(&format!("layers.{layer}.ffn.experts.{e}.{w}.weight"))?
                        .flatten_all()?
                        .to_vec1::<f32>()
                };
                let next = std::sync::atomic::AtomicUsize::new(0);
                let results: Mutex<Vec<Option<Result<Vec<u8>>>>> =
                    Mutex::new((0..cfg.n_routed_experts).map(|_| None).collect());
                let claim = || {
                    let e = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    (e < cfg.n_routed_experts).then_some(e)
                };
                let store = |e: usize, r: Result<Vec<u8>>| results.lock().unwrap()[e] = Some(r);
                // Experts that fall back on one record share its factor: computed by the first to
                // need it, keyed by the record's address, dropped with the stack.
                type Shared = Arc<std::sync::OnceLock<std::result::Result<Factor, String>>>;
                let factors: Mutex<std::collections::HashMap<usize, Shared>> = Mutex::default();
                let factor_of = |record: &(Vec<f32>, Vec<f32>, usize)| -> Result<Shared> {
                    let slot = factors
                        .lock()
                        .unwrap()
                        .entry(record as *const _ as usize)
                        .or_default()
                        .clone();
                    let (_, sample, kept) = record;
                    let moment = Moment {
                        rows: sample,
                        n: *kept,
                        damping: recipe.damping,
                    };
                    let done =
                        slot.get_or_init(|| Factor::new(&moment, cols).map_err(|e| e.to_string()));
                    done.as_ref().map_err(|e| Error::msg(e.clone()))?;
                    Ok(slot)
                };
                // The same factors on each card, uploaded by the first worker there to need one.
                #[cfg(feature = "cuda")]
                let on_cards: Mutex<
                    std::collections::HashMap<(usize, usize), Arc<crate::tensor::cuda::CardFactor>>,
                > = Mutex::default();
                let one = |device: usize, e: usize| -> Result<Vec<u8>> {
                    let (record, compensate) = record_for(e);
                    let v = weights(e)?;
                    let shared = match record {
                        Some(r) if compensate && r.2 >= recipe.min_rows => Some(factor_of(r)?),
                        _ => None,
                    };
                    let factor = shared
                        .as_ref()
                        .and_then(|s| s.get())
                        .and_then(|f| f.as_ref().ok());
                    #[cfg(feature = "cuda")]
                    if let (Some(dev), Some(r), Some(f)) = (devices.get(device), record, factor) {
                        if matches!(dtype, GgmlDType::Iq2Xxs | GgmlDType::Q2K) {
                            use crate::inference::load::compensate::block_inverse;
                            use crate::tensor::cuda::{card_factor, quantize_compensated_on_card};
                            let card = {
                                let mut held = on_cards.lock().unwrap();
                                let key = (r as *const _ as usize, device);
                                match held.get(&key) {
                                    Some(c) => c.clone(),
                                    None => {
                                        let c = Arc::new(card_factor(
                                            dev,
                                            &r.1,
                                            f.solved(),
                                            (r.2, cols),
                                            f.lambda(),
                                        )?);
                                        held.insert(key, c.clone());
                                        c
                                    }
                                }
                            };
                            return quantize_compensated_on_card(
                                dev,
                                dtype,
                                &v,
                                (rows, cols),
                                &r.0,
                                &card,
                                &block_inverse,
                            );
                        }
                    }
                    #[cfg(feature = "cuda")]
                    if let Some(dev) = devices.get(device) {
                        return quantize_on_device(
                            dev,
                            dtype,
                            &v,
                            (rows, cols),
                            record,
                            compensate,
                            recipe,
                            factor,
                        );
                    }
                    let _ = device;
                    quantize_projection_on(
                        dtype,
                        &v,
                        (rows, cols),
                        record,
                        compensate,
                        recipe,
                        factor,
                        None,
                    )
                };
                #[cfg(feature = "cuda")]
                let cards = devices.len().max(1);
                #[cfg(not(feature = "cuda"))]
                let cards = 1;
                std::thread::scope(|scope| {
                    for worker in 0..WORKERS.max(cards) {
                        let (claim, store, one) = (&claim, &store, &one);
                        scope.spawn(move || {
                            while let Some(e) = claim() {
                                store(e, one(worker % cards, e));
                            }
                        });
                    }
                });
                let parts = results
                    .into_inner()
                    .unwrap()
                    .into_iter()
                    .enumerate()
                    .map(|(e, r)| {
                        r.unwrap_or_else(|| {
                            Err(Error::msg(format!(
                                "layer {layer}: expert {e} was never quantised"
                            )))
                        })
                    });
                writer.append_parts(parts)?;
            }
        }
        progress(i + 1, total, &entry.name);
    }
    writer.finish_with_digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Excerpts come from every text in turn, never cross a text's end, and stop at the budget.
    #[test]
    fn excerpts_take_each_text_in_turn_within_the_budget() {
        let a: Vec<u32> = (0..10).collect();
        let b: Vec<u32> = (100..125).collect();
        let got = excerpts(&[a, b], 4, 16);
        assert_eq!(
            got,
            vec![
                vec![0, 1, 2, 3],
                vec![100, 101, 102, 103],
                vec![4, 5, 6, 7],
                vec![104, 105, 106, 107]
            ]
        );
        let short = excerpts(&[vec![1, 2, 3]], 4, 100);
        assert!(short.is_empty());
    }
}
