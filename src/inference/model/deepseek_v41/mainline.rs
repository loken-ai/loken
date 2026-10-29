//! llama.cpp's layout of this architecture, `deepseek41`: the tag, the keys its loaders read, the
//! names and shapes it reads the tensors under, and the engram tables as one table. A file in
//! this layout loads in llama.cpp, in the engines that vendor it, and here.
//!
//! [`metadata`] writes the keys from a configuration; [`tensor_names`] lists the tensors the
//! loader creates for it, no more and no fewer, since the loader refuses a file that carries a
//! tensor it does not create. [`relayout`] rewrites a file in loken's own layout into this one,
//! carrying every tensor's bytes.

use super::engram::{qk_weight, LLAMACPP_KEYS};
use super::source::{
    expert_unit_name, gguf_names, llamacpp_name, WeightSource, BLOCK_TENSORS, ENGRAM_TABLE,
    EXPERT_STACKS,
};
use super::DeepseekV41Config;
use crate::inference::load::model_dirs::ensure_room;
use crate::tensor::gguf_write::{planned_len, GgufStreamWriter, Placement, PlannedEntry};
use crate::tensor::quantized::gguf_file::{MappedGguf, Value};
use crate::tensor::quantized::GgmlDType;
use crate::tensor::{Error, Result};
use std::path::Path;

/// The architecture tag.
pub const ARCH: &str = "deepseek41";

/// The scoring functions by the id `expert_gating_func` stores.
const GATING_FUNCS: [(&str, u32); 3] = [("softmax", 1), ("sigmoid", 2), ("sqrtsoftplus", 4)];

/// The id `expert_gating_func` stores for a scoring function.
pub fn gating_func(score_func: &str) -> Option<u32> {
    GATING_FUNCS
        .iter()
        .find(|(n, _)| *n == score_func)
        .map(|(_, id)| *id)
}

/// The scoring function an `expert_gating_func` id names.
pub fn scoring_func(id: u32) -> Option<&'static str> {
    GATING_FUNCS.iter().find(|(_, i)| *i == id).map(|(n, _)| *n)
}

/// The tokenizer model, and the pre-tokenizer llama.cpp names this checkpoint's tokenizer by.
pub const TOKENIZER_MODEL: &str = "gpt2";
pub const TOKENIZER_PRE: &str = "deepseek-v3";

/// Token types as `tokenizer.ggml.token_type` stores them.
const TOKEN_NORMAL: i32 = 1;
const TOKEN_CONTROL: i32 = 3;
const TOKEN_USER_DEFINED: i32 = 4;
const TOKEN_UNUSED: i32 = 5;

/// The format the engram table is written in: ggml has no fp8 type, and Q8_0 holds the released
/// e4m3 values in about the same bytes, every value but the smallest of a block exactly.
pub const ENGRAM_TABLE_FORMAT: GgmlDType = GgmlDType::Q8_0;

/// Engram table rows decoded and encoded as one piece of the file, on every core.
pub const TABLE_ROWS_PER_PIECE: usize = 1 << 12;

/// The bytes of an engram table of `rows` rows of `head_dim` values in [`ENGRAM_TABLE_FORMAT`].
/// A row that does not tile into the format's blocks is refused.
pub fn engram_table_bytes(rows: usize, head_dim: usize) -> Result<u64> {
    let block = ENGRAM_TABLE_FORMAT.block_size();
    if !head_dim.is_multiple_of(block) {
        return Err(Error::msg(format!(
            "{ARCH}: engram rows of {head_dim} values do not tile into {} blocks of {block}",
            ENGRAM_TABLE_FORMAT.gguf_name()
        )));
    }
    Ok((rows * head_dim / block * ENGRAM_TABLE_FORMAT.type_size()) as u64)
}

/// The vocabulary arrays llama.cpp reads, from a `tokenizer.json`: one token per id below
/// `vocab_size`, an id the tokenizer leaves out named `[PAD{id}]` and unused, the merges as
/// `left right`.
pub fn vocab_metadata(json: &str, vocab_size: usize) -> Result<Vec<(String, Value)>> {
    let tok: serde_json::Value =
        serde_json::from_str(json).map_err(|e| Error::msg(format!("tokenizer.json: {e}")))?;
    let model = tok
        .get("model")
        .ok_or_else(|| Error::msg("tokenizer.json: no model"))?;
    let mut named: Vec<Option<(String, i32)>> = vec![None; vocab_size];
    if let Some(vocab) = model.get("vocab").and_then(|v| v.as_object()) {
        for (text, id) in vocab {
            if let Some(slot) = id.as_u64().and_then(|i| named.get_mut(i as usize)) {
                *slot = Some((text.clone(), TOKEN_NORMAL));
            }
        }
    }
    for added in tok
        .get("added_tokens")
        .and_then(|a| a.as_array())
        .into_iter()
        .flatten()
    {
        let (Some(id), Some(text)) = (
            added.get("id").and_then(|i| i.as_u64()),
            added.get("content").and_then(|c| c.as_str()),
        ) else {
            continue;
        };
        let special = added.get("special").and_then(|s| s.as_bool()) == Some(true);
        if let Some(slot) = named.get_mut(id as usize) {
            let kind = if special {
                TOKEN_CONTROL
            } else {
                TOKEN_USER_DEFINED
            };
            *slot = Some((text.to_string(), kind));
        }
    }
    let (tokens, types): (Vec<Value>, Vec<Value>) = named
        .into_iter()
        .enumerate()
        .map(|(id, t)| {
            let (text, kind) = t.unwrap_or_else(|| (format!("[PAD{id}]"), TOKEN_UNUSED));
            (Value::String(text), Value::I32(kind))
        })
        .unzip();
    // A merge is written `left right`, or as the pair.
    let merges: Vec<Value> = model
        .get("merges")
        .and_then(|m| m.as_array())
        .into_iter()
        .flatten()
        .filter_map(|m| match m {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Array(pair) => {
                let parts: Option<Vec<&str>> = pair.iter().map(|p| p.as_str()).collect();
                parts.map(|p| p.join(" "))
            }
            _ => None,
        })
        .map(Value::String)
        .collect();
    let s = |v: &str| Value::String(v.to_string());
    Ok(vec![
        ("tokenizer.ggml.model".into(), s(TOKENIZER_MODEL)),
        ("tokenizer.ggml.pre".into(), s(TOKENIZER_PRE)),
        ("tokenizer.ggml.tokens".into(), Value::Array(tokens)),
        ("tokenizer.ggml.token_type".into(), Value::Array(types)),
        ("tokenizer.ggml.merges".into(), Value::Array(merges)),
    ])
}

/// What a file's keys hold beyond the model's configuration.
pub struct Extras {
    pub name: String,
    pub context_length: usize,
    /// `tokenizer.*` keys, written as they are.
    pub tokenizer: Vec<(String, Value)>,
}

/// The context the model was trained to: the original context stretched by the YaRN factor.
pub fn trained_context(cfg: &DeepseekV41Config) -> usize {
    (cfg.original_seq_len as f64 * f64::from(cfg.rope_factor.max(1.0))).round() as usize
}

/// The engram layers a file of the first `n_layers` layers carries, as indices into the
/// configuration's lists.
pub fn engram_layers(cfg: &DeepseekV41Config, n_layers: usize) -> Vec<usize> {
    cfg.engram.as_ref().map_or_else(Vec::new, |e| {
        (0..e.layer_ids.len())
            .filter(|&i| e.layer_ids[i] < n_layers)
            .collect()
    })
}

/// llama.cpp's keys for `cfg`, of which the file carries the first `n_layers` layers.
/// `engram_rows` holds the rows of each engram layer's table, in the order
/// [`engram_layers`] lists them, as the file's one table stacks them.
pub fn metadata(
    cfg: &DeepseekV41Config,
    n_layers: usize,
    engram_rows: &[usize],
    extras: &Extras,
) -> Result<Vec<(String, Value)>> {
    let key = |k: &str| format!("{ARCH}.{k}");
    let u = |x: usize| Value::U32(x as u32);
    let f = Value::F32;
    let layers_below = |xs: &[usize]| {
        Value::Array(
            xs.iter()
                .filter(|&&l| l < n_layers)
                .map(|&l| Value::U32(l as u32))
                .collect(),
        )
    };
    let gating = gating_func(&cfg.score_func).ok_or_else(|| {
        Error::msg(format!(
            "{ARCH}: llama.cpp has no gating function {}",
            cfg.score_func
        ))
    })?;
    let clamps: Vec<Value> = cfg
        .swiglu_limits
        .iter()
        .take(n_layers)
        .map(|&x| Value::F32(x))
        .collect();
    // Every layer the file carries has its ratio; a file of every layer keeps the draft's too.
    let ratios = if n_layers < cfg.n_layers {
        &cfg.compress_ratios[..n_layers.min(cfg.compress_ratios.len())]
    } else {
        &cfg.compress_ratios[..]
    };
    let mut md = vec![
        ("general.architecture".into(), Value::String(ARCH.into())),
        ("general.name".into(), Value::String(extras.name.clone())),
        (key("context_length"), u(extras.context_length)),
        (key("block_count"), u(n_layers)),
        (key("embedding_length"), u(cfg.d_model)),
        (key("vocab_size"), u(cfg.vocab_size)),
        (key("attention.head_count"), u(cfg.n_head)),
        (key("attention.head_count_kv"), Value::U32(1)),
        (key("attention.key_length"), u(cfg.head_dim)),
        (key("attention.value_length"), u(cfg.head_dim)),
        (
            key("attention.layer_norm_rms_epsilon"),
            f(cfg.rms_eps as f32),
        ),
        (key("attention.q_lora_rank"), u(cfg.q_lora_rank)),
        (key("attention.output_group_count"), u(cfg.o_groups)),
        (key("attention.output_lora_rank"), u(cfg.o_lora_rank)),
        (key("attention.sliding_window"), u(cfg.window_size)),
        (key("rope.dimension_count"), u(cfg.rope_head_dim)),
        (key("rope.freq_base"), f(cfg.rope_theta)),
        (key("expert_count"), u(cfg.n_routed_experts)),
        (key("expert_used_count"), u(cfg.n_activated_experts)),
        (key("expert_shared_count"), u(cfg.n_shared_experts)),
        (key("expert_feed_forward_length"), u(cfg.moe_inter_dim)),
        (key("expert_gating_func"), Value::U32(gating)),
        (key("expert_weights_scale"), f(cfg.route_scale)),
        (key("expert_weights_norm"), Value::Bool(cfg.norm_topk)),
        (key("swiglu_clamp_exp"), Value::Array(clamps.clone())),
        (key("swiglu_clamp_shexp"), Value::Array(clamps)),
        // No layer of this model routes by token hash.
        (key("hash_layer_count"), Value::U32(0)),
        (
            key("attention.compress_ratios"),
            Value::Array(ratios.iter().map(|&r| Value::I32(r as i32)).collect()),
        ),
        (
            key("attention.compress_rope_freq_base"),
            f(cfg.compress_rope_theta),
        ),
        (
            key("attention.kv_source_layers"),
            layers_below(&cfg.kv_source_layers),
        ),
        (
            key("attention.indexer.source_layers"),
            layers_below(&cfg.index_source_layers),
        ),
        (key("attention.indexer.head_count"), u(cfg.index_n_heads)),
        (key("attention.indexer.key_length"), u(cfg.index_head_dim)),
        (key("attention.indexer.top_k"), u(cfg.index_topk)),
        (key("hyper_connection.count"), u(cfg.hc_mult)),
        (
            key("hyper_connection.sinkhorn_iterations"),
            u(cfg.hc_sinkhorn_iters),
        ),
        (key("hyper_connection.epsilon"), f(cfg.hc_eps)),
    ];
    if cfg.rope_factor != 1.0 && cfg.original_seq_len > 0 {
        md.extend([
            (key("rope.scaling.type"), Value::String("yarn".into())),
            (key("rope.scaling.factor"), f(cfg.rope_factor)),
            (
                key("rope.scaling.original_context_length"),
                u(cfg.original_seq_len),
            ),
        ]);
    }
    if cfg.candidate_source_layer >= 0 && (cfg.candidate_source_layer as usize) < n_layers {
        md.extend([
            (
                key("attention.indexer.candidate_source_layer"),
                Value::I32(cfg.candidate_source_layer as i32),
            ),
            (
                key("attention.indexer.candidate_top_k_blocks"),
                u(cfg.candidate_topk_blocks),
            ),
            (
                key("attention.indexer.candidate_block_size"),
                u(cfg.candidate_block_size),
            ),
        ]);
    }
    // loken's own keys, which llama.cpp passes over: a router temperature and the routing prior.
    if cfg.gate_temp != 1.0 {
        md.push((key("expert_gate_temp"), f(cfg.gate_temp)));
    }
    for (l, hot) in cfg.hot_experts.iter().take(n_layers).enumerate() {
        if !hot.is_empty() {
            md.push((
                key(&format!("hot_experts.{l}")),
                Value::Array(hot.iter().map(|&e| Value::U32(e as u32)).collect()),
            ));
        }
    }
    let kept = engram_layers(cfg, n_layers);
    if let (Some(e), false) = (cfg.engram.as_ref(), kept.is_empty()) {
        if engram_rows.len() != kept.len() {
            return Err(Error::msg(format!(
                "{ARCH}: {} engram tables for {} engram layers",
                engram_rows.len(),
                kept.len()
            )));
        }
        // Each layer's bucket ranges move down by the rows of the tables stacked above its own.
        let (mut multipliers, mut primes, mut offsets) = (Vec::new(), Vec::new(), Vec::new());
        let mut base = 0u64;
        for (&i, &rows) in kept.iter().zip(engram_rows) {
            multipliers.extend(e.multipliers[i].iter().map(|&m| Value::U64(m as u64)));
            primes.extend(e.primes[i].iter().map(|&p| Value::U64(p)));
            offsets.extend(e.offsets[i].iter().map(|&o| Value::U64(o + base)));
            base += rows as u64;
        }
        // The pad as a token id: the first token whose compressed id is the pad's.
        let pad = e
            .token_map
            .iter()
            .position(|&c| c == e.pad_id)
            .ok_or_else(|| Error::msg(format!("{ARCH}: no token maps to the engram pad")))?;
        let k = &LLAMACPP_KEYS;
        md.extend([
            (
                key(k.layer_ids),
                Value::Array(kept.iter().map(|&i| u(e.layer_ids[i])).collect()),
            ),
            (key(k.max_ngram), u(e.max_ngram)),
            (key(k.n_heads), u(e.n_heads)),
            (key(k.head_dim), u(e.head_dim)),
            (key(k.multipliers), Value::Array(multipliers)),
            (key(k.offsets), Value::Array(offsets)),
            (key(k.primes), Value::Array(primes)),
            (key(k.pad_id), u(pad)),
            (
                key(k.token_map),
                Value::Array(e.token_map.iter().map(|&c| Value::I32(c as i32)).collect()),
            ),
        ]);
    }
    md.extend(extras.tokenizer.iter().cloned());
    Ok(md)
}

/// The tensors llama.cpp's loader creates for `cfg`, of which the file carries the first
/// `n_layers` layers: a layer that sources the compressed KV carries the compressor and the
/// index keys, its gate when it compresses by more than one; a layer that sources the top-k
/// carries the index query and weights; an engram layer its projection and gate.
pub fn tensor_names(cfg: &DeepseekV41Config, n_layers: usize) -> Vec<String> {
    let mut names: Vec<String> = ["token_embd", "output_norm", "output"]
        .iter()
        .map(|n| format!("{n}.weight"))
        .collect();
    if !engram_layers(cfg, n_layers).is_empty() {
        names.push(ENGRAM_TABLE.into());
    }
    let engram = |l: usize| {
        cfg.engram
            .as_ref()
            .is_some_and(|e| e.layer_ids.contains(&l))
    };
    for l in 0..n_layers {
        let mut rests = vec![
            "attn_norm",
            "attn_sinks",
            "attn_q_a",
            "attn_q_a_norm",
            "attn_q_b",
            "attn_kv",
            "attn_kv_a_norm",
            "attn_output_a",
            "attn_output_b",
            "hc_attn_fn",
            "hc_attn_base",
            "hc_attn_scale",
            "hc_ffn_fn",
            "hc_ffn_base",
            "hc_ffn_scale",
        ];
        if cfg.kv_source_layers.contains(&l) {
            rests.extend(["attn_compressor_kv", "attn_compressor_norm"]);
            if cfg.compress_ratios.get(l).is_some_and(|&r| r > 1) {
                rests.push("attn_compressor_gate");
            }
            rests.extend(["indexer.attn_k", "indexer.k_norm"]);
        }
        if cfg.index_source_layers.contains(&l) {
            rests.extend(["indexer.attn_q_b", "indexer.proj"]);
        }
        if engram(l) {
            rests.extend(["engram_kv", "engram_gate"]);
        }
        rests.extend([
            "ffn_gate_inp",
            "ffn_norm",
            "ffn_gate_exps",
            "ffn_down_exps",
            "ffn_up_exps",
            "ffn_gate_shexp",
            "ffn_down_shexp",
            "ffn_up_shexp",
        ]);
        names.extend(rests.iter().map(|r| format!("blk.{l}.{r}.weight")));
        names.push(format!("blk.{l}.exp_probs_b.bias"));
    }
    names
}

/// A tensor's bytes where the mapping holds them.
fn bytes_of<'a>(src: &'a MappedGguf, name: &str) -> Result<&'a [u8]> {
    let i = src
        .content
        .tensor_infos
        .get(name)
        .ok_or_else(|| Error::msg(format!("no tensor {name}")))?;
    let start = (src.content.tensor_data_offset + i.offset) as usize;
    Ok(&src.mmap[start..start + i.size_in_bytes()])
}

/// Where one tensor of a re-laid file comes from.
enum Part {
    /// A source tensor's bytes.
    Carried(String),
    /// Experts stored one tensor each, stacked in expert order.
    Stacked(Vec<String>),
    /// An engram layer's query and key weights multiplied, f32.
    Gate(usize),
    /// Every engram layer's table in [`ENGRAM_TABLE_FORMAT`], one after the other.
    Table(Vec<usize>),
}

/// Write `out` in llama.cpp's layout from `src`, a file in loken's own: every tensor llama.cpp
/// reads, its bytes carried; experts stored one tensor each stacked again; each engram layer's
/// query and key weights multiplied into its gate; the engram tables laid out as one table.
/// The draft, and any tensor llama.cpp has no place for, is left out. A disk that cannot hold the
/// file is refused before its first byte. Returns the SHA-256 of the file written.
pub fn relayout(
    src: &MappedGguf,
    out: &Path,
    progress: &dyn Fn(usize, usize, &str),
) -> Result<String> {
    let cfg = DeepseekV41Config::from_gguf(&src.content)?;
    let info = |n: &str| src.content.tensor_infos.get(n);
    let present = |reference: &str| {
        gguf_names(reference)
            .into_iter()
            .find(|n| info(n).is_some())
    };
    let mut plan: Vec<(PlannedEntry, Part)> = Vec::new();
    let carry = |plan: &mut Vec<(PlannedEntry, Part)>, reference: &str| {
        let (Some(from), Some(to)) = (present(reference), llamacpp_name(reference)) else {
            return;
        };
        let i = &src.content.tensor_infos[&from];
        plan.push((
            PlannedEntry {
                name: to,
                dims: i.shape.dims().to_vec(),
                dtype: i.ggml_dtype,
                byte_len: i.size_in_bytes() as u64,
            },
            Part::Carried(from),
        ));
    };
    for reference in ["embed.weight", "norm.weight", "head.weight"] {
        carry(&mut plan, reference);
    }
    let n_layers = cfg.n_layers;
    let kept = engram_layers(&cfg, n_layers);
    for l in 0..n_layers {
        for rest in BLOCK_TENSORS {
            carry(&mut plan, &format!("layers.{l}.{rest}"));
        }
        let engram_layer = cfg
            .engram
            .as_ref()
            .is_some_and(|e| e.layer_ids.contains(&l));
        if engram_layer && present(&format!("layers.{l}.engram.gate.weight")).is_none() {
            let q = present(&format!("layers.{l}.engram.q_weight")).ok_or_else(|| {
                Error::msg(format!("layer {l}: an engram layer without its weights"))
            })?;
            let dims = src.content.tensor_infos[&q].shape.dims().to_vec();
            let elems: usize = dims.iter().product();
            plan.push((
                PlannedEntry {
                    name: format!("blk.{l}.engram_gate.weight"),
                    dims,
                    dtype: GgmlDType::F32,
                    byte_len: (elems * std::mem::size_of::<f32>()) as u64,
                },
                Part::Gate(l),
            ));
        }
        for (stack, w) in EXPERT_STACKS {
            let reference = format!("layers.{l}.ffn.experts.0.{w}.weight");
            let to = llamacpp_name(&reference)
                .ok_or_else(|| Error::msg(format!("no llama.cpp name for {reference}")))?;
            match present(&reference) {
                Some(from) if src.content.tensor_infos[&from].shape.dims().len() == 3 => {
                    let i = &src.content.tensor_infos[&from];
                    plan.push((
                        PlannedEntry {
                            name: to,
                            dims: i.shape.dims().to_vec(),
                            dtype: i.ggml_dtype,
                            byte_len: i.size_in_bytes() as u64,
                        },
                        Part::Carried(from),
                    ));
                }
                _ => {
                    let units: Vec<String> = (0..cfg.n_routed_experts)
                        .map(|e| expert_unit_name(l, stack, e))
                        .collect();
                    let first = info(&units[0])
                        .ok_or_else(|| Error::msg(format!("layer {l}: no {stack}")))?;
                    let mut dims = vec![cfg.n_routed_experts];
                    dims.extend(first.shape.dims());
                    let (dtype, one) = (first.ggml_dtype, first.size_in_bytes() as u64);
                    for u in &units {
                        let i = info(u).ok_or_else(|| Error::msg(format!("no tensor {u}")))?;
                        if i.ggml_dtype != dtype || i.size_in_bytes() as u64 != one {
                            return Err(Error::msg(format!("{u}: not stored as {}", units[0])));
                        }
                    }
                    plan.push((
                        PlannedEntry {
                            name: to,
                            dims,
                            dtype,
                            byte_len: one * cfg.n_routed_experts as u64,
                        },
                        Part::Stacked(units),
                    ));
                }
            }
        }
    }
    let mut engram_rows = Vec::with_capacity(kept.len());
    if let Some(e) = cfg.engram.as_ref().filter(|_| !kept.is_empty()) {
        for &i in &kept {
            let table =
                src.engram_table(&format!("layers.{}.engram.embed.weight", e.layer_ids[i]))?;
            engram_rows.push(table.n_rows());
        }
        let rows: usize = engram_rows.iter().sum();
        plan.push((
            PlannedEntry {
                name: ENGRAM_TABLE.into(),
                dims: vec![rows, e.head_dim],
                dtype: ENGRAM_TABLE_FORMAT,
                byte_len: engram_table_bytes(rows, e.head_dim)?,
            },
            Part::Table(kept.iter().map(|&i| e.layer_ids[i]).collect()),
        ));
    }

    let md = &src.content.metadata;
    let string = |k: &str| md.get(k).and_then(|v| v.to_string().ok().cloned());
    let arch = string("general.architecture").unwrap_or_default();
    let mut tokenizer: Vec<(String, Value)> = md
        .iter()
        .filter(|(k, _)| k.starts_with("tokenizer."))
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    tokenizer.sort_by(|a, b| a.0.cmp(&b.0));
    if !md.contains_key("tokenizer.ggml.tokens") {
        let json = string(super::convert::TOKENIZER_KEY).ok_or_else(|| {
            Error::msg("relayout: the file carries no tokenizer llama.cpp can read")
        })?;
        tokenizer.extend(vocab_metadata(&json, cfg.vocab_size)?);
    }
    let extras = Extras {
        name: string("general.name").unwrap_or_else(|| ARCH.to_string()),
        context_length: md
            .get(&format!("{arch}.context_length"))
            .and_then(|v| v.to_u32().ok())
            .map(|v| v as usize)
            .unwrap_or_else(|| trained_context(&cfg)),
        tokenizer,
    };
    let metadata = metadata(&cfg, n_layers, &engram_rows, &extras)?;

    let entries: Vec<PlannedEntry> = plan
        .iter()
        .map(|(e, _)| PlannedEntry {
            name: e.name.clone(),
            dims: e.dims.clone(),
            dtype: e.dtype,
            byte_len: e.byte_len,
        })
        .collect();
    let need = planned_len(out, &metadata, &entries, &Placement::packed())?;
    let room = out.parent().unwrap_or(Path::new("."));
    ensure_room("the re-laid file", room, need).map_err(Error::msg)?;
    let mut writer = GgufStreamWriter::create(out, &metadata, &entries)?;
    let total = plan.len();
    for (n, (entry, part)) in plan.iter().enumerate() {
        match part {
            Part::Carried(name) => writer.append(bytes_of(src, name)?)?,
            Part::Stacked(units) => {
                writer.append_parts(units.iter().map(|u| bytes_of(src, u).map(<[u8]>::to_vec)))?
            }
            Part::Gate(l) => {
                let gate = qk_weight(src, &format!("layers.{l}.engram"))?;
                writer.append(
                    &gate
                        .iter()
                        .flatten()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?
            }
            Part::Table(layers) => {
                let mut tables = Vec::with_capacity(layers.len());
                for l in layers {
                    let table = src.engram_table(&format!("layers.{l}.engram.embed.weight"))?;
                    table.advise_sequential();
                    tables.push(table);
                }
                writer.append_parts(tables.iter().flat_map(|t| {
                    let rows = t.n_rows();
                    (0..rows).step_by(TABLE_ROWS_PER_PIECE).map(move |start| {
                        t.encode_rows(
                            start..(start + TABLE_ROWS_PER_PIECE).min(rows),
                            ENGRAM_TABLE_FORMAT,
                        )
                    })
                }))?
            }
        }
        progress(n + 1, total, &entry.name);
    }
    writer.finish_with_digest()
}

/// Every key llama.cpp's `deepseek41` loader reads without a default: the generic loader's,
/// `llama_model_deepseek4::load_arch_hparams`', `llama_model_deepseek41::load_arch_hparams`', and
/// the vocabulary's.
pub const REQUIRED_KEYS: &[&str] = &[
    "general.architecture",
    "deepseek41.context_length",
    "deepseek41.embedding_length",
    "deepseek41.block_count",
    "deepseek41.attention.layer_norm_rms_epsilon",
    "deepseek41.attention.q_lora_rank",
    "deepseek41.attention.sliding_window",
    "deepseek41.expert_feed_forward_length",
    "deepseek41.expert_shared_count",
    "deepseek41.expert_weights_scale",
    "deepseek41.expert_weights_norm",
    "deepseek41.swiglu_clamp_exp",
    "deepseek41.attention.indexer.head_count",
    "deepseek41.attention.indexer.key_length",
    "deepseek41.attention.indexer.top_k",
    "deepseek41.attention.output_group_count",
    "deepseek41.attention.output_lora_rank",
    "deepseek41.attention.compress_rope_freq_base",
    "deepseek41.hyper_connection.count",
    "deepseek41.hyper_connection.sinkhorn_iterations",
    "deepseek41.hyper_connection.epsilon",
    "deepseek41.hash_layer_count",
    "deepseek41.attention.compress_ratios",
    "deepseek41.expert_gating_func",
    "deepseek41.attention.kv_source_layers",
    "deepseek41.attention.indexer.source_layers",
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges",
];

/// The keys an engram layer adds, read once `ple.layers` names one.
pub const ENGRAM_KEYS: &[&str] = &[
    "deepseek41.ple.layers",
    "deepseek41.ple.ngram_size",
    "deepseek41.ple.heads_per_ngram",
    "deepseek41.embedding_length_per_layer_input",
    "deepseek41.ple.layer_multipliers",
    "deepseek41.ple.head_offsets",
    "deepseek41.ple.head_vocab_sizes",
    "deepseek41.engram.pad_token_id",
    "deepseek41.engram.token_map",
];

#[cfg(test)]
mod tests {
    use super::*;

    /// The gating ids are llama.cpp's enum values, both ways.
    #[test]
    fn gating_functions_map_both_ways() {
        assert_eq!(gating_func("sqrtsoftplus"), Some(4));
        assert_eq!(scoring_func(4), Some("sqrtsoftplus"));
        assert_eq!(gating_func("sigmoid"), Some(2));
        assert_eq!(gating_func("tanh"), None);
    }

    /// Every id gets a token, the added ones typed by whether they are special, the missing ones
    /// padded; merges written either way read as `left right`.
    #[test]
    fn vocabulary_arrays_cover_every_id() {
        let json = serde_json::json!({
            "model": { "vocab": { "a": 0, "b": 1, "ab": 2 }, "merges": ["a b", ["ab", "b"]] },
            "added_tokens": [
                { "id": 3, "content": "<s>", "special": true },
                { "id": 4, "content": "<x>", "special": false }
            ]
        })
        .to_string();
        let md = vocab_metadata(&json, 6).unwrap();
        let get = |k: &str| {
            md.iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v.clone())
                .unwrap()
        };
        let tokens: Vec<String> = get("tokenizer.ggml.tokens")
            .to_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_string().unwrap().clone())
            .collect();
        assert_eq!(tokens, ["a", "b", "ab", "<s>", "<x>", "[PAD5]"]);
        let types: Vec<i64> = get("tokenizer.ggml.token_type")
            .to_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_i64().unwrap())
            .collect();
        assert_eq!(types, [1, 1, 1, 3, 4, 5]);
        let merges: Vec<String> = get("tokenizer.ggml.merges")
            .to_vec()
            .unwrap()
            .iter()
            .map(|v| v.to_string().unwrap().clone())
            .collect();
        assert_eq!(merges, ["a b", "ab b"]);
        assert_eq!(
            get("tokenizer.ggml.pre").to_string().unwrap(),
            TOKENIZER_PRE
        );
    }

    /// llama.cpp's name of every reference tensor: `.weight` on every block tensor, `.bias` on
    /// the routing bias, the engram projection as `engram_kv`, every engram table as the one
    /// table, the query and key weights as nothing.
    #[test]
    fn reference_names_map_to_llamacpp_names() {
        assert_eq!(llamacpp_name("embed.weight").unwrap(), "token_embd.weight");
        assert_eq!(
            llamacpp_name("layers.3.attn.kv_norm.weight").unwrap(),
            "blk.3.attn_kv_a_norm.weight"
        );
        assert_eq!(
            llamacpp_name("layers.3.attn.attn_sink").unwrap(),
            "blk.3.attn_sinks.weight"
        );
        assert_eq!(
            llamacpp_name("layers.3.hc_attn_base").unwrap(),
            "blk.3.hc_attn_base.weight"
        );
        assert_eq!(
            llamacpp_name("layers.3.ffn.gate.bias").unwrap(),
            "blk.3.exp_probs_b.bias"
        );
        assert_eq!(
            llamacpp_name("layers.3.ffn.experts.17.w2.weight").unwrap(),
            "blk.3.ffn_down_exps.weight"
        );
        assert_eq!(
            llamacpp_name("layers.3.engram.wkv.weight").unwrap(),
            "blk.3.engram_kv.weight"
        );
        assert_eq!(
            llamacpp_name("layers.3.engram.gate.weight").unwrap(),
            "blk.3.engram_gate.weight"
        );
        assert_eq!(
            llamacpp_name("layers.3.engram.embed.weight").unwrap(),
            ENGRAM_TABLE
        );
        assert!(llamacpp_name("layers.3.engram.q_weight").is_none());
        assert!(llamacpp_name("layers.3.engram.k_weight").is_none());
        // Every name llama.cpp reads is one the reader finds again.
        for rest in BLOCK_TENSORS {
            let reference = format!("layers.3.{rest}");
            if let Some(n) = llamacpp_name(&reference) {
                assert!(gguf_names(&reference).contains(&n), "{reference} -> {n}");
            }
        }
    }
}
