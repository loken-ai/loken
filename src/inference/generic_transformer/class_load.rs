//! A transformer of one class of a dense safetensors checkpoint, loaded through `HeteroPlan`
//! and the ladder.
//!
//! The class is the checkpoint's own or one nested in it.
//! A nested class reads the checkpoint's configuration under its own class row.

use super::GenericHeteroTransformer;
use crate::inference::engine::llm_engine::KvQuant;
use crate::inference::load::awq_loader as awql;
use crate::inference::load::hf_config::{every_layer_window, hf_class, nested_config};
use crate::inference::place::ladder;
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::safetensors::MmapedSafetensors;
use crate::tensor::{Device, Error, Result};
use std::collections::HashMap;

/// A transformer loaded by [`load_class`], and the device its last layer runs on.
pub struct LoadedClass {
    pub model: GenericHeteroTransformer,
    /// The device of the last layer: the normed rows land there, and the head sits there.
    pub last_device: Device,
    /// Rows the cache holds.
    pub context: usize,
}

/// The rows a class's cache is asked to hold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rows {
    /// This many rows, whatever the cards hold.
    Exactly(usize),
    /// The most rows the placement affords, up to this many.
    UpTo(usize),
}

/// What a class asks of the cards, its cache priced by the row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClassDemand {
    pub layers: usize,
    pub block_bytes: u64,
    /// The head's bytes and the bytes the caller places beside it.
    pub head_bytes: u64,
    /// Cache bytes a row adds to a layer.
    pub row_bytes: u64,
    /// The most rows a layer's cache holds whatever the context, for a class whose every layer
    /// sees a window of its past.
    pub held_rows: Option<u64>,
    pub runtime_reserve: u64,
}

impl ClassDemand {
    /// The demand at a cache of `rows` rows.
    ///
    /// A windowed cache holds no more than its window's rows.
    pub fn at(&self, rows: u64) -> ladder::Demand {
        let rows = self.held_rows.map_or(rows, |held| rows.min(held));
        ladder::Demand {
            layers: self.layers,
            block_bytes: self.block_bytes,
            head_bytes: self.head_bytes,
            kv_bytes_per_layer: self.row_bytes.saturating_mul(rows),
            runtime_reserve: self.runtime_reserve,
        }
    }

    /// The rows `rows` asks for on `cards`, by the cards' `bandwidth`.
    ///
    /// An exact count is kept.
    /// A ceiling gives the context a language model takes for a window nobody asked for.
    /// It halves from the ceiling down to the working window while its plan sends a layer to
    /// the host, spreads over more cards, or reads a token's weights more slowly.
    /// With no card it is the ceiling.
    pub fn rows_on(
        &self,
        rows: Rows,
        cards: &[(usize, u64)],
        bandwidth: impl Fn(usize) -> Option<f64>,
    ) -> usize {
        match rows {
            Rows::Exactly(n) => n,
            Rows::UpTo(full) => crate::inference::place::llm::context_the_cards_hold(
                full as u64,
                crate::inference::cache::KV_WORKING_WINDOW_TOKENS as u64,
                cards,
                bandwidth,
                false,
                false,
                |n| self.at(n),
            ) as usize,
        }
    }
}

/// The transformer of `class` in the checkpoint at `dir`, its cache holding `rows`, placed on
/// the cards the ladder sees.
///
/// The configuration is the directory's `config.json`, the weights its safetensors files.
pub fn load_class(
    dir: &std::path::Path,
    class: &str,
    rows: Rows,
    beside: u64,
) -> Result<LoadedClass> {
    let msg = |e: anyhow::Error| Error::msg(e.to_string());
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    let root: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| Error::msg(format!("{}: {e}", path.display())))?;
    let (_, shards) = awql::checkpoint_shards(dir).map_err(msg)?;
    load_class_from(&root, &shards, class, rows, beside, &ladder::cards(0))
}

/// What `class` read from the configuration `root` and the safetensors files `shards` asks of
/// the cards, with `beside` bytes counted with its head.
///
/// The cache is unquantised, of the width the route gives an unquantised cache of the class's
/// architecture.
/// The cache of a class windowed in every layer holds its window and a prefill chunk at most.
pub fn class_demand(
    root: &serde_json::Value,
    shards: &[String],
    class: &str,
    beside: u64,
) -> Result<ClassDemand> {
    let (cfg, st) = class_config(root, shards, class)?;
    demand_of(&cfg, &st, class, beside)
}

/// The configuration of `class` read from `root`, and the mapped `shards`.
fn class_config(
    root: &serde_json::Value,
    shards: &[String],
    class: &str,
) -> Result<(awql::AwqConfigJson, MmapedSafetensors)> {
    let msg = |e: anyhow::Error| Error::msg(e.to_string());
    let row = hf_class(class).ok_or_else(|| Error::msg(format!("no class row for {class}")))?;
    let cfg = awql::config_from_value(nested_config(root, class), class.into(), row.arch.into())
        .map_err(msg)?;
    let st = unsafe { MmapedSafetensors::multi(shards)? };
    Ok((cfg, st))
}

/// [`class_demand`] of the configuration `cfg` and the mapped tensors `st`.
fn demand_of(
    cfg: &awql::AwqConfigJson,
    st: &MmapedSafetensors,
    class: &str,
    beside: u64,
) -> Result<ClassDemand> {
    let (block_bytes, head_bytes) = super::text_bytes(st, cfg)
        .ok_or_else(|| Error::msg(format!("{class}: no tensor layout reads the checkpoint")))?;
    let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.n_head.max(1));
    // An unquantised cache holds F16 keys and values on a card for an architecture that
    // keeps its cache unquantised, and F32 otherwise.
    let width = match crate::inference::model_facts::facts_or_default(&cfg.gguf_arch)
        .quantized_kv_degrades
    {
        true => std::mem::size_of::<half::f16>(),
        false => std::mem::size_of::<f32>(),
    };
    Ok(ClassDemand {
        layers: cfg.n_layers,
        block_bytes,
        head_bytes: head_bytes + beside,
        row_bytes: (2 * cfg.n_kv_head * head_dim * width) as u64,
        held_rows: every_layer_window(cfg)
            .map(|w| (w + crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS) as u64),
        runtime_reserve: crate::inference::place::runtime_demand::llm_work_reserve(
            cfg.hidden_size.max(cfg.n_kv_head * head_dim),
            crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS,
        ),
    })
}

/// The transformer of `class` read from the configuration `root` and the safetensors files
/// `shards`, its cache holding `rows`, placed on `cards`.
///
/// `class` is the class of `root` or a class nested in it.
/// The ladder places [`class_demand`]: the blocks, the head, the cache and the work reserve.
/// An empty `cards` places every layer on the host.
pub fn load_class_from(
    root: &serde_json::Value,
    shards: &[String],
    class: &str,
    rows: Rows,
    beside: u64,
    cards: &[(usize, u64)],
) -> Result<LoadedClass> {
    let (cfg, st) = class_config(root, shards, class)?;
    let priced = demand_of(&cfg, &st, class, beside)?;
    let context = priced.rows_on(rows, cards, ladder::card_bandwidth);
    if let Rows::UpTo(full) = rows {
        tracing::info!("{class}: a cache of {context} rows of {full}");
    }
    let holds = |tail: &str| st.names().iter().any(|n| n.ends_with(tail));
    let present = awql::AwqTensorPresence {
        has_q_bias: holds("layers.0.self_attn.q_proj.bias"),
        has_qk_norm: holds("layers.0.self_attn.q_norm.weight"),
    };
    let mut header = awql::synth_gguf_content(&cfg, present);
    let arch = cfg.gguf_arch.clone();
    use crate::tensor::quantized::gguf_file::Value;
    header
        .metadata
        .insert(format!("{arch}.context_length"), Value::U32(context as u32));
    header.metadata.insert(
        format!("{arch}.vocab_size"),
        Value::U32(cfg.vocab_size as u32),
    );
    let ck = super::DenseCheckpoint::open(&header, shards, &cfg)?;
    let layers = cfg.n_layers;
    ladder::load_on(
        class,
        &priced.at(context as u64),
        cards,
        |plan: &HeteroPlan| {
            let mut cards = HashMap::new();
            for idx in ladder::cards_of(plan) {
                cards.insert(idx, Device::new_cuda(idx)?);
            }
            let model = GenericHeteroTransformer::from_dense_checkpoint(
                &ck,
                &cards,
                plan,
                KvQuant::Off,
                Some(context),
                false,
                false,
            )?;
            let last_device = ladder::layer_devices(plan, layers)
                .pop()
                .unwrap_or(Device::Cpu);
            Ok(LoadedClass {
                model,
                last_device,
                context,
            })
        },
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::inference::load::hf_config::tests::{speech_config, PREDICTOR_CLASS, SPEECH_CLASS};
    use crate::tensor::Tensor;
    use serde_json::Value as Json;

    /// A scratch directory of its own for `tag`.
    pub(crate) fn scratch(tag: &str) -> std::path::PathBuf {
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let call = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("loken-class-{tag}-{}-{call}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Values of a deterministic pattern in `[-scale, scale)` around `centre`.
    pub(crate) fn values(seed: &mut u32, n: usize, centre: f32, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                centre + ((*seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * scale
            })
            .collect()
    }

    /// The stored tensors of a speech checkpoint of configuration `root`, with `text_vocab`
    /// rows of text table: name, shape, centre and spread of the values.
    fn speech_tensors(root: &Json, text_vocab: usize) -> Vec<(String, Vec<usize>, f32, f32)> {
        let t = &root["talker_config"];
        let p = &t["code_predictor_config"];
        let u = |v: &Json, k: &str| v[k].as_u64().unwrap() as usize;
        let mut out = Vec::new();
        let mut model = |prefix: &str, c: &Json| {
            let (h, f, hd) = (
                u(c, "hidden_size"),
                u(c, "intermediate_size"),
                u(c, "head_dim"),
            );
            let (q, kv) = (
                u(c, "num_attention_heads") * hd,
                u(c, "num_key_value_heads") * hd,
            );
            for il in 0..u(c, "num_hidden_layers") {
                for (part, dims, centre, scale) in [
                    ("input_layernorm", vec![h], 1.0, 0.2),
                    ("post_attention_layernorm", vec![h], 1.0, 0.2),
                    ("self_attn.q_norm", vec![hd], 1.0, 0.2),
                    ("self_attn.k_norm", vec![hd], 1.0, 0.2),
                    ("self_attn.q_proj", vec![q, h], 0.0, 0.1),
                    ("self_attn.k_proj", vec![kv, h], 0.0, 0.1),
                    ("self_attn.v_proj", vec![kv, h], 0.0, 0.1),
                    ("self_attn.o_proj", vec![h, q], 0.0, 0.1),
                    ("mlp.gate_proj", vec![f, h], 0.0, 0.1),
                    ("mlp.up_proj", vec![f, h], 0.0, 0.1),
                    ("mlp.down_proj", vec![h, f], 0.0, 0.1),
                ] {
                    out.push((
                        format!("{prefix}layers.{il}.{part}.weight"),
                        dims,
                        centre,
                        scale,
                    ));
                }
            }
            out.push((format!("{prefix}norm.weight"), vec![h], 1.0, 0.2));
        };
        model("talker.model.", t);
        model("talker.code_predictor.model.", p);
        let (h, ph) = (u(t, "hidden_size"), u(p, "hidden_size"));
        let (tv, pv) = (u(t, "vocab_size"), u(p, "vocab_size"));
        out.push((
            "talker.model.codec_embedding.weight".into(),
            vec![tv, h],
            0.0,
            1.0,
        ));
        out.push((
            "talker.model.text_embedding.weight".into(),
            vec![text_vocab, h],
            0.0,
            1.0,
        ));
        out.push(("talker.codec_head.weight".into(), vec![tv, h], 0.0, 0.1));
        for fc in ["linear_fc1", "linear_fc2"] {
            out.push((
                format!("talker.text_projection.{fc}.weight"),
                vec![h, h],
                0.0,
                0.1,
            ));
            out.push((
                format!("talker.text_projection.{fc}.bias"),
                vec![h],
                0.0,
                0.1,
            ));
        }
        let mtp = "talker.code_predictor.small_to_mtp_projection";
        out.push((format!("{mtp}.weight"), vec![ph, h], 0.0, 0.1));
        out.push((format!("{mtp}.bias"), vec![ph], 0.0, 0.1));
        for g in 0..u(t, "num_code_groups") - 1 {
            out.push((
                format!("talker.code_predictor.model.codec_embedding.{g}.weight"),
                vec![pv, h],
                0.0,
                1.0,
            ));
            out.push((
                format!("talker.code_predictor.lm_head.{g}.weight"),
                vec![pv, ph],
                0.0,
                0.1,
            ));
        }
        out
    }

    /// A speech checkpoint of configuration `root` written to `dir` in BF16: its
    /// `config.json` and its `model.safetensors`.
    pub(crate) fn write_speech_toy(dir: &std::path::Path, root: &Json, text_vocab: usize) {
        let mut seed = 7u32;
        let tensors: Vec<(String, Vec<usize>, Vec<u8>)> = speech_tensors(root, text_vocab)
            .into_iter()
            .map(|(name, dims, centre, scale)| {
                let v = values(&mut seed, dims.iter().product(), centre, scale);
                let bytes = v
                    .iter()
                    .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
                    .collect();
                (name, dims, bytes)
            })
            .collect();
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|(name, dims, bytes)| {
                let view = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    dims.clone(),
                    bytes,
                )
                .unwrap();
                (name.clone(), view)
            })
            .collect();
        safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
        std::fs::write(dir.join("config.json"), root.to_string()).unwrap();
    }

    /// A class nested in a checkpoint loads through the ladder with its own geometry.
    /// A layout with no head gives the model none: it gives normed rows and refuses logits.
    #[test]
    fn a_nested_class_loads_and_a_layout_without_a_head_gives_none() {
        let root = speech_config();
        let dir = scratch("nested");
        write_speech_toy(&dir, &root, 16);
        let t = &root["talker_config"];
        let p = &t["code_predictor_config"];
        let width = |c: &Json| c["hidden_size"].as_u64().unwrap() as usize;
        let rows = 3;
        let mut seed = 3u32;
        let mut talker = load_class(&dir, SPEECH_CLASS, Rows::Exactly(rows), 0)
            .unwrap()
            .model;
        let x = values(&mut seed, rows * width(t), 0.0, 1.0);
        let x = Tensor::from_vec(x, (rows, width(t)), &Device::Cpu).unwrap();
        let out = talker.forward_embeddings_prefill(&x, true).unwrap();
        let vocab = t["vocab_size"].as_u64().unwrap() as usize;
        assert_eq!(out.logits.unwrap().dims(), [1, vocab]);
        let mut predictor = load_class(&dir, PREDICTOR_CLASS, Rows::Exactly(rows), 0)
            .unwrap()
            .model;
        let x = values(&mut seed, rows * width(p), 0.0, 1.0);
        let x = Tensor::from_vec(x, (rows, width(p)), &Device::Cpu).unwrap();
        let out = predictor.forward_embeddings_prefill(&x, false).unwrap();
        assert!(out.logits.is_none());
        assert_eq!(out.normed.dims(), [1, width(p)]);
        assert!(predictor.forward_embeddings_prefill(&x, true).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    const GB: u64 = 1 << 30;

    /// A class of 28 layers whose cache at 32768 rows outweighs its weights.
    fn talker_like() -> ClassDemand {
        ClassDemand {
            layers: 28,
            block_bytes: 3 * GB,
            head_bytes: GB / 2,
            row_bytes: 8192,
            held_rows: None,
            runtime_reserve: GB,
        }
    }

    /// The layers of the plan of `demand` on `cards` that run on the host.
    fn host_layers(demand: &ladder::Demand, cards: &[(usize, u64)]) -> usize {
        demand
            .plan(cards)
            .segments
            .iter()
            .filter(|s| {
                !matches!(
                    s.kind,
                    crate::inference::place::layer_executor::DeviceKind::Cuda(_)
                )
            })
            .map(|s| s.num_layers())
            .sum()
    }

    /// A card that holds the class at its ceiling gives the ceiling, every layer on the card.
    #[test]
    fn a_card_that_holds_everything_gives_the_ceiling() {
        let c = talker_like();
        let cards = [(0, 24 * GB)];
        let rows = c.rows_on(Rows::UpTo(32768), &cards, |_| None);
        assert_eq!(rows, 32768);
        assert_eq!(host_layers(&c.at(rows as u64), &cards), 0);
        assert_eq!(ladder::cards_of(&c.at(rows as u64).plan(&cards)), [0]);
    }

    /// On a small card the cache shrinks before a layer leaves the card.
    /// Below the working window the layers move.
    #[test]
    fn a_small_card_shrinks_the_cache_before_a_layer_moves() {
        let c = talker_like();
        let cards = [(0, 7 * GB)];
        assert!(host_layers(&c.at(32768), &cards) > 0);
        let rows = c.rows_on(Rows::UpTo(32768), &cards, |_| None);
        assert!(rows < 32768);
        assert!(rows >= crate::inference::cache::KV_WORKING_WINDOW_TOKENS);
        assert_eq!(host_layers(&c.at(rows as u64), &cards), 0);
        let tiny = [(0, 4 * GB)];
        let floor = c.rows_on(Rows::UpTo(32768), &tiny, |_| None);
        assert_eq!(floor, crate::inference::cache::KV_WORKING_WINDOW_TOKENS);
        assert!(host_layers(&c.at(floor as u64), &tiny) > 0);
    }

    /// With no card the cache keeps its ceiling and the layers run on the host.
    /// An exact count is kept on any card list.
    #[test]
    fn a_host_load_keeps_the_ceiling_and_an_exact_count_is_kept() {
        let c = talker_like();
        let rows = c.rows_on(Rows::UpTo(32768), &[], |_| None);
        assert_eq!(rows, 32768);
        assert_eq!(host_layers(&c.at(rows as u64), &[]), c.layers);
        let ceiling = c.rows_on(Rows::UpTo(2048), &[(0, 24 * GB)], |_| None);
        assert_eq!(ceiling, 2048);
        for cards in [&[][..], &[(0, 7 * GB)][..]] {
            assert_eq!(c.rows_on(Rows::Exactly(16), cards, |_| None), 16);
        }
    }
}
