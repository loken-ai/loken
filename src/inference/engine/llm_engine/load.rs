//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// If `arch` is phi2-class AND a projector blob was located for this
/// model, load the CLIP vision tower and return a `GenericHeteroVision`
/// variant. Otherwise (text-only model, no projector, or non-phi2 arch)
/// return the plain `GenericHetero(text)`. A CLIP load failure is
/// non-fatal: it falls back to text-only so a corrupt projector doesn't
/// take down the entire LM load.
/// Last-resort CPU placement for a generic-arch GGUF. Re-parses the
/// mmap (the prior `Content` was consumed by the failed GPU attempt),
/// builds an all-CPU plan, and forces kv_quant off (Q8/Q4 KV require
/// CUDA). Used by the load-time OOM-recovery ladder when neither the
/// optimistic single-GPU placement nor a forced CUDA split fits.
/// Where a generic transformer's weights come from. The hetero route plans the placement
/// and recovers from out-of-memory the same way for every source; only reading a layer
/// differs.
pub(super) enum GenericWeights<'a> {
    Gguf {
        mmap_bytes: &'a [u8],
        mmap: &'a std::sync::Arc<memmap2::Mmap>,
    },
    /// A split GGUF set: one model across several files, each listing only the tensors it
    /// carries, read as one source.
    GgufSplit {
        split: &'a crate::tensor::quantized::gguf_source::SplitGguf,
    },
    Awq {
        dir: &'a str,
        shards: &'a [String],
    },
}

impl GenericWeights<'_> {
    /// The model built on `plan`, its embedding table on the first layer's card when
    /// `stem_on_card` says the plan holds room for it there.
    pub(super) fn build(
        &self,
        cuda_devices: &std::collections::HashMap<usize, Device>,
        plan: &crate::inference::place::layer_executor::HeteroPlan,
        kv_quant: KvQuant,
        max_kv_seq_len: Option<usize>,
        stem_on_card: bool,
        draft_resident: bool,
    ) -> crate::tensor::Result<GenericHeteroTransformer> {
        match self {
            GenericWeights::Gguf { mmap_bytes, mmap } => {
                let content = gguf_file::Content::read_mapped(
                    &mut Cursor::new(*mmap_bytes),
                    (*mmap).clone(),
                )?;
                GenericHeteroTransformer::from_gguf_placed(
                    content,
                    mmap_bytes,
                    cuda_devices,
                    plan,
                    kv_quant,
                    max_kv_seq_len,
                    stem_on_card,
                    draft_resident,
                )
            }
            GenericWeights::GgufSplit { split } => GenericHeteroTransformer::from_split_gguf(
                split,
                cuda_devices,
                plan,
                kv_quant,
                max_kv_seq_len,
                stem_on_card,
                draft_resident,
            ),
            GenericWeights::Awq { dir, shards } => {
                use crate::inference::load::awq_loader as awql;
                let quantized = awql::parse_config_json(dir)
                    .map_err(|e| crate::tensor::Error::msg(e.to_string()))?
                    .quantized;
                if quantized {
                    return GenericHeteroTransformer::from_awq_safetensors(
                        dir,
                        shards,
                        cuda_devices,
                        plan,
                        kv_quant,
                        max_kv_seq_len,
                    );
                }
                // A dense checkpoint: the header is read as the quantised route reads it, the
                // tensors are mapped under the names the loader asks for.
                let cfg = awql::parse_config_json(dir)
                    .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
                let (header, _) =
                    awq_checkpoint(dir).map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
                let ck = crate::inference::generic_transformer::DenseCheckpoint::open(
                    &header, shards, &cfg,
                )?;
                GenericHeteroTransformer::from_dense_checkpoint(
                    &ck,
                    cuda_devices,
                    plan,
                    kv_quant,
                    max_kv_seq_len,
                    stem_on_card,
                    draft_resident,
                )
            }
        }
    }

    /// Bytes the embedding table takes on a card: zero, since a table is read row by row from
    /// the file's own blocks on the host and no source places it on a card.
    pub(super) fn stem_bytes(&self, _content: &gguf_file::Content) -> u64 {
        0
    }

    /// Each block's `(dense bytes, routed expert bytes)`, the experts being the stacked
    /// `_exps` tensors. `None` when the source cannot say or holds no routed experts.
    pub(super) fn layer_dense_and_expert_bytes(
        &self,
        content: &gguf_file::Content,
        layers: usize,
    ) -> Option<Vec<(u64, u64)>> {
        match self {
            GenericWeights::Gguf { .. } | GenericWeights::GgufSplit { .. } => {
                let mut out = vec![(0u64, 0u64); layers];
                for (name, t) in &content.tensor_infos {
                    let Some(rest) = name.strip_prefix("blk.") else {
                        continue;
                    };
                    let Some((idx, tail)) = rest.split_once('.') else {
                        continue;
                    };
                    let Some(slot) = idx.parse::<usize>().ok().and_then(|i| out.get_mut(i)) else {
                        continue;
                    };
                    let bytes = t.size_in_bytes() as u64;
                    if is_routed_expert_tensor(tail) {
                        slot.1 += bytes;
                    } else {
                        slot.0 += bytes;
                    }
                }
                out.iter().any(|&(_, e)| e > 0).then_some(out)
            }
            GenericWeights::Awq { .. } => None,
        }
    }

    /// Bytes of the repeating blocks and of the output head, as the cards will hold them.
    /// `None` when the source cannot say, and the file size stands in.
    ///
    /// With `share`, the layers `[from, to)` of a stack of `layers`: the blocks of those layers
    /// alone, and the head only when the share ends the stack.
    pub(super) fn block_and_head_bytes(
        &self,
        content: &gguf_file::Content,
        share: Option<(usize, usize)>,
        layers: usize,
    ) -> Option<(u64, u64)> {
        let in_share = |name: &str| -> bool {
            let Some((from, to)) = share else {
                return true;
            };
            name.strip_prefix("blk.")
                .and_then(|rest| rest.split_once('.'))
                .and_then(|(idx, _)| idx.parse::<usize>().ok())
                .is_some_and(|i| (from..to).contains(&i))
        };
        let holds_head = share.is_none_or(|(_, to)| to == layers);
        match self {
            GenericWeights::Gguf { .. } | GenericWeights::GgufSplit { .. } => {
                let tensor_bytes = |name: &str| {
                    content
                        .tensor_infos
                        .get(name)
                        .map_or(0, |t| t.size_in_bytes() as u64)
                };
                let block_bytes: u64 = content
                    .tensor_infos
                    .iter()
                    .filter(|(n, _)| n.starts_with("blk.") && in_share(n))
                    .map(|(_, t)| t.size_in_bytes() as u64)
                    .sum();
                // An untied model uploads `output.weight`; a tied one uploads its
                // embedding table as the projection.
                let head_bytes = match tensor_bytes("output.weight") {
                    _ if !holds_head => 0,
                    0 => tensor_bytes("token_embd.weight"),
                    b => b,
                };
                (block_bytes > 0).then_some((block_bytes, head_bytes))
            }
            GenericWeights::Awq { shards, .. } => {
                let st =
                    unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(shards) }.ok()?;
                let block_bytes: u64 = st
                    .byte_sizes()
                    .iter()
                    .filter(|(n, _)| n.starts_with("model.layers."))
                    .map(|(_, b)| *b as u64)
                    .sum();
                // The head is uploaded as dense F16.
                let head = if st.get("lm_head.weight").is_ok() {
                    "lm_head.weight"
                } else {
                    "model.embed_tokens.weight"
                };
                let head_bytes = st
                    .shape_of(head)
                    .map_or(0, |d| d.iter().product::<usize>() as u64 * 2);
                Some((block_bytes, head_bytes))
            }
        }
    }
}

/// An AWQ checkpoint described as the header the loading route reads: its config, its
/// tokenizer, its end-of-sequence ids and its declared context, beside the shards that
/// carry the layers.
pub(super) fn awq_checkpoint(dir: &str) -> AnyResult<(gguf_file::Content, Vec<String>)> {
    use crate::inference::load::awq_loader as awql;
    let cfg = awql::parse_config_json(dir)?;
    let mut shards: Vec<String> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(anyhow!("AWQ: no *.safetensors shards in {dir}"));
    }
    let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(&shards)? };
    let present = awql::AwqTensorPresence {
        has_q_bias: st.get("model.layers.0.self_attn.q_proj.bias").is_ok(),
        has_qk_norm: st.get("model.layers.0.self_attn.q_norm.weight").is_ok(),
    };
    let mut content = awql::synth_gguf_content(&cfg, present);
    let arch = cfg.gguf_arch.clone();
    let meta = &mut content.metadata;
    // The checkpoint's tokenizer.json, else its tekken.json read into the same form.
    let tokenizer_json = std::path::Path::new(dir).join("tokenizer.json");
    let tekken_json = std::path::Path::new(dir).join(super::tekken::TEKKEN_FILE);
    let (tokenizer, tekken) = if tokenizer_json.exists() || !tekken_json.exists() {
        let json = std::fs::read_to_string(&tokenizer_json)
            .map_err(|e| anyhow!("AWQ tokenizer {}: {e}", tokenizer_json.display()))?;
        (json, None)
    } else {
        let tekken = super::tekken::Tekken::open(&tekken_json)?;
        let json = tekken
            .build()?
            .to_string(false)
            .map_err(|e| anyhow!("AWQ tokenizer {}: {e}", tekken_json.display()))?;
        (json, Some(tekken))
    };
    meta.insert(
        super::gguf::EMBEDDED_TOKENIZER_KEY.to_string(),
        gguf_file::Value::String(tokenizer),
    );
    // With no end-of-sequence id declared anywhere, the sampler still honours the
    // tokenizer's own stops.
    let (eos, extra) = awq_eos_tokens(dir)
        .or_else(|| Some((tekken.as_ref()?.eos()?, Vec::new())))
        .unwrap_or((0, Vec::new()));
    meta.insert(
        "tokenizer.ggml.eos_token_id".into(),
        gguf_file::Value::U32(eos),
    );
    let bos = awq_token_id(dir, "bos_token_id").or_else(|| tekken.as_ref()?.bos());
    if let Some(bos) = bos {
        meta.insert(
            "tokenizer.ggml.bos_token_id".into(),
            gguf_file::Value::U32(bos),
        );
    }
    if !extra.is_empty() {
        meta.insert(
            "tokenizer.ggml.eos_token_ids".into(),
            gguf_file::Value::Array(extra.into_iter().map(gguf_file::Value::U32).collect()),
        );
    }
    meta.insert(
        format!("{arch}.context_length"),
        gguf_file::Value::U32(declared_context(std::path::Path::new(dir)) as u32),
    );
    meta.insert(
        format!("{arch}.vocab_size"),
        gguf_file::Value::U32(cfg.vocab_size as u32),
    );
    Ok((content, shards))
}

pub(super) fn resolve_awq_dir(candidate: &str) -> Option<String> {
    use crate::inference::load::awq_loader::{find_awq_model_dir, find_dense_model_dir};
    let is_checkpoint = |dir: &str| find_awq_model_dir(dir).or_else(|| find_dense_model_dir(dir));
    if candidate.is_empty() {
        return None;
    }
    if is_checkpoint(candidate).is_some() {
        return Some(candidate.to_string());
    }
    let snaps = format!("{candidate}/snapshots");
    if let Ok(rd) = std::fs::read_dir(&snaps) {
        // Pick the first snapshot whose config.json is AWQ (HF caches usually
        // hold one snapshot per revision).
        let mut dirs: Vec<String> = rd
            .flatten()
            .map(|e| e.path().to_string_lossy().to_string())
            .collect();
        dirs.sort();
        for p in dirs {
            if is_checkpoint(&p).is_some() {
                return Some(p);
            }
        }
    }
    None
}

/// Resolve an HF repo id (`org/name[:tag]`) to an AWQ hub directory by mapping
/// it to the standard HF cache layout `models--org--name/snapshots/<hash>` under
/// each candidate root. Lets a request use the natural model name
/// `Quickpanda/deepcoder-14b-preview-awq` (the validator allows single slashes).
pub(super) fn resolve_awq_repo(model_id: &str, roots: &[PathBuf]) -> Option<String> {
    let base = model_id.split(':').next().unwrap_or(model_id);
    let (org, name) = base.split_once('/')?;
    if org.is_empty() || name.is_empty() || name.contains('/') {
        return None;
    }
    let hub = format!("models--{org}--{name}");
    for root in roots {
        let cand = root.join(&hub);
        if let Some(d) = resolve_awq_dir(&cand.to_string_lossy()) {
            return Some(d);
        }
    }
    None
}

/// The released DeepSeek V4.1 checkpoint - safetensors shards, `config.json`,
/// `inference/config.json`, `tokenizer.json` - loaded on the CPU through the deepseek_v41
/// source: the always-resident path dequantised to f32, the routed experts read in place, the
/// engram layers on when the tokenizer lets their hash layout be derived.
pub(super) fn load_deepseek_v41_checkpoint(
    dir: &std::path::Path,
    model_id: &str,
    user_context_length: usize,
    state_cache: Option<std::sync::Arc<crate::inference::cache::state_disk::StateStore>>,
) -> AnyResult<LoadedModelState> {
    use crate::inference::model::deepseek_v41::engram::EngramConfig;
    use crate::inference::model::deepseek_v41::model::DeepseekV41Model;
    use crate::inference::model::deepseek_v41::safetensors_source::SafeTensorsSource;
    use crate::inference::model::deepseek_v41::{token_map, DeepseekV41Config};

    let t0 = std::time::Instant::now();
    let source = SafeTensorsSource::open(dir).map_err(|e| anyhow!("deepseek_v41: {e}"))?;
    let tokenizer_path = dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("{}: {e}", tokenizer_path.display()))?;
    let engram = match token_map::build_from_file(&tokenizer_path) {
        Ok((map, _)) => EngramConfig::derive(&source.inference_config, map),
        Err(e) => {
            warn!("deepseek_v41: no engram, the token map could not be built: {e}");
            None
        }
    };
    if engram.is_none() {
        warn!("deepseek_v41: running without the engram layers");
    }
    let cfg = DeepseekV41Config::from_reference_config(&source.inference_config, engram)
        .map_err(|e| anyhow!("deepseek_v41: {e}"))?;
    let model = DeepseekV41Model::load(&source, &cfg)
        .map_err(|e| anyhow!("deepseek_v41 load failed: {e}"))?;
    // The pinned expert tier takes what memory is left once the page cache can hold the
    // always-read path and one token's worth of routed experts beside it.
    let slots = model
        .size_expert_cache_to_memory(&cfg)
        .map_err(|e| anyhow!("deepseek_v41: {e}"))?;
    let expert_bytes = model
        .expert_bytes()
        .map_err(|e| anyhow!("deepseek_v41: {e}"))? as u64;
    let file_size: u64 = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    let eos_token_id = super::gguf::sidecar_token_id(&dir.join("config.json"), "eos_token")
        .ok_or_else(|| anyhow!("deepseek_v41: no eos_token in tokenizer_config.json"))?;
    let context_length = if user_context_length > 0 {
        user_context_length
    } else {
        declared_context(dir)
    };
    info!(
        "✅ deepseek_v41 loaded on CPU from {} in {:.1}s ({} layers, {} routed + {} shared experts, engram {}, {} experts pinned per layer of {} MB)",
        dir.display(),
        t0.elapsed().as_secs_f32(),
        cfg.n_layers,
        cfg.n_routed_experts,
        cfg.n_shared_experts,
        if cfg.engram.is_some() { "on" } else { "off" },
        slots,
        expert_bytes / 1_000_000,
    );
    let mut backend = crate::inference::engine::model_backend::DeepseekV41Backend::new(
        model,
        cfg.moe_inter_dim,
        context_length,
    );
    // The states of a prompt are stored under the shards and the configuration they were
    // computed with.
    if let Some(store) = state_cache {
        let mut files = crate::inference::cache::state_disk::FilesDigest::new();
        let mut shards: Vec<std::path::PathBuf> = std::fs::read_dir(dir)?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .is_some_and(|x| x == "safetensors" || x == "json")
            })
            .collect();
        shards.sort();
        match shards.iter().try_for_each(|p| files.path(p)) {
            Ok(()) => {
                backend = backend.with_state_cache(
                    store,
                    crate::inference::model::deepseek_v41::snapshot::identity(&[
                        &files.finish(),
                        &(cfg.n_layers as u64).to_le_bytes(),
                    ]),
                    cfg.state_bytes_per_token(),
                );
            }
            Err(e) => warn!("state cache off: {e}"),
        }
    }
    let backend: BoxedModelBackend = Box::new(backend);
    Ok(LoadedModelState {
        name: model_id.to_string(),
        num_layers: cfg.n_layers,
        hidden_size: cfg.d_model,
        num_heads: cfg.n_head,
        vocab_size: cfg.vocab_size,
        context_length,
        eos_token_id,
        #[cfg(feature = "cuda")]
        moondream_graph: None,
        moondream_decode_count: 0,
        #[cfg(feature = "cuda")]
        kept_graphs: Default::default(),
        eos_token_ids_extra: Vec::new(),
        file_size,
        model: backend,
        tokenizer: std::sync::Arc::new(tokenizer),
        device: Device::Cpu,
        image_embeds: None,
        audio: None,
        qwen35_image: None,
        image_embed_cache: None,
        grammar_factory: std::sync::OnceLock::new(),
        generation_defaults: Default::default(),
    })
}

/// The context a checkpoint directory declares (`max_position_embeddings` of its
/// `config.json`), else the working window a cache is built for.
pub(super) fn declared_context(dir: &std::path::Path) -> usize {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v.get("max_position_embeddings").and_then(|x| x.as_u64()))
        .map_or(crate::inference::cache::KV_WORKING_WINDOW_TOKENS, |c| {
            c as usize
        })
}

/// An AWQ checkpoint's generation_config.json, else its config.json.
fn awq_generation_config(dir: &str) -> Option<serde_json::Value> {
    let read = |f: &str| {
        std::fs::read_to_string(format!("{dir}/{f}"))
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
    };
    read("generation_config.json").or_else(|| read("config.json"))
}

/// A scalar token id an AWQ checkpoint's generation config declares under `key`.
fn awq_token_id(dir: &str, key: &str) -> Option<u32> {
    awq_generation_config(dir)?
        .get(key)?
        .as_u64()
        .map(|i| i as u32)
}

/// Read the EOS token id(s) for an AWQ checkpoint from generation_config.json
/// (falling back to config.json). `eos_token_id` may be a scalar or an array.
/// `None` when neither file declares one.
pub(super) fn awq_eos_tokens(dir: &str) -> Option<(u32, Vec<u32>)> {
    let mut ids: Vec<u32> = Vec::new();
    if let Some(v) = awq_generation_config(dir) {
        match v.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                if let Some(i) = n.as_u64() {
                    ids.push(i as u32);
                }
            }
            Some(serde_json::Value::Array(a)) => {
                for x in a {
                    if let Some(i) = x.as_u64() {
                        ids.push(i as u32);
                    }
                }
            }
            _ => {}
        }
    }
    let (&first, rest) = ids.split_first()?;
    Some((first, rest.to_vec()))
}

pub(super) fn try_wrap_with_vision(
    text: GenericHeteroTransformer,
    projector_blob: Option<&std::path::Path>,
    keep: u64,
) -> AnyResult<BoxedModelBackend> {
    let Some(blob) = projector_blob else {
        return Ok(Box::new(GenericBackend(text)));
    };
    // The projector's clip.projector_type picks the tower: "pixtral" loads the
    // Pixtral ViT, "mlp" the CLIP tower (which checks the file's vision shapes
    // and leaves the model text-only when they differ). Any other type, or a
    // projector without the key, leaves the model text-only.
    // A directory is a checkpoint carrying the tower; a file is an `mmproj`.
    #[cfg(feature = "audio")]
    if blob.is_dir() {
        return Ok(attach_audio_tower(text, blob, keep));
    }
    let projector_type = sniff_projector_type(blob).unwrap_or_default();
    #[cfg(feature = "audio")]
    if crate::inference::model_facts::audio_facts(&projector_type).is_some() {
        return Ok(attach_audio_tower(text, blob, keep));
    }
    if !matches!(projector_type.as_str(), "pixtral" | "mlp") {
        info!(
            "📷 projector {} has type {:?}, which no tower here loads; text only",
            blob.display(),
            projector_type
        );
        return Ok(Box::new(GenericBackend(text)));
    }
    // The tower runs once per image, so it is offered the cards the text model holds no
    // layer on first, then the text model's cards with `keep` left free for its work and
    // cache. It holds its weights and the staging of loading them.
    let tower = if projector_type == "pixtral" {
        "Pixtral"
    } else {
        "CLIP"
    };
    let bytes = std::fs::metadata(blob).map(|m| m.len()).unwrap_or(0);
    let need = bytes.saturating_add(crate::inference::place::runtime_demand::load_runtime_floor(
        bytes,
    ));
    let hot: Vec<usize> = text.layer_cuda_ordinals().into_iter().collect();
    let cards = crate::inference::place::ladder::aside(
        &crate::inference::place::ladder::cards(0),
        &hot,
        keep,
    );
    let t0 = std::time::Instant::now();
    let placed = crate::inference::place::ladder::load_whole_on(
        &format!("{tower} vision tower"),
        need,
        need,
        &cards,
        None,
        |on| {
            // The towers take a plain card handle, not the placement's stream-bound one.
            let device = match on.location() {
                crate::tensor::DeviceLocation::Cuda { gpu_id } => {
                    crate::tensor::Device::new_cuda(gpu_id)?
                }
                _ => crate::tensor::Device::Cpu,
            };
            info!(
                "📷 Loading {tower} vision tower from {} onto {:?}",
                blob.display(),
                device.location()
            );
            if projector_type == "pixtral" {
                crate::inference::model::pixtral::PixtralVision::load_mmproj(
                    &blob.to_string_lossy(),
                    &device,
                )
                .map(VisionTower::Pixtral)
                .map_err(|e| anyhow!("{e}"))
            } else {
                crate::inference::model::moondream::vision::MoondreamVisionEncoder::from_clip_gguf(
                    blob, &device,
                )
                .map(VisionTower::Clip)
                .map_err(|e| anyhow!("{e}"))
            }
        },
    );
    match placed {
        Ok((vision, _)) => {
            info!("✅ Vision tower loaded in {} ms", t0.elapsed().as_millis());
            Ok(Box::new(GenericVisionBackend { text, vision }))
        }
        Err(e) => {
            warn!("⚠️  {tower} vision load failed ({e}); model will work for text only");
            Ok(Box::new(GenericBackend(text)))
        }
    }
}

/// The speech tower of an `mmproj` GGUF, loaded beside the text model it feeds. The tower
/// runs once per clip, so it is offered the cards the text model holds no layer on first,
/// then the text model's cards with `keep` left free. A tower that does not load leaves the
/// model text only.
#[cfg(feature = "audio")]
fn attach_audio_tower(
    mut text: GenericHeteroTransformer,
    blob: &std::path::Path,
    keep: u64,
) -> BoxedModelBackend {
    use crate::inference::model::audio_in::{resident_bytes, AudioTower};
    let bytes = resident_bytes(blob);
    let need = bytes.saturating_add(crate::inference::place::runtime_demand::load_runtime_floor(
        bytes,
    ));
    let hot: Vec<usize> = text.layer_cuda_ordinals().into_iter().collect();
    let cards = crate::inference::place::ladder::aside(
        &crate::inference::place::ladder::cards(0),
        &hot,
        keep,
    );
    let t0 = std::time::Instant::now();
    let placed = crate::inference::place::ladder::load_whole_on(
        "audio tower",
        need,
        need,
        &cards,
        None,
        |on| {
            let device = match on.location() {
                crate::tensor::DeviceLocation::Cuda { gpu_id } => {
                    crate::tensor::Device::new_cuda(gpu_id)?
                }
                _ => crate::tensor::Device::Cpu,
            };
            info!(
                "Loading the audio tower from {} onto {:?}",
                blob.display(),
                device.location()
            );
            if blob.is_dir() {
                AudioTower::from_checkpoint(blob, &device)
            } else {
                AudioTower::from_mmproj(blob, &device)
            }
            .map_err(|e| anyhow!("{e}"))
        },
    );
    match placed
        .map(|(tower, _)| tower)
        .and_then(|tower| text.set_audio_tower(tower).map_err(|e| anyhow!("{e}")))
    {
        Ok(()) => info!("Audio tower loaded in {} ms", t0.elapsed().as_millis()),
        Err(e) => warn!("audio tower not loaded ({e}); model will work for text only"),
    }
    Box::new(GenericBackend(text))
}

/// Read `clip.projector_type` from an mmproj GGUF (cheap: metadata only).
pub(super) fn sniff_projector_type(path: &std::path::Path) -> Option<String> {
    let content = crate::tensor::quantized::gguf_file::open_header(path).ok()?;
    content
        .metadata
        .iter()
        .find(|(k, _)| *k == "clip.projector_type")
        .and_then(|(_, v)| v.to_string().ok().cloned())
}

/// Try to move a loaded `cb_eligible` GPU `GenericHetero` model into a
/// `ContinuousServer` worker for batched concurrent serving. The card's room is divided by
/// `place::paged` from the model's shape. Returns the model back unchanged on any failure
/// so the caller keeps the serial path.
#[cfg(feature = "cuda")]
pub(super) fn try_spawn_continuous(
    model: GenericHeteroTransformer,
    eos: Vec<u32>,
    context_length: usize,
) -> std::result::Result<
    crate::inference::serve::continuous_serve::ContinuousServer,
    GenericHeteroTransformer,
> {
    use crate::inference::place::{paged, runtime_demand};
    use crate::inference::serve::continuous_serve::ContinuousServer;
    let (n_kv, hd, n_layers, _vocab) = model.paged_geometry();
    let block_size = crate::inference::serve::continuous_serve::CB_BLOCK_SIZE;
    // K and V, F16, every layer.
    let block_bytes = (2 * n_kv * hd * 2 * n_layers * block_size) as u64;
    let hidden = model.hidden_size();
    let ffn = model.widest_ffn();
    let shape = paged::PagedShape {
        block_bytes,
        block_size,
        window: context_length.min(crate::inference::cache::KV_WORKING_WINDOW_TOKENS),
        lane_step_bytes: paged::lane_step_bytes(n_layers, hidden, ffn),
        work_reserve: runtime_demand::llm_work_reserve(hidden.max(ffn), PREFILL_CHUNK_TOKENS),
    };
    let Some(plan) = paged::plan(&model.compute_device(), &shape) else {
        warn!(
            "CB serve: the card holds fewer than two lanes of {} tokens - staying serial",
            shape.window
        );
        return Err(model);
    };
    let (num_blocks, max_running) = (plan.blocks, plan.lanes);
    info!(
        "CB serve: ContinuousServer - {num_blocks} blocks x {block_size} = {} KV slots, \
           max_running {max_running}, pool {} MB",
        num_blocks * block_size,
        (num_blocks as u64 * block_bytes) >> 20
    );
    let arc = std::sync::Arc::new(model);
    match ContinuousServer::spawn(
        std::sync::Arc::clone(&arc),
        eos,
        num_blocks,
        block_size,
        max_running,
        1 << 20,
    ) {
        Ok(s) => Ok(s),
        Err(e) => {
            warn!("CB serve: spawn failed ({e}) - staying serial");
            // spawn drops its model ref on every error path before the worker
            // thread takes ownership, so the original Arc is now unique.
            Err(std::sync::Arc::try_unwrap(arc)
                .unwrap_or_else(|_| unreachable!("spawn failed yet worker holds the model")))
        }
    }
}

/// Moves the served model into its batch worker: DeepSeek V4.1 into its own, a
/// `cb_eligible` GPU `GenericHetero` into a paged `ContinuousServer` (batched decode, with
/// automatic prefix caching). Returns whether the model now serves batched; a model that
/// cannot batch stays as it was.
pub(super) fn promote_to_batched(state: &mut LoadedModelState) -> bool {
    #[cfg(feature = "cuda")]
    {
        if state.model.batched_server().is_some() {
            return true;
        }
        let (eos, ctx) = (state.eos_token_id, state.context_length);
        // The concurrent-decode batch-width cap. The V4.1 bench saturates aggregate throughput
        // by a batch of a few, so a handful is the useful width; a larger prompt set simply
        // queues behind it.
        const V41_MAX_BATCH: usize = 8;
        // The graphs kept between requests are for the serial decode of this model as it stands.
        #[cfg(feature = "cuda")]
        if !state.kept_graphs.is_empty() {
            state.kept_graphs.clear();
            state.model.note_graphs_kept(None);
            state.model.invalidate_graph_state();
        }
        let taken: BoxedModelBackend = std::mem::replace(&mut state.model, Box::new(TakenBackend));
        // DeepSeek V4.1: move a card-placed model into its batch worker. A host-only load has
        // no lanes to batch and hands itself back.
        let taken = match taken.into_batched(eos, V41_MAX_BATCH) {
            Ok(server) => {
                state.model = Box::new(BatchedBackend(server));
                return true;
            }
            Err(back) => back,
        };
        // Generic dense (Qwen2/3): the paged continuous server, when the GPU is eligible.
        if !taken.cb_eligible_gpu() {
            state.model = taken;
            return false;
        }
        let m = match taken.take_generic() {
            Ok(m) => m,
            Err(other) => {
                state.model = other;
                return false;
            }
        };
        let mut ends = state.eos_token_ids_extra.clone();
        ends.push(eos);
        let (model, batched): (BoxedModelBackend, bool) = match try_spawn_continuous(m, ends, ctx) {
            Ok(s) => (Box::new(ContinuousBackend(std::sync::Arc::new(s))), true),
            Err(m) => (Box::new(GenericBackend(m)), false),
        };
        state.model = model;
        batched
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = state;
        false
    }
}

/// Takes a model served by a paged `ContinuousServer` back to serial serving, once no
/// stream still holds the server. Returns whether the model now serves serially.
pub(super) fn demote_to_serial(state: &mut LoadedModelState) -> bool {
    #[cfg(feature = "cuda")]
    {
        if state.model.batched_server().is_none() {
            return true;
        }
        // The graphs kept between requests are for the serial decode of this model as it stands.
        #[cfg(feature = "cuda")]
        if !state.kept_graphs.is_empty() {
            state.kept_graphs.clear();
            state.model.note_graphs_kept(None);
            state.model.invalidate_graph_state();
        }
        let taken: BoxedModelBackend = std::mem::replace(&mut state.model, Box::new(TakenBackend));
        match taken.into_serial() {
            Ok(serial) => {
                state.model = serial;
                true
            }
            Err(back) => {
                state.model = back;
                false
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = state;
        true
    }
}

#[cfg(feature = "cuda")]
unsafe impl Send for MoondreamGraphState {}
#[cfg(feature = "cuda")]
unsafe impl Sync for MoondreamGraphState {}

/// Full state when a model is loaded
/// CUDA graph capture state for QuantizedMoondream.
/// Captured once after warmup; replayed each decode token to collapse
/// per-launch overhead from ~720 launches/token down to 1 graph launch.
/// Falls back to plain forward on any replay failure (cleared on
/// failure; recaptured on next eligible warmup).
#[cfg(feature = "cuda")]
pub(super) struct MoondreamGraphState {
    pub(super) graph: crate::tensor::cuda_ext::CapturedGraph,
}

pub(crate) struct LoadedModelState {
    pub(super) name: String,
    pub(super) num_layers: usize,
    pub(super) hidden_size: usize,
    pub(super) num_heads: usize,
    pub(super) vocab_size: usize,
    pub(super) context_length: usize,
    pub(super) eos_token_id: u32,
    /// Lazily captured CUDA graph for QuantizedMoondream decode (Step 5).
    #[cfg(feature = "cuda")]
    pub(super) moondream_graph: Option<MoondreamGraphState>,
    /// Decode tokens seen since model load. Drives moondream graph
    /// capture warmup gating (capture fires once after N warmup tokens).
    pub(super) moondream_decode_count: u32,
    /// The decode's recorded graphs, kept between requests while nothing they read has moved.
    #[cfg(feature = "cuda")]
    pub(super) kept_graphs: super::decode::KeptGraphs<super::decode::DecodeGraphs>,
    /// Additional EOS-equivalent token IDs (e.g. Gemma4's
    /// `tokenizer.ggml.eos_token_ids` = [1, 106, 50] where 106 and 50
    /// are turn-end markers used in the chat template). Single
    /// `eos_token_id` is the primary one; this set captures all that
    /// should also terminate generation.
    pub(super) eos_token_ids_extra: Vec<u32>,
    pub(super) file_size: u64,
    pub(super) model: BoxedModelBackend,
    /// Shared, so a request takes a handle rather than a copy of the vocabulary.
    pub(super) tokenizer: std::sync::Arc<Tokenizer>,
    pub(super) device: Device,
    /// Pre-encoded image embeddings for the current request (vision models only)
    pub(super) image_embeds: Option<Tensor>,
    /// The clip of the current request, encoded (speech models only).
    pub(super) audio: Option<AudioClip>,
    /// qwen35moe (Qwen3-VL) raw preprocessed image for the current request:
    /// `(pixel_values [n_patches,1536], patch grid (gh,gw))`. Unlike moondream
    /// (pre-encoded + prepended), qwen35 keeps the raw patches and runs the ViT
    /// + splice inside `forward_with_image` at prefill. None = no image.
    pub(super) qwen35_image: Option<(Tensor, (usize, usize))>,
    /// One-slot cache for the most-recent vision encode: (image_hash,
    /// projector_output). Same-image requests (bench, chat with shared
    /// image, multi-turn captioning) skip the CLIP forward and reuse the
    /// stored projector output - matches Ollama's prompt-cache semantics
    /// for vision. Single-entry to keep memory bounded; a bigger LRU
    /// could ship later if the use case shows up.
    pub(super) image_embed_cache: Option<(u64, Tensor)>,
    /// Lazily-built llguidance parser factory. Constructed on first
    /// grammar-constrained request from the live tokenizer; cached for reuse
    /// across requests on the same loaded model.
    pub(super) grammar_factory: std::sync::OnceLock<std::sync::Arc<llguidance::ParserFactory>>,
    /// The sampling the model recommends, applied to every knob a request leaves unset.
    pub(super) generation_defaults: super::generation_defaults::GenerationDefaults,
}

/// A request's clip as the model reads it: the embeddings that replace the prompt's
/// placeholder, and a hash of the clip that keeps the prompt cache from matching across clips.
pub(super) struct AudioClip {
    pub embeds: Tensor,
    pub hash: u64,
}

/// Snapshot of one session's cached conversation state. The KV cache itself
/// lives inside `LoadedModelState.model` (a single shared cache) - this
/// struct only records which token sequence the cache currently contains, so
/// the next request can find a common prefix and skip redundant prefill.
#[derive(Clone)]
pub(crate) struct SessionState {
    /// Full token sequence currently represented in the KV cache
    /// (prompt + generated). For vision sessions, this records ONLY the
    /// post-image text tokens (the image embeds occupy positions
    /// [0, image_prefix_len) but aren't in the token stream).
    pub(super) tokens: Vec<u32>,
    /// Model name that owned the cache. If the engine swaps models the
    /// cache is invalidated.
    pub(super) model_name: String,
    /// Hash of the image whose embeddings are in cache positions
    /// [0, image_prefix_len). None for text-only sessions. Set by the
    /// vision prefill path to enable cross-call prefix-KV reuse when the
    /// same image is sent again.
    pub(super) image_hash: Option<u64>,
    /// Number of cache positions occupied by the image prefix
    /// (typically 1 BOS + 729 image embeds = 730 for moondream).
    /// Tokens in `tokens` are at positions [image_prefix_len, ...).
    pub(super) image_prefix_len: usize,
    /// KV rows the cache ACTUALLY holds.
    ///
    /// NOT `tokens.len()`, which is what every reuse decision used to assume. A
    /// sampled token is recorded the moment it is produced but only written to the
    /// cache when it is FORWARDED, so the last one of a generation is always named by
    /// `tokens` and absent from the KV; a rejected speculative draft trims rows the
    /// list still names, widening the gap further. Reusing a prefix longer than the
    /// cache holds builds the attention mask for positions that do not exist, and the
    /// next request dies on the shape - `cannot broadcast [191, 620] to
    /// [1, 16, 191, 617]`, the two lengths in plain sight.
    pub(super) kv_len: usize,
}

/// Longest prefix of `prompt` the resident cache can actually serve.
///
/// Two things must agree and did not: the token list names what was PRODUCED, `kv_len`
/// counts what was WRITTEN. Taking the common prefix of the lists alone yields a start
/// position past the end of the cache, and the mask built for it names columns the
/// attention tensor does not have.
/// How many leading tokens of `prompt_tokens` the resident KV can be reused for.
///
/// ONE implementation, called by both the streaming and non-streaming paths. They each
/// carried a copy of a four-branch version of this and the copies had drifted: the
/// streaming one returned 0 without trimming when there was nothing to reuse, leaving the
/// previous request's KV in place while the new prompt prefilled from position 0.
///
/// The rule, stated once: cut the KV to exactly what is being reused, and always leave at
/// least one token for the forward to run on. Every case follows - no entry gives 0, a
/// full match gives len-1, a partial or extending match gives the shared prefix.
/// The reuse offset the cache can actually honour, having been asked.
///
/// The offset above it comes from a table describing the resident KV. The two can
/// disagree: a trim that failed, a cache reset nothing wrote down, a request that trimmed
/// and never finished. Prefilling at an offset the cache does not hold builds the
/// attention mask for a context that is not there - the forward dies on a shape mismatch,
/// or, when the gap is small enough to broadcast, the model reads rows belonging to
/// another prompt and answers with text that is almost right.
///
/// Short means the cache is a shorter prefix of the same tokens, which is sound to
/// continue from. Long means the trim did not take, so the rows past the offset belong to
/// a prompt this one does not share, and the only safe answer is to start cold. A backend
/// that does not report a length is left alone.
pub(crate) fn honest_reuse_start(
    model: &mut dyn crate::inference::engine::model_backend::ModelBackend,
    want: usize,
) -> usize {
    match reuse_verdict(model.kv_len(), want) {
        Reuse::AsAsked(n) => n,
        Reuse::Shorten(n) => {
            tracing::warn!(
                "prefix reuse offset {want} is past what the cache holds ({n}); \
                 prefilling from there instead"
            );
            n
        }
        Reuse::Cold(held) => {
            tracing::warn!(
                "cache still holds {held} rows after a trim to {want}; starting cold \
                 rather than reading rows this prompt does not share"
            );
            model.reset_kv_from(0);
            0
        }
    }
}

/// What to do about an offset and the cache that is supposed to honour it.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Reuse {
    AsAsked(usize),
    Shorten(usize),
    Cold(usize),
}

pub(crate) fn reuse_verdict(held: Option<usize>, want: usize) -> Reuse {
    match held {
        None => Reuse::AsAsked(want),
        Some(held) if held == want => Reuse::AsAsked(want),
        Some(held) if held < want => Reuse::Shorten(held),
        Some(held) => Reuse::Cold(held),
    }
}

pub(crate) fn kv_reuse_start(
    model: &mut dyn crate::inference::engine::model_backend::ModelBackend,
    sessions: &std::sync::Arc<tokio::sync::Mutex<std::collections::HashMap<String, SessionState>>>,
    model_name: &str,
    prompt_tokens: &[u32],
    disabled: bool,
    disk: Option<&crate::inference::cache::kv_disk::KvDiskStore>,
) -> usize {
    let mut reuse = if disabled || !model.capabilities().trim_kv {
        None
    } else {
        let g = sessions.blocking_lock();
        match g.get(GLOBAL_PROMPT_CACHE_KEY) {
            Some(s) if s.model_name == model_name && s.image_hash.is_none() => {
                Some(reusable_prefix(&s.tokens, prompt_tokens, s.kv_len))
            }
            _ => None,
        }
    };
    // Reuse is only sound on a cold-run chunk boundary. A cold run prefills in fixed
    // chunks from position 0; a warm run re-prefills the tail from `keep`. Off a boundary
    // the tail runs GEMM shapes a cold run never used at those positions, and the
    // reassociated arithmetic flips greedy argmax wherever the top two candidates are
    // close. Rounding down makes the warm tail exactly cold's tail chunks - a fixed
    // remainder was measured insufficient, since any shape difference reassociates.
    let chunk = crate::inference::engine::model_backend::adaptive_prefill_chunk(
        crate::inference::engine::model_backend::PrefillShape::from_widest_ffn(model.widest_ffn()),
    );
    // A snapshot of another conversation that shares more of this prompt than the
    // resident KV does becomes the resident: the resident sequence itself was copied
    // aside when its request ended, so nothing is lost by the swap.
    if !disabled && model.capabilities().trim_kv {
        let cur_keep = reuse
            .map(|c| chunk_aligned_keep(c, prompt_tokens.len(), chunk))
            .unwrap_or(0);
        if let Some((index, snap_common)) = model.best_kv_snapshot(prompt_tokens) {
            if chunk_aligned_keep(snap_common, prompt_tokens.len(), chunk) > cur_keep {
                match model.restore_kv_snapshot(index) {
                    Ok((tokens, kv_len)) => {
                        let mut g = sessions.blocking_lock();
                        g.insert(
                            GLOBAL_PROMPT_CACHE_KEY.to_string(),
                            SessionState {
                                tokens,
                                model_name: model_name.to_string(),
                                image_hash: None,
                                image_prefix_len: 0,
                                kv_len,
                            },
                        );
                        tracing::info!(
                            "kv snapshot restored: {snap_common} tokens shared with the prompt, {kv_len} resident"
                        );
                        reuse = Some(snap_common);
                    }
                    Err(e) => tracing::warn!("kv snapshot restore failed: {e}; resident KV kept"),
                }
            }
        }
    }
    // Then the disk: a sequence written after an earlier request, or an earlier run,
    // that covers more of the prompt than anything in memory. Its blocks come back as
    // a snapshot, which is restored like the others.
    if let (Some(store), false) = (disk, disabled) {
        if let Some((n_layers, _, _)) = model.kv_layout() {
            let cur_keep = reuse
                .map(|c| chunk_aligned_keep(c, prompt_tokens.len(), chunk))
                .unwrap_or(0);
            let layout = kv_layout_id(model);
            if let Some((mi, covered)) = store.best(model_name, layout, prompt_tokens) {
                // A windowed manifest's side blob describes one exact prefix length; reuse
                // it only when the whole prefix matches, never a shorter shared run.
                let windowed_ok =
                    !store.manifest_windowed(mi) || covered == store.manifest_covered(mi);
                if windowed_ok && chunk_aligned_keep(covered, prompt_tokens.len(), chunk) > cur_keep
                {
                    let blocks = covered / store.block_tokens();
                    let loaded = store
                        .load(mi, blocks, n_layers)
                        .map_err(|e| crate::tensor::Error::msg(e.to_string()))
                        .and_then(|rows| {
                            let dt = store.manifest_kv_dtype(mi);
                            model.import_kv_snapshot(
                                prompt_tokens[..covered].to_vec(),
                                covered,
                                rows,
                                dt,
                            )
                        })
                        .and_then(|()| {
                            let (index, _) =
                                model.best_kv_snapshot(prompt_tokens).ok_or_else(|| {
                                    crate::tensor::Error::msg(
                                        "kv disk: imported snapshot not found".to_string(),
                                    )
                                })?;
                            model.restore_kv_snapshot(index)
                        });
                    match loaded {
                        Ok((tokens, kv_len)) => {
                            let mut g = sessions.blocking_lock();
                            g.insert(
                                GLOBAL_PROMPT_CACHE_KEY.to_string(),
                                SessionState {
                                    tokens,
                                    model_name: model_name.to_string(),
                                    image_hash: None,
                                    image_prefix_len: 0,
                                    kv_len,
                                },
                            );
                            tracing::info!(
                                "kv disk: {covered} tokens of the prompt restored from disk"
                            );
                            reuse = Some(covered);
                        }
                        Err(e) => tracing::warn!("kv disk restore failed: {e}; resident KV kept"),
                    }
                }
            }
        }
    }
    // A prompt clamped to the window keeps its first token and its tail; once a
    // conversation outgrows the window the common prefix with the resident sequence
    // is that first token alone, and every turn re-prefills the whole window. The tail
    // is still resident, at positions `discard` further on: shift it down and prefill
    // only what is new. The shifted keys are re-phased, not recomputed, so the result
    // is not the cold run's bit for bit; the option says the caller accepts that, and
    // with it the chunk grid below: a shifted tail is kept whole, since rounding it to
    // a grid that can exceed the window would keep nothing.
    let mut shifted: Option<usize> = None;
    if model.capabilities().shift_kv && reuse.is_some_and(|c| c < chunk) {
        let mut g = sessions.blocking_lock();
        if let Some(s) = g.get_mut(GLOBAL_PROMPT_CACHE_KEY) {
            if let Some((discard, reused)) = shift_reuse_plan(
                &s.tokens,
                prompt_tokens,
                s.kv_len,
                SHIFT_KEEP,
                SHIFT_PROBE,
                SHIFT_PROBE,
            ) {
                match model.shift_kv_tail(SHIFT_KEEP + discard, discard) {
                    Ok(()) => {
                        s.tokens.drain(SHIFT_KEEP..SHIFT_KEEP + discard);
                        s.kv_len = s.kv_len.saturating_sub(discard);
                        tracing::info!(
                            "context shift: dropped {discard} tokens after the first {SHIFT_KEEP}, {reused} reused"
                        );
                        shifted = Some(SHIFT_KEEP + reused);
                    }
                    Err(e) => {
                        tracing::warn!("context shift refused: {e}; cold prefill");
                        model.trim_kv(0);
                        s.tokens.clear();
                        s.kv_len = 0;
                        reuse = None;
                    }
                }
            }
        }
    }
    let keep = match shifted {
        Some(kept) => kept.min(prompt_tokens.len().saturating_sub(1)),
        None => reuse
            .map(|common| chunk_aligned_keep(common, prompt_tokens.len(), chunk))
            .unwrap_or(0),
    };
    if model.capabilities().trim_kv {
        model.trim_kv(keep);
    }
    keep
}

/// What identifies a KV layout on disk: the row geometry. Two builds of one model with
/// the same geometry read each other's blocks; a different quantisation of the weights
/// changes the values, not the layout, and the model name keeps those apart.
pub(crate) fn kv_layout_id(
    model: &dyn crate::inference::engine::model_backend::ModelBackend,
) -> u64 {
    match model.kv_layout() {
        Some((layers, n_kv, hd)) => ((layers as u64) << 40) | ((n_kv as u64) << 20) | hd as u64,
        None => 0,
    }
}

/// Writes the snapshot just taken for `tokens` to the disk tier, block by block.
/// Writes the snapshot that holds `tokens` to the disk tier, off the request path. A
/// thread takes the model lock once, copies out only the blocks not yet on disk, most
/// turns one, and writes with the lock released; a response never waits on the copies or
/// the writes, and requests wait at most that one copy.
pub(crate) fn spawn_persist(
    model_state: std::sync::Arc<tokio::sync::Mutex<Option<LoadedModelState>>>,
    store: std::sync::Arc<crate::inference::cache::kv_disk::KvDiskStore>,
    tokens: Vec<u32>,
) {
    let spawned = std::thread::Builder::new()
        .name("kv-disk-persist".to_string())
        .spawn(move || persist_kv_snapshot(&model_state, &store, &tokens));
    if let Err(e) = spawned {
        tracing::warn!("kv disk persist thread: {e}");
    }
}

/// Finds the snapshot holding `tokens` whole: its index and resident length.
fn snapshot_index_for(
    model: &dyn crate::inference::engine::model_backend::ModelBackend,
    tokens: &[u32],
) -> Option<(usize, usize)> {
    let (index, common) = model.best_kv_snapshot(tokens)?;
    let (_, kv_len) = model.kv_snapshot_ref(index)?;
    if common < kv_len.min(tokens.len()) {
        return None; // the table holds another sequence; nothing of this one to write
    }
    Some((index, kv_len))
}

fn persist_kv_snapshot(
    model_state: &tokio::sync::Mutex<Option<LoadedModelState>>,
    store: &crate::inference::cache::kv_disk::KvDiskStore,
    tokens: &[u32],
) {
    // Under the lock: the snapshot's facts and the rows of the blocks still to write.
    let (model_name, layout, kv_len, kv_dtype, window, staged) = {
        let g = model_state.blocking_lock();
        let Some(state) = g.as_ref() else { return };
        let model = state.model.as_ref();
        let Some((index, kv_len)) = snapshot_index_for(model, tokens) else {
            return;
        };
        let covered = store.covered_len(tokens.len(), kv_len);
        let layout = kv_layout_id(model);
        let window = match model.export_window_rows(index, covered) {
            Ok(w) => w,
            Err(e) => {
                tracing::warn!("kv disk: window rows export failed: {e}; not persisted");
                return;
            }
        };
        let mut staged = std::collections::HashMap::new();
        for (b, from, to) in store.missing_blocks(&state.name, layout, tokens, covered) {
            match model.export_kv_rows(index, from, to) {
                Ok(rows) => {
                    staged.insert(b, rows);
                }
                Err(e) => {
                    tracing::warn!("kv disk: block export failed: {e}; not persisted");
                    return;
                }
            }
        }
        (
            state.name.clone(),
            layout,
            kv_len,
            model.kv_snapshot_dtype(index).unwrap_or(0),
            window,
            staged,
        )
    };
    // Lock released: the writes.
    let mut staged = staged;
    let result = store.persist(
        &model_name,
        layout,
        tokens,
        kv_len,
        kv_dtype,
        &window,
        |from, _to| {
            staged
                .remove(&(from / store.block_tokens()))
                .ok_or_else(|| anyhow::anyhow!("kv disk: block at {from} was not staged"))
        },
    );
    if let Err(e) = result {
        tracing::warn!("kv disk persist failed: {e}");
    }
}

const SHIFT_KEEP: usize = 1;
/// Tokens compared to locate the prompt's tail in the resident sequence before the
/// match is extended; long enough that chat text does not repeat it by chance.
const SHIFT_PROBE: usize = 64;

/// Where a clamped prompt's tail sits in the resident sequence: `(discard, reused)` such
/// that `cached[n_keep + discard..][..reused] == prompt[n_keep..][..reused]`, found by
/// matching a `probe`-token window then extending it. `None` when the heads differ, the
/// window occurs nowhere after the head, or the match is shorter than `min_reuse`.
pub(crate) fn shift_reuse_plan(
    cached: &[u32],
    prompt: &[u32],
    kv_len: usize,
    n_keep: usize,
    probe: usize,
    min_reuse: usize,
) -> Option<(usize, usize)> {
    let cached = &cached[..kv_len.min(cached.len())];
    if cached.len() <= n_keep || prompt.len() <= n_keep || cached[..n_keep] != prompt[..n_keep] {
        return None;
    }
    let head = &prompt[n_keep..];
    let body = &cached[n_keep..];
    let probe = probe.min(head.len());
    if probe == 0 || body.len() < probe + 1 {
        return None;
    }
    for discard in 1..=body.len() - probe {
        if body[discard..discard + probe] == head[..probe] {
            let reused = body[discard..]
                .iter()
                .zip(head)
                .take_while(|(a, b)| a == b)
                .count();
            return (reused >= min_reuse).then_some((discard, reused));
        }
    }
    None
}

/// The largest reusable prefix that sits on a cold-run chunk boundary, always leaving at
/// least the final token to re-prefill so the call produces logits.
pub(crate) fn chunk_aligned_keep(common: usize, prompt_len: usize, chunk: usize) -> usize {
    let capped = common.min(prompt_len.saturating_sub(1));
    capped - capped % chunk.max(1)
}

pub(crate) fn reusable_prefix(cached: &[u32], prompt: &[u32], kv_len: usize) -> usize {
    cached
        .iter()
        .zip(prompt.iter())
        .take_while(|(a, b)| a == b)
        .count()
        .min(kv_len)
}

#[cfg(test)]
mod prompt_cache_reuse_tests {
    use super::shift_reuse_plan;

    #[test]
    fn shift_plan_finds_the_tail_of_a_grown_conversation() {
        // Resident: BOS then 1..=100. The next turn was clamped to BOS + the last 60
        // of the old window + 10 new tokens: 40 tokens fell out after the head.
        let cached: Vec<u32> = std::iter::once(0).chain(1..=100).collect();
        let prompt: Vec<u32> = std::iter::once(0).chain(41..=110).collect();
        assert_eq!(
            shift_reuse_plan(&cached, &prompt, cached.len(), 1, 8, 16),
            Some((40, 60))
        );
    }

    #[test]
    fn shift_plan_refuses_a_different_head_or_an_absent_tail() {
        let cached: Vec<u32> = std::iter::once(0).chain(1..=100).collect();
        let other_head: Vec<u32> = std::iter::once(7).chain(41..=110).collect();
        assert_eq!(
            shift_reuse_plan(&cached, &other_head, cached.len(), 1, 8, 16),
            None
        );
        let elsewhere: Vec<u32> = std::iter::once(0).chain(500..=560).collect();
        assert_eq!(
            shift_reuse_plan(&cached, &elsewhere, cached.len(), 1, 8, 16),
            None
        );
    }

    #[test]
    fn shift_plan_wants_a_match_worth_a_chunk_and_honours_kv_len() {
        let cached: Vec<u32> = std::iter::once(0).chain(1..=100).collect();
        let short: Vec<u32> = std::iter::once(0)
            .chain(91..=100)
            .chain(200..=230)
            .collect();
        assert_eq!(
            shift_reuse_plan(&cached, &short, cached.len(), 1, 8, 16),
            None
        );
        // Only 50 tokens are in the KV: the match cannot extend past them.
        let prompt: Vec<u32> = std::iter::once(0).chain(21..=110).collect();
        assert_eq!(
            shift_reuse_plan(&cached, &prompt, 51, 1, 8, 16),
            Some((20, 30))
        );
    }

    use super::{chunk_aligned_keep, reusable_prefix};

    /// Reuse snaps DOWN to the chunk grid of a cold prefill: the warm tail must be the
    /// cold run's exact tail chunks, or the boundary arithmetic differs and greedy flips.
    #[test]
    fn reuse_snaps_down_to_a_cold_chunk_boundary() {
        assert_eq!(chunk_aligned_keep(1200, 1300, 512), 1024);
        assert_eq!(chunk_aligned_keep(512, 1300, 512), 512);
        assert_eq!(
            chunk_aligned_keep(511, 1300, 512),
            0,
            "below one chunk nothing is safe"
        );
    }

    /// A fully-cached prompt still re-prefills its final chunk: the last token must go
    /// through the forward to produce logits, from the same boundary a cold run used.
    #[test]
    fn a_fully_cached_prompt_reprefills_from_the_last_boundary() {
        assert_eq!(chunk_aligned_keep(1024, 1024, 512), 512);
        assert_eq!(chunk_aligned_keep(513, 513, 512), 512);
        assert_eq!(chunk_aligned_keep(512, 512, 512), 0);
    }

    /// The failure this exists for: a generation leaves its last sampled token in the
    /// list and out of the cache, so a repeat of the same prompt asked to resume from a
    /// position the cache never held.
    #[test]
    fn never_reuses_more_than_the_cache_holds() {
        let cached = [1u32, 2, 3, 4, 5];
        // Four rows written, five tokens named - the classic commit-delayed tail.
        assert_eq!(reusable_prefix(&cached, &[1, 2, 3, 4, 5], 4), 4);
        // A rejected speculative draft can trim several rows at once.
        assert_eq!(reusable_prefix(&cached, &[1, 2, 3, 4, 5], 2), 2);
    }

    #[test]
    fn a_shorter_divergence_still_wins() {
        // Divergence before the cache end is the binding limit, not kv_len.
        assert_eq!(reusable_prefix(&[1, 2, 9], &[1, 2, 3, 4], 3), 2);
    }

    #[test]
    fn an_empty_cache_reuses_nothing() {
        assert_eq!(reusable_prefix(&[1, 2, 3], &[1, 2, 3], 0), 0);
    }
}

#[cfg(test)]
mod reuse_tests {
    use super::{reuse_verdict, Reuse};

    /// A backend that does not report a length is left alone: nothing is known, and
    /// guessing would be worse than the offset the caller already computed.
    #[test]
    fn a_cache_that_says_nothing_is_taken_at_its_word() {
        assert_eq!(reuse_verdict(None, 500), Reuse::AsAsked(500));
    }

    /// Short means the cache is a shorter prefix of the same tokens, which is sound to
    /// continue from. This is the shape that used to kill the forward.
    #[test]
    fn an_offset_past_the_last_row_comes_back_to_it() {
        assert_eq!(reuse_verdict(Some(347), 577), Reuse::Shorten(347));
    }

    /// Long means the trim did not take, so the rows past the offset belong to a prompt
    /// this one does not share and none of them may be read.
    #[test]
    fn a_trim_that_did_not_take_starts_cold() {
        assert_eq!(reuse_verdict(Some(900), 577), Reuse::Cold(900));
    }

    #[test]
    fn a_cache_that_matches_is_used_as_asked() {
        assert_eq!(reuse_verdict(Some(577), 577), Reuse::AsAsked(577));
    }
}

/// Whether a block tensor, named after its `blk.N.` prefix, is a routed expert stack: the
/// weights and per-expert biases the host-expert placement keeps off the card. The router
/// (`ffn_gate_inp`) and the shared expert (`_shexp`) are not.
pub(crate) fn is_routed_expert_tensor(tail: &str) -> bool {
    let stem = tail.split('.').next().unwrap_or(tail);
    stem.ends_with("_exps")
}
