//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

/// A model's layers as the cluster's planner sees them; see `LlmEngine::segment_inputs`.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentInputs {
    pub layer_bytes: Vec<u64>,
    pub head_bytes: u64,
    pub card_ms: f64,
    pub host_ms: f64,
    /// The bytes of those layers, and of the head with them, that this node's cards carry.
    pub card_bytes: u64,
}

impl LlmEngine {
    /// Cross-encoder relevance score in [0,1] for one (query, document) pair,
    /// using this engine's causal LLM as a Qwen3-Reranker (P(yes) at the final
    /// position). `instruction` empty -> the default retrieval instruction.
    pub async fn rerank_score(
        &self,
        query: String,
        document: String,
        instruction: String,
    ) -> Result<f32, Box<dyn std::error::Error>> {
        use crate::inference::model::reranker::{build_prompt, relevance_from_logits};
        let model_state = self.model_state.clone();
        let prompt = build_prompt(&instruction, &query, &document);
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<f32> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Reranker model not loaded"))?;
            // "yes"/"no" token ids (try the bare token, then a leading-space variant
            // that BPE tokenizers commonly use for a word after whitespace).
            let tok_id = |w: &str, ws: &str| -> AnyResult<u32> {
                state
                    .tokenizer
                    .token_to_id(w)
                    .or_else(|| state.tokenizer.token_to_id(ws))
                    .ok_or_else(|| anyhow!("reranker: token '{w}' not in vocab"))
            };
            let yes_id = tok_id("yes", "Ġyes")?;
            let no_id = tok_id("no", "Ġno")?;
            let encoding = state
                .tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow!("reranker encode: {e}"))?;
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            let seq = ids.len();
            let vocab = state.vocab_size;
            let device = state.device.clone();
            let x = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
            let logits = state.model.forward_all(&x, 0)?;
            let logits_cpu = match logits.device() {
                Device::Cpu => logits,
                _ => logits.to_device(&Device::Cpu)?,
            };
            let flat = logits_cpu
                .squeeze(0)?
                .to_dtype(crate::tensor::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            // last position's row [vocab]
            let base = (seq - 1) * vocab;
            let ly = flat[base + yes_id as usize];
            let ln = flat[base + no_id as usize];
            Ok(relevance_from_logits(ly, ln))
        })
        .await;
        match inner {
            Ok(Ok(s)) => Ok(s),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }

    /// Get model metadata
    pub async fn get_metadata(&self) -> Option<ModelGeometry> {
        let state = self.model_state.lock().await;
        state.as_ref().map(|model| ModelGeometry {
            name: model.name.clone(),
            size: 0,                   // Would read from file
            parameters: 4_000_000_000, // 7B parameters (in millions, fits u32)
            num_layers: model.num_layers,
            hidden_size: model.hidden_size,
            num_heads: model.num_heads,
            vocab_size: model.vocab_size,
            context_length: model.context_length,
        })
    }

    /// The shape a node measures its miss cost against, when the loaded model has streamed experts:
    /// `(dim, inter, gate_up_dtype, down_dtype)`. `None` for a model with nothing to miss, or when
    /// none is loaded. The engine forwards what the model declares; the cluster turns it into a cost.
    pub async fn expert_cost_shape(
        &self,
    ) -> Option<(
        usize,
        usize,
        crate::tensor::quantized::GgmlDType,
        crate::tensor::quantized::GgmlDType,
    )> {
        let state = self.model_state.lock().await;
        state.as_ref().and_then(|s| s.model.expert_cost_shape())
    }

    /// The model-side inputs the cluster's cut planner needs, when a streamed-expert model is
    /// loaded: `(attn_bytes_per_layer, expert_bytes, working_set, activated)`. `None` otherwise.
    pub async fn placement_model_inputs(&self) -> Option<(Vec<u64>, u64, u64, u64)> {
        let state = self.model_state.lock().await;
        state
            .as_ref()
            .and_then(|s| s.model.placement_model_inputs())
    }

    /// What the cluster's planner needs to place this model's layers on nodes: each layer's
    /// bytes with its cache, the output head's bytes, and what a layer costs on this node's
    /// cards and host, zero for a device no layer ran on. `None` when the model cannot be cut
    /// or its file cannot be read. Runs decode steps, so it is called when the node is idle.
    pub async fn segment_inputs(&self) -> Option<SegmentInputs> {
        let file = self.resolve_model_file(&self.config.model_id)?;
        let model_state = self.model_state.clone();
        let share = self.config.layer_share;
        tokio::task::spawn_blocking(move || {
            let content = crate::tensor::quantized::gguf_source::SplitGguf::open(&file)
                .ok()?
                .merged_content();
            let mut guard = model_state.blocking_lock();
            let state = guard.as_mut()?;
            let profile = state.model.segment_profile()?;
            let placed = state.model.device_layer_distribution();
            let mut layer_bytes = block_bytes(&content);
            // A quantised cache starts at the working window and grows on demand, and is
            // placed at that size; an unquantised one is allocated at the whole window.
            let planned_window = match profile.kv_quant {
                KvQuant::Off => profile.window,
                _ => profile
                    .window
                    .min(crate::inference::cache::KV_WORKING_WINDOW_TOKENS),
            };
            let kv_bytes = super::loading::kv_cache_bytes(
                profile.kv_quant,
                std::mem::size_of::<f32>() as u64,
                profile.kv_elems_per_token * planned_window as u64,
            );
            for bytes in &mut layer_bytes {
                *bytes += kv_bytes;
            }
            let tensor_bytes = |n: &str| {
                content
                    .tensor_infos
                    .get(n)
                    .map_or(0, |t| t.size_in_bytes() as u64)
            };
            // An untied model uploads `output.weight`; a tied one its embedding table.
            let head_bytes = match tensor_bytes("output.weight") {
                0 => tensor_bytes("token_embd.weight"),
                b => b,
            };
            // The layers a card carries are the ranges the placement reports for a device that is
            // not the host; the head rides on the cards when any layer does and the loaded layers
            // end the stack.
            let on_card: Vec<usize> = placed
                .iter()
                .filter(|(device, ..)| device != "CPU")
                .flat_map(|&(_, _, first, last)| first as usize..=last as usize)
                .collect();
            let ends_stack = share.is_none_or(|(_, to)| to >= layer_bytes.len());
            let card_bytes = on_card
                .iter()
                .filter_map(|&l| layer_bytes.get(l))
                .sum::<u64>()
                + if on_card.is_empty() || !ends_stack {
                    0
                } else {
                    head_bytes
                };
            Some(SegmentInputs {
                layer_bytes,
                head_bytes,
                card_bytes,
                card_ms: profile.rates.card_ms.unwrap_or(0.0),
                host_ms: profile.rates.host_ms.unwrap_or(0.0),
            })
        })
        .await
        .ok()
        .flatten()
    }

    /// The layers `[0, k)` this node measures the model on without loading the whole: as many as
    /// `card_room` bytes of its cards hold and one more, so that its placement runs layers on the
    /// cards and on the host and both costs come out of one load. `None` when the file is not
    /// found or has no layers.
    pub async fn sample_share(&self, card_room: u64) -> Option<(usize, usize)> {
        let file = self.resolve_model_file(&self.config.model_id)?;
        tokio::task::spawn_blocking(move || {
            let content = crate::tensor::quantized::gguf_source::SplitGguf::open(&file)
                .ok()?
                .merged_content();
            let layers = block_bytes(&content);
            let mut used = 0u64;
            let on_cards = layers
                .iter()
                .take_while(|&&b| {
                    used += b;
                    used <= card_room
                })
                .count();
            (!layers.is_empty()).then(|| (0, (on_cards + 1).min(layers.len())))
        })
        .await
        .ok()
        .flatten()
    }

    /// Load the layers `config.layer_share` names, measure what a layer costs on the cards and on
    /// the host, and unload them: the inputs a cut planner weighs this node by.
    pub async fn measure_share(&self) -> Result<SegmentInputs, String> {
        if self.config.layer_share.is_none() {
            return Err("no layer share is configured".to_string());
        }
        self.load_model_only().await.map_err(|e| e.to_string())?;
        let inputs = self.segment_inputs().await;
        let unloaded = self.unload().await.map_err(|e| e.to_string());
        let inputs = inputs.ok_or_else(|| "this model cannot be cut into layers".to_string())?;
        unloaded?;
        Ok(inputs)
    }

    /// Load the layers `config.layer_share` names and hand the model out of the engine, which
    /// then holds nothing: the model belongs to whoever serves the segment. `window` is the
    /// context the head serves; a segment that cannot hold it is refused, as is a model that
    /// cannot be cut.
    pub async fn load_segment(
        &self,
        window: Option<usize>,
    ) -> Result<GenericHeteroTransformer, String> {
        if self.config.layer_share.is_none() {
            return Err("no layer share is configured".to_string());
        }
        self.load_model_only().await.map_err(|e| e.to_string())?;
        let state = self.model_state.lock().await.take();
        let state = state.ok_or_else(|| "no model loaded".to_string())?;
        let model = state
            .model
            .take_generic()
            .map_err(|_| "this model is not a generic transformer".to_string())?;
        if let Some(why) = model.segment_unsupported_reason() {
            return Err(format!("this model cannot be cut: {why}"));
        }
        if let Some(window) = window.filter(|&w| w > model.kv_capacity()) {
            return Err(format!(
                "the segment holds {} positions and the head serves {window}",
                model.kv_capacity()
            ));
        }
        Ok(model)
    }

    /// Load the layers `[0, cut)` named by `config.layer_share` and make them the head of the
    /// pipeline whose next segment listens at `addr`.
    pub async fn load_head(&self, addr: std::net::SocketAddr, cut: usize) -> Result<(), String> {
        if self.config.layer_share != Some((0, cut)) {
            return Err(format!("the share is not layers [0, {cut})"));
        }
        self.load_model_only().await.map_err(|e| e.to_string())?;
        self.connect_pipeline_tail(addr, cut).await
    }

    /// Make the loaded model the head of a cross-node pipeline, running `[0, cut)` here and handing
    /// `[cut, n)` to the tail at `addr`. Called when the cluster's auto-tune decides a split pays.
    pub async fn connect_pipeline_tail(
        &self,
        addr: std::net::SocketAddr,
        cut: usize,
    ) -> Result<(), String> {
        let mut state = self.model_state.lock().await;
        let s = state
            .as_mut()
            .ok_or_else(|| "no model loaded".to_string())?;
        // A generic transformer keeps layers `[0, cut)` and the rest leaves with the link; any
        // other backend forms the pipeline its own way, or declines.
        // The graphs kept between requests are for the serial decode of this model as it stands.
        #[cfg(feature = "cuda")]
        if !s.kept_graphs.is_empty() {
            s.kept_graphs.clear();
            s.model.note_graphs_kept(None);
            s.model.invalidate_graph_state();
        }
        let taken: BoxedModelBackend = std::mem::replace(&mut s.model, Box::new(TakenBackend));
        match taken.take_generic() {
            Ok(model) => {
                match crate::inference::engine::pipeline_head::PipelineHeadBackend::connect(
                    model, addr, cut,
                ) {
                    Ok(head) => {
                        s.model = Box::new(head);
                        Ok(())
                    }
                    Err((model, e)) => {
                        s.model = Box::new(GenericBackend(*model));
                        Err(e.to_string())
                    }
                }
            }
            Err(other) => {
                s.model = other;
                s.model
                    .connect_pipeline_tail(addr, cut)
                    .map_err(|e| e.to_string())
            }
        }
    }

    /// Get device name
    pub async fn get_device_name(&self) -> String {
        let state = self.model_state.lock().await;
        match state.as_ref() {
            Some(s) => {
                if s.device.is_cuda() {
                    "cuda".to_string()
                } else {
                    "cpu".to_string()
                }
            }
            None => "cpu".to_string(),
        }
    }

    /// Get GPU portion size for multi-device models (for PROCESSOR field calculation)
    /// Returns the portion of model size that's on GPU (CUDA + OpenCL combined)
    pub async fn get_gpu_portion_size(&self) -> u64 {
        let state = self.model_state.lock().await;
        let Some(s) = state.as_ref() else {
            return 0;
        };
        // What a client reads as the split between card and host, so it has to come from
        // where the LAYERS are and not from the base device. A model spread over several
        // cards keeps a CPU base device - that is how the hetero placement addresses the
        // host side - so asking the base device answers "entirely on the host" for a model
        // that is entirely on two cards, and every reader of `size_vram` is told the
        // opposite of the truth.
        let segs = s.model.device_layer_distribution();
        if segs.is_empty() {
            // A variant that publishes no map: the base device is all there is to go on.
            return if s.device.is_cuda() { s.file_size } else { 0 };
        }
        // Counted over the layers that reached a card, NOT summed over segments: a
        // tensor-parallel model publishes every card across the WHOLE layer range, because
        // each one holds a slice of every layer. Adding those segments up would report a
        // model twice its own size. A layer is on the GPU or it is not, so the question is
        // how many distinct layers are, and each is counted once.
        let mut on_gpu = vec![false; s.num_layers];
        for (dtype, _, start, end) in &segs {
            if dtype == "CPU" {
                continue;
            }
            for layer in (*start as usize)..=(*end as usize).min(s.num_layers.saturating_sub(1)) {
                on_gpu[layer] = true;
            }
        }
        let n = on_gpu.iter().filter(|held| **held).count() as u64;
        // Apportioned by layer count, the same way the topology view apportions it - the
        // two must not answer differently about one model.
        s.file_size.saturating_mul(n) / (s.num_layers.max(1)) as u64
    }

    /// Get model size in bytes (actual GGUF file size)
    pub async fn get_model_size(&self) -> u64 {
        let state = self.model_state.lock().await;
        match state.as_ref() {
            Some(model) => model.file_size,
            None => 0,
        }
    }

    /// Get model size in bytes without blocking (for sync contexts like stats monitoring)
    pub fn model_size_nonblocking(&self) -> u64 {
        // Use cached size to avoid lock contention during stats monitoring
        match self.cached_model_size.try_lock() {
            Ok(size) => *size,
            Err(_) => {
                // Fallback: try to get from model_state if cache lock fails
                match self.model_state.try_lock() {
                    Ok(state) => state.as_ref().map(|s| s.file_size).unwrap_or(0),
                    Err(_) => 0,
                }
            }
        }
    }

    /// Get actual GPU/CPU layer split info for the loaded model.
    /// Returns `(total_layers, Vec<LayerDistribution>)` or None if not loaded.
    pub async fn get_layer_distribution(
        &self,
    ) -> Option<(usize, Vec<crate::api::LayerDistribution>)> {
        use crate::api::LayerDistribution;
        let state = self.model_state.lock().await;
        state.as_ref().map(|s| {
            let total = s.num_layers;
            // A streamed placement keeps the weights every token reads, and as many routed
            // experts as fit, resident on the cards, and streams the rest from the host. Its
            // layers have no fixed device, so the honest report is the bytes resident on each
            // card - not a layer map that reads as the whole model living on the CPU. The host
            // remainder is left off: naming it as a device holding layers is what produced the
            // "everything on the CPU" misread in the first place.
            if let Some(cards) = s.model.card_residency() {
                if !cards.is_empty() {
                    let last = (total.saturating_sub(1)) as u32;
                    let distributions: Vec<LayerDistribution> = cards
                        .into_iter()
                        .map(|(ordinal, bytes)| {
                            LayerDistribution::new("CUDA".to_string(), ordinal, 0, last, bytes)
                        })
                        .collect();
                    return (total, distributions);
                }
            }
            // Prefer the model's REAL per-layer device map (the multi-GPU /
            // CPU hetero split). Previously this reported EVERY layer on the
            // primary device (`s.device`), so the GUI topology showed a
            // 2-GPU model entirely on GPU0 - the long-standing bug.
            // Placement comes from the plan, not from the model type: GenericHetero,
            // the tensor-parallel path and DeepSeek each read back the device their
            // layers were loaded onto. A variant with no map at all falls through to the single-entry
            // heuristic below.
            let segs: Vec<(String, usize, u32, u32)> = s.model.device_layer_distribution();
            let distributions: Vec<LayerDistribution> = if segs.is_empty() {
                // Fallback (non-GenericHetero variants): single entry on the
                // primary device, as before.
                let dtype = if s.device.is_cuda() { "CUDA" } else { "CPU" };
                vec![LayerDistribution::new(
                    dtype.to_string(),
                    0,
                    0,
                    (total.saturating_sub(1)) as u32,
                    s.file_size,
                )]
            } else {
                // Apportion the model's on-disk size across segments by layer
                // count (approximate - ignores embed/lm-head weight - but a
                // faithful device MAP, which is what the topology view needs).
                let total_layers = (total.max(1)) as u64;
                segs.into_iter()
                    .map(|(dtype, did, start, end)| {
                        let n = (end - start + 1) as u64;
                        let mem = s.file_size.saturating_mul(n) / total_layers;
                        LayerDistribution::new(dtype, did, start, end, mem)
                    })
                    .collect()
            };
            (total, distributions)
        })
    }
}

/// The bytes of each layer `blk.i` of a file, its tensors summed.
fn block_bytes(content: &crate::tensor::quantized::gguf_file::Content) -> Vec<u64> {
    let mut layer_bytes = Vec::new();
    for (name, info) in &content.tensor_infos {
        let Some(idx) = name
            .strip_prefix("blk.")
            .and_then(|r| r.split_once('.'))
            .and_then(|(i, _)| i.parse::<usize>().ok())
        else {
            continue;
        };
        if layer_bytes.len() <= idx {
            layer_bytes.resize(idx + 1, 0);
        }
        layer_bytes[idx] += info.size_in_bytes() as u64;
    }
    layer_bytes
}
