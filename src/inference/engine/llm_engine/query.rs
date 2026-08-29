//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl LlmEngine {
    /// Multi-position forward returning logits at every input position.
    /// Used by the speculative-decode verify step. Logits are returned as
    /// a flat `Vec<f32>` of length `seq * vocab_size`; caller indexes by
    /// `i * vocab_size .. (i+1) * vocab_size`.
    pub async fn target_forward_all(
        &self,
        input: Vec<u32>,
        pos: usize,
    ) -> Result<(Vec<f32>, usize, usize), Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<(Vec<f32>, usize, usize)> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Target model not loaded"))?;
            let device = state.device.clone();
            let vocab_size = state.vocab_size;
            let seq = input.len();
            let x = Tensor::new(input.as_slice(), &device)?.unsqueeze(0)?;
            let logits = state.model.forward_all(&x, pos)?;
            let logits_cpu = match logits.device() {
                Device::Cpu => logits,
                _ => logits.to_device(&Device::Cpu)?,
            };
            // logits_cpu shape: [1, seq, vocab] or [seq, vocab]. Flatten to seq * vocab.
            let flat = logits_cpu
                .squeeze(0)?
                .to_dtype(crate::tensor::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            Ok((flat, seq, vocab_size))
        })
        .await;
        match inner {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }

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

    /// Trim the model's KV cache to a specific length. Public wrapper
    /// for spec decode rollback after partial draft acceptance.
    pub async fn trim_kv(&self, new_len: usize) -> Result<(), Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            let mut guard = model_state.blocking_lock();
            let state = guard.as_mut().ok_or_else(|| anyhow!("Model not loaded"))?;
            if state.model.supports_trim_kv() {
                state.model.trim_kv(new_len);
            }
            Ok(())
        })
        .await;
        match inner {
            Ok(Ok(())) => Ok(()),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }

    /// Reset the draft model's internal KV state. Called at the start
    /// of every speculative decode session so the draft's positional
    /// cache starts at offset 0.
    pub async fn draft_reset(&self) -> Result<(), Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<()> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Draft model not loaded"))?;
            // forward(_, 0) resets per-layer KV in every supported variant.
            // Use a no-op single-token forward to trigger the reset path.
            // Actually cheaper: trim to zero where supported.
            if state.model.supports_trim_kv() {
                state.model.trim_kv(0);
            }
            Ok(())
        })
        .await;
        match inner {
            Ok(Ok(())) => Ok(()),
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
            // Prefer the model's REAL per-layer device map (the multi-GPU /
            // CPU hetero split). Previously this reported EVERY layer on the
            // primary device (`s.device`), so the GUI topology showed a
            // 2-GPU model entirely on GPU0 - the long-standing bug.
            // Placement comes from the plan, not from the model type: GenericHetero,
            // the tensor-parallel path and the multi-device variants (MoE-multi,
            // nemotron, lfm2moe) each read back the device their layers were loaded
            // onto. A variant with no map at all falls through to the single-entry
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
