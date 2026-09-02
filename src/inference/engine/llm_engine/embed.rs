//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl LlmEngine {
    /// Generate an L2-normalized embedding for `input` using this engine's model.
    ///
    /// The model is treated as a Qwen3-Embedding-style encoder: the GGUF backing
    /// this engine (resolved by `model_id`) is loaded once into a cached
    /// `EmbeddingModel` (last-token pooling + L2 norm) and reused for every request.
    pub async fn generate_embeddings(
        &self,
        input: &str,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Route through the ALREADY-LOADED model (last-token hidden + L2 norm), exactly
        // as rerank routes through its logits. Works with any dense llama-arch GGUF  -
        // an embedding model (qwen3-embedding, e5-mistral, ...) or a chat model as a
        // fallback. Replaces the old separate ACE-Step-encoder-only path that failed on
        // standard GGUFs with `no tensor layers.0.input_layernorm.weight`.
        let model_state = self.model_state.clone();
        let input = input.to_string();
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<Vec<f32>> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Embedding model not loaded"))?;
            let encoding = state
                .tokenizer
                .encode(input.as_str(), true)
                .map_err(|e| anyhow!("embedding encode: {e}"))?;
            let ids: Vec<u32> = encoding.get_ids().to_vec();
            if ids.is_empty() {
                return Err(anyhow!("embedding: empty token sequence"));
            }
            let device = state.device.clone();
            let x = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
            let mut feat = state.model.embed_last_hidden(&x)?;
            // L2-normalize (cosine-ready).
            let norm = feat.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
            for v in &mut feat {
                *v /= norm;
            }
            Ok(feat)
        })
        .await;
        match inner {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }
}
