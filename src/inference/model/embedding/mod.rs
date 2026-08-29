//! Text embedding model (RAG #1) - produces a single L2-normalized vector per input
//! text for `/v1/embeddings` and `/api/embed`.
//!
//! Reuses the validated Qwen3 encoder stack from `native_acestep_textenc::TextEncoder`
//! (28 causal pre-norm Qwen3 layers, GQA, qk-norm, RoPE θ1e6) - the same machinery
//! ACE-Step already runs for Qwen3-Embedding-0.6B. The only embedding-specific logic
//! added here is tokenization, pooling and normalization:
//!   - **last-token** pooling (Qwen3-Embedding convention: causal attention, the final
//!     token's hidden state summarizes the sequence), or
//!   - **mean** pooling (masked average - for encoder-style models like BGE).
//! The pooled vector is L2-normalized so cosine similarity reduces to a dot product.

use crate::inference::model::acestep::textenc::TextEncoder;
use tokenizers::Tokenizer;

/// How the per-token hidden states `[S, H]` are reduced to one `[H]` vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// Hidden state of the last token (Qwen3-Embedding / decoder-style models).
    LastToken,
    /// Mean over all token hidden states (encoder-style models, e.g. BGE).
    Mean,
}

/// A loaded text-embedding model: tokenizer + Qwen3 encoder + pooling strategy.
pub struct EmbeddingModel {
    encoder: TextEncoder,
    tokenizer: Tokenizer,
    hidden: usize,
    pooling: Pooling,
}

impl EmbeddingModel {
    /// Output embedding dimension (model hidden size).
    pub fn dim(&self) -> usize {
        self.hidden
    }

    /// Load a Qwen3-Embedding GGUF: the tokenizer is read from the embedded
    /// `tokenizer.ggml.*` metadata, the weights via the shared `TextEncoder` loader.
    /// `pooling` selects last-token (Qwen3-Embedding) or mean (BGE-style).
    pub fn from_gguf(path: &str, pooling: Pooling) -> crate::tensor::Result<Self> {
        // Build the tokenizer from the same GGUF's embedded vocab (Ollama-style GGUFs
        // never ship a separate tokenizer.json - vocab+merges live in the file).
        let tokenizer = {
            use crate::tensor::quantized::gguf_file;
            let f = std::fs::File::open(path)?;
            let content = gguf_file::read_mapped_file(&f)?;
            crate::inference::engine::llm_engine::build_tokenizer_from_gguf(&content)
                .map_err(|e| crate::tensor::Error::msg(format!("embedding tokenizer: {e}")))?
        };
        let encoder = TextEncoder::from_gguf(path)?;
        let hidden = encoder.hidden;
        Ok(Self {
            encoder,
            tokenizer,
            hidden,
            pooling,
        })
    }

    /// Embed one text -> L2-normalized vector `[hidden]`.
    pub fn embed(&self, text: &str) -> crate::tensor::Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| crate::tensor::Error::msg(format!("embedding encode: {e}")))?;
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Err(crate::tensor::Error::msg("empty token sequence"));
        }
        // Causal Qwen3 forward -> row-major hidden states `[S, hidden]`.
        let hid = self.encoder.forward(ids)?;
        let s = ids.len();
        let h = self.hidden;
        debug_assert_eq!(hid.len(), s * h, "encoder output shape");

        let mut pooled = vec![0f32; h];
        match self.pooling {
            Pooling::LastToken => {
                pooled.copy_from_slice(&hid[(s - 1) * h..s * h]);
            }
            Pooling::Mean => {
                for t in 0..s {
                    let row = &hid[t * h..(t + 1) * h];
                    for (p, &v) in pooled.iter_mut().zip(row) {
                        *p += v;
                    }
                }
                let inv = 1.0f32 / s as f32;
                for p in &mut pooled {
                    *p *= inv;
                }
            }
        }
        l2_normalize(&mut pooled);
        Ok(pooled)
    }

    /// Embed a batch of texts (sequential - the encoder is single-sequence).
    pub fn embed_batch(&self, texts: &[String]) -> crate::tensor::Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// In-place L2 normalization; leaves an all-zero vector unchanged (degenerate input).
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v {
            *x *= inv;
        }
    }
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn l2_normalize_unit_and_zero() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);
        let mut z = vec![0.0, 0.0];
        l2_normalize(&mut z);
        assert_eq!(z, vec![0.0, 0.0]);
    }

    // End-to-end semantic sanity: synonyms should embed closer than unrelated words.
    // Reuses the Qwen3-Embedding-0.6B GGUF already present for the ACE-Step text-encoder
    // parity test (resolved via config, no hardcoded path).
    #[test]
    #[ignore = "needs Qwen3-Embedding GGUF (config huggingface_models_dir)"]
    fn embedding_semantic_order() {
        let gguf = crate::inference::cache::hf::file(
            "models--Serveurperso--ACE-Step-1.5-GGUF",
            "Qwen3-Embedding-0.6B-Q8_0.gguf",
        );
        let m = EmbeddingModel::from_gguf(gguf.to_str().unwrap(), Pooling::LastToken).unwrap();
        let cat = m.embed("a cat").unwrap();
        let dog = m.embed("a dog").unwrap();
        let car = m.embed("a car").unwrap();
        assert_eq!(cat.len(), m.dim());
        // Vectors are L2-normalized -> cosine == dot.
        let cat_dog = cosine(&cat, &dog);
        let cat_car = cosine(&cat, &car);
        println!("cos(cat,dog)={cat_dog:.4}  cos(cat,car)={cat_car:.4}");
        assert!(
            cat_dog > cat_car,
            "synonym pair should be closer than unrelated"
        );
    }
}
