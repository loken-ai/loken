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

    /// Get hidden states for a text and their sequence length.
    ///
    /// Returns the row-major hidden states `[S, hidden]` as a `Vec<f32>`
    /// and the sequence length `S`. The hidden states are produced by
    /// the causal Qwen3 forward pass without pooling or normalization.
    fn hidden_states(&self, text: &str) -> crate::tensor::Result<(Vec<f32>, usize)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| crate::tensor::Error::msg(format!("embedding encode: {e}")))?;
        let ids = encoding.get_ids();
        if ids.is_empty() {
            return Err(crate::tensor::Error::msg("empty token sequence"));
        }
        let hid = self.encoder.forward(ids)?;
        let s = ids.len();
        let h = self.hidden;
        debug_assert_eq!(hid.len(), s * h, "encoder output shape");
        Ok((hid, s))
    }

    /// Embed one text -> L2-normalized vector `[hidden]`.
    ///
    /// Returns the pooled and L2-normalized embedding vector of length `hidden`.
    /// For `Pooling::LastToken`, the last token's hidden state is used.
    /// For `Pooling::Mean`, the mean over all token hidden states is used.
    pub fn embed(&self, text: &str) -> crate::tensor::Result<Vec<f32>> {
        let (hid, s) = self.hidden_states(text)?;
        let h = self.hidden;

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

    /// Embed a text and return one L2-normalized vector per token.
    ///
    /// Returns a vector of hidden states, one per token, each L2-normalized.
    /// The returned vectors are in token order and each has length `hidden`.
    /// An empty token sequence returns an empty vector.
    pub fn embed_tokens(&self, text: &str) -> crate::tensor::Result<Vec<Vec<f32>>> {
        let (hid, s) = self.hidden_states(text)?;
        let h = self.hidden;
        let mut result = Vec::with_capacity(s);
        for t in 0..s {
            let row = &hid[t * h..(t + 1) * h];
            let mut v: Vec<f32> = row.to_vec();
            l2_normalize(&mut v);
            result.push(v);
        }
        Ok(result)
    }

    /// Embed a batch of texts (sequential - the encoder is single-sequence).
    pub fn embed_batch(&self, texts: &[String]) -> crate::tensor::Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

/// In-place L2 normalization; leaves an all-zero vector unchanged (degenerate input).
pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v {
            *x *= inv;
        }
    }
}

/// Compute late interaction score between query and document vectors.
///
/// For each query vector, computes the largest dot product with any document vector,
/// and returns the sum over all query vectors. An empty query or document returns 0.
/// Vectors of different lengths are compared over their common length via `zip`.
pub fn late_interaction(query: &[Vec<f32>], document: &[Vec<f32>]) -> f32 {
    if query.is_empty() || document.is_empty() {
        return 0.0;
    }
    let mut score = 0.0f32;
    for q in query {
        let mut best = f32::NEG_INFINITY;
        for d in document {
            let dot = q.iter().zip(d).map(|(x, y)| x * y).sum::<f32>();
            if dot > best {
                best = dot;
            }
        }
        score += best;
    }
    score
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

    #[test]
    fn late_interaction_sums_the_best_match_of_each_query_vector() {
        // query [[1,0],[0,1]], document [[1,0],[0.6,0.8]]
        // first query vector: dot with [1,0] = 1.0, dot with [0.6,0.8] = 0.6 -> best = 1.0
        // second query vector: dot with [1,0] = 0.0, dot with [0.6,0.8] = 0.8 -> best = 0.8
        // score = 1.0 + 0.8 = 1.8
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let document = vec![vec![1.0, 0.0], vec![0.6, 0.8]];
        let score = late_interaction(&query, &document);
        assert!((score - 1.8).abs() < 1e-6);
    }

    #[test]
    fn late_interaction_of_nothing_is_zero() {
        let query: Vec<Vec<f32>> = vec![];
        let document = vec![vec![1.0, 0.0], vec![0.6, 0.8]];
        let score = late_interaction(&query, &document);
        assert_eq!(score, 0.0);

        let query = vec![vec![1.0, 0.0]];
        let document: Vec<Vec<f32>> = vec![];
        let score = late_interaction(&query, &document);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn a_document_that_holds_the_query_scores_highest() {
        // query [[1,0],[0,1]]
        // document1 [[0,1],[1,0],[0.7,0.7]]:
        //   q[0] best = max(0.0, 1.0, 0.7) = 1.0
        //   q[1] best = max(1.0, 0.0, 0.7) = 1.0
        //   score = 2.0
        // document2 [[0.7,0.7]]:
        //   q[0] best = 0.7
        //   q[1] best = 0.7
        //   score = 1.4
        let query = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let doc1 = vec![vec![0.0, 1.0], vec![1.0, 0.0], vec![0.7, 0.7]];
        let doc2 = vec![vec![0.7, 0.7]];
        assert!(late_interaction(&query, &doc1) > late_interaction(&query, &doc2));
    }

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

    #[test]
    #[ignore = "needs Qwen3-Embedding GGUF (config huggingface_models_dir)"]
    fn token_vectors_are_unit_and_one_per_token() {
        let gguf = crate::inference::cache::hf::file(
            "models--Serveurperso--ACE-Step-1.5-GGUF",
            "Qwen3-Embedding-0.6B-Q8_0.gguf",
        );
        let m = EmbeddingModel::from_gguf(gguf.to_str().unwrap(), Pooling::LastToken).unwrap();
        let texts = ["a cat", "a cat sat on the mat", "stock prices fell"];
        let tokens = texts
            .iter()
            .map(|t| m.embed_tokens(t).unwrap())
            .collect::<Vec<_>>();
        // One vector per token
        let vecs = &tokens[0];
        assert_eq!(vecs.len(), 2); // "a cat" = 2 tokens
        assert!(vecs.iter().all(|v| v.len() == m.dim()));
        // Each vector has norm 1 within 1e-5
        assert!(vecs.iter().all(|v| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            (norm - 1.0).abs() < 1e-5
        }));
        // late_interaction ranks "a cat sat on the mat" higher than "stock prices fell"
        // for the query "where is the cat"
        let query_vecs = m.embed_tokens("where is the cat").unwrap();
        let score_cat = late_interaction(&query_vecs, &tokens[0]);
        let score_full = late_interaction(&query_vecs, &tokens[1]);
        let score_stock = late_interaction(&query_vecs, &tokens[2]);
        println!("score(a cat)={score_cat:.4}, score(a cat sat on the mat)={score_full:.4}, score(stock prices fell)={score_stock:.4}");
        assert!(score_cat > score_stock);
        assert!(score_full > score_stock);
    }
}
