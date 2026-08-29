//! Cross-encoder reranker (RAG #1) - scores a (query, document) pair for `/v1/rerank`.
//!
//! Follows the Qwen3-Reranker recipe: a causal LLM is prompted to answer "yes"/"no"
//! on whether the document satisfies the query, and the relevance score is the softmax
//! probability of the "yes" token at the final position. This reuses the existing full
//! LLM forward (`LlmEngine::target_forward_all`) - no new model weights or kernels: any
//! causal Qwen-family model can act as the reranker, Qwen3-Reranker being the tuned one.

/// Default retrieval instruction used when the caller doesn't supply one.
pub const DEFAULT_INSTRUCTION: &str =
    "Given a web search query, retrieve relevant passages that answer the query";

/// Build the Qwen3-Reranker chat prompt for one (query, document) pair.
/// The model judges relevance and is expected to emit "yes" or "no" next.
pub fn build_prompt(instruction: &str, query: &str, document: &str) -> String {
    let instruction = if instruction.trim().is_empty() {
        DEFAULT_INSTRUCTION
    } else {
        instruction
    };
    format!(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on \
the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".\
<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n\
<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

/// Relevance score in [0,1]: softmax over the "no"/"yes" logits, returning P(yes).
/// Numerically stable (subtracts the max before exp).
pub fn relevance_from_logits(logit_yes: f32, logit_no: f32) -> f32 {
    let m = logit_yes.max(logit_no);
    let ey = (logit_yes - m).exp();
    let en = (logit_no - m).exp();
    ey / (ey + en)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relevance_monotonic_and_bounded() {
        // Equal logits -> 0.5; yes ≫ no -> ->1; no ≫ yes -> ->0.
        assert!((relevance_from_logits(1.0, 1.0) - 0.5).abs() < 1e-6);
        assert!(relevance_from_logits(10.0, -10.0) > 0.999);
        assert!(relevance_from_logits(-10.0, 10.0) < 0.001);
        // Strictly increasing in logit_yes.
        assert!(relevance_from_logits(2.0, 0.0) > relevance_from_logits(1.0, 0.0));
    }

    #[test]
    fn prompt_contains_fields() {
        let p = build_prompt("", "cats", "felines are cats");
        assert!(p.contains(DEFAULT_INSTRUCTION));
        assert!(p.contains("<Query>: cats"));
        assert!(p.contains("<Document>: felines are cats"));
    }
}
