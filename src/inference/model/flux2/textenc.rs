//! FLUX.2 Klein conditioning: Qwen3-4B hidden states -> the DiT's 7680-wide context.
//!
//! The encoder itself is the Qwen3 decoder this repo already runs
//! ([`crate::inference::model::qwen3vl::textenc`]) at a narrower config, so nothing here loads or
//! runs weights. What IS specific to FLUX.2 is how a prompt becomes conditioning, and every part
//! of it is load-bearing in a way that fails quietly rather than loudly:
//!
//! * **Three MID-STACK layers, concatenated.** `hidden_states[9]`, `[18]` and `[27]` of 36,
//!   joined on the feature axis - `3 x 2560 = 7680`, which is exactly the DiT's
//!   `joint_attention_dim`. Using the last layer instead would have the right rank and the wrong
//!   width; using the right width from the wrong layers would have both and still be wrong.
//! * **Raw residual-stream values, not normalised.** HuggingFace applies the final RMSNorm only
//!   to the LAST hidden state, so intermediate taps come out unnormalised - which is what the
//!   reference pipeline concatenates.
//! * **The chat template, verbatim.** Qwen3 with `enable_thinking=False` still emits an EMPTY
//!   `<think></think>` block; a hand-written "user ... assistant" template omits it and shifts
//!   every token.
//! * **Padded to a fixed 512** on the right with `<|endoftext|>`. The DiT attends over all 512
//!   positions, so the padded tail is part of the conditioning, not slack to be trimmed.

use crate::inference::model::qwen3vl::textenc::Qwen3VlTextEncoder;
use crate::tensor::{Error, Result, Tensor as NT};

/// Hidden-state indices the pipeline concatenates, in HuggingFace's `output_hidden_states`
/// numbering (`k` = the output of the k-th block).
pub const HIDDEN_LAYERS: [usize; 3] = [9, 18, 27];
/// The fixed context length. Not a maximum - the reference pads TO it unconditionally.
pub const TEXT_LEN: usize = 512;
/// `<|endoftext|>`, the pad token the reference right-pads with.
pub const PAD_ID: u32 = 151_643;

/// Qwen3's chat template with thinking disabled, exactly as `apply_chat_template(...,
/// add_generation_prompt=True, enable_thinking=False)` renders it. The empty `<think></think>`
/// is not a typo - it is what the template emits when thinking is off.
pub fn chat_template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// Tokenize a prompt through the chat template and right-pad to [`TEXT_LEN`].
///
/// Returns the ids and how many of them are real, because the count is worth asserting on: a
/// template that silently changed would show up here as a different prefix length long before it
/// showed up as a worse image.
pub fn tokenize(tok: &tokenizers::Tokenizer, prompt: &str) -> Result<(Vec<u32>, usize)> {
    let text = chat_template(prompt);
    let mut ids: Vec<u32> = tok
        .encode(text.as_str(), false)
        .map_err(|e| Error(format!("flux2 tokenize: {e}")))?
        .get_ids()
        .to_vec();
    let real = ids.len().min(TEXT_LEN);
    ids.truncate(TEXT_LEN);
    ids.resize(TEXT_LEN, PAD_ID);
    Ok((ids, real))
}

/// Prompt -> `[512, 7680]` conditioning for the DiT.
pub fn encode(enc: &Qwen3VlTextEncoder, tok: &tokenizers::Tokenizer, prompt: &str) -> Result<NT> {
    let (ids, real) = tokenize(tok, prompt)?;
    encode_ids(enc, &ids, real)
}

/// As [`encode`], but from already-tokenized ids - the form a parity run needs so it can feed the
/// reference's exact tokens rather than trusting two tokenizers to agree.
pub fn encode_ids(enc: &Qwen3VlTextEncoder, ids: &[u32], n_real: usize) -> Result<NT> {
    // The padded tail is CONDITIONING, not slack - the DiT attends over all 512 positions - so
    // the pads have to be computed the way the reference computes them, which means masking them
    // out as attention keys. Without that the real tokens are still exact and the tail is noise.
    let taps = enc.forward_taps_padded(ids, &HIDDEN_LAYERS, n_real)?;
    if taps.len() != HIDDEN_LAYERS.len() {
        return Err(Error(format!(
            "flux2 conditioning: asked for {} taps, got {}",
            HIDDEN_LAYERS.len(),
            taps.len()
        )));
    }
    // Concatenate on the FEATURE axis, in layer order: [s, 2560] x3 -> [s, 7680]. The reference
    // stacks then permutes then reshapes, which lands each token's three layers adjacent in
    // ascending layer order - the same thing this cat produces.
    let refs: Vec<&NT> = taps.iter().collect();
    NT::cat(&refs, 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concatenated_width_is_the_dit_context_dim() {
        let cfg = crate::inference::model::qwen3vl::textenc::Config::qwen3_4b();
        let dit = crate::inference::model::flux2::dit::Config::klein_4b();
        assert_eq!(cfg.hidden * HIDDEN_LAYERS.len(), dit.joint_attention_dim);
        // The taps must exist in a 36-block stack, and be mid-stack rather than the last layer.
        assert!(HIDDEN_LAYERS.iter().all(|&k| k >= 1 && k < cfg.n_layers));
    }

    /// The empty `<think></think>` block is what `enable_thinking=False` renders. Dropping it is
    /// the single easiest way to get conditioning that is subtly off for every prompt.
    #[test]
    fn chat_template_keeps_the_empty_thinking_block() {
        let t = chat_template("a fox");
        assert_eq!(
            t,
            "<|im_start|>user\na fox<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
