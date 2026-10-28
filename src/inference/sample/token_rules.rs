//! The masks a request's token rules put on a row of logits before the draw.
//!
//! Each rule removes candidates by setting their logit to negative infinity, so the draw that
//! follows, greedy or sampled, never picks them: the tokens outside an allowed list, the last
//! token of a banned word wherever the tokens before it were just written, the end tokens
//! before a minimum length, and the tokens whose probability falls below a share of the
//! leader's.

use tokenizers::Tokenizer;

use crate::inference::engine::llm_engine::RequestRules;

/// A request's rules resolved against the model's vocabulary.
#[derive(Debug, Clone, Default)]
pub struct TokenRules {
    min_p: Option<f32>,
    /// Sorted, deduplicated.
    allowed: Option<Vec<u32>>,
    /// Token sequences of the banned words; the last token of each is banned after the others.
    banned: Vec<Vec<u32>>,
    min_tokens: usize,
    /// The tokens held back before `min_tokens` are generated.
    ends: Vec<u32>,
}

/// The token sequences of `word` as it opens a text and as it follows a space, the second
/// kept only when the space makes a different sequence of the same length, as vLLM bans them.
fn word_sequences(tokenizer: &Tokenizer, word: &str) -> Vec<Vec<u32>> {
    let bare = word.trim_start();
    let mut out: Vec<Vec<u32>> = Vec::new();
    for text in [bare.to_string(), format!(" {bare}")] {
        let Ok(enc) = tokenizer.encode(text.as_str(), false) else {
            continue;
        };
        let ids = enc.get_ids().to_vec();
        if ids.is_empty() {
            continue;
        }
        let distinct = match out.first() {
            None => true,
            Some(first) => first.len() == ids.len() && first[0] != ids[0],
        };
        if distinct {
            out.push(ids);
        }
    }
    out
}

impl TokenRules {
    /// `rules` against `tokenizer`, with `ends` the tokens that end the request.
    pub fn new(rules: &RequestRules, tokenizer: &Tokenizer, ends: &[u32]) -> Self {
        let allowed = rules.allowed_token_ids.as_ref().map(|ids| {
            let mut ids = ids.clone();
            ids.sort_unstable();
            ids.dedup();
            ids
        });
        let banned = rules
            .bad_words
            .iter()
            .flat_map(|w| word_sequences(tokenizer, w))
            .collect();
        Self {
            min_p: rules.min_p.filter(|&p| p > 0.0),
            allowed,
            banned,
            min_tokens: rules.min_tokens,
            ends: ends.to_vec(),
        }
    }

    /// Whether any rule is set.
    pub fn is_empty(&self) -> bool {
        self.min_p.is_none()
            && self.allowed.is_none()
            && self.banned.is_empty()
            && self.min_tokens == 0
    }

    /// Whether the generation may end after `generated` tokens.
    pub fn may_end(&self, generated: usize) -> bool {
        generated >= self.min_tokens
    }

    /// Masks `logits` for the next token after `generated`, the draw then made at
    /// `temperature` (zero or below is greedy, which a probability floor does not change).
    /// Refused when the allowed list names no token of the row.
    pub fn apply(
        &self,
        logits: &mut [f32],
        generated: &[u32],
        temperature: f32,
    ) -> Result<(), String> {
        if let Some(allowed) = &self.allowed {
            let mut kept = 0usize;
            for (i, l) in logits.iter_mut().enumerate() {
                if allowed.binary_search(&(i as u32)).is_ok() {
                    kept += 1;
                } else {
                    *l = f32::NEG_INFINITY;
                }
            }
            if kept == 0 {
                return Err("`allowed_token_ids` names no token of the vocabulary".into());
            }
        }
        for seq in &self.banned {
            let (last, head) = seq.split_last().expect("a banned sequence holds a token");
            if generated.ends_with(head) {
                if let Some(l) = logits.get_mut(*last as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        if !self.may_end(generated.len()) {
            for &t in &self.ends {
                if let Some(l) = logits.get_mut(t as usize) {
                    *l = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(p) = self.min_p.filter(|_| temperature > 0.0) {
            // Probability ratio to the leader at temperature t: exp((x - max) / t).
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            if max.is_finite() {
                let floor = max + p.ln() * temperature;
                for l in logits.iter_mut() {
                    if *l < floor {
                        *l = f32::NEG_INFINITY;
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokenizer() -> Tokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::pre_tokenizers::whitespace::Whitespace;
        let vocab: ahash::AHashMap<String, u32> = ["[UNK]", "a", "b", "c", "d", "e"]
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tok = Tokenizer::new(model);
        tok.with_pre_tokenizer(Some(Whitespace {}));
        tok
    }

    fn rules(f: impl FnOnce(&mut RequestRules)) -> TokenRules {
        let mut r = RequestRules::default();
        f(&mut r);
        TokenRules::new(&r, &tokenizer(), &[5])
    }

    fn allowed(row: &[f32]) -> Vec<usize> {
        (0..row.len()).filter(|&i| row[i].is_finite()).collect()
    }

    /// Only the listed tokens stay; a list naming no token of the row is refused.
    #[test]
    fn the_allowed_list_keeps_its_tokens_alone() {
        let r = rules(|r| r.allowed_token_ids = Some(vec![3, 1, 3]));
        let mut row = vec![0.0f32; 6];
        r.apply(&mut row, &[], 1.0).unwrap();
        assert_eq!(allowed(&row), vec![1, 3]);
        let r = rules(|r| r.allowed_token_ids = Some(vec![99]));
        assert!(r.apply(&mut [0.0f32; 6], &[], 1.0).is_err());
    }

    /// A one-token word is banned everywhere; the last token of a longer one only after the
    /// tokens before it.
    #[test]
    fn a_bad_word_bans_its_last_token_after_its_head() {
        let r = rules(|r| r.bad_words = vec!["b".into(), "c d".into()]);
        let mut row = vec![0.0f32; 6];
        r.apply(&mut row, &[1], 1.0).unwrap();
        assert_eq!(allowed(&row), vec![0, 1, 3, 4, 5]);
        let mut row = vec![0.0f32; 6];
        r.apply(&mut row, &[1, 3], 1.0).unwrap();
        assert_eq!(allowed(&row), vec![0, 1, 3, 5]);
    }

    /// The end tokens are held back until the minimum length is generated.
    #[test]
    fn the_end_waits_for_the_minimum_length() {
        let r = rules(|r| r.min_tokens = 2);
        let mut row = vec![0.0f32; 6];
        r.apply(&mut row, &[1], 0.0).unwrap();
        assert!(!row[5].is_finite());
        let mut row = vec![0.0f32; 6];
        r.apply(&mut row, &[1, 2], 0.0).unwrap();
        assert!(row[5].is_finite());
        assert!(!r.may_end(1) && r.may_end(2));
    }

    /// The floor drops what is less probable than `min_p` of the leader at the draw's
    /// temperature, and leaves a greedy row alone.
    #[test]
    fn min_p_drops_the_improbable_tail() {
        let r = rules(|r| r.min_p = Some(0.5));
        // Ratios to the leader at t = 1: 1, e^-0.5 (0.61), e^-1 (0.37).
        let base = [2.0f32, 1.5, 1.0, 2.0, -1.0, 0.0];
        let mut row = base.to_vec();
        r.apply(&mut row, &[], 1.0).unwrap();
        assert_eq!(allowed(&row), vec![0, 1, 3]);
        let mut row = base.to_vec();
        r.apply(&mut row, &[], 0.0).unwrap();
        assert_eq!(allowed(&row), vec![0, 1, 2, 3, 4, 5]);
    }
}
