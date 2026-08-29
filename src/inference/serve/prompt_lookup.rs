//! Prompt-lookup decoding (PLD): draft n-gram continuations from the prompt
//! and recent generation to enable speculative decoding without a separate
//! draft model.
//!
//! Algorithm: keep a rolling buffer of recent tokens (prompt + generated).
//! At each decode step, take the last N tokens as a "query" n-gram and scan
//! the buffer backwards for a previous occurrence of that n-gram. The
//! tokens that followed the previous occurrence become the draft.
//!
//! References:
//!  - arXiv 2304.05128 (Assisted Generation)
//!  - llama.cpp `--draft-min` / `--draft-max` (prompt lookup)
//!  - vLLM `ngram` speculative decoder
//!
//! Best for repetitive content (code, structured output). For general chat
//! the hit rate is lower but still >0 because of common tokens (articles,
//! punctuation, code blocks).

use std::collections::VecDeque;

/// Rolling buffer of recent tokens with n-gram lookup.
#[derive(Debug)]
pub struct NgramDraftCache {
    /// Last-N tokens (prompt + generated, in order).
    tokens: VecDeque<u32>,
    /// Maximum buffer size. Older tokens drop off the front.
    max_buffer: usize,
    /// Minimum n-gram length to try (inclusive).
    min_ngram: usize,
    /// Maximum n-gram length to try (inclusive).
    max_ngram: usize,
    /// Maximum draft length returned by lookup.
    max_draft: usize,
}

impl NgramDraftCache {
    /// Create a cache that tries a single fixed n-gram length.
    /// Convenience wrapper; prefer `new_cascade` for adaptive lookup.
    pub fn new(ngram_size: usize, max_draft: usize, max_buffer: usize) -> Self {
        Self::new_cascade(ngram_size, ngram_size, max_draft, max_buffer)
    }

    /// Create a cache with a **cascading** n-gram lookup: try the longest
    /// n-gram first (higher precision - fewer but more reliable matches),
    /// and fall back to shorter n-grams when no match is found.
    ///
    /// - `min_ngram` / `max_ngram`: inclusive range, typical 2..=4. Longer
    ///   n-grams filter out coincidental matches (e.g. `[", ", ","]` matches
    ///   everywhere). Shorter catch repetition that longer would miss
    ///   (e.g. `self.x = ` appearing with different prefixes).
    /// - `max_draft`: how many tokens to draft after a match (typical 4-8).
    /// - `max_buffer`: rolling buffer size (typical 4096 tokens).
    pub fn new_cascade(
        min_ngram: usize,
        max_ngram: usize,
        max_draft: usize,
        max_buffer: usize,
    ) -> Self {
        assert!(min_ngram >= 1, "min_ngram must be >= 1");
        assert!(max_ngram >= min_ngram, "max_ngram must be >= min_ngram");
        assert!(max_draft >= 1, "max_draft must be >= 1");
        Self {
            tokens: VecDeque::with_capacity(max_buffer),
            max_buffer,
            min_ngram,
            max_ngram,
            max_draft,
        }
    }

    /// Append a single token, dropping oldest if the buffer is full.
    pub fn push(&mut self, token: u32) {
        if self.tokens.len() == self.max_buffer {
            self.tokens.pop_front();
        }
        self.tokens.push_back(token);
    }

    /// Append multiple tokens.
    pub fn push_many(&mut self, toks: &[u32]) {
        for &t in toks {
            self.push(t);
        }
    }

    /// Look up a draft by trying each n-gram length from `max_ngram` down to
    /// `min_ngram`, returning the first non-empty match.
    ///
    /// The cascade prefers longer (more specific) matches - a 4-gram match
    /// is stronger evidence of repetition than a 2-gram. This trades a few
    /// extra scans for much higher precision: on code workloads, 4-grams
    /// like `["self", ".", "name", " = "]` catch class-body repetition that
    /// a bare 2-gram `[" = ", "name"]` would produce spurious matches for.
    ///
    /// Complexity: O((max_ngram - min_ngram + 1) x buffer_size) worst case.
    /// For a 4K buffer with 2..=4, this is <10 μs on CPU - negligible
    /// compared to a single CUDA kernel launch.
    pub fn lookup(&self) -> Vec<u32> {
        // Snapshot the buffer once - the cascade does multiple scans.
        let len = self.tokens.len();
        if len < self.min_ngram + 1 {
            return Vec::new();
        }
        let slice: &[u32] = self.tokens.as_slices().0;
        let owned;
        let tokens: &[u32] = if slice.len() == len {
            slice
        } else {
            owned = self.tokens.iter().copied().collect::<Vec<_>>();
            &owned
        };

        for n in (self.min_ngram..=self.max_ngram).rev() {
            if let Some(draft) = Self::lookup_at_n(tokens, n, self.max_draft) {
                return draft;
            }
        }
        Vec::new()
    }

    /// Single-n lookup over a contiguous slice. Returns `Some(draft)` on a
    /// match (possibly empty draft if there's no room), `None` if no
    /// occurrence of the trailing n-gram exists earlier in the buffer.
    fn lookup_at_n(tokens: &[u32], n: usize, max_draft: usize) -> Option<Vec<u32>> {
        let len = tokens.len();
        if len < n + 1 {
            return None;
        }
        let query = &tokens[len - n..];
        let max_start = len - n - 1;
        for start in (0..=max_start).rev() {
            if tokens[start..start + n] == *query {
                let draft_start = start + n;
                // Clip against the buffer end AND the start of the trailing
                // n-gram (which is the query itself, not a continuation).
                let draft_end = (draft_start + max_draft).min(len - n);
                if draft_end <= draft_start {
                    return Some(Vec::new());
                }
                return Some(tokens[draft_start..draft_end].to_vec());
            }
        }
        None
    }

    /// Current buffer length.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the buffer has enough content for any lookup to succeed.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_on_no_match() {
        let mut c = NgramDraftCache::new(3, 4, 256);
        c.push_many(&[1, 2, 3, 4, 5]);
        assert!(c.lookup().is_empty());
    }

    #[test]
    fn finds_simple_repetition() {
        let mut c = NgramDraftCache::new(2, 4, 256);
        // "self . data ... self . data foo bar ... self ."
        c.push_many(&[10, 20, 30, 99, 10, 20, 30, 40, 41, 42, 88, 10, 20]);
        let draft = c.lookup();
        // Query is last-2 = [10, 20]. Most recent earlier match is at
        // index 4 (10, 20, 30, 40, 41). Continuation = [30, 40, 41, 42].
        assert_eq!(draft, vec![30, 40, 41, 42]);
    }

    #[test]
    fn respects_max_draft() {
        let mut c = NgramDraftCache::new(1, 2, 256);
        c.push_many(&[1, 2, 3, 4, 5, 6, 1]);
        // Query [1], match at index 0, continuation [2, 3, 4, 5, 6] capped to 2.
        assert_eq!(c.lookup(), vec![2, 3]);
    }

    #[test]
    fn picks_most_recent_match() {
        let mut c = NgramDraftCache::new(1, 4, 256);
        c.push_many(&[9, 1, 100, 100, 9, 1, 200, 200, 9]);
        // Query [9], multiple matches: indices 0, 4. Most recent is 4.
        // Continuation from index 5 = [1, 200, 200] - but wait, need to
        // stop before the trailing n-gram starts (index 8). That gives
        // [1, 200, 200] (length 3, bounded by max_draft=4 and len-n=8).
        assert_eq!(c.lookup(), vec![1, 200, 200]);
    }

    #[test]
    fn empty_query_too_short() {
        let mut c = NgramDraftCache::new(3, 4, 256);
        c.push_many(&[1, 2, 3]);
        // Only 3 tokens, ngram=3 -> need 4+ for query+continuation.
        assert!(c.lookup().is_empty());
    }

    #[test]
    fn buffer_eviction() {
        let mut c = NgramDraftCache::new(1, 4, 4);
        c.push_many(&[1, 2, 3, 4, 5, 6]);
        assert_eq!(c.len(), 4);
        // Buffer now contains [3, 4, 5, 6]. Query [6] has no match.
        assert!(c.lookup().is_empty());
    }
}
