//! Shared per-token decode-step logic for `LlmEngine::generate` and
//! `LlmEngine::generate_stream`.
//!
//! The two decode loops used to duplicate ~800 lines of sampling / penalty /
//! stop-token / PLD-verification logic. The pieces that are genuinely
//! identical live here, moved verbatim so the numerics and sampling order do
//! not change:
//!
//! - [`resolve_gen_params`]: per-request parameter resolution (request
//!   overrides -> config defaults, max_tokens clamped to the KV window).
//! - [`StopTracker`]: rolling stop-sequence suffix buffer.
//! - [`incremental_chunk_text`]: O(1) 2-token sliding-window detokenize for
//!   streamed chunks.
//! - [`sample_row_sync`]: the GPU-argmax-or-CPU-sampler dispatch.
//! - [`pld_verify_commit`]: the commit-delayed PLD/spec-draft verification
//!   loop (sample every position of the multi-token verify forward, accept
//!   the longest matching draft prefix).
//! - [`pld_window_update`]: PLD acceptance-window cooldown bookkeeping.
//! - [`spec_draft_lockstep`]: keep the external drafter's KV aligned with the
//!   target after a PLD/spec commit.
//!
//! The paths intentionally KEPT apart (behavioral divergences, not
//! duplication): the CUDA-graph capture/replay section (non-stream only), the
//! Path-B device-resident next-token tensor (stream only), the session /
//! global-prompt-cache prefix reuse (different policies), and the legacy
//! `opencl` speculative blocks (compiled out on CUDA builds).

use tokenizers::Tokenizer;

use crate::inference::sample::token_sampling::LogitsProcessor;
use crate::inference::serve::prompt_lookup::NgramDraftCache;
use crate::tensor::{IndexOp, Tensor};

use crate::inference::engine::llm_engine::{
    apply_repeat_penalty, build_sampling_from, gpu_sample, GenerationParams, LlmEngine,
};

/// Per-request sampling/limit parameters after applying the
/// request-overrides-config-defaults policy shared by both decode paths.
pub(crate) struct ResolvedGenParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
}

/// Resolve generation parameters: the request's value, else the model's recommendation, else
/// the API default. Generation runs to the end of the KV window (`effective_context`) unless
/// the request caps it, and a request that names no seed samples its own stream.
pub(crate) fn resolve_gen_params(
    params: &GenerationParams,
    defaults: &crate::inference::engine::llm_engine::generation_defaults::GenerationDefaults,
    effective_context: usize,
    prompt_len: usize,
) -> ResolvedGenParams {
    let room = effective_context.saturating_sub(prompt_len);
    ResolvedGenParams {
        max_tokens: params.max_tokens.map_or(room, |m| m.min(room)),
        temperature: params.temperature.unwrap_or_else(|| defaults.temperature()),
        top_p: params.top_p.unwrap_or_else(|| defaults.top_p()),
        top_k: params.top_k.unwrap_or_else(|| defaults.top_k()),
        seed: params
            .seed
            .unwrap_or_else(crate::inference::engine::llm_engine::generation_defaults::fresh_seed),
        repeat_penalty: params
            .repeat_penalty
            .unwrap_or_else(|| defaults.repeat_penalty()),
        repeat_last_n: params
            .repeat_last_n
            .unwrap_or_else(|| defaults.repeat_last_n()),
    }
}

impl ResolvedGenParams {
    /// Build the logits processor for these parameters (same construction the
    /// two paths previously duplicated).
    pub(crate) fn make_logits_processor(&self) -> LogitsProcessor {
        let sampling = build_sampling_from(self.temperature, self.top_p, self.top_k);
        LogitsProcessor::from_sampling(self.seed, sampling)
    }
}

/// Rolling stop-sequence suffix buffer. Only the last `2 x max stop length`
/// chars of decoded text are retained, so the per-token check is O(1) in the
/// generation length.
pub(crate) struct StopTracker {
    max_len: usize,
    buf: String,
    hit: Option<String>,
}

impl StopTracker {
    pub(crate) fn new(stop_sequences: &[String]) -> Self {
        Self {
            max_len: stop_sequences.iter().map(String::len).max().unwrap_or(0),
            buf: String::new(),
            hit: None,
        }
    }

    /// False when there are no stop sequences (all tracking is skipped).
    pub(crate) fn is_active(&self) -> bool {
        self.max_len > 0
    }

    /// Append newly decoded text; returns true when any stop sequence now
    /// terminates the rolling suffix (generation must stop).
    pub(crate) fn push_and_check(&mut self, text: &str, stop_sequences: &[String]) -> bool {
        self.buf.push_str(text);
        // Keep buffer trimmed to 2x max stop sequence length, cut on a char boundary.
        if self.buf.len() > self.max_len * 2 {
            let mut trim_at = self.buf.len() - self.max_len * 2;
            while !self.buf.is_char_boundary(trim_at) {
                trim_at += 1;
            }
            self.buf.drain(..trim_at);
        }
        match stop_sequences
            .iter()
            .find(|s| self.buf.ends_with(s.as_str()))
        {
            Some(s) => {
                self.hit = Some(s.clone());
                true
            }
            None => false,
        }
    }
    /// The stop sequence that ended the generation, once one has.
    pub(crate) fn matched(&self) -> Option<&str> {
        self.hit.as_deref()
    }
}

/// Incremental detokenize for streaming (2-token sliding window).
///
/// Decoding the full cumulative sequence every iter is O(N) per call -> O(N²)
/// over the stream; decoding a lone token loses leading spaces with
/// Metaspace/SentencePiece tokenizers. Decode `[prev_token, this_token]` and
/// subtract `[prev_token]` to get just this token's contribution - preserves
/// the space/metaspace boundary at O(1) work per iter.
/// `skip_special` travels with the caller rather than being assumed here. It used to be
/// hardcoded, so a vocabulary whose channel tokens must reach the reader lost them on the
/// streamed path while keeping them on the whole-response one: the same answer arrived clean
/// when returned in one piece and carried a bare channel name when streamed.
pub(crate) fn incremental_chunk_text(
    tokenizer: &Tokenizer,
    generated_token_ids: &[u32],
    skip_special: bool,
) -> String {
    let n = generated_token_ids.len();
    if n >= 2 {
        let two_text = tokenizer
            .decode(&generated_token_ids[n - 2..], skip_special)
            .unwrap_or_default();
        let one_text = tokenizer
            .decode(&generated_token_ids[n - 2..n - 1], skip_special)
            .unwrap_or_default();
        two_text
            .strip_prefix(one_text.as_str())
            .map(String::from)
            .unwrap_or(two_text)
    } else {
        tokenizer
            .decode(generated_token_ids, skip_special)
            .unwrap_or_default()
    }
}

/// Sample one logits row: fused GPU penalty+argmax when the row lives on
/// CUDA, otherwise the host repeat-penalty + `LogitsProcessor` path.
pub(crate) fn sample_row_sync(
    row: &Tensor,
    recent_tokens: &[u32],
    repeat_penalty: f32,
    repeat_last_n: usize,
    logits_processor: &mut LogitsProcessor,
) -> crate::tensor::Result<u32> {
    if row.device().is_cuda() {
        gpu_sample(
            row,
            recent_tokens,
            repeat_penalty,
            repeat_last_n,
            logits_processor,
        )
    } else {
        let row_cpu = apply_repeat_penalty(row, recent_tokens, repeat_penalty, repeat_last_n)?;
        logits_processor.sample(&row_cpu)
    }
}

/// Result of one PLD/spec-draft verification pass.
pub(crate) struct PldCommitOutcome {
    /// The last committed sample - becomes `next_token` (pushed/streamed by
    /// the caller's next loop iteration, completing the commit-delayed
    /// pattern).
    pub next_token: u32,
    /// Device-resident `[1,1]` U32 tensor for `next_token` when the sampler
    /// produced one (stream Path-B greedy path); `None` otherwise.
    pub next_token_dev: Option<Tensor>,
    /// How many draft tokens were accepted (committed = 1 + accepted).
    pub accepted_drafts: usize,
}

/// Commit-delayed PLD/spec-draft verification loop, shared verbatim by the
/// stream and non-stream decode paths.
///
/// `logits` is the multi-position verify forward's output `[1, k+1, vocab]`
/// over `[current_token, draft_0, .., draft_{k-1}]`. Position `i` predicts
/// the token AFTER input `i`:
///   - `i=0`: prediction after `current_token` (always "true")
///   - `i>0`: prediction after `draft[i-1]` (valid iff ALL `draft[0..i]`
///     matched previous samples)
///
/// We don't push a sample immediately - we push the PREVIOUS sample once we
/// know it's committed, so the final sample (the new `next_token`) gets
/// pushed naturally by the caller's outer loop on the next iteration.
/// Committed tokens go into `committed_out` (the non-stream path passes its
/// `generated` vec directly; the stream path passes a scratch vec it then
/// emits), and are mirrored into `recent_tokens` (so the repeat penalty at
/// position `i` sees earlier commits) and `pld_cache`.
///
/// Capacity: stop when `committed_base + committed_out.len() + 1 >=
/// max_tokens` - `committed_base` is the count of already-emitted tokens NOT
/// included in `committed_out` (0 for the non-stream path, `token_count` for
/// the stream path).
pub(crate) fn pld_verify_commit(
    logits: &Tensor,
    draft: &[u32],
    committed_out: &mut Vec<u32>,
    recent_tokens: &mut Vec<u32>,
    pld_cache: &mut NgramDraftCache,
    pld_enabled: bool,
    eos_primary: u32,
    eos_extra: &[u32],
    committed_base: usize,
    max_tokens: usize,
    sampler: &mut dyn FnMut(&Tensor, &[u32]) -> crate::tensor::Result<(u32, Option<Tensor>)>,
) -> crate::tensor::Result<PldCommitOutcome> {
    let k = draft.len();
    let mut prev_sampled: Option<u32> = None;
    let mut prev_sampled_dev: Option<Tensor> = None;
    let mut accepted_drafts = 0usize;

    // i needs to be a literal value: passed to logits.i((.., i, ..)) and used
    // to index draft[i].
    #[allow(clippy::needless_range_loop)]
    for i in 0..=k {
        let row = logits.i((.., i, ..))?.squeeze(0)?;
        let (sampled, dev_next) = sampler(&row, recent_tokens)?;

        // Push the PREVIOUS sample now that we know it's committed (we're
        // about to sample the next).
        if let Some(p) = prev_sampled {
            committed_out.push(p);
            recent_tokens.push(p);
            if pld_enabled {
                pld_cache.push(p);
            }
        }
        prev_sampled = Some(sampled);
        prev_sampled_dev = dev_next;

        // Stop conditions evaluated on the new sample: EOS / max_tokens ->
        // the caller's outer loop will catch them when next_token is set.
        if sampled == eos_primary || eos_extra.contains(&sampled) {
            break;
        }
        if committed_base + committed_out.len() + 1 >= max_tokens {
            break;
        }
        if i == k {
            break;
        }
        if draft[i] != sampled {
            break;
        }
        accepted_drafts += 1;
    }

    Ok(PldCommitOutcome {
        next_token: prev_sampled.expect("at least one sample committed"),
        next_token_dev: prev_sampled_dev,
        accepted_drafts,
    })
}

/// Longest draft a drafter is asked for.
pub(crate) const DRAFT_MAX: usize = 8;
/// Rows of a verify forward: the token carried forward, then the draft.
pub(crate) const VERIFY_ROWS: usize = DRAFT_MAX + 1;
/// Weight of the newest cycle in the running cost of a draft length.
const CYCLE_COST_WEIGHT: f64 = 0.2;

/// How many tokens to ask the drafter for, learned from the cycles themselves.
///
/// A cycle drafts `k` tokens and verifies them in one target forward; it commits one token
/// plus the accepted prefix of the draft. Two measured quantities decide the best `k`: how
/// often each draft position survives once the ones before it did, and what a cycle costs
/// (the drafter's steps plus a verify whose price depends on where the target runs). A
/// length already used is priced at its own running cycle time. One never used is priced
/// from the nearest used length, its verify scaled by the rows verified plus its drafting
/// time: shorter lengths look cheaper than they may be, so they get tried and are then
/// judged on their own time. Length 0 is plain decoding, one token per step, priced the
/// same way. The chosen `k` maximises committed tokens per second, the longer length on a
/// tie.
#[derive(Debug, Clone, Default)]
pub(crate) struct DraftLength {
    /// Cycles that reached draft position `i`.
    reached: [u32; DRAFT_MAX],
    /// Cycles whose draft position `i` was accepted.
    kept: [u32; DRAFT_MAX],
    /// Running cycle seconds and verify seconds, per draft length.
    cycle: [Option<(f64, f64)>; DRAFT_MAX + 1],
    /// Running drafter seconds per drafted token.
    draft_token: Option<f64>,
}

impl DraftLength {
    pub(crate) fn next(&self) -> usize {
        let draft_token = self.draft_token.unwrap_or(0.0);
        // A verify covers the carried token plus the draft: `k + 1` rows.
        let scaled_verify = |k: usize| {
            (0..=DRAFT_MAX)
                .filter_map(|j| self.cycle[j].map(|(_, v)| (j.abs_diff(k), j, v)))
                .min_by(|a, b| a.0.cmp(&b.0))
                .map(|(_, j, v)| v * (k + 1) as f64 / (j + 1) as f64)
        };
        let mut best = (0, f64::MIN);
        let mut reach = 1.0;
        let mut expected = 1.0;
        for k in 0..=DRAFT_MAX {
            if k > 0 {
                // Laplace-smoothed survival of position k-1 given the earlier ones survived.
                reach *= (self.kept[k - 1] as f64 + 1.0) / (self.reached[k - 1] as f64 + 2.0);
                expected += reach;
            }
            let cost = match self.cycle[k] {
                Some((c, _)) => c,
                None => match scaled_verify(k) {
                    Some(v) => v + k as f64 * draft_token,
                    // Nothing measured yet: every length costs the same.
                    None => 1.0,
                },
            };
            let rate = expected / cost.max(f64::EPSILON);
            if rate >= best.1 {
                best = (k, rate);
            }
        }
        best.0
    }

    /// Records a cycle that drafted `k` tokens in `draft_secs`, accepted `accepted` of them,
    /// and took `cycle_secs` in all; `k` 0 is a plain decode step.
    pub(crate) fn record(&mut self, k: usize, accepted: usize, draft_secs: f64, cycle_secs: f64) {
        let k = k.min(DRAFT_MAX);
        for i in 0..k.min(accepted + 1) {
            self.reached[i] += 1;
            if i < accepted {
                self.kept[i] += 1;
            }
        }
        let blend = |old: Option<f64>, new: f64| match old {
            Some(o) => o + (new - o) * CYCLE_COST_WEIGHT,
            None => new,
        };
        let verify = (cycle_secs - draft_secs).max(0.0);
        let old = self.cycle[k];
        self.cycle[k] = Some((
            blend(old.map(|c| c.0), cycle_secs),
            blend(old.map(|c| c.1), verify),
        ));
        if k > 0 {
            self.draft_token = Some(blend(self.draft_token, draft_secs / k as f64));
        }
    }
}

/// PLD acceptance-window cooldown: when recent acceptance drops SIGNIFICANTLY
/// below break-even, skip PLD for 500 iters. Threshold 0.20 over a 24-draft
/// window - Q4/Q8 main-path arches have lower natural acceptance (~25-35 %)
/// than the F-dtype fallback path; an aggressive 0.35/6-draft cooldown caused
/// ~10 pp regression on deepcoder+qwen3 at medium, and one short unlucky
/// window must not disable PLD for 500 iters.
pub(crate) fn pld_window_update(
    window_drafted: &mut u32,
    window_accepted: &mut u32,
    cooldown: &mut u32,
    k: usize,
    accepted_drafts: usize,
) {
    *window_drafted += k as u32;
    *window_accepted += accepted_drafts as u32;
    if *window_drafted >= 24 {
        let rate = *window_accepted as f32 / *window_drafted as f32;
        if rate < 0.20 {
            *cooldown = 500;
        }
        *window_drafted = 0;
        *window_accepted = 0;
    }
}

/// Spec-decode: keep the DRAFTER KV in lockstep with the target after
/// a verification commit. The drafter forwarded `k` tokens this cycle (its KV
/// = pos_before + k); after commit the target sits at `new_pos`.
///  - partial accept (accepted < k): trim drafter to `new_pos`.
///  - all accepted: the drafter never forwarded the LAST draft (it only
///    predicted it), so its KV is one short - feed `draft[k-1]` @ `new_pos-1`
///    to add that entry (-> KV = new_pos).
pub(crate) fn spec_draft_lockstep(
    spec_engine: &LlmEngine,
    draft: &[u32],
    accepted_drafts: usize,
    new_pos: usize,
) {
    let k = draft.len();
    if accepted_drafts == k {
        let _ = spec_engine.draft_k(draft[k - 1], 1, new_pos - 1, &[], 1.0, 0);
    } else {
        spec_engine.draft_trim_kv(new_pos);
    }
}

/// Moves the sampler's last log-probabilities, if it kept any, into `sink`, decoded.
pub(crate) fn record_logprobs(
    tokenizer: &Tokenizer,
    logits_processor: &mut LogitsProcessor,
    sink: &mut Vec<crate::inference::engine::llm_engine::TokenLogprob>,
) {
    let Some(lp) = logits_processor.take_last_logprobs() else {
        return;
    };
    let piece = |id: u32| tokenizer.decode(&[id], false).unwrap_or_default();
    sink.push(crate::inference::engine::llm_engine::TokenLogprob {
        token: lp.token,
        text: piece(lp.token),
        logprob: lp.logprob,
        top: lp
            .top
            .into_iter()
            .map(|(id, l)| (id, piece(id), l))
            .collect(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_drafter_that_rarely_agrees_gets_short_drafts() {
        let mut d = DraftLength::default();
        // Verify costs the same at any length; drafting costs a tenth of it per token.
        for _ in 0..200 {
            let k = d.next();
            d.record(k, 0, 0.1 * k as f64, 1.0 + 0.1 * k as f64);
        }
        assert!(d.next() <= 1, "chose {}", d.next());
    }

    #[test]
    fn plain_decoding_wins_when_a_cycle_costs_more_than_it_commits() {
        let mut d = DraftLength::default();
        // Half the positions survive, but a plain step costs a fifth of any cycle.
        for n in 0..400u32 {
            let k = d.next();
            if k == 0 {
                d.record(0, 0, 0.0, 0.2);
                continue;
            }
            let accepted = (0..k).take_while(|i| (n >> i) & 1 == 1).count();
            d.record(k, accepted, 0.01 * k as f64, 1.0 + 0.01 * k as f64);
        }
        assert_eq!(d.next(), 0);
    }

    #[test]
    fn a_drafter_that_always_agrees_gets_long_drafts() {
        let mut d = DraftLength::default();
        for _ in 0..200 {
            let k = d.next();
            d.record(k, k, 0.01 * k as f64, 1.0 + 0.01 * k as f64);
        }
        assert_eq!(d.next(), DRAFT_MAX);
    }

    #[test]
    fn a_flat_verify_takes_long_drafts_even_at_half_acceptance() {
        let mut d = DraftLength::default();
        for n in 0..400u32 {
            let k = d.next();
            let accepted = (0..k).take_while(|i| (n >> i) & 1 == 1).count();
            d.record(k, accepted, 0.005 * k as f64, 1.0 + 0.005 * k as f64);
        }
        assert!(d.next() >= 4, "chose {}", d.next());
    }

    #[test]
    fn a_verify_that_grows_with_length_caps_the_draft() {
        let mut d = DraftLength::default();
        // Half the positions survive; each verified row adds half a single-row verify.
        for n in 0..400u32 {
            let k = d.next();
            let accepted = (0..k).take_while(|i| (n >> i) & 1 == 1).count();
            d.record(k, accepted, 0.0, 1.0 + 0.5 * k as f64);
        }
        assert!(d.next() <= 2, "chose {}", d.next());
    }

    #[test]
    fn the_stop_buffer_trims_on_a_char_boundary() {
        let stops = vec!["<|end|>".to_string()];
        let mut t = StopTracker::new(&stops);
        for piece in ["héllo ", "wörld ", "日本語", "é", "<|end", "|>"] {
            if t.push_and_check(piece, &stops) {
                break;
            }
        }
        assert_eq!(t.matched(), Some("<|end|>"));
    }
}
