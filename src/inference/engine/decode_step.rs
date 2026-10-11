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
//! - [`StreamDetokenizer`]: incremental detokenize for streamed chunks, holding back an
//!   incomplete multi-byte character until a later token completes it.
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

/// Incremental detokenizer for streaming.
///
/// Each step decodes from the last committed character boundary forward and emits only the
/// text beyond what was already sent: the window never starts mid-word, so a SentencePiece
/// leading space survives, and a trailing UTF-8 sequence a later token will finish is held
/// back rather than sent as U+FFFD, however many tokens its bytes straddle. The boundary
/// advances each time a character completes, so a step decodes a few tokens.
///
/// `skip_special` travels with the caller: a vocabulary whose channel tokens must reach the
/// reader keeps them streamed as it does whole.
#[derive(Debug, Default)]
pub(crate) struct StreamDetokenizer {
    /// First token of the open window: the last committed character boundary.
    prev_index: usize,
    /// End of the tokens whose text has been emitted.
    current_index: usize,
}

impl StreamDetokenizer {
    /// Text the newest token of `ids` completes beyond what was already emitted; empty while
    /// the trailing character is incomplete.
    pub(crate) fn step(
        &mut self,
        tokenizer: &Tokenizer,
        ids: &[u32],
        skip_special: bool,
    ) -> String {
        let sent = tokenizer
            .decode(&ids[self.prev_index..self.current_index], skip_special)
            .unwrap_or_default();
        let text = tokenizer
            .decode(&ids[self.prev_index..], skip_special)
            .unwrap_or_default();
        if text.len() > sent.len()
            && !text.ends_with('\u{FFFD}')
            && text.is_char_boundary(sent.len())
        {
            let out = text[sent.len()..].to_string();
            self.prev_index = self.current_index;
            self.current_index = ids.len();
            out
        } else {
            String::new()
        }
    }

    /// What remains unsent at the end of the stream, an incomplete trailing sequence
    /// included, since no later token will finish it.
    pub(crate) fn finish(&self, tokenizer: &Tokenizer, ids: &[u32], skip_special: bool) -> String {
        let sent = tokenizer
            .decode(&ids[self.prev_index..self.current_index], skip_special)
            .unwrap_or_default();
        let text = tokenizer
            .decode(&ids[self.prev_index..], skip_special)
            .unwrap_or_default();
        if text.len() > sent.len() && text.is_char_boundary(sent.len()) {
            text[sent.len()..].to_string()
        } else {
            String::new()
        }
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
    /// Plain steps since a cycle with a draft last ran.
    plain_run: u32,
    /// The run of plain steps the next losing probe holds the drafter off for.
    backoff: u32,
}

/// Plain steps between the probes of a drafter whose cycles were losing, doubled at each probe
/// that loses again, up to [`DRAFT_RETRY_MAX`], and restored by a cycle that wins. Without a probe a
/// cost measured once, on a cold cache, would keep the drafter off for good.
const DRAFT_RETRY_STEPS: u32 = 48;
const DRAFT_RETRY_MAX: u32 = 1536;
/// The odds a drafted token must have while the policy has not measured what a row costs.
const DRAFT_EVEN_ODDS: f32 = 0.5;
/// Cycles a drafter is given before its acceptance is trusted against the plain step: the
/// smoothed survival of its first cycle alone undercounts a drafter that is mostly right.
const DRAFT_PROBES: u32 = 4;

impl DraftLength {
    pub(crate) fn next(&self) -> usize {
        let draft_token = self.draft_token.unwrap_or(0.0);
        // A verify covers the carried token plus the draft: `k + 1` rows.
        // A window costs the plain step plus a share per row: the share comes from the
        // nearest measured length, and the plain step is the length-zero cycle.
        let scaled_verify = |k: usize| {
            let plain = self.cycle[0].map(|(_, v)| v);
            (1..=DRAFT_MAX)
                .filter_map(|j| self.cycle[j].map(|(_, v)| (j.abs_diff(k), j, v)))
                .min_by(|a, b| a.0.cmp(&b.0))
                .map(|(_, j, v)| match plain {
                    Some(p) if v > p => p + (v - p) * k as f64 / j as f64,
                    _ => v * (k + 1) as f64 / (j + 1) as f64,
                })
                .or_else(|| plain.map(|p| p * (k + 1) as f64))
        };
        let mut best = (0, f64::MIN);
        let mut best_drafting = (0, f64::MIN);
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
            if k > 0 && rate >= best_drafting.1 {
                best_drafting = (k, rate);
            }
        }
        if best.0 == 0
            && (self.reached[0] < DRAFT_PROBES
                || self.plain_run >= self.backoff.max(DRAFT_RETRY_STEPS))
        {
            return best_drafting.0;
        }
        best.0
    }

    /// The odds a drafted token has to be given for its row to pay: a row of the window costs
    /// its share of a plain step, measured from the length-one cycles against the plain ones;
    /// even odds until both are measured.
    pub(crate) fn min_odds(&self) -> f32 {
        match (self.cycle[0], self.cycle[1]) {
            (Some((_, plain)), Some((_, two))) if plain > 0.0 && two > plain => {
                ((two - plain) / plain).clamp(0.1, 0.9) as f32
            }
            _ => DRAFT_EVEN_ODDS,
        }
    }

    /// The costs and acceptances the choice rests on, for the request's log line.
    pub(crate) fn summary(&self) -> String {
        let cycles: Vec<String> = (0..=DRAFT_MAX)
            .filter_map(|k| {
                self.cycle[k].map(|(c, v)| format!("{k}:{:.1}/{:.1}", c * 1e3, v * 1e3))
            })
            .collect();
        let kept: Vec<String> = (0..DRAFT_MAX)
            .filter(|&i| self.reached[i] > 0)
            .map(|i| format!("{}/{}", self.kept[i], self.reached[i]))
            .collect();
        format!(
            "cycle ms (cycle/verify) {}; kept {}; draft token {:.1} ms; min odds {:.2}; plain run {}, backoff {}",
            cycles.join(" "),
            kept.join(" "),
            self.draft_token.unwrap_or(0.0) * 1e3,
            self.min_odds(),
            self.plain_run,
            self.backoff
        )
    }

    /// Records what a cycle that drafted `k` tokens accepted, and nothing of what it cost: a
    /// cycle whose verify was recorded as it ran paid the recording too.
    pub(crate) fn record_accepted(&mut self, k: usize, accepted: usize) {
        let k = k.min(DRAFT_MAX);
        for i in 0..k.min(accepted + 1) {
            self.reached[i] += 1;
            if i < accepted {
                self.kept[i] += 1;
            }
        }
        if k > 0 {
            self.plain_run = 0;
        }
    }

    /// Records a cycle that drafted `k` tokens in `draft_secs`, accepted `accepted` of them,
    /// and took `cycle_secs` in all; `k` 0 is a plain decode step.
    pub(crate) fn record(&mut self, k: usize, accepted: usize, draft_secs: f64, cycle_secs: f64) {
        let k = k.min(DRAFT_MAX);
        let probe = k > 0 && self.plain_run >= self.backoff.max(DRAFT_RETRY_STEPS);
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
            self.plain_run = 0;
            if probe {
                // A probe that now wins restores the short wait; one that loses again doubles it.
                self.backoff = if self.next() > 0 {
                    0
                } else {
                    (self.backoff.max(DRAFT_RETRY_STEPS) * 2).min(DRAFT_RETRY_MAX)
                };
            }
        } else {
            self.plain_run = self.plain_run.saturating_add(1);
        }
    }
}

/// What a prompt-lookup verify costs against the plain steps it replaces, measured on this
/// request: a round is worth running while the tokens it commits per second beat the plain
/// decode's. Where a verify costs about one step per row, a round that commits fewer tokens than
/// it has rows loses, whatever share of its drafts was accepted; where a verify costs hardly more
/// than a step, nearly any acceptance wins.
#[derive(Debug, Default, Clone)]
pub(crate) struct PldCost {
    /// Seconds of a plain decode step, smoothed.
    plain: Option<f64>,
    /// Seconds and committed tokens of a verify round, smoothed.
    round: Option<(f64, f64)>,
    /// Plain steps left before a round is tried again after a losing one.
    hold: u32,
    /// The run of plain steps the next losing round holds the lookup off for.
    backoff: u32,
}

/// Plain steps between the trials of a lookup whose rounds were losing, doubled at each trial
/// that loses again, up to [`PLD_RETRY_MAX`], and restored by a round that wins.
const PLD_RETRY_STEPS: u32 = 48;
const PLD_RETRY_MAX: u32 = 1536;

impl PldCost {
    fn blend(old: Option<f64>, new: f64) -> f64 {
        match old {
            Some(o) => o + (new - o) * CYCLE_COST_WEIGHT,
            None => new,
        }
    }

    /// What the next request starts from: the costs and the length of the wait that earlier
    /// requests learned, and a trial at its first lookup hit.
    pub(crate) fn carried(&self) -> Self {
        Self {
            hold: 0,
            ..self.clone()
        }
    }

    /// A plain decode step took `secs`.
    pub(crate) fn plain_step(&mut self, secs: f64) {
        self.plain = Some(Self::blend(self.plain, secs));
        self.hold = self.hold.saturating_sub(1);
    }

    /// A verify round took `secs` and committed `tokens`.
    pub(crate) fn round(&mut self, secs: f64, tokens: usize) {
        let (s, t) = self.round.unwrap_or((secs, tokens as f64));
        self.round = Some((
            Self::blend(Some(s), secs),
            Self::blend(Some(t), tokens as f64),
        ));
        if let (Some(plain), Some((s, t))) = (self.plain, self.round) {
            let held = self.backoff.max(PLD_RETRY_STEPS);
            if t / s.max(f64::EPSILON) < plain.recip() {
                self.hold = held;
                self.backoff = (held * 2).min(PLD_RETRY_MAX);
            } else {
                self.backoff = PLD_RETRY_STEPS;
            }
        }
    }

    /// Whether the next lookup hit should be verified.
    pub(crate) fn worth_trying(&self) -> bool {
        self.hold == 0
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
    /// A replay that picks the other of two logits the eager step has within the numerical
    /// noise between them agrees with it; one whose logits are not the eager step's does not.
    #[test]
    fn a_replay_agrees_through_a_near_tie_and_not_through_a_fault() {
        let eager: Vec<f32> = (0..1000).map(|i| (i % 17) as f32 - 8.0).collect();
        let mut top_two = eager.clone();
        top_two[3] = 20.0;
        top_two[4] = 19.999;
        let mut flipped = top_two.clone();
        flipped[3] = 19.998;
        flipped[4] = 20.001;
        assert!(replay_agrees(&flipped, &top_two));
        assert!(replay_agrees(&top_two, &top_two));
        let mut wrong = top_two.clone();
        wrong[3] = 19.0;
        wrong[4] = 20.0;
        wrong.iter_mut().step_by(2).for_each(|v| *v += 5.0);
        assert!(!replay_agrees(&wrong, &top_two));
        let mut far = top_two.clone();
        far[3] = 5.0;
        far[4] = 20.0;
        assert!(
            !replay_agrees(&far, &top_two),
            "a gap larger than the noise"
        );
    }

    use super::*;

    /// A request starts from what the earlier ones learned: the costs and the longer wait, with a
    /// trial at its first lookup hit.
    #[test]
    fn a_request_starts_from_the_cost_the_earlier_ones_learned() {
        let mut c = PldCost::default();
        c.plain_step(0.040);
        for _ in 0..12 {
            if c.worth_trying() {
                c.round(0.180, 2);
            }
        }
        assert!(!c.worth_trying());
        let next = c.carried();
        assert!(next.worth_trying(), "a trial at the first hit");
        let mut next = next;
        next.round(0.180, 2);
        assert!(!next.worth_trying());
        for _ in 0..PLD_RETRY_STEPS {
            next.plain_step(0.040);
        }
        assert!(
            !next.worth_trying(),
            "the wait grew: the earlier request lost too"
        );
    }

    /// A lookup is verified while its rounds commit tokens faster than plain steps do, held off
    /// for a run of plain steps once they do not, and tried again after it.
    #[test]
    fn a_lookup_is_verified_only_while_its_rounds_beat_the_plain_steps() {
        let mut c = PldCost::default();
        assert!(c.worth_trying(), "nothing measured: try");
        c.plain_step(0.040);
        c.round(0.180, 6);
        assert!(c.worth_trying(), "33 tokens a second against 25");
        // Rounds that commit 2 tokens in 0.18 s lose: 11 tokens a second against 25. The lookup
        // is held off for a run of plain steps, and each trial that loses holds it off twice as
        // long as the last.
        let mut held = PLD_RETRY_STEPS;
        for trial in 0..3 {
            // The first losses have to outweigh the round that won: the cost is smoothed.
            let rounds = if trial == 0 { 12 } else { 1 };
            for _ in 0..rounds {
                if c.worth_trying() {
                    c.round(0.180, 2);
                }
            }
            assert!(!c.worth_trying());
            for _ in 0..held - 1 {
                c.plain_step(0.040);
                assert!(!c.worth_trying());
            }
            c.plain_step(0.040);
            assert!(c.worth_trying(), "tried again after the run of plain steps");
            held *= 2;
        }
        // A round that wins restores the short wait.
        for _ in 0..12 {
            c.round(0.100, 8);
        }
        for _ in 0..12 {
            if c.worth_trying() {
                c.round(0.180, 2);
            }
        }
        assert!(!c.worth_trying());
        for _ in 0..PLD_RETRY_STEPS {
            c.plain_step(0.040);
        }
        assert!(c.worth_trying(), "the wait is the short one again");
    }

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
        // It still probes now and then, with a wait that doubles when the probe loses.
        let mut probes = 0;
        for _ in 0..400u32 {
            let k = d.next();
            if k == 0 {
                d.record(0, 0, 0.0, 0.2);
            } else {
                probes += 1;
                d.record(k, 0, 0.01 * k as f64, 1.0 + 0.01 * k as f64);
            }
        }
        assert!((1..=4).contains(&probes), "probed {probes} times");
    }

    #[test]
    fn a_drafter_whose_cost_was_measured_cold_is_tried_again() {
        let mut d = DraftLength::default();
        // The probe cycles at length 3 cost five plain steps each; then plain steps win.
        for _ in 0..DRAFT_PROBES {
            d.record(3, 1, 0.01, 1.0);
        }
        for _ in 0..40 {
            d.record(0, 0, 0.0, 0.2);
        }
        assert_eq!(d.next(), 0, "plain wins for now");
        for _ in 0..DRAFT_RETRY_STEPS {
            d.record(0, 0, 0.0, 0.2);
        }
        assert!(d.next() > 0, "a probe after the run of plain steps");
        // A probe that finds the cycle cheap now keeps the drafter on.
        for _ in 0..12 {
            let k = d.next().max(1);
            d.record(k, k, 0.01, 0.25);
        }
        assert!(d.next() > 0);
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

/// Whether the logits of a replayed graph step agree with the eager step over the same token:
/// the same top token, or two logit vectors the same to within [`REPLAY_RELATIVE_ERROR`] of the
/// eager norm (an expert computed on the card in one run and on the host in the other differs
/// in its last bits, which a near tie flips on). A replay that is wrong fails the second.
pub(crate) fn replay_agrees(graph: &[f32], eager: &[f32]) -> bool {
    if graph.len() != eager.len() || eager.is_empty() {
        return false;
    }
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                if x > b.1 {
                    (i, x)
                } else {
                    b
                }
            })
            .0
    };
    top(graph) == top(eager) || replay_distance(graph, eager) <= REPLAY_RELATIVE_ERROR
}

/// The distance between a replay's logits and the eager step's, relative to the eager norm.
pub(crate) fn replay_distance(graph: &[f32], eager: &[f32]) -> f64 {
    let (mut diff2, mut norm2) = (0f64, 0f64);
    for (g, e) in graph.iter().zip(eager) {
        let d = f64::from(g - e);
        diff2 += d * d;
        norm2 += f64::from(*e) * f64::from(*e);
    }
    diff2.sqrt() / norm2.sqrt().max(f64::EPSILON)
}

/// The relative distance between a replay's logits and the eager step's, from which a replay is
/// not trusted. Between two runs of the same rows an expert may have moved between the host
/// and the card, whose products differ in their last bits: the drift measured is three to
/// five percent of the norm; a fault moves one logit by a tenth of it or more.
const REPLAY_RELATIVE_ERROR: f64 = 5e-2;
