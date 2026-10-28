//! Beam search on the sequential path: the best continuations of a prompt by cumulative
//! log-probability, scored and stopped as transformers' `generate(num_beams=...)` scores and
//! stops them.
//!
//! Each step reads the log-softmax of the raw logits after every running beam. All their
//! continuations are ranked by cumulative log-probability. Of the first `width`, those that end
//! on an end token, and on the last step every one of them, become finished hypotheses, scored
//! by their cumulative log-probability over their generated length, the end token counted,
//! raised to `length_penalty`. The first `width` that do not end are the next step's beams.
//! Ties rank the earlier beam, then the lower token id, first; a finished hypothesis keeps its
//! place ahead of a later one of the same score.
//!
//! The model holds one KV cache. The beams share the prompt's rows. A beam's own rows are
//! written by rolling the cache back to what the resident sequence shares with the beam and
//! feeding the rest one token at a time, the forwards a plain decode of that beam runs. The
//! beams are visited in token order, so the children of one beam follow each other and each
//! costs one forward.

use super::params::{FinishReason, TokenLogprob};
use super::*;
use crate::inference::engine::model_backend::ModelBackend;

/// When a beam search stops before its token cap, as transformers' `early_stopping` says.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EarlyStopping {
    /// `false`: once `width` hypotheses are finished, stops when the best running beam, scored
    /// at its current length, does not beat the worst of them.
    #[default]
    Heuristic,
    /// `true`: stops as soon as `width` hypotheses are finished.
    Done,
    /// `"never"`: as `Heuristic`, the running beam scored at the token cap when the length
    /// penalty is positive.
    Never,
}

/// What a beam search is asked for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamSearch {
    /// Beams kept at each step, and hypotheses answered.
    pub width: usize,
    /// The exponent of the generated length a hypothesis' cumulative log-probability is
    /// divided by: above 0 favours longer hypotheses, below 0 shorter ones.
    pub length_penalty: f64,
    pub early_stopping: EarlyStopping,
}

/// One token of a hypothesis: its log-probability and the most probable alternatives.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct BeamStep {
    pub token: u32,
    pub logprob: f64,
    pub top: Vec<(u32, f64)>,
}

/// A sequence of generated tokens and the sum of their log-probabilities.
#[derive(Debug, Clone, Default)]
pub(crate) struct Hypothesis {
    pub tokens: Vec<u32>,
    pub steps: Vec<BeamStep>,
    pub cumulative: f64,
}

/// A hypothesis that stopped: its score, and whether its last token is an end token.
#[derive(Debug, Clone)]
pub(crate) struct Finished {
    pub hypothesis: Hypothesis,
    pub score: f64,
    pub ended: bool,
}

/// A continuation of a running beam.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    cumulative: f64,
    beam: usize,
    token: u32,
    logprob: f64,
}

/// Ranks candidates best first: higher cumulative log-probability, then earlier beam, then
/// lower token id.
fn rank(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.cumulative
        .total_cmp(&a.cumulative)
        .then(a.beam.cmp(&b.beam))
        .then(a.token.cmp(&b.token))
}

/// The log-softmax of a row of logits, in f64; a NaN logit is an impossible token.
pub(crate) fn log_softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits
        .iter()
        .filter(|x| !x.is_nan())
        .fold(f64::NEG_INFINITY, |m, &x| m.max(f64::from(x)));
    if !max.is_finite() {
        return vec![f64::NEG_INFINITY; logits.len()];
    }
    let sum: f64 = logits
        .iter()
        .filter(|x| !x.is_nan())
        .map(|&x| (f64::from(x) - max).exp())
        .sum();
    let lse = max + sum.ln();
    logits
        .iter()
        .map(|&x| {
            if x.is_nan() {
                f64::NEG_INFINITY
            } else {
                f64::from(x) - lse
            }
        })
        .collect()
}

/// The `k` tokens of `row` whose `offset + logprob` ranks first, best first, ties to the lower
/// id: the order [`rank`] puts one beam's candidates in.
fn top_tokens(row: &[f64], k: usize, offset: f64) -> Vec<(u32, f64)> {
    let key = |i: u32| offset + row[i as usize];
    let order = |a: &u32, b: &u32| key(*b).total_cmp(&key(*a)).then(a.cmp(b));
    let mut ids: Vec<u32> = (0..row.len() as u32).collect();
    let k = k.min(ids.len());
    if k == 0 {
        return Vec::new();
    }
    if k < ids.len() {
        ids.select_nth_unstable_by(k - 1, order);
        ids.truncate(k);
    }
    ids.sort_unstable_by(order);
    ids.into_iter().map(|i| (i, row[i as usize])).collect()
}

/// The state of one beam search, apart from the model: the running beams, the finished
/// hypotheses, best first, and the step count.
pub(crate) struct BeamSearcher {
    search: BeamSearch,
    ends: Vec<u32>,
    max_new: usize,
    top_n: usize,
    running: Vec<Hypothesis>,
    finished: Vec<Finished>,
    step: usize,
}

impl BeamSearcher {
    /// A search from the prompt alone, ended by `ends`, at most `max_new` tokens long,
    /// recording `top_n` alternatives per token.
    pub fn new(search: BeamSearch, ends: Vec<u32>, max_new: usize, top_n: usize) -> Self {
        Self {
            search,
            ends,
            max_new,
            top_n,
            running: vec![Hypothesis::default()],
            finished: Vec::new(),
            step: 0,
        }
    }

    /// The beams the next step continues, best first.
    pub fn running(&self) -> &[Hypothesis] {
        &self.running
    }

    /// The divisor of a cumulative log-probability over `len` generated tokens.
    fn norm(&self, len: usize) -> f64 {
        (len.max(1) as f64).powf(self.search.length_penalty)
    }

    /// Keeps `f` among the `width` best finished hypotheses, after those of the same score.
    fn finish(&mut self, f: Finished) {
        let at = self
            .finished
            .iter()
            .position(|o| o.score < f.score)
            .unwrap_or(self.finished.len());
        self.finished.insert(at, f);
        self.finished.truncate(self.search.width);
    }

    /// Advances by one token, `rows[i]` the log-probabilities of the token after
    /// `running()[i]`. False once the search is over.
    pub fn advance(&mut self, rows: &[Vec<f64>]) -> bool {
        let width = self.search.width;
        self.step += 1;
        let last = self.step >= self.max_new;
        // A beam's best `width` continuations that do not end are among its best
        // `width + ends` ones.
        let per_beam = width + self.ends.len();
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut tops: Vec<Vec<(u32, f64)>> = Vec::with_capacity(rows.len());
        for (beam, (hyp, row)) in self.running.iter().zip(rows).enumerate() {
            candidates.extend(top_tokens(row, per_beam, hyp.cumulative).into_iter().map(
                |(token, logprob)| Candidate {
                    cumulative: hyp.cumulative + logprob,
                    beam,
                    token,
                    logprob,
                },
            ));
            tops.push(if self.top_n > 0 {
                top_tokens(row, self.top_n, 0.0)
            } else {
                Vec::new()
            });
        }
        candidates.sort_by(rank);

        let norm = self.norm(self.step);
        let mut next: Vec<Hypothesis> = Vec::with_capacity(width);
        let mut done: Vec<Finished> = Vec::new();
        for (place, c) in candidates.iter().enumerate() {
            if place >= width && (last || next.len() >= width) {
                break;
            }
            let ended = self.ends.contains(&c.token);
            let stops = ended || last;
            if (stops && place >= width) || (!stops && next.len() >= width) {
                continue;
            }
            let parent = &self.running[c.beam];
            let mut hyp = parent.clone();
            hyp.tokens.push(c.token);
            hyp.steps.push(BeamStep {
                token: c.token,
                logprob: c.logprob,
                top: tops[c.beam].clone(),
            });
            hyp.cumulative = c.cumulative;
            if stops {
                done.push(Finished {
                    score: c.cumulative / norm,
                    hypothesis: hyp,
                    ended,
                });
            } else {
                next.push(hyp);
            }
        }
        for f in done {
            self.finish(f);
        }
        self.running = next;

        if last || self.running.is_empty() {
            return false;
        }
        if self.finished.len() < width {
            return true;
        }
        let lp = self.search.length_penalty;
        let hypothetical = match self.search.early_stopping {
            EarlyStopping::Done => return false,
            EarlyStopping::Never if lp > 0.0 => self.max_new,
            EarlyStopping::Heuristic | EarlyStopping::Never => self.step,
        };
        let best_possible = self.running[0].cumulative / self.norm(hypothetical);
        let worst = self.finished.last().map_or(f64::NEG_INFINITY, |f| f.score);
        best_possible > worst
    }

    /// The finished hypotheses, best first, at most `width`. A search stopped before its end
    /// finishes its running beams where they are.
    pub fn best(mut self) -> Vec<Finished> {
        let running = std::mem::take(&mut self.running);
        if self.finished.len() < self.search.width {
            for hyp in running {
                let score = hyp.cumulative / self.norm(hyp.tokens.len());
                self.finish(Finished {
                    hypothesis: hyp,
                    score,
                    ended: false,
                });
            }
        }
        self.finished
    }
}

/// One beam a search answers.
#[derive(Debug, Clone)]
pub struct BeamAnswer {
    /// The generated tokens, the end token it stopped on left out.
    pub tokens: Vec<u32>,
    /// The text of `tokens`.
    pub text: String,
    /// Per generated token, the end token included, when the request asked for them.
    pub logprobs: Vec<TokenLogprob>,
    /// The sum of the log-probabilities of every generated token, the end token included.
    pub cumulative_logprob: f64,
    /// The cumulative log-probability over the length penalty: what the beams are ranked by.
    pub score: f64,
    pub finish_reason: FinishReason,
}

/// The answer of a beam search: its beams, best first, and the figures of the run.
#[derive(Debug, Clone)]
pub struct BeamOutcome {
    pub beams: Vec<BeamAnswer>,
    pub prompt_eval_count: u64,
    pub prompt_eval_duration: u64,
    /// Forwards run after the prefill, the replays of a beam's rows included.
    pub forwards: u64,
    pub eval_duration: u64,
}

/// The logits of `token` at `pos`, the cache holding the `pos` positions before it.
fn feed(
    model: &mut dyn ModelBackend,
    device: &Device,
    pos: usize,
    token: u32,
) -> AnyResult<Tensor> {
    let x = Tensor::new(&[token], device)?.unsqueeze(0)?;
    Ok(model.forward(&x, pos)?)
}

/// A row of logits on the host, in f32.
fn host_row(logits: &Tensor) -> AnyResult<Vec<f32>> {
    Ok(logits
        .to_device(&Device::Cpu)?
        .squeeze(0)?
        .to_dtype(crate::tensor::DType::F32)?
        .to_vec1::<f32>()?)
}

/// The length of the run two sequences share from their start.
fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Rolls the cache back to `len` positions; an error when the model does not say it holds
/// exactly that many.
fn roll_back(model: &mut dyn ModelBackend, len: usize) -> AnyResult<()> {
    model.reset_kv_from(len);
    match model.kv_len() {
        Some(held) if held == len => Ok(()),
        held => Err(anyhow!(
            "beam search: the KV cache holds {held:?} positions after a roll back to {len}"
        )),
    }
}

/// The log-probabilities of the token after each beam, in the beams' order. `resident` is
/// the generated sequence the cache holds after the `base` prompt positions; the beams are
/// visited in token order.
fn beam_rows(
    model: &mut dyn ModelBackend,
    device: &Device,
    base: usize,
    resident: &mut Vec<u32>,
    beams: &[Hypothesis],
    forwards: &mut u64,
) -> AnyResult<Vec<Vec<f64>>> {
    let mut order: Vec<usize> = (0..beams.len()).collect();
    order.sort_by(|&a, &b| beams[a].tokens.cmp(&beams[b].tokens));
    let mut rows = vec![Vec::new(); beams.len()];
    for i in order {
        let Some((&last, before)) = beams[i].tokens.split_last() else {
            return Err(anyhow!("beam search: a running beam holds no token"));
        };
        let keep = common_prefix(resident, before);
        if keep < resident.len() {
            roll_back(model, base + keep)?;
            resident.truncate(keep);
        }
        for &t in &before[keep..] {
            feed(model, device, base + resident.len(), t)?;
            resident.push(t);
            *forwards += 1;
        }
        let logits = feed(model, device, base + resident.len(), last)?;
        resident.push(last);
        *forwards += 1;
        rows[i] = log_softmax(&host_row(&logits)?);
    }
    Ok(rows)
}

/// One beam search, under the model lock.
fn decode_beams(
    state: &mut LoadedModelState,
    config: &InferenceConfig,
    prompt: &str,
    params: GenerationParams,
    search: BeamSearch,
    cancel: &std::sync::atomic::AtomicBool,
) -> AnyResult<BeamOutcome> {
    if state.audio.is_some() {
        return Err(anyhow!("a beam search cannot carry audio"));
    }
    if !state.model.kv_rolls_back() {
        return Err(anyhow!(
            "beam search needs a model whose KV cache can be cut back: no recurrent or \
             sliding-window layer"
        ));
    }
    if search.width == 0 {
        return Err(anyhow!("beam search needs a width of at least 1"));
    }
    let encoding = super::encode_prompt(&state.tokenizer, prompt)
        .map_err(|e| anyhow!("Tokenization error: {e}"))?;
    let kv_window = effective_kv_window(
        state.context_length,
        config.context_length,
        params.context_length,
    );
    let prompt_tokens = clamp_prompt_to_window(
        request_prompt_tokens(&params, encoding.get_ids()),
        kv_window,
        params.max_tokens.unwrap_or(0),
    );
    let max_tokens = resolve_gen_params(
        &params,
        &state.generation_defaults,
        kv_window,
        prompt_tokens.len(),
    )
    .max_tokens;
    let ends = params
        .rules
        .end_tokens(state.eos_token_id, &state.eos_token_ids_extra);
    let skip_special = !params.rules.keep_special_tokens;
    let device = state.device.clone();
    state.model.set_early_exit_threshold(None);

    // The graphs kept from the last request go, with the buffers they read: the single-token
    // forwards below would read their frozen rope values.
    #[cfg(feature = "gpu")]
    state.kept_graphs.clear();
    state.model.note_graphs_kept(None);
    state.model.invalidate_graph_state();
    // The prompt is written from the first row, so nothing the cache held stays usable.
    state.exact_prefill = None;
    state.model.reset_kv_from(0);

    let prefill_start = std::time::Instant::now();
    let x = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let first = state.model.forward_prefill_chunked(&x, 0)?;
    let prompt_eval_duration = prefill_start.elapsed().as_nanos() as u64;
    let base = prompt_tokens.len();
    if state.model.kv_len() != Some(base) {
        return Err(anyhow!(
            "beam search needs a model whose KV cache says how many positions it holds"
        ));
    }

    let mut searcher =
        BeamSearcher::new(search, ends, max_tokens, params.top_logprobs.unwrap_or(0));
    let mut rows = vec![log_softmax(&host_row(&first)?)];
    let mut resident: Vec<u32> = Vec::new();
    let mut forwards: u64 = 0;
    let mut cancelled = false;
    let decode_start = std::time::Instant::now();
    if max_tokens > 0 {
        while searcher.advance(&rows) {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                cancelled = true;
                break;
            }
            rows = beam_rows(
                &mut *state.model,
                &device,
                base,
                &mut resident,
                searcher.running(),
                &mut forwards,
            )?;
        }
    }
    let eval_duration = decode_start.elapsed().as_nanos() as u64;

    let piece = |id: u32| state.tokenizer.decode(&[id], false).unwrap_or_default();
    let mut beams = Vec::new();
    for f in searcher.best() {
        let Finished {
            hypothesis,
            score,
            ended,
        } = f;
        let mut tokens = hypothesis.tokens;
        if ended {
            tokens.pop();
        }
        let text = state
            .tokenizer
            .decode(&tokens, skip_special)
            .map_err(|e| anyhow!("decode: {e}"))?;
        let logprobs = if params.top_logprobs.is_some() {
            hypothesis
                .steps
                .iter()
                .map(|s| TokenLogprob {
                    token: s.token,
                    text: piece(s.token),
                    logprob: s.logprob as f32,
                    top: s
                        .top
                        .iter()
                        .map(|&(id, l)| (id, piece(id), l as f32))
                        .collect(),
                })
                .collect()
        } else {
            Vec::new()
        };
        let finish_reason = if ended {
            FinishReason::Eos
        } else if cancelled {
            FinishReason::Disconnect
        } else {
            FinishReason::MaxTokens
        };
        beams.push(BeamAnswer {
            tokens,
            text,
            logprobs,
            cumulative_logprob: hypothesis.cumulative,
            score,
            finish_reason,
        });
    }
    Ok(BeamOutcome {
        beams,
        prompt_eval_count: base as u64,
        prompt_eval_duration,
        forwards,
        eval_duration,
    })
}

impl LlmEngine {
    /// The `search.width` best beams of a prompt, best first, on the sequential path. The
    /// request's end tokens end a beam; its sampling knobs, stop sequences and token rules
    /// are not read.
    pub async fn generate_beams(
        &self,
        prompt: &str,
        params: GenerationParams,
        search: BeamSearch,
    ) -> Result<BeamOutcome, Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let sessions = self.sessions.clone();
        let config = self.config.clone();
        let cancel = params.cancel.clone().unwrap_or_default();
        let prompt = prompt.to_string();
        let arrived = std::time::Instant::now();
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<BeamOutcome> {
            let mut guard = model_state.blocking_lock();
            let queue_s = arrived.elapsed().as_secs_f64();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Model not loaded. Load a model first."))?;
            sessions.blocking_lock().clear();
            let out = decode_beams(state, &config, &prompt, params, search, &cancel)?;
            crate::request_metrics::record(
                &config.model_id,
                &crate::request_metrics::RequestFigures {
                    queue_s,
                    prefill_s: out.prompt_eval_duration as f64 / 1e9,
                    decode_s: out.eval_duration as f64 / 1e9,
                    prompt_tokens: out.prompt_eval_count,
                    cached_prompt_tokens: 0,
                    generated_tokens: out.forwards,
                },
            );
            Ok(out)
        })
        .await;
        match inner {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EOS: u32 = 3;

    fn search(width: usize, length_penalty: f64, early_stopping: EarlyStopping) -> BeamSearch {
        BeamSearch {
            width,
            length_penalty,
            early_stopping,
        }
    }

    /// A row of log-probabilities from probabilities.
    fn row(probs: &[f64]) -> Vec<f64> {
        probs.iter().map(|p| p.ln()).collect()
    }

    /// A toy model: the next-token distribution read from the tokens so far.
    type Toy = fn(&[u32]) -> Vec<f64>;

    /// Runs a search over a toy model, as the engine runs it over the real one.
    fn run(s: BeamSearch, max_new: usize, model: Toy) -> Vec<Finished> {
        let mut searcher = BeamSearcher::new(s, vec![EOS], max_new, 0);
        let mut rows = vec![model(&[])];
        while searcher.advance(&rows) {
            rows = searcher
                .running()
                .iter()
                .map(|h| model(&h.tokens))
                .collect();
        }
        searcher.best()
    }

    fn tokens(f: &[Finished]) -> Vec<Vec<u32>> {
        f.iter().map(|f| f.hypothesis.tokens.clone()).collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn log_softmax_sums_to_one_and_drops_nan() {
        let lp = log_softmax(&[1.0, 2.0, 3.0, f32::NAN]);
        let total: f64 = lp.iter().map(|l| l.exp()).sum();
        assert!(close(total, 1.0));
        assert_eq!(lp[3], f64::NEG_INFINITY);
        assert!(close(lp[2] - lp[1], 1.0));
    }

    #[test]
    fn the_top_tokens_rank_by_logprob_then_lower_id() {
        let r = vec![-1.0, -0.5, -0.5, -3.0];
        assert_eq!(
            top_tokens(&r, 3, 0.0),
            vec![(1, -0.5), (2, -0.5), (0, -1.0)]
        );
        assert_eq!(top_tokens(&r, 9, 0.0).len(), 4);
    }

    /// Greedy would take 0 then 1 (0.5 * 0.4 = 0.2); the beam finds 1 then 0 (0.4 * 0.9 = 0.36).
    #[test]
    fn a_beam_finds_what_greedy_misses() {
        fn model(t: &[u32]) -> Vec<f64> {
            match t {
                [] => row(&[0.5, 0.4, 0.05, 0.05]),
                [0] => row(&[0.3, 0.4, 0.3, 1e-9]),
                [1] => row(&[0.9, 0.05, 0.05, 1e-9]),
                _ => row(&[0.25, 0.25, 0.25, 0.25]),
            }
        }
        let best = run(search(2, 0.0, EarlyStopping::Heuristic), 2, model);
        assert_eq!(tokens(&best), vec![vec![1, 0], vec![0, 1]]);
        assert!(close(best[0].hypothesis.cumulative, (0.4f64 * 0.9).ln()));
        // length_penalty 0 scores by the cumulative log-probability itself.
        assert!(close(best[0].score, best[0].hypothesis.cumulative));
        assert!(best.iter().all(|f| !f.ended));
    }

    /// The end token is counted in the length and in the cumulative log-probability, and the
    /// length penalty divides by the length raised to it.
    #[test]
    fn the_score_counts_the_end_token() {
        fn model(t: &[u32]) -> Vec<f64> {
            match t {
                [] => row(&[0.6, 0.3, 0.05, 0.05]),
                [0] => row(&[0.1, 0.1, 0.1, 0.7]),
                _ => row(&[0.4, 0.4, 0.1, 0.1]),
            }
        }
        let lp = 2.0;
        let best = run(search(1, lp, EarlyStopping::Heuristic), 4, model);
        let f = &best[0];
        assert_eq!(f.hypothesis.tokens, vec![0, EOS]);
        assert!(f.ended);
        let cum = (0.6f64 * 0.7).ln();
        assert!(close(f.hypothesis.cumulative, cum));
        assert!(close(f.score, cum / 2f64.powf(lp)));
    }

    /// Only the first `width` candidates may finish: an end token ranked past them is dropped,
    /// and the running beams are the first `width` candidates that do not end.
    #[test]
    fn an_end_token_finishes_only_within_the_first_width() {
        let s = search(2, 1.0, EarlyStopping::Heuristic);
        let mut searcher = BeamSearcher::new(s, vec![EOS], 5, 0);
        // Candidates: 0 (0.4), 1 (0.3), EOS (0.2), 2 (0.1).
        assert!(searcher.advance(&[row(&[0.4, 0.3, 0.1, 0.2])]));
        assert!(searcher.finished.is_empty());
        let running: Vec<_> = searcher
            .running()
            .iter()
            .map(|h| h.tokens.clone())
            .collect();
        assert_eq!(running, vec![vec![0], vec![1]]);
        // Now EOS after [0] ranks first, and EOS after [1] ranks past the width.
        let rows = [row(&[0.2, 0.1, 0.1, 0.6]), row(&[0.5, 0.2, 0.1, 0.2])];
        searcher.advance(&rows);
        assert_eq!(searcher.finished.len(), 1);
        assert_eq!(searcher.finished[0].hypothesis.tokens, vec![0, EOS]);
        let running: Vec<_> = searcher
            .running()
            .iter()
            .map(|h| h.tokens.clone())
            .collect();
        assert_eq!(running, vec![vec![1, 0], vec![0, 0]]);
    }

    /// The last step finishes the first `width` candidates whatever their token, scored at the
    /// cap.
    #[test]
    fn the_last_step_finishes_the_first_width() {
        fn model(_: &[u32]) -> Vec<f64> {
            row(&[0.5, 0.3, 0.15, 0.05])
        }
        let best = run(search(2, 1.0, EarlyStopping::Heuristic), 3, model);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].hypothesis.tokens, vec![0, 0, 0]);
        let cum = best[0].hypothesis.cumulative;
        assert!((cum - 3.0 * 0.5f64.ln()).abs() < 1e-12);
        assert!(close(best[0].score, cum / 3.0));
        // The second is one of the two hypotheses of probability 0.5 * 0.5 * 0.3.
        let second = &best[1].hypothesis.tokens;
        assert_eq!(second.len(), 3);
        assert_eq!(second.iter().filter(|&&t| t == 1).count(), 1);
        assert!(best.iter().all(|f| !f.ended));
    }

    /// `early_stopping: true` stops once `width` hypotheses are finished; the heuristic runs on
    /// while the best running beam may still beat the worst of them.
    #[test]
    fn early_stopping_follows_transformers() {
        // After [0], EOS (0.5) edges out 0 (0.45); after [0, 0], 0 runs on at 0.99 a token.
        fn model(t: &[u32]) -> Vec<f64> {
            match t {
                [] => row(&[0.6, 0.35, 0.025, 0.025]),
                [0] => row(&[0.45, 0.025, 0.025, 0.5]),
                _ => row(&[0.99, 0.004, 0.003, 0.003]),
            }
        }
        let done = run(search(1, 1.0, EarlyStopping::Done), 8, model);
        assert_eq!(done[0].hypothesis.tokens, vec![0, EOS]);
        // Heuristic, length_penalty 1: [0, EOS] scores ln(0.6 * 0.5) / 2 = -0.602; the
        // running [0, 0] scores ln(0.6 * 0.45) / 2 = -0.655 at its length and cannot beat it.
        let heuristic = run(search(1, 1.0, EarlyStopping::Heuristic), 8, model);
        assert_eq!(heuristic[0].hypothesis.tokens, vec![0, EOS]);
        // "never" with a positive penalty scores the running beam at the cap of 8 tokens:
        // ln(0.27) / 8 = -0.164 beats -0.602, so the beam runs to the cap, where
        // ln(0.27 * 0.99^6) / 8 = -0.171 wins.
        let never = run(search(1, 1.0, EarlyStopping::Never), 8, model);
        assert_eq!(never[0].hypothesis.tokens, vec![0; 8]);
        assert!(!never[0].ended);
        assert!((never[0].score - (0.27f64 * 0.99f64.powi(6)).ln() / 8.0).abs() < 1e-9);
    }

    /// Equal cumulative log-probabilities rank the earlier beam, then the lower token, first.
    #[test]
    fn ties_rank_the_earlier_beam_then_the_lower_token() {
        let s = search(2, 1.0, EarlyStopping::Heuristic);
        let mut searcher = BeamSearcher::new(s, vec![EOS], 4, 0);
        assert!(searcher.advance(&[row(&[0.4, 0.4, 0.1, 0.1])]));
        let running: Vec<_> = searcher
            .running()
            .iter()
            .map(|h| h.tokens.clone())
            .collect();
        assert_eq!(running, vec![vec![0], vec![1]]);
        let flat = row(&[0.3, 0.3, 0.3, 0.1]);
        searcher.advance(&[flat.clone(), flat]);
        let running: Vec<_> = searcher
            .running()
            .iter()
            .map(|h| h.tokens.clone())
            .collect();
        assert_eq!(running, vec![vec![0, 0], vec![0, 1]]);
    }

    /// A search whose end tokens are none, as `ignore_eos` asks, runs every beam to the cap.
    #[test]
    fn without_end_tokens_every_beam_runs_to_the_cap() {
        let s = search(2, 1.0, EarlyStopping::Done);
        let mut searcher = BeamSearcher::new(s, Vec::new(), 3, 0);
        let eos_heavy = row(&[0.1, 0.1, 0.1, 0.7]);
        let mut rows = vec![eos_heavy.clone()];
        while searcher.advance(&rows) {
            rows = vec![eos_heavy.clone(); searcher.running().len()];
        }
        let best = searcher.best();
        assert_eq!(best[0].hypothesis.tokens, vec![EOS, EOS, EOS]);
        assert!(best
            .iter()
            .all(|f| f.hypothesis.tokens.len() == 3 && !f.ended));
    }

    /// Each token records its log-probability and the alternatives asked for.
    #[test]
    fn each_step_records_its_logprob_and_alternatives() {
        let s = search(1, 1.0, EarlyStopping::Heuristic);
        let mut searcher = BeamSearcher::new(s, vec![EOS], 1, 2);
        searcher.advance(&[row(&[0.2, 0.5, 0.25, 0.05])]);
        let best = searcher.best();
        let step = &best[0].hypothesis.steps[0];
        assert_eq!(step.token, 1);
        assert!(close(step.logprob, 0.5f64.ln()));
        assert_eq!(step.top.iter().map(|t| t.0).collect::<Vec<_>>(), vec![1, 2]);
    }

    /// transformers' `_beam_search` for one prompt, run as its tensors run it: the first
    /// `max(2, 1 + ends) * width` of all continuations, the running and the finished beams picked
    /// by `topk` under `-1e9` masks, the early-stop heuristic and the loop's three conditions.
    /// The finished hypotheses, best first, with their scores.
    fn transformers_search(
        s: BeamSearch,
        ends: &[u32],
        max_new: usize,
        model: &dyn Fn(&[u32]) -> Vec<f64>,
    ) -> Vec<(Vec<u32>, f64)> {
        const MASK: f64 = -1.0e9;
        let (width, lp) = (s.width, s.length_penalty);
        let keep = 2.max(1 + ends.len()) * width;
        // `torch.topk`, best first; equal values keep the lower index first.
        let topk = |v: &[f64], k: usize| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
            idx.truncate(k);
            idx
        };
        let mut running: Vec<(Vec<u32>, f64)> = (0..width)
            .map(|b| (Vec::new(), if b == 0 { 0.0 } else { MASK }))
            .collect();
        let mut finished: Vec<(Vec<u32>, f64, bool)> = vec![(Vec::new(), MASK, false); width];
        let mut improvable = true;
        let mut cur = 0;
        loop {
            let mut vocab = 0;
            let mut flat = Vec::new();
            for (seq, score) in &running {
                let row = model(seq);
                vocab = row.len();
                flat.extend(row.iter().map(|l| score + l));
            }
            let candidates: Vec<(Vec<u32>, f64, bool)> = topk(&flat, keep)
                .into_iter()
                .map(|i| {
                    let token = (i % vocab) as u32;
                    let mut seq = running[i / vocab].0.clone();
                    seq.push(token);
                    let hits = ends.contains(&token) || cur + 1 >= max_new;
                    (seq, flat[i], hits)
                })
                .collect();
            let masked: Vec<f64> = candidates
                .iter()
                .map(|c| if c.2 { c.1 + MASK } else { c.1 })
                .collect();
            let next: Vec<(Vec<u32>, f64)> = topk(&masked, width)
                .into_iter()
                .map(|i| (candidates[i].0.clone(), masked[i]))
                .collect();
            let full = s.early_stopping == EarlyStopping::Done && finished.iter().all(|f| f.2);
            let norm = ((cur + 1) as f64).powf(lp);
            let mut merged = finished.clone();
            for (i, c) in candidates.iter().enumerate() {
                let just = c.2 && i < width;
                let mut score = c.1 / norm;
                for masked_out in [full, !improvable, !just] {
                    if masked_out {
                        score += MASK;
                    }
                }
                merged.push((c.0.clone(), score, just));
            }
            let scores: Vec<f64> = merged.iter().map(|m| m.1).collect();
            finished = topk(&scores, width)
                .into_iter()
                .map(|i| merged[i].clone())
                .collect();
            running = next;
            cur += 1;
            let hypothetical = if s.early_stopping == EarlyStopping::Never && lp > 0.0 {
                max_new
            } else {
                cur
            };
            let best = running[0].1 / (hypothetical as f64).powf(lp);
            // Per slot: the worst finished score where the slot is finished, the mask elsewhere.
            let worst = finished.iter().map(|f| f.1).fold(f64::INFINITY, f64::min);
            let bars: Vec<f64> = finished
                .iter()
                .map(|f| if f.2 { worst } else { MASK })
                .collect();
            improvable &= bars.iter().any(|&bar| best > bar);
            let open = !(finished.iter().all(|f| f.2) && s.early_stopping == EarlyStopping::Done);
            let valid = !candidates.iter().all(|c| c.2);
            if !(improvable && open && valid) {
                break;
            }
        }
        finished.into_iter().map(|f| (f.0, f.1)).collect()
    }

    /// Tokens of the toy model the searches are compared on.
    const TOY_VOCAB: usize = 8;

    /// A toy model: logits drawn from `seed` and the tokens so far, the end token `EOS` the more
    /// likely the longer the sequence.
    fn toy(seed: u64, tokens: &[u32]) -> Vec<f64> {
        let mut h = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        for &t in tokens {
            h = (h ^ u64::from(t)).wrapping_mul(0x0000_0100_0000_01b3);
        }
        let logits: Vec<f32> = (0..TOY_VOCAB as u32)
            .map(|i| {
                h = h
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let x = (h >> 40) as f32 / (1u64 << 24) as f32 * 6.0 - 3.0;
                if i == EOS {
                    x + 0.5 * tokens.len() as f32
                } else {
                    x
                }
            })
            .collect();
        log_softmax(&logits)
    }

    /// Asserts that `got` holds the hypotheses and scores of `want`, in its order.
    fn assert_same(got: &[Finished], want: &[(Vec<u32>, f64)], case: &str) {
        assert_eq!(got.len(), want.len(), "{case}");
        for (g, (tokens, score)) in got.iter().zip(want) {
            assert_eq!(&g.hypothesis.tokens, tokens, "{case}");
            assert!(
                (g.score - score).abs() < 1e-9,
                "{case}: {} against {score}",
                g.score
            );
        }
    }

    /// Over every width, length penalty, early stopping, end-token set and token cap, on toy
    /// models whose end token turns likely at various lengths, the search finishes the
    /// hypotheses transformers' search finishes, in its order and with its scores.
    #[test]
    fn the_search_answers_what_transformers_answers() {
        let stops = [
            EarlyStopping::Heuristic,
            EarlyStopping::Done,
            EarlyStopping::Never,
        ];
        let end_sets: [&[u32]; 3] = [&[], &[EOS], &[EOS, 5]];
        for seed in 0..12u64 {
            let model = |t: &[u32]| toy(seed, t);
            for width in 1..=4 {
                for lp in [-1.0, 0.0, 0.5, 1.0, 2.0] {
                    for es in stops {
                        for ends in end_sets {
                            for max_new in [1, 3, 6] {
                                let s = search(width, lp, es);
                                let mut searcher = BeamSearcher::new(s, ends.to_vec(), max_new, 0);
                                let mut rows = vec![model(&[])];
                                while searcher.advance(&rows) {
                                    rows = searcher
                                        .running()
                                        .iter()
                                        .map(|h| model(&h.tokens))
                                        .collect();
                                }
                                let case = format!(
                                    "seed {seed}, width {width}, lp {lp}, {es:?}, ends {ends:?}, \
                                     {max_new} tokens"
                                );
                                let want = transformers_search(s, ends, max_new, &model);
                                assert_same(&searcher.best(), &want, &case);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The first token alone ranks first, but only the second has a likely continuation: width
    /// 2 keeps the second, and the answer starts with it, as transformers' does.
    #[test]
    fn the_second_first_token_wins_on_its_continuation() {
        fn model(t: &[u32]) -> Vec<f64> {
            match t.first() {
                None => row(&[0.45, 0.30, 0.20, 0.05]),
                Some(0) => row(&[0.35, 0.30, 0.30, 0.05]),
                Some(_) => row(&[0.90, 0.04, 0.04, 0.02]),
            }
        }
        let s = search(2, 1.0, EarlyStopping::Heuristic);
        let best = run(s, 4, model);
        assert_eq!(tokens(&best), vec![vec![1, 0, 0, 0], vec![0, 0, 0, 0]]);
        let cum = (0.30f64 * 0.9f64.powi(3)).ln();
        assert!(close(best[0].hypothesis.cumulative, cum));
        assert!(close(best[0].score, cum / 4.0));
        assert_same(
            &best,
            &transformers_search(s, &[EOS], 4, &model),
            "the second token first",
        );
    }

    /// The logits of each beam, read on the one cache rolled back and replayed across steps whose
    /// beams part at the first token, share a prefix or come back to one the cache dropped, are
    /// bit for bit those a plain decode of that beam alone reads on a fresh model: the roll back
    /// leaves no row of another beam behind, in the full-width cache or the host's quantised one.
    #[test]
    fn a_replayed_beam_reads_what_its_plain_decode_reads() {
        use crate::inference::engine::model_backend::GenericBackend;
        use crate::inference::generic_transformer::tiny_model;
        const LAYERS: usize = 2;
        const PROMPT: [u32; 5] = [1, 4, 2, 9, 3];
        let steps: [&[&[u32]]; 4] = [
            &[&[7], &[5], &[8]],
            &[&[7, 6], &[5, 2], &[7, 1], &[8, 8]],
            &[&[5, 2, 9], &[7, 6, 3], &[5, 2, 10], &[7, 1, 4]],
            &[&[7, 1, 4, 4], &[5, 2, 9, 30], &[5, 2, 10, 2]],
        ];
        let ids = |t: &[u32]| Tensor::new(t, &Device::Cpu).unwrap().unsqueeze(0).unwrap();
        let path = tiny_model::write_quant("beam-rows", LAYERS);
        let plain = |beam: &[u32]| -> Vec<f64> {
            let mut m = GenericBackend(tiny_model::load(&path, LAYERS));
            let mut logits = m.forward(&ids(&PROMPT), 0).unwrap();
            for (i, &t) in beam.iter().enumerate() {
                logits = m.forward(&ids(&[t]), PROMPT.len() + i).unwrap();
            }
            log_softmax(&host_row(&logits).unwrap())
        };
        let mut m = GenericBackend(tiny_model::load(&path, LAYERS));
        m.forward(&ids(&PROMPT), 0).unwrap();
        let mut resident = Vec::new();
        let mut forwards = 0;
        for beams in steps {
            let running: Vec<Hypothesis> = beams
                .iter()
                .map(|t| Hypothesis {
                    tokens: t.to_vec(),
                    ..Hypothesis::default()
                })
                .collect();
            let rows = beam_rows(
                &mut m,
                &Device::Cpu,
                PROMPT.len(),
                &mut resident,
                &running,
                &mut forwards,
            )
            .unwrap();
            for (beam, row) in beams.iter().zip(&rows) {
                assert!(
                    row == &plain(beam),
                    "beam {beam:?} read other logits than its plain decode"
                );
            }
        }
        tiny_model::remove(&path);
    }

    /// The beams are visited in token order: the replays cost the divergent tails alone.
    #[test]
    fn the_shared_prefix_is_measured_from_the_start() {
        assert_eq!(common_prefix(&[1, 2, 3], &[1, 2, 4]), 2);
        assert_eq!(common_prefix(&[], &[1]), 0);
        assert_eq!(common_prefix(&[5, 6], &[5, 6]), 2);
    }
}
