//! Continuous-batching engine - wires the [`scheduler`](crate::inference::serve::scheduler) +
//! [`paged_kv`](crate::inference::cache::paged_kv) store into one decode loop behind a model
//! interface. This is the integration layer of the throughput-vs-vLLM feature.
//!
//! Each `step()`: ask the scheduler for the per-step [`BatchPlan`], turn it into
//! [`StepItem`]s (the input tokens + their paged KV slots / block tables), run
//! ONE batched model forward, append the sampled tokens, and let the scheduler
//! free finished sequences. New requests join and finished ones leave between
//! every step - no head-of-line blocking, no idle slots.
//!
//! The model plugs in via [`BatchedModel`]: it runs the batched forward (batched
//! GEMMs + paged attention reading KV through the block tables), writes new KV to
//! the store at the given slots, and returns one sampled token per item. Keeping
//! the model behind a trait lets the whole control flow be validated now with a
//! mock (the test below), with the real CUDA batched-paged forward as a drop-in
//! impl - exactly the data-plane work that remains.

use crate::inference::cache::paged_kv::{PagedKvAllocator, SeqId};
use crate::inference::serve::scheduler::{BatchPlan, FinishReason, Scheduler, SchedulerConfig};
use crate::tensor::Result;
use std::collections::HashMap;

/// One unit of work for the model this step. `tokens` are the inputs to embed:
/// the whole prompt for a prefill, the single most-recent token for a decode.
/// `slots`/`block_table` come from the paged allocator (write KV at `slots`,
/// attend over `block_table`).
pub struct StepItem {
    pub seq: SeqId,
    pub is_prefill: bool,
    pub tokens: Vec<u32>,
    pub slots: Vec<usize>,
    pub block_table: Vec<u32>,
    pub context_len: usize,
    /// Per-request sampling controls applied by the model adapter to this item's
    /// logits (greedy by default -> bit-identical argmax).
    pub sampling: SamplingParams,
    /// Tail of (prompt+generated) for the repeat-penalty window - the adapter has
    /// no per-seq history of its own, so the engine supplies it each step.
    pub recent_tokens: Vec<u32>,
    /// Prefix cache (prefill only): tokens 0..cached_len are already resident, so
    /// the model computes only `tokens[cached_len..]`, gathering the prefix KV via
    /// `block_table`. 0 = full prefill (the default).
    pub cached_len: usize,
}

/// Per-request sampling controls (mirrors the serial path's GenerationParams
/// subset). `temperature == 0` (the default) ⇒ greedy argmax - bit-identical to
/// the non-sampling path, so the greedy bench is unaffected.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: 0,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self::default()
    }
    /// True when this reduces to plain argmax (no temperature, no penalty) - lets
    /// the adapter keep the exact argmax fast path for greedy requests.
    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0 && self.repeat_penalty <= 1.0
    }
}

/// The last `last_n` tokens of (prompt ++ generated) - the repeat-penalty window
/// the model adapter applies to this step's logits. Empty when penalty is off.
fn recent_window(prompt: &[u32], generated: &[u32], last_n: usize) -> Vec<u32> {
    if last_n == 0 {
        return Vec::new();
    }
    let total = prompt.len() + generated.len();
    let start = total.saturating_sub(last_n);
    let mut out = Vec::with_capacity(total - start);
    out.extend(prompt.iter().chain(generated.iter()).skip(start).copied());
    out
}

/// The model side of the loop. One call = one batched forward over all items,
/// returning the sampled next token per item (same order). The MODEL owns its KV
/// (one paged store per layer - all layers share the allocator's block tables, so
/// the same `slots`/`block_table` in each [`StepItem`] address every layer); the
/// engine owns only block bookkeeping. This matches real transformers (L layers,
/// L KV caches) and keeps the engine model-agnostic.
pub trait BatchedModel {
    fn step(&mut self, items: &[StepItem]) -> Result<Vec<u32>>;
}

/// A generation request.
pub struct GenReq {
    pub id: SeqId,
    pub prompt: Vec<u32>,
    pub max_new: usize,
    pub sampling: SamplingParams,
}

/// A completed sequence's output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenDone {
    pub id: SeqId,
    pub tokens: Vec<u32>,
    pub reason: FinishReason,
}

struct ReqState {
    prompt: Vec<u32>,
    generated: Vec<u32>,
    last_was_prefill: bool,
    sampling: SamplingParams,
}

pub struct ContinuousBatchEngine<M: BatchedModel> {
    sched: Scheduler,
    model: M,
    eos: u32,
    reqs: HashMap<SeqId, ReqState>,
}

impl<M: BatchedModel> ContinuousBatchEngine<M> {
    pub fn new(
        model: M,
        eos: u32,
        num_blocks: usize,
        block_size: usize,
        max_running: usize,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        // Automatic prefix caching (shared prompt prefixes reuse KV blocks across
        // requests - solves multi-turn chat re-prefill) is always enabled.
        let alloc = PagedKvAllocator::new_with_prefix(num_blocks, block_size);
        let sched = Scheduler::new(
            alloc,
            SchedulerConfig {
                max_running,
                max_prefill_tokens,
            },
        );
        Ok(Self {
            sched,
            model,
            eos,
            reqs: HashMap::new(),
        })
    }

    pub fn submit(&mut self, req: GenReq) {
        self.sched
            .add_request(req.id, req.prompt.clone(), req.max_new);
        self.reqs.insert(
            req.id,
            ReqState {
                prompt: req.prompt,
                generated: Vec::new(),
                last_was_prefill: false,
                sampling: req.sampling,
            },
        );
    }

    pub fn is_idle(&self) -> bool {
        self.sched.is_idle()
    }

    /// Abort all in-flight sequences (free their KV, clear per-seq state) after a
    /// model step error, returning their ids so the caller can close their channels.
    /// Lets the worker recover and keep serving instead of dying on one bad step.
    pub fn abort_all(&mut self) -> Vec<SeqId> {
        let ids: Vec<SeqId> = self.reqs.keys().copied().collect();
        self.sched.reset();
        self.reqs.clear();
        ids
    }

    /// Run one batched step. Returns sequences that finished this step.
    pub fn step(&mut self) -> Result<Vec<GenDone>> {
        Ok(self.step_stream()?.1)
    }

    /// Run one batched step, returning BOTH the per-sequence tokens emitted this
    /// step (for incremental streaming) AND the sequences that finished. Each
    /// active sequence emits exactly one new token per step (a prefill emits its
    /// first generated token); prompt tokens are never emitted here.
    pub fn step_stream(&mut self) -> Result<(Vec<(SeqId, u32)>, Vec<GenDone>)> {
        let plan: BatchPlan = self.sched.schedule();
        if plan.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        // Build model inputs from the plan + per-request token state.
        let mut items = Vec::with_capacity(plan.num_seqs());
        for p in &plan.prefills {
            let r = self.reqs.get_mut(&p.seq).expect("submitted");
            r.last_was_prefill = true;
            let recent = recent_window(&r.prompt, &r.generated, r.sampling.repeat_last_n);
            items.push(StepItem {
                seq: p.seq,
                is_prefill: true,
                tokens: r.prompt.clone(),
                slots: p.slots.clone(),
                // Full prefill: self-attends over its own slots (no block table).
                // Prefix-cached prefill (cached_len>0): the block table lets the
                // suffix gather the cached prefix KV; `slots` are the suffix slots.
                block_table: p.block_table.clone(),
                context_len: p.prompt_len,
                sampling: r.sampling.clone(),
                recent_tokens: recent,
                cached_len: p.cached_len,
            });
        }
        for d in &plan.decodes {
            let r = self.reqs.get_mut(&d.seq).expect("running");
            // input is the most recently produced token
            let last = *r
                .generated
                .last()
                .unwrap_or_else(|| r.prompt.last().expect("non-empty prompt"));
            r.last_was_prefill = false;
            let recent = recent_window(&r.prompt, &r.generated, r.sampling.repeat_last_n);
            items.push(StepItem {
                seq: d.seq,
                is_prefill: false,
                tokens: vec![last],
                slots: vec![d.slot],
                block_table: d.block_table.clone(),
                context_len: d.context_len,
                sampling: r.sampling.clone(),
                recent_tokens: recent,
                cached_len: 0,
            });
        }

        let sampled = self.model.step(&items)?;
        debug_assert_eq!(sampled.len(), items.len());

        // Publish freshly-prefilled prompts' full blocks to the prefix cache so
        // later requests with a shared prefix reuse them (no-op when disabled).
        if self.sched.prefix_enabled() {
            for p in &plan.prefills {
                let prompt = match self.reqs.get(&p.seq) {
                    Some(r) => r.prompt.clone(),
                    None => continue,
                };
                self.sched.commit_prefix(p.seq, &prompt);
            }
        }

        // Record tokens + tell the scheduler which sequences hit EOS.
        let mut results = Vec::with_capacity(items.len());
        let mut emits = Vec::with_capacity(items.len());
        for (it, &tok) in items.iter().zip(sampled.iter()) {
            self.reqs.get_mut(&it.seq).unwrap().generated.push(tok);
            emits.push((it.seq, tok));
            results.push((it.seq, tok == self.eos));
        }
        let finished = self.sched.on_step_done(&results);

        let done = finished
            .into_iter()
            .map(|(id, reason)| {
                let st = self.reqs.remove(&id).unwrap();
                GenDone {
                    id,
                    tokens: st.generated,
                    reason,
                }
            })
            .collect();
        Ok((emits, done))
    }

    /// Drive to completion (test/offline-batch convenience). Returns all outputs.
    pub fn run_to_idle(&mut self) -> Result<Vec<GenDone>> {
        let mut out = Vec::new();
        let mut guard = 0usize;
        while !self.sched.is_idle() {
            out.extend(self.step()?);
            guard += 1;
            assert!(guard < 1_000_000, "continuous-batch loop runaway");
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock model: deterministic, KV-free - validates the LOOP, not attention.
    /// Emits a per-seq counter; EOS after the seq's `stop_after` tokens. Verifies
    /// the engine feeds it the right items and tracks tokens/finish correctly.
    struct MockModel {
        eos: u32,
        stop_after: HashMap<SeqId, usize>, // seq -> emit EOS on this token index
        seen_prefill: std::cell::RefCell<Vec<SeqId>>,
    }
    impl BatchedModel for MockModel {
        fn step(&mut self, items: &[StepItem]) -> Result<Vec<u32>> {
            let mut out = Vec::with_capacity(items.len());
            for it in items {
                if it.is_prefill {
                    self.seen_prefill.borrow_mut().push(it.seq);
                }
                // token index this seq is about to produce = (its prior count).
                // We don't track count here; encode it via context_len for prefill
                // (=prompt_len -> token 0) and decode carries growing context.
                let idx = if it.is_prefill { 0 } else { it.context_len }; // monotone-ish
                let stop = *self.stop_after.get(&it.seq).unwrap_or(&usize::MAX);
                out.push(if idx >= stop {
                    self.eos
                } else {
                    1000 + it.seq as u32
                });
            }
            Ok(out)
        }
    }

    fn engine(num_blocks: usize, stop: HashMap<SeqId, usize>) -> ContinuousBatchEngine<MockModel> {
        let m = MockModel {
            eos: 0,
            stop_after: stop,
            seen_prefill: Default::default(),
        };
        ContinuousBatchEngine::new(m, 0, num_blocks, 16, 8, 4096).unwrap()
    }

    #[test]
    fn single_request_generates_to_length() {
        let mut e = engine(64, HashMap::new());
        e.submit(GenReq {
            id: 1,
            prompt: vec![5, 6, 7],
            max_new: 4,
            sampling: SamplingParams::greedy(),
        });
        let done = e.run_to_idle().unwrap();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].id, 1);
        assert_eq!(done[0].tokens.len(), 4); // exactly max_new
        assert_eq!(done[0].reason, FinishReason::Length);
        assert!(e.is_idle());
    }

    #[test]
    fn abort_all_clears_state_and_frees_blocks() {
        let mut e = engine(64, HashMap::new());
        let free0 = e.sched.free_blocks();
        for id in 1..=4 {
            e.submit(GenReq {
                id,
                prompt: vec![1, 2, 3],
                max_new: 10,
                sampling: SamplingParams::greedy(),
            });
        }
        let _ = e.step().unwrap(); // prefill: allocate blocks for the 4 seqs
        assert!(e.sched.free_blocks() < free0, "blocks in use after prefill");
        let aborted = e.abort_all();
        assert_eq!(aborted.len(), 4, "all in-flight seqs reported");
        assert!(e.is_idle(), "engine idle after abort");
        assert_eq!(
            e.sched.free_blocks(),
            free0,
            "all KV blocks reclaimed - no leak"
        );
    }

    #[test]
    fn concurrent_requests_all_complete() {
        let mut e = engine(64, HashMap::new());
        for id in 1..=5 {
            e.submit(GenReq {
                id,
                prompt: vec![1, 2],
                max_new: 3,
                sampling: SamplingParams::greedy(),
            });
        }
        let done = e.run_to_idle().unwrap();
        assert_eq!(done.len(), 5);
        for d in &done {
            assert_eq!(d.tokens.len(), 3);
        }
        // all five distinct ids returned
        let ids: std::collections::HashSet<_> = done.iter().map(|d| d.id).collect();
        assert_eq!(ids.len(), 5);
    }

    #[test]
    fn eos_stops_before_max_new() {
        // seq 1 emits EOS once context_len >= 5 (prompt 3 + a couple decodes).
        let mut stop = HashMap::new();
        stop.insert(1u64, 5usize);
        let mut e = engine(64, stop);
        e.submit(GenReq {
            id: 1,
            prompt: vec![9, 9, 9],
            max_new: 100,
            sampling: SamplingParams::greedy(),
        });
        let done = e.run_to_idle().unwrap();
        assert_eq!(done.len(), 1);
        assert_eq!(done[0].reason, FinishReason::Eos);
        assert!(done[0].tokens.len() < 100); // stopped early on EOS
        assert_eq!(*done[0].tokens.last().unwrap(), 0); // last token is EOS
    }

    #[test]
    fn more_requests_than_kv_capacity_still_all_finish() {
        // tiny KV (2 blocks x 16 = 32 slots) but 6 short requests -> they must
        // queue/admit as capacity frees, and all complete (no deadlock).
        let mut e = engine(2, HashMap::new());
        for id in 1..=6 {
            e.submit(GenReq {
                id,
                prompt: vec![1],
                max_new: 2,
                sampling: SamplingParams::greedy(),
            });
        }
        let done = e.run_to_idle().unwrap();
        assert_eq!(done.len(), 6);
        assert!(e.is_idle());
    }
}
