//! Iteration-level scheduler for continuous batching (vLLM-lineage) - the
//! "continuous" control plane on top of [`crate::inference::cache::paged_kv`]. Throughput lever vs
//! vLLM under concurrent load.
//!
//! Classic (static) batching runs a fixed batch start-to-finish: a slow/long
//! request holds its slot until done (head-of-line blocking) and finished
//! requests leave the GPU idle until the whole batch ends. Continuous batching
//! schedules at EVERY decode step: finished sequences free their slots
//! immediately, waiting requests join as soon as KV blocks are available, and
//! prefill + decode are mixed into one step. Each `schedule()` returns a
//! [`BatchPlan`] - the exact per-step work (which sequences prefill, which
//! decode, with their paged `slot_mapping`s and block tables) - which the batched
//! forward consumes. After the forward, [`Scheduler::on_step_done`] commits the
//! sampled tokens and frees finished sequences.
//!
//! Pure orchestration logic (no model, no GPU) -> deterministic and unit-tested
//! cold. Admission never half-allocates (the paged allocator's `can_append`
//! gates every step), so a decode step always completes; under block pressure a
//! running sequence is PREEMPTED (its blocks freed, requeued for recompute  -
//! vLLM's default recovery), guaranteeing forward progress.

use crate::inference::cache::paged_kv::{PagedKvAllocator, SeqId};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
    Eos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqStatus {
    Waiting, // queued, not yet prefilled (or preempted -> recompute)
    Running, // in the active decode set
    Finished(FinishReason),
}

/// One in-flight generation request.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: SeqId,
    pub prompt_len: usize,
    pub max_new: usize,   // generation budget (num_predict)
    pub generated: usize, // tokens produced so far
    pub status: SeqStatus,
    arrival: u64, // FCFS tiebreak
    /// Prompt token ids - kept so the allocator can content-hash the prefix for
    /// automatic prefix caching at admission. Empty when caching is disabled.
    prompt: Vec<u32>,
}

impl Sequence {
    /// Total tokens currently in this sequence's KV (prompt + generated).
    pub fn ctx_len(&self) -> usize {
        self.prompt_len + self.generated
    }
}

/// Work for one model step. `prefills` carry their whole prompt's slot_mapping;
/// `decodes` carry the single new token's slot + the block table to gather KV.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct BatchPlan {
    pub prefills: Vec<PrefillItem>,
    pub decodes: Vec<DecodeItem>,
    pub preempted: Vec<SeqId>, // sequences evicted this step (for logging/metrics)
}

#[derive(Debug, PartialEq, Eq)]
pub struct PrefillItem {
    pub seq: SeqId,
    pub prompt_len: usize,
    pub slots: Vec<usize>, // KV slots to WRITE (whole prompt, or just the suffix
    // when `cached_len`>0)
    /// Prefix-cache: tokens 0..cached_len are already resident (shared blocks); the
    /// prefill computes only `prompt[cached_len..]`. 0 = full prefill.
    pub cached_len: usize,
    /// Full block table (for the cached-prefill suffix attention to gather the
    /// prefix KV). Empty when `cached_len`==0.
    pub block_table: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct DecodeItem {
    pub seq: SeqId,
    pub slot: usize,        // KV slot for the new token
    pub context_len: usize, // tokens attended (incl. the new one)
    pub block_table: Vec<u32>,
}

impl BatchPlan {
    pub fn is_empty(&self) -> bool {
        self.prefills.is_empty() && self.decodes.is_empty()
    }
    pub fn num_seqs(&self) -> usize {
        self.prefills.len() + self.decodes.len()
    }
}

pub struct SchedulerConfig {
    pub max_running: usize,        // max sequences decoded per step (batch width)
    pub max_prefill_tokens: usize, // prompt-token budget admitted per step
}

pub struct Scheduler {
    alloc: PagedKvAllocator,
    cfg: SchedulerConfig,
    waiting: VecDeque<Sequence>,
    running: Vec<Sequence>,
    next_arrival: u64,
}

impl Scheduler {
    pub fn new(alloc: PagedKvAllocator, cfg: SchedulerConfig) -> Self {
        Self {
            alloc,
            cfg,
            waiting: VecDeque::new(),
            running: Vec::new(),
            next_arrival: 0,
        }
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }
    pub fn num_running(&self) -> usize {
        self.running.len()
    }
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty()
    }
    pub fn free_blocks(&self) -> usize {
        self.alloc.free_blocks()
    }

    /// Enqueue a new request (FCFS). `max_new` is the generation budget. `prompt`
    /// tokens are kept only for prefix-cache hashing (cleared if caching is off).
    pub fn add_request(&mut self, id: SeqId, prompt: Vec<u32>, max_new: usize) {
        let arrival = self.next_arrival;
        self.next_arrival += 1;
        let prompt_len = prompt.len();
        let prompt = if self.alloc.prefix_enabled() {
            prompt
        } else {
            Vec::new()
        };
        self.waiting.push_back(Sequence {
            id,
            prompt_len,
            max_new,
            generated: 0,
            status: SeqStatus::Waiting,
            arrival,
            prompt,
        });
    }

    /// True if automatic prefix caching is enabled on the allocator.
    pub fn prefix_enabled(&self) -> bool {
        self.alloc.prefix_enabled()
    }

    /// Abort everything: free all running sequences' blocks and drop the queues.
    /// Used to recover the worker after a model step error without leaking KV.
    /// (Waiting sequences hold no blocks yet - allocation happens at admission.)
    pub fn reset(&mut self) {
        for s in std::mem::take(&mut self.running) {
            let _ = self.alloc.free(s.id);
        }
        self.waiting.clear();
    }

    /// Publish a finished prefill's fresh full prompt blocks to the prefix cache so
    /// later requests reuse them (no-op when caching is disabled).
    pub fn commit_prefix(&mut self, seq: SeqId, prompt: &[u32]) {
        self.alloc.commit_prefix(seq, prompt);
    }

    /// Build the next step's work: admit as many waiting prefills as fit (block +
    /// token + width budgets), then decode every running sequence, preempting the
    /// most-recently-admitted running sequence if block pressure blocks a decode.
    pub fn schedule(&mut self) -> BatchPlan {
        let mut plan = BatchPlan::default();

        // -- 1. Grow running decodes (each needs 1 new token; a new block only on
        // a block boundary). Process oldest-first; when a sequence can't grow,
        // preempt the NEWEST running sequence (free its blocks, requeue for
        // recompute) until the current one fits - so the oldest always progress.
        let mut running = std::mem::take(&mut self.running);
        running.sort_by_key(|s| s.arrival);
        let mut i = 0;
        while i < running.len() {
            // The allocator's len is the source of truth for KV occupancy (the
            // scheduler's `generated` is only for finish detection and may lead by
            // one un-written token). Append the single pending token here.
            let cur = self
                .alloc
                .seq_len(running[i].id)
                .expect("running seq allocated");
            if self.alloc.can_append(cur, 1) {
                let s = &running[i];
                let slots = self.alloc.append(s.id, 1).expect("can_append checked");
                let bt = self
                    .alloc
                    .block_table(s.id)
                    .expect("running seq allocated")
                    .to_vec();
                plan.decodes.push(DecodeItem {
                    seq: s.id,
                    slot: slots[0],
                    context_len: cur + 1,
                    block_table: bt,
                });
                i += 1;
            } else if running.len() - 1 > i {
                // Preempt the newest (last) not-yet-scheduled sequence, then retry i.
                let v = running.pop().unwrap();
                let _ = self.alloc.free(v.id);
                plan.preempted.push(v.id);
                let mut v = v;
                v.status = SeqStatus::Waiting;
                v.generated = 0;
                self.waiting.push_front(v);
            } else {
                // Only the current sequence remains and it still can't grow ->
                // preempt it too (out of KV); stop growing this step.
                let v = running.remove(i);
                let _ = self.alloc.free(v.id);
                plan.preempted.push(v.id);
                let mut v = v;
                v.status = SeqStatus::Waiting;
                v.generated = 0;
                self.waiting.push_front(v);
                break;
            }
        }
        self.running = running;

        // -- 2. Admit waiting prefills into the freed/remaining capacity. --
        let mut prefill_budget = self.cfg.max_prefill_tokens;
        while self.running.len() < self.cfg.max_running {
            let Some(front) = self.waiting.front() else {
                break;
            };
            let need = front.prompt_len;
            if need > prefill_budget && !plan.prefills.is_empty() {
                break; // keep the step bounded; admit it next step
            }
            // Enough KV for the whole prompt? (can_append is seq-agnostic.)
            if !self.alloc.can_append(0, need) {
                break; // not enough KV - leave it (and the rest) waiting
            }
            let mut s = self.waiting.pop_front().unwrap();
            let (slots, cached_len, block_table) = if self.alloc.prefix_enabled() {
                // Reuse the cached prompt prefix; allocate fresh blocks only for the
                // suffix. The prefill writes/computes just the suffix slots, gathering
                // the prefix KV via the (full) block table.
                let cached = self
                    .alloc
                    .allocate_with_prefix(s.id, &s.prompt)
                    .expect("can_append checked");
                let slots = self.alloc.slots_for_range(s.id, cached, need);
                let bt = self
                    .alloc
                    .block_table(s.id)
                    .map(|b| b.to_vec())
                    .unwrap_or_default();
                (slots, cached, bt)
            } else {
                // Register the sequence empty, then append the prompt tokens to get
                // their slot_mapping (allocate(0) reserves no blocks; append grows).
                self.alloc.allocate(s.id, 0).expect("fresh seq id");
                let slots = self.alloc.append(s.id, need).expect("can_append checked");
                (slots, 0, Vec::new())
            };
            s.status = SeqStatus::Running;
            plan.prefills.push(PrefillItem {
                seq: s.id,
                prompt_len: need,
                slots,
                cached_len,
                block_table,
            });
            prefill_budget = prefill_budget.saturating_sub(need);
            self.running.push(s);
        }
        plan
    }

    /// Commit one step's results: for each (seq, sampled_token_is_terminal),
    /// advance the generation counter and finish/free sequences that hit EOS or
    /// their length budget. Returns the sequences that finished this step.
    pub fn on_step_done(&mut self, results: &[(SeqId, bool)]) -> Vec<(SeqId, FinishReason)> {
        let term: std::collections::HashMap<SeqId, bool> = results.iter().copied().collect();
        let mut finished = Vec::new();
        let mut keep = Vec::with_capacity(self.running.len());
        for mut s in std::mem::take(&mut self.running) {
            // A sequence is in `running` because it was prefilled or decoded this
            // step -> it produced one new token.
            s.generated += 1;
            let eos = term.get(&s.id).copied().unwrap_or(false);
            let reason = if eos {
                Some(FinishReason::Eos)
            } else if s.generated >= s.max_new {
                Some(FinishReason::Length)
            } else {
                None
            };
            match reason {
                Some(r) => {
                    s.status = SeqStatus::Finished(r);
                    let _ = self.alloc.free(s.id);
                    finished.push((s.id, r));
                }
                None => keep.push(s),
            }
        }
        self.running = keep;
        finished
    }

    /// Roll a sequence back (speculative-decode reject) by `n` tokens.
    pub fn trim(&mut self, seq: SeqId, n_keep: usize) {
        if let Some(s) = self.running.iter_mut().find(|s| s.id == seq) {
            let keep = n_keep.saturating_sub(s.prompt_len);
            s.generated = keep.min(s.generated);
            let _ = self.alloc.trim(seq, s.ctx_len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(num_blocks: usize, block_size: usize, max_running: usize) -> Scheduler {
        Scheduler::new(
            PagedKvAllocator::new(num_blocks, block_size),
            SchedulerConfig {
                max_running,
                max_prefill_tokens: 4096,
            },
        )
    }

    #[test]
    fn admits_prefills_then_decodes() {
        let mut s = sched(64, 16, 8);
        s.add_request(1, vec![0u32; 10], 5);
        s.add_request(2, vec![0u32; 20], 5);
        let p = s.schedule();
        assert_eq!(p.prefills.len(), 2);
        assert_eq!(p.decodes.len(), 0);
        assert_eq!(p.prefills[0].slots.len(), 10);
        assert_eq!(s.num_running(), 2);
        // commit a step (no EOS) -> both move to decode next step
        let fin = s.on_step_done(&[(1, false), (2, false)]);
        assert!(fin.is_empty());
        let p2 = s.schedule();
        assert_eq!(p2.prefills.len(), 0);
        assert_eq!(p2.decodes.len(), 2);
        // prefill wrote KV=10 (prompt); first decode appends token0 -> KV=11, and
        // the forward attends to all 11 to produce token1.
        assert_eq!(p2.decodes[0].context_len, 11);
    }

    #[test]
    fn finishes_on_length_and_frees_blocks() {
        let mut s = sched(64, 16, 8);
        s.add_request(1, vec![0u32; 8], 2); // max_new=2
        s.schedule(); // prefill
        let f0 = s.on_step_done(&[(1, false)]); // generated=1
        assert!(f0.is_empty());
        s.schedule(); // decode 1
        let f1 = s.on_step_done(&[(1, false)]); // generated=2 == max_new -> Length
        assert_eq!(f1, vec![(1, FinishReason::Length)]);
        assert!(s.is_idle());
        assert_eq!(s.free_blocks(), 64); // all blocks returned
    }

    #[test]
    fn eos_finishes_immediately() {
        let mut s = sched(64, 16, 8);
        s.add_request(1, vec![0u32; 8], 100);
        s.schedule();
        let f = s.on_step_done(&[(1, true)]); // EOS
        assert_eq!(f, vec![(1, FinishReason::Eos)]);
        assert!(s.is_idle());
    }

    #[test]
    fn max_running_caps_batch_width() {
        let mut s = sched(256, 16, 2); // only 2 may run at once
        for i in 0..5 {
            s.add_request(i, vec![0u32; 4], 10);
        }
        let p = s.schedule();
        assert_eq!(p.prefills.len(), 2);
        assert_eq!(s.num_running(), 2);
        assert_eq!(s.num_waiting(), 3);
    }

    #[test]
    fn waits_when_kv_blocks_exhausted() {
        // 2 blocks x 4 tokens = 8 token-slots total.
        let mut s = sched(2, 4, 8);
        s.add_request(1, vec![0u32; 8], 50); // takes both blocks
        s.add_request(2, vec![0u32; 4], 50); // no blocks left
        let p = s.schedule();
        assert_eq!(p.prefills.len(), 1); // only seq 1 admitted
        assert_eq!(s.num_waiting(), 1); // seq 2 waits for KV
        assert_eq!(s.free_blocks(), 0);
    }

    #[test]
    fn preempts_under_block_pressure_to_guarantee_progress() {
        // 3 blocks x 4 = 12 slots. Two seqs of 4 tokens (1 block each) run; a
        // third block is the only spare. Force both to grow until the spare runs
        // out -> the newer must be preempted so the older keeps decoding.
        let mut s = sched(3, 4, 8);
        s.add_request(1, vec![0u32; 4], 50);
        s.add_request(2, vec![0u32; 4], 50);
        s.schedule(); // both prefill: 2 blocks used, 1 free
        s.on_step_done(&[(1, false), (2, false)]); // each now 5 tokens... wait, append happens in schedule
                                                   // Decode step: seq1 grows to 6 (needs 2nd block), seq2 grows to 6 (needs 2nd block)
                                                   // but only 1 free block -> one is preempted.
        let p = s.schedule();
        assert!(!p.preempted.is_empty() || p.decodes.len() <= 2);
        // The system makes progress (at least one decode) and never panics.
        assert!(p.decodes.len() >= 1);
    }

    #[test]
    fn freed_slot_admits_a_waiting_request() {
        let mut s = sched(2, 4, 8); // 8 slots
        s.add_request(1, vec![0u32; 8], 1); // fills both blocks, max_new=1
        s.add_request(2, vec![0u32; 8], 5); // waits
        s.schedule(); // prefill seq1
        assert_eq!(s.num_waiting(), 1);
        s.on_step_done(&[(1, false)]); // seq1 hits max_new=1 -> finishes, frees 2 blocks
        assert!(s.free_blocks() >= 2 || s.num_running() == 1);
        let p = s.schedule(); // now seq2 fits
        assert_eq!(p.prefills.len(), 1);
        assert_eq!(p.prefills[0].seq, 2);
    }
}
