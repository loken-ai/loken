//! The host work of the continuous-batching worker between 2 decode steps, for B running
//! sequences: the scheduler's plan and commit, the step items the engine builds, and the
//! vectors the decode path builds from them before any card call. No model, no card and no
//! server is touched: the scheduler and the engine are the crate's own, the model answers at once.
//!
//! The host stretch is the sum of the first 3 variants, during which the card is idle.

use super::stats::run_for;
use super::{Context, Measured};
use crate::inference::cache::paged_kv::{PagedKvAllocator, SeqId};
use crate::inference::serve::continuous_batch::{
    BatchedModel, ContinuousBatchEngine, GenReq, SamplingParams, StepItem,
};
use crate::inference::serve::continuous_serve::CB_GRAPH_BUCKET;
use crate::inference::serve::scheduler::{BatchPlan, Scheduler, SchedulerConfig};
use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

/// A sampled request, as a client of the API sends one.
fn sampling() -> SamplingParams {
    SamplingParams {
        temperature: 0.8,
        top_k: Some(40),
        top_p: Some(0.95),
        repeat_penalty: 1.1,
        repeat_last_n: 64,
        seed: 0,
        ends: None,
    }
}

/// The shape of the run: the sequences, their context and the cache's blocks.
struct Setup {
    batch: usize,
    context: usize,
    prompt: usize,
    block_size: usize,
}

impl Setup {
    fn of(cx: &Context) -> Result<Self, String> {
        let o = cx.options;
        if o.batch == 0 || o.block_size == 0 || o.rope_head_dim() < 2 {
            return Err("--batch, --block-size and --head-dim must be above zero".to_string());
        }
        if o.prompt_tokens == 0 || o.context <= o.prompt_tokens {
            return Err("--context must exceed --prompt-tokens, which is above zero".to_string());
        }
        Ok(Self {
            batch: o.batch,
            context: o.context,
            prompt: o.prompt_tokens,
            block_size: o.block_size,
        })
    }

    /// Steps a timed sample runs: the context grows by 1 block per sequence over a sample, as
    /// it does in a served run.
    fn steps(&self) -> usize {
        self.block_size
    }

    /// Steps run before the one timed: the prefill, then the decodes up to the context.
    fn warm_steps(&self) -> usize {
        self.context - self.prompt
    }

    fn num_blocks(&self) -> usize {
        let blocks = (self.context + self.steps() + 1).div_ceil(self.block_size);
        self.batch * (blocks + 1)
    }

    fn prompt_of(&self, seq: SeqId) -> Vec<u32> {
        let base = seq as u32 * self.prompt as u32;
        (0..self.prompt as u32).map(|i| base + i).collect()
    }

    fn max_prefill_tokens(&self) -> usize {
        self.batch * self.prompt
    }

    /// A scheduler holding the batch's sequences, all decoding, one step from `context`.
    fn scheduler(&self) -> Scheduler {
        let alloc = PagedKvAllocator::new_with_prefix(self.num_blocks(), self.block_size);
        let mut sched = Scheduler::new(
            alloc,
            SchedulerConfig {
                max_running: self.batch,
                max_prefill_tokens: self.max_prefill_tokens(),
            },
        );
        for seq in 0..self.batch as SeqId {
            sched.add_request(seq, self.prompt_of(seq), usize::MAX);
        }
        for _ in 0..self.warm_steps() {
            let plan = sched.schedule();
            let results: Vec<(SeqId, bool)> = plan
                .prefills
                .iter()
                .map(|p| p.seq)
                .chain(plan.decodes.iter().map(|d| d.seq))
                .map(|s| (s, false))
                .collect();
            sched.on_step_done(&results);
        }
        sched
    }

    /// An engine over a model that answers at once, in the same state.
    fn engine(&self) -> ContinuousBatchEngine<InstantModel> {
        let mut engine = ContinuousBatchEngine::new(
            InstantModel,
            vec![0],
            self.num_blocks(),
            self.block_size,
            self.batch,
            self.max_prefill_tokens(),
        )
        .expect("the engine opens");
        for seq in 0..self.batch as SeqId {
            engine.submit(GenReq {
                id: seq,
                prompt: self.prompt_of(seq),
                max_new: usize::MAX,
                sampling: sampling(),
            });
        }
        for _ in 0..self.warm_steps() {
            engine.step_stream().expect("the model answers");
        }
        engine
    }

    /// The token state the engine keeps for each sequence at `context`.
    fn requests(&self) -> HashMap<SeqId, Request> {
        (0..self.batch as SeqId)
            .map(|seq| {
                let generated = (0..(self.context - self.prompt) as u32).collect();
                let request = Request {
                    prompt: self.prompt_of(seq),
                    generated,
                    sampling: sampling(),
                };
                (seq, request)
            })
            .collect()
    }
}

/// A model that answers 1 token per item and does nothing else.
struct InstantModel;

impl BatchedModel for InstantModel {
    fn step(&mut self, items: &[StepItem]) -> crate::tensor::Result<Vec<u32>> {
        Ok(vec![1; items.len()])
    }
}

/// What the engine keeps of a request between steps.
struct Request {
    prompt: Vec<u32>,
    generated: Vec<u32>,
    sampling: SamplingParams,
}

/// The last `last_n` tokens of the prompt followed by what was generated.
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

/// The step items of a plan's decodes, built as the engine builds them: every vector copied.
fn items_with_clones(plan: &BatchPlan, reqs: &mut HashMap<SeqId, Request>) -> Vec<StepItem> {
    let mut items = Vec::with_capacity(plan.num_seqs());
    for d in &plan.decodes {
        let r = reqs.get_mut(&d.seq).expect("running");
        let last = *r
            .generated
            .last()
            .unwrap_or_else(|| r.prompt.last().expect("non-empty prompt"));
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
            cancel: None,
        });
    }
    items
}

/// A decode row that borrows what the plan and the request already hold.
struct RowView<'a> {
    seq: SeqId,
    token: u32,
    slot: usize,
    block_table: &'a [u32],
    context_len: usize,
    sampling: &'a SamplingParams,
    /// The repeat window as 2 slices: the end of the prompt, then the end of the generated
    /// tokens.
    recent: (&'a [u32], &'a [u32]),
}

/// The same rows with every vector borrowed.
fn items_with_borrows<'a>(
    plan: &'a BatchPlan,
    reqs: &'a HashMap<SeqId, Request>,
) -> Vec<RowView<'a>> {
    let mut rows = Vec::with_capacity(plan.num_seqs());
    for d in &plan.decodes {
        let r = reqs.get(&d.seq).expect("running");
        let token = *r
            .generated
            .last()
            .unwrap_or_else(|| r.prompt.last().expect("non-empty prompt"));
        let last_n = r.sampling.repeat_last_n;
        let from_generated = last_n.min(r.generated.len());
        let from_prompt = (last_n - from_generated).min(r.prompt.len());
        rows.push(RowView {
            seq: d.seq,
            token,
            slot: d.slot,
            block_table: &d.block_table,
            context_len: d.context_len,
            sampling: &r.sampling,
            recent: (
                &r.prompt[r.prompt.len() - from_prompt..],
                &r.generated[r.generated.len() - from_generated..],
            ),
        });
    }
    rows
}

/// The vectors `GhBatched::step` and `decode_graph` build from the items before the first card
/// call: the per-field vectors, the batch padded to a power of two, the flat block table and
/// the sequence lengths and slots as 32 bit indices. Answers the bytes it built.
fn decode_vectors(items: &[StepItem], block_size: usize) -> usize {
    let dec: Vec<usize> = (0..items.len()).filter(|&i| !items[i].is_prefill).collect();
    let toks: Vec<u32> = dec.iter().map(|&i| items[i].tokens[0]).collect();
    let pos: Vec<usize> = dec.iter().map(|&i| items[i].context_len - 1).collect();
    let slots: Vec<usize> = dec.iter().map(|&i| items[i].slots[0]).collect();
    let bts: Vec<Vec<u32>> = dec.iter().map(|&i| items[i].block_table.clone()).collect();
    let ctx: Vec<usize> = dec.iter().map(|&i| items[i].context_len).collect();
    let greedy = dec.iter().all(|&i| items[i].sampling.temperature <= 0.0);

    let raw_b = toks.len();
    let bpad = if raw_b <= 1 {
        1
    } else {
        raw_b.next_power_of_two()
    };
    let pad = bpad - raw_b;
    let toks_v: Vec<u32> = toks
        .iter()
        .copied()
        .chain(std::iter::repeat_n(toks[0], pad))
        .collect();
    let pos_v: Vec<usize> = pos
        .iter()
        .copied()
        .chain(std::iter::repeat_n(pos[0], pad))
        .collect();
    let slots_v: Vec<usize> = slots
        .iter()
        .copied()
        .chain(std::iter::repeat_n(slots[0], pad))
        .collect();
    let ctx_v: Vec<usize> = ctx
        .iter()
        .copied()
        .chain(std::iter::repeat_n(ctx[0], pad))
        .collect();
    let bts_v: Vec<Vec<u32>> = bts
        .iter()
        .cloned()
        .chain(std::iter::repeat_n(bts[0].clone(), pad))
        .collect();
    let b = toks_v.len();
    let max_ctx = *ctx_v.iter().max().unwrap();
    let cap = max_ctx.div_ceil(CB_GRAPH_BUCKET) * CB_GRAPH_BUCKET;
    let max_blocks = cap / block_size;
    let slot_i32: Vec<i32> = slots_v.iter().map(|&s| s as i32).collect();
    let seq_lens: Vec<i32> = ctx_v.iter().map(|&c| c as i32).collect();
    let mut btf = vec![0i32; b * max_blocks];
    for (bi, bt) in bts_v.iter().enumerate() {
        for (j, &blk) in bt.iter().take(max_blocks).enumerate() {
            btf[bi * max_blocks + j] = blk as i32;
        }
    }
    black_box((&toks_v, &pos_v, &slot_i32, &seq_lens, &btf, greedy));
    btf.len() * 4 + (slot_i32.len() + seq_lens.len()) * 4
}

/// The cosine and sine rows `rope_cos_sin` computes on the host for the padded batch, as f32.
fn rope_rows(positions: &[usize], head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    let rows = positions.len().next_power_of_two();
    let half = head_dim / 2;
    let mut cos = vec![0f32; rows * half];
    let mut sin = vec![0f32; rows * half];
    let inv_freq = crate::inference::model::rope::inverse_frequencies(head_dim, base);
    for b in 0..rows {
        let position = positions[b.min(positions.len() - 1)];
        for i in 0..half {
            let (s, c) = (position as f32 * inv_freq[i]).sin_cos();
            cos[b * half + i] = c;
            sin[b * half + i] = s;
        }
    }
    (cos, sin)
}

/// `steps` calls of `step` on a state built by `build`, which the clock leaves out. Answers the
/// time of 1 step in microseconds.
fn per_step<S>(steps: usize, build: impl FnOnce() -> S, mut step: impl FnMut(&mut S)) -> f64 {
    let mut state = build();
    let t = Instant::now();
    for _ in 0..steps {
        step(&mut state);
    }
    t.elapsed().as_secs_f64() * 1e6 / steps as f64
}

/// The time of 1 call of `f` in microseconds.
fn once(mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1e6
}

fn median(samples: &[f64]) -> f64 {
    super::stats::Summary::of(samples).median_us
}

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let setup = Setup::of(cx)?;
    let target = Duration::from_secs_f64(cx.options.seconds);
    let (head_dim, base) = (cx.options.rope_head_dim(), cx.options.rope_base);
    let steps = setup.steps();
    let mut rows = Vec::new();

    // The scheduler alone: a plan, then the commit of the tokens.
    let schedule = run_for(target, || {
        per_step(
            steps,
            || setup.scheduler(),
            |sched| {
                let plan = sched.schedule();
                let results: Vec<(SeqId, bool)> =
                    plan.decodes.iter().map(|d| (d.seq, false)).collect();
                black_box(sched.on_step_done(&results));
                black_box(&plan);
            },
        )
    });
    let schedule_us = median(&schedule);
    rows.push(Measured {
        variant: format!(
            "schedule and commit, {} decoding rows at {} tokens",
            setup.batch, setup.context
        ),
        bytes: None,
        samples_us: schedule,
        note: Some(format!(
            "each sample is {steps} consecutive steps, divided; the results list is built as the engine builds it"
        )),
    });

    // A plan frozen at the context, and the request state, for the item variants.
    let plan = setup.scheduler().schedule();
    let mut reqs = setup.requests();
    let items_clone = run_for(target, || {
        once(|| {
            black_box(items_with_clones(&plan, &mut reqs));
        })
    });
    let items_us = median(&items_clone);
    rows.push(Measured {
        variant: "step items with clones, as the engine builds them".to_string(),
        bytes: None,
        samples_us: items_clone,
        note: Some(
            "tokens, slots, block table and repeat window are vectors of their own, the sampling options are cloned"
                .to_string(),
        ),
    });
    let items_borrow = run_for(target, || {
        once(|| {
            black_box(items_with_borrows(&plan, &reqs));
        })
    });
    rows.push(Measured {
        variant: "step items with borrows".to_string(),
        bytes: None,
        samples_us: items_borrow,
        note: Some(
            "rows of a type local to the bench: a model taking them would need its own signature changed"
                .to_string(),
        ),
    });

    // The decode path's vectors, from items built once.
    let items = items_with_clones(&plan, &mut reqs);
    let mut built = 0;
    let vectors = run_for(target, || {
        once(|| built = black_box(decode_vectors(&items, setup.block_size)))
    });
    let vectors_us = median(&vectors);
    rows.push(Measured {
        variant: "decode inputs: indices, padded block tables".to_string(),
        bytes: Some(built as u64),
        samples_us: vectors,
        note: Some(
            "the vectors of the step and of the graph path up to the first card call".to_string(),
        ),
    });
    let positions: Vec<usize> = items.iter().map(|i| i.context_len - 1).collect();
    let rope = run_for(target, || {
        once(|| {
            black_box(rope_rows(&positions, head_dim, base));
        })
    });
    let rope_us = median(&rope);
    rows.push(Measured {
        variant: "decode inputs: rope cosine and sine rows".to_string(),
        bytes: None,
        samples_us: rope,
        note: Some("the f32 rows, before the conversion and the upload".to_string()),
    });

    let sum = schedule_us + items_us + vectors_us + rope_us;
    rows.push(Measured {
        variant: "host stretch, sum of the medians of the variants with clones".to_string(),
        bytes: None,
        samples_us: vec![sum],
        note: Some("the card is idle for this long between 2 decode steps".to_string()),
    });

    // The whole engine step over a model that answers at once, as a cross-check.
    let whole = run_for(target, || {
        per_step(
            steps,
            || setup.engine(),
            |engine| {
                black_box(engine.step_stream().expect("the model answers"));
            },
        )
    });
    rows.push(Measured {
        variant: "engine step over a model that answers at once".to_string(),
        bytes: None,
        samples_us: whole,
        note: Some(
            "schedule, items, the model's call, the tokens recorded and the commit".to_string(),
        ),
    });
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn setup() -> Setup {
        Setup {
            batch: 5,
            context: 40,
            prompt: 7,
            block_size: 4,
        }
    }

    #[test]
    fn the_scheduler_is_one_step_from_the_context() {
        let s = setup();
        let plan = s.scheduler().schedule();
        assert_eq!(plan.decodes.len(), s.batch);
        assert!(plan.prefills.is_empty());
        assert!(plan.decodes.iter().all(|d| d.context_len == s.context));
    }

    #[test]
    fn the_borrowed_rows_name_what_the_cloned_items_hold() {
        let s = setup();
        let plan = s.scheduler().schedule();
        let mut reqs = s.requests();
        let items = items_with_clones(&plan, &mut reqs);
        let views = items_with_borrows(&plan, &reqs);
        for (item, view) in items.iter().zip(&views) {
            assert_eq!(item.seq, view.seq);
            assert_eq!(item.tokens[0], view.token);
            assert_eq!(item.slots[0], view.slot);
            assert_eq!(item.block_table, view.block_table);
            assert_eq!(item.context_len, view.context_len);
            assert_eq!(item.sampling.top_k, view.sampling.top_k);
            let mut window = view.recent.0.to_vec();
            window.extend_from_slice(view.recent.1);
            assert_eq!(item.recent_tokens, window);
        }
    }

    #[test]
    fn the_engine_reaches_the_same_context() {
        let s = setup();
        let mut engine = s.engine();
        let (emits, _) = engine.step_stream().unwrap();
        assert_eq!(emits.len(), s.batch);
    }

    #[test]
    fn the_bench_runs_and_names_its_variants() {
        let o = crate::microbench::Options::parse_from([
            "microbench",
            "schedule-host",
            "--batch",
            "6",
            "--context",
            "40",
            "--prompt-tokens",
            "7",
            "--seconds",
            "0.01",
        ]);
        let shapes = crate::microbench::shapes::resolve(&o).unwrap();
        let cx = Context {
            options: &o,
            shapes: &shapes,
            #[cfg(feature = "gpu")]
            card: None,
        };
        let rows = run(&cx).unwrap();
        assert_eq!(rows.len(), 7);
        assert!(rows.iter().all(|r| !r.samples_us.is_empty()));
    }
}
