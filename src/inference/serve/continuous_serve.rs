//! Live continuous-batching SERVER: a background worker thread that owns the
//! model + per-layer paged KV stores and multiplexes many concurrent requests
//! into batched decode steps, streaming each request's tokens back as they are
//! produced. This is what turns `batched_paged_decode` (the throughput lever)
//! into a concurrent serving mode - many in-flight requests share each weight
//! read (memory-bound single-stream decode -> compute-bound batched GEMM).
//!
//! Ownership model: the worker is the SOLE caller of the model forward, so the
//! model needs no lock - requests only exchange a prompt + a token channel with
//! the worker. The model is held as an `Arc<GenericHeteroTransformer>` so it can
//! be SHARED with the owning `LlmEngine` (which delegates a CB-served model's
//! requests here and must NOT call the model itself - that sole-caller contract
//! is what keeps the lock-free sharing sound). Supported for the models
//! `cb_unsupported_reason` lets through; callers fall back to the serial path otherwise.
//!
//! A model whose layers keep each sequence's state is served by `spawn_sequences`.
//! Its worker owns the model and holds each sequence's state, see `SequenceBatched`.
use crate::inference::cache::paged_kv::{PagedKvStore, SeqId};
use crate::inference::generic_transformer::wide_sparse_layer::SequenceMixers;
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::inference::serve::continuous_batch::{
    BatchedModel, ContinuousBatchEngine, GenReq, SamplingParams, StepItem,
};
use crate::inference::serve::scheduler::FinishReason;
use crate::tensor::{DType, Tensor, D};
use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

/// One token (or terminal marker) streamed back to a request.
pub enum CbToken {
    /// The request's prompt started at `at`, `cached_prompt_tokens` of it served from the
    /// cache; its tokens follow.
    Admitted {
        at: std::time::Instant,
        cached_prompt_tokens: usize,
    },
    Tok(u32),
    Done(FinishReason),
}

/// A submitted generation request the worker should serve.
struct Submission {
    prompt: Vec<u32>,
    max_new: usize,
    sampling: SamplingParams,
    tx: Sender<CbToken>,
}

/// The paged KV pool of a stopped worker, kept for its next start.
///
/// A parked pool keeps its blocks on the card while the model serves serially.
/// The next worker reuses them and takes no new card memory.
pub struct ParkedPool {
    stores: Vec<PagedKvStore>,
    num_blocks: usize,
    block_size: usize,
    max_running: usize,
    max_prefill_tokens: usize,
}

// The stores are touched by one thread at a time: the worker that owns them, or none while
// parked.
unsafe impl Send for ParkedPool {}

impl ParkedPool {
    /// The blocks of the pool and the sequences it runs at once.
    pub fn size(&self) -> (usize, usize) {
        (self.num_blocks, self.max_running)
    }
}

/// Handle to the running continuous-batch worker. Cheap to clone-share via Arc.
pub struct ContinuousServer {
    submit_tx: Sender<Submission>,
    /// The worker thread, joined by `stop`.
    worker: Worker,
    /// The served model when the worker shares it; `stop` takes it back once the worker ends.
    /// `None` when the worker owns it.
    model: Option<Arc<GenericHeteroTransformer>>,
}

/// The model-side adapter: decode items run in ONE batched `batched_paged_decode`
/// (the throughput win); prefill items loop their prompt positions. Identical
/// math to the validated `cb_bench` adapter.
struct GhBatched {
    model: Arc<GenericHeteroTransformer>,
    stores: Vec<PagedKvStore>,
    // CUDA-graph cache for the batched decode step, keyed by (batch_width, ctx
    // capacity bucket). Each entry holds the captured graph + the persistent
    // input/output buffers refreshed (slice_set) before every replay.
    #[cfg(feature = "gpu")]
    graphs: std::collections::HashMap<(usize, usize), CapturedDecode>,
    // Set once if capture ever fails -> permanently use the eager path.
    #[cfg(feature = "gpu")]
    graph_failed: bool,
    pace: StepPace,
}

/// The time the batched steps of the last second took, logged once a second while the
/// server works: what a batch costs against its width.
#[derive(Default)]
struct StepPace {
    since: Option<std::time::Instant>,
    decode_steps: u32,
    decode_rows: u32,
    decode: std::time::Duration,
    prompts: u32,
    prompt_tokens: u32,
    prefill: std::time::Duration,
}

impl StepPace {
    fn report_each_second(&mut self) {
        let since = *self.since.get_or_insert_with(std::time::Instant::now);
        if since.elapsed() < std::time::Duration::from_secs(1) {
            return;
        }
        let steps = self.decode_steps.max(1);
        tracing::info!(
            "CB pace: {} decode steps, {:.1} rows each, {:.2} ms a step; {} prompts of {} tokens \
             in {:.0} ms",
            self.decode_steps,
            self.decode_rows as f64 / steps as f64,
            self.decode.as_secs_f64() * 1e3 / steps as f64,
            self.prompts,
            self.prompt_tokens,
            self.prefill.as_secs_f64() * 1e3
        );
        *self = Self {
            since: Some(std::time::Instant::now()),
            ..Self::default()
        };
    }
}

#[cfg(feature = "gpu")]
impl Drop for GhBatched {
    fn drop(&mut self) {
        // The captured graphs go before the model and the paged KV stores they read.
        self.graphs.clear();
    }
}

/// A captured batched-decode graph + its stable I/O buffers (one per (B,cap)).
#[cfg(feature = "gpu")]
struct CapturedDecode {
    graph: crate::tensor::cuda_ext::CapturedGraph,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    block_table: Tensor,
    slot: Tensor,
    seq_lens: Tensor,
    hidden: Tensor,
}

/// Context capacity bucket: gather/mask are padded to the next multiple of this so
/// the captured shapes stay fixed as context grows (recapture only on bucket cross).
pub const CB_GRAPH_BUCKET: usize = 256;

/// Tokens per block of the paged cache.
pub const CB_BLOCK_SIZE: usize = 16;

fn argmax_row(logits: &Tensor, row: usize) -> crate::tensor::Result<u32> {
    let row = logits.narrow(0, row, 1)?.flatten_all()?;
    // An f32 row on a card is reduced by the card sampler, which allocates nothing per token.
    if row.device().is_cuda() && row.dtype() == DType::F32 {
        return crate::inference::kernel::fused::fused_penalty_argmax(&row, &[], 1.0);
    }
    row.to_dtype(DType::F32)?
        .argmax(D::Minus1)?
        .to_dtype(DType::U32)?
        .to_scalar::<u32>()
}

/// Select one token from `logits` row `row` under per-request sampling controls.
/// Greedy (`temperature==0`, no penalty) takes the exact argmax fast path, so
/// greedy requests are bit-identical to the non-sampling worker. Otherwise apply
/// repeat-penalty over `recent` then temperature/top-k/top-p via the shared
/// `LogitsProcessor` (same sampler the serial engine uses -> parity).
fn sample_row(
    logits: &Tensor,
    row: usize,
    sp: &SamplingParams,
    recent: &[u32],
) -> crate::tensor::Result<u32> {
    if sp.is_greedy() {
        return argmax_row(logits, row);
    }
    use crate::inference::sample::token_sampling::{LogitsProcessor, Sampling};
    let mut v: Vec<f32> = logits
        .narrow(0, row, 1)?
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1()?;
    if sp.repeat_penalty > 1.0 && !recent.is_empty() {
        for &tok in recent {
            let idx = tok as usize;
            if idx < v.len() {
                if v[idx] > 0.0 {
                    v[idx] /= sp.repeat_penalty;
                } else {
                    v[idx] *= sp.repeat_penalty;
                }
            }
        }
    }
    // Penalty WITHOUT temperature (temp==0) = greedy over the penalized logits.
    // Routing this through a temperature sampler would divide by zero -> NaN.
    if sp.temperature <= 0.0 {
        let mut best = 0u32;
        let mut bestv = f32::NEG_INFINITY;
        for (i, &x) in v.iter().enumerate() {
            if x > bestv {
                bestv = x;
                best = i as u32;
            }
        }
        return Ok(best);
    }
    let n = v.len();
    // Sample on the HOST (native top-k/top-p/multinomial), NOT on the device: building
    // the tensor on `logits.device()` (GPU) routed each row through the device sort/
    // softmax per step - profiled ~31% slower than the greedy native-kernel path at B=8.
    // The serial sampler already samples on host; this matches it (parity) and detaches
    // the CB sampled path from the device GPU ops.
    let logits_1d = Tensor::from_vec(v, &[n][..], &crate::tensor::Device::Cpu)?;
    let sampling = match (sp.top_k, sp.top_p) {
        (Some(k), Some(p)) => Sampling::TopKThenTopP {
            k,
            p,
            temperature: sp.temperature,
        },
        (Some(k), None) => Sampling::TopK {
            k,
            temperature: sp.temperature,
        },
        (None, Some(p)) => Sampling::TopP {
            p,
            temperature: sp.temperature,
        },
        (None, None) => Sampling::All {
            temperature: sp.temperature,
        },
    };
    // Vary the seed per decode step (recent grows each step) so a fixed-seed
    // request still advances its sample stream rather than redrawing identically.
    let seed = sp
        .seed
        .wrapping_add(row as u64)
        .wrapping_add(recent.len() as u64);
    LogitsProcessor::from_sampling(seed, sampling).sample(&logits_1d)
}

/// Host-side per-row sampler over already-transferred logits `v` (one row of [B, vocab]).
/// Same math as `sample_row` (parity) but takes a host slice so B rows can be sampled in
/// PARALLEL (rayon) after ONE batched device->host transfer - the per-row softmax + top-k
/// select + multinomial are compute-heavy, so fanning them across cores cuts wall time.
/// `seed_off` varies the RNG per row (mirrors `sample_row`'s `row`-offset seed).
pub(crate) fn sample_row_host(
    mut v: Vec<f32>,
    sp: &SamplingParams,
    recent: &[u32],
    seed_off: usize,
) -> crate::tensor::Result<u32> {
    use crate::inference::sample::token_sampling::{LogitsProcessor, Sampling};
    if sp.repeat_penalty > 1.0 && !recent.is_empty() {
        for &tok in recent {
            let idx = tok as usize;
            if idx < v.len() {
                if v[idx] > 0.0 {
                    v[idx] /= sp.repeat_penalty;
                } else {
                    v[idx] *= sp.repeat_penalty;
                }
            }
        }
    }
    if sp.temperature <= 0.0 {
        let (mut best, mut bv) = (0u32, f32::NEG_INFINITY);
        for (i, &x) in v.iter().enumerate() {
            if x > bv {
                bv = x;
                best = i as u32;
            }
        }
        return Ok(best);
    }
    let n = v.len();
    let logits_1d = Tensor::from_vec(v, &[n][..], &crate::tensor::Device::Cpu)?;
    let sampling = match (sp.top_k, sp.top_p) {
        (Some(k), Some(p)) => Sampling::TopKThenTopP {
            k,
            p,
            temperature: sp.temperature,
        },
        (Some(k), None) => Sampling::TopK {
            k,
            temperature: sp.temperature,
        },
        (None, Some(p)) => Sampling::TopP {
            p,
            temperature: sp.temperature,
        },
        (None, None) => Sampling::All {
            temperature: sp.temperature,
        },
    };
    let seed = sp
        .seed
        .wrapping_add(seed_off as u64)
        .wrapping_add(recent.len() as u64);
    LogitsProcessor::from_sampling(seed, sampling).sample(&logits_1d)
}

/// Finish sampling for one row from the GPU top-k+denom output: reconstruct full-vocab
/// probs `exp((l-rowmax)/T)/denom` (descending), apply top-p, and draw a multinomial token.
/// Correct top-k/top-p/temperature sampling (same distribution as `sample_row_host`); the
/// exact token can differ (the multinomial RNG order differs), which is fine for random
/// sampling. Deterministic given the seed.
#[cfg(feature = "gpu")]
fn sampled_finish_host(
    top_logit: &[f32],
    top_idx: &[u32],
    rowmax: f32,
    denom: f32,
    inv_t: f32,
    top_p: Option<f64>,
    sp: &SamplingParams,
    seed_off: usize,
) -> u32 {
    use rand::distr::Distribution;
    use rand::SeedableRng;
    let mut probs: Vec<f32> = top_logit
        .iter()
        .map(|&l| ((l - rowmax) * inv_t).exp() / denom)
        .collect();
    if let Some(p) = top_p {
        let p = p as f32;
        let mut cum = 0.0f32;
        for pr in probs.iter_mut() {
            if cum >= p {
                *pr = 0.0;
            } else {
                cum += *pr;
            }
        }
    }
    let seed = sp.seed.wrapping_add(seed_off as u64);
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    match rand::distr::weighted::WeightedIndex::new(&probs) {
        Ok(w) => top_idx[w.sample(&mut rng)],
        Err(_) => top_idx[0], // all-zero (degenerate) -> the argmax
    }
}

impl BatchedModel for GhBatched {
    fn step(&mut self, items: &[StepItem]) -> crate::tensor::Result<Vec<u32>> {
        let mut out = vec![0u32; items.len()];
        // -- batched decode (B = #decode items, one forward) --
        let dec: Vec<usize> = (0..items.len()).filter(|&i| !items[i].is_prefill).collect();
        let decode_start = std::time::Instant::now();
        if !dec.is_empty() {
            let toks: Vec<u32> = dec.iter().map(|&i| items[i].tokens[0]).collect();
            let pos: Vec<usize> = dec.iter().map(|&i| items[i].context_len - 1).collect();
            let slots: Vec<usize> = dec.iter().map(|&i| items[i].slots[0]).collect();
            let bts: Vec<Vec<u32>> = dec.iter().map(|&i| items[i].block_table.clone()).collect();
            let ctx: Vec<usize> = dec.iter().map(|&i| items[i].context_len).collect();
            // CUDA-graph decode is what delivers the throughput win (eager batching
            // alone roughly ties serial on big models); always on, with a safe eager
            // fallback latched via `graph_failed` on any capture/replay failure.
            // Every row decodes greedily: the head may answer in half precision.
            #[cfg(feature = "gpu")]
            let greedy = dec.iter().all(|&i| items[i].sampling.temperature <= 0.0);
            #[cfg(feature = "gpu")]
            let graph_on = !self.graph_failed
                && !self.model.compute_device().is_cpu()
                && self.model.cb_eligible()
                && self.model.cb_graph_serves();
            #[cfg(not(feature = "gpu"))]
            let graph_on = false;
            let logits = if graph_on {
                #[cfg(feature = "gpu")]
                {
                    // Try the captured-graph path; on ANY capture/replay failure
                    // fall back to the eager path (correct, just unaccelerated) so
                    // the flag can never break serving.
                    match self.decode_graph(&toks, &pos, &slots, &bts, &ctx, greedy) {
                        Ok(l) => l,
                        Err(e) => {
                            let msg = format!("{e}");
                            // Any capture or replay failure latches the graph off.
                            eprintln!("cb graph disabled permanently (fallback to eager): {msg}");
                            self.graph_failed = true;
                            self.model.batched_paged_decode(
                                &toks,
                                &pos,
                                &slots,
                                &bts,
                                &ctx,
                                &mut self.stores,
                            )?
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    unreachable!()
                }
            } else {
                self.model.batched_paged_decode(
                    &toks,
                    &pos,
                    &slots,
                    &bts,
                    &ctx,
                    &mut self.stores,
                )?
            };
            // Sampling. When every decode row is DETERMINISTIC (temperature==0 - argmax,
            // optionally over repeat-penalized logits), pick all B tokens with ONE custom
            // batched-argmax CUDA launch instead of B per-row argmax + host scans
            // (profiled ~2ms/step at B=8; the per-row argmax and host paths are memory/
            // overhead-bound). The penalty only LOWERS `recent` tokens, so the raw argmax
            // is exact unless the winner is itself penalized - recompute those rare rows.
            #[cfg(feature = "gpu")]
            let fast_argmax = dec.iter().all(|&i| items[i].sampling.temperature <= 0.0)
                && !self.model.compute_device().is_cpu();
            #[cfg(not(feature = "gpu"))]
            let fast_argmax = false;
            if fast_argmax {
                #[cfg(feature = "gpu")]
                {
                    let raw = crate::inference::moe_cuda::batched_argmax(&logits)?;
                    for (j, &i) in dec.iter().enumerate() {
                        let sp = &items[i].sampling;
                        out[i] = if sp.repeat_penalty > 1.0
                            && items[i].recent_tokens.contains(&raw[j])
                        {
                            sample_row(&logits, j, sp, &items[i].recent_tokens)?
                        } else {
                            raw[j]
                        };
                    }
                }
            } else {
                // UNIFORM sampled batch (same temp>0, top_k) -> our GPU top-k+denom+penalty
                // kernel + a host top-p/multinomial over just k entries (transfers [B,k] not
                // [B,vocab]; vLLM-style on-device sampler). Else fall back to the per-row
                // host sampler across cores (rayon over a single [B,vocab] transfer).
                let s0 = &items[dec[0]].sampling;
                // The GPU top-k does k sequential block-argmax passes -> latency-bound at
                // small batch (underfills the SMs); it only beats the host-rayon sampler
                // once enough rows are in flight (measured crossover ~B=12: B=8 loses, B>=16
                // wins +19-24%). Gate on batch width.
                #[cfg(feature = "gpu")]
                let gpu_ok = dec.len() >= 12
                    && s0.temperature > 0.0
                    && s0.top_k.is_some()
                    && !self.model.compute_device().is_cpu()
                    && dec.iter().all(|&i| {
                        let s = &items[i].sampling;
                        s.temperature == s0.temperature
                            && s.top_k == s0.top_k
                            && s.top_p == s0.top_p
                            && s.repeat_penalty == s0.repeat_penalty
                    });
                #[cfg(not(feature = "gpu"))]
                let gpu_ok = false;
                if gpu_ok {
                    #[cfg(feature = "gpu")]
                    {
                        use rayon::prelude::*;
                        let k = s0.top_k.unwrap();
                        let (temp, rp, top_p) = (s0.temperature, s0.repeat_penalty, s0.top_p);
                        let (mut prows, mut ptoks) = (Vec::new(), Vec::new());
                        if rp > 1.0 {
                            for (j, &i) in dec.iter().enumerate() {
                                for &t in &items[i].recent_tokens {
                                    prows.push(j as i32);
                                    ptoks.push(t);
                                }
                            }
                        }
                        let (tl, ti, ts) = crate::inference::moe_cuda::batched_topk_denom(
                            &logits, k, temp, &prows, &ptoks, rp,
                        )?;
                        let inv_t = (1.0 / temp) as f32;
                        let outs: Vec<u32> = dec
                            .par_iter()
                            .enumerate()
                            .map(|(j, &i)| {
                                sampled_finish_host(
                                    &tl[j * k..(j + 1) * k],
                                    &ti[j * k..(j + 1) * k],
                                    ts[j * 2],
                                    ts[j * 2 + 1],
                                    inv_t,
                                    top_p,
                                    &items[i].sampling,
                                    j,
                                )
                            })
                            .collect();
                        for (j, &i) in dec.iter().enumerate() {
                            out[i] = outs[j];
                        }
                    }
                } else {
                    use rayon::prelude::*;
                    let host = logits.to_dtype(DType::F32)?.to_vec2::<f32>()?;
                    let ids: crate::tensor::Result<Vec<u32>> = dec
                        .par_iter()
                        .enumerate()
                        .map(|(j, &i)| {
                            sample_row_host(
                                host[j].clone(),
                                &items[i].sampling,
                                &items[i].recent_tokens,
                                j,
                            )
                        })
                        .collect();
                    let ids = ids?;
                    for (j, &i) in dec.iter().enumerate() {
                        out[i] = ids[j];
                    }
                }
            }
        }
        if !dec.is_empty() {
            self.pace.decode_steps += 1;
            self.pace.decode_rows += dec.len() as u32;
            self.pace.decode += decode_start.elapsed();
        }
        // -- prefills: every prompt in ONE forward, the weights read once. A prompt whose prefix
        // the cache holds computes only its suffix. --
        let prefill_start = std::time::Instant::now();
        let mut plain = Vec::new();
        let mut cached = Vec::new();
        for (i, it) in items.iter().enumerate() {
            if !it.is_prefill {
                continue;
            }
            self.pace.prompts += 1;
            self.pace.prompt_tokens += it.tokens.len() as u32;
            if it.cached_len == 0 {
                plain.push(i);
            } else {
                cached.push(i);
            }
        }
        if !plain.is_empty() || !cached.is_empty() {
            let plain_seqs: Vec<(&[u32], &[usize])> = plain
                .iter()
                .map(|&i| (items[i].tokens.as_slice(), items[i].slots.as_slice()))
                .collect();
            let cached_seqs: Vec<crate::inference::generic_transformer::CachedPrompt<'_>> = cached
                .iter()
                .map(|&i| crate::inference::generic_transformer::CachedPrompt {
                    tokens: items[i].tokens.as_slice(),
                    suffix_slots: items[i].slots.as_slice(),
                    cached_len: items[i].cached_len,
                    block_table: items[i].block_table.as_slice(),
                })
                .collect();
            let l = self
                .model
                .paged_prefill_mixed(&plain_seqs, &cached_seqs, &mut self.stores)?;
            for (row, &i) in plain.iter().chain(cached.iter()).enumerate() {
                out[i] = sample_row(&l, row, &items[i].sampling, &items[i].recent_tokens)?;
            }
        }
        self.pace.prefill += prefill_start.elapsed();
        self.pace.report_each_second();
        Ok(out)
    }
}

/// The model-side adapter of a model whose layers keep each sequence's state beside the paged
/// store: a Sinkhorn wide residual over latent and band layers, its token mixers on the host or
/// on a card.
///
/// It holds 1 [`SequenceMixers`] per sequence the engine runs, with the tokens its state stands
/// after.
/// A prompt starts its sequence again from position 0, whatever part of it the paged prefix
/// cache holds.
/// With a state cache on the model, the prompt starts from the longest stored state it extends.
/// The states its prefill passes and the state a finished sequence stands in are kept.
/// The decode items of a step run as 1 batch across their sequences.
pub(crate) struct SequenceBatched {
    model: GenericHeteroTransformer,
    seqs: HashMap<SeqId, Sequence>,
}

/// What [`SequenceBatched`] keeps for 1 sequence.
struct Sequence {
    mixers: SequenceMixers,
    /// The tokens the state stands after, the prompt first.
    fed: Vec<u32>,
    /// The prompt tokens a stored state covered at the prefill.
    covered: usize,
}

impl SequenceBatched {
    /// Refused for a model whose layers do not keep latent and band states.
    pub(crate) fn new(model: GenericHeteroTransformer) -> crate::tensor::Result<Self> {
        model.sequence_mixers()?;
        Ok(Self {
            model,
            seqs: HashMap::new(),
        })
    }

    /// The logits of the last row of `tokens` read from position 0 over `seq`, `[1, vocab]`, and
    /// the tokens a stored state covered.
    ///
    /// Without a state cache on the model, the prompt is 1 forward and covers none.
    fn prefill(
        &mut self,
        tokens: &[u32],
        seq: &mut SequenceMixers,
    ) -> crate::tensor::Result<(Tensor, usize)> {
        let Some(cache) = self.model.state_cache().filter(|_| tokens.len() > 1) else {
            return Ok((self.model.prefill_sequence(tokens, seq)?, 0));
        };
        self.model.with_sequence(seq, |model| {
            let covered = cache.restore(model, tokens);
            let rest = tokens.len() - covered;
            let started = std::time::Instant::now();
            let (logits, grid) = model.prefill_with_grid(
                tokens,
                covered,
                rest,
                cache.stride(rest),
                cache.layout(),
            )?;
            let secs = started.elapsed().as_secs_f64();
            cache.tuning().prefilled(rest, secs);
            cache.save(model, tokens, grid, Some(secs));
            Ok((logits, covered))
        })
    }
}

impl BatchedModel for SequenceBatched {
    fn step(&mut self, items: &[StepItem]) -> crate::tensor::Result<Vec<u32>> {
        let mut out = vec![0u32; items.len()];
        for (i, it) in items.iter().enumerate().filter(|(_, it)| it.is_prefill) {
            let mut mixers = self.model.sequence_mixers()?;
            // A prompt that runs out of card memory runs again once the lent card room is given
            // back.
            let (logits, covered) = match self.prefill(&it.tokens, &mut mixers) {
                Err(e) if e.is_oom() => {
                    self.model.give_back_all_lent_room();
                    mixers = self.model.sequence_mixers()?;
                    self.prefill(&it.tokens, &mut mixers)?
                }
                done => done?,
            };
            out[i] = sample_row(&logits, 0, &it.sampling, &it.recent_tokens)?;
            let seq = Sequence {
                mixers,
                fed: it.tokens.clone(),
                covered,
            };
            self.seqs.insert(it.seq, seq);
        }
        let dec: Vec<usize> = (0..items.len()).filter(|&i| !items[i].is_prefill).collect();
        if dec.is_empty() {
            return Ok(out);
        }
        let tokens: Vec<u32> = dec.iter().map(|&i| items[i].tokens[0]).collect();
        let positions: Vec<usize> = dec.iter().map(|&i| items[i].context_len - 1).collect();
        // Each decoded sequence's state leaves the map for the step and comes back after it.
        let mut taken = Vec::with_capacity(dec.len());
        for &i in &dec {
            match self.seqs.remove(&items[i].seq) {
                Some(seq) => taken.push((items[i].seq, seq)),
                None => {
                    for (id, seq) in taken {
                        self.seqs.insert(id, seq);
                    }
                    return Err(crate::tensor::Error::msg(format!(
                        "sequence {} decodes without a prompt",
                        items[i].seq
                    )));
                }
            }
        }
        let logits = {
            let mut refs: Vec<&mut SequenceMixers> =
                taken.iter_mut().map(|(_, s)| &mut s.mixers).collect();
            self.model.decode_sequences(&tokens, &positions, &mut refs)
        };
        let fed = logits.is_ok();
        for ((id, mut seq), &t) in taken.into_iter().zip(&tokens) {
            if fed {
                seq.fed.push(t);
            }
            self.seqs.insert(id, seq);
        }
        let logits = logits?;
        for (j, &i) in dec.iter().enumerate() {
            out[i] = sample_row(&logits, j, &items[i].sampling, &items[i].recent_tokens)?;
        }
        Ok(out)
    }

    fn cached_prompt(&self, seq: SeqId) -> Option<usize> {
        self.seqs.get(&seq).map(|s| s.covered)
    }

    /// The state the sequence stands in is kept when the model has a state cache.
    /// The cache refuses a state whose layers do not all stand after the fed tokens.
    fn release(&mut self, seq: SeqId) {
        let Some(mut s) = self.seqs.remove(&seq) else {
            return;
        };
        let Some(cache) = self.model.state_cache() else {
            return;
        };
        let fed = &s.fed;
        let kept = self.model.with_sequence(&mut s.mixers, |model| {
            cache.save(model, fed, Vec::new(), None);
            Ok(())
        });
        if let Err(e) = kept {
            tracing::info!("state cache: sequence {seq} not stored: {e}");
        }
    }
}

/// What the adapter of a worker hands back once the worker ends.
trait Served: BatchedModel {
    /// The model the adapter owned, `None` when it shared it, and its paged stores, `None`
    /// when it keeps none.
    fn into_parts(self) -> (Option<GenericHeteroTransformer>, Option<Vec<PagedKvStore>>);
}

impl Served for GhBatched {
    fn into_parts(mut self) -> (Option<GenericHeteroTransformer>, Option<Vec<PagedKvStore>>) {
        // The captured graphs go before the stores they read leave the adapter.
        #[cfg(feature = "gpu")]
        self.graphs.clear();
        (None, Some(std::mem::take(&mut self.stores)))
    }
}

impl Served for SequenceBatched {
    fn into_parts(self) -> (Option<GenericHeteroTransformer>, Option<Vec<PagedKvStore>>) {
        (Some(self.model), None)
    }
}

#[cfg(feature = "gpu")]
impl GhBatched {
    /// Batched decode via a captured CUDA graph: capture once per (B, ctx-cap
    /// bucket), then replay (just refresh the stable input buffers + launch)  -
    /// removing the ~per-step kernel-launch overhead that caps the eager path.
    /// Returns logits `[B, vocab]` in a persistent buffer.
    fn decode_graph(
        &mut self,
        toks: &[u32],
        pos: &[usize],
        slots: &[usize],
        bts: &[Vec<u32>],
        ctx: &[usize],
        greedy: bool,
    ) -> crate::tensor::Result<Tensor> {
        use crate::tensor::cuda_ext;
        let Self {
            model,
            stores,
            graphs,
            ..
        } = self;
        let dev = model.compute_device();
        let (n_kv, hd, _nl, vocab) = model.paged_geometry();
        let feat = n_kv * hd;
        let base = model.rope_base();
        let _ = feat;
        // Batch-width PADDING: round the live batch up to a power-of-two bucket and
        // pad by REPLICATING row 0. The duplicate rows recompute row 0's identical
        // K/V and write it to row 0's own slot (idempotent - no corruption), and
        // their logits (rows raw_b..bpad) are ignored by the caller (it reads only
        // the real rows 0..raw_b). This caps the distinct (B,cap) graphs to the few
        // buckets {1,2,4,8,16,32} instead of one per live width -> far fewer
        // recaptures as concurrency churns.
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
        let (toks, pos, slots, bts, ctx) = (
            &toks_v[..],
            &pos_v[..],
            &slots_v[..],
            &bts_v[..],
            &ctx_v[..],
        );
        let b = toks.len(); // == bpad
        let block_size = stores[0].block_size();
        let max_ctx = *ctx.iter().max().unwrap();
        let cap = max_ctx.div_ceil(CB_GRAPH_BUCKET) * CB_GRAPH_BUCKET; // ctx capacity bucket
        let max_blocks = cap / block_size; // fixed block-table width
        let key = (b, max_blocks);

        // Per-step dynamic inputs (host->device), refreshed into stable buffers each
        // replay. All paged indices live in DEVICE i32 buffers the capture-safe
        // kernels read at exec time (slot/block_table/seq_lens).
        let x = model.embed_batch(toks)?;
        let (cos, sin) = crate::inference::cache::paged_attention::rope_cos_sin(
            pos,
            hd,
            base,
            &dev,
            DType::F16,
        )?;
        let slot_dev = Tensor::from_vec(
            slots.iter().map(|&s| s as i32).collect::<Vec<_>>(),
            &[b][..],
            &dev,
        )?;
        let seq_lens_dev = Tensor::from_vec(
            ctx.iter().map(|&c| c as i32).collect::<Vec<_>>(),
            &[b][..],
            &dev,
        )?;
        let mut btf = vec![0i32; b * max_blocks];
        for (bi, bt) in bts.iter().enumerate() {
            for (j, &blk) in bt.iter().take(max_blocks).enumerate() {
                btf[bi * max_blocks + j] = blk as i32;
            }
        }
        let block_table_dev = Tensor::from_vec(btf, &[b, max_blocks][..], &dev)?;

        if let Some(g) = graphs.get(&key) {
            // replay: refresh the stable buffers in place, then launch.
            let lbl = |n: &str, r: crate::tensor::Result<()>| {
                r.map_err(|e| crate::tensor::Error::msg(format!("slice_set[{n}]: {e}")))
            };
            lbl("x", g.x.slice_set(&x, 0, 0))?;
            lbl("cos", g.cos.slice_set(&cos, 0, 0))?;
            lbl("sin", g.sin.slice_set(&sin, 0, 0))?;
            lbl("slot", g.slot.slice_set(&slot_dev, 0, 0))?;
            lbl("seq", g.seq_lens.slice_set(&seq_lens_dev, 0, 0))?;
            lbl("bt", g.block_table.slice_set(&block_table_dev, 0, 0))?;
            // The buffer-refresh slice_sets and the graph replay must be ordered:
            // sync so the refreshed device buffers are visible before the graph reads
            // them (else the graph replays on stale capture-time inputs).
            if let Ok(cd) = dev.as_cuda_device() {
                let _ = cd.cuda_stream().synchronize();
            }
            g.graph
                .launch()
                .map_err(|e| crate::tensor::Error::msg(format!("cb graph launch: {e:?}")))?;
            // The graph launch is ASYNC; the eager lm_head downloads hidden_buf to
            // host (CPU output_proj) - sync so the graph has finished WRITING
            // hidden_buf before lm_head reads it (else it reads uninitialized data).
            if let Ok(cd) = dev.as_cuda_device() {
                let _ = cd.cuda_stream().synchronize();
            }
            // lm_head runs eager (may live on CPU); the graph produced the hidden.
            return if greedy {
                model.lm_head_greedy(&g.hidden)
            } else {
                model.lm_head(&g.hidden)
            };
        }

        // capture: the just-built tensors become the persistent buffers. Keep the
        // captured region single-dtype (F16) - no F32 cast inside (cast kernels
        // break capture stream isolation); argmax converts F16->F32 outside.
        let hidden_sz = x.dims()[1];
        let _ = vocab;
        let cd = dev.as_cuda_device()?;
        let stream = cd.cuda_stream();
        // WARMUP: run the region eagerly ONCE before capture. (a) loads every
        // kernel module (lazy cuModuleLoad during capture is illegal -> the masked
        // "previous error"); (b) computes + writes THIS step's KV/logits, since
        // stream-capture only RECORDS (doesn't execute) - so this eager pass is the
        // first step's real result. Capture then records the op sequence for replay.
        let eager_hidden = model.decode_layers_gpu(
            &x,
            &cos,
            &sin,
            &block_table_dev,
            &slot_dev,
            &seq_lens_dev,
            b,
            max_blocks,
            stores,
        )?;
        let eager_logits = model.lm_head(&eager_hidden)?;
        // Settle all pre-capture input prep (embed, to_device, the index from_vecs)
        // so the captured region has no live cross-stream dependency on uncaptured
        // work (else CUDA_ERROR_STREAM_CAPTURE_ISOLATION).
        let _ = stream.synchronize();
        // Re-home the stable per-step INPUT buffers just before the capture: the graph reads
        // them at fixed addresses and each replay refreshes them in place, so they are
        // allocated outside the capture and live as long as the graph.
        let stable_copy = |t: &Tensor| -> crate::tensor::Result<Tensor> {
            let a = Tensor::zeros_on(t.dims(), t.dtype(), &dev)?;
            a.slice_set(t, 0, 0)?;
            Ok(a)
        };
        let x = stable_copy(&x)?;
        let cos = stable_copy(&cos)?;
        let sin = stable_copy(&sin)?;
        let slot_dev = stable_copy(&slot_dev)?;
        let seq_lens_dev = stable_copy(&seq_lens_dev)?;
        let block_table_dev = stable_copy(&block_table_dev)?;
        // The final-norm output is copied inside the capture into a buffer allocated before
        // it: the graph then allocates nothing it keeps, and the buffer holds the hidden
        // state after each replay.
        let _ = hidden_sz;
        let hidden_buf = Tensor::zeros_on(eager_hidden.dims(), eager_hidden.dtype(), &dev)?;
        let cap_res: crate::tensor::Result<cuda_ext::CapturedGraph> = (|| {
            cuda_ext::begin_capture(&stream)?;
            let hf = model.decode_layers_gpu(
                &x,
                &cos,
                &sin,
                &block_table_dev,
                &slot_dev,
                &seq_lens_dev,
                b,
                max_blocks,
                stores,
            )?;
            hidden_buf.slice_set(&hf, 0, 0)?;
            drop(hf);
            cuda_ext::end_capture(&stream)?
                .ok_or_else(|| crate::tensor::Error::msg("cb capture: null graph"))
        })();
        let graph = match cap_res {
            Ok(g) => {
                let nnodes = g.num_nodes().unwrap_or(0);
                // DEFENSE IN DEPTH: a 0-node graph recorded no work - replaying
                // it would leave the hidden/logits buffers frozen -> constant
                // tokens for every sequence in the batch. Err here latches
                // `graph_failed` in the caller -> permanent eager fallback.
                if nnodes == 0 {
                    drop(g);
                    return Err(crate::tensor::Error::msg(
                        "cb capture: empty graph (0 nodes) - refusing to replay".to_string(),
                    ));
                }
                eprintln!("cb graph captured (B={b} cap={cap}) nodes={nnodes}");
                crate::tensor::cuda::trim_pool(dev.as_cuda_device()?.ordinal());
                g
            }
            Err(e) => {
                let _ = cuda_ext::end_capture(&stream);
                let _ = stream.synchronize();
                return Err(e);
            }
        };
        graphs.insert(
            key,
            CapturedDecode {
                graph,
                x,
                cos,
                sin,
                block_table: block_table_dev,
                slot: slot_dev,
                seq_lens: seq_lens_dev,
                hidden: hidden_buf,
            },
        );
        Ok(eager_logits)
    }
}

// The worker thread is the sole owner/caller of the model; CUDA handles inside
// are only ever touched from that one thread (same contract as ModelVariant).
struct SendWrap<T>(T);
unsafe impl<T> Send for SendWrap<T> {}

/// The worker thread a server joins: it ends with the model its adapter owned, if it owned one,
/// and with its paged pool, if it kept one.
type Worker =
    std::thread::JoinHandle<SendWrap<(Option<GenericHeteroTransformer>, Option<ParkedPool>)>>;

/// Starts the worker thread over `adapter`, and the channel requests reach it through.
///
/// The adapter crosses to the thread once the thread runs.
/// `Err` hands the adapter back when no thread takes it.
#[allow(clippy::type_complexity)]
fn launch<M: Served + 'static>(
    adapter: M,
    eos: Vec<u32>,
    num_blocks: usize,
    block_size: usize,
    max_running: usize,
    max_prefill_tokens: usize,
) -> std::result::Result<(Sender<Submission>, Worker), (M, crate::tensor::Error)> {
    let (submit_tx, submit_rx) = mpsc::channel::<Submission>();
    let (hand_tx, hand_rx) = mpsc::channel::<SendWrap<M>>();
    let worker = std::thread::Builder::new()
        .name("cb-worker".into())
        .spawn(move || {
            let Ok(SendWrap(adapter)) = hand_rx.recv() else {
                return SendWrap((None, None));
            };
            let ended = worker_loop(
                adapter,
                eos,
                num_blocks,
                block_size,
                max_running,
                max_prefill_tokens,
                submit_rx,
            );
            let Some((model, stores)) = ended.map(Served::into_parts) else {
                return SendWrap((None, None));
            };
            let parked = stores.map(|stores| ParkedPool {
                stores,
                num_blocks,
                block_size,
                max_running,
                max_prefill_tokens,
            });
            SendWrap((model, parked))
        });
    let worker = match worker {
        Ok(w) => w,
        Err(e) => {
            return Err((
                adapter,
                crate::tensor::Error::msg(format!("cb-worker spawn: {e}")),
            ))
        }
    };
    match hand_tx.send(SendWrap(adapter)) {
        Ok(()) => Ok((submit_tx, worker)),
        Err(mpsc::SendError(SendWrap(adapter))) => Err((
            adapter,
            crate::tensor::Error::msg("cb-worker ended before it took its model"),
        )),
    }
}

impl ContinuousServer {
    /// Spawn the worker. `model` is moved in and owned for the worker's lifetime;
    /// `eos` terminates a sequence; `num_blocks`/`block_size` size the paged KV
    /// pool; `max_running` caps the concurrent decode batch width.
    pub fn spawn(
        model: Arc<GenericHeteroTransformer>,
        eos: Vec<u32>,
        num_blocks: usize,
        block_size: usize,
        max_running: usize,
        max_prefill_tokens: usize,
    ) -> crate::tensor::Result<Self> {
        let (n_kv, hd, n_layers, _vocab) = model.paged_geometry();
        let cdev = model.compute_device();
        let store_dt = if cdev.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };
        let stores: Vec<PagedKvStore> = (0..n_layers)
            .map(|_| PagedKvStore::new(num_blocks, block_size, n_kv, hd, store_dt, &cdev))
            .collect::<crate::tensor::Result<_>>()?;
        Self::spawn_over(
            model,
            eos,
            ParkedPool {
                stores,
                num_blocks,
                block_size,
                max_running,
                max_prefill_tokens,
            },
        )
    }

    /// Spawn the worker over the paged `pool` a stopped worker kept.
    pub fn spawn_over(
        model: Arc<GenericHeteroTransformer>,
        eos: Vec<u32>,
        pool: ParkedPool,
    ) -> crate::tensor::Result<Self> {
        let ParkedPool {
            stores,
            num_blocks,
            block_size,
            max_running,
            max_prefill_tokens,
        } = pool;
        let cdev = model.compute_device();
        // Pin a cuBLAS workspace so the batched-decode GEMMs stop allocating a
        // workspace per call (per-call allocs are pure overhead in the hot loop,
        // and are also required to be absent before any future graph capture).
        #[cfg(feature = "gpu")]
        if !cdev.is_cpu() {
            let _ = crate::tensor::cuda_ext::pin_cublas_workspace(&cdev, 256 << 20);
            // Disable event tracking: the CB worker is single-stream, so the
            // multi-stream sync events aren't needed - and creating CUDA events
            // during allocations INSIDE a graph capture invalidates the capture
            // (required for the captured decode graph).
            if let Ok(cd) = cdev.as_cuda_device() {
                unsafe {
                    cd.disable_event_tracking();
                }
            }
        }
        let served = Arc::clone(&model);
        let adapter = GhBatched {
            model,
            stores,
            #[cfg(feature = "gpu")]
            graphs: std::collections::HashMap::new(),
            #[cfg(feature = "gpu")]
            graph_failed: false,
            pace: StepPace::default(),
        };
        let (submit_tx, worker) = launch(
            adapter,
            eos,
            num_blocks,
            block_size,
            max_running,
            max_prefill_tokens,
        )
        .map_err(|(_, e)| e)?;

        Ok(Self {
            submit_tx,
            worker,
            model: Some(served),
        })
    }

    /// Spawn the worker over a model whose layers keep each sequence's state, see
    /// [`SequenceBatched`]. The worker owns the model, and `stop` hands it back.
    ///
    /// `max_running` caps the concurrent decode batch width.
    /// The paged allocator counts the blocks of `context_length` tokens for each of
    /// `max_running` sequences.
    /// No store holds those blocks: each sequence's state lives beside them.
    /// `Err` hands the model back with the reason it is not served.
    pub fn spawn_sequences(
        model: GenericHeteroTransformer,
        eos: Vec<u32>,
        context_length: usize,
        max_running: usize,
        max_prefill_tokens: usize,
    ) -> std::result::Result<Self, (GenericHeteroTransformer, crate::tensor::Error)> {
        if let Err(e) = model.sequence_mixers() {
            return Err((model, e));
        }
        let adapter = SequenceBatched {
            model,
            seqs: HashMap::new(),
        };
        let block_size = CB_BLOCK_SIZE;
        let num_blocks = max_running.max(1) * context_length.max(1).div_ceil(block_size);
        match launch(
            adapter,
            eos,
            num_blocks,
            block_size,
            max_running,
            max_prefill_tokens,
        ) {
            Ok((submit_tx, worker)) => Ok(Self {
                submit_tx,
                worker,
                model: None,
            }),
            Err((adapter, e)) => Err((adapter.model, e)),
        }
    }

    /// Stops the worker once its sequences finish and hands the model back for serial
    /// serving. `Err` says why: another holder still shares the model, or the worker ended
    /// without it.
    pub fn stop(self) -> std::result::Result<GenericHeteroTransformer, String> {
        self.park().map(|(model, _)| model)
    }

    /// Stops the worker as `stop` does, and keeps its paged pool for the next worker.
    pub fn park(
        self,
    ) -> std::result::Result<(GenericHeteroTransformer, Option<ParkedPool>), String> {
        let Self {
            submit_tx,
            worker,
            model,
        } = self;
        drop(submit_tx);
        let SendWrap((owned, parked)) = worker
            .join()
            .map_err(|_| "cb-worker panicked".to_string())?;
        match (owned, model) {
            (Some(owned), _) => Ok((owned, parked)),
            (None, Some(shared)) => Arc::try_unwrap(shared)
                .map(|model| (model, parked))
                .map_err(|_| "the model is still shared".to_string()),
            (None, None) => Err("the worker ended without its model".to_string()),
        }
    }

    /// Submit a request; returns a receiver that yields its tokens as they are
    /// produced, ending with `CbToken::Done`. Dropping the receiver lets the
    /// worker reclaim the slot once the sequence finishes.
    pub fn submit(&self, prompt: Vec<u32>, max_new: usize) -> Receiver<CbToken> {
        self.submit_sampled(prompt, max_new, SamplingParams::greedy())
    }

    /// Submit with explicit sampling controls (temperature / top-k / top-p /
    /// repeat-penalty). `SamplingParams::greedy()` ⇒ bit-identical to `submit`.
    pub fn submit_sampled(
        &self,
        prompt: Vec<u32>,
        max_new: usize,
        sampling: SamplingParams,
    ) -> Receiver<CbToken> {
        let (tx, rx) = mpsc::channel();
        // If the worker is gone, the receiver just yields nothing.
        let _ = self.submit_tx.send(Submission {
            prompt,
            max_new,
            sampling,
            tx,
        });
        rx
    }
}

/// A spawned batched-decode server the engine streams a request through, whatever model backs it:
/// one `submit_cb` per request, tokens streamed back as `CbToken` until a `Done`. Both the generic
/// continuous-batch server and the DeepSeek batch worker implement it, so the serving path has one
/// route, not one per model.
pub trait BatchedServer: Send + Sync {
    fn submit_cb(
        &self,
        prompt: Vec<u32>,
        max_new: usize,
        sampling: SamplingParams,
    ) -> Receiver<CbToken>;
}

impl BatchedServer for ContinuousServer {
    fn submit_cb(
        &self,
        prompt: Vec<u32>,
        max_new: usize,
        sampling: SamplingParams,
    ) -> Receiver<CbToken> {
        self.submit_sampled(prompt, max_new, sampling)
    }
}

/// Serves the submissions until every submitter is gone and no sequence runs, then hands the
/// adapter back. `None` when the engine did not start.
fn worker_loop<M: BatchedModel>(
    model: M,
    eos: Vec<u32>,
    num_blocks: usize,
    block_size: usize,
    max_running: usize,
    max_prefill_tokens: usize,
    submit_rx: Receiver<Submission>,
) -> Option<M> {
    let mut engine = match ContinuousBatchEngine::new(
        model,
        eos,
        num_blocks,
        block_size,
        max_running,
        max_prefill_tokens,
    ) {
        Ok(e) => e,
        Err(_) => return None,
    };
    // Tokens reach their clients from a thread of their own, in the order they were produced:
    // waking a client from the step loop lets the woken threads take the core before the next
    // step is launched.
    let (emit_tx, emit_rx) = mpsc::channel::<Vec<(Sender<CbToken>, CbToken)>>();
    let emitter = std::thread::Builder::new()
        .name("cb-emit".into())
        .spawn(move || {
            for batch in emit_rx {
                for (tx, token) in batch {
                    let _ = tx.send(token);
                }
            }
        })
        .is_ok();
    let emit = |batch: Vec<(Sender<CbToken>, CbToken)>| {
        if batch.is_empty() {
            return;
        }
        if emitter {
            let _ = emit_tx.send(batch);
        } else {
            for (tx, token) in batch {
                let _ = tx.send(token);
            }
        }
    };
    let mut next_id: u64 = 0;
    // seq id -> (token sender, prompt length already streamed marker)
    let mut chans: HashMap<u64, Sender<CbToken>> = HashMap::new();
    let mut fails: u32 = 0; // consecutive step-error count (worker recovery backstop)

    loop {
        // Drain any pending submissions (non-blocking) before stepping.
        loop {
            match submit_rx.try_recv() {
                Ok(sub) => {
                    let id = next_id;
                    next_id += 1;
                    engine.submit(GenReq {
                        id,
                        prompt: sub.prompt,
                        max_new: sub.max_new,
                        sampling: sub.sampling,
                    });
                    chans.insert(id, sub.tx);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if engine.is_idle() {
                        return Some(engine.into_model());
                    }
                    break;
                }
            }
        }

        if engine.is_idle() {
            // Nothing to do - block for the next submission (or exit if all
            // handles dropped). This keeps the worker off-CPU while idle.
            match submit_rx.recv() {
                Ok(sub) => {
                    let id = next_id;
                    next_id += 1;
                    engine.submit(GenReq {
                        id,
                        prompt: sub.prompt,
                        max_new: sub.max_new,
                        sampling: sub.sampling,
                    });
                    chans.insert(id, sub.tx);
                }
                Err(_) => return Some(engine.into_model()), // all submitters gone, no work left
            }
            continue;
        }

        let (emits, done) = match engine.step_stream() {
            Ok(v) => {
                fails = 0;
                v
            }
            Err(e) => {
                // A step failed (e.g. a transient CUDA OOM on a too-wide batch).
                // RECOVER instead of dying: abort the in-flight sequences (free
                // their KV, notify their clients with Done so nobody hangs), then
                // keep serving. Bail only after several CONSECUTIVE failures (a
                // genuinely poisoned context) so we don't spin failing forever.
                fails += 1;
                eprintln!(
                    "cb-worker step error #{fails} (recovering, aborting {} in-flight): {e}",
                    chans.len()
                );
                let mut batch = Vec::new();
                for id in engine.abort_all() {
                    if let Some(tx) = chans.remove(&id) {
                        batch.push((tx, CbToken::Done(FinishReason::Stop)));
                    }
                }
                if fails >= 3 {
                    eprintln!("cb-worker: 3 consecutive step errors - stopping worker");
                    for (_, tx) in chans.drain() {
                        batch.push((tx, CbToken::Done(FinishReason::Stop)));
                    }
                    emit(batch);
                    return Some(engine.into_model());
                }
                emit(batch);
                continue;
            }
        };
        let mut batch = Vec::new();
        for (id, at, cached_prompt_tokens) in engine.take_admitted() {
            if let Some(tx) = chans.get(&id) {
                batch.push((
                    tx.clone(),
                    CbToken::Admitted {
                        at,
                        cached_prompt_tokens,
                    },
                ));
            }
        }
        for (id, tok) in emits {
            if let Some(tx) = chans.get(&id) {
                // The token a sequence ended on is no content; Done carries it.
                let ended = done
                    .iter()
                    .any(|d| d.id == id && d.reason == FinishReason::Eos);
                if !ended {
                    batch.push((tx.clone(), CbToken::Tok(tok)));
                }
            }
        }
        for d in done {
            if let Some(tx) = chans.remove(&d.id) {
                batch.push((tx, CbToken::Done(d.reason)));
            }
        }
        emit(batch);
    }
}
