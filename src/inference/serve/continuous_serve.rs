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
//! is what keeps the lock-free sharing sound). Supported for the dense archs
//! `batched_paged_decode` covers (Qwen2/Qwen3 today); callers fall back to the
//! serial path otherwise.
use crate::inference::cache::paged_kv::PagedKvStore;
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

/// Handle to the running continuous-batch worker. Cheap to clone-share via Arc.
pub struct ContinuousServer {
    submit_tx: Sender<Submission>,
    /// The worker thread, joined by `stop`.
    worker: std::thread::JoinHandle<()>,
    /// The served model, shared with the worker; `stop` takes it back once the worker ends.
    model: Arc<GenericHeteroTransformer>,
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
    #[cfg(feature = "cuda")]
    graphs: std::collections::HashMap<(usize, usize), CapturedDecode>,
    // Set once if capture ever fails -> permanently use the eager path.
    #[cfg(feature = "cuda")]
    graph_failed: bool,
}

#[cfg(feature = "cuda")]
impl Drop for GhBatched {
    fn drop(&mut self) {
        // The captured graphs go before the model and the paged KV stores they read.
        self.graphs.clear();
    }
}

/// A captured batched-decode graph + its stable I/O buffers (one per (B,cap)).
#[cfg(feature = "cuda")]
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
#[cfg(feature = "cuda")]
const CB_GRAPH_BUCKET: usize = 256;

/// Tokens per block of the paged cache.
pub const CB_BLOCK_SIZE: usize = 16;

fn argmax_row(logits: &Tensor, row: usize) -> crate::tensor::Result<u32> {
    logits
        .narrow(0, row, 1)?
        .flatten_all()?
        .to_dtype(DType::F32)?
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
#[cfg(feature = "cuda")]
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
        if !dec.is_empty() {
            let toks: Vec<u32> = dec.iter().map(|&i| items[i].tokens[0]).collect();
            let pos: Vec<usize> = dec.iter().map(|&i| items[i].context_len - 1).collect();
            let slots: Vec<usize> = dec.iter().map(|&i| items[i].slots[0]).collect();
            let bts: Vec<Vec<u32>> = dec.iter().map(|&i| items[i].block_table.clone()).collect();
            let ctx: Vec<usize> = dec.iter().map(|&i| items[i].context_len).collect();
            // CUDA-graph decode is what delivers the throughput win (eager batching
            // alone roughly ties serial on big models); always on, with a safe eager
            // fallback latched via `graph_failed` on any capture/replay failure.
            #[cfg(feature = "cuda")]
            let graph_on = !self.graph_failed
                && !self.model.compute_device().is_cpu()
                && self.model.cb_eligible();
            #[cfg(not(feature = "cuda"))]
            let graph_on = false;
            let logits = if graph_on {
                #[cfg(feature = "cuda")]
                {
                    // Try the captured-graph path; on ANY capture/replay failure
                    // fall back to the eager path (correct, just unaccelerated) so
                    // the flag can never break serving.
                    match self.decode_graph(&toks, &pos, &slots, &bts, &ctx) {
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
                #[cfg(not(feature = "cuda"))]
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
            #[cfg(feature = "cuda")]
            let fast_argmax = dec.iter().all(|&i| items[i].sampling.temperature <= 0.0)
                && !self.model.compute_device().is_cpu();
            #[cfg(not(feature = "cuda"))]
            let fast_argmax = false;
            if fast_argmax {
                #[cfg(feature = "cuda")]
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
                #[cfg(feature = "cuda")]
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
                #[cfg(not(feature = "cuda"))]
                let gpu_ok = false;
                if gpu_ok {
                    #[cfg(feature = "cuda")]
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
        // -- prefills: each prompt in ONE forward (the TTFT lever) --
        for (i, it) in items.iter().enumerate() {
            if it.is_prefill {
                let l = if it.cached_len > 0 {
                    // Prefix-cached: compute only the suffix, gather the prefix KV.
                    self.model.paged_prefill_seq_cached(
                        &it.tokens,
                        &it.slots,
                        it.cached_len,
                        &it.block_table,
                        &mut self.stores,
                    )?
                } else {
                    self.model
                        .paged_prefill_seq(&it.tokens, &it.slots, &mut self.stores)?
                };
                out[i] = sample_row(&l, 0, &it.sampling, &it.recent_tokens)?;
            }
        }
        Ok(out)
    }
}

#[cfg(feature = "cuda")]
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
            return model.lm_head(&g.hidden);
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
struct SendWrap(GhBatched);
unsafe impl Send for SendWrap {}

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
        // Pin a cuBLAS workspace so the batched-decode GEMMs stop allocating a
        // workspace per call (per-call allocs are pure overhead in the hot loop,
        // and are also required to be absent before any future graph capture).
        #[cfg(feature = "cuda")]
        if !cdev.is_cpu() {
            let _ = crate::tensor::cuda_ext::pin_cublas_workspace(&cdev, 256 << 20);
            // Disable event tracking: the CB worker is single-stream, so the
            // multi-stream sync events aren't needed - and creating CUDA events
            // during allocations INSIDE a graph capture invalidates the capture
            // (required for the captured decode graph; mirrors tp_model + test_graph_capture).
            if let Ok(cd) = cdev.as_cuda_device() {
                unsafe {
                    cd.disable_event_tracking();
                }
            }
        }
        let store_dt = if cdev.is_cpu() {
            DType::F32
        } else {
            DType::F16
        };
        let stores: Vec<PagedKvStore> = (0..n_layers)
            .map(|_| PagedKvStore::new(num_blocks, block_size, n_kv, hd, store_dt, &cdev))
            .collect::<crate::tensor::Result<_>>()?;
        let served = Arc::clone(&model);
        let adapter = SendWrap(GhBatched {
            model,
            stores,
            #[cfg(feature = "cuda")]
            graphs: std::collections::HashMap::new(),
            #[cfg(feature = "cuda")]
            graph_failed: false,
        });
        let (submit_tx, submit_rx) = mpsc::channel::<Submission>();

        let worker = std::thread::Builder::new()
            .name("cb-worker".into())
            .spawn(move || {
                let gh = adapter; // SendWrap (Send); unwrap inside the worker thread
                worker_loop(
                    gh.0,
                    eos,
                    num_blocks,
                    block_size,
                    max_running,
                    max_prefill_tokens,
                    submit_rx,
                )
            })
            .map_err(|e| crate::tensor::Error::msg(format!("cb-worker spawn: {e}")))?;

        Ok(Self {
            submit_tx,
            worker,
            model: served,
        })
    }

    /// Stops the worker once its sequences finish and hands the model back for serial
    /// serving. `Err` returns the server when another holder still shares the model.
    pub fn stop(self) -> std::result::Result<GenericHeteroTransformer, String> {
        let Self {
            submit_tx,
            worker,
            model,
        } = self;
        drop(submit_tx);
        worker
            .join()
            .map_err(|_| "cb-worker panicked".to_string())?;
        Arc::try_unwrap(model).map_err(|_| "the model is still shared".to_string())
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

fn worker_loop(
    model: GhBatched,
    eos: Vec<u32>,
    num_blocks: usize,
    block_size: usize,
    max_running: usize,
    max_prefill_tokens: usize,
    submit_rx: Receiver<Submission>,
) {
    let mut engine = match ContinuousBatchEngine::new(
        model,
        eos.clone(),
        num_blocks,
        block_size,
        max_running,
        max_prefill_tokens,
    ) {
        Ok(e) => e,
        Err(_) => return,
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
                        return;
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
                Err(_) => return, // all submitters gone, no work left
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
                for id in engine.abort_all() {
                    if let Some(tx) = chans.remove(&id) {
                        let _ = tx.send(CbToken::Done(FinishReason::Stop));
                    }
                }
                if fails >= 3 {
                    eprintln!("cb-worker: 3 consecutive step errors - stopping worker");
                    for (_, tx) in chans.drain() {
                        let _ = tx.send(CbToken::Done(FinishReason::Stop));
                    }
                    return;
                }
                continue;
            }
        };
        for (id, tok) in emits {
            if let Some(tx) = chans.get(&id) {
                // Suppress the terminal EOS as a content token; Done carries it.
                if !eos.contains(&tok) {
                    let _ = tx.send(CbToken::Tok(tok));
                }
            }
        }
        for d in done {
            if let Some(tx) = chans.remove(&d.id) {
                let _ = tx.send(CbToken::Done(d.reason));
            }
        }
    }
}
