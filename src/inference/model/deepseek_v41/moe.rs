//! DeepSeek V4.1 mixture of experts (bet phase 3).
//!
//! Top-k routed experts under a sqrt-softplus gate, plus one shared expert every token also passes
//! through. The gate's correction bias steers selection only: the routing weights come from the
//! unbiased scores. Each expert is a SwiGLU FFN with the training clamps that keep fp8/fp4
//! activations in range (the up branch clamped both sides, the gate branch only from above).
//!
//! The reference is `notes/deepseek-oracle`. This forward reproduces its `MoE.forward` (world_size
//! 1), and is judged against a dump of one layer's MoE in the module test.

use crate::inference::offload::experts::{Expert, ExpertOffload};
use crate::inference::offload::store::ExpertSet;
use crate::tensor::{Device, Result, Tensor};
use std::sync::Arc;

/// What a calibration run is told about one MoE block's forward: `expert` is `None` for the block's
/// whole input, `Some(e)` for the rows routed to expert `e`. `rows` are the inputs, row-major;
/// `weights` their routing weights; `down` the rows entering the expert's down projection, scaled
/// by those weights as the reference applies them - the moment a quantiser fits against is the one
/// the weighted rows make.
pub type ExpertObserver = dyn Fn(Option<usize>, &[f32], &[f32], &[f32]) + Send + Sync;

/// A full MoE block: gate, routed experts, and the shared expert.
pub struct Moe {
    pub gate_weight: Tensor, // [n_routed, dim]
    pub gate_bias: Tensor,   // [n_routed]
    pub experts: ExpertSet,
    pub shared: Expert,

    pub n_routed: usize,
    pub n_activated: usize,
    pub dim: usize,
    pub gate_temp: f32,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub norm_topk: bool,
    /// The last forward's routing, per token: the chosen experts with their weights, heaviest
    /// first. Observability for the cost of each expert read against what it contributes.
    pub last_routing: std::sync::Mutex<Vec<Vec<(usize, f32)>>>,
    /// Told what every forward routes where, when a calibration run asks; `None` otherwise, and
    /// then the forward does no extra work.
    pub observer: std::sync::RwLock<Option<std::sync::Arc<ExpertObserver>>>,
    /// Where routed experts run when not on the CPU; `None` keeps them here.
    pub offload: std::sync::RwLock<Option<std::sync::Arc<ExpertOffload>>>,
}

impl Moe {
    /// The router's gate and bias, in bytes: read whole for every token, as the attention is.
    pub fn gate_bytes(&self) -> usize {
        (self.gate_weight.elem_count() + self.gate_bias.elem_count()) * std::mem::size_of::<f32>()
    }

    /// Numerically stable softplus, matching torch's threshold at 20.
    fn softplus(z: f32) -> f32 {
        if z > 20.0 {
            z
        } else {
            z.exp().ln_1p()
        }
    }

    /// Resize the routed experts' hot cache when they are streamed.
    pub fn set_expert_cache(&self, slots: usize) {
        self.experts.set_capacity(slots);
    }

    /// `x` is [b, s, dim]; returns [b, s, dim].
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let n: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((n, self.dim))?;

        use crate::inference::offload::stage;
        // Gate: sqrt-softplus scores, top-k by score+bias, weights from the unbiased scores.
        let gate_started = std::time::Instant::now();
        let raw = stage("moe gate product", || {
            crate::inference::offload::linear(&x2, &self.gate_weight)?.to_vec2::<f32>()
        })?;
        let bias = self.gate_bias.flatten_all()?.to_vec1::<f32>()?;
        let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n_routed];
        let mut routing: Vec<Vec<(usize, f32)>> = Vec::with_capacity(n);
        for (t, row) in raw.iter().enumerate() {
            let scores: Vec<f32> = row
                .iter()
                .map(|&z| Self::softplus(z / self.gate_temp).sqrt())
                .collect();
            // Top-k experts by biased score; the weights are the unbiased scores at those experts.
            let mut order: Vec<usize> = (0..self.n_routed).collect();
            order.sort_by(|&a, &b| {
                (scores[b] + bias[b])
                    .partial_cmp(&(scores[a] + bias[a]))
                    .unwrap()
            });
            let chosen = &order[..self.n_activated];
            let mut w: Vec<f32> = chosen.iter().map(|&e| scores[e]).collect();
            if self.norm_topk && self.n_activated > 1 {
                let s: f32 = w.iter().sum::<f32>() + 1e-20;
                for wi in w.iter_mut() {
                    *wi /= s;
                }
            }
            for (slot, &e) in chosen.iter().enumerate() {
                assign[e].push((t, w[slot] * self.route_scale));
            }
            routing.push(
                chosen
                    .iter()
                    .zip(&w)
                    .map(|(&e, &wi)| (e, wi * self.route_scale))
                    .collect(),
            );
        }
        *self.last_routing.lock().unwrap() = routing;
        if let Some(off) = crate::inference::offload::current() {
            off.record("moe routing", gate_started.elapsed().as_nanos() as u64);
        }

        // The routed experts this batch needs are known from the top-k before any of them runs, so
        // a streamed set prefetches that working set before the compute loop reads it.
        let uses: Vec<(usize, u64, u64)> = assign
            .iter()
            .enumerate()
            .filter(|(_, tokens)| !tokens.is_empty())
            .map(|(e, tokens)| {
                let last = tokens.iter().map(|&(t, _)| t as u64).max().unwrap_or(0);
                (e, tokens.len() as u64, last)
            })
            .collect();
        self.experts.prefetch(&uses, n as u64)?;
        let active: Vec<usize> = uses.iter().map(|&(e, _, _)| e).collect();
        // Every routed expert's bytes are asked for at once, so the reads of one overlap the
        // compute of another instead of each faulting in alone. In turn rather than from
        // several threads: issuing the same requests across a thread pool was measured on a
        // cold prefill and changed nothing, so the loop stays the simpler of the two.
        for &e in &active {
            self.experts.fetch(e)?.will_need();
        }

        // Every token passes through the shared expert; routed experts add on top, each run over
        // the rows of the tokens that selected it and nothing else.
        let mut y = stage("moe shared expert", || {
            self.shared
                .forward(&x2, self.swiglu_limit)?
                .to_vec2::<f32>()
        })?;
        let xv = x2.to_vec2::<f32>()?;
        let observer = self.observer.read().unwrap().clone();
        if let Some(obs) = &observer {
            let all: Vec<f32> = xv.iter().flatten().copied().collect();
            obs(None, &all, &[], &[]);
        }
        let offload = self.offload.read().unwrap().clone();
        let rows_of = |e: usize| -> Vec<f32> {
            assign[e]
                .iter()
                .flat_map(|&(t, _)| xv[t].iter().copied())
                .collect()
        };
        // Each active expert's (rows entering w2, output), computed across the offload's lanes when
        // there is one; observed and added in expert order either way, so a record does not depend
        // on which lane finished first.
        let mut done: Vec<Option<Result<(Vec<f32>, Vec<f32>)>>> =
            (0..active.len()).map(|_| None).collect();
        let routed_started = std::time::Instant::now();
        if let Some(off) = &offload {
            let fetch = |e: usize| self.experts.fetch(e);
            let (ran, fetch_ns, run_ns) =
                off.run_all(&active, n, self.swiglu_limit, &fetch, &rows_of);
            done = ran;
            if let Some(o) = crate::inference::offload::current() {
                // The lanes' own time, split: their threads carry no recorder, so it is
                // recorded here after the join.
                for (name, a) in [
                    ("moe expert hold", &off.timings[0]),
                    ("moe expert kernels", &off.timings[1]),
                    ("count experts on card", &off.timings[2]),
                ] {
                    o.record(name, a.swap(0, std::sync::atomic::Ordering::Relaxed));
                }
                o.record("moe expert fetch", fetch_ns);
                o.record("moe expert run", run_ns);
            }
        }
        if let Some(off) = crate::inference::offload::current() {
            off.record(
                "moe routed experts",
                routed_started.elapsed().as_nanos() as u64,
            );
        }
        let gather_started = std::time::Instant::now();
        // The experts an offload did not take are computed across the cores, one task each: a
        // single expert's product does not saturate the memory its weights stream through, and
        // the experts of a layer depend on nothing but their own rows. Held here rather than
        // released inside the task, so the weights go back in expert order below.
        type Ready = (Vec<f32>, Vec<f32>, Option<Arc<Expert>>);
        let mut ready: Vec<Option<Ready>> = done
            .iter_mut()
            .map(|d| {
                d.take()
                    .map(|r| r.map(|(h, out)| (h, out, None)))
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        let host_started = std::time::Instant::now();
        let host_count: u64;
        {
            use rayon::prelude::*;
            let todo: Vec<usize> = (0..active.len()).filter(|&i| ready[i].is_none()).collect();
            host_count = todo.len() as u64;
            let made: Vec<Result<Ready>> = todo
                .par_iter()
                .map(|&i| -> Result<Ready> {
                    let e = active[i];
                    let expert = self.experts.fetch(e)?;
                    let xe =
                        Tensor::from_vec(rows_of(e), (assign[e].len(), self.dim), &Device::Cpu)?;
                    let h = std::cell::RefCell::new(Vec::new());
                    let seen = |rows: &[f32]| *h.borrow_mut() = rows.to_vec();
                    let out = expert
                        .forward_seen(
                            &xe,
                            self.swiglu_limit,
                            observer.as_ref().map(|_| &seen as &dyn Fn(&[f32])),
                        )?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    Ok((h.into_inner(), out, Some(expert)))
                })
                .collect();
            for (i, r) in todo.into_iter().zip(made) {
                ready[i] = Some(r?);
            }
        }
        if let Some(o) = crate::inference::offload::current() {
            o.record("moe host experts", host_started.elapsed().as_nanos() as u64);
            o.record("count experts on host", host_count);
        }
        let add_started = std::time::Instant::now();
        for (i, &e) in active.iter().enumerate() {
            let (h, out, held) = ready[i]
                .take()
                .ok_or_else(|| crate::tensor::Error::msg("an active expert produced nothing"))?;
            if let Some(obs) = &observer {
                let weights: Vec<f32> = assign[e].iter().map(|&(_, w)| w).collect();
                let inter = h.len() / weights.len().max(1);
                let weighted: Vec<f32> = h
                    .chunks(inter.max(1))
                    .zip(&weights)
                    .flat_map(|(row, &w)| row.iter().map(move |v| v * w))
                    .collect();
                obs(Some(e), &rows_of(e), &weights, &weighted);
            }
            for (r, &(t, w)) in assign[e].iter().enumerate() {
                for j in 0..self.dim {
                    y[t][j] += w * out[r * self.dim + j];
                }
            }
            if let Some(expert) = held {
                self.experts.release(e, &expert);
            }
        }
        if let Some(o) = crate::inference::offload::current() {
            o.record("moe add", add_started.elapsed().as_nanos() as u64);
        }
        // A batch of many tokens read the layer nearly whole; what read-ahead brought in
        // beside the kept experts goes now, before the next layer's reads need the room.
        if n > 1 {
            self.experts.sweep()?;
        }

        if let Some(off) = crate::inference::offload::current() {
            off.record(
                "moe gather and add",
                gather_started.elapsed().as_nanos() as u64,
            );
        }
        let flat: Vec<f32> = y.into_iter().flatten().collect();
        Tensor::from_vec(flat, (n, self.dim), &Device::Cpu)?.reshape(dims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::offload::projection::Projection;
    use crate::inference::offload::store::{ExpertStore, VecExpertLoader};
    use std::sync::Arc;

    // A toy configuration, weights and input drawn from a fixed stream: the tests compare the MoE
    // with itself under different storage, caches and offloads.
    const N_ROUTED: usize = 8;
    const N_ACTIVATED: usize = 2;
    const DIM: usize = 256;
    const INTER: usize = 256;
    const TOKENS: usize = 64;
    const GATE_TEMP: f32 = 1.0;
    const ROUTE_SCALE: f32 = 1.5;
    const SWIGLU_LIMIT: f32 = 10.0;

    fn stream(seed: u64, n: usize, scale: f32) -> Vec<f32> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                ((x >> 11) as f32 / (1u64 << 53) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn drawn(seed: u64, dims: &[usize], scale: f32) -> Tensor {
        Tensor::from_vec(
            stream(seed, dims.iter().product(), scale),
            dims,
            &Device::Cpu,
        )
        .unwrap()
    }

    /// Projection `w` (w1, w2 or w3) of routed expert `i`.
    fn expert_weight(i: usize, w: &str) -> Tensor {
        let seed = 100 + 3 * i as u64;
        match w {
            "w1" => drawn(seed, &[INTER, DIM], 0.05),
            "w2" => drawn(seed + 1, &[DIM, INTER], 0.05),
            _ => drawn(seed + 2, &[INTER, DIM], 0.05),
        }
    }

    fn expert(i: usize) -> Expert {
        Expert::dense(
            expert_weight(i, "w1"),
            expert_weight(i, "w2"),
            expert_weight(i, "w3"),
        )
    }

    fn experts_arc() -> Vec<Arc<Expert>> {
        (0..N_ROUTED).map(|i| Arc::new(expert(i))).collect()
    }

    fn input() -> Tensor {
        drawn(7, &[1, TOKENS, DIM], 1.0)
    }

    fn build_moe(experts: ExpertSet) -> Moe {
        Moe {
            gate_weight: drawn(1, &[N_ROUTED, DIM], 0.2),
            gate_bias: drawn(2, &[N_ROUTED], 0.1),
            experts,
            shared: expert(N_ROUTED),
            n_routed: N_ROUTED,
            n_activated: N_ACTIVATED,
            dim: DIM,
            gate_temp: GATE_TEMP,
            route_scale: ROUTE_SCALE,
            swiglu_limit: SWIGLU_LIMIT,
            norm_topk: true,
            last_routing: std::sync::Mutex::new(Vec::new()),
            observer: std::sync::RwLock::new(None),
            offload: std::sync::RwLock::new(None),
        }
    }

    /// An observer changes nothing about the output, and is told exactly what was routed: the
    /// block's input once, then, for every token and expert the routing chose, that token's input
    /// row, its routing weight, and a down-projection input scaled by that weight.
    #[test]
    fn an_observer_sees_the_routing_and_changes_nothing() {
        use std::sync::Mutex;
        let moe = build_moe(ExpertSet::Resident(experts_arc()));
        let x = input();
        let plain = moe
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let routing = moe.last_routing.lock().unwrap().clone();

        type Call = (Option<usize>, Vec<f32>, Vec<f32>, Vec<f32>);
        let calls: Arc<Mutex<Vec<Call>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = calls.clone();
        *moe.observer.write().unwrap() = Some(Arc::new(
            move |e: Option<usize>, xs: &[f32], w: &[f32], h: &[f32]| {
                sink.lock()
                    .unwrap()
                    .push((e, xs.to_vec(), w.to_vec(), h.to_vec()));
            },
        ));
        let seen = moe
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(plain, seen);

        let calls = calls.lock().unwrap();
        let input = x.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (whole, per_expert): (Vec<&Call>, Vec<&Call>) =
            calls.iter().partition(|c| c.0.is_none());
        assert_eq!(whole.len(), 1);
        assert_eq!(whole[0].1, input);
        let routed: usize = routing.iter().map(|r| r.len()).sum();
        assert_eq!(per_expert.iter().map(|c| c.2.len()).sum::<usize>(), routed);
        for (e, xs, weights, down) in per_expert {
            let e = e.unwrap();
            let tokens: Vec<(usize, f32)> = routing
                .iter()
                .enumerate()
                .filter_map(|(t, r)| r.iter().find(|&&(ex, _)| ex == e).map(|&(_, w)| (t, w)))
                .collect();
            assert_eq!(weights, &tokens.iter().map(|&(_, w)| w).collect::<Vec<_>>());
            for (row, &(t, _)) in xs.chunks(DIM).zip(&tokens) {
                assert_eq!(row, &input[t * DIM..(t + 1) * DIM]);
            }
            assert_eq!(down.len() % weights.len(), 0);
        }
    }

    /// Inside `with_offload` a projection's product goes to the offload, which may decline it;
    /// outside, it runs here again.
    #[test]
    fn a_projection_offload_takes_the_product_within_its_scope() {
        use crate::inference::offload::{with_offload, Offload};
        struct Sum(bool);
        impl Offload for Sum {
            fn projection(&self, p: &Projection, xs: &[f32]) -> Option<Result<Vec<f32>>> {
                self.0
                    .then(|| Ok(vec![xs.iter().sum::<f32>(); p.dims()[0]]))
            }
            fn sparse_attention(
                &self,
                _: &[f32],
                _: &[f32],
                _: &[f32],
                _: &[i32],
                _: (usize, usize, usize, usize),
                _: f32,
            ) -> Option<Result<Vec<f32>>> {
                None
            }
        }
        let w = Tensor::from_vec(
            (0..12).map(|i| i as f32 * 0.1).collect(),
            (3, 4),
            &Device::Cpu,
        )
        .unwrap();
        let p = Projection::Dense(w);
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 4), &Device::Cpu).unwrap();
        let here = p
            .apply(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let there = with_offload(Arc::new(Sum(true)), || p.apply(&x)).unwrap();
        assert_eq!(there.dims(), &[1, 3]);
        assert_eq!(
            there.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![10.0; 3]
        );
        let again = with_offload(Arc::new(Sum(false)), || p.apply(&x))
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(again, here);
        assert_eq!(
            p.apply(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            here
        );
    }

    /// Routed experts handed to an offload over several lanes give the output and the observations
    /// the CPU gives, in the same order; an expert the offload declines runs on the CPU.
    #[test]
    fn an_offload_changes_nothing_but_where_experts_run() {
        use std::sync::Mutex;
        let moe = build_moe(ExpertSet::Resident(experts_arc()));
        let x = input();
        type Call = (Option<usize>, Vec<f32>, Vec<f32>, Vec<f32>);
        let record = |moe: &Moe| -> (Vec<f32>, Vec<Call>) {
            let calls: Arc<Mutex<Vec<Call>>> = Arc::new(Mutex::new(Vec::new()));
            let sink = calls.clone();
            *moe.observer.write().unwrap() = Some(Arc::new(
                move |e: Option<usize>, xs: &[f32], w: &[f32], h: &[f32]| {
                    sink.lock()
                        .unwrap()
                        .push((e, xs.to_vec(), w.to_vec(), h.to_vec()));
                },
            ));
            let out = moe
                .forward(&x)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let calls = calls.lock().unwrap().clone();
            (out, calls)
        };
        let (cpu_out, cpu_calls) = record(&moe);
        let ran = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counter = ran.clone();
        *moe.offload.write().unwrap() = Some(Arc::new(ExpertOffload {
            lanes: 3,
            timings: Arc::new(Default::default()),
            warm: None,
            run: Box::new(move |_lane, _tokens, expert, rows, limit| {
                let n = rows.len() / DIM;
                // Declines every other call, to leave some experts to the CPU.
                if counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2 == 1 {
                    return None;
                }
                let xe = Tensor::from_vec(rows.to_vec(), (n, DIM), &Device::Cpu).unwrap();
                let h = std::cell::RefCell::new(Vec::new());
                let seen = |r: &[f32]| *h.borrow_mut() = r.to_vec();
                let out = expert
                    .forward_seen(&xe, limit, Some(&seen))
                    .and_then(|t| t.flatten_all()?.to_vec1::<f32>());
                Some(out.map(|o| (h.into_inner(), o)))
            }),
        }));
        let (off_out, off_calls) = record(&moe);
        assert!(ran.load(std::sync::atomic::Ordering::Relaxed) > 0);
        assert_eq!(cpu_out, off_out);
        assert_eq!(cpu_calls, off_calls);
    }

    /// Unit 4 gate: an MoE whose experts are streamed through a hot cache too small to hold them all
    /// produces exactly the resident MoE's output. Capacity below the number of experts the batch
    /// routes to forces eviction, so this proves the least-frequently-used cache and the router
    /// prefetch never change the numbers - only what is resident. The memory win itself comes when
    /// the loader reads a quantised, disk-backed source instead of cloning resident tensors.
    #[test]
    fn streamed_experts_match_resident() {
        let x = input();

        let resident = build_moe(ExpertSet::Resident(experts_arc()));
        let want = resident
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let capacity = 1; // one slot, so every further routed expert forces an eviction
        let store = ExpertStore::new(Box::new(VecExpertLoader::new(experts_arc())), capacity);
        let streamed = build_moe(ExpertSet::Streamed(store));
        let got = streamed
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        assert_eq!(got, want, "streamed output differs from resident");

        let stats = match &streamed.experts {
            ExpertSet::Streamed(s) => s.stats(),
            _ => unreachable!(),
        };
        // The gate is only meaningful if the tier was too small for the batch: one slot keeps
        // the busiest expert and every other one is streamed past it.
        assert!(
            stats.misses > stats.hits,
            "the batch was served from the tier: {stats:?}, cache was never pressured"
        );
    }

    /// A batch keeps the experts its last tokens routed to: recency within the batch decides,
    /// not the order the experts were asked for in.
    #[test]
    fn a_batch_keeps_the_experts_its_last_tokens_route_to() {
        let store = ExpertStore::new(Box::new(VecExpertLoader::new(experts_arc())), 2);
        store
            .prefetch(&[(0, 1, 0), (1, 1, 599), (2, 40, 300), (3, 30, 598)], 600)
            .unwrap();
        assert!(
            !store.is_kept(0) && !store.is_kept(2),
            "the experts of early tokens stayed"
        );
        assert!(
            store.is_kept(1) && store.is_kept(3),
            "the experts of the last tokens were not kept"
        );
    }

    /// The quantised path: the routed experts stacked, block-quantised and viewed in place, judged
    /// against the resident forward over the same stacks dequantised. The only difference left is
    /// the dot engine's activation quantisation, so the outputs agree closely; a wrong view range,
    /// a transposed projection or a mis-sized block would miss by order one.
    #[test]
    fn quantised_experts_match_the_dequantised_ones() {
        use crate::inference::offload::store::QuantExpertLoader;
        use crate::tensor::quantized::{GgmlDType, QTensor};
        let x = input();
        let stack = |w: &str| -> Tensor {
            let parts: Vec<Tensor> = (0..N_ROUTED).map(|i| expert_weight(i, w)).collect();
            Tensor::stack(&parts, 0).unwrap()
        };
        let (gate, up, down) = (stack("w1"), stack("w3"), stack("w2"));
        let q = |t: &Tensor| Arc::new(QTensor::quantize(t, GgmlDType::Q8_0).unwrap());
        let (qg, qu, qd) = (q(&gate), q(&up), q(&down));

        let slice = |t: &Tensor, i: usize| {
            t.narrow(0, i, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let dq = |t: &Arc<QTensor>| t.dequantize(&Device::Cpu).unwrap();
        let (dg, du, dd) = (dq(&qg), dq(&qu), dq(&qd));
        let resident: Vec<Arc<Expert>> = (0..N_ROUTED)
            .map(|i| Arc::new(Expert::dense(slice(&dg, i), slice(&dd, i), slice(&du, i))))
            .collect();
        let want = build_moe(ExpertSet::Resident(resident))
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        let loader = QuantExpertLoader::new(qg, qu, qd, N_ROUTED);
        let got = build_moe(ExpertSet::Streamed(ExpertStore::new(Box::new(loader), 0)))
            .forward(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let num: f32 = got.iter().zip(&want).map(|(a, b)| (a - b) * (a - b)).sum();
        let den: f32 = want.iter().map(|b| b * b).sum();
        let dot: f32 = got.iter().zip(&want).map(|(a, b)| a * b).sum();
        let cos = dot / (got.iter().map(|a| a * a).sum::<f32>().sqrt() * den.sqrt());
        let rel = (num / den).sqrt();
        assert!(
            cos > 0.999 && rel < 2e-2,
            "in-place experts vs dequantised: cos {cos} rel {rel}"
        );
    }
}
