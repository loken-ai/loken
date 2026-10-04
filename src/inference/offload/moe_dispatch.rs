//! The routed experts of one MoE decode or verify block, split across the cards and the cores.
//! Model-neutral: it speaks experts, their assigned rows and the offload's lanes, hands the misses
//! to the CPU kernels to run beside the cards' block, and returns each expert's product for the
//! caller to gather. The gate, the shared expert and the weighted gather stay with the model.

use super::experts::{Expert, ExpertOffload};
use crate::tensor::{Device, Result, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

/// Per active expert: its product from the cards (`done`), or the host-side product computed
/// alongside the cards on the overlapped block path (`cpu_side`, carrying the held handle to
/// release). The caller reads whichever is present and runs any expert neither took.
#[allow(clippy::type_complexity)]
pub(crate) type Routed = (
    Vec<Option<Result<(Vec<f32>, Vec<f32>)>>>,
    Vec<Option<Result<(Vec<f32>, Vec<f32>, Option<Arc<Expert>>)>>>,
    Option<DeferredExperts>,
);

/// Routed experts the host runs on a thread of its own while the layers below go on: the weighted
/// sum of their outputs, read once the stream is ready to take it. Dropping one waits for the
/// thread, so no read or kernel outlives the decode that started it.
pub struct DeferredExperts {
    job: Option<std::thread::JoinHandle<Result<Vec<f32>>>>,
    /// Positions in the block's active list that this job took.
    pub indexes: Vec<usize>,
    /// The experts of the job by number, handed back for the store to release once they are done.
    pub held: Vec<(usize, Arc<Expert>)>,
}

impl DeferredExperts {
    /// Start `experts` over the row `x`, each output scaled by its weight and summed.
    fn start(
        indexes: Vec<usize>,
        held: Vec<(usize, Arc<Expert>)>,
        weights: Vec<f32>,
        x: Vec<f32>,
        swiglu_limit: f32,
    ) -> Self {
        let experts: Vec<Arc<Expert>> = held.iter().map(|(_, e)| e.clone()).collect();
        let job = std::thread::spawn(move || -> Result<Vec<f32>> {
            let outs =
                match crate::inference::offload::cpu_experts::batched(&experts, &x, swiglu_limit) {
                    Some(outs) => outs,
                    None => experts
                        .iter()
                        .map(|e| {
                            let row = Tensor::from_vec(x.clone(), (1, x.len()), &Device::Cpu)?;
                            e.forward(&row, swiglu_limit)?
                                .flatten_all()?
                                .to_vec1::<f32>()
                        })
                        .collect::<Result<Vec<_>>>()?,
                };
            let mut sum = vec![0f32; x.len()];
            for (out, w) in outs.iter().zip(&weights) {
                for (s, v) in sum.iter_mut().zip(out) {
                    *s += w * v;
                }
            }
            Ok(sum)
        });
        Self {
            job: Some(job),
            indexes,
            held,
        }
    }

    /// The summed output, and the experts to release.
    pub fn wait(mut self) -> Result<(Vec<f32>, Vec<(usize, Arc<Expert>)>)> {
        let job = self
            .job
            .take()
            .ok_or_else(|| crate::tensor::Error::msg("a deferred expert job joined twice"))?;
        let sum = job
            .join()
            .map_err(|_| crate::tensor::Error::msg("a deferred expert job panicked"))??;
        Ok((sum, std::mem::take(&mut self.held)))
    }
}

impl Drop for DeferredExperts {
    fn drop(&mut self) {
        if let Some(job) = self.job.take() {
            let _ = job.join();
        }
    }
}

/// Run the `active` experts of one block: resident ones on the cards, the misses on the cores at
/// the same instant. `assign[e]` is expert `e`'s `(token, weight)` rows; `xv[t]` is token `t`'s
/// row; `active_refs[i]` is the handle for `active[i]`; `n` is the batch width; `observed` is true
/// when a calibration wants each expert's rows, which forces the per-expert path. A decode that
/// runs its misses beside the cards may hand up to `defer` of the host's experts, lightest routing
/// weight first, to a job that outlives the call; the caller finds them in the third answer.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_routed(
    off: &ExpertOffload,
    active: &[usize],
    active_refs: &[Arc<Expert>],
    assign: &[Vec<(usize, f32)>],
    xv: &[Vec<f32>],
    n: usize,
    dim: usize,
    swiglu_limit: f32,
    observed: bool,
    defer: usize,
) -> Routed {
    let ref_of: HashMap<usize, Arc<Expert>> = active
        .iter()
        .copied()
        .zip(active_refs.iter().cloned())
        .collect();
    let rows_of = |e: usize| -> Vec<f32> {
        assign[e]
            .iter()
            .flat_map(|&(t, _)| xv[t].iter().copied())
            .collect()
    };
    let mut done: Vec<Option<Result<(Vec<f32>, Vec<f32>)>>> =
        (0..active.len()).map(|_| None).collect();
    #[allow(clippy::type_complexity)]
    let mut cpu_side: Vec<Option<Result<(Vec<f32>, Vec<f32>, Option<Arc<Expert>>)>>> =
        (0..active.len()).map(|_| None).collect();
    let fetch = |e: usize| {
        ref_of
            .get(&e)
            .cloned()
            .ok_or_else(|| crate::tensor::Error::msg("expert not in the layer's set"))
    };
    // A decode's active experts all read the one token's row, so they run as one block per card;
    // the per-expert path serves a prompt's batch and stands in when the block path is absent.
    let mut deferred: Option<DeferredExperts> = None;
    let batched = n == 1 && off.run_batch.is_some();
    if batched {
        let rb = off.run_batch.as_ref().unwrap();
        if let (true, Some(is_on)) = (!observed, off.on_card.as_ref()) {
            // The cards run the experts they hold while the cores run, at the same instant, the
            // ones they do not. One core keeps up with a decode's handful of misses beside the
            // block, so the two finish together instead of summing.
            let mut cpu_pos: Vec<usize> = (0..active.len())
                .filter(|&i| !is_on(&active_refs[i]))
                .collect();
            // The lightest of the host's experts leave for the deferred job: their share of the
            // layer's output is the smallest, and the stream takes it a layer late.
            let weight_of = |i: usize| assign[active[i]].first().map_or(0.0, |&(_, w)| w);
            let mut later: Vec<usize> = cpu_pos.clone();
            later.sort_by(|&a, &b| weight_of(a).total_cmp(&weight_of(b)));
            later.truncate(defer.min(cpu_pos.len()));
            cpu_pos.retain(|i| !later.contains(i));
            done = std::thread::scope(|s| {
                let g = s.spawn(|| {
                    let t = std::time::Instant::now();
                    let r = rb(active, &fetch, &xv[0], swiglu_limit);
                    crate::inference::offload::prof_record(
                        "moe cards block",
                        t.elapsed().as_nanos() as u64,
                    );
                    r
                });
                let cpu_t = std::time::Instant::now();
                let miss_refs: Vec<Arc<Expert>> =
                    cpu_pos.iter().map(|&i| active_refs[i].clone()).collect();
                if let Some(outs) =
                    crate::inference::offload::cpu_experts::batched(&miss_refs, &xv[0], swiglu_limit)
                {
                    for ((&i, out), expert) in cpu_pos.iter().zip(outs).zip(miss_refs) {
                        cpu_side[i] = Some(Ok((Vec::new(), out, Some(expert))));
                    }
                } else {
                    for &i in &cpu_pos {
                        let e = active[i];
                        let r = (|| {
                            let expert = active_refs[i].clone();
                            let xe =
                                Tensor::from_vec(rows_of(e), (assign[e].len(), dim), &Device::Cpu)?;
                            let out = expert
                                .forward(&xe, swiglu_limit)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            Ok((Vec::new(), out, Some(expert)))
                        })();
                        cpu_side[i] = Some(r);
                    }
                }
                // Started once the cores are free of this layer's own experts, so the job and
                // the cards' block run together and the job outlives both.
                let host_ns = cpu_t.elapsed().as_nanos() as u64;
                if !later.is_empty() {
                    deferred = Some(DeferredExperts::start(
                        later.clone(),
                        later
                            .iter()
                            .map(|&i| (active[i], active_refs[i].clone()))
                            .collect(),
                        later.iter().map(|&i| weight_of(i)).collect(),
                        xv[0].clone(),
                        swiglu_limit,
                    ));
                }
                crate::inference::offload::prof_record("moe host misses", host_ns);
                g.join().unwrap()
            });
        } else {
            done = rb(active, &fetch, &xv[0], swiglu_limit);
        }
    } else {
        // A small batch is a speculative verify, not a prompt's prefill. Its resident experts run
        // in one grouped launch per card over their rows, so the per-layer launch a decode pays for
        // every token is paid once for the whole block; the non-resident ones run in one pool region
        // over their rows, the same amortising. A prefill's wide batch stays on the lanes, where the
        // cards earn their crossing.
        const VERIFY_BATCH_MAX: usize = 32;
        let pooled = n <= VERIFY_BATCH_MAX
            && !observed
            && off.run_multi.is_some()
            && (|| -> Option<()> {
                // One instance per (expert, routed row), the rows kept in the expert's assignment
                // order so a card's answers scatter back where the host's would.
                let mut eidx: Vec<usize> = Vec::new();
                let mut xrows: Vec<&[f32]> = Vec::new();
                let mut inst_of: Vec<Vec<usize>> = vec![Vec::new(); active.len()];
                for (i, &e) in active.iter().enumerate() {
                    for &(t, _w) in &assign[e] {
                        inst_of[i].push(eidx.len());
                        eidx.push(e);
                        xrows.push(&xv[t]);
                    }
                }
                let rm = off.run_multi.as_ref().unwrap();
                let gpu = rm(&eidx, &xrows, &fetch, swiglu_limit);
                // An expert is resident when its first instance came back from a card; then all of
                // its instances did. The rest go to the host in one region.
                let miss_i: Vec<usize> = (0..active.len())
                    .filter(|&i| {
                        inst_of[i]
                            .first()
                            .map(|&k| gpu[k].is_none())
                            .unwrap_or(false)
                    })
                    .collect();
                let miss_refs: Vec<Arc<Expert>> =
                    miss_i.iter().map(|&i| active_refs[i].clone()).collect();
                let miss_rows: Vec<Vec<f32>> = miss_i.iter().map(|&i| rows_of(active[i])).collect();
                let cpu = crate::inference::offload::cpu_experts::multirow(
                    &miss_refs,
                    &miss_rows,
                    dim,
                    swiglu_limit,
                )?;
                let mut miss_at = 0usize;
                for i in 0..active.len() {
                    if miss_at < miss_i.len() && miss_i[miss_at] == i {
                        done[i] = Some(Ok((Vec::new(), cpu[miss_at].clone())));
                        miss_at += 1;
                        continue;
                    }
                    let mut out = Vec::with_capacity(inst_of[i].len() * dim);
                    for &k in &inst_of[i] {
                        let row = gpu[k].as_ref()?.as_ref().ok()?;
                        out.extend_from_slice(row);
                    }
                    done[i] = Some(Ok((Vec::new(), out)));
                }
                Some(())
            })()
            .is_some();
        if !pooled {
            let (ran, fetch_ns, run_ns) = off.run_all(active, n, swiglu_limit, &fetch, &rows_of);
            done = ran;
            if crate::inference::offload::current().is_some() {
                crate::inference::offload::prof_record("moe expert fetch", fetch_ns);
                crate::inference::offload::prof_record("moe expert run", run_ns);
            }
        }
    }
    if crate::inference::offload::current().is_some() {
        // The lanes' own time, split: their threads carry no recorder, so it is recorded here after
        // the join.
        for (name, a) in [
            ("moe expert hold", &off.timings[0]),
            ("moe expert kernels", &off.timings[1]),
            ("count experts on card", &off.timings[2]),
        ] {
            crate::inference::offload::prof_record(
                name,
                a.swap(0, std::sync::atomic::Ordering::Relaxed),
            );
        }
    }
    (done, cpu_side, deferred)
}
