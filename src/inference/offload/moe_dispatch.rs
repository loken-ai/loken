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
);

/// Run the `active` experts of one block: resident ones on the cards, the misses on the cores at
/// the same instant. `assign[e]` is expert `e`'s `(token, weight)` rows; `xv[t]` is token `t`'s
/// row; `active_refs[i]` is the handle for `active[i]`; `n` is the batch width; `observed` is true
/// when a calibration wants each expert's rows, which forces the per-expert path.
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
    let batched = n == 1 && off.run_batch.is_some();
    if batched {
        let rb = off.run_batch.as_ref().unwrap();
        if let (true, Some(is_on)) = (!observed, off.on_card.as_ref()) {
            // The cards run the experts they hold while the cores run, at the same instant, the
            // ones they do not. One core keeps up with a decode's handful of misses beside the
            // block, so the two finish together instead of summing.
            let cpu_pos: Vec<usize> = (0..active.len())
                .filter(|&i| !is_on(&active_refs[i]))
                .collect();
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
                crate::inference::offload::prof_record(
                    "moe host misses",
                    cpu_t.elapsed().as_nanos() as u64,
                );
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
    (done, cpu_side)
}
