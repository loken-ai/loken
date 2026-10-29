//! The routed experts of one prompt layer, split between the cards and the host.
//!
//! After the router, each expert of the layer has its rows. An expert a card keeps runs there
//! from the card's copy. An expert no card keeps runs on a card when a copy of it costs less
//! than its rows cost the host, from the measured time of a copy per byte against the measured
//! time of a host row; it stays on the host otherwise. The copies the cards need are read from
//! the file into page-locked memory in contiguous runs of adjacent experts, around the page
//! cache, while the cards compute the experts already there and the host computes its own. A
//! copy made for the layer is let go after it unless the card has the room free to keep it: a
//! prompt never takes a kept expert's place.
//!
//! Model-neutral: it speaks experts, their rows and their storage. The gate, the shared expert
//! and the weighted gather stay with the model.

use super::experts::{DeviceLayer, Expert, ExpertOffload};
use super::moe_dispatch::Routed;
use crate::tensor::{Device, Result, Tensor};
use std::sync::Arc;

/// Bytes per KiB: the copy cost is kept per KiB so that an integer holds it to a percent.
pub(crate) const KIB: usize = 1024;

/// One active expert of a prompt layer, as the plan sees it.
pub struct PromptExpert {
    /// The rows the batch routes to it.
    pub rows: usize,
    /// Its three weights' bytes as stored.
    pub bytes: usize,
    /// A card keeps it: it runs there, with no copy.
    pub held: bool,
    /// Its bytes are in the host's memory now: a copy reads them from there, not from the file.
    pub resident: bool,
    /// The address and length of its three weights' storage, in the order the file reads them,
    /// when they are mapped.
    pub storage: Option<[(usize, usize); 3]>,
    /// Nanoseconds a copy to the card that would run it takes per KiB; 0 when none was measured.
    pub copy_ns_per_kib: u64,
}

/// Where each active expert of a prompt layer runs, and the file spans the cards' copies are
/// read from.
#[derive(Debug, Default, PartialEq)]
pub struct PromptPlan {
    /// Positions in the active list the cards run, ascending.
    pub on_cards: Vec<usize>,
    /// Positions in the active list the host runs, ascending.
    pub on_host: Vec<usize>,
    /// The mapped spans (address, length) read into page-locked memory for the cards' copies:
    /// one per run of adjacent experts and per weight, in the order the cards consume them.
    pub spans: Vec<(usize, usize)>,
    /// Positions among `on_cards` whose expert is copied for the layer, ascending.
    pub copied: Vec<usize>,
}

/// Whether a copy of `bytes` to a card at `copy_ns_per_kib` costs less than `rows` rows cost the
/// host at `host_row_ns`. A cost not measured yet counts as nothing, as an admission takes an
/// expert on sight before an upload was timed.
pub fn copy_pays(bytes: usize, rows: usize, copy_ns_per_kib: u64, host_row_ns: u64) -> bool {
    if copy_ns_per_kib == 0 || host_row_ns == 0 {
        return true;
    }
    let copy = copy_ns_per_kib as u128 * bytes.div_ceil(KIB) as u128;
    let host = host_row_ns as u128 * rows as u128;
    copy <= host
}

/// Split `experts` between the cards and the host at `host_row_ns` nanoseconds per host row, and
/// gather the spans the cards' copies are read from. An expert without rows stays on the host,
/// where it runs nothing.
pub fn plan(experts: &[PromptExpert], host_row_ns: u64) -> PromptPlan {
    let mut out = PromptPlan::default();
    for (i, e) in experts.iter().enumerate() {
        let to_card =
            e.held || (e.rows > 0 && copy_pays(e.bytes, e.rows, e.copy_ns_per_kib, host_row_ns));
        if to_card {
            out.on_cards.push(i);
            if !e.held {
                out.copied.push(i);
            }
        } else {
            out.on_host.push(i);
        }
    }
    let from_file: Vec<[(usize, usize); 3]> = out
        .copied
        .iter()
        .filter(|&&i| !experts[i].resident)
        .filter_map(|&i| experts[i].storage)
        .collect();
    out.spans = runs(&from_file);
    out
}

/// The spans of `storage`, each an expert's weights in file order, joined where the experts lie
/// side by side: one span per run and per weight, the runs in the order given, each run's weights
/// in file order. Two experts are side by side when each weight of the second starts where the
/// first's ends.
pub fn runs(storage: &[[(usize, usize); 3]]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut i = 0;
    while i < storage.len() {
        let mut run = storage[i];
        let mut j = i + 1;
        while j < storage.len() && (0..3).all(|w| storage[j][w].0 == run[w].0 + run[w].1) {
            for w in 0..3 {
                run[w].1 += storage[j][w].1;
            }
            j += 1;
        }
        spans.extend(run);
        i = j;
    }
    spans
}

/// Run the active experts of a prompt layer by `plan`: the cards' on the lanes over the layer's
/// context `ctx`, the host's on the cores at the same instant. `assign[e]` is expert `e`'s
/// `(token, weight)` rows, `xv[t]` token `t`'s row, `active_refs[i]` the handle of `active[i]`.
/// An expert a lane declines answers `None` for the caller to run. The host's time per row is
/// smoothed into the offload's figure.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_prompt(
    off: &ExpertOffload,
    hooks: &DeviceLayer,
    ctx: &Arc<dyn std::any::Any + Send + Sync>,
    active: &[usize],
    active_refs: &[Arc<Expert>],
    assign: &[Vec<(usize, f32)>],
    xv: &[&[f32]],
    dim: usize,
    swiglu_limit: f32,
    plan: &PromptPlan,
) -> Routed {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let mut done: Vec<Option<Result<(Vec<f32>, Vec<f32>)>>> =
        (0..active.len()).map(|_| None).collect();
    #[allow(clippy::type_complexity)]
    let mut cpu_side: Vec<Option<Result<(Vec<f32>, Vec<f32>, Option<Arc<Expert>>)>>> =
        (0..active.len()).map(|_| None).collect();
    let lanes = off.lanes.max(1);
    let next = AtomicUsize::new(0);
    let slots = std::sync::Mutex::new(&mut done);
    let host_ns = std::thread::scope(|scope| {
        for _ in 0..lanes {
            let (next, slots) = (&next, &slots);
            scope.spawn(move || loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                let Some(&i) = plan.on_cards.get(k) else {
                    break;
                };
                let e = active[i];
                let toks: Vec<u32> = assign[e].iter().map(|&(t, _)| t as u32).collect();
                let r = (hooks.prompt)(e % lanes, ctx, &active_refs[i], &toks, swiglu_limit)
                    .map(|r| r.map(|out| (Vec::new(), out)));
                slots.lock().unwrap_or_else(|e| e.into_inner())[i] = r;
            });
        }
        // The host's experts run while the lanes run the cards': the two finish together.
        let started = std::time::Instant::now();
        let made: Vec<(usize, Result<(Vec<f32>, Vec<f32>, Option<Arc<Expert>>)>)> = {
            use rayon::prelude::*;
            plan.on_host
                .par_iter()
                .map(|&i| {
                    let e = active[i];
                    let expert = active_refs[i].clone();
                    let r = (|| {
                        let rows: Vec<f32> = assign[e]
                            .iter()
                            .flat_map(|&(t, _)| xv[t].iter().copied())
                            .collect();
                        let xe = Tensor::from_vec(rows, (assign[e].len(), dim), &Device::Cpu)?;
                        let out = super::on_host(|| expert.forward(&xe, swiglu_limit))?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        Ok((Vec::new(), out, Some(expert)))
                    })();
                    (i, r)
                })
                .collect()
        };
        for (i, r) in made {
            cpu_side[i] = Some(r);
        }
        started.elapsed().as_nanos() as u64
    });
    let rows: usize = plan.on_host.iter().map(|&i| assign[active[i]].len()).sum();
    if rows > 0 {
        use std::sync::atomic::Ordering::Relaxed;
        let per = host_ns / rows as u64;
        let old = off.prompt_host_row_ns.load(Relaxed);
        off.prompt_host_row_ns.store(
            old - old / super::admission::SMOOTHING + per / super::admission::SMOOTHING,
            Relaxed,
        );
    }
    super::prof_record("moe prompt host experts", host_ns);
    super::prof_record("count prompt experts on host", plan.on_host.len() as u64);
    super::prof_record("count prompt experts on cards", plan.on_cards.len() as u64);
    super::prof_record("count prompt experts copied", plan.copied.len() as u64);
    (done, cpu_side, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn expert(rows: usize, held: bool, resident: bool, at: usize) -> PromptExpert {
        PromptExpert {
            rows,
            bytes: 3 * KIB,
            held,
            resident,
            storage: Some([(at, KIB), (1000 * KIB + at, KIB), (2000 * KIB + at, KIB)]),
            copy_ns_per_kib: 100,
        }
    }

    /// A kept expert runs on its card with no copy; one not kept goes to a card when the copy
    /// costs less than its rows cost the host, and stays on the host otherwise; an unmeasured
    /// cost sends it to a card.
    #[test]
    fn the_cards_take_the_kept_experts_and_the_copies_that_pay() {
        // A copy of 3 KiB at 100 ns per KiB costs 300 ns; a host row costs 100 ns.
        let experts = vec![
            expert(1, true, false, 0),
            expert(1, false, false, KIB),
            expert(3, false, false, 2 * KIB),
            expert(8, false, false, 3 * KIB),
            expert(0, false, false, 4 * KIB),
        ];
        let p = plan(&experts, 100);
        assert_eq!(p.on_cards, vec![0, 2, 3]);
        assert_eq!(p.on_host, vec![1, 4]);
        assert_eq!(p.copied, vec![2, 3]);
        let unmeasured = plan(&experts, 0);
        assert_eq!(unmeasured.on_cards, vec![0, 1, 2, 3]);
        assert_eq!(unmeasured.on_host, vec![4]);
        let mut free = experts;
        for e in &mut free {
            e.copy_ns_per_kib = 0;
        }
        assert_eq!(plan(&free, 100).on_host, vec![4]);
    }

    /// Only the experts copied from the file are read, adjacent ones as one span per weight; a
    /// resident one is copied from memory and a kept one not at all.
    #[test]
    fn the_spans_cover_the_copied_experts_only_in_runs() {
        let experts = vec![
            expert(8, false, false, 0),
            expert(8, false, false, KIB),
            expert(8, true, false, 2 * KIB),
            expert(8, false, false, 3 * KIB),
            expert(8, false, true, 4 * KIB),
            expert(8, false, false, 5 * KIB),
            expert(1, false, false, 6 * KIB),
            expert(8, false, false, 7 * KIB),
        ];
        let p = plan(&experts, 100);
        assert_eq!(p.on_host, vec![6]);
        assert_eq!(p.copied, vec![0, 1, 3, 4, 5, 7]);
        assert_eq!(
            p.spans,
            vec![
                (0, 2 * KIB),
                (1000 * KIB, 2 * KIB),
                (2000 * KIB, 2 * KIB),
                (3 * KIB, KIB),
                (1003 * KIB, KIB),
                (2003 * KIB, KIB),
                (5 * KIB, KIB),
                (1005 * KIB, KIB),
                (2005 * KIB, KIB),
                (7 * KIB, KIB),
                (1007 * KIB, KIB),
                (2007 * KIB, KIB),
            ]
        );
        let bytes: usize = p.spans.iter().map(|s| s.1).sum();
        assert_eq!(
            bytes,
            5 * 3 * KIB,
            "the file is read once per copied expert"
        );
    }

    /// On a card, a prompt's experts give the host's bits whether the card keeps them, copies
    /// them for the layer or leaves them to the host. A copy takes no kept expert's place: it
    /// goes with the layer when the card has no room to keep it, and stays when it has.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_card_runs_a_prompts_copies_beside_its_kept_experts() {
        use crate::inference::offload::cuda::{lanes, Card};
        use crate::inference::offload::projection::Projection;
        use crate::inference::offload::room;
        use crate::tensor::quantized::{GgmlDType, QTensor};
        let Ok(dev) = crate::tensor::cuda::CudaDevice::new(0) else {
            eprintln!("no CUDA device; the card's prompt experts are NOT covered by this run");
            return;
        };
        let (dim, inter, limit, n) = (1024usize, 512usize, 7.0f32, 40usize);
        let draw = |seed: u32, len: usize, scale: f32| -> Vec<f32> {
            (0..len)
                .map(|i| {
                    (((i as u32)
                        .wrapping_add(seed.wrapping_mul(0x9E37_79B9))
                        .wrapping_mul(2_654_435_761)
                        .rotate_left(11)
                        >> 8) as f32
                        / (1 << 24) as f32
                        - 0.5)
                        * scale
                })
                .collect()
        };
        let quant = |w: Vec<f32>, rows: usize, cols: usize, dtype: GgmlDType| {
            let t = Tensor::from_vec(w, (rows, cols), &Device::Cpu).unwrap();
            Projection::Quant(Arc::new(QTensor::quantize(&t, dtype).unwrap()))
        };
        let experts: Vec<Arc<Expert>> = (0..4u32)
            .map(|j| {
                Arc::new(Expert {
                    w1: quant(
                        draw(10 * j + 1, inter * dim, 0.08),
                        inter,
                        dim,
                        GgmlDType::Iq2Xxs,
                    ),
                    w3: quant(
                        draw(10 * j + 2, inter * dim, 0.08),
                        inter,
                        dim,
                        GgmlDType::Iq2Xxs,
                    ),
                    w2: quant(
                        draw(10 * j + 3, dim * inter, 0.08),
                        dim,
                        inter,
                        GgmlDType::Q2K,
                    ),
                })
            })
            .collect();
        let x = draw(99, n * dim, 6.0);
        let xv: Vec<&[f32]> = x.chunks(dim).collect();
        // Expert j serves the rows t with (t + j) % 3 != 0.
        let assign: Vec<Vec<(usize, f32)>> = (0..4)
            .map(|j| {
                (0..n)
                    .filter(|t| (t + j) % 3 != 0)
                    .map(|t| (t, 1.0))
                    .collect()
            })
            .collect();
        let active: Vec<usize> = (0..4).collect();
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
        let host: Vec<Vec<u32>> = (0..4)
            .map(|j| {
                let rows: Vec<f32> = assign[j]
                    .iter()
                    .flat_map(|&(t, _)| xv[t].iter().copied())
                    .collect();
                let xe = Tensor::from_vec(rows, (assign[j].len(), dim), &Device::Cpu).unwrap();
                let out = super::super::on_host(|| experts[j].forward(&xe, limit)).unwrap();
                bits(&out.flatten_all().unwrap().to_vec1::<f32>().unwrap())
            })
            .collect();
        let bytes = experts[0].bytes();
        // A room with a place for two kept experts and none beyond.
        let room = room::open(dev.ordinal(), 2 * bytes + 1, 0);
        let card = Arc::new(Card::new(dev.clone(), room, 1));
        let off = lanes(vec![card.clone()]);
        let hooks = off.device.as_ref().unwrap();
        assert!(
            card.admit(&experts[0], true).is_some() && card.admit(&experts[1], true).is_some(),
            "the card took no expert"
        );
        let describe = |costs: [u64; 4]| -> Vec<PromptExpert> {
            experts
                .iter()
                .enumerate()
                .map(|(j, e)| PromptExpert {
                    rows: assign[j].len(),
                    bytes: e.bytes(),
                    held: card.holds(e),
                    resident: true,
                    storage: None,
                    copy_ns_per_kib: costs[j],
                })
                .collect()
        };
        let check = |plan: &PromptPlan, what: &str| {
            let ctx = (hooks.begin)(&x, dim).expect("the card has room for the layer's rows");
            let (done, cpu_side, _) = run_prompt(
                &off, hooks, &ctx, &active, &experts, &assign, &xv, dim, limit, plan,
            );
            for j in 0..4 {
                let out = match (&done[j], &cpu_side[j]) {
                    (Some(Ok((_, out))), _) => out.clone(),
                    (_, Some(Ok((_, out, _)))) => out.clone(),
                    (Some(Err(e)), _) | (_, Some(Err(e))) => panic!("{what}: expert {j}: {e}"),
                    _ => panic!("{what}: expert {j} ran nowhere"),
                };
                assert_eq!(bits(&out), host[j], "{what}: expert {j}");
            }
        };
        // Expert 2 copied for the layer, expert 3 costed to the host: the card keeps 0 and 1 and
        // has no room for 2.
        let mixed = plan(&describe([0, 0, 0, 1 << 40]), 1);
        assert_eq!(
            (&mixed.on_cards, &mixed.on_host),
            (&vec![0, 1, 2], &vec![3])
        );
        assert_eq!(mixed.copied, vec![2]);
        check(&mixed, "the card full");
        assert!(
            card.holds(&experts[0]) && card.holds(&experts[1]),
            "a kept expert was traded"
        );
        assert!(
            !card.holds(&experts[2]),
            "a copy stayed on a card without room"
        );
        assert!((hooks.copy_ns_per_kib)(0) > 0, "the copy went untimed");
        // The room opened: the copies stay.
        let (_, total) = crate::tensor::cuda_ext::mem_get_info(&Device::Cuda(dev.clone())).unwrap();
        room::open(dev.ordinal(), total, 0);
        let roomy = plan(&describe([0; 4]), 1);
        assert_eq!(roomy.copied, vec![2, 3]);
        check(&roomy, "the card with room");
        assert!(
            (0..4).all(|j| card.holds(&experts[j])),
            "a copy went with the layer though the card had room"
        );
        let kept = plan(&describe([0; 4]), 1);
        assert!(kept.copied.is_empty() && kept.on_cards == vec![0, 1, 2, 3]);
        check(&kept, "every expert kept");
    }

    /// Experts whose weights lie apart, or are not mapped, make runs of one.
    #[test]
    fn a_run_ends_where_the_storage_does() {
        let mut gap = expert(8, false, false, 3 * KIB);
        gap.storage = Some([(3 * KIB, KIB), (1003 * KIB, KIB), (2004 * KIB, KIB)]);
        let mut unmapped = expert(8, false, false, 4 * KIB);
        unmapped.storage = None;
        let experts = vec![
            expert(8, false, false, 0),
            expert(8, false, false, KIB),
            gap,
            unmapped,
        ];
        let p = plan(&experts, 100);
        assert_eq!(p.copied, vec![0, 1, 2, 3]);
        assert_eq!(
            p.spans,
            vec![
                (0, 2 * KIB),
                (1000 * KIB, 2 * KIB),
                (2000 * KIB, 2 * KIB),
                (3 * KIB, KIB),
                (1003 * KIB, KIB),
                (2004 * KIB, KIB),
            ]
        );
    }
}
