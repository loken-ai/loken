//! One routed expert, and a set of lanes that run experts somewhere other than the CPU.

use super::projection::Projection;
use crate::tensor::{Device, Result, Tensor};
use std::sync::Arc;

/// `e^x` in single precision from additions, multiplications and a division alone, each rounded
/// once in the order written: a range reduction by ln 2 and a degree-five polynomial, scaled by
/// two powers of two. The cards compute it with the same steps, so the SwiGLU of an expert is
/// the same bits wherever the expert runs. Within two units in the last place of `exp`.
pub fn exp(x: f32) -> f32 {
    const LOG2E: f32 = std::f32::consts::LOG2_E;
    const LN2_HI: f32 = 0.693_359_4;
    const LN2_LO: f32 = -2.121_944_4e-4;
    const P: [f32; 6] = [
        1.987_569_1e-4,
        1.398_199_9e-3,
        8.333_452e-3,
        4.166_579_6e-2,
        0.166_666_65,
        0.5,
    ];
    if x.is_nan() {
        return x;
    }
    if x > 88.722_84 {
        return f32::INFINITY;
    }
    if x < -103.972_08 {
        return 0.0;
    }
    let k = (x * LOG2E).round_ties_even();
    let r = (x - k * LN2_HI) - k * LN2_LO;
    let mut p = P[0];
    for c in &P[1..] {
        p = p * r + c;
    }
    let y = p * (r * r) + r + 1.0;
    let k = k as i32;
    let (k1, k2) = (k / 2, k - k / 2);
    let two = |e: i32| f32::from_bits(((e + 127) as u32) << 23);
    y * two(k1) * two(k2)
}

/// The SwiGLU of one gate and up value, the gate clamped from above and the up value on both
/// sides when `limit` is positive: `silu(g) * u`, its steps in the order the cards take them.
#[inline]
pub fn swiglu(g: f32, u: f32, limit: f32) -> f32 {
    let (g, u) = if limit > 0.0 {
        (g.min(limit), u.clamp(-limit, limit))
    } else {
        (g, u)
    };
    g / (1.0 + exp(-g)) * u
}

/// One SwiGLU expert. `w1`/`w3` are [inter, dim], `w2` is [dim, inter].
pub struct Expert {
    pub w1: Projection,
    pub w2: Projection,
    pub w3: Projection,
}

impl Expert {
    /// Resident and total pages over the three projections.
    #[cfg(target_os = "linux")]
    pub fn resident_pages(&self) -> (usize, usize) {
        let (a, b, c) = (
            self.w1.resident_pages(),
            self.w2.resident_pages(),
            self.w3.resident_pages(),
        );
        (a.0 + b.0 + c.0, a.1 + b.1 + c.1)
    }

    /// The bytes of the three projections as stored.
    pub fn bytes(&self) -> usize {
        self.w1.bytes() + self.w2.bytes() + self.w3.bytes()
    }

    /// Drop the three projections from the page cache.
    pub fn dont_need(&self) {
        self.w1.dont_need();
        self.w2.dont_need();
        self.w3.dont_need();
    }

    /// The address range the three projections' storage covers when they lie side by side in
    /// one mapping, apart by less than a page of padding: an expert stored as one unit. `None`
    /// when they lie apart, as the slices of three stacks do, or are not mapped.
    #[cfg(unix)]
    pub fn storage_unit(&self) -> Option<(usize, usize)> {
        let page = crate::tensor::mapped::page_size()?;
        let mut spans = [
            self.w1.storage_span()?,
            self.w3.storage_span()?,
            self.w2.storage_span()?,
        ];
        spans.sort_unstable();
        for pair in spans.windows(2) {
            let end = pair[0].0 + pair[0].1;
            if pair[1].0 < end || pair[1].0 - end >= page {
                return None;
            }
        }
        let last = spans[2];
        Some((spans[0].0, last.0 + last.1 - spans[0].0))
    }

    /// Read the three projections into memory now, returning once they are there: the blocking
    /// form of `will_need`, for a thread whose job is to wait on the read.
    pub fn populate(&self) {
        #[cfg(unix)]
        {
            // MADV_POPULATE_READ: fault the range in as a read would, without mapping it for a
            // caller to touch.
            const POPULATE_READ: libc::c_int = 22;
            if let Some((start, len)) = self.storage_unit() {
                crate::tensor::mapped::advise_span(start, len, POPULATE_READ, false);
                return;
            }
            for w in [&self.w1, &self.w3, &self.w2] {
                if let Some((start, len)) = w.storage_span() {
                    crate::tensor::mapped::advise_span(start, len, POPULATE_READ, false);
                }
            }
        }
    }

    /// Start reading the three projections in: one request over the unit when the expert is
    /// stored as one, a request per projection otherwise.
    pub fn will_need(&self) {
        #[cfg(unix)]
        if let Some((start, len)) = self.storage_unit() {
            crate::tensor::mapped::advise_span(start, len, libc::MADV_WILLNEED, false);
            return;
        }
        self.w1.will_need();
        self.w2.will_need();
        self.w3.will_need();
    }

    pub fn dense(w1: Tensor, w2: Tensor, w3: Tensor) -> Self {
        Self {
            w1: Projection::Dense(w1),
            w2: Projection::Dense(w2),
            w3: Projection::Dense(w3),
        }
    }

    /// SwiGLU over every row of `x` [n, dim], returning [n, dim]. A per-token routing weight is a
    /// scalar and w2 is linear, so it is applied by the caller after this rather than folded in.
    pub fn forward(&self, x: &Tensor, swiglu_limit: f32) -> Result<Tensor> {
        self.forward_seen(x, swiglu_limit, None)
    }

    /// `forward`, handing the rows that enter w2 to `seen` before w2 runs.
    pub fn forward_seen(
        &self,
        x: &Tensor,
        swiglu_limit: f32,
        seen: Option<&dyn Fn(&[f32])>,
    ) -> Result<Tensor> {
        let gate = self.w1.apply(x)?.to_vec2::<f32>()?;
        let up = self.w3.apply(x)?.to_vec2::<f32>()?;
        let (n, inter) = (gate.len(), gate[0].len());
        let mut h = vec![0f32; n * inter];
        for t in 0..n {
            for k in 0..inter {
                h[t * inter + k] = swiglu(gate[t][k], up[t][k], swiglu_limit);
            }
        }
        if let Some(seen) = seen {
            seen(&h);
        }
        let h = Tensor::from_vec(h, (n, inter), &Device::Cpu)?;
        self.w2.apply(&h)
    }
}

/// Routed experts run somewhere other than the CPU, `lanes` of them at once. `run(lane, expert, rows,
/// swiglu_limit)` takes an expert's input rows, row-major, and returns the rows entering its w2 and
/// its output, or `None` for an expert whose storage it cannot run, which the CPU then runs.
/// The token context of one layer's routed experts held where the offload computes: the layer's
/// input rows on each card, sent once for all the experts of the layer that card runs. The experts
/// read their rows there and answer their outputs, which the host adds in the order it adds every
/// expert in, so a prompt's sum is the same whichever experts ran on a card.
pub struct DeviceLayer {
    /// Open a layer's context from its input `x` [n, dim]. `None` when the offload has no room.
    #[allow(clippy::type_complexity)]
    pub begin:
        Box<dyn Fn(&[f32], usize) -> Option<Arc<dyn std::any::Any + Send + Sync>> + Send + Sync>,
    /// Run `expert` on lane `lane` over its rows `toks` of the layer, answering its outputs
    /// `[toks, dim]` in the order of `toks`. `None` where the lane declines and the host runs it.
    #[allow(clippy::type_complexity)]
    pub run: Box<
        dyn Fn(
                usize,
                &Arc<dyn std::any::Any + Send + Sync>,
                &Expert,
                &[u32],
                f32,
            ) -> Option<Result<Vec<f32>>>
            + Send
            + Sync,
    >,
}

pub struct ExpertOffload {
    /// Nanoseconds the host takes for one routed expert of a decode, smoothed over the blocks
    /// that ran some there: what a use on a card saves.
    pub host_expert_ns: Arc<std::sync::atomic::AtomicU64>,
    /// A prompt's experts held on the cards' side (see `DeviceLayer`), where the offload has it.
    pub device: Option<DeviceLayer>,
    pub lanes: usize,
    #[allow(clippy::type_complexity)]
    /// `(lane, tokens in the batch, expert, its rows, the SwiGLU clamp)`. The batch's size is
    /// there because an expert that one token routed to looks the same in a batch of six hundred
    /// as in a decode, and what is worth keeping on a card differs between the two.
    pub run: Box<
        dyn Fn(usize, usize, &Expert, &[f32], f32) -> Option<Result<(Vec<f32>, Vec<f32>)>>
            + Send
            + Sync,
    >,
    /// Across the lanes, since last read: nanoseconds holding a weight on a card, nanoseconds
    /// in its kernels and their wait, and the expert rows the cards answered. The lanes'
    /// threads carry no recorder; the caller records these after the join.
    pub timings: Arc<[std::sync::atomic::AtomicU64; 3]>,
    /// Keep `expert` (number `id`) ahead of any request, where the offload keeps experts.
    /// Answers whether it was taken.
    pub warm: Option<Box<dyn Fn(usize, &Expert) -> bool + Send + Sync>>,
    /// A decode's active experts, which all read the one token's row, run as a block: `active`
    /// the expert numbers, `fetch` builds one, `x` the shared row. Answers in the order asked,
    /// `None` where the expert is not resident and the host must run it. `None` on the field when
    /// the offload has no block path.
    #[allow(clippy::type_complexity)]
    pub run_batch: Option<
        Box<
            dyn Fn(
                    &[usize],
                    &(dyn Fn(usize) -> Result<Arc<Expert>> + Sync),
                    &[f32],
                    f32,
                ) -> Vec<Option<Result<(Vec<f32>, Vec<f32>)>>>
                + Send
                + Sync,
        >,
    >,
    /// Whether `expert` is resident where the block path would run it with no upload. Read only,
    /// so the caller can start the host's own experts on the cores in the same instant the cards
    /// run theirs, instead of after. `None` on the field when the offload has no block path.
    #[allow(clippy::type_complexity)]
    pub on_card: Option<Box<dyn Fn(&Expert) -> bool + Send + Sync>>,
    /// A verify's routed instances, one activation row each: `eidx` the expert number per instance,
    /// `xrows` its row, `fetch` builds an expert. The resident instances run in one grouped launch
    /// per card, so the block pays the launch once for all its tokens; `None` where the instance's
    /// expert is not resident and the host runs it. `None` on the field when the offload has no
    /// block path.
    #[allow(clippy::type_complexity)]
    pub run_multi: Option<
        Box<
            dyn Fn(
                    &[usize],
                    &[&[f32]],
                    &(dyn Fn(usize) -> Result<Arc<Expert>> + Sync),
                    f32,
                ) -> Vec<Option<Result<Vec<f32>>>>
                + Send
                + Sync,
        >,
    >,
}

impl ExpertOffload {
    /// Run the `active` experts of a layer over the context `ctx` of the device layer, each on the
    /// lane its number names. An expert a lane ran answers `Some(Ok(..))` with its output rows (the
    /// rows entering its down projection are not read back), and one a lane declined answers `None`
    /// for the host.
    #[allow(clippy::type_complexity)]
    pub fn run_all_dev(
        &self,
        hooks: &DeviceLayer,
        ctx: &Arc<dyn std::any::Any + Send + Sync>,
        active: &[usize],
        swiglu_limit: f32,
        fetch: &(dyn Fn(usize) -> Result<Arc<Expert>> + Sync),
        rows_of: &(dyn Fn(usize) -> Vec<u32> + Sync),
    ) -> Vec<Option<Result<(Vec<f32>, Vec<f32>)>>> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let mut done: Vec<Option<Result<(Vec<f32>, Vec<f32>)>>> =
            (0..active.len()).map(|_| None).collect();
        let next = AtomicUsize::new(0);
        let slots = std::sync::Mutex::new(&mut done);
        let lanes = self.lanes.max(1);
        std::thread::scope(|scope| {
            for _ in 0..lanes {
                let (next, slots) = (&next, &slots);
                scope.spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&e) = active.get(i) else { break };
                    let home = e % lanes;
                    let r = match fetch(e) {
                        Ok(expert) => (hooks.run)(home, ctx, &expert, &rows_of(e), swiglu_limit)
                            .map(|r| r.map(|out| (Vec::new(), out))),
                        Err(err) => Some(Err(err)),
                    };
                    slots.lock().unwrap()[i] = r;
                });
            }
        });
        done
    }

    /// Run the `active` experts across the lanes, each on the lane its number names, and
    /// answer in the order asked, `None` where a lane declined and the caller runs the expert
    /// itself. `fetch` builds an expert by number and `rows_of` gives its input rows;
    /// `tokens` is the batch's size. Answers with the lanes' own time, fetching and running,
    /// in nanoseconds.
    #[allow(clippy::type_complexity)]
    pub fn run_all(
        &self,
        active: &[usize],
        tokens: usize,
        swiglu_limit: f32,
        fetch: &(dyn Fn(usize) -> Result<Arc<Expert>> + Sync),
        rows_of: &(dyn Fn(usize) -> Vec<f32> + Sync),
    ) -> (Vec<Option<Result<(Vec<f32>, Vec<f32>)>>>, u64, u64) {
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
        let mut done: Vec<Option<Result<(Vec<f32>, Vec<f32>)>>> =
            (0..active.len()).map(|_| None).collect();
        let next = AtomicUsize::new(0);
        let (fetch_ns, run_ns) = (AtomicU64::new(0), AtomicU64::new(0));
        let slots = std::sync::Mutex::new(&mut done);
        let lanes = self.lanes.max(1);
        std::thread::scope(|scope| {
            for _ in 0..lanes {
                let (next, slots, fetch_ns, run_ns) = (&next, &slots, &fetch_ns, &run_ns);
                scope.spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&e) = active.get(i) else { break };
                    // Which lane runs an expert is the expert's own number, not whichever lane
                    // reached it first: a lane that keeps weights keeps the same ones from one
                    // token to the next only if the same experts come back to it.
                    let home = e % lanes;
                    let fetch_started = std::time::Instant::now();
                    let fetched = fetch(e);
                    fetch_ns
                        .fetch_add(fetch_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let run_started = std::time::Instant::now();
                    let r = fetched
                        .map(|expert| (self.run)(home, tokens, &expert, &rows_of(e), swiglu_limit));
                    run_ns.fetch_add(run_started.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    let r = match r {
                        Ok(Some(r)) => Some(r),
                        Ok(None) => None,
                        Err(err) => Some(Err(err)),
                    };
                    slots.lock().unwrap()[i] = r;
                });
            }
        });
        (
            done,
            fetch_ns.load(Ordering::Relaxed),
            run_ns.load(Ordering::Relaxed),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `exp` is within two units in the last place of the true value over the whole range it does
    /// not saturate in, and meets the limits as `f32::exp` does.
    #[test]
    fn exp_is_within_two_units_in_the_last_place() {
        let ulps = |a: f32, b: f32| (a.to_bits() as i64 - b.to_bits() as i64).abs();
        let mut x = -103.9f32;
        while x < 88.7 {
            let want = (x as f64).exp() as f32;
            assert!(
                ulps(exp(x), want) <= 2,
                "exp({x}) = {} against {want}",
                exp(x)
            );
            x += 0.0137;
        }
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(100.0), f32::INFINITY);
        assert_eq!(exp(f32::INFINITY), f32::INFINITY);
        assert_eq!(exp(f32::NEG_INFINITY), 0.0);
        assert!(exp(f32::NAN).is_nan());
    }
}
