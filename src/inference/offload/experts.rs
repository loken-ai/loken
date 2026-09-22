//! One routed expert, and a set of lanes that run experts somewhere other than the CPU.

use super::projection::Projection;
use crate::tensor::{Device, Result, Tensor};
use std::sync::Arc;

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

    /// Start reading the three projections in.
    pub fn will_need(&self) {
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
                let mut g = gate[t][k];
                let mut u = up[t][k];
                if swiglu_limit > 0.0 {
                    u = u.clamp(-swiglu_limit, swiglu_limit);
                    g = g.min(swiglu_limit);
                }
                let silu = g / (1.0 + (-g).exp());
                h[t * inter + k] = silu * u;
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
pub struct ExpertOffload {
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
