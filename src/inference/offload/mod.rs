//! Where the heavy steps of a forward run when not on the CPU: the engine's streamed placement.
//!
//! A model larger than the cards and the host together keeps its weights in the mapping and
//! runs on the host; the steps worth a crossing go to a card. A forward asks the offload of its
//! thread, if one is set, to run a projection's product or a sparse attention; the offload runs
//! it elsewhere or declines, and the CPU then runs it as always. The setting is scoped to the
//! thread and the call, so a model served elsewhere in the process keeps its own path.
//!
//! `projection` and `experts` are what any model hands over; `store` keeps a working set of
//! routed experts in the page cache; `room` is what a card may hold beside its own work;
//! `cuda` is the card behind the trait; `streamed` opens the placement from what a model
//! declares, under the engine's own switches.

#[cfg(feature = "cuda")]
pub mod cuda;
pub mod experts;
pub mod projection;
pub mod room;
pub mod store;
pub mod streamed;

use crate::tensor::Result;
use projection::Projection;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// A profile of the named stages a forward runs through, kept whole for a model that offloads its
/// steps (its lanes discard the per-thread recorder). Off by default: a lock on every stage of
/// every layer is a cost only a diagnostic wants, and what makes the shares readable is that they
/// are summed across the lanes rather than lost with the thread that ran them.
static STAGE_PROF: Mutex<BTreeMap<&'static str, (u64, u64)>> = Mutex::new(BTreeMap::new());
static STAGE_PROF_ON: AtomicBool = AtomicBool::new(false);

/// Switch the stage profile on (zeroing it) or off.
pub fn stage_prof_enable(on: bool) {
    if on {
        STAGE_PROF.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }
    STAGE_PROF_ON.store(on, Ordering::Relaxed);
}

pub fn stage_prof_enabled() -> bool {
    STAGE_PROF_ON.load(Ordering::Relaxed)
}

/// Add `ns` to a stage the offload timed itself (its lanes' fetch and kernels, the host experts):
/// these are summed on the caller after the lanes join rather than wrapped by `stage`, so they
/// reach the profile through here. A no-op when the profile is off.
pub fn prof_record(stage: &'static str, ns: u64) {
    if STAGE_PROF_ON.load(Ordering::Relaxed) {
        let mut m = STAGE_PROF.lock().unwrap_or_else(|e| e.into_inner());
        let e = m.entry(stage).or_default();
        e.0 += ns;
        e.1 += 1;
    }
}

/// Each stage's total nanoseconds and call count since it was switched on.
pub fn stage_prof_snapshot() -> Vec<(&'static str, u64, u64)> {
    STAGE_PROF
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(k, v)| (*k, v.0, v.1))
        .collect()
}

/// One ratio-0 attention block's weights and this token's inputs, for a card to run whole. The
/// weights are the layer's kept projections and dense norms; `x` is the layer input row [dim],
/// `cos`/`sin` the rope tables' row for this position [rd/2], `window` the last keys [w, hd] the
/// window held before this token, `sink` the per-head attention sink [h].
pub struct AttnBlock<'a> {
    pub wq_a: &'a Projection,
    pub q_norm: &'a crate::tensor::Tensor,
    pub wq_b: &'a Projection,
    pub wkv: &'a Projection,
    pub kv_norm: &'a crate::tensor::Tensor,
    pub wo_a: &'a Projection,
    pub wo_b: &'a Projection,
    pub sink: &'a crate::tensor::Tensor,
    pub x: &'a [f32],
    pub cos: &'a [f32],
    pub sin: &'a [f32],
    pub window: &'a [f32],
    pub n_heads: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub o_groups: usize,
    pub o_lora: usize,
    pub dim: usize,
    pub eps: f32,
    pub scale: f32,
}

pub trait Offload: Send + Sync {
    /// `p` applied to `xs` rows, row-major; `None` to run it here.
    fn projection(&self, p: &Projection, xs: &[f32]) -> Option<Result<Vec<f32>>>;

    /// Sparse attention with a per-head sink: `q` [s, h, d], `kv` [n, d], `idxs` [s, topk] into
    /// `kv` (negative: empty). `None` to run it here.
    #[allow(clippy::too_many_arguments)]
    fn sparse_attention(
        &self,
        q: &[f32],
        kv: &[f32],
        sink: &[f32],
        idxs: &[i32],
        dims: (usize, usize, usize, usize),
        scale: f32,
    ) -> Option<Result<Vec<f32>>>;

    /// An indexer's scores [s, g]: `q` [s, nh, ihd], `k` [g, ihd], `weights` [s, nh]; the sum over
    /// heads of relu(q . k) * weight * scale where query `i` reaches, `(i + 1) / ratio` positions,
    /// -inf past. `None` to run it here.
    fn index_scores(
        &self,
        _q: &[f32],
        _k: &[f32],
        _weights: &[f32],
        _dims: (usize, usize, usize, usize),
        _ratio: usize,
        _scale: f32,
    ) -> Option<Result<Vec<f32>>> {
        None
    }

    /// A ratio-0 attention block of a decode, run whole on one card so the activation crosses once
    /// for the layer instead of once per projection. Given the layer's kept weights and this token's
    /// input, it runs `wq_a`->q_norm->`wq_b`->rope, `wkv`->kv_norm->rope, windowed attention with a
    /// per-head sink over `window` plus this token's key, output rope and the grouped output
    /// projection - all on the card. Returns `(o, kv_row)`: the block output [dim] and this token's
    /// rope'd key row [hd] to append to the window for the next token. `None` to run it here.
    fn attn_block(&self, _p: &AttnBlock) -> Option<Result<(Vec<f32>, Vec<f32>)>> {
        None
    }

    /// A dense SwiGLU expert (the mixture's shared expert) run whole on one card so its three
    /// products and the SwiGLU stay on the device and the activation crosses the bus once. `x` is
    /// [rows, dim] row-major; `limit` is the SwiGLU clamp (the up branch clamped both sides, the
    /// gate branch from above); returns [rows, dim]. `None` to run it on the host.
    fn expert_dev(
        &self,
        _w1: &Projection,
        _w3: &Projection,
        _w2: &Projection,
        _x: &[f32],
        _limit: f32,
    ) -> Option<Result<Vec<f32>>> {
        None
    }

    /// The grouped output projection of a decode's attention, run on one card so the activation
    /// crosses once instead of once per group. `o` is [o_groups, p] row-major; group g is
    /// multiplied by rows `[g*o_lora, (g+1)*o_lora)` of `wo_a`, the group results concatenated and
    /// put through `wo_b` to [dim]. `None` to run it here, group by group, as a plain projection.
    #[allow(clippy::too_many_arguments)]
    fn attn_out(
        &self,
        _wo_a: &Projection,
        _wo_b: &Projection,
        _o: &[f32],
        _o_groups: usize,
        _p: usize,
        _o_lora: usize,
        _dim: usize,
    ) -> Option<Result<Vec<f32>>> {
        None
    }

    /// Told how long a stage of the forward took on this thread, for a profile. Nothing by default.
    fn record(&self, _stage: &'static str, _nanos: u64) {}
}

thread_local! {
    static CURRENT: std::cell::RefCell<Option<Arc<dyn Offload>>> = const { std::cell::RefCell::new(None) };
}

/// Run `body` with this thread's heavy steps handed to `offload`.
pub fn with_offload<R>(offload: Arc<dyn Offload>, body: impl FnOnce() -> R) -> R {
    let previous = CURRENT.with(|c| c.replace(Some(offload)));
    let result = body();
    CURRENT.with(|c| *c.borrow_mut() = previous);
    result
}

/// The offload set on this thread, if any.
pub fn current() -> Option<Arc<dyn Offload>> {
    CURRENT.with(|c| c.borrow().clone())
}

/// `x` [.., in] through a dense `w` [out, in], on this thread's offload when it takes the product.
pub fn linear(
    x: &crate::tensor::Tensor,
    w: &crate::tensor::Tensor,
) -> Result<crate::tensor::Tensor> {
    Projection::Dense(w.clone()).apply(x)
}

/// Run `f` as stage `stage`: timed and recorded when this thread has an offload, run plainly
/// otherwise.
pub fn stage<R>(stage: &'static str, f: impl FnOnce() -> R) -> R {
    let prof = STAGE_PROF_ON.load(Ordering::Relaxed);
    match current() {
        Some(offload) => {
            let t = std::time::Instant::now();
            let r = f();
            let ns = t.elapsed().as_nanos() as u64;
            offload.record(stage, ns);
            if prof {
                let mut m = STAGE_PROF.lock().unwrap_or_else(|e| e.into_inner());
                let e = m.entry(stage).or_default();
                e.0 += ns;
                e.1 += 1;
            }
            r
        }
        None if prof => {
            let t = std::time::Instant::now();
            let r = f();
            let ns = t.elapsed().as_nanos() as u64;
            let mut m = STAGE_PROF.lock().unwrap_or_else(|e| e.into_inner());
            let e = m.entry(stage).or_default();
            e.0 += ns;
            e.1 += 1;
            r
        }
        None => f(),
    }
}
