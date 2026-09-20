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
use std::sync::Arc;

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
    match current() {
        Some(offload) => {
            let t = std::time::Instant::now();
            let r = f();
            offload.record(stage, t.elapsed().as_nanos() as u64);
            r
        }
        None => f(),
    }
}
