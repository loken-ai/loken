//! The routing calibration a load leaves, run between requests.
//!
//! A thread takes the model lock for 1 run of the calibration at a time, while the server
//! handles no request and no request runs.
//! A request that arrives during a run waits for that run at most.
//! The decoded run of the noise gives way at the next token instead.
//! After each run the resident KV and what the engine records of it are dropped.

use super::load::LoadedModelState;
use super::LlmEngine;
use crate::inference::generic_transformer::routing_calibration::{self as calibration, Step};
use crate::inference::generic_transformer::GenericHeteroTransformer;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use tracing::warn;

impl LlmEngine {
    /// Starts the thread that runs the loaded model's routing calibration between requests,
    /// when its load left one.
    pub(super) async fn calibrate_between_requests(&self) {
        let id = {
            let mut guard = self.model_state.lock().await;
            guard
                .as_mut()
                .and_then(|s| s.model.generic_mut())
                .and_then(|m| m.routing_pending().map(|p| p.id()))
        };
        let Some(id) = id else { return };
        // The thread keeps the engine's model only for the length of a step.
        let state = Arc::downgrade(&self.model_state);
        let sessions = Arc::downgrade(&self.sessions);
        let active = Arc::clone(&self.active);
        let spawned = std::thread::Builder::new()
            .name("routing-calibration".to_string())
            .spawn(move || {
                between_requests(id, &active, || {
                    let (state, sessions) = (state.upgrade()?, sessions.upgrade()?);
                    let mut guard = state.blocking_lock();
                    if super::batching::busy(&active) {
                        return Some(Step::Yielded);
                    }
                    let stepped = run_step(guard.as_mut()?, id, &|| super::batching::busy(&active));
                    sessions.blocking_lock().clear();
                    stepped
                })
            });
        if let Err(e) = spawned {
            warn!("routing calibration thread: {e}; routing stays exact");
        }
    }

    /// Runs what is left of the loaded model's routing calibration, holding the model until
    /// it ends.
    pub async fn finish_routing_calibration(&self) {
        let mut guard = self.model_state.lock().await;
        let Some(state) = guard.as_mut() else { return };
        let id = state
            .model
            .generic_mut()
            .and_then(|m| m.routing_pending().map(|p| p.id()));
        let Some(id) = id else { return };
        while let Some(Step::Measured) = run_step(state, id, &|| false) {}
        self.sessions.lock().await.clear();
    }
}

/// Calls `step` each time no request counted in `active` runs, until it returns `None` or
/// [`Step::Done`].
///
/// `id` names the calibration in the log.
fn between_requests(id: u64, active: &AtomicUsize, mut step: impl FnMut() -> Option<Step>) {
    loop {
        super::batching::wait_idle(active);
        match step() {
            Some(Step::Measured | Step::Yielded) => {}
            Some(Step::Done(_)) | None => {
                tracing::debug!("routing calibration {id} ended");
                return;
            }
        }
    }
}

/// Runs the next run of the calibration `id` on the loaded model.
///
/// The resident KV and the graphs kept are dropped before the run and after it.
///
/// `None` when the model no longer holds that calibration or the run failed.
fn run_step(state: &mut LoadedModelState, id: u64, yielded: &dyn Fn() -> bool) -> Option<Step> {
    if state.model.batched_server().is_some() && !super::load::demote_to_serial(state) {
        return Some(Step::Yielded);
    }
    let pending = state
        .model
        .generic_mut()?
        .routing_pending()
        .filter(|p| p.id() == id)?
        .clone();
    // The graphs kept would replay over the single-token forwards of the noise.
    forget_resident(state);
    // A prefill of the text stores no state under the boundary of the last request.
    state.model.set_prefix_boundary(None);
    let model = state.model.generic_mut()?;
    let mut other = |m: &mut GenericHeteroTransformer, ids: &[u32]| {
        calibration::decode_run_yielding(m, ids, yielded)
    };
    let dir = calibration::cache_dir();
    let started = std::time::Instant::now();
    let stepped = pending.step(
        model,
        dir.as_deref(),
        &mut calibration::prefill_run,
        &mut other,
        yielded,
    );
    let step = match stepped {
        Ok(step) => {
            tracing::info!(
                "quality calibration: {step:?} between requests in {:.1}s",
                started.elapsed().as_secs_f64()
            );
            Some(step)
        }
        Err(e) => {
            warn!("quality calibration failed, routing stays exact: {e}");
            None
        }
    };
    if !matches!(step, Some(Step::Measured | Step::Yielded)) {
        model.set_routing_pending(None);
    }
    forget_resident(state);
    step
}

/// Drops what the model and the engine keep of the resident KV: the KV, the logits of the
/// last prefill and the graphs kept between requests.
fn forget_resident(state: &mut LoadedModelState) {
    #[cfg(feature = "gpu")]
    state.kept_graphs.clear();
    state.model.note_graphs_kept(None);
    state.model.invalidate_graph_state();
    state.exact_prefill = None;
    state.model.reset_kv_from(0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::engine::llm_engine::batching::ActiveRequest;
    use std::sync::atomic::Ordering;

    /// A request admitted while the calibration runs waits for 1 step at most, and no step
    /// starts while a request runs.
    #[test]
    fn a_request_waits_for_one_step_at_most() {
        let active = Arc::new(AtomicUsize::new(0));
        let model = Arc::new(tokio::sync::Mutex::new(0usize));
        let (to_request, at_request) = std::sync::mpsc::channel::<()>();
        let (to_steps, at_steps) = std::sync::mpsc::channel::<()>();
        let request = {
            let (active, model) = (Arc::clone(&active), Arc::clone(&model));
            std::thread::spawn(move || {
                at_request.recv().unwrap();
                let admitted = ActiveRequest::enter(&active);
                to_steps.send(()).unwrap();
                let started = *model.blocking_lock();
                drop(admitted);
                started
            })
        };
        let steps = {
            let (active, model) = (Arc::clone(&active), Arc::clone(&model));
            std::thread::spawn(move || {
                let mut under_request = 0;
                between_requests(0, &active, || {
                    let mut guard = model.blocking_lock();
                    if active.load(Ordering::Acquire) > 0 {
                        under_request += 1;
                        return Some(Step::Yielded);
                    }
                    *guard += 1;
                    match *guard {
                        2 => {
                            to_request.send(()).unwrap();
                            at_steps.recv().unwrap();
                        }
                        6 => return Some(Step::Done(None)),
                        _ => {}
                    }
                    Some(Step::Measured)
                });
                under_request
            })
        };
        let seen = request.join().unwrap();
        let under_request = steps.join().unwrap();
        assert_eq!(seen, 2, "the request ran after the step it arrived in");
        assert!(
            under_request <= 1,
            "{under_request} steps passed while a request ran"
        );
        assert_eq!(*model.blocking_lock(), 6);
    }
}
