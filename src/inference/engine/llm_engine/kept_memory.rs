//! What a loaded language model keeps on its cards beyond what serves the request under way,
//! made known to the other components on those cards.
//!
//! The model is listed in `place::reclaim` as a holder: another component short of room on one
//! of its cards has it give back its batch worker's pool, its recorded graphs, its conversation
//! states or its extra expert slots, the cheapest first.

use super::load::LoadedModelState;
use super::LlmEngine;
use crate::inference::place::reclaim::{Holder, Kind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

type State = tokio::sync::Mutex<Option<LoadedModelState>>;

/// A loaded model as the registry of kept memory sees it.
pub(super) struct ModelHolder {
    owner: String,
    cards: Vec<usize>,
    state: Weak<State>,
    /// The requests the engine runs now.
    active: Weak<AtomicUsize>,
    /// The slot the model keeps its stopped batch worker's pool in.
    #[cfg(feature = "gpu")]
    kept: Option<crate::inference::generic_transformer::hetero::KeptPool>,
}

impl ModelHolder {
    /// Runs `f` on the loaded state when no request holds it.
    fn when_free<T>(&self, f: impl FnOnce(&mut LoadedModelState) -> Option<T>) -> Option<T> {
        let state = self.state.upgrade()?;
        let mut guard = state.try_lock().ok()?;
        f(guard.as_mut()?)
    }

    /// Whether the engine runs no request.
    #[cfg(feature = "gpu")]
    fn idle(&self) -> bool {
        self.active
            .upgrade()
            .is_some_and(|a| a.load(Ordering::Acquire) == 0)
    }

    /// Drops the pool the model keeps for its next batch worker. True when it kept one.
    #[cfg(feature = "gpu")]
    fn drop_kept_pool(&self, card: usize) -> bool {
        let Some(kept) = &self.kept else {
            return false;
        };
        let pool = kept.lock().unwrap_or_else(|e| e.into_inner()).take();
        let Some(pool) = pool else {
            return false;
        };
        drop(pool);
        crate::inference::place::room::settle(card);
        true
    }
}

impl Holder for ModelHolder {
    fn owner(&self) -> &str {
        &self.owner
    }

    fn cards(&self) -> Vec<usize> {
        self.cards.clone()
    }

    fn give_back(&self, kind: Kind, card: usize, bytes: u64) -> bool {
        match kind {
            #[cfg(feature = "gpu")]
            Kind::ParkedPool => self.drop_kept_pool(card),
            // A worker serving no request stops; the model serves serially until its next
            // batch, which sizes a new pool over what its cards offer then.
            #[cfg(feature = "gpu")]
            Kind::IdlePool => {
                self.idle()
                    && self
                        .when_free(|s| {
                            (s.model.batched_server().is_some() && super::load::demote_to_serial(s))
                                .then_some(())
                        })
                        .is_some()
                    && self.drop_kept_pool(card)
            }
            // The next request records its graphs again.
            #[cfg(feature = "gpu")]
            Kind::KeptGraphs => {
                let dropped = self
                    .when_free(|s| {
                        if s.kept_graphs.is_empty() && s.draft_graph.is_empty() {
                            return None;
                        }
                        s.kept_graphs.clear();
                        s.draft_graph.clear();
                        s.model.note_graphs_kept(None);
                        s.model.invalidate_graph_state();
                        Some(())
                    })
                    .is_some();
                if dropped {
                    crate::inference::place::room::settle(card);
                }
                dropped
            }
            Kind::Snapshots => self
                .when_free(|s| {
                    let model = s.model.generic_mut()?;
                    (model.kv_snapshot_bytes() > 0).then(|| model.drop_kv_snapshots_for(bytes))
                })
                .is_some(),
            Kind::ExpertSlots => self
                .when_free(|s| {
                    s.model
                        .generic_mut()?
                        .lend_extra_slots_on(card, bytes)
                        .then_some(())
                })
                .is_some(),
            #[cfg(not(feature = "gpu"))]
            Kind::ParkedPool | Kind::IdlePool | Kind::KeptGraphs => false,
        }
    }
}

impl LlmEngine {
    /// Lists the loaded model as a holder of kept memory.
    /// Replaces what an earlier load of this engine listed.
    pub(super) async fn list_kept_memory(&self) {
        let (cards, kept) = {
            let mut guard = self.model_state.lock().await;
            let Some(state) = guard.as_mut() else {
                return;
            };
            let mut cards: Vec<usize> = state.model.cuda_device_ordinals().into_iter().collect();
            cards.sort_unstable();
            #[cfg(feature = "gpu")]
            let kept = state.model.generic_mut().map(|m| m.kept_pool());
            #[cfg(not(feature = "gpu"))]
            let kept: Option<()> = None;
            (cards, kept)
        };
        let holder = Arc::new(ModelHolder {
            owner: self.config.model_id.clone(),
            cards,
            state: Arc::downgrade(&self.model_state),
            active: Arc::downgrade(&self.active),
            #[cfg(feature = "gpu")]
            kept,
        });
        #[cfg(not(feature = "gpu"))]
        let _ = kept;
        let listed: Arc<dyn Holder> = holder;
        crate::inference::place::reclaim::register(&listed);
        *self.kept_memory.lock().unwrap_or_else(|e| e.into_inner()) = Some(listed);
    }

    /// Withdraws the model from the holders of kept memory.
    pub(super) fn unlist_kept_memory(&self) {
        self.kept_memory
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
    }
}
