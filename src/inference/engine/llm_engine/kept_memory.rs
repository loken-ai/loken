//! What a loaded language model keeps on its cards beyond what serves the request under way,
//! made known to the other components on those cards.
//!
//! The model is listed in `place::reclaim` as a holder: another component short of room on one
//! of its cards has it give back its batch worker's pool, its recorded graphs, its conversation
//! states or its extra expert slots, the cheapest first.
//! The growth its quantised caches were planned for and have not taken is declared on each
//! card, so no other placement takes it.
//! A request declares the memory its prompt reads in while it runs.
//! On a card another model is resident on, or is loading onto, that memory stays declared
//! between requests too.

use super::load::LoadedModelState;
use super::LlmEngine;
use crate::inference::place::reclaim::{Holder, Kind};
use crate::inference::place::vram_manager::DemandLease;
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
    /// The growth of the quantised caches still to come, declared on each card.
    growth: std::sync::Mutex<Vec<(usize, DemandLease)>>,
    /// The context the load planned the caches at, in positions; 0 when it planned none.
    planned: Arc<AtomicUsize>,
    /// The memory a prompt read takes on each card, as the last request declared it.
    work: std::sync::Mutex<Vec<(usize, u64)>>,
    /// The requests that declared their work and still run.
    working: AtomicUsize,
    /// The model's cards another owner is resident on or loading onto.
    shared: std::sync::Mutex<Vec<usize>>,
    /// The work declared on the shared cards while no request declares its own.
    kept_work: std::sync::Mutex<Vec<(usize, DemandLease)>>,
}

impl ModelHolder {
    fn new(
        owner: String,
        cards: Vec<usize>,
        state: Weak<State>,
        active: Weak<AtomicUsize>,
        #[cfg(feature = "gpu")] kept: Option<
            crate::inference::generic_transformer::hetero::KeptPool,
        >,
        planned: Arc<AtomicUsize>,
        work: Vec<(usize, u64)>,
    ) -> Self {
        Self {
            owner,
            cards,
            state,
            active,
            #[cfg(feature = "gpu")]
            kept,
            growth: Default::default(),
            planned,
            work: std::sync::Mutex::new(work),
            working: AtomicUsize::new(0),
            shared: Default::default(),
            kept_work: Default::default(),
        }
    }

    /// Declares the prompt work on each shared card while no request runs, and nothing on
    /// the other cards.
    fn keep_work(&self) {
        let mut kept = self.kept_work.lock().unwrap_or_else(|e| e.into_inner());
        let idle = self.working.load(Ordering::Acquire) == 0;
        let shared = self
            .shared
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let work = self.work.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let keep: Vec<(usize, u64)> = work
            .into_iter()
            .filter(|(card, _)| idle && shared.contains(card))
            .collect();
        kept.retain(|(card, _)| keep.iter().any(|(c, _)| c == card));
        for (card, bytes) in keep {
            match kept.iter().find(|(c, _)| *c == card) {
                Some((_, lease)) => lease.set(bytes),
                None => kept.push((
                    card,
                    crate::inference::place::room::hold_on(&self.owner, card, bytes),
                )),
            }
        }
    }

    /// Declares `work`, the memory a prompt read takes on each card, until the returned guard
    /// drops. The work kept between requests gives way to it meanwhile.
    pub(super) fn begin_work(self: &Arc<Self>, work: Vec<(usize, u64)>) -> RequestWork {
        let leases = work
            .iter()
            .map(|&(card, bytes)| crate::inference::place::room::hold_on(&self.owner, card, bytes))
            .collect();
        *self.work.lock().unwrap_or_else(|e| e.into_inner()) = work;
        self.working.fetch_add(1, Ordering::AcqRel);
        self.keep_work();
        RequestWork {
            _leases: leases,
            holder: Arc::downgrade(self),
        }
    }

    /// Declares what the model's caches still grow by on each card, from the model as it is now.
    /// Nothing changes while a request holds the model.
    fn declare_growth(&self) {
        let Some(state) = self.state.upgrade() else {
            return;
        };
        let Ok(mut guard) = state.try_lock() else {
            return;
        };
        let Some(model) = guard.as_mut().and_then(|s| s.model.generic_mut()) else {
            return;
        };
        let growth = model.kv_growth_by_card(self.planned.load(Ordering::Relaxed));
        drop(guard);
        let mut leases = self.growth.lock().unwrap_or_else(|e| e.into_inner());
        for (card, bytes) in growth {
            match leases.iter().find(|(c, _)| *c == card) {
                Some((_, lease)) => lease.set(bytes),
                None => leases.push((
                    card,
                    crate::inference::place::room::hold_on(&self.owner, card, bytes),
                )),
            }
        }
    }

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

    fn refresh(&self, shared: &[usize]) {
        self.declare_growth();
        *self.shared.lock().unwrap_or_else(|e| e.into_inner()) = shared.to_vec();
        self.keep_work();
    }

    /// The conversation states are counted, shared evenly by the model's cards.
    /// The pools, the graphs and the expert slots are not counted: their bytes are known only
    /// once dropped.
    fn keeps(&self, kind: Kind, card: usize) -> u64 {
        if kind != Kind::Snapshots || !self.cards.contains(&card) {
            return 0;
        }
        self.when_free(|s| Some(s.model.generic_mut()?.kv_snapshot_bytes()))
            .map_or(0, |bytes| bytes / self.cards.len() as u64)
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

/// The memory a request's prompt reads take, declared until it drops.
pub(super) struct RequestWork {
    _leases: Vec<DemandLease>,
    holder: Weak<ModelHolder>,
}

impl Drop for RequestWork {
    fn drop(&mut self) {
        // The kept work comes back before the request's own leases drop with the fields.
        if let Some(holder) = self.holder.upgrade() {
            holder.working.fetch_sub(1, Ordering::AcqRel);
            holder.keep_work();
        }
    }
}

/// The memory the prompt reads of `state` take on each of its cards.
fn prompt_work(state: &LoadedModelState) -> Vec<(usize, u64)> {
    let shape = state.model.prefill_shape();
    let chunk = crate::inference::engine::model_backend::adaptive_prefill_chunk(shape);
    let bytes = (chunk * shape.bytes_per_token + shape.fixed_bytes) as u64;
    #[cfg(feature = "gpu")]
    let cards = state.model.cuda_device_ordinals();
    #[cfg(not(feature = "gpu"))]
    let cards = std::collections::HashSet::<usize>::new();
    let mut work: Vec<(usize, u64)> = cards
        .into_iter()
        .filter(|card| shape.cards == 0 || shape.cards >> card & 1 == 1)
        .map(|card| (card, bytes))
        .collect();
    work.sort_unstable();
    work
}

impl LlmEngine {
    /// Lists the loaded model as a holder of kept memory, and declares the growth its caches
    /// were planned for. Replaces what an earlier load of this engine listed.
    pub(super) async fn list_kept_memory(&self) {
        let (cards, kept, work) = {
            let mut guard = self.model_state.lock().await;
            let Some(state) = guard.as_mut() else {
                return;
            };
            #[cfg(feature = "gpu")]
            let mut cards: Vec<usize> = state.model.cuda_device_ordinals().into_iter().collect();
            #[cfg(not(feature = "gpu"))]
            let mut cards: Vec<usize> = Vec::new();
            cards.sort_unstable();
            #[cfg(feature = "gpu")]
            let kept = state.model.generic_mut().map(|m| m.kept_pool());
            #[cfg(not(feature = "gpu"))]
            let kept: Option<()> = None;
            (cards, kept, prompt_work(state))
        };
        let holder = Arc::new(ModelHolder::new(
            self.config.model_id.clone(),
            cards,
            Arc::downgrade(&self.model_state),
            Arc::downgrade(&self.active),
            #[cfg(feature = "gpu")]
            kept,
            self.planned_kv.clone(),
            work,
        ));
        #[cfg(not(feature = "gpu"))]
        let _ = kept;
        holder.declare_growth();
        let listed: Arc<dyn Holder> = holder.clone();
        *self.kept_memory.lock().unwrap_or_else(|e| e.into_inner()) = Some(holder);
        crate::inference::place::reclaim::register(&listed);
    }

    /// Withdraws the model from the holders of kept memory, and its declarations with it.
    pub(super) fn unlist_kept_memory(&self) {
        let listed = self
            .kept_memory
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if listed.is_some() {
            drop(listed);
            crate::inference::place::reclaim::refresh();
        }
    }

    /// The cards of the loaded model, `None` while no model is listed.
    pub(super) fn listed_cards(&self) -> Option<Vec<usize>> {
        self.kept_memory
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|h| h.cards.clone())
    }

    /// Declares, for as long as the returned guard lives, the memory the loaded model's prompt
    /// reads take on each of its cards: what another placement leaves free while a request
    /// runs.
    pub(super) async fn declare_request_work(&self) -> RequestWork {
        let work = {
            let guard = self.model_state.lock().await;
            guard.as_ref().map(prompt_work).unwrap_or_default()
        };
        let holder = self
            .kept_memory
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        match holder {
            Some(holder) => holder.begin_work(work),
            None => RequestWork {
                _leases: work
                    .into_iter()
                    .map(|(card, bytes)| {
                        crate::inference::place::room::hold_on(&self.config.model_id, card, bytes)
                    })
                    .collect(),
                holder: Weak::new(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::place::reclaim;
    use crate::inference::place::vram_manager::card_demand_excluding;

    /// Cards no machine has: the holders of other tests never run on them.
    const CARD: usize = usize::MAX - 21;
    const ALONE: usize = usize::MAX - 22;
    const SAME: usize = usize::MAX - 23;

    /// A model listed on `cards`, whose prompt reads take `work` on each.
    fn listed(owner: &str, cards: &[usize], work: u64) -> Arc<ModelHolder> {
        let holder = Arc::new(ModelHolder::new(
            owner.to_string(),
            cards.to_vec(),
            Weak::new(),
            Weak::new(),
            #[cfg(feature = "gpu")]
            None,
            Arc::default(),
            cards.iter().map(|&c| (c, work)).collect(),
        ));
        let as_holder: Arc<dyn Holder> = holder.clone();
        reclaim::register(&as_holder);
        holder
    }

    /// An idle model declares its prompt work on a card while another model loads onto it or
    /// is resident on it, and stops when the other leaves.
    /// A request's own work replaces the kept work, so it is never counted twice.
    #[test]
    fn an_idle_model_keeps_its_work_while_another_is_on_its_card() {
        let (mine, theirs) = ("kept-work-test-mine", "kept-work-test-theirs");
        let work = 3 << 20;
        let model = listed(mine, &[CARD, ALONE], work);
        assert_eq!(card_demand_excluding(theirs, CARD), 0, "alone, as before");
        let arrival = reclaim::arrive_on(theirs, vec![CARD]);
        assert_eq!(card_demand_excluding(theirs, CARD), work);
        assert_eq!(
            card_demand_excluding(theirs, ALONE),
            0,
            "a card the other does not reach"
        );
        let request = model.begin_work(vec![(CARD, work), (ALONE, work)]);
        assert_eq!(card_demand_excluding(theirs, CARD), work);
        assert_eq!(card_demand_excluding(theirs, ALONE), work);
        drop(request);
        assert_eq!(card_demand_excluding(theirs, CARD), work);
        assert_eq!(card_demand_excluding(theirs, ALONE), 0);
        drop(arrival);
        assert_eq!(card_demand_excluding(theirs, CARD), 0, "the other left");
        let resident = arrive_resident(theirs);
        assert_eq!(card_demand_excluding(theirs, CARD), work);
        drop(resident);
        reclaim::refresh();
        assert_eq!(card_demand_excluding(theirs, CARD), 0);
        drop(model);
    }

    /// Two copies of one model never keep their work for each other.
    #[test]
    fn a_model_does_not_keep_its_work_for_its_own_owner() {
        let owner = "kept-work-test-same";
        let model = listed(owner, &[SAME], 5 << 20);
        let arrival = reclaim::arrive_on(owner, vec![SAME]);
        assert_eq!(card_demand_excluding("kept-work-test-reader", SAME), 0);
        drop(arrival);
        drop(model);
    }

    /// Another model listed on [`CARD`], as a load lists it once placed.
    fn arrive_resident(owner: &str) -> Arc<ModelHolder> {
        listed(owner, &[CARD], 0)
    }
}
