//! The routed experts of a wide-residual layer that a card keeps.
//!
//! A card keeps an expert's 3 weights in the blocks the file stores.
//! It runs them with the expert kernels that give the host's bits (`gpu_experts_exact`).
//! Only the formats those kernels read are kept: gate and up in `IQ2_XXS` with down in `Q2_K`,
//! or gate and up in `IQ3_XXS` with down in `Q3_K`.
//!
//! A layer's experts on 1 card run in 1 call.
//! The layer's rows go up once, every (expert, row) pair runs, and the outputs come back once.
//! A layer whose experts sit on several cards makes 1 call per card, each card on its own thread.
//! A card that keeps routed experts only takes the rows and gives back its outputs, nothing more.
//!
//! The card that runs the layer's token mixer may also keep 2 weights every token reads:
//! - the shared expert, in the blocks the file stores, its products measured per unit as the
//!   mixers' are (`card_packed.rs`),
//! - the router's F32 values, which a decoded row takes with the host's bits.
//!
//! In a forward of several rows, a card may also run experts it does not keep, in transit.
//! Their blocks go up from the host's memory, run over their rows with the same kernels, and
//! are dropped once their outputs are down.
//! Each card runs its kept experts first, then its transit, in batches that fit what it has
//! free at that moment.
//! The blocks of the transit go up on a stream of their own.
//! A card whose reads end before its kept experts, by the schedule, reads while they run.
//! An expert that does not fit, or whose batch fails, answers `None`: the host runs it.

use super::block_scaled_experts::BlockScaledExpert;
use crate::tensor::Result;

/// What a card's measure at the load gives a schedule of experts in transit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransitCost {
    /// Seconds to move 1 byte of blocks from the host's memory to the card.
    pub secs_per_byte: f64,
    /// Seconds of the card for 1 expert over 1 tile of rows: the measured run over 1 row, per
    /// expert.
    pub secs_per_tile: f64,
    /// Seconds of 1 host core for 1 expert over 1 row.
    pub host_secs_per_pair: f64,
}

impl TransitCost {
    /// The costs of `m`, `None` when it timed no upload or no product.
    pub fn of(m: &super::wide_sparse_card_tier::CardMeasure) -> Option<Self> {
        let per = m.experts as f64;
        // The host ran the experts at once, each on a core while there are cores.
        let cores = per.min(rayon::current_num_threads() as f64);
        let cost = Self {
            secs_per_byte: m.upload_secs / m.bytes as f64,
            secs_per_tile: m.card_secs / per,
            host_secs_per_pair: m.host_secs * cores / per,
        };
        let fine = |v: f64| v.is_finite() && v > 0.0;
        (fine(cost.secs_per_byte) && fine(cost.secs_per_tile) && fine(cost.host_secs_per_pair))
            .then_some(cost)
    }
}

/// Rows of 1 expert that the card kernels run over 1 read of its blocks.
pub fn tile_rows() -> usize {
    #[cfg(feature = "gpu")]
    {
        crate::tensor::cuda::EXACT_TILE
    }
    #[cfg(not(feature = "gpu"))]
    1
}

/// Bytes a card holds while it routes a prompt's rows `group` at a time: the rows of a group,
/// `inp` wide, and their `out` logits, in F32.
pub fn router_rows_bytes(inp: usize, out: usize, group: usize) -> usize {
    group * (inp + out) * std::mem::size_of::<f32>()
}

/// The seconds a card takes for its kept experts, `kept`, and for experts in transit whose reads
/// take `reads` and whose runs take `runs`.
///
/// The reads go on while the kept experts run when they end before them.
/// The kernels then bound the card, and the runs follow the kept experts.
/// Reads that outlast the kept experts start after them: the disk bounds the card.
pub fn card_finish(kept: f64, reads: f64, runs: f64) -> f64 {
    match reads <= kept {
        true => kept + runs,
        false => kept + reads + runs,
    }
}

/// How a schedule splits the experts no card keeps between the cards and the host's cores.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TransitSchedule {
    /// Per card that takes experts in transit, its ordinal and the experts, in the order taken.
    pub moving: Vec<(usize, Vec<usize>)>,
    /// The experts left to the host's cores.
    pub left: Vec<usize>,
    /// The ordinals of the cards that read their experts in transit while their kept experts
    /// run.
    pub beside_kept: Vec<usize>,
    /// Per card of the costs, its ordinal and the seconds the schedule expects it to take.
    pub expected: Vec<(usize, f64)>,
}

/// Splits the experts no card keeps between the cards in transit and the host's cores.
///
/// `candidates` are (expert, rows, bytes).
/// `busy[k]` is the seconds card `costs[k]` spends on its kept experts.
/// Each queue is a card or a host core.
/// With `beside`, a card finishes by [`card_finish`] over its kept experts and the reads and
/// runs of its transit.
/// Without it, a card finishes at its kept experts plus the reads and runs of each expert of its
/// transit, one after the other, and no card reads beside its kept experts.
/// A core's expert costs its rows times the host's seconds per pair: the least the cards
/// measured.
/// The experts go, the most rows first, each to the queue that finishes it soonest.
pub fn schedule_transit(
    costs: &[(usize, TransitCost)],
    busy: &[f64],
    candidates: &[(usize, usize, u64)],
    cores: usize,
    beside: bool,
) -> TransitSchedule {
    let mut order: Vec<&(usize, usize, u64)> = candidates.iter().collect();
    order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let host_pair = costs
        .iter()
        .map(|(_, c)| c.host_secs_per_pair)
        .fold(f64::INFINITY, f64::min);
    let kept = |k: usize| busy.get(k).copied().unwrap_or(0.0);
    // Per card: its ordinal, its seconds of reads, its seconds of runs, the experts it takes
    // and the time it finishes without reading beside its kept experts.
    let mut cards: Vec<(usize, f64, f64, Vec<usize>, f64)> = costs
        .iter()
        .enumerate()
        .map(|(k, (o, _))| (*o, 0.0, 0.0, Vec::new(), kept(k)))
        .collect();
    let mut host: Vec<f64> = vec![0.0; cores.max(1)];
    let mut left = Vec::new();
    for &&(e, rows, bytes) in &order {
        let core = (0..host.len())
            .min_by(|&a, &b| host[a].total_cmp(&host[b]))
            .unwrap_or(0);
        let on_core = host[core] + rows as f64 * host_pair;
        let card = (0..cards.len())
            .map(|k| {
                let c = &costs[k].1;
                let read = bytes as f64 * c.secs_per_byte;
                let run = rows.div_ceil(tile_rows()) as f64 * c.secs_per_tile;
                let (reads, runs) = (cards[k].1 + read, cards[k].2 + run);
                let after = cards[k].4 + (read + run);
                let done = match beside {
                    true => card_finish(kept(k), reads, runs),
                    false => after,
                };
                (k, reads, runs, after, done)
            })
            .min_by(|a, b| a.4.total_cmp(&b.4));
        match card {
            Some((k, reads, runs, after, done)) if done < on_core => {
                cards[k].1 = reads;
                cards[k].2 = runs;
                cards[k].3.push(e);
                cards[k].4 = after;
            }
            _ => {
                host[core] = on_core;
                left.push(e);
            }
        }
    }
    let expected = cards
        .iter()
        .enumerate()
        .map(|(k, c)| match beside {
            true => (c.0, card_finish(kept(k), c.1, c.2)),
            false => (c.0, c.4),
        })
        .collect();
    let beside_kept = cards
        .iter()
        .enumerate()
        .filter(|(k, c)| beside && !c.3.is_empty() && c.1 <= kept(*k))
        .map(|(_, c)| c.0)
        .collect();
    let moving = cards
        .into_iter()
        .filter(|c| !c.3.is_empty())
        .map(|(o, _, _, ids, _)| (o, ids))
        .collect();
    TransitSchedule {
        moving,
        left,
        beside_kept,
        expected,
    }
}

/// The routed experts of 1 layer that the cards keep, by expert id, and the layer's shared
/// expert and router when a card keeps them.
///
/// Empty on the host and in a build without cards.
#[derive(Default)]
pub struct CardExperts {
    #[cfg(feature = "gpu")]
    sets: Vec<gpu::CardSet>,
    #[cfg(feature = "gpu")]
    shared: Option<gpu::SharedCard>,
    #[cfg(feature = "gpu")]
    router: Option<gpu::RouterCard>,
    /// The ids of the experts the cards keep, in the order they were kept: the hottest first.
    #[cfg(feature = "gpu")]
    order: Vec<usize>,
    /// (expert, row) pairs the cards ran since the load.
    ran: std::sync::atomic::AtomicU64,
    /// Per card whose run failed since the last take, the most bytes a failed run asked for.
    #[cfg(feature = "gpu")]
    short: std::sync::Mutex<Vec<(usize, u64)>>,
    /// The cards that may run experts they do not keep, shared by the layers.
    #[cfg(feature = "gpu")]
    transit: Vec<std::sync::Arc<gpu::Transit>>,
    /// (expert, row) pairs the cards ran in transit since the load.
    moved: std::sync::atomic::AtomicU64,
}

impl CardExperts {
    /// Whether no card keeps an expert of the layer.
    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Experts the cards keep.
    pub fn count(&self) -> usize {
        #[cfg(feature = "gpu")]
        {
            self.sets.iter().map(|s| s.experts.len()).sum()
        }
        #[cfg(not(feature = "gpu"))]
        0
    }

    /// Whether a card keeps expert `e`.
    pub fn holds(&self, e: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.sets.iter().any(|s| s.experts.contains_key(&e))
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = e;
            false
        }
    }

    /// The ordinal of the card that keeps expert `e`, `None` when no card keeps it.
    pub fn card_of(&self, e: usize) -> Option<usize> {
        #[cfg(feature = "gpu")]
        {
            self.sets
                .iter()
                .find(|s| s.experts.contains_key(&e))
                .map(|s| s.ordinal)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = e;
            None
        }
    }

    /// Per card whose run of routed experts failed since the last call, the most bytes a failed
    /// run asked for.
    ///
    /// The notes are emptied.
    pub fn take_short(&self) -> Vec<(usize, u64)> {
        #[cfg(feature = "gpu")]
        {
            std::mem::take(&mut *self.short.lock().unwrap_or_else(|e| e.into_inner()))
        }
        #[cfg(not(feature = "gpu"))]
        Vec::new()
    }

    /// Experts card `ordinal` keeps.
    pub fn count_on(&self, ordinal: usize) -> usize {
        #[cfg(feature = "gpu")]
        {
            self.sets
                .iter()
                .filter(|s| s.ordinal == ordinal)
                .map(|s| s.experts.len())
                .sum()
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            0
        }
    }

    /// Lets card `ordinal` drop the last expert it kept, the coldest: the bytes it freed, `None`
    /// when the card keeps none.
    ///
    /// The host runs the expert from then on, with the same bits.
    pub fn release_coldest(&mut self, ordinal: usize) -> Option<u64> {
        #[cfg(feature = "gpu")]
        {
            let set = self.sets.iter_mut().find(|s| s.ordinal == ordinal)?;
            let at = self
                .order
                .iter()
                .rposition(|e| set.experts.contains_key(e))?;
            let e = self.order.remove(at);
            let freed = set.release(e);
            self.sets.retain(|s| !s.experts.is_empty());
            freed
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            None
        }
    }

    /// The expert card `ordinal` would drop first, the last it kept: `None` when it keeps none.
    pub fn coldest_on(&self, ordinal: usize) -> Option<usize> {
        #[cfg(feature = "gpu")]
        {
            let set = self.sets.iter().find(|s| s.ordinal == ordinal)?;
            self.order
                .iter()
                .rev()
                .find(|e| set.experts.contains_key(e))
                .copied()
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            None
        }
    }

    /// Whether card `ordinal` keeps the layer's shared expert.
    pub fn keeps_shared_on(&self, ordinal: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.shared.as_ref().is_some_and(|s| s.ordinal() == ordinal)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            false
        }
    }

    /// Whether card `ordinal` keeps the layer's router.
    pub fn keeps_router_on(&self, ordinal: usize) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.router.as_ref().is_some_and(|r| r.ordinal() == ordinal)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = ordinal;
            false
        }
    }

    /// Drops the shared expert from the card: the bytes it held, `None` when no card keeps it.
    ///
    /// The host runs it from then on.
    pub fn release_shared(&mut self) -> Option<u64> {
        #[cfg(feature = "gpu")]
        {
            self.shared.take().map(|s| s.bytes() as u64)
        }
        #[cfg(not(feature = "gpu"))]
        None
    }

    /// Drops the router from the card: the bytes it held, `None` when no card keeps it.
    ///
    /// The host runs it from then on.
    pub fn release_router(&mut self) -> Option<u64> {
        #[cfg(feature = "gpu")]
        {
            self.router.take().map(|r| r.bytes() as u64)
        }
        #[cfg(not(feature = "gpu"))]
        None
    }

    /// (expert, row) pairs the cards ran since the load.
    pub fn pairs_run(&self) -> u64 {
        self.ran.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// (expert, row) pairs the cards ran in transit since the load.
    pub fn pairs_moved(&self) -> u64 {
        self.moved.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Lets `cards` run, in a forward of several rows, experts they do not keep.
    #[cfg(feature = "gpu")]
    pub(super) fn set_transit(&mut self, cards: Vec<std::sync::Arc<gpu::Transit>>) {
        self.transit = cards;
    }

    /// Notes, on every card that may run experts it does not keep, `ns` nanoseconds of host
    /// cores spent on `pairs` (expert, row) pairs in a forward, of a prompt when `prompt`.
    pub fn note_host(&self, ns: u64, pairs: u64, prompt: bool) {
        #[cfg(feature = "gpu")]
        self.transit
            .iter()
            .for_each(|t| t.note_host(ns, pairs, prompt));
        #[cfg(not(feature = "gpu"))]
        let _ = (ns, pairs, prompt);
    }

    /// Whether every card that may run experts it does not keep reads them from their file
    /// itself, past the page cache.
    pub fn transit_reads_direct(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            !self.transit.is_empty() && self.transit.iter().all(|t| t.reads_direct())
        }
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// Bytes of experts in transit the cards read past the page cache since the load.
    pub fn transit_bytes_read_direct(&self) -> u64 {
        #[cfg(feature = "gpu")]
        {
            self.transit.iter().map(|t| t.bytes_read_direct()).sum()
        }
        #[cfg(not(feature = "gpu"))]
        0
    }

    /// Per card that may run experts it does not keep, its ordinal and its costs now, for a
    /// forward of a prompt when `prompt`.
    pub fn transit_costs(&self, prompt: bool) -> Vec<(usize, TransitCost)> {
        #[cfg(feature = "gpu")]
        {
            self.transit
                .iter()
                .map(|t| (t.ordinal, t.cost(prompt)))
                .collect()
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = prompt;
            Vec::new()
        }
    }

    /// Per card ordinal, the bytes of experts kept there, with the shared expert and the router.
    pub fn bytes_by_card(&self) -> Vec<(usize, u64)> {
        #[cfg(feature = "gpu")]
        {
            let mut held: Vec<(usize, u64)> =
                self.sets.iter().map(|s| (s.ordinal, s.bytes)).collect();
            let always = self
                .shared
                .iter()
                .map(|s| (s.ordinal(), s.bytes() as u64))
                .chain(self.router.iter().map(|r| (r.ordinal(), r.bytes() as u64)));
            for (ordinal, bytes) in always {
                match held.iter_mut().find(|(o, _)| *o == ordinal) {
                    Some((_, b)) => *b += bytes,
                    None => held.push((ordinal, bytes)),
                }
            }
            held
        }
        #[cfg(not(feature = "gpu"))]
        Vec::new()
    }

    /// Whether a card keeps the layer's shared expert.
    pub fn keeps_shared(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.shared.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// Whether a card keeps the layer's router.
    pub fn keeps_router(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.router.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// The shared expert's outputs over the rows `x` `[n, dim]`, `[n * dim]` row-major, from the
    /// card that keeps it.
    ///
    /// `None` without one, or when the card call fails: the caller runs it on the host.
    pub fn run_shared(&self, x: &[f32], n: usize, limit: f32) -> Option<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            let shared = self.shared.as_ref()?;
            match shared.run(x, n, limit) {
                Ok(y) => Some(y),
                Err(e) => {
                    tracing::warn!(
                        "card {}: shared expert not run, the host runs it: {e}",
                        shared.ordinal()
                    );
                    None
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (x, n, limit);
            None
        }
    }

    /// The router's logits of the single row `x`, from the card that keeps it, with the host's
    /// bits.
    ///
    /// `None` without one, or when the card call fails: the caller runs it on the host.
    pub fn router_logits(&self, x: &[f32]) -> Option<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            let router = self.router.as_ref()?;
            match router.logits(x) {
                Ok(y) => Some(y),
                Err(e) => {
                    tracing::warn!(
                        "card {}: router not run, the host runs it: {e}",
                        router.ordinal()
                    );
                    None
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = x;
            None
        }
    }

    /// The router's logits of the `n` rows `x` `[n, dim]`, `[n, experts]` row-major, from the
    /// card that keeps it, each row with the host's bits for that row alone.
    ///
    /// `None` without one, or when the card call fails: the caller runs it on the host.
    pub fn router_rows(&self, x: &[f32], n: usize) -> Option<Vec<f32>> {
        #[cfg(feature = "gpu")]
        {
            let router = self.router.as_ref()?;
            match router.logits_rows(x, n) {
                Ok(y) => Some(y),
                Err(e) => {
                    tracing::warn!(
                        "card {}: router of {n} rows not run, the host runs it: {e}",
                        router.ordinal()
                    );
                    None
                }
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (x, n);
            None
        }
    }

    /// The router a card keeps, when it is card `ordinal`.
    #[cfg(feature = "gpu")]
    pub(super) fn router_on(&self, ordinal: usize) -> Option<&gpu::RouterCard> {
        self.router.as_ref().filter(|r| r.ordinal() == ordinal)
    }

    /// Keeps `shared`, the layer's shared expert, on card `ordinal`, its products in their units
    /// of `choices`.
    #[cfg(feature = "gpu")]
    pub(super) fn keep_shared(
        &mut self,
        ordinal: usize,
        shared: &BlockScaledExpert,
        choices: &super::card_packed::Choices,
    ) -> Result<()> {
        self.shared = Some(gpu::SharedCard::new(ordinal, shared, choices)?);
        Ok(())
    }

    /// Keeps `router`, the layer's router, on card `ordinal` as F32 values.
    #[cfg(feature = "gpu")]
    pub(super) fn keep_router(
        &mut self,
        ordinal: usize,
        router: &crate::tensor::blockscaled::projection::Projection,
    ) -> Result<()> {
        self.router = Some(gpu::RouterCard::new(ordinal, router)?);
        Ok(())
    }

    /// The outputs of the experts `ids`, kept on the cards, over their rows.
    ///
    /// `rows[i]` are the row indices of `x` `[n, dim]` that expert `ids[i]` runs over.
    /// Answers per expert its outputs `[rows[i].len(), dim]` row-major, in the order of `ids`.
    /// `None` when a card call fails: the caller runs them on the host.
    pub fn run(
        &self,
        ids: &[usize],
        rows: &[Vec<usize>],
        x: &[f32],
        dim: usize,
        limit: f32,
    ) -> Option<Vec<Vec<f32>>> {
        let (_, outs) = self.run_cards(ids, rows, x, dim, limit, false);
        outs.into_iter().collect()
    }

    /// The outputs of the experts `ids` kept on the cards over their rows, and, when `shared`
    /// and a card keeps it, the shared expert's outputs over every row of `x`.
    ///
    /// `rows[i]` are the row indices of `x` `[n, dim]` that expert `ids[i]` runs over.
    /// Each card with work runs on a thread of its own, the calling thread drives the first.
    /// The card that keeps the shared expert runs it before its routed experts.
    /// Answers the shared expert's outputs `[n * dim]` row-major, `None` when it did not run.
    /// Answers per expert its outputs `[rows[i].len(), dim]` row-major, in the order of `ids`.
    /// An expert no card keeps, or whose card call failed, answers `None`: the caller runs it on
    /// the host.
    pub fn run_cards(
        &self,
        ids: &[usize],
        rows: &[Vec<usize>],
        x: &[f32],
        dim: usize,
        limit: f32,
        shared: bool,
    ) -> (Option<Vec<f32>>, Vec<Option<Vec<f32>>>) {
        let (shared, kept, _) =
            self.run_cards_moving(ids, rows, x, dim, limit, shared, &[], &[], false, &[], &[]);
        (shared, kept)
    }

    /// [`Self::run_cards`], and on each card of `moving`, after its kept experts, the experts
    /// it takes in transit, read from `experts` by id.
    ///
    /// A card of `beside_kept` reads its experts in transit while its kept experts run.
    /// `prompt` says the forward reads a prompt: its transit then measures the costs of the next
    /// prompt forward.
    /// The blocks of an expert of `through_cache` go up through the page cache.
    /// Answers per entry of `moving`, per expert it gives, its outputs `[rows, dim]` row-major.
    /// An expert in transit that did not run answers `None`: the caller runs it on the host.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn run_cards_moving(
        &self,
        ids: &[usize],
        rows: &[Vec<usize>],
        x: &[f32],
        dim: usize,
        limit: f32,
        shared: bool,
        moving: &[(usize, Vec<(usize, Vec<usize>)>)],
        beside_kept: &[usize],
        prompt: bool,
        through_cache: &[usize],
        experts: &[BlockScaledExpert],
    ) -> (
        Option<Vec<f32>>,
        Vec<Option<Vec<f32>>>,
        Vec<Vec<Option<Vec<f32>>>>,
    ) {
        #[cfg(feature = "gpu")]
        {
            let n = x.len() / dim.max(1);
            let shared = self.shared.as_ref().filter(|_| shared);
            // Per card with work, its set and the indices of `ids` it keeps.
            let mut work: Vec<(usize, Option<&gpu::CardSet>, Vec<usize>)> = self
                .sets
                .iter()
                .map(|set| {
                    let mine: Vec<usize> = (0..ids.len())
                        .filter(|&i| set.experts.contains_key(&ids[i]))
                        .collect();
                    (set.ordinal, Some(set), mine)
                })
                .filter(|(ordinal, _, mine)| {
                    !mine.is_empty() || moving.iter().any(|(o, m)| o == ordinal && !m.is_empty())
                })
                .collect();
            if let Some(s) = shared {
                if !work.iter().any(|w| w.0 == s.ordinal()) {
                    work.push((s.ordinal(), None, Vec::new()));
                }
            }
            for (ordinal, m) in moving {
                if !m.is_empty() && !work.iter().any(|w| w.0 == *ordinal) {
                    work.push((*ordinal, None, Vec::new()));
                }
            }
            // The shared expert's card stands first.
            work.sort_by_key(|w| Some(w.0) != shared.map(|s| s.ordinal()));
            let card_prof = |ordinal: usize, stage: &'static str, since: std::time::Instant| {
                crate::inference::stage_prof::prof_record(
                    crate::inference::stage_prof::on_card(ordinal, stage),
                    since.elapsed().as_nanos() as u64,
                );
            };
            // A card's experts in transit, after its kept ones.
            let transit = |ordinal: usize| -> Vec<(usize, Vec<Option<Vec<f32>>>)> {
                let Some(t) = self.transit.iter().find(|t| t.ordinal == ordinal) else {
                    return Vec::new();
                };
                moving
                    .iter()
                    .enumerate()
                    .filter(|(_, (o, m))| *o == ordinal && !m.is_empty())
                    .map(|(k, (_, m))| {
                        let given: Vec<(&BlockScaledExpert, &[usize])> = m
                            .iter()
                            .filter_map(|(e, r)| experts.get(*e).map(|x| (x, r.as_slice())))
                            .collect();
                        let cached: Vec<bool> =
                            m.iter().map(|(e, _)| through_cache.contains(e)).collect();
                        let started = std::time::Instant::now();
                        let outs = match given.len() == m.len() {
                            true => t.run(&given, &cached, x, dim, limit, prompt),
                            false => vec![None; m.len()],
                        };
                        card_prof(ordinal, "experts: transit", started);
                        crate::inference::stage_prof::prof_record(
                            "experts: transit",
                            started.elapsed().as_nanos() as u64,
                        );
                        let pairs: usize = m
                            .iter()
                            .zip(&outs)
                            .filter(|(_, o)| o.is_some())
                            .map(|((_, r), _)| r.len())
                            .sum();
                        self.moved
                            .fetch_add(pairs as u64, std::sync::atomic::Ordering::Relaxed);
                        crate::inference::expert_counters::global().card(ordinal, pairs as u64);
                        (k, outs)
                    })
                    .collect()
            };
            let transit = &transit;
            let kept_one = |(ordinal, set, mine): &(usize, Option<&gpu::CardSet>, Vec<usize>)| {
                let started = std::time::Instant::now();
                let shared_out = shared.filter(|s| s.ordinal() == *ordinal).and_then(|s| {
                    match s.run(x, n, limit) {
                        Ok(y) => Some(y),
                        Err(e) => {
                            tracing::warn!(
                                "card {ordinal}: shared expert not run, the host runs it: {e}"
                            );
                            None
                        }
                    }
                });
                card_prof(*ordinal, "experts: shared", started);
                let started = std::time::Instant::now();
                // The kept experts' kernels are queued on the card's stream.
                let launched = set.map(|set| {
                    let ids_here: Vec<usize> = mine.iter().map(|&i| ids[i]).collect();
                    let rows_here: Vec<Vec<usize>> =
                        mine.iter().map(|&i| rows[i].clone()).collect();
                    let pairs: usize = rows_here.iter().map(Vec::len).sum();
                    crate::inference::stage_prof::prof_count(
                        crate::inference::stage_prof::on_card(*ordinal, "experts: kept pairs"),
                        pairs as u64,
                    );
                    let launched = set.launch(&ids_here, &rows_here, x, dim, limit);
                    (pairs, set.run_bytes(pairs, n), launched)
                });
                card_prof(*ordinal, "experts: kept launch", started);
                (shared_out, launched)
            };
            let kept_one = &kept_one;
            // The outputs of a card's kept experts, once their kernels end.
            let finish = |ordinal: usize, launched: Option<(usize, u64, Result<gpu::Launched>)>| {
                let started = std::time::Instant::now();
                // A prompt forward's outputs come down through the readers of the card's transit.
                let through = self.transit.iter().find(|t| prompt && t.ordinal == ordinal);
                let outs = launched.and_then(|(pairs, bytes, launched)| {
                    match launched.and_then(|l| l.finish_through(through.map(|t| &**t))) {
                        Ok(ys) => {
                            self.ran
                                .fetch_add(pairs as u64, std::sync::atomic::Ordering::Relaxed);
                            Some(ys)
                        }
                        Err(e) => {
                            tracing::warn!(
                                "card {ordinal}: routed experts not run, the host runs them: {e}"
                            );
                            self.note_short(ordinal, bytes);
                            None
                        }
                    }
                });
                card_prof(ordinal, "experts: kept finish", started);
                outs
            };
            let finish = &finish;
            // A card of `beside_kept` reads its transit before its kept experts end.
            let one = |w: &(usize, Option<&gpu::CardSet>, Vec<usize>)| {
                let started = std::time::Instant::now();
                let (shared_out, launched) = kept_one(w);
                let (outs, carried) = match beside_kept.contains(&w.0) {
                    true => {
                        let carried = transit(w.0);
                        (finish(w.0, launched), carried)
                    }
                    false => {
                        let outs = finish(w.0, launched);
                        (outs, transit(w.0))
                    }
                };
                card_prof(w.0, "experts: thread", started);
                (shared_out, outs, carried)
            };
            let one = &one;
            type Ran = (
                Option<Vec<f32>>,
                Option<Vec<Vec<f32>>>,
                Vec<(usize, Vec<Option<Vec<f32>>>)>,
            );
            let ran: Vec<Ran> = std::thread::scope(|scope| {
                let others: Vec<_> = work
                    .iter()
                    .skip(1)
                    .map(|w| scope.spawn(move || one(w)))
                    .collect();
                let first = work.first().map(one);
                first
                    .into_iter()
                    .chain(
                        others
                            .into_iter()
                            .map(|h| h.join().unwrap_or((None, None, Vec::new()))),
                    )
                    .collect()
            });
            let mut out: Vec<Option<Vec<f32>>> = vec![None; ids.len()];
            let mut moved: Vec<Vec<Option<Vec<f32>>>> =
                moving.iter().map(|(_, m)| vec![None; m.len()]).collect();
            let mut shared_out = None;
            for ((_, _, mine), (sh, outs, carried)) in work.iter().zip(ran) {
                shared_out = shared_out.or(sh);
                for (&i, y) in mine.iter().zip(outs.into_iter().flatten()) {
                    out[i] = Some(y);
                }
                for (k, ys) in carried {
                    moved[k] = ys;
                }
            }
            (shared_out, out, moved)
        }
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (
                rows,
                x,
                dim,
                limit,
                shared,
                beside_kept,
                prompt,
                through_cache,
                experts,
            );
            (
                None,
                vec![None; ids.len()],
                moving.iter().map(|(_, m)| vec![None; m.len()]).collect(),
            )
        }
    }

    /// Notes that a run on card `ordinal` that asked for `bytes` failed.
    #[cfg(feature = "gpu")]
    fn note_short(&self, ordinal: usize, bytes: u64) {
        let mut short = self.short.lock().unwrap_or_else(|e| e.into_inner());
        match short.iter_mut().find(|(o, _)| *o == ordinal) {
            Some((_, b)) => *b = (*b).max(bytes),
            None => short.push((ordinal, bytes)),
        }
    }

    /// Keeps `experts` (id and weights) on card `ordinal`, beside what the layer keeps.
    ///
    /// Refused when their format has no kernel that gives the host's bits, or the upload fails.
    #[cfg(feature = "gpu")]
    pub fn keep(&mut self, ordinal: usize, experts: &[(usize, &BlockScaledExpert)]) -> Result<()> {
        let set = gpu::CardSet::new(ordinal, experts)?;
        self.order.extend(experts.iter().map(|&(e, _)| e));
        match self.sets.iter_mut().find(|s| s.ordinal == ordinal) {
            Some(held) => held.absorb(set),
            None => self.sets.push(set),
        }
        Ok(())
    }
}

/// Whether the cards have a kernel that gives the host's bits for `expert`'s format and shape.
pub fn served(expert: &BlockScaledExpert) -> bool {
    #[cfg(feature = "gpu")]
    {
        gpu::format_of(expert).is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
        let _ = expert;
        false
    }
}

/// The host's outputs of `experts` over the rows `x` `[n, dim]`, each over all the rows.
///
/// The experts run at once, as a layer's forward runs them.
pub fn host_outputs(
    experts: &[&BlockScaledExpert],
    x: &crate::tensor::Tensor,
    limit: f32,
) -> Result<Vec<Vec<f32>>> {
    use rayon::prelude::*;
    experts
        .par_iter()
        .map(|e| e.forward(x, limit)?.flatten_all()?.to_vec1::<f32>())
        .collect()
}

#[cfg(feature = "gpu")]
pub(super) mod gpu {
    use super::super::card_packed::{Choices, PackedDev};
    use super::BlockScaledExpert;
    use crate::tensor::blockscaled::projection::Projection;
    use crate::tensor::cuda::{
        alloc_f32, expert_tables, gpu_experts_exact, q8k32_rows_to_card, swiglu_limit, upload_f32,
        CudaDevice, ExpertFormat, ExpertTables,
    };
    use crate::tensor::gpu_runtime::driver::{CudaSlice, CudaView, LaunchConfig, PushKernelArg};
    use crate::tensor::quant_cpu::QK_K;
    use crate::tensor::{Device, Error, Result};
    use std::collections::BTreeMap;
    use std::sync::Arc;

    /// The kernels' format of `expert`, `None` when no kernel reads it or its shape.
    ///
    /// The kernels read the rows as q8_K blocks: `dim` and `inter` are whole blocks of it.
    pub(crate) fn format_of(expert: &BlockScaledExpert) -> Option<ExpertFormat> {
        let format = ExpertFormat::of(
            expert.gate.quant_dtype()?,
            expert.up.quant_dtype()?,
            expert.down.quant_dtype()?,
        )?;
        // The gate is `[inter, dim]`.
        let dims = expert.gate.dims();
        let (inter, dim) = (*dims.first()?, *dims.get(1)?);
        (dim % QK_K == 0 && inter % QK_K == 0).then_some(format)
    }

    /// The stored blocks of a projection in a GGUF format.
    fn blocks(p: &Projection) -> Result<std::borrow::Cow<'_, [u8]>> {
        match p {
            Projection::Quant(q) => q.data(),
            _ => Err(Error::msg("a card expert is stored in GGUF blocks")),
        }
    }

    /// Experts of 1 format kept on 1 card.
    pub(crate) struct CardSet {
        pub(crate) ordinal: usize,
        dev: Arc<CudaDevice>,
        tables: Arc<ExpertTables>,
        /// `(dim, inter)`.
        dims: (usize, usize),
        /// Per expert id, gate, up and down blocks.
        pub(crate) experts: BTreeMap<usize, [CudaSlice<u8>; 3]>,
        /// Bytes of the blocks kept.
        pub(crate) bytes: u64,
    }

    impl CardSet {
        /// `experts` uploaded to card `ordinal`.
        ///
        /// Refused when they are not all of 1 format and shape the kernels read.
        pub(crate) fn new(ordinal: usize, experts: &[(usize, &BlockScaledExpert)]) -> Result<Self> {
            let Some(&(_, first)) = experts.first() else {
                return Err(Error::msg("no expert to keep on a card"));
            };
            let format = format_of(first)
                .ok_or_else(|| Error::msg("no card kernel reads these experts' format"))?;
            let dims = first.gate.dims();
            let dims = (dims[1], dims[0]);
            let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
                return Err(Error::msg("card experts need a CUDA device"));
            };
            let tables = Arc::new(expert_tables(&dev, format)?);
            let mut set = Self {
                ordinal,
                dev,
                tables,
                dims,
                experts: BTreeMap::new(),
                bytes: 0,
            };
            for &(id, e) in experts {
                if format_of(e) != Some(format) || (e.gate.dims()[1], e.gate.dims()[0]) != dims {
                    return Err(Error::msg(format!(
                        "expert {id} differs in format or shape from the others on card {ordinal}"
                    )));
                }
                let stream = set.dev.stream();
                let mut up = |p: &Projection| -> Result<CudaSlice<u8>> {
                    let b = blocks(p)?;
                    set.bytes += b.len() as u64;
                    stream
                        .clone_htod(b.as_ref())
                        .map_err(|err| Error::msg(format!("expert {id} upload: {err}")))
                };
                let weights = [up(&e.gate)?, up(&e.up)?, up(&e.down)?];
                set.experts.insert(id, weights);
            }
            Ok(set)
        }

        /// Takes the experts of `other`, kept on the same card.
        pub(crate) fn absorb(&mut self, other: CardSet) {
            self.bytes += other.bytes;
            self.experts.extend(other.experts);
        }

        /// Drops expert `e` from the card: the bytes its blocks held, `None` when the card does
        /// not keep it.
        pub(crate) fn release(&mut self, e: usize) -> Option<u64> {
            let blocks = self.experts.remove(&e)?;
            let freed: u64 = blocks.iter().map(|b| b.len() as u64).sum();
            self.bytes = self.bytes.saturating_sub(freed);
            Some(freed)
        }

        /// Bytes a run of `pairs` (expert, row) pairs over `n` rows takes on the card: the rows,
        /// each pair's intermediate values and each pair's outputs, as F32.
        pub(crate) fn run_bytes(&self, pairs: usize, n: usize) -> u64 {
            let (dim, inter) = self.dims;
            let floats = n * dim + pairs * (inter + dim);
            (floats * std::mem::size_of::<f32>()) as u64
        }

        /// The outputs of `ids` over their `rows` of `x` `[n, dim]`, in the order of `ids`.
        pub(crate) fn run(
            &self,
            ids: &[usize],
            rows: &[Vec<usize>],
            x: &[f32],
            dim: usize,
            limit: f32,
        ) -> Result<Vec<Vec<f32>>> {
            self.launch(ids, rows, x, dim, limit)?.finish()
        }

        /// [`Self::run`] queued on the card's stream.
        ///
        /// The outputs come down with [`Launched::finish`].
        pub(crate) fn launch(
            &self,
            ids: &[usize],
            rows: &[Vec<usize>],
            x: &[f32],
            dim: usize,
            limit: f32,
        ) -> Result<Launched> {
            if dim != self.dims.0 {
                return Err(Error::msg(format!(
                    "rows of {dim} for experts of {}",
                    self.dims.0
                )));
            }
            let weights = ids
                .iter()
                .map(|e| {
                    self.experts
                        .get(e)
                        .map(|[g, u, d]| (g, u, d))
                        .ok_or_else(|| Error::msg(format!("expert {e} is not on the card")))
                })
                .collect::<Result<Vec<_>>>()?;
            // Instances of 1 expert stand together, its rows in row order.
            let instances: Vec<(usize, usize)> = rows
                .iter()
                .enumerate()
                .flat_map(|(i, r)| r.iter().map(move |&t| (i, t)))
                .collect();
            let lens: Vec<usize> = rows.iter().map(|r| r.len() * dim).collect();
            if instances.is_empty() {
                return Ok(Launched {
                    dev: self.dev.clone(),
                    lens,
                    held: None,
                });
            }
            let xq = q8k32_rows_to_card(&self.dev, x)?;
            let (h, y) = gpu_experts_exact(
                &self.dev,
                &weights,
                &self.tables,
                self.dims,
                &xq,
                &instances,
                limit,
            )?;
            Ok(Launched {
                dev: self.dev.clone(),
                lens,
                held: Some((xq, h, y)),
            })
        }
    }

    /// Records the time since `since` as stage `stage` of card `ordinal`.
    fn card_stage(ordinal: usize, stage: &'static str, since: std::time::Instant) {
        use crate::inference::stage_prof::{on_card, prof_record};
        prof_record(on_card(ordinal, stage), since.elapsed().as_nanos() as u64);
    }

    /// A run of kept experts queued on a card's stream.
    pub(crate) struct Launched {
        dev: Arc<CudaDevice>,
        /// Per expert, the length of its outputs.
        lens: Vec<usize>,
        /// The rows, the down projection's inputs and the outputs, on the card until the run
        /// ends.
        held: Option<(CudaSlice<u8>, CudaSlice<f32>, CudaSlice<f32>)>,
    }

    impl Launched {
        /// The outputs of each expert, once the run ends.
        pub(crate) fn finish(self) -> Result<Vec<Vec<f32>>> {
            self.finish_through(None)
        }

        /// [`Self::finish`], the outputs brought down through the page-locked buffers of
        /// `through` when it has them open.
        pub(crate) fn finish_through(self, through: Option<&Transit>) -> Result<Vec<Vec<f32>>> {
            let Some((_xq, _h, y)) = &self.held else {
                return Ok(vec![Vec::new(); self.lens.len()]);
            };
            let ordinal = self.dev.ordinal();
            let stream = self.dev.stream();
            // Under the stage profile the kernels' end is waited for apart from the copy.
            if crate::inference::stage_prof::stage_prof_enabled() {
                let since = std::time::Instant::now();
                let _ = stream.synchronize();
                card_stage(ordinal, "experts: kept wait", since);
            }
            let since = std::time::Instant::now();
            if let Some(outs) = through.and_then(|t| t.download(stream, y, &self.lens)) {
                card_stage(ordinal, "experts: kept down", since);
                return Ok(outs);
            }
            let since = std::time::Instant::now();
            let flat = stream
                .clone_dtoh(y)
                .map_err(|e| Error::msg(format!("card expert outputs: {e}")))?;
            card_stage(ordinal, "experts: kept down", since);
            let since = std::time::Instant::now();
            let mut at = 0;
            let outs = self
                .lens
                .iter()
                .map(|&n| {
                    let part = flat[at..at + n].to_vec();
                    at += n;
                    part
                })
                .collect();
            card_stage(ordinal, "experts: kept split", since);
            Ok(outs)
        }
    }

    /// A card that runs, in a forward, experts it does not keep.
    pub(crate) struct Transit {
        pub(crate) ordinal: usize,
        device: Device,
        dev: Arc<CudaDevice>,
        tables: ExpertTables,
        /// `(dim, inter)`.
        dims: (usize, usize),
        /// The costs the load measured.
        measured: super::TransitCost,
        /// Totals measured since the load: nanoseconds of uploads and their bytes, nanoseconds
        /// of runs and their tiles, host core nanoseconds and their pairs.
        totals: [std::sync::atomic::AtomicU64; 6],
        /// The same measures over the last prompt forward that took experts in transit.
        last: [std::sync::atomic::AtomicU64; 6],
        /// The reads and runs of the prompt forward under way, summed over its batches.
        forward: [std::sync::atomic::AtomicU64; 4],
        /// Bytes of the largest weight of an expert: what 1 direct read carries.
        unit: usize,
        /// The direct reader of the experts' file.
        bounce: std::sync::Mutex<Reader>,
        /// Bytes read past the page cache since the load.
        direct: std::sync::atomic::AtomicU64,
        /// The stream the blocks go up on, beside the kernels of the card's stream.
        copy: Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
    }

    /// The state of a transit's direct reader.
    enum Reader {
        /// Not opened yet, or opened on another file.
        Closed,
        /// 1 direct reader a reader thread.
        #[cfg(target_os = "linux")]
        Open(Vec<super::super::expert_tier::direct::Bounce>),
        /// Failed once: the experts go up from where they lie.
        Off,
    }

    impl Transit {
        /// Card `ordinal` ready for experts of the format and shape of `like`.
        pub(crate) fn new(
            ordinal: usize,
            like: &BlockScaledExpert,
            cost: super::TransitCost,
        ) -> Result<Self> {
            let format = format_of(like)
                .ok_or_else(|| Error::msg("no card kernel reads these experts' format"))?;
            let device = Device::new_cuda(ordinal)?;
            let Device::Cuda(dev) = device.clone() else {
                return Err(Error::msg("experts in transit need a CUDA device"));
            };
            let tables = expert_tables(&dev, format)?;
            let copy = dev
                .context()
                .new_stream()
                .map_err(|e| Error::msg(format!("expert copy stream: {e}")))?;
            let dims = like.gate.dims();
            let unit = like
                .gate
                .bytes()
                .max(like.up.bytes())
                .max(like.down.bytes());
            Ok(Self {
                ordinal,
                device,
                dev,
                tables,
                dims: (dims[1], dims[0]),
                measured: cost,
                totals: Default::default(),
                last: Default::default(),
                forward: Default::default(),
                unit,
                bounce: std::sync::Mutex::new(Reader::Closed),
                direct: Default::default(),
                copy,
            })
        }

        /// Bytes read past the page cache since the load.
        pub(crate) fn bytes_read_direct(&self) -> u64 {
            self.direct.load(std::sync::atomic::Ordering::Relaxed)
        }

        /// Whether the experts go up through the direct reader, until a read fails.
        pub(crate) fn reads_direct(&self) -> bool {
            let state = self.bounce.lock().unwrap_or_else(|e| e.into_inner());
            cfg!(target_os = "linux") && !matches!(*state, Reader::Off)
        }

        /// The stream the blocks and rows of a transit go up on: the copy stream in a prompt
        /// forward, the card's stream in any other.
        fn up_stream(&self, prompt: bool) -> &Arc<crate::tensor::gpu_runtime::driver::CudaStream> {
            match prompt {
                true => &self.copy,
                false => self.dev.stream(),
            }
        }

        /// The blocks `parts` on the card, in their order, on the stream of [`Self::up_stream`].
        ///
        /// Blocks a mapped file holds and the page cache does not are read past the cache into
        /// page-locked memory by the reader threads, each queueing its copies on that stream
        /// while it reads the next ones.
        /// The reader threads read a block of `through` through the page cache instead.
        /// Other blocks, and those of a direct read that fails, go up from where they lie.
        fn upload(
            &self,
            parts: &[&[u8]],
            through: &[bool],
            prompt: bool,
        ) -> Result<Vec<CudaSlice<u8>>> {
            let mut on_card: Vec<Option<CudaSlice<u8>>> = parts.iter().map(|_| None).collect();
            #[cfg(target_os = "linux")]
            self.upload_direct(parts, through, prompt, &mut on_card);
            #[cfg(not(target_os = "linux"))]
            let _ = through;
            parts
                .iter()
                .zip(on_card)
                .map(|(b, up)| match up {
                    Some(up) => Ok(up),
                    None => self
                        .up_stream(prompt)
                        .clone_htod(*b)
                        .map_err(|err| Error::msg(format!("expert upload in transit: {err}"))),
                })
                .collect()
        }

        /// [`Self::upload`] through the direct readers: fills `on_card[i]` for each part of
        /// `parts` they read whole.
        ///
        /// A part of `through` is read through the page cache.
        #[cfg(target_os = "linux")]
        fn upload_direct(
            &self,
            parts: &[&[u8]],
            through: &[bool],
            prompt: bool,
            on_card: &mut [Option<CudaSlice<u8>>],
        ) {
            use super::super::expert_tier::direct::{residency, Bounce};
            use crate::tensor::gpu_runtime::driver::DevicePtr;
            use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
            static WARNED: AtomicBool = AtomicBool::new(false);
            let warn = |e: &str| {
                if !WARNED.swap(true, Ordering::Relaxed) {
                    tracing::warn!("card {}: direct expert reads off: {e}", self.ordinal);
                }
            };
            let Some(page) = crate::tensor::mapped::page_size() else {
                return;
            };
            // Per part a mapped file holds and the page cache does not hold whole: its place,
            // its file and its offset there.
            let wanted: Vec<(usize, Arc<std::fs::File>, u64)> = parts
                .iter()
                .enumerate()
                .filter_map(|(i, b)| {
                    let start = b.as_ptr() as usize;
                    let (file, off) = crate::tensor::mapped::file_span(start, b.len())?;
                    let cached = residency(start, b.len(), page)
                        .is_some_and(|runs| runs.iter().all(|r| r.2));
                    (!cached).then_some((i, file, off))
                })
                .collect();
            let Some(file) = wanted.first().map(|w| w.1.clone()) else {
                return;
            };
            let mut state = self.bounce.lock().unwrap_or_else(|e| e.into_inner());
            if matches!(&*state, Reader::Open(r) if r.first().is_some_and(|b| !b.reads(&file))) {
                *state = Reader::Closed;
            }
            if matches!(*state, Reader::Closed) {
                // As many readers as reach the disk's rate on reads of 1 weight.
                let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
                let count = super::super::expert_readahead::probe_readers(&file, self.unit, cores)
                    .unwrap_or(1);
                *state = match (0..count)
                    .map(|_| Bounce::open(&file, self.unit.max(page), page))
                    .collect::<std::result::Result<Vec<Bounce>, String>>()
                {
                    Ok(r) => Reader::Open(r),
                    Err(e) => {
                        warn(&e);
                        Reader::Off
                    }
                };
            }
            let Reader::Open(readers) = &mut *state else {
                return;
            };
            let stream = self.up_stream(prompt);
            // Every byte is written by the copies the readers queue before a kernel reads it.
            let jobs: Vec<(usize, u64, CudaSlice<u8>)> = wanted
                .iter()
                .filter(|w| Arc::ptr_eq(&w.1, &file))
                .filter_map(|(i, _, off)| {
                    let up = unsafe { stream.alloc::<u8>(parts[*i].len()) }.ok()?;
                    Some((*i, *off, up))
                })
                .collect();
            let next = AtomicUsize::new(0);
            let failed: Vec<AtomicBool> = jobs.iter().map(|_| AtomicBool::new(false)).collect();
            std::thread::scope(|scope| {
                for reader in readers.iter_mut() {
                    let (jobs, next, failed, warn) = (&jobs, &next, &failed, &warn);
                    scope.spawn(move || {
                        if self.dev.context().bind_to_thread().is_err() {
                            return;
                        }
                        loop {
                            let k = next.fetch_add(1, Ordering::Relaxed);
                            let Some((i, off, up)) = jobs.get(k) else {
                                return;
                            };
                            let len = parts[*i].len();
                            let at = {
                                let (at, _sync) = up.device_ptr(stream);
                                at
                            };
                            let cached = through.get(*i).copied().unwrap_or(false);
                            let (done, err) = reader.upload(*off, len, at, stream, cached);
                            if err.is_some() || done < len {
                                failed[k].store(true, Ordering::Relaxed);
                                warn(&err.unwrap_or_else(|| "a short read".to_string()));
                            }
                        }
                    });
                }
            });
            // A part no reader took, or one whose read failed, goes up from the mapping.
            let taken = next.load(Ordering::Relaxed);
            let mut off = false;
            for (k, (i, _, up)) in jobs.into_iter().enumerate() {
                if k >= taken || failed[k].load(Ordering::Relaxed) {
                    off |= k < taken;
                    continue;
                }
                if !through.get(i).copied().unwrap_or(false) {
                    self.direct
                        .fetch_add(parts[i].len() as u64, Ordering::Relaxed);
                }
                on_card[i] = Some(up);
            }
            if off {
                *state = Reader::Off;
            }
        }

        /// The outputs `y` of a run on `stream`, `lens[i]` values for its expert `i` in order.
        ///
        /// They come down through the page-locked buffers of the direct readers, 1 thread a
        /// reader, each copying its chunks out while the next ones come down.
        /// `None` when the readers are not open or a copy fails: the caller brings `y` down
        /// itself.
        pub(crate) fn download(
            &self,
            stream: &Arc<crate::tensor::gpu_runtime::driver::CudaStream>,
            y: &CudaSlice<f32>,
            lens: &[usize],
        ) -> Option<Vec<Vec<f32>>> {
            #[cfg(target_os = "linux")]
            {
                use crate::tensor::gpu_runtime::driver::DevicePtr;
                use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
                let f32s = std::mem::size_of::<f32>();
                let total: usize = lens.iter().sum();
                if y.len() < total {
                    return None;
                }
                let mut state = self.bounce.lock().unwrap_or_else(|e| e.into_inner());
                let Reader::Open(readers) = &mut *state else {
                    return None;
                };
                if readers.is_empty() {
                    return None;
                }
                let mut outs: Vec<Vec<f32>> = lens.iter().map(|&n| vec![0f32; n]).collect();
                // Per expert, the byte its outputs start at in `y`, and where they go.
                let mut starts = Vec::with_capacity(outs.len());
                let mut at = 0;
                for o in &outs {
                    starts.push(at);
                    at += o.len() * f32s;
                }
                let dsts: Vec<(usize, usize)> = outs
                    .iter_mut()
                    .map(|o| (o.as_mut_ptr() as usize, o.len() * f32s))
                    .collect();
                // Writes the bytes from `off` of `y` to the outputs they belong to.
                let put = |off: usize, bytes: &[u8]| {
                    let mut e = starts.partition_point(|&s| s <= off).saturating_sub(1);
                    let mut done = 0;
                    while done < bytes.len() && e < dsts.len() {
                        let (ptr, cap) = dsts[e];
                        let inside = off + done - starts[e];
                        let n = (cap - inside).min(bytes.len() - done);
                        // Safety: each byte of each output is written by 1 chunk alone, and
                        // `outs` outlives the threads.
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                bytes.as_ptr().add(done),
                                (ptr as *mut u8).add(inside),
                                n,
                            )
                        };
                        done += n;
                        e += 1;
                    }
                };
                let src = {
                    let (src, _sync) = y.device_ptr(stream);
                    src
                };
                let next = AtomicUsize::new(0);
                let failed = AtomicBool::new(false);
                std::thread::scope(|scope| {
                    for reader in readers.iter_mut() {
                        let (next, failed, put) = (&next, &failed, &put);
                        scope.spawn(move || {
                            if self.dev.context().bind_to_thread().is_err() {
                                failed.store(true, Ordering::Relaxed);
                                return;
                            }
                            let len = total * f32s;
                            if let Some(e) = reader.download(src, len, next, stream, put) {
                                failed.store(true, Ordering::Relaxed);
                                tracing::warn!(
                                    "card {}: expert outputs not brought down through the \
                                     readers: {e}",
                                    self.ordinal
                                );
                            }
                        });
                    }
                });
                // A reader that does not fail hands out chunks until none is left.
                (!failed.load(Ordering::Relaxed)).then_some(outs)
            }
            #[cfg(not(target_os = "linux"))]
            {
                let _ = (stream, y, lens);
                None
            }
        }

        /// The costs now, the load's measure before any.
        ///
        /// For a prompt forward, each is the ratio of the last prompt forward's measure.
        /// For any other forward, each is the ratio of its totals since the load.
        pub(crate) fn cost(&self, prompt: bool) -> super::TransitCost {
            let of = match prompt {
                true => &self.last,
                false => &self.totals,
            };
            let at = |i: usize| of[i].load(std::sync::atomic::Ordering::Relaxed);
            let ratio = |ns: usize, of: usize, load: f64| match at(of) {
                0 => load,
                n => at(ns) as f64 * 1e-9 / n as f64,
            };
            super::TransitCost {
                secs_per_byte: ratio(0, 1, self.measured.secs_per_byte),
                secs_per_tile: ratio(2, 3, self.measured.secs_per_tile),
                host_secs_per_pair: ratio(4, 5, self.measured.host_secs_per_pair),
            }
        }

        /// Adds `ns` nanoseconds over `count` to the totals at `at` and the one after it, and
        /// to the prompt forward's sums when `prompt`.
        fn add(&self, at: usize, ns: u64, count: u64, prompt: bool) {
            use std::sync::atomic::Ordering::Relaxed;
            self.totals[at].fetch_add(ns, Relaxed);
            self.totals[at + 1].fetch_add(count, Relaxed);
            if prompt {
                self.forward[at].fetch_add(ns, Relaxed);
                self.forward[at + 1].fetch_add(count, Relaxed);
            }
        }

        /// Makes the prompt forward's sums the last measure, each pair that counted something.
        fn close_forward(&self) {
            use std::sync::atomic::Ordering::Relaxed;
            for at in [0, 2] {
                let ns = self.forward[at].swap(0, Relaxed);
                let count = self.forward[at + 1].swap(0, Relaxed);
                if count > 0 {
                    self.last[at].store(ns, Relaxed);
                    self.last[at + 1].store(count, Relaxed);
                }
            }
        }

        /// Notes `ns` nanoseconds of host cores spent on `pairs` (expert, row) pairs in a
        /// forward, of a prompt when `prompt`.
        pub(crate) fn note_host(&self, ns: u64, pairs: u64, prompt: bool) {
            use std::sync::atomic::Ordering::Relaxed;
            self.totals[4].fetch_add(ns, Relaxed);
            self.totals[5].fetch_add(pairs, Relaxed);
            if prompt && pairs > 0 {
                self.last[4].store(ns, Relaxed);
                self.last[5].store(pairs, Relaxed);
            }
        }

        /// Bytes a batch in transit may take now: what the driver has free.
        ///
        /// The bytes the card's memory pool reserves without using are the prompt's work room.
        /// The prompt's next layers take them back, so a batch leaves them alone.
        fn room_now(&self) -> u64 {
            crate::inference::place::vram_manager::free_now(&self.device).unwrap_or(0)
        }

        /// The rows `x` as q8_K blocks on the card, on the stream of [`Self::up_stream`].
        fn rows_up(&self, x: &[f32], prompt: bool) -> Result<CudaSlice<u8>> {
            self.up_stream(prompt)
                .clone_htod(&crate::tensor::quant_cpu::format::q8k32_bytes(x))
                .map_err(|e| Error::msg(format!("expert rows in transit: {e}")))
        }

        /// Gives the driver back what the pool reserved beyond `kept` for a batch.
        ///
        /// The batch's frees run on the card's stream first, and on the copy stream in a prompt
        /// forward.
        fn give_back(&self, kept: u64, prompt: bool) {
            let _ = self.dev.stream().synchronize();
            if prompt {
                let _ = self.copy.synchronize();
            }
            crate::tensor::cuda::trim_pool_to(self.ordinal, kept);
        }

        /// Bytes 1 (expert, row) pair takes on the card in a run: its rows entering the down
        /// projection, as F32 and as q8_K blocks, its output and its row index.
        fn pair_bytes(&self) -> u64 {
            let (dim, inter) = self.dims;
            let block = std::mem::size_of::<crate::tensor::quant_cpu::format::BlockQ8K>();
            let f32s = std::mem::size_of::<f32>();
            (inter * f32s + inter / QK_K * block + dim * f32s + std::mem::size_of::<i32>()) as u64
        }

        /// Bytes expert `e` takes on the card in a run over `rows` rows: its blocks and its pairs.
        pub(crate) fn need(&self, e: &BlockScaledExpert, rows: usize) -> u64 {
            expert_bytes(e) + rows as u64 * self.pair_bytes()
        }

        /// The outputs of each of `experts` over its rows of `x` `[n, dim]`, `[rows, dim]`
        /// row-major, in the order given.
        ///
        /// The rows go up once.
        /// The experts go up in batches, in the order given, each batch as large as what the
        /// driver has free then holds.
        /// The prompt's work room, which the pool reserves, is never part of a batch's room.
        /// Each batch runs, its outputs come down, and its blocks are dropped.
        /// The pool then gives the driver back what it reserved for the batch.
        /// An expert of another format or shape, one that does not fit alone, and those of a
        /// batch that fails answer `None`.
        /// The blocks of `experts[i]` go up through the page cache when `cached[i]` holds.
        /// `prompt` says the forward reads a prompt: its measures then feed the next prompt
        /// forward's costs, and its blocks go up on the copy stream.
        pub(crate) fn run(
            &self,
            experts: &[(&BlockScaledExpert, &[usize])],
            cached: &[bool],
            x: &[f32],
            dim: usize,
            limit: f32,
            prompt: bool,
        ) -> Vec<Option<Vec<f32>>> {
            self.run_within(experts, cached, x, dim, limit, prompt, || self.room_now())
        }

        /// [`Self::run`] with each batch's room read from `room`.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn run_within(
            &self,
            experts: &[(&BlockScaledExpert, &[usize])],
            cached: &[bool],
            x: &[f32],
            dim: usize,
            limit: f32,
            prompt: bool,
            room: impl Fn() -> u64,
        ) -> Vec<Option<Vec<f32>>> {
            let mut out: Vec<Option<Vec<f32>>> = vec![None; experts.len()];
            let format = Some(self.tables.format);
            let fits = |e: &BlockScaledExpert| {
                format_of(e) == format && (e.gate.dims()[1], e.gate.dims()[0]) == self.dims
            };
            if dim != self.dims.0 || experts.is_empty() {
                return out;
            }
            let (kept, _) = crate::tensor::cuda::pool_use(self.ordinal);
            let rows_up = std::time::Instant::now();
            let xq = self.rows_up(x, prompt);
            self.prof("experts: transit rows up", rows_up);
            let xq = match xq {
                Ok(xq) => xq,
                Err(e) => {
                    tracing::warn!(
                        "card {}: rows not moved up, the host runs the experts in transit: {e}",
                        self.ordinal
                    );
                    return out;
                }
            };
            let mut i = 0;
            while i < experts.len() {
                let room = room();
                let mut taken = 0u64;
                let mut end = i;
                while end < experts.len() {
                    let (e, rows) = experts[end];
                    let need = self.need(e, rows.len());
                    if !fits(e) || taken + need > room {
                        break;
                    }
                    taken += need;
                    end += 1;
                }
                if end == i {
                    i += 1;
                    continue;
                }
                let through = cached.get(i..end).unwrap_or(&[]);
                match self.run_batch(&experts[i..end], through, &xq, limit, prompt) {
                    Ok(ys) => {
                        for (k, y) in ys.into_iter().enumerate() {
                            out[i + k] = Some(y);
                        }
                    }
                    Err(e) => tracing::warn!(
                        "card {}: {} experts in transit not run, the host runs them: {e}",
                        self.ordinal,
                        end - i
                    ),
                }
                let trim = std::time::Instant::now();
                self.give_back(kept, prompt);
                self.prof("experts: transit trim", trim);
                i = end;
            }
            drop(xq);
            let trim = std::time::Instant::now();
            self.give_back(kept, prompt);
            self.prof("experts: transit trim", trim);
            if prompt {
                self.close_forward();
            }
            out
        }

        /// Records the time since `since` as stage `stage` of this card.
        fn prof(&self, stage: &'static str, since: std::time::Instant) {
            use crate::inference::stage_prof::{on_card, prof_record};
            prof_record(
                on_card(self.ordinal, stage),
                since.elapsed().as_nanos() as u64,
            );
        }

        /// Records `count` as stage `stage` of this card.
        fn count(&self, stage: &'static str, count: u64) {
            use crate::inference::stage_prof::{on_card, prof_count};
            prof_count(on_card(self.ordinal, stage), count);
        }

        /// The outputs of `experts` over their rows of `xq`, their blocks moved up for the run.
        ///
        /// The blocks of `experts[i]` go up through the page cache when `cached[i]` holds.
        /// In a prompt forward they go up on the copy stream, and the run is measured from the
        /// end of the copies and of the kernels queued before them.
        fn run_batch(
            &self,
            experts: &[(&BlockScaledExpert, &[usize])],
            cached: &[bool],
            xq: &CudaSlice<u8>,
            limit: f32,
            prompt: bool,
        ) -> Result<Vec<Vec<f32>>> {
            let stream = self.dev.stream();
            let started = std::time::Instant::now();
            let stored = experts
                .iter()
                .flat_map(|(e, _)| [&e.gate, &e.up, &e.down])
                .map(blocks)
                .collect::<Result<Vec<_>>>()?;
            let parts: Vec<&[u8]> = stored.iter().map(|b| b.as_ref()).collect();
            // 3 blocks an expert.
            let through: Vec<bool> = (0..parts.len())
                .map(|p| cached.get(p / 3).copied().unwrap_or(false))
                .collect();
            let direct_before = self.bytes_read_direct();
            let mut up = self.upload(&parts, &through, prompt)?.into_iter();
            // The kernels wait on the card for the copies queued so far.
            if prompt {
                stream
                    .join(&self.copy)
                    .map_err(|e| Error::msg(format!("expert copies in transit: {e}")))?;
            }
            let mut held: Vec<[CudaSlice<u8>; 3]> = Vec::with_capacity(experts.len());
            while let (Some(g), Some(u), Some(d)) = (up.next(), up.next(), up.next()) {
                held.push([g, u, d]);
            }
            let bytes: u64 = experts.iter().map(|(e, _)| expert_bytes(e)).sum();
            self.add(0, started.elapsed().as_nanos() as u64, bytes, prompt);
            self.prof("experts: transit read", started);
            let profiled = crate::inference::stage_prof::stage_prof_enabled();
            if profiled {
                let cached: u64 = parts
                    .iter()
                    .zip(&through)
                    .filter(|(_, t)| **t)
                    .map(|(p, _)| p.len() as u64)
                    .sum();
                let direct = self.bytes_read_direct() - direct_before;
                self.count("experts: transit kB", bytes / 1000);
                self.count("experts: transit kB direct", direct / 1000);
                self.count("experts: transit kB through the cache", cached / 1000);
                self.count("experts: transit experts", experts.len() as u64);
                let pairs: usize = experts.iter().map(|(_, r)| r.len()).sum();
                self.count("experts: transit pairs", pairs as u64);
            }
            let started = std::time::Instant::now();
            if prompt || profiled {
                let _ = stream.synchronize();
                self.prof("experts: transit copy tail", started);
            }
            // The run of a prompt forward is measured from the end of the copies.
            let started = match prompt {
                true => std::time::Instant::now(),
                false => started,
            };
            let launched = std::time::Instant::now();
            let weights: Vec<(&CudaSlice<u8>, &CudaSlice<u8>, &CudaSlice<u8>)> =
                held.iter().map(|[g, u, d]| (g, u, d)).collect();
            // Instances of 1 expert stand together, its rows in row order.
            let instances: Vec<(usize, usize)> = experts
                .iter()
                .enumerate()
                .flat_map(|(i, (_, r))| r.iter().map(move |&t| (i, t)))
                .collect();
            let (_h, y) = gpu_experts_exact(
                &self.dev,
                &weights,
                &self.tables,
                self.dims,
                xq,
                &instances,
                limit,
            )?;
            if profiled {
                let _ = stream.synchronize();
                self.prof("experts: transit run", launched);
            }
            let down = std::time::Instant::now();
            let dim = self.dims.0;
            let lens: Vec<usize> = experts.iter().map(|(_, r)| r.len() * dim).collect();
            // A prompt forward's outputs come down through the readers' buffers.
            let through = match prompt {
                true => self.download(stream, &y, &lens),
                false => None,
            };
            let flat = match through {
                Some(_) => Vec::new(),
                None => stream
                    .clone_dtoh(&y)
                    .map_err(|e| Error::msg(format!("expert outputs in transit: {e}")))?,
            };
            self.prof("experts: transit down", down);
            let tiles: usize = experts
                .iter()
                .map(|(_, r)| r.len().div_ceil(crate::tensor::cuda::EXACT_TILE))
                .sum();
            self.add(2, started.elapsed().as_nanos() as u64, tiles as u64, prompt);
            self.count("experts: transit tiles", tiles as u64);
            if let Some(outs) = through {
                return Ok(outs);
            }
            let split = std::time::Instant::now();
            let mut at = 0;
            let outs = lens
                .iter()
                .map(|&n| {
                    let part = flat[at..at + n].to_vec();
                    at += n;
                    part
                })
                .collect();
            self.prof("experts: transit split", split);
            Ok(outs)
        }
    }

    /// Bytes of the 3 weights of `e`.
    fn expert_bytes(e: &BlockScaledExpert) -> u64 {
        (e.gate.bytes() + e.up.bytes() + e.down.bytes()) as u64
    }

    /// A layer's shared expert on the card that runs its token mixer.
    pub(crate) struct SharedCard {
        ordinal: usize,
        dev: Arc<CudaDevice>,
        gate: PackedDev,
        up: PackedDev,
        down: PackedDev,
    }

    impl SharedCard {
        /// `expert` uploaded to card `ordinal` as stored, its products in their units of
        /// `choices`.
        pub(crate) fn new(
            ordinal: usize,
            expert: &BlockScaledExpert,
            choices: &Choices,
        ) -> Result<Self> {
            let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
                return Err(Error::msg("a card shared expert needs a CUDA device"));
            };
            Ok(Self {
                ordinal,
                gate: PackedDev::new(&dev, &expert.gate, choices)?,
                up: PackedDev::new(&dev, &expert.up, choices)?,
                down: PackedDev::new(&dev, &expert.down, choices)?,
                dev,
            })
        }

        pub(crate) fn ordinal(&self) -> usize {
            self.ordinal
        }

        /// Bytes the card holds for the 3 weights.
        pub(crate) fn bytes(&self) -> usize {
            self.gate.bytes() + self.up.bytes() + self.down.bytes()
        }

        /// The outputs over the rows `x` `[n, dim]`, `[n * dim]` row-major.
        ///
        /// The rows go up once, the gate and up products, the SwiGLU and the down product run on
        /// the card, and the outputs come back once.
        pub(crate) fn run(&self, x: &[f32], n: usize, limit: f32) -> Result<Vec<f32>> {
            let dev: &CudaDevice = &self.dev;
            let rows = upload_f32(dev, "shared expert rows", x)?;
            let inter = self.gate.out();
            let g = self.gate.apply(&rows, n)?;
            let u = self.up.apply(&rows, n)?;
            let h = swiglu_limit(dev, &g.slice(0..n * inter), &u.slice(0..n * inter), limit)?;
            let y = self.down.apply(&h, n)?;
            super::super::card_mixers::download(dev, &y, n * self.down.out())
        }
    }

    /// A layer's router on the card that runs its token mixer, as the F32 values the host
    /// holds.
    pub(crate) struct RouterCard {
        ordinal: usize,
        dev: Arc<CudaDevice>,
        /// `[out, inp]` row-major.
        values: CudaSlice<f32>,
        out: usize,
        inp: usize,
    }

    impl RouterCard {
        /// `router`'s values uploaded to card `ordinal`.
        pub(crate) fn new(ordinal: usize, router: &Projection) -> Result<Self> {
            let (out, inp) = match router.dims().as_slice() {
                [out, inp] => (*out, *inp),
                _ => return Err(Error::msg("a router is a matrix")),
            };
            let Device::Cuda(dev) = Device::new_cuda(ordinal)? else {
                return Err(Error::msg("a card router needs a CUDA device"));
            };
            dev.quantized_fn("seq_row_f32")?;
            dev.quantized_fn("seq_rows_f32")?;
            let values = upload_f32(&dev, "router", &router.dense_f32()?)?;
            Ok(Self {
                ordinal,
                dev,
                values,
                out,
                inp,
            })
        }

        pub(crate) fn ordinal(&self) -> usize {
            self.ordinal
        }

        /// Bytes the card holds for the values.
        pub(crate) fn bytes(&self) -> usize {
            self.out * self.inp * std::mem::size_of::<f32>()
        }

        /// The logits of the single row `x`.
        ///
        /// Each is the fold from -0, in input order, of the rounded products of a weight row
        /// and the row: the steps the host takes for 1 row.
        pub(crate) fn logits(&self, x: &[f32]) -> Result<Vec<f32>> {
            let row = upload_f32(&self.dev, "router row", x)?;
            let y = self.logits_dev(&row.slice(0..x.len()))?;
            super::super::card_mixers::download(&self.dev, &y, self.out)
        }

        /// The logits of the single row `x` on the card, left there, by the steps of
        /// [`Self::logits`].
        pub(crate) fn logits_dev(&self, x: &CudaView<'_, f32>) -> Result<CudaSlice<f32>> {
            if x.len() != self.inp {
                return Err(Error::msg(format!(
                    "a row of {} values for a router of {}",
                    x.len(),
                    self.inp
                )));
            }
            let dev: &CudaDevice = &self.dev;
            let y = alloc_f32(dev, "router logits", self.out)?;
            let func = dev.quantized_fn("seq_row_f32")?;
            let inp = self.inp as i32;
            let mut b = dev.stream().launch_builder(&func);
            b.arg(&self.values);
            b.arg(x);
            b.arg(&y);
            b.arg(&inp);
            // A warp per output.
            let cfg = LaunchConfig {
                grid_dim: (self.out as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { b.launch(cfg) }.map_err(|e| Error::msg(format!("seq_row_f32 launch: {e}")))?;
            Ok(y)
        }

        /// The logits of the `n` rows `x` `[n, inp]` row-major: `[n, out]` row-major.
        ///
        /// Each row's logits take the steps of [`Self::logits`].
        /// The rows go up in groups of the rows a mixer works on at once.
        pub(crate) fn logits_rows(&self, x: &[f32], n: usize) -> Result<Vec<f32>> {
            self.logits_rows_in(x, n, super::super::hyper::mixer_block_rows())
        }

        /// [`Self::logits_rows`] with the rows going up `group` at a time.
        ///
        /// 1 buffer of `group` rows and 1 of their logits serve every group.
        /// Both go back to the card's pool when the call returns.
        pub(crate) fn logits_rows_in(&self, x: &[f32], n: usize, group: usize) -> Result<Vec<f32>> {
            if x.len() != n * self.inp || n == 0 {
                return Err(Error::msg(format!(
                    "{} values for {n} rows of a router of {}",
                    x.len(),
                    self.inp
                )));
            }
            let group = group.clamp(1, n);
            let dev: &CudaDevice = &self.dev;
            let stream = dev.stream();
            let mut rows = alloc_f32(dev, "router rows", group * self.inp)?;
            let y = alloc_f32(dev, "router logits", group * self.out)?;
            let func = dev.quantized_fn("seq_rows_f32")?;
            let mut logits = vec![0f32; n * self.out];
            for (part, at) in x
                .chunks(group * self.inp)
                .zip(logits.chunks_mut(group * self.out))
            {
                let count = part.len() / self.inp;
                stream
                    .memcpy_htod(part, &mut rows.slice_mut(0..part.len()))
                    .map_err(|e| Error::msg(format!("router rows upload: {e}")))?;
                let (inp, out, count_i) = (self.inp as i32, self.out as i32, count as i32);
                let mut b = stream.launch_builder(&func);
                b.arg(&self.values);
                b.arg(&rows);
                b.arg(&y);
                b.arg(&inp);
                b.arg(&out);
                b.arg(&count_i);
                // A thread per output, 16 rows by 16 outputs a block.
                let cfg = LaunchConfig {
                    grid_dim: (self.out.div_ceil(16) as u32, count.div_ceil(16) as u32, 1),
                    block_dim: (16, 16, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { b.launch(cfg) }
                    .map_err(|e| Error::msg(format!("seq_rows_f32 launch: {e}")))?;
                stream
                    .memcpy_dtoh(&y.slice(0..at.len()), at)
                    .map_err(|e| Error::msg(format!("router logits download: {e}")))?;
            }
            Ok(logits)
        }

        /// The logits' count.
        pub(crate) fn out(&self) -> usize {
            self.out
        }

        /// The card's seconds for the logits of the row `x` already on the card, as `timed`
        /// times them: the kernel and a wait for it, no crossing.
        pub(crate) fn time_on_card(
            &self,
            x: &[f32],
            timed: &mut dyn FnMut(
                &mut dyn FnMut() -> Result<()>,
            ) -> std::result::Result<f64, String>,
        ) -> std::result::Result<f64, String> {
            let text = |e: Error| e.to_string();
            let row = upload_f32(&self.dev, "router row", x).map_err(text)?;
            let view = row.slice(0..x.len());
            timed(&mut || {
                self.logits_dev(&view)?;
                self.dev
                    .stream()
                    .synchronize()
                    .map_err(|e| Error::msg(format!("router wait: {e}")))
            })
        }
    }
}
