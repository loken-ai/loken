//! The host tier of the routed experts of a Sinkhorn wide-residual model.
//!
//! Each wide layer keeps up to a bound of its routed experts in host memory, in the pages of the
//! file's mapping.
//! The bound is the same for every layer.
//! It is the host memory the process may take, less a burst's headroom, less one layer's routed
//! experts and less a sequence's state at the open window, over one expert of every layer.
//! A bound that holds every expert of every layer builds no tier.
//!
//! The rows that route to an expert admit it, on a clock of 1 tick a row:
//! - a decoded row admits every expert it computes on the host,
//! - a prompt admits its experts at the tick of the last row that routed to each,
//! - an expert the tier does not keep leaves the page cache once computed.
//!
//! The kept expert that gives way is ranked 1 of 2 ways:
//! - by recency, the least recent first,
//! - by decoded count, the one the fewest decoded rows routed to since the load first, then the
//!   least recent.
//!
//! A shadow of each layer, which holds no page, ranks the other way.
//! At a prompt that follows a decode the tier takes the ranking that missed fewer of the decoded
//! rows' experts since the last such prompt.
//! It starts ranked by decoded count.
//!
//! The page cache then holds the kept experts and the experts in flight.
//!
//! The tier starts from the routing heat: the hottest experts the cards do not keep.
//! The pages the load read from the experts' files leave the page cache when the tier is built.
//! The kept experts' pages are read at their first use.
//!
//! A forward asks for the bytes of every host expert it computes once its routing is known, on
//! reader threads, the experts the layer does not keep first.
//! Its cores then wait only for the pages they compute from.
//!
//! A decoded row reads ahead the experts the next layer is expected to route to and does not keep.
//! The guess is the next layer's router over this layer's input.
//! The reads run while a measured token runs faster with them (`expert_readahead::ReadAhead`).
//! A guessed expert the next layer does not route to and does not keep leaves the page cache.
//!
//! The bound moves at a prompt that follows a decode, by the memory stalls the host reports
//! (`expert_host_tune::TierTune`).
//!
//! The tier moves pages, never a value a forward computes.

use super::expert_host_tune::TierTune;
use super::expert_readahead::{drop_later, drop_now, read_now, Asked, ReadAhead, Span};
use super::moe_weights::{sqrt_softplus_routed_choice, Truncation};
use super::wide_sparse_layer::RoutedFfn;
use super::GenericHeteroTransformer;
use crate::inference::expert_counters::{Found, HostExpert, TierEvent};
use crate::inference::place::expert_variants::order;
use crate::tensor::blockscaled::projection::Projection;
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The rows of a forward that count as decoded rows: a decode step or a verify of drafted rows.
pub(crate) fn decoded_rows(rows: usize) -> bool {
    rows <= crate::inference::engine::decode_step::VERIFY_ROWS
}

/// Which kept expert gives way first.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Ranking {
    /// The least recent.
    #[default]
    Recency,
    /// The one the fewest decoded rows routed to since the load, then the least recent.
    DecodedCount,
}

impl Ranking {
    /// The other ranking.
    pub(crate) fn other(self) -> Self {
        match self {
            Ranking::Recency => Ranking::DecodedCount,
            Ranking::DecodedCount => Ranking::Recency,
        }
    }
}

/// The kept experts of 1 layer, ordered by the clock of the rows that routed to them.
#[derive(Debug, Clone, Default)]
pub(crate) struct KeepSet {
    /// Per expert, the clock of the last row that routed to it, `None` when not kept.
    last: Vec<Option<u64>>,
    /// Per expert not kept, which kind of rows let it go, `NeverKept` before any.
    left: Vec<Found>,
    /// Per expert, the decoded rows that routed to it since the load, kept or not.
    decodes: Vec<u64>,
    ranking: Ranking,
    kept: usize,
    clock: u64,
}

impl KeepSet {
    /// No expert kept among `experts`, the least recent giving way first.
    pub(crate) fn new(experts: usize) -> Self {
        Self {
            last: vec![None; experts],
            left: vec![Found::NeverKept; experts],
            decodes: vec![0; experts],
            ranking: Ranking::Recency,
            kept: 0,
            clock: 0,
        }
    }

    /// The set, its experts giving way by `ranking` from now on.
    pub(crate) fn ranked(mut self, ranking: Ranking) -> Self {
        self.ranking = ranking;
        self
    }

    pub(crate) fn ranking(&self) -> Ranking {
        self.ranking
    }

    pub(crate) fn is_kept(&self, e: usize) -> bool {
        self.last.get(e).is_some_and(Option::is_some)
    }

    /// What the set holds of expert `e`: kept, or which kind of rows let it go.
    pub(crate) fn found(&self, e: usize) -> Found {
        match self.is_kept(e) {
            true => Found::Kept,
            false => self.left.get(e).copied().unwrap_or(Found::NeverKept),
        }
    }

    /// The kept experts, the least recent first.
    pub(crate) fn by_recency(&self) -> Vec<usize> {
        let mut order: Vec<(u64, usize)> = self
            .last
            .iter()
            .enumerate()
            .filter_map(|(e, c)| c.map(|c| (c, e)))
            .collect();
        order.sort_unstable();
        order.into_iter().map(|(_, e)| e).collect()
    }

    /// The kept experts, the first to give way first.
    fn by_rank(&self) -> Vec<usize> {
        let mut order = self.by_recency();
        if self.ranking == Ranking::DecodedCount {
            // A stable sort keeps the least recent first among equal counts.
            order.sort_by_key(|&e| self.decodes[e]);
        }
        order
    }

    /// Experts kept now.
    pub(crate) fn len(&self) -> usize {
        self.kept
    }

    /// Keeps the first `capacity` experts of `order`, the hottest first.
    ///
    /// The hottest counts as the most recently decoded.
    pub(crate) fn seed(&mut self, order: &[usize], capacity: usize) {
        let take: Vec<usize> = order
            .iter()
            .copied()
            .filter(|&e| e < self.last.len())
            .take(capacity)
            .collect();
        let n = take.len() as u64;
        for (rank, e) in take.into_iter().enumerate() {
            if self.last[e].is_none() {
                self.kept += 1;
            }
            self.last[e] = Some(self.clock + n - rank as u64);
        }
        self.clock += n;
    }

    /// A decoded row routed to `routed`: each is kept from this tick on.
    ///
    /// Returns the experts let go to keep at most `capacity`, never one of `routed`.
    pub(crate) fn decoded(&mut self, routed: &[usize], capacity: usize) -> Vec<usize> {
        self.clock += 1;
        for &e in routed {
            if let Some(slot) = self.last.get_mut(e) {
                if slot.is_none() {
                    self.kept += 1;
                }
                *slot = Some(self.clock);
                self.decodes[e] += 1;
            }
        }
        self.shrink_by(capacity, routed, Found::LeftAtDecode)
    }

    /// A prompt of `rows` rows routed to `routed`, each with the last row that routed to it: each
    /// is kept from the tick of that row, the rows ticking in order.
    ///
    /// Returns the experts let go to keep at most `capacity`, in the order they went.
    pub(crate) fn prompted(
        &mut self,
        routed: &[(usize, usize)],
        rows: usize,
        capacity: usize,
    ) -> Vec<usize> {
        let base = self.clock;
        for &(e, last) in routed {
            if let Some(slot) = self.last.get_mut(e) {
                if slot.is_none() {
                    self.kept += 1;
                }
                *slot = Some(base + last as u64 + 1);
            }
        }
        self.clock += rows as u64;
        self.shrink_by(capacity, &[], Found::LeftAtPrompt)
    }

    /// Lets the experts go in their ranking until at most `capacity` are kept.
    ///
    /// An expert of `spare` stays.
    /// Returns the experts let go, in the order they went.
    #[cfg(test)]
    pub(crate) fn shrink_to(&mut self, capacity: usize, spare: &[usize]) -> Vec<usize> {
        self.shrink_by(capacity, spare, Found::LeftAtDecode)
    }

    /// [`Self::shrink_to`], the experts let go noted as let go by rows of kind `by`.
    fn shrink_by(&mut self, capacity: usize, spare: &[usize], by: Found) -> Vec<usize> {
        let mut gone = Vec::new();
        if self.kept <= capacity {
            return gone;
        }
        let order: Vec<usize> = self
            .by_rank()
            .into_iter()
            .filter(|e| !spare.contains(e))
            .collect();
        for e in order {
            if self.kept <= capacity {
                break;
            }
            self.last[e] = None;
            self.left[e] = by;
            self.kept -= 1;
            gone.push(e);
        }
        gone
    }
}

/// The bytes of whole pages `spans` can make resident, whatever their alignment.
pub(crate) fn page_cost(spans: &[Span], page: usize) -> u64 {
    spans
        .iter()
        .map(|&(_, len)| ((len.div_ceil(page.max(1)) + 1) * page) as u64)
        .sum()
}

/// What 1 more expert of a layer takes in the host tier: the cost of the largest of `costs`
/// whose expert no card keeps.
///
/// An expert a card keeps may be read from a larger file: it takes no slot here.
pub(crate) fn host_slot(costs: &[u64], on_card: &dyn Fn(usize) -> bool) -> u64 {
    (0..costs.len())
        .filter(|&e| !on_card(e))
        .map(|e| costs[e])
        .max()
        .unwrap_or(0)
}

/// Experts each layer keeps when `room` bytes of host memory are left to the tier after
/// `reserve`, 1 more expert in every layer costing `step` bytes.
pub(crate) fn slots_for(room: u64, reserve: u64, step: u64) -> usize {
    usize::try_from(room.saturating_sub(reserve) / step.max(1)).unwrap_or(usize::MAX)
}

/// What every layer's tier shares: the bound, its tuning and the reads ahead.
pub struct HostTier {
    /// Experts each layer keeps at most.
    capacity: AtomicUsize,
    /// What one more expert in every layer takes, in whole pages.
    step: u64,
    /// The bound's tuning by the host's memory stalls.
    tune: Mutex<TierTune>,
    /// A decoded row ran since the bound was last looked at.
    decoded: AtomicBool,
    /// The reads ahead, `None` where they cannot run.
    readahead: Option<Arc<ReadAhead>>,
    /// The kept experts give way by their decoded count, else by recency.
    by_count: AtomicBool,
    /// Decoded rows' host experts missed since the bound was last looked at: by the tier, and by
    /// a shadow of each layer ranked the other way.
    missed: [AtomicU64; 2],
}

impl HostTier {
    /// A tier of `capacity` experts a layer, at most `ceiling`.
    ///
    /// `reserve` is what the host keeps outside the tier, `step` what 1 more expert in every
    /// layer takes.
    pub(crate) fn new(
        capacity: usize,
        ceiling: usize,
        reserve: u64,
        step: u64,
        readahead: Option<Arc<ReadAhead>>,
    ) -> Self {
        Self {
            capacity: AtomicUsize::new(capacity),
            step,
            tune: Mutex::new(TierTune::new(capacity, ceiling, reserve, step)),
            decoded: AtomicBool::new(false),
            readahead,
            by_count: AtomicBool::new(true),
            missed: Default::default(),
        }
    }

    /// Experts each layer keeps at most.
    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    /// How the kept experts give way now.
    pub(crate) fn ranking(&self) -> Ranking {
        match self.by_count.load(Ordering::Relaxed) {
            true => Ranking::DecodedCount,
            false => Ranking::Recency,
        }
    }

    /// A decoded row's host experts: `tier` the tier missed, `shadow` its shadow missed.
    pub(crate) fn note_missed(&self, tier: u64, shadow: u64) {
        self.missed[0].fetch_add(tier, Ordering::Relaxed);
        self.missed[1].fetch_add(shadow, Ordering::Relaxed);
    }

    /// Takes the other ranking when its shadow missed fewer experts than the tier since the
    /// last look.
    pub(crate) fn choose_ranking(&self) {
        let tier = self.missed[0].swap(0, Ordering::Relaxed);
        let shadow = self.missed[1].swap(0, Ordering::Relaxed);
        if shadow < tier {
            let next = self.ranking().other();
            self.by_count
                .store(next == Ranking::DecodedCount, Ordering::Relaxed);
            tracing::info!(
                "expert host tier: ranked by {next:?} from now on, {shadow} misses against {tier} since the last prompt"
            );
        }
    }

    /// Moves the bound by the memory stalls since the last look, and the ranking by the misses,
    /// once a decode ran since then.
    ///
    /// `room` is the host memory the process may take now, the tier's pages left out.
    fn look(&self, room: u64) {
        if !self.decoded.swap(false, Ordering::Relaxed) {
            return;
        }
        self.choose_ranking();
        let mut tune = self.tune.lock().unwrap_or_else(|e| e.into_inner());
        let before = tune.slots;
        let beyond = room.saturating_sub(before as u64 * self.step);
        if let Some(slots) = tune.look(beyond) {
            if slots != before {
                self.capacity.store(slots, Ordering::Relaxed);
                tracing::info!("expert host tier: {before} to {slots} experts a layer");
            }
        }
    }
}

/// The guess of a layer's routing that the layer before made for the row decoding now.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Guess {
    /// The experts the guess named.
    pub named: Vec<usize>,
    /// The experts of `named` that were read ahead.
    pub read: Vec<usize>,
}

/// How a guess compared with the routing of its layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct GuessCount {
    /// Experts routed.
    pub routed: u64,
    /// Experts routed that the guess named.
    pub named: u64,
    /// Experts routed that no tier kept.
    pub unkept: u64,
    /// Experts routed that no tier kept and the guess named.
    pub unkept_named: u64,
}

/// `guess` against the experts `routed` to, `held` saying which a tier kept.
///
/// Returns the count and the experts read ahead that were not routed to.
pub(crate) fn score_guess(
    guess: &Guess,
    routed: &[usize],
    held: impl Fn(usize) -> bool,
) -> (GuessCount, Vec<usize>) {
    let mut c = GuessCount::default();
    for &e in routed {
        let named = guess.named.contains(&e);
        c.routed += 1;
        c.named += u64::from(named);
        if !held(e) {
            c.unkept += 1;
            c.unkept_named += u64::from(named);
        }
    }
    let wasted = guess
        .read
        .iter()
        .copied()
        .filter(|e| !routed.contains(e))
        .collect();
    (c, wasted)
}

/// 1 layer's share of the host tier.
pub struct LayerKeep {
    tier: Arc<HostTier>,
    set: Mutex<KeepSet>,
    /// What the layer would keep ranked the other way, which holds no page.
    shadow: Mutex<KeepSet>,
    /// Per expert, the byte ranges of its projections in the file's mapping.
    spans: Vec<Vec<Span>>,
    /// Per expert, whether a card keeps it.
    on_card: Vec<bool>,
    /// The guess the layer before made for the row decoding now.
    guess: Mutex<Option<Guess>>,
    /// The experts the scored guess read ahead.
    ahead: Mutex<Vec<usize>>,
    /// The layer's place among the wide layers.
    index: usize,
    /// This layer's router, what the layer before guesses with.
    router: Projection,
    bias: Vec<f32>,
    used: usize,
    temperature: f32,
    normalize: bool,
    /// The truncation this layer's router chooses under, which the layer before guesses with.
    truncation: Mutex<Truncation>,
    /// The next wide layer's share.
    next: OnceLock<Arc<LayerKeep>>,
    /// Hits, misses and experts let go since the load.
    stats: [std::sync::atomic::AtomicU64; 3],
    /// The first wide layer: a decoded token starts here.
    first: bool,
    /// The last wide layer: a decoded token ends here.
    last: bool,
}

impl LayerKeep {
    /// The share of `ffn`'s layer in `tier`, keeping the hottest of `order` the cards do not keep.
    ///
    /// `index` is the layer's place among the wide layers, `last` whether it is the last of them.
    pub(crate) fn new(
        tier: Arc<HostTier>,
        ffn: &RoutedFfn,
        order: &[usize],
        index: usize,
        last: bool,
    ) -> Self {
        let experts = ffn.experts.len();
        let on_card: Vec<bool> = (0..experts).map(|e| ffn.card.holds(e)).collect();
        let spans = ffn.experts.iter().map(expert_spans).collect();
        let mut set = KeepSet::new(experts);
        let hottest: Vec<usize> = order.iter().copied().filter(|&e| !on_card[e]).collect();
        set.seed(&hottest, tier.capacity());
        let set = set.ranked(tier.ranking());
        let shadow = set.clone().ranked(set.ranking().other());
        Self {
            tier,
            set: Mutex::new(set),
            shadow: Mutex::new(shadow),
            spans,
            on_card,
            guess: Mutex::new(None),
            ahead: Mutex::new(Vec::new()),
            index,
            router: ffn.router.clone(),
            bias: ffn.bias.clone(),
            used: ffn.used,
            temperature: ffn.temperature,
            normalize: ffn.normalize,
            truncation: Mutex::new(ffn.lossy.truncation),
            next: OnceLock::new(),
            stats: Default::default(),
            first: index == 0,
            last,
        }
    }

    fn set(&self) -> std::sync::MutexGuard<'_, KeepSet> {
        self.set.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The tier this layer shares.
    pub fn tier(&self) -> &Arc<HostTier> {
        &self.tier
    }

    /// Experts the layer keeps now.
    pub fn kept_count(&self) -> usize {
        self.set().len()
    }

    /// How the layer's kept experts give way now, and how its shadow's do.
    #[cfg(test)]
    pub(crate) fn rankings(&self) -> (Ranking, Ranking) {
        let shadow = self.shadow.lock().unwrap_or_else(|e| e.into_inner());
        (self.set().ranking(), shadow.ranking())
    }

    /// Whether the layer keeps expert `e` now.
    pub fn keeps(&self, e: usize) -> bool {
        self.set().is_kept(e)
    }

    /// Per expert of the first `experts`, whether the layer keeps it now, read under 1 lock.
    pub(crate) fn kept_mask(&self, experts: usize) -> Vec<bool> {
        let set = self.set();
        (0..experts).map(|e| set.is_kept(e)).collect()
    }

    /// The guess the layer before made for the row decoding now, `None` when none waits.
    #[cfg(test)]
    pub(crate) fn guess(&self) -> Option<Guess> {
        self.guess.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Sets the truncation this layer's router chooses under.
    pub(crate) fn set_truncation(&self, t: Truncation) {
        *self.truncation.lock().unwrap_or_else(|e| e.into_inner()) = t;
    }

    /// Bytes of expert `e` in the mapping.
    pub(crate) fn expert_bytes(&self, e: usize) -> u64 {
        self.spans
            .get(e)
            .map_or(0, |s| s.iter().map(|&(_, len)| len as u64).sum())
    }

    /// Sets the share of the next wide layer, which this layer reads ahead for.
    pub(crate) fn link(&self, next: Arc<LayerKeep>) {
        let _ = self.next.set(next);
    }

    /// A forward of `rows` rows opens, routed to `routed` on the host and the cards.
    ///
    /// The first layer of a prompt that follows a decode moves the bound.
    /// A decoded row's guess is scored and its wasted reads leave the page cache.
    /// Returns the score and the experts whose pages left, `None` without a guess.
    pub(crate) fn opens(&self, routed: &[usize], rows: usize) -> Option<(GuessCount, Vec<usize>)> {
        if !decoded_rows(rows) {
            if self.first {
                let room = crate::inference::place::layer_executor::available_host_ram_bytes();
                self.tier.look(room);
            }
            return None;
        }
        if rows == 1 && self.first {
            if let Some(ra) = &self.tier.readahead {
                ra.token_starts(std::time::Instant::now());
            }
        }
        let guess = self.guess.lock().unwrap_or_else(|e| e.into_inner()).take();
        *self.ahead.lock().unwrap_or_else(|e| e.into_inner()) =
            guess.as_ref().map(|g| g.read.clone()).unwrap_or_default();
        let guess = guess?;
        let (count, wasted) = {
            let set = self.set();
            score_guess(&guess, routed, |e| {
                set.is_kept(e) || self.on_card.get(e).copied().unwrap_or(false)
            })
        };
        let counters = crate::inference::expert_counters::global();
        counters.guess(count.routed, count.named, count.unkept, count.unkept_named);
        let set = self.set();
        let gone: Vec<usize> = wasted.into_iter().filter(|&e| !set.is_kept(e)).collect();
        drop(set);
        self.drop_pages(&gone);
        Some((count, gone))
    }

    /// Whether each of `on_host` is kept now, in a forward of `rows` rows whose experts on the
    /// cards are `card`.
    ///
    /// `routes[i]` is the last row routed to `on_host[i]` and how many rows were.
    /// Counts a hit for each kept and a miss for each other.
    /// While the expert measure runs, a decoded row's experts are counted by what the tier held
    /// of them and by their bytes not in memory, and the forward joins the trace.
    pub(crate) fn kept(
        &self,
        on_host: &[usize],
        routes: &[(usize, usize)],
        card: &[usize],
        rows: usize,
    ) -> Vec<bool> {
        let mut set = self.set();
        let mut shadow = self.shadow.lock().unwrap_or_else(|e| e.into_inner());
        // The tier took the other ranking: the shadow follows the one it left.
        let ranking = self.tier.ranking();
        if set.ranking() != ranking {
            *shadow = set.clone();
            *set = std::mem::take(&mut *set).ranked(ranking);
        }
        let kept: Vec<bool> = on_host.iter().map(|&e| set.is_kept(e)).collect();
        let hits = kept.iter().filter(|&&k| k).count() as u64;
        self.stats[0].fetch_add(hits, Ordering::Relaxed);
        self.stats[1].fetch_add(kept.len() as u64 - hits, Ordering::Relaxed);
        if decoded_rows(rows) {
            let held = on_host.iter().filter(|&&e| shadow.is_kept(e)).count() as u64;
            self.tier
                .note_missed(kept.len() as u64 - hits, on_host.len() as u64 - held);
        }
        drop(shadow);
        let counters = crate::inference::expert_counters::global();
        if !counters.on() {
            return kept;
        }
        if counters.trace_starts(self.index) {
            counters.trace(TierEvent::Kept {
                layer: self.index,
                experts: set.by_recency(),
                capacity: self.tier.capacity(),
            });
        }
        let ahead = std::mem::take(&mut *self.ahead.lock().unwrap_or_else(|e| e.into_inner()));
        let decoded = decoded_rows(rows);
        let host: Vec<HostExpert> = on_host
            .iter()
            .enumerate()
            .map(|(i, &e)| {
                let found = set.found(e);
                let read = ahead.contains(&e);
                let absent = match decoded {
                    true => self.absent_bytes(e),
                    false => 0,
                };
                if decoded {
                    counters.found(found, read, absent);
                }
                let (last, routed) = routes.get(i).copied().unwrap_or((0, 0));
                HostExpert {
                    expert: e,
                    last,
                    rows: routed,
                    found,
                    absent,
                    ahead: read,
                }
            })
            .collect();
        counters.trace(TierEvent::Forward {
            layer: self.index,
            rows,
            card: card.to_vec(),
            host,
        });
        kept
    }

    /// Asks on reader threads for the bytes of `experts`, which a forward computes now.
    ///
    /// The experts the layer does not keep are asked first.
    /// At least as many threads serve the reads as read ahead, and as a row routes to experts.
    /// `None` where the tier reads nothing ahead: the caller asks for the bytes itself.
    pub(crate) fn read_now(&self, experts: &[usize]) -> Option<Asked> {
        let ra = self.tier.readahead.as_ref()?;
        let mut order = experts.to_vec();
        {
            let set = self.set();
            order.sort_by_key(|&e| set.is_kept(e));
        }
        let spans = order
            .iter()
            .filter_map(|&e| self.spans.get(e).cloned())
            .collect();
        Some(read_now(spans, ra.readers().max(self.used)))
    }

    /// Bytes of expert `e`'s pages not in memory now.
    fn absent_bytes(&self, e: usize) -> u64 {
        self.spans.get(e).map_or(0, |s| {
            s.iter()
                .map(|&(at, len)| crate::tensor::mapped::absent_bytes(at, len))
                .sum()
        })
    }

    /// Hits, misses and experts let go since the load.
    pub fn stats(&self) -> (u64, u64, u64) {
        let at = |i: usize| self.stats[i].load(Ordering::Relaxed);
        (at(0), at(1), at(2))
    }

    /// Reads ahead, from the input `x` `[1, hidden]` of a decoded row, the experts the next layer
    /// is expected to route to and does not keep, while the reads run.
    ///
    /// The guess is the choice the next layer's router makes under its truncation.
    /// The experts the cards and the next layer's share keep now count as in memory.
    pub(crate) fn read_ahead(&self, x: &Tensor) {
        let Some(ra) = self.tier.readahead.as_ref().filter(|ra| ra.reads()) else {
            return;
        };
        let Some(next) = self.next.get() else {
            return;
        };
        let Ok(logits) = next.router.apply_here(x).and_then(|t| t.to_vec2::<f32>()) else {
            return;
        };
        let Some(row) = logits.first() else {
            return;
        };
        let read: Vec<usize>;
        let named = {
            let t = *next.truncation.lock().unwrap_or_else(|e| e.into_inner());
            let set = next.set();
            let held = |e: usize| set.is_kept(e) || next.on_card.get(e).copied().unwrap_or(true);
            let (named, _, _) = sqrt_softplus_routed_choice(
                row,
                Some(next.bias.as_slice()),
                next.used,
                next.temperature,
                next.normalize,
                t,
                &held,
            );
            read = named.iter().copied().filter(|&e| !held(e)).collect();
            named
        };
        let counters = crate::inference::expert_counters::global();
        for &e in &read {
            if let Some(spans) = next.spans.get(e) {
                counters.read(true, next.expert_bytes(e));
                ra.read(spans.clone());
            }
        }
        *next.guess.lock().unwrap_or_else(|e| e.into_inner()) = Some(Guess { named, read });
    }

    /// A forward of `rows` rows computed `on_host` on the host, `last[i]` the last row routed to
    /// `on_host[i]`.
    ///
    /// Decoded rows admit them at this tick.
    /// Prompt rows admit them at the tick of their last row, so the newest stay.
    /// The experts the tier does not keep after it leave the page cache.
    /// Returns those experts.
    pub(crate) fn computed(&self, on_host: &[usize], last: &[usize], rows: usize) -> Vec<usize> {
        let capacity = self.tier.capacity();
        let routed: Vec<(usize, usize)> =
            on_host.iter().copied().zip(last.iter().copied()).collect();
        let admit = |set: &mut KeepSet| match decoded_rows(rows) {
            true => set.decoded(on_host, capacity),
            false => set.prompted(&routed, rows, capacity),
        };
        let gone = admit(&mut self.set());
        admit(&mut self.shadow.lock().unwrap_or_else(|e| e.into_inner()));
        if decoded_rows(rows) {
            self.tier.decoded.store(true, Ordering::Relaxed);
        }
        self.stats[2].fetch_add(gone.len() as u64, Ordering::Relaxed);
        self.drop_pages(&gone);
        if rows == 1 && self.last {
            if let Some(ra) = &self.tier.readahead {
                ra.token_ends(std::time::Instant::now());
            }
        }
        gone
    }

    /// Drops the pages of `experts` from the page cache, off this thread.
    fn drop_pages(&self, experts: &[usize]) {
        let spans: Vec<Span> = experts
            .iter()
            .filter_map(|&e| self.spans.get(e))
            .flatten()
            .copied()
            .collect();
        drop_later(spans);
    }
}

/// The byte ranges of an expert's 3 projections in the file's mapping.
pub(crate) fn expert_spans(e: &super::block_scaled_experts::BlockScaledExpert) -> Vec<Span> {
    [&e.gate, &e.up, &e.down]
        .into_iter()
        .flat_map(Projection::mapped_spans)
        .collect()
}

/// Reader threads for the reads ahead of experts of `unit` bytes whose first byte is at `at`:
/// the fewest that reach the rate of the device the file is on, measured now over the host's
/// cores, else `used`, the experts a row routes to.
fn readers_for(at: usize, unit: usize, used: usize) -> usize {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    crate::tensor::mapped::file_span(at, unit)
        .and_then(|(file, _)| super::expert_readahead::probe_readers(&file, unit, cores))
        .unwrap_or(used)
        .max(1)
}

impl GenericHeteroTransformer {
    /// Builds the host tier of the routed experts of a Sinkhorn wide-residual model, once the
    /// cards keep theirs.
    ///
    /// The pages the load read from the experts' files leave the page cache.
    /// Returns the experts each layer keeps, `None` when no tier is built: a model of another
    /// kind, experts the host does not read from a mapping, or a host that holds them all.
    pub fn keep_wide_host_experts(&mut self) -> Option<usize> {
        self.wide_sparse.as_ref()?;
        let page = crate::tensor::mapped::page_size()?;
        let held: Vec<usize> = (0..self.layers.len())
            .filter(|&i| self.layers[i].sinkhorn.is_some())
            .collect();
        let ffns: Vec<&RoutedFfn> = held
            .iter()
            .filter_map(|&i| self.layers[i].sinkhorn.as_ref().map(|b| &b.ffn))
            .collect();
        // Per layer, the whole pages of the largest expert the host reads, and of all of them.
        let mut step = 0u64;
        let mut transient = 0u64;
        let mut ceiling = 0usize;
        for ffn in &ffns {
            let costs: Vec<u64> = ffn
                .experts
                .iter()
                .map(|x| page_cost(&expert_spans(x), page))
                .collect();
            if costs.contains(&0) {
                return None;
            }
            let host: Vec<usize> = (0..ffn.experts.len())
                .filter(|&e| !ffn.card.holds(e))
                .collect();
            step += host_slot(&costs, &|e| ffn.card.holds(e));
            transient = transient.max(host.iter().map(|&e| costs[e]).sum());
            ceiling = ceiling.max(host.len());
        }
        if ffns.is_empty() || step == 0 {
            return None;
        }
        let state = self.sequence_state_bytes(self.kv_capacity()).unwrap_or(0);
        let headroom = crate::inference::place::layer_executor::host_burst_headroom_bytes();
        let reserve = headroom + transient + state;
        let room = crate::inference::place::layer_executor::available_host_ram_bytes();
        let slots = slots_for(room, reserve, step);
        if slots >= ceiling {
            tracing::info!(
                "expert host tier: none, the {:.1} GB of host memory left hold every routed expert",
                room as f64 / 1e9
            );
            return None;
        }
        // The mappings of the files the experts are read from.
        let mut files: Vec<Span> = ffns
            .iter()
            .flat_map(|f| f.experts.iter().flat_map(expert_spans))
            .filter_map(|(at, len)| crate::tensor::mapped::mapping_span(at, len))
            .collect();
        files.sort_unstable();
        files.dedup();
        let (first_at, unit, used) = {
            let f = ffns[0];
            let spans = f.experts.first().map(expert_spans).unwrap_or_default();
            let unit: usize = spans.iter().map(|&(_, len)| len).sum();
            (spans.first().map_or(0, |s| s.0), unit, f.used)
        };
        let readahead = (unit > 0).then(|| {
            Arc::new(ReadAhead::new(
                readers_for(first_at, unit, used),
                ffns.len(),
            ))
        });
        let readers = readahead.as_ref().map_or(0, |r| r.readers());
        let tier = Arc::new(HostTier::new(slots, ceiling, reserve, step, readahead));
        self.attach_host_tier(&tier);
        // What the load read from those files: the weights the cards took, the experts they keep.
        drop_now(&files);
        tracing::info!(
            "expert host tier: {slots} of {ceiling} routed experts a layer kept in host memory, {:.1} GB of {:.1} GB left after {:.1} GB for a burst, {:.1} GB for a layer's experts in flight and {:.1} GB for a sequence's state; {readers} threads read ahead the next layer's experts while a token runs faster with them",
            slots as f64 * step as f64 / 1e9,
            room as f64 / 1e9,
            headroom as f64 / 1e9,
            transient as f64 / 1e9,
            state as f64 / 1e9
        );
        Some(slots)
    }

    /// Gives every wide layer its share of `tier`, the layers linked in order.
    ///
    /// Each share starts with the hottest experts the cards do not keep.
    /// Returns the shares, the first wide layer's first.
    pub(crate) fn attach_host_tier(&mut self, tier: &Arc<HostTier>) -> Vec<Arc<LayerKeep>> {
        let ranks = self.held_heat();
        let held: Vec<usize> = (0..self.layers.len())
            .filter(|&i| self.layers[i].sinkhorn.is_some())
            .collect();
        let keeps: Vec<Arc<LayerKeep>> = held
            .iter()
            .enumerate()
            .filter_map(|(k, &i)| {
                let ffn = &self.layers[i].sinkhorn.as_ref()?.ffn;
                let order = order(&ranks, k, ffn.experts.len());
                Some(Arc::new(LayerKeep::new(
                    tier.clone(),
                    ffn,
                    &order,
                    k,
                    k + 1 == held.len(),
                )))
            })
            .collect();
        for pair in keeps.windows(2) {
            pair[0].link(pair[1].clone());
        }
        for (&i, keep) in held.iter().zip(&keeps) {
            if let Some(block) = self.layers[i].sinkhorn.as_mut() {
                block.ffn.host = Some(keep.clone());
            }
        }
        keeps
    }
}
