//! The card tier of the routed experts of a Sinkhorn wide-residual model.
//!
//! After the load, each card the load was offered has a room for routed experts.
//! The room of a card the plan uses is what the card has free once the token mixers are on it,
//! less the cache the plan keeps for its layers and less the room a request's work takes.
//! The room of a card outside the plan is what it has free less the room a request's work takes.
//! It keeps routed experts only, and it lends and takes back nothing.
//!
//! A card takes experts only when it pays, measured at the load:
//! - the card's variant of the experts has a kernel that gives the host's bits,
//! - on a decode's routed set of the first wide layer, over 1 row, the card gives the host's bits,
//! - the card runs that set in less time than the host's cores.
//!
//! The experts then fill the rooms in the order of the heat, rank by rank across the layers,
//! through the variant plan: the first card takes the hottest, the next card the next ones.
//! Each card keeps the precise variant of its experts under `Plan`.
//!
//! The routed experts and the weights every token reads also borrow the part of a card's cache
//! room that the working window leaves (`LentRoom`).
//! Before a forward, the cache takes back what the tokens after it need (`take_back_card_room`).
//! The card then drops what saves the least per byte by the load's measures: its coldest
//! experts and its cheapest weights every token reads.
//! The host runs each of them from then on, as it runs what the load leaves on the host.
//! The next load places them again.
//!
//! A card whose run of routed experts fails leaves them to the host for that step.
//! Before the next forward the card drops its coldest experts until the bytes the run asked
//! for are free (`release_after_card_failures`).

use super::wide_always_read::{expert_share, AlwaysRead};
use super::GenericHeteroTransformer;
use crate::inference::place::expert_variants::{order, VariantChoice, VariantPlan};
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};

/// The part of a card's cache room that the routed experts borrow.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LentRoom {
    pub ordinal: usize,
    /// Layers whose cache the card keeps.
    pub layers: u64,
    /// Bytes the load reserved on the card for those layers' cache.
    pub reserved: u64,
    /// Bytes of that reservation the experts may hold now.
    pub lent: u64,
    /// Bytes the load leaves free on the card for a request's work.
    pub work: u64,
}

impl LentRoom {
    /// Bytes the card's cache may take once a sequence holds `tokens` tokens, with `per_layer`
    /// the cache of the largest layer at `tokens`.
    ///
    /// A growing card buffer holds up to `GROWTH` times its rows.
    /// The need never passes the reservation.
    pub fn need(&self, per_layer: u64) -> u64 {
        #[cfg(feature = "gpu")]
        let growth = crate::tensor::cuda::GROWTH as u64;
        #[cfg(not(feature = "gpu"))]
        let growth = 1;
        per_layer
            .saturating_mul(growth)
            .saturating_mul(self.layers)
            .min(self.reserved)
    }

    /// Bytes the cache must take back from the experts before a sequence holds the tokens whose
    /// largest layer's cache is `per_layer`.
    pub fn shortfall(&self, per_layer: u64) -> u64 {
        self.need(per_layer)
            .saturating_sub(self.reserved.saturating_sub(self.lent))
    }
}

/// What the load measured of a card against the host's cores.
#[derive(Debug, Clone, PartialEq)]
pub struct CardMeasure {
    /// Experts timed: a decode's routed set of the first wide layer.
    pub experts: usize,
    /// Seconds the host's cores took for them over 1 row.
    pub host_secs: f64,
    /// Seconds the card took for them over 1 row: the row up, the kernels, the outputs down.
    pub card_secs: f64,
}

impl CardMeasure {
    /// Whether the card runs the routed set in less time than the host.
    pub fn pays(&self) -> bool {
        self.card_secs < self.host_secs
    }
}

impl GenericHeteroTransformer {
    /// Per card ordinal, the bytes the routed experts may take on it.
    pub fn wide_card_rooms(&self) -> &[(usize, u64)] {
        self.wide_sparse
            .as_ref()
            .map_or(&[], |ws| ws.card_rooms.as_slice())
    }

    /// Sets the bytes the routed experts may take on each card, cards in ordinal order.
    ///
    /// A model of another kind is left as it is.
    pub fn set_wide_card_rooms(&mut self, mut rooms: Vec<(usize, u64)>) {
        rooms.retain(|&(_, bytes)| bytes > 0);
        rooms.sort_unstable();
        if let Some(ws) = self.wide_sparse.as_mut() {
            ws.card_rooms = rooms;
        }
    }

    /// Reads, after the load, the room each card of `plan` and each card of `offered` has for
    /// routed experts.
    ///
    /// The room is what the card has free now, less `kv_bytes_per_layer` for each layer the plan
    /// gives it and less `runtime_reserve`.
    /// A card of `offered` outside the plan keeps no cache: its room is what it has free less
    /// `runtime_reserve`, and it lends nothing.
    /// A card that reads no free memory, or a model of another kind, gets no room.
    pub fn note_wide_card_rooms(
        &mut self,
        plan: &HeteroPlan,
        offered: &[usize],
        kv_bytes_per_layer: u64,
        runtime_reserve: u64,
    ) {
        if self.wide_sparse.is_none() {
            return;
        }
        let mut rooms: Vec<(usize, u64)> = Vec::new();
        let mut lent: Vec<LentRoom> = Vec::new();
        for s in &plan.segments {
            let DeviceKind::Cuda(ordinal) = s.kind else {
                continue;
            };
            let layers = s.layer_end.saturating_sub(s.layer_start) as u64;
            let cache = kv_bytes_per_layer.saturating_mul(layers);
            match rooms.iter_mut().find(|(o, _)| *o == ordinal) {
                Some((_, kept)) => *kept = kept.saturating_add(cache),
                None => rooms.push((ordinal, cache)),
            }
            match lent.iter_mut().find(|l| l.ordinal == ordinal) {
                Some(l) => {
                    l.layers += layers;
                    l.reserved = l.reserved.saturating_add(cache);
                }
                None => lent.push(LentRoom {
                    ordinal,
                    layers,
                    reserved: cache,
                    lent: 0,
                    work: runtime_reserve,
                }),
            }
        }
        if let Some(ws) = self.wide_sparse.as_mut() {
            ws.lent = lent;
        }
        for &ordinal in offered {
            if !rooms.iter().any(|&(o, _)| o == ordinal) {
                rooms.push((ordinal, 0));
            }
        }
        let rooms = rooms
            .into_iter()
            .filter_map(|(ordinal, cache)| {
                let dev = crate::tensor::Device::new_cuda(ordinal).ok()?;
                let free = crate::inference::place::vram_manager::free_now(&dev)?;
                let room = free.saturating_sub(cache).saturating_sub(runtime_reserve);
                tracing::info!(
                    "card {ordinal}: {} MB free after the load, {} MB kept for the cache, {} MB for a request's work: {} MB for routed experts",
                    free / 1_000_000,
                    cache / 1_000_000,
                    runtime_reserve / 1_000_000,
                    room / 1_000_000
                );
                Some((ordinal, room))
            })
            .collect();
        self.set_wide_card_rooms(rooms);
    }

    /// Per held wide layer, per card ordinal, the experts the cards keep from the load on.
    ///
    /// They are the experts `plan` gives that card that the heat ranks.
    /// Each layer's card experts are a prefix of its heat order, the first card's first.
    pub fn wide_card_experts_of(&self, plan: &VariantPlan) -> Vec<Vec<(usize, Vec<usize>)>> {
        let rooms = self.wide_card_rooms();
        let ranks = self.held_heat();
        let experts = |l: usize| plan.chosen.get(l).map_or(0, Vec::len);
        (0..plan.chosen.len())
            .map(|l| {
                let ids = order(&ranks, l, experts(l));
                // The order starts with the experts the heat ranks.
                let ranked: std::collections::HashSet<usize> = ranks
                    .get(l)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|&e| e < experts(l))
                    .collect();
                let mut at = 0usize;
                rooms
                    .iter()
                    .enumerate()
                    .filter_map(|(k, &(ordinal, _))| {
                        let n = plan.kept.get(k).and_then(|t| t.get(l)).copied()?;
                        let mine: Vec<usize> = ids[at..at + n]
                            .iter()
                            .copied()
                            .filter(|e| ranked.contains(e))
                            .collect();
                        at += n;
                        (!mine.is_empty()).then_some((ordinal, mine))
                    })
                    .collect()
            })
            .collect()
    }

    /// Experts the cards keep, over every held layer.
    pub fn wide_card_expert_count(&self) -> usize {
        self.layers
            .iter()
            .filter_map(|l| l.sinkhorn.as_ref())
            .map(|b| b.ffn.card.count())
            .sum()
    }

    /// Places the weights every token reads, then the routed experts: the cards that pay keep
    /// the hottest, each expert reads the variant `choice` gives it, and the host tier keeps the
    /// hottest of the others (`keep_wide_host_experts`).
    ///
    /// The weights every token reads take the part of the rooms, lent room included, that saves
    /// more per byte than the experts (`place_always_read`).
    ///
    /// A card that does not pay, by the measure of the load, keeps none and the log says why.
    /// Returns the plan, `None` for a model of another kind.
    pub fn place_wide_experts(&mut self, choice: VariantChoice) -> Option<VariantPlan> {
        self.wide_sparse.as_ref()?;
        // The lent room joins each card's room before the weights every token reads and the
        // experts share it, by measurement.
        self.lend_card_room();
        self.place_always_read(choice);
        let offered = self.wide_card_rooms().to_vec();
        let measured: Vec<(usize, u64, std::result::Result<CardMeasure, String>)> = offered
            .into_iter()
            .map(|(ordinal, bytes)| (ordinal, bytes, self.measure_wide_card(ordinal, choice)))
            .collect();
        // A card that keeps no weight every token reads notes the experts' saving here.
        if let Some(ws) = self.wide_sparse.as_mut() {
            for (ordinal, _, m) in &measured {
                let Ok(m) = m else { continue };
                if m.experts == 0 || ws.savings.iter().any(|s| s.ordinal == *ordinal) {
                    continue;
                }
                ws.savings.push(super::wide_always_read::CardSaving {
                    ordinal: *ordinal,
                    kinds: Vec::new(),
                    per_expert: (m.host_secs - m.card_secs) / m.experts as f64,
                });
            }
        }
        let paying: Vec<(usize, u64)> = measured
            .into_iter()
            .filter(|(ordinal, bytes, m)| match m {
                Ok(m) if m.pays() => {
                    tracing::info!(
                        "card {ordinal} keeps routed experts in {} MB: {} experts of 1 row in {:.3} ms on the card, {:.3} ms on the host",
                        bytes / 1_000_000,
                        m.experts,
                        m.card_secs * 1e3,
                        m.host_secs * 1e3
                    );
                    true
                }
                Ok(m) => {
                    tracing::info!(
                        "card {ordinal} keeps no routed expert: {} experts of 1 row take {:.3} ms on the card, {:.3} ms on the host",
                        m.experts,
                        m.card_secs * 1e3,
                        m.host_secs * 1e3
                    );
                    false
                }
                Err(why) => {
                    tracing::info!("card {ordinal} keeps no routed expert: {why}");
                    false
                }
            })
            .map(|(ordinal, bytes, _)| (ordinal, bytes))
            .collect();
        self.set_wide_card_rooms(paying);
        let plan = match self.expert_variants() {
            1 => self.plan_wide_experts(choice)?,
            _ => self.place_expert_variants(choice)?,
        };
        self.keep_wide_card_experts(&plan);
        // The host keeps the hottest of the experts the cards do not.
        self.keep_wide_host_experts();
        Some(plan)
    }

    /// Adds to each card's room for routed experts the part of its cache room that the working
    /// window leaves, and notes it as lent.
    ///
    /// Only a model that holds every layer of its stack lends: its forwards see every token.
    pub(super) fn lend_card_room(&mut self) {
        let whole = self
            .wide_sparse
            .as_ref()
            .is_some_and(|ws| self.first_layer == 0 && self.layers.len() == ws.meta.n_layers);
        let Some(ws) = self.wide_sparse.as_mut() else {
            return;
        };
        if !whole {
            ws.lent.clear();
            return;
        }
        let working = ws
            .meta
            .kv_bytes_per_layer(crate::inference::cache::KV_WORKING_WINDOW_TOKENS);
        let mut rooms = ws.card_rooms.clone();
        for l in &mut ws.lent {
            l.lent = l.reserved.saturating_sub(l.need(working));
            if l.lent == 0 {
                continue;
            }
            match rooms.iter_mut().find(|(o, _)| *o == l.ordinal) {
                Some((_, room)) => *room = room.saturating_add(l.lent),
                None => rooms.push((l.ordinal, l.lent)),
            }
            tracing::info!(
                "card {}: {} MB of the {} MB kept for the cache lent to routed experts until a sequence passes {} tokens",
                l.ordinal,
                l.lent / 1_000_000,
                l.reserved / 1_000_000,
                crate::inference::cache::KV_WORKING_WINDOW_TOKENS
            );
        }
        self.set_wide_card_rooms(rooms);
    }

    /// Takes back, before a forward after which a sequence holds `tokens` tokens, the cache room
    /// the routed experts borrow and the tokens need.
    ///
    /// Each card short of room drops what saves the least per byte (`release_card_room`) until
    /// the freed bytes cover the shortfall or the card keeps nothing it can drop.
    pub fn take_back_card_room(&mut self, tokens: usize) {
        self.release_after_card_failures();
        let Some(ws) = self.wide_sparse.as_ref() else {
            return;
        };
        if ws.lent.iter().all(|l| l.lent == 0) {
            return;
        }
        let per_layer = ws.meta.kv_bytes_per_layer(tokens);
        let mut lent = ws.lent.clone();
        for l in &mut lent {
            let short = l.shortfall(per_layer);
            if short == 0 {
                continue;
            }
            let (freed, dropped, weights) = self.release_card_room(l.ordinal, short);
            l.lent = match freed >= short {
                true => l.lent.saturating_sub(freed),
                false => 0,
            };
            let listed = AlwaysRead::ALL
                .iter()
                .map(|&k| (k, weights.iter().filter(|&&x| x == k).count()))
                .filter(|&(_, n)| n > 0)
                .map(|(k, n)| format!("{} {n}", k.name()))
                .collect::<Vec<_>>()
                .join(", ");
            tracing::info!(
                "card {}: the cache takes back {} MB lent at {tokens} tokens: {dropped} experts dropped, {} kept, always-read weights dropped: {}, {} MB still lent",
                l.ordinal,
                freed / 1_000_000,
                self.wide_card_expert_count(),
                if listed.is_empty() { "none" } else { listed.as_str() },
                l.lent / 1_000_000
            );
        }
        if let Some(ws) = self.wide_sparse.as_mut() {
            ws.lent = lent;
        }
    }

    /// Drops, on each card whose run of routed experts failed since the last call, what saves
    /// the least per byte (`release_card_room`) until the most bytes a failed run asked for are
    /// free.
    ///
    /// The host runs what the cards drop from then on.
    pub fn release_after_card_failures(&mut self) {
        let mut short: Vec<(usize, u64)> = Vec::new();
        for (ordinal, bytes) in self
            .layers
            .iter()
            .filter_map(|l| l.sinkhorn.as_ref())
            .flat_map(|b| b.ffn.card.take_short())
        {
            match short.iter_mut().find(|(o, _)| *o == ordinal) {
                Some((_, b)) => *b = (*b).max(bytes),
                None => short.push((ordinal, bytes)),
            }
        }
        for (ordinal, want) in short {
            let (freed, dropped, weights) = self.release_card_room(ordinal, want);
            tracing::warn!(
                "card {ordinal}: a run of routed experts that asked for {} MB failed: {} MB freed, {dropped} experts and {} always-read weights dropped, {} experts kept on the cards",
                want / 1_000_000,
                freed / 1_000_000,
                weights.len(),
                self.wide_card_expert_count()
            );
        }
    }

    /// Makes room on the cards before a prompt that runs a sequence from `from` to `to` tokens.
    ///
    /// The token mixers drop their card copies of a decode's state first: the next decode
    /// rebuilds them from the host state.
    /// Then each card gives back the lent room it lacks to hold the request's work and the
    /// cache those tokens add.
    /// The card's free bytes are read now: the driver's free bytes and the bytes its memory pool
    /// reserves without using.
    /// They count whatever the card holds beside the tokens of this sequence.
    /// Each card short of free bytes drops what saves the least per byte (`release_card_room`)
    /// until the freed bytes cover the shortfall or nothing is lent.
    pub fn give_back_for_prompt(&mut self, from: usize, to: usize) {
        let Some(ws) = self.wide_sparse.as_ref() else {
            return;
        };
        self.layers.iter().for_each(|l| l.mixer.drop_card_copies());
        if ws.lent.iter().all(|l| l.lent == 0) {
            return;
        }
        let (start, end) = (
            ws.meta.kv_bytes_per_layer(from),
            ws.meta.kv_bytes_per_layer(to),
        );
        let mut lent = ws.lent.clone();
        for l in &mut lent {
            if l.lent == 0 {
                continue;
            }
            let Ok(dev) = crate::tensor::Device::new_cuda(l.ordinal) else {
                continue;
            };
            let Some(driver) = crate::inference::place::vram_manager::free_now(&dev) else {
                continue;
            };
            let (reserved, used) = card_pool_use(l.ordinal);
            let free = driver.saturating_add(reserved.saturating_sub(used));
            let grows = l.need(end).saturating_sub(l.need(start));
            let short = l
                .work
                .saturating_add(grows)
                .saturating_sub(free)
                .min(l.lent);
            let (freed, dropped, weights) = match short {
                0 => (0, 0, Vec::new()),
                _ => self.release_card_room(l.ordinal, short),
            };
            l.lent = match short {
                0 => l.lent,
                _ => l.lent.saturating_sub(freed.max(short)),
            };
            tracing::info!(
                "card {}: a prompt from {from} to {to} tokens finds {} MB free ({} MB by the driver, the pool holds {} MB with {} MB in use) for {} MB of work and {} MB of cache: {} MB of lent room given back, {dropped} experts and {} always-read weights dropped, {} MB still lent",
                l.ordinal,
                free / 1_000_000,
                driver / 1_000_000,
                reserved / 1_000_000,
                used / 1_000_000,
                l.work / 1_000_000,
                grows / 1_000_000,
                freed / 1_000_000,
                weights.len(),
                l.lent / 1_000_000
            );
        }
        if let Some(ws) = self.wide_sparse.as_mut() {
            ws.lent = lent;
        }
    }

    /// Gives back every byte of lent room: the cards drop what they hold in it.
    ///
    /// A prompt that ran out of card memory runs again after it.
    pub fn give_back_all_lent_room(&mut self) {
        let Some(ws) = self.wide_sparse.as_ref() else {
            return;
        };
        let mut lent = ws.lent.clone();
        for l in &mut lent {
            if l.lent == 0 {
                continue;
            }
            let (freed, dropped, weights) = self.release_card_room(l.ordinal, l.lent);
            tracing::warn!(
                "card {}: out of memory in a prompt: {} MB of lent room given back, {dropped} experts and {} always-read weights dropped",
                l.ordinal,
                freed / 1_000_000,
                weights.len()
            );
            l.lent = 0;
        }
        if let Some(ws) = self.wide_sparse.as_mut() {
            ws.lent = lent;
        }
    }

    /// Drops from card `ordinal` what saves the least host time per byte, until `want` bytes
    /// are freed or the card keeps nothing it can drop: the bytes freed, the experts dropped and
    /// the kinds of the weights every token reads dropped.
    ///
    /// The candidates are each layer's coldest expert on the card and each weight every token
    /// reads that the card keeps.
    /// A weight saves its kind's measure of the load (`CardSaving`).
    /// An expert saves its share of the tokens times the load's measure of an expert, per byte.
    /// At an equal saving an expert goes before a weight, the reverse of the load's split.
    /// Of equal experts the layer with the most experts on the card goes first.
    /// Of equal weights the last the load offered goes first.
    fn release_card_room(&mut self, ordinal: usize, want: u64) -> (u64, usize, Vec<AlwaysRead>) {
        let saving = self
            .wide_sparse
            .as_ref()
            .and_then(|ws| ws.savings.iter().find(|s| s.ordinal == ordinal).cloned())
            .unwrap_or_default();
        let used = self.routed_per_token();
        let ranks = self.held_heat();
        let (mut freed, mut dropped) = (0u64, 0usize);
        let mut weights: Vec<AlwaysRead> = Vec::new();
        // Weights the card cannot drop now: a forward in flight shares the mixer's card.
        let mut stuck: Vec<(AlwaysRead, usize)> = Vec::new();
        while freed < want {
            let experts = self.coldest_experts_on(ordinal, &ranks, used, saving.per_expert);
            let cheapest_expert = experts
                .iter()
                .min_by(|a, b| {
                    a.0.total_cmp(&b.0)
                        .then_with(|| (b.1, b.2).cmp(&(a.1, a.2)))
                })
                .copied();
            let cheapest_weight = self
                .always_read_on(ordinal)
                .into_iter()
                .filter(|w| !stuck.contains(w))
                .enumerate()
                .map(|(at, (k, il))| (saving.of(k), at, k, il))
                .min_by(|a, b| a.0.total_cmp(&b.0).then_with(|| b.1.cmp(&a.1)));
            let expert_first = match (cheapest_expert, cheapest_weight) {
                (Some(e), Some(w)) => e.0 <= w.0,
                (Some(_), None) => true,
                (None, Some(_)) => false,
                (None, None) => break,
            };
            if expert_first {
                let Some((_, _, il)) = cheapest_expert else {
                    break;
                };
                let released = self.layers[il]
                    .sinkhorn
                    .as_mut()
                    .and_then(|b| b.ffn.card.release_coldest(ordinal));
                match released {
                    Some(bytes) => {
                        freed += bytes;
                        dropped += 1;
                    }
                    None => break,
                }
                continue;
            }
            let Some((_, _, kind, il)) = cheapest_weight else {
                break;
            };
            match self.release_always_read(kind, il) {
                Some(bytes) => {
                    freed += bytes;
                    weights.push(kind);
                }
                None => stuck.push((kind, il)),
            }
        }
        (freed, dropped, weights)
    }

    /// Per held layer whose card `ordinal` keeps experts, the saving per byte of the coldest of
    /// them, the experts the card keeps of the layer and the layer's index.
    ///
    /// The saving is the expert's share of the tokens times `per_expert`, over its bytes.
    /// The share is the routing profile's when it counted the layer, else the bound of its heat
    /// rank in `ranks` (`expert_share`).
    fn coldest_experts_on(
        &self,
        ordinal: usize,
        ranks: &[Vec<usize>],
        used: usize,
        per_expert: f64,
    ) -> Vec<(f64, usize, usize)> {
        let shares = self.wide_sparse.as_ref().and_then(|ws| ws.shares.as_ref());
        let mut out = Vec::new();
        let mut held = 0usize;
        for (il, layer) in self.layers.iter().enumerate() {
            let Some(block) = layer.sinkhorn.as_ref() else {
                continue;
            };
            let h = held;
            held += 1;
            let ffn = &block.ffn;
            let n = ffn.card.count_on(ordinal);
            let Some(e) = ffn.card.coldest_on(ordinal).filter(|_| n > 0) else {
                continue;
            };
            let counted = shares
                .and_then(|s| s.get(self.first_layer + il))
                .and_then(Option::as_ref)
                .and_then(|c| c.get(e).copied());
            let rank = order(ranks, h, ffn.experts.len())
                .iter()
                .position(|&x| x == e)
                .unwrap_or(ffn.experts.len());
            let share = expert_share(counted, rank, used);
            let bytes = ffn.variant_bytes(e, 0).unwrap_or(0).max(1);
            out.push((share * per_expert / bytes as f64, n, il));
        }
        out
    }

    /// Uploads the experts `plan` gives the cards, each in the variant the layer reads.
    ///
    /// An upload that fails stops the tier there: the experts not uploaded run on the host.
    pub fn keep_wide_card_experts(&mut self, plan: &VariantPlan) {
        #[cfg(feature = "gpu")]
        {
            let per_layer = self.wide_card_experts_of(plan);
            let held: Vec<usize> = (0..self.layers.len())
                .filter(|&i| self.layers[i].sinkhorn.is_some())
                .collect();
            let started = std::time::Instant::now();
            'layers: for (l, cards) in per_layer.into_iter().enumerate() {
                let Some(block) = self.layers[held[l]].sinkhorn.as_mut() else {
                    continue;
                };
                let ffn = &mut block.ffn;
                for (ordinal, ids) in cards {
                    let experts: Vec<(usize, &super::block_scaled_experts::BlockScaledExpert)> =
                        ids.iter().map(|&e| (e, &ffn.experts[e])).collect();
                    if let Err(e) = ffn.card.keep(ordinal, &experts) {
                        tracing::warn!(
                            "card {ordinal}: routed experts of layer {} not kept, the rest run on the host: {e}",
                            self.first_layer + held[l]
                        );
                        break 'layers;
                    }
                }
            }
            let kept = self.wide_card_expert_count();
            if kept > 0 {
                tracing::info!(
                    "{kept} routed experts kept on the cards in {:.1}s",
                    started.elapsed().as_secs_f32()
                );
            }
        }
        #[cfg(not(feature = "gpu"))]
        let _ = plan;
    }

    /// Times a decode's routed set of the first wide layer on card `ordinal` and on the host,
    /// in the variant the card would keep under `choice`, and checks the card gives the host's
    /// bits.
    ///
    /// `Err` names why the card cannot take the experts.
    #[cfg(feature = "gpu")]
    pub fn measure_wide_card(
        &self,
        ordinal: usize,
        choice: VariantChoice,
    ) -> std::result::Result<CardMeasure, String> {
        use super::wide_expert_card::{host_outputs, served, CardExperts};
        let held = (0..self.layers.len())
            .find(|&i| self.layers[i].sinkhorn.is_some())
            .ok_or("the model holds no wide layer")?;
        let block = self.layers[held].sinkhorn.as_ref().ok_or("no wide layer")?;
        let ffn = &block.ffn;
        let ranks = self.held_heat();
        let ids: Vec<usize> = order(&ranks, 0, ffn.experts.len())
            .into_iter()
            .take(ffn.used)
            .collect();
        // The variant a card tier takes: the largest unless every expert reads the smallest.
        let largest = choice != VariantChoice::Smallest;
        let pick = |e: usize| {
            let stored = &ffn.variants.stored;
            if stored.is_empty() {
                return &ffn.experts[e];
            }
            let size = |v: usize| ffn.variant_bytes(e, v).unwrap_or(0);
            let vs = 0..stored.len();
            let v = match largest {
                true => vs.rev().max_by_key(|&v| size(v)).unwrap_or(0),
                false => vs.min_by_key(|&v| size(v)).unwrap_or(0),
            };
            &stored[v][e]
        };
        let experts: Vec<&super::block_scaled_experts::BlockScaledExpert> =
            ids.iter().map(|&e| pick(e)).collect();
        if experts.is_empty() || !experts.iter().all(|e| served(e)) {
            return Err(
                "no card kernel gives the host's bits for the format of its experts".into(),
            );
        }
        let dim = block.ffn_norm.elem_count();
        let row = block
            .ffn_norm
            .reshape((1, dim))
            .map_err(|e| e.to_string())?;
        let values = row
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| e.to_string())?;
        let limit = ffn.limit;
        let mut card = CardExperts::default();
        let numbered: Vec<(usize, &super::block_scaled_experts::BlockScaledExpert)> =
            experts.iter().copied().enumerate().collect();
        card.keep(ordinal, &numbered).map_err(|e| e.to_string())?;
        let slots: Vec<usize> = (0..experts.len()).collect();
        let rows = vec![vec![0usize]; experts.len()];
        // The first pass reads the bytes in and loads the kernels, the second is timed.
        let mut host = Vec::new();
        let mut host_secs = 0.0;
        let mut on_card = None;
        let mut card_secs = 0.0;
        for _ in 0..2 {
            let started = std::time::Instant::now();
            host = host_outputs(&experts, &row, limit).map_err(|e| e.to_string())?;
            host_secs = started.elapsed().as_secs_f64();
            let started = std::time::Instant::now();
            on_card = card.run(&slots, &rows, &values, dim, limit);
            card_secs = started.elapsed().as_secs_f64();
        }
        let on_card = on_card.ok_or("the card did not run the experts")?;
        let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<u32>>();
        if host.iter().zip(&on_card).any(|(h, c)| bits(h) != bits(c)) {
            return Err("the card's products differ from the host's bits".into());
        }
        Ok(CardMeasure {
            experts: experts.len(),
            host_secs,
            card_secs,
        })
    }

    /// A build without cards measures no card.
    #[cfg(not(feature = "gpu"))]
    pub fn measure_wide_card(
        &self,
        _ordinal: usize,
        _choice: VariantChoice,
    ) -> std::result::Result<CardMeasure, String> {
        Err("the build has no card".into())
    }
}

/// What the default memory pool of card `ordinal` reserves and what of it is in use.
fn card_pool_use(ordinal: usize) -> (u64, u64) {
    #[cfg(feature = "gpu")]
    return crate::tensor::cuda::pool_use(ordinal);
    #[cfg(not(feature = "gpu"))]
    {
        let _ = ordinal;
        (0, 0)
    }
}
