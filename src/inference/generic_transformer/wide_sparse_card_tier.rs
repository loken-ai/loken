//! The card tier of the routed experts of a Sinkhorn wide-residual model.
//!
//! After the load, each card the plan uses has a room for routed experts.
//! The room is what the card has free once the token mixers are on it, less the cache the plan
//! keeps for its layers and less the room a request's work takes.
//!
//! A card takes experts only when it pays, measured at the load:
//! - the card's variant of the experts has a kernel that gives the host's bits,
//! - on a decode's routed set of the first wide layer, over 1 row, the card gives the host's bits,
//! - the card runs that set in less time than the host's cores.
//!
//! The experts then fill the rooms in the order of the heat, rank by rank across the layers,
//! through the variant plan: the first card takes the hottest, the next card the next ones.
//! Each card keeps the precise variant of its experts under `Plan`.

use super::GenericHeteroTransformer;
use crate::inference::place::expert_variants::{order, VariantChoice, VariantPlan};
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};

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

    /// Reads, after the load, the room each card of `plan` has for routed experts.
    ///
    /// The room is what the card has free now, less `kv_bytes_per_layer` for each layer the plan
    /// gives it and less `runtime_reserve`.
    /// A card that reads no free memory, or a model of another kind, gets no room.
    pub fn note_wide_card_rooms(
        &mut self,
        plan: &HeteroPlan,
        kv_bytes_per_layer: u64,
        runtime_reserve: u64,
    ) {
        if self.wide_sparse.is_none() {
            return;
        }
        let mut rooms: Vec<(usize, u64)> = Vec::new();
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

    /// Places the routed experts: the cards that pay keep the hottest, each expert reads the
    /// variant `choice` gives it.
    ///
    /// A card that does not pay, by the measure of the load, keeps none and the log says why.
    /// Returns the plan, `None` for a model of another kind.
    pub fn place_wide_experts(&mut self, choice: VariantChoice) -> Option<VariantPlan> {
        self.wide_sparse.as_ref()?;
        let offered = self.wide_card_rooms().to_vec();
        let paying: Vec<(usize, u64)> = offered
            .into_iter()
            .filter(|&(ordinal, bytes)| match self.measure_wide_card(ordinal, choice) {
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
            .collect();
        self.set_wide_card_rooms(paying);
        let plan = match self.expert_variants() {
            1 => self.plan_wide_experts(choice)?,
            _ => self.place_expert_variants(choice)?,
        };
        self.keep_wide_card_experts(&plan);
        Some(plan)
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
