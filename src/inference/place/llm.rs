//! Where a language model runs: the cards it is offered, the context its plan can hold, and
//! whether it runs tensor-parallel. The loader asks here and builds what it is told; it does
//! not read the cards itself.

use super::ladder::{on_one_card, Demand};

/// The owner a language model's placement reads the cards as.
const OWNER: &str = "llm";

/// The cards a language model is offered.
#[derive(Debug, Clone, Default)]
pub struct Offer {
    /// Cards fastest first, a pinned card first, each with the bytes it offers once the
    /// loader's staging is kept free on it.
    pub cards: Vec<(usize, u64)>,
    /// The card that holds the model alone: the pinned card when it does, else the first
    /// card that does. `None` when the model spreads.
    pub whole_on: Option<usize>,
}

impl Offer {
    /// The card the model's work is anchored on: the one holding it whole, else the first
    /// card offered.
    pub fn primary(&self) -> Option<usize> {
        self.whole_on
            .or_else(|| self.cards.first().map(|(idx, _)| *idx))
    }
}

/// The cards a language model is offered, read under the fleet's pressure. `whole` is what
/// one card must hold for the model to live on it alone, beside the loader's `staging`.
///
/// Every card an offer lists had its context made before it was measured, so a card reads
/// the same on every load: the fastest card first, and the others only once the model does
/// not sit whole on one card and the spread needs their budgets.
pub fn offer(pin: Option<usize>, whole: u64, staging: u64) -> Offer {
    let fastest = pin.or_else(|| super::room::fastest_card(OWNER));
    if let Some(card) = fastest {
        warm(card);
    }
    let cards = read(staging);
    let whole_on = holder(&cards, pin, whole);
    if whole_on.is_some() || cards.len() < 2 {
        return Offer {
            cards: super::ladder::prefer(&cards, pin),
            whole_on,
        };
    }
    for (idx, _) in &cards {
        warm(*idx);
    }
    let cards = read(staging);
    Offer {
        whole_on: holder(&cards, pin, whole),
        cards: super::ladder::prefer(&cards, pin),
    }
}

/// The card among `cards` that holds `whole`: the pinned card or none when one is pinned,
/// else the first that does.
fn holder(cards: &[(usize, u64)], pin: Option<usize>, whole: u64) -> Option<usize> {
    let fits = |&&(_, free): &&(usize, u64)| free >= whole && whole > 0;
    match pin {
        Some(p) => cards
            .iter()
            .filter(|(idx, _)| *idx == p)
            .find(fits)
            .map(|(idx, _)| *idx),
        None => cards.iter().find(fits).map(|(idx, _)| *idx),
    }
}

/// Free bytes each card offers after the pools are handed back, under the fleet's
/// pressure, less `staging`; cards offering nothing are left out.
fn read(staging: u64) -> Vec<(usize, u64)> {
    crate::inference::engine::llm_engine::release_cuda_pools();
    let boost = super::vram_manager::vram_reserve_boost();
    super::device_probe::probe_cuda_gpus_for(OWNER)
        .into_iter()
        .map(|g| {
            (
                g.index,
                g.stable_free.saturating_sub(boost).saturating_sub(staging),
            )
        })
        .filter(|(_, free)| *free > 0)
        .collect()
}

/// Makes the card's context and cuBLAS workspace, once per process.
fn warm(card: usize) {
    #[cfg(feature = "gpu")]
    crate::inference::engine::llm_engine::warm_card(card);
    #[cfg(not(feature = "gpu"))]
    let _ = card;
}

/// Blocks of `demand`'s first plan over `cards` that land on the host.
fn host_blocks(demand: &Demand, cards: &[(usize, u64)]) -> usize {
    demand
        .plan(cards)
        .segments
        .iter()
        .filter(|s| !matches!(s.kind, super::layer_executor::DeviceKind::Cuda(_)))
        .map(super::layer_executor::HeteroSegment::num_layers)
        .sum()
}

/// Cards the first plan of `demand` runs on.
fn cards_used(demand: &Demand, cards: &[(usize, u64)]) -> usize {
    super::ladder::cards_of(&demand.plan(cards)).len()
}

/// The context a model's cache is sized for: `full`, halved down to `floor` while the first
/// plan of `demand_at` that context would send blocks to the host, spread the model over more
/// cards than the plan at `floor` needs, or leave the weights a token reads on slower cards than
/// that plan does (by the cards' `bandwidth`, within a few percent). A context nobody opens costs
/// nothing; a block on the host and a hop between two cards cost every token.
pub fn context_the_cards_hold(
    full: u64,
    floor: u64,
    cards: &[(usize, u64)],
    bandwidth: impl Fn(usize) -> Option<f64>,
    demand_at: impl Fn(u64) -> Demand,
) -> u64 {
    if cards.is_empty() {
        return full;
    }
    let floor = floor.min(full);
    let fewest = cards_used(&demand_at(floor), cards);
    // Seconds per token (bytes over GB/s) of the plan a token takes at `ctx`.
    let token = |ctx: u64| -> Option<f64> {
        let demand = demand_at(ctx);
        let plan = demand.plan_for_one_stream_by(cards, &bandwidth);
        demand.token_cost(&plan, &bandwidth)
    };
    let best = token(floor);
    let slower = |ctx: u64| match (best, token(ctx)) {
        (Some(best), Some(now)) => now > best * 1.03,
        _ => false,
    };
    let mut chosen = full;
    while chosen > floor
        && (host_blocks(&demand_at(chosen), cards) > 0
            || cards_used(&demand_at(chosen), cards) > fewest
            || slower(chosen))
    {
        chosen = (chosen / 2).max(floor);
    }
    if chosen < full {
        let demand = demand_at(full);
        tracing::info!(
            "KV context {full} -> {chosen}: at {full}, {} of {} blocks would have gone to the host, the model would have spread over {} cards instead of {fewest}, or a token would have read its weights more slowly",
            host_blocks(&demand, cards),
            demand.layers,
            cards_used(&demand, cards)
        );
    }
    chosen
}

/// The cards that hold `demand` with every block split over the first `ranks` of `cards` in
/// proportion to `weight` of each card: chosen when the first plan does not hold it on one card
/// and each of them holds its share of the weights and of the cache with the runtime reserve
/// beside it.
pub fn tensor_parallel(
    demand: &Demand,
    cards: &[(usize, u64)],
    ranks: usize,
    weight: impl Fn(usize) -> f64,
) -> Option<Vec<usize>> {
    if ranks < 2 || cards.len() < ranks || on_one_card(&demand.plan(cards)) {
        return None;
    }
    let ranked = &cards[..ranks];
    let total: f64 = ranked.iter().map(|(idx, _)| weight(*idx)).sum();
    if total <= 0.0 {
        return None;
    }
    let held = demand
        .block_bytes
        .saturating_add(demand.head_bytes)
        .saturating_add(
            demand
                .kv_bytes_per_layer
                .saturating_mul(demand.layers as u64),
        );
    ranked
        .iter()
        .all(|(idx, free)| {
            let share = (held as f64 * weight(*idx) / total) as u64;
            share <= free.saturating_sub(demand.runtime_reserve)
        })
        .then(|| ranked.iter().map(|(idx, _)| *idx).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn demand(layers: usize, weights_gb: u64) -> Demand {
        Demand {
            layers,
            block_bytes: weights_gb * GB,
            runtime_reserve: GB,
            ..Demand::default()
        }
    }

    #[test]
    fn a_pinned_card_holds_the_model_only_when_it_has_the_room() {
        let cards = [(0, 10 * GB), (1, 20 * GB)];
        assert_eq!(holder(&cards, None, 12 * GB), Some(1));
        assert_eq!(holder(&cards, Some(0), 12 * GB), None);
        assert_eq!(holder(&cards, Some(1), 12 * GB), Some(1));
        assert_eq!(holder(&cards, None, 30 * GB), None);
    }

    #[test]
    fn the_context_halves_until_no_block_leaves_the_cards() {
        let cards = [(0, 11 * GB)];
        let at = |ctx: u64| Demand {
            kv_bytes_per_layer: ctx * (GB / 32768),
            ..demand(8, 8)
        };
        assert_eq!(
            context_the_cards_hold(32768, 4096, &cards, |_| None, at),
            8192
        );
        assert_eq!(
            context_the_cards_hold(32768, 4096, &[], |_| None, at),
            32768
        );
        assert_eq!(
            context_the_cards_hold(32768, 32768, &cards, |_| None, at),
            32768
        );
    }

    /// A model one card holds at a smaller context is not spread over two cards for a context
    /// nobody asked for: a hop between cards costs every token.
    #[test]
    fn the_context_halves_until_one_card_holds_the_model() {
        let cards = [(0, 12 * GB), (1, 12 * GB)];
        let at = |ctx: u64| Demand {
            kv_bytes_per_layer: ctx * (GB / 32768),
            ..demand(8, 8)
        };
        // At 131072 the cache is 32 GB over the 8 blocks: two cards hold it, one does not.
        assert_eq!(cards_used(&at(131072), &cards), 2);
        let held = context_the_cards_hold(131072, 4096, &cards, |_| None, at);
        assert_eq!(cards_used(&at(held), &cards), 1);
        assert!(held >= 4096 && held < 131072);
        // A model that needs two cards at the floor keeps its context.
        let big = |ctx: u64| Demand {
            kv_bytes_per_layer: ctx * (GB / 32768),
            ..demand(8, 20)
        };
        assert_eq!(
            context_the_cards_hold(8192, 4096, &cards, |_| None, big),
            8192
        );
    }

    /// Two cards hold the model at any context, but the larger one leaves fewer blocks on the
    /// faster card: the context gives way until the fast card holds as many as it can.
    #[test]
    fn the_context_gives_way_until_the_fast_card_holds_the_most() {
        let cards = [(0, 16 * GB), (1, 16 * GB)];
        let bandwidth = |idx: usize| Some(if idx == 0 { 900.0 } else { 450.0 });
        // 64 blocks of 0.375 GB; the cache of a block is 0.25 GB at 32768 keys.
        let at = |ctx: u64| Demand {
            layers: 64,
            block_bytes: 24 * GB,
            kv_bytes_per_layer: ctx * (GB / 4) / 32768,
            runtime_reserve: GB,
            ..Demand::default()
        };
        let on_fast = |ctx: u64| {
            at(ctx)
                .plan_for_one_stream_by(&cards, bandwidth)
                .segments
                .iter()
                .filter(|s| s.kind == super::super::layer_executor::DeviceKind::Cuda(0))
                .map(|s| s.num_layers())
                .sum::<usize>()
        };
        assert!(on_fast(131072) < on_fast(4096));
        let held = context_the_cards_hold(131072, 4096, &cards, bandwidth, at);
        assert!(held < 131072);
        let cost = |ctx: u64| {
            let d = at(ctx);
            d.token_cost(&d.plan_for_one_stream_by(&cards, bandwidth), &bandwidth)
                .unwrap()
        };
        assert!(cost(held) <= cost(4096) * 1.03);
        // Without bandwidths the speeds are unknown: only the host and the card count give way.
        assert!(context_the_cards_hold(131072, 4096, &cards, |_| None, at) >= held);
    }

    #[test]
    fn a_model_one_card_holds_is_not_split_by_rank() {
        let cards = [(0, 20 * GB), (1, 20 * GB)];
        assert_eq!(tensor_parallel(&demand(10, 8), &cards, 2, |_| 1.0), None);
    }

    #[test]
    fn two_cards_split_a_model_each_holds_half_of() {
        let cards = [(0, 16 * GB), (1, 16 * GB)];
        assert_eq!(
            tensor_parallel(&demand(10, 20), &cards, 2, |_| 1.0),
            Some(vec![0, 1])
        );
    }

    #[test]
    fn a_split_each_card_cannot_hold_its_half_of_is_refused() {
        let cards = [(0, 16 * GB), (1, 8 * GB)];
        assert_eq!(tensor_parallel(&demand(10, 20), &cards, 2, |_| 1.0), None);
        assert_eq!(
            tensor_parallel(&demand(10, 20), &cards[..1], 2, |_| 1.0),
            None
        );
        // Weighted to the larger card, the smaller one's share fits.
        assert_eq!(
            tensor_parallel(&demand(10, 20), &cards, 2, |i| if i == 0 {
                2.0
            } else {
                1.0
            }),
            Some(vec![0, 1])
        );
    }
}
