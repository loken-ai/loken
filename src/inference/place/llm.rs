//! Where a language model runs: the cards it is offered, the context its plan can hold, and
//! whether it runs tensor-parallel. The loader asks here and builds what it is told; it does
//! not read the cards itself.

use super::ladder::Demand;

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
    #[cfg(feature = "cuda")]
    crate::inference::engine::llm_engine::warm_card(card);
    #[cfg(not(feature = "cuda"))]
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

/// The context a model's cache is sized for: `full`, halved down to `floor` while the first
/// plan of `demand_at` that context would send blocks to the host. A context nobody opens
/// costs nothing, a block on the host costs every token.
pub fn context_the_cards_hold(
    full: u64,
    floor: u64,
    cards: &[(usize, u64)],
    demand_at: impl Fn(u64) -> Demand,
) -> u64 {
    if cards.is_empty() {
        return full;
    }
    let floor = floor.min(full);
    let mut chosen = full;
    while chosen > floor && host_blocks(&demand_at(chosen), cards) > 0 {
        chosen = (chosen / 2).max(floor);
    }
    if chosen < full {
        let demand = demand_at(full);
        tracing::info!(
            "KV context {full} -> {chosen}: at {full}, {} of {} blocks would have gone to the host",
            host_blocks(&demand, cards),
            demand.layers
        );
    }
    chosen
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
        assert_eq!(context_the_cards_hold(32768, 4096, &cards, at), 8192);
        assert_eq!(context_the_cards_hold(32768, 4096, &[], at), 32768);
        assert_eq!(context_the_cards_hold(32768, 32768, &cards, at), 32768);
    }
}
