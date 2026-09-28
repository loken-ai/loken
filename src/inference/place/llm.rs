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

/// A tensor-parallel placement: every block runs on each of `cards`, the feed-forward rows
/// split so the first card takes `ffn_share.0 / ffn_share.1` of them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorParallel {
    pub cards: Vec<usize>,
    pub ffn_share: (usize, usize),
}

/// Tensor parallelism over the first `ranks` cards, when the cards' speeds are read now.
pub fn tensor_parallel(
    demand: &Demand,
    cards: &[(usize, u64)],
    ranks: usize,
) -> Option<TensorParallel> {
    let speeds: Vec<(usize, u64)> = super::room::card_rooms()
        .into_iter()
        .map(|c| (c.index, c.perf))
        .collect();
    tensor_parallel_on(demand, cards, &speeds, ranks)
}

/// Tensor parallelism for `demand` over the first `ranks` of `cards`, chosen when the first
/// plan does not hold it on one card and each rank holds its share with the runtime reserve
/// and its part of the cache beside it. The first card takes feed-forward rows in proportion
/// to its speed among the ranks when that share fits, else an even share.
pub fn tensor_parallel_on(
    demand: &Demand,
    cards: &[(usize, u64)],
    speeds: &[(usize, u64)],
    ranks: usize,
) -> Option<TensorParallel> {
    if ranks < 2 || cards.len() < ranks || on_one_card(&demand.plan(cards)) {
        return None;
    }
    let ranked = &cards[..ranks];
    let weights = demand.block_bytes.saturating_add(demand.head_bytes);
    let cache = demand
        .kv_bytes_per_layer
        .saturating_mul(demand.layers as u64)
        / ranks as u64;
    let room: Vec<u64> = ranked
        .iter()
        .map(|(_, free)| {
            free.saturating_sub(demand.runtime_reserve)
                .saturating_sub(cache)
        })
        .collect();
    let speed = |idx: usize| {
        speeds
            .iter()
            .find(|(i, _)| *i == idx)
            .map_or(0, |(_, s)| *s)
    };
    let first = speed(ranked[0].0);
    let all: u64 = ranked.iter().map(|(idx, _)| speed(*idx)).sum();
    let fits = |num: u64, den: u64| {
        let first_share = weights.saturating_mul(num) / den;
        let rest = weights.saturating_sub(first_share) / (ranks as u64 - 1);
        first_share <= room[0] && room[1..].iter().all(|r| rest <= *r)
    };
    let by_speed = (first > 0 && all > first).then(|| reduced(first, all));
    let even = (1, ranks as u64);
    let share = by_speed
        .filter(|&(n, d)| fits(n, d))
        .or_else(|| fits(even.0, even.1).then_some(even))?;
    Some(TensorParallel {
        cards: ranked.iter().map(|(idx, _)| *idx).collect(),
        ffn_share: (share.0 as usize, share.1 as usize),
    })
}

/// `num / den` in lowest terms.
fn reduced(num: u64, den: u64) -> (u64, u64) {
    let (mut a, mut b) = (num, den);
    while b != 0 {
        (a, b) = (b, a % b);
    }
    let g = a.max(1);
    (num / g, den / g)
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

    #[test]
    fn a_model_one_card_holds_is_not_split_by_rank() {
        let cards = [(0, 20 * GB), (1, 20 * GB)];
        assert_eq!(
            tensor_parallel_on(&demand(10, 8), &cards, &[(0, 2), (1, 1)], 2),
            None
        );
    }

    #[test]
    fn the_faster_card_takes_the_larger_share_when_it_fits() {
        let cards = [(0, 16 * GB), (1, 16 * GB)];
        let tp = tensor_parallel_on(&demand(10, 20), &cards, &[(0, 2), (1, 1)], 2).unwrap();
        assert_eq!(tp.cards, vec![0, 1]);
        assert_eq!(tp.ffn_share, (2, 3));
    }

    #[test]
    fn a_speed_share_that_does_not_fit_falls_back_to_even_then_to_none() {
        let cards = [(0, 14 * GB), (1, 14 * GB)];
        let tp = tensor_parallel_on(&demand(10, 24), &cards, &[(0, 2), (1, 1)], 2).unwrap();
        assert_eq!(tp.ffn_share, (1, 2));
        assert_eq!(
            tensor_parallel_on(&demand(10, 30), &cards, &[(0, 2), (1, 1)], 2),
            None
        );
        assert_eq!(
            tensor_parallel_on(&demand(10, 20), &cards[..1], &[(0, 2)], 2),
            None
        );
    }
}
