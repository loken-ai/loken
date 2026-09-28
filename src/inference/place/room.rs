//! What the cards offer right now, for the callers that decide whether and where to load
//! before the planner places anything: eviction, migration, hand-over to a peer.

use super::device_probe::{probe_cuda_gpus, probe_cuda_gpus_for};
use super::ladder::Demand;
use super::CardRoom;

/// Every card as a move or an eviction judges it, fastest first.
pub fn card_rooms() -> Vec<CardRoom> {
    probe_cuda_gpus()
        .iter()
        .map(|g| CardRoom {
            index: g.index,
            perf: g.perf_score,
            available: g.available,
            total: g.total,
        })
        .collect()
}

/// Bytes the cards offer the blocks of `demand`, each card keeping the demand's runtime
/// reserve free: the roomiest card when one card could hold the whole demand, else all of
/// them together. A component one card can hold wants one card; one no card can hold wants
/// the room of all of them.
pub fn room_for(demand: &Demand, cards: &[CardRoom]) -> u64 {
    let offered = cards
        .iter()
        .map(|c| c.available.saturating_sub(demand.runtime_reserve));
    if cards.iter().any(|c| demand.whole() <= c.total) {
        offered.max().unwrap_or(0)
    } else {
        offered.sum()
    }
}

/// Each card's size, probed once: a card's size does not change, and a caller on a
/// periodic path must not pay a probe each time.
pub fn card_sizes() -> Vec<u64> {
    static SIZES: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    SIZES
        .get_or_init(|| probe_cuda_gpus().iter().map(|g| g.total).collect())
        .clone()
}

/// Whether one card has `bytes` free now.
pub fn a_card_holds(bytes: u64) -> bool {
    probe_cuda_gpus()
        .iter()
        .map(|g| g.available)
        .max()
        .unwrap_or(0)
        >= bytes
}

/// The fastest card for `owner`, after what other subsystems declared they are about to use.
pub fn fastest_card(owner: &str) -> Option<usize> {
    probe_cuda_gpus_for(owner).first().map(|g| g.index)
}

/// How the cards' free memory takes a component: whole on one card, and across all of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreeFit {
    pub whole: bool,
    pub across: bool,
}

/// How the cards, at what they have free now with the pools handed back, take `bytes`.
pub fn free_fit(bytes: u64) -> FreeFit {
    let free: Vec<u64> = super::vram_manager::probe(0)
        .into_iter()
        .map(|(_, f, _)| f)
        .collect();
    fit_of(bytes, &free)
}

fn fit_of(bytes: u64, free: &[u64]) -> FreeFit {
    FreeFit {
        whole: free.iter().any(|f| *f >= bytes),
        across: free.iter().sum::<u64>() >= bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    fn card(available: u64, total: u64) -> CardRoom {
        CardRoom {
            index: 0,
            perf: 0,
            available,
            total,
        }
    }

    fn demand(weights: u64, reserve: u64) -> Demand {
        Demand {
            layers: 1,
            block_bytes: weights,
            runtime_reserve: reserve,
            ..Demand::default()
        }
    }

    #[test]
    fn a_component_one_card_holds_is_offered_the_roomiest_card() {
        let cards = [card(6 * GB, 16 * GB), card(10 * GB, 16 * GB)];
        assert_eq!(room_for(&demand(8 * GB, GB), &cards), 9 * GB);
    }

    #[test]
    fn a_component_no_card_holds_is_offered_every_card() {
        let cards = [card(6 * GB, 16 * GB), card(10 * GB, 16 * GB)];
        assert_eq!(room_for(&demand(20 * GB, GB), &cards), 14 * GB);
    }

    #[test]
    fn free_memory_fits_whole_or_across() {
        assert_eq!(
            fit_of(8, &[10, 2]),
            FreeFit {
                whole: true,
                across: true
            }
        );
        assert_eq!(
            fit_of(11, &[10, 2]),
            FreeFit {
                whole: false,
                across: true
            }
        );
        assert_eq!(
            fit_of(13, &[10, 2]),
            FreeFit {
                whole: false,
                across: false
            }
        );
    }
}
