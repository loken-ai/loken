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

/// Every card as a load by `owner` is offered it, fastest first.
/// Its free memory is net of what other subsystems declared they are about to use.
pub fn card_rooms_for(owner: &str) -> Vec<CardRoom> {
    probe_cuda_gpus_for(owner)
        .iter()
        .map(|g| CardRoom {
            index: g.index,
            perf: g.perf_score,
            available: g.stable_free,
            total: g.total,
        })
        .collect()
}

/// What a model asks of the cards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CardNeed {
    /// Bytes the cards hold.
    pub bytes: u64,
    /// Bytes each card the model uses keeps free beside them.
    pub per_card: u64,
}

impl CardNeed {
    /// Bytes one card needs to hold the model alone.
    pub fn whole(&self) -> u64 {
        self.bytes.saturating_add(self.per_card)
    }

    /// Bytes the cards leave to the host, the model spread over every card.
    pub fn spill(&self, cards: &[CardRoom]) -> u64 {
        let room: u64 = cards
            .iter()
            .map(|c| c.available.saturating_sub(self.per_card))
            .sum();
        self.bytes.saturating_sub(room)
    }
}

/// What a drafter asks of one card: all of it, or its layers with its routed experts on the
/// host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrafterNeed {
    pub whole: CardNeed,
    pub layers: CardNeed,
}

/// How the cards take a drafter held for before its target is placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterHold {
    /// The target sits on the cards: a drafter gains nothing.
    TargetFits,
    /// The fastest card offers `room` bytes, too few for the drafter's layers.
    NotWhole { room: u64 },
    /// `bytes` held on `card`, `None` without a card.
    /// The target spills `alone` bytes without the hold and `pushed` bytes more with it.
    Held {
        card: Option<usize>,
        bytes: u64,
        alone: u64,
        pushed: u64,
    },
}

impl DrafterHold {
    /// How many times the host bytes of a plain step grow under the hold.
    /// A verify must commit more tokens than this for the hold to pay.
    pub fn host_ratio(&self) -> Option<f64> {
        match *self {
            DrafterHold::Held { alone, pushed, .. } if alone > 0 => {
                Some((alone + pushed) as f64 / alone as f64)
            }
            _ => None,
        }
    }
}

/// How the cards take `drafter` held beside `target` before the target is placed, `cards`
/// fastest first.
///
/// The hold is taken on the fastest card.
/// It is the whole drafter when that card holds it, else its layers, its routed experts on the
/// host.
/// Without a card the drafter runs on the host and takes nothing from the target.
pub fn drafter_hold(target: CardNeed, drafter: DrafterNeed, cards: &[CardRoom]) -> DrafterHold {
    let alone = target.spill(cards);
    if alone == 0 {
        return DrafterHold::TargetFits;
    }
    let Some(first) = cards.first() else {
        return DrafterHold::Held {
            card: None,
            bytes: 0,
            alone,
            pushed: 0,
        };
    };
    let Some(bytes) = [drafter.whole.whole(), drafter.layers.whole()]
        .into_iter()
        .find(|b| *b <= first.available)
    else {
        return DrafterHold::NotWhole {
            room: first.available,
        };
    };
    let mut held = cards.to_vec();
    held[0].available -= bytes;
    DrafterHold::Held {
        card: Some(first.index),
        bytes,
        alone,
        pushed: target.spill(&held) - alone,
    }
}

/// The drafter of `drafters` whose hold grows the target's host bytes least, held when a verify
/// is `expected` to commit more tokens than that growth, with the verdict of each one.
pub fn choose_hold(
    target: CardNeed,
    drafters: &[DrafterNeed],
    cards: &[CardRoom],
    expected: f64,
) -> (Option<usize>, Vec<DrafterHold>) {
    let verdicts: Vec<DrafterHold> = drafters
        .iter()
        .map(|d| drafter_hold(target, *d, cards))
        .collect();
    let least = verdicts
        .iter()
        .enumerate()
        .filter_map(|(n, v)| v.host_ratio().map(|r| (n, r)))
        .min_by(|a, b| a.1.total_cmp(&b.1));
    let chosen = least.filter(|(_, r)| *r < expected).map(|(n, _)| n);
    (chosen, verdicts)
}

/// How the cards take a drafter beside a target already placed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrafterFit {
    /// No card holds the drafter whole and keeps the target's reserve beside it.
    /// `room` is the most a card offers beside that reserve.
    NoRoom { room: u64 },
    /// Runs on `card`, `None` without a card, in room the target does not use.
    Fits { card: Option<usize> },
}

/// How the cards take `drafter` beside a placed target that keeps `reserve` free on each card,
/// `cards` fastest first, at what they have free now.
///
/// A drafter runs whole on the fastest card that holds it, as the ladder places it.
/// It is taken only when that card still keeps the target's reserve beside it.
/// So the drafter moves none of the target's layers to the host.
///
/// Without a card the drafter runs on the host beside the target.
pub fn drafter_fit(reserve: u64, drafter: CardNeed, cards: &[CardRoom]) -> DrafterFit {
    if cards.is_empty() {
        return DrafterFit::Fits { card: None };
    }
    let need = drafter.whole();
    let beside = |c: &CardRoom| c.available.saturating_sub(reserve);
    match cards.iter().find(|c| c.available >= need) {
        Some(c) if beside(c) >= need => DrafterFit::Fits {
            card: Some(c.index),
        },
        _ => DrafterFit::NoRoom {
            room: cards.iter().map(beside).max().unwrap_or(0),
        },
    }
}

/// The first of `drafters`, in their order, that the cards take beside a placed target keeping
/// `reserve` free on each card, with the verdict of each one considered.
pub fn choose_drafter(
    reserve: u64,
    drafters: &[CardNeed],
    cards: &[CardRoom],
) -> (Option<usize>, Vec<DrafterFit>) {
    let mut verdicts = Vec::new();
    for (n, drafter) in drafters.iter().enumerate() {
        let fit = drafter_fit(reserve, *drafter, cards);
        verdicts.push(fit);
        if let DrafterFit::Fits { .. } = fit {
            return (Some(n), verdicts);
        }
    }
    (None, verdicts)
}

/// Holds `bytes` of the cards for `owner` until the lease drops: every placement made
/// meanwhile, but the owner's own, is planned around it.
pub fn hold(owner: &'static str, bytes: u64) -> super::vram_manager::DemandLease {
    super::vram_manager::declare_demand(owner, bytes)
}

/// Holds `bytes` of card `card` for `owner` until the lease drops: every placement made
/// meanwhile, but the owner's own, is planned around it.
pub fn hold_on(owner: &str, card: usize, bytes: u64) -> super::vram_manager::DemandLease {
    super::vram_manager::declare_card_demand(owner, card, bytes)
}

/// The highest memory bandwidth among the cards, in GB/s; zero without a card.
pub fn fastest_bandwidth_gbs() -> f64 {
    probe_cuda_gpus()
        .iter()
        .map(|g| g.mem_bw_gbs)
        .fold(0.0, f64::max)
}

/// Bytes free on the card `device` runs on, as the driver reports them now, the pools left
/// alone: for a component sizing work inside the room its placement already gave it.
/// `None` for the host or a card that does not answer.
pub fn free_now(device: &crate::tensor::Device) -> Option<u64> {
    #[cfg(feature = "gpu")]
    if device.is_cuda() {
        return crate::tensor::cuda_ext::mem_get_info(device)
            .ok()
            .map(|(free, _)| free as u64);
    }
    let _ = device;
    None
}

/// Settles card `card`: the work queued on it runs, so the buffers freed in its order are freed,
/// and its pool hands the free blocks back to the driver. A card no context holds is left alone.
pub fn settle(card: usize) {
    #[cfg(feature = "gpu")]
    if crate::tensor::cuda::card_context_active(card) {
        // A graph another thread records on the card is left whole: the card settles after it.
        crate::tensor::cuda::outside_recordings_on(card, || {
            if let Ok(device) = crate::tensor::Device::new_cuda(card) {
                let _ = device.synchronize();
            }
            crate::tensor::cuda::trim_pool(card);
        });
    }
    super::ladder::release_pools();
}

/// What `card` offers `owner` now: its free memory once settled, less what other owners declared
/// they are about to take there.
pub fn offered(card: usize, owner: &str) -> u64 {
    settle(card);
    super::device_probe::probe_cuda_gpus_for(owner)
        .into_iter()
        .find(|g| g.index == card)
        .map_or(0, |g| g.stable_free)
}

/// The ordinal of the card `device` runs on, `None` for the host.
pub fn card_of(device: &crate::tensor::Device) -> Option<usize> {
    #[cfg(feature = "gpu")]
    if let Ok(cd) = device.as_cuda_device() {
        return Some(cd.0.ordinal());
    }
    let _ = device;
    None
}

/// [`free_now`] for a caller holding the card's context rather than a device.
#[cfg(feature = "gpu")]
pub fn free_in(ctx: &crate::tensor::gpu_runtime::driver::CudaContext) -> Option<u64> {
    ctx.mem_get_info().ok().map(|(free, _)| free as u64)
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

    /// 2 cards of 16 GB, each with 16 GB free, the fastest first.
    fn two_cards() -> [CardRoom; 2] {
        [
            CardRoom {
                index: 1,
                ..card(16 * GB, 16 * GB)
            },
            card(16 * GB, 16 * GB),
        ]
    }

    fn need(bytes: u64, per_card: u64) -> CardNeed {
        CardNeed { bytes, per_card }
    }

    fn drafter(whole: u64, layers: u64, per_card: u64) -> DrafterNeed {
        DrafterNeed {
            whole: need(whole, per_card),
            layers: need(layers, per_card),
        }
    }

    #[test]
    fn a_target_the_cards_hold_gets_no_hold() {
        let target = need(26 * GB, 2 * GB);
        let (chosen, verdicts) = choose_hold(target, &[drafter(GB, GB, GB)], &two_cards(), 1.5);
        assert_eq!(chosen, None);
        assert_eq!(verdicts, [DrafterHold::TargetFits]);
    }

    /// A drafter whose experts stream holds its layers alone, never its file.
    #[test]
    fn a_drafter_whose_experts_stream_holds_its_layers() {
        let target = need(40 * GB, 2 * GB);
        let streamed = drafter(368 * GB, 3 * GB, GB);
        assert_eq!(
            drafter_hold(target, streamed, &two_cards()),
            DrafterHold::Held {
                card: Some(1),
                bytes: 4 * GB,
                alone: 12 * GB,
                pushed: 4 * GB,
            }
        );
        let too_wide = drafter(368 * GB, 20 * GB, GB);
        assert_eq!(
            drafter_hold(target, too_wide, &two_cards()),
            DrafterHold::NotWhole { room: 16 * GB }
        );
    }

    /// The hold pays only when a verify commits more tokens than the host bytes grow.
    /// A small push against a large spill is held.
    /// A push near the spill is not.
    #[test]
    fn a_hold_is_taken_when_a_verify_commits_more_than_the_host_bytes_grow() {
        let small = drafter(2 * GB, 2 * GB, GB / 2);
        // Spills 14 GB alone, 2.5 GB more with the hold: a growth below one and a half.
        let large_spill = need(42 * GB, 2 * GB);
        let (chosen, verdicts) = choose_hold(large_spill, &[small], &two_cards(), 1.5);
        assert_eq!(chosen, Some(0));
        assert!(verdicts[0].host_ratio().unwrap() < 1.5);
        // Spills 3 GB alone, 2.5 GB more with the hold: a growth above one and a half.
        let small_spill = need(31 * GB, 2 * GB);
        let (chosen, verdicts) = choose_hold(small_spill, &[small], &two_cards(), 1.5);
        assert_eq!(chosen, None);
        assert!(verdicts[0].host_ratio().unwrap() > 1.5);
    }

    #[test]
    fn the_hold_that_grows_the_host_bytes_least_is_chosen() {
        let target = need(42 * GB, 2 * GB);
        let larger = drafter(6 * GB, 6 * GB, GB);
        let smaller = drafter(GB, GB, GB / 2);
        let (chosen, _) = choose_hold(target, &[larger, smaller], &two_cards(), 2.0);
        assert_eq!(chosen, Some(1));
    }

    #[test]
    fn without_a_card_a_hold_takes_nothing() {
        let (chosen, verdicts) =
            choose_hold(need(40 * GB, 2 * GB), &[drafter(GB, GB, GB)], &[], 1.5);
        assert_eq!(chosen, Some(0));
        assert_eq!(verdicts[0].host_ratio(), Some(1.0));
    }

    #[test]
    fn a_drafter_streamed_from_the_host_is_not_taken() {
        let streamed = need(368 * GB, 4 * GB);
        let (chosen, verdicts) = choose_drafter(2 * GB, &[streamed], &two_cards());
        assert_eq!(chosen, None);
        assert_eq!(verdicts, [DrafterFit::NoRoom { room: 14 * GB }]);
    }

    #[test]
    fn a_rejected_drafter_leaves_the_next_one_the_same_room() {
        let streamed = need(368 * GB, 4 * GB);
        let small = need(GB, GB);
        let (chosen, verdicts) = choose_drafter(2 * GB, &[streamed, small], &two_cards());
        assert_eq!(chosen, Some(1));
        assert_eq!(verdicts[1], drafter_fit(2 * GB, small, &two_cards()));
    }

    /// A spilled target leaves each card its reserve and little more.
    /// A drafter that would eat into that reserve is not taken.
    #[test]
    fn a_drafter_never_takes_the_reserve_of_a_spilled_target() {
        let full = [
            CardRoom {
                index: 1,
                ..card(3 * GB, 16 * GB)
            },
            card(3 * GB, 16 * GB),
        ];
        let drafter = need(2 * GB, GB / 2);
        assert_eq!(
            drafter_fit(2 * GB, drafter, &full),
            DrafterFit::NoRoom { room: GB }
        );
    }

    /// The drafter goes where the ladder puts it, the fastest card that holds it.
    /// A slower card with room beside the reserve does not make that card's reserve safe.
    #[test]
    fn a_drafter_is_judged_on_the_card_the_ladder_gives_it() {
        let cards = [
            CardRoom {
                index: 1,
                ..card(3 * GB, 16 * GB)
            },
            card(8 * GB, 16 * GB),
        ];
        let drafter = need(2 * GB, GB / 2);
        assert!(matches!(
            drafter_fit(2 * GB, drafter, &cards),
            DrafterFit::NoRoom { .. }
        ));
        let roomy = [
            CardRoom {
                index: 1,
                ..card(GB, 16 * GB)
            },
            card(8 * GB, 16 * GB),
        ];
        assert_eq!(
            drafter_fit(2 * GB, drafter, &roomy),
            DrafterFit::Fits { card: Some(0) }
        );
    }

    #[test]
    fn a_small_drafter_runs_on_the_fastest_card_with_room_left() {
        assert_eq!(
            drafter_fit(2 * GB, need(GB, GB), &two_cards()),
            DrafterFit::Fits { card: Some(1) }
        );
    }

    #[test]
    fn without_a_card_the_drafter_runs_beside_the_target() {
        assert_eq!(
            drafter_fit(2 * GB, need(GB, GB), &[]),
            DrafterFit::Fits { card: None }
        );
    }
}
