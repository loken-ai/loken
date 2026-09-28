//! Deciding WHERE a component runs, and what that costs. The heterogeneous
//! plan, the VRAM budgets it is checked against, and the device probes under both.
//!
//! Callers name a module through this directory - `crate::inference::place::<module>` - so the path says which
//! part of the system a file belongs to, which is the whole reason the directory exists.

pub mod audio_demand;
pub mod device_probe;
pub mod dry_plan;
pub mod fit_report;
#[cfg(all(test, feature = "image"))]
mod geometry_invariants;
pub mod ladder;
pub mod layer_executor;
pub mod layer_perf;
pub mod model_request;
pub mod paged;
#[cfg(all(test, feature = "image"))]
pub mod placement_invariants;
pub mod plan;
pub mod recover;
pub mod room;
pub mod runtime_demand;
#[cfg(feature = "video")]
pub mod video_plan;
pub mod vram_manager;

/// Where a model's layers run: the cards holding them and how many run on the host.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Placement {
    pub cards: std::collections::BTreeSet<usize>,
    pub host_layers: usize,
}

/// A card as a move is judged against it: its index, its speed score, its free memory and
/// its size.
#[derive(Debug, Clone, Copy)]
pub struct CardRoom {
    pub index: usize,
    pub perf: u64,
    pub available: u64,
    pub total: u64,
}

impl Placement {
    /// The card this model would run faster on if reloaded there whole, given `need` bytes to
    /// hold it: the fastest card it does not use now that has the room, when the model is split
    /// across cards or partly on the host, or when that card is faster than the one it uses.
    pub fn better_card(&self, cards: &[CardRoom], need: u64) -> Option<usize> {
        let target = cards
            .iter()
            .filter(|c| !self.cards.contains(&c.index) && c.available >= need)
            .max_by_key(|c| c.perf)?;
        let whole_on_one = self.cards.len() == 1 && self.host_layers == 0;
        if !whole_on_one {
            return Some(target.index);
        }
        let current = cards
            .iter()
            .filter(|c| self.cards.contains(&c.index))
            .map(|c| c.perf)
            .max()
            .unwrap_or(0);
        (target.perf > current).then_some(target.index)
    }
}

#[cfg(test)]
mod placement_tests {
    use super::*;

    fn card(index: usize, perf: u64, available: u64) -> CardRoom {
        CardRoom {
            index,
            perf,
            available,
            total: available,
        }
    }

    fn on(cards: &[usize], host_layers: usize) -> Placement {
        Placement {
            cards: cards.iter().copied().collect(),
            host_layers,
        }
    }

    #[test]
    fn a_model_on_the_slow_card_moves_to_a_faster_one_with_room() {
        let cards = [card(0, 2, 10), card(1, 1, 0)];
        assert_eq!(on(&[1], 0).better_card(&cards, 8), Some(0));
        assert_eq!(on(&[1], 0).better_card(&cards, 12), None);
        assert_eq!(
            on(&[0], 0).better_card(&[card(0, 2, 0), card(1, 1, 16)], 8),
            None
        );
    }

    #[test]
    fn a_split_or_spilled_model_moves_to_any_card_that_holds_it_whole() {
        let cards = [card(0, 2, 0), card(1, 1, 16)];
        assert_eq!(on(&[0], 4).better_card(&cards, 8), Some(1));
        assert_eq!(on(&[0, 1], 0).better_card(&cards, 8), None);
        assert_eq!(on(&[0], 0).better_card(&cards, 8), None);
    }
}
