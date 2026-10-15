//! Tensor parallelism: every dense layer's heads and feed-forward split across a group of cards,
//! each card reading its share of every layer while the others read theirs.
//!
//! Whether it pays is predicted from measured quantities only, against the layer by layer split
//! the planner otherwise makes, where a token crosses the cards in turn and each reads its own
//! layers whole. Split, a layer's bytes are read by the group together, and each sublayer pays a
//! trip of the activations to every other card and of its partial output back. A machine whose
//! links cost more than the reads they save keeps the layer split.

use super::expert_parallel::ShardCard;

/// A dense stack as the split sees it.
#[derive(Clone, Copy, Debug)]
pub struct DenseStack {
    pub layers: usize,
    /// Bytes of one layer's weights.
    pub layer_bytes: u64,
    /// Bytes of one row of activations as it crosses a link.
    pub row_bytes: u64,
}

/// Predicted microseconds of a token through the stack with each layer split over `cards`, the
/// first being the layers' own card: per layer, the slowest card's read of its share, and two
/// sublayers each sending `rows` rows out to every other card and the partial outputs back.
pub fn split_us(stack: &DenseStack, cards: &[ShardCard], rows: usize) -> f64 {
    let ranks = cards.len() as f64;
    let read = cards
        .iter()
        .map(|c| stack.layer_bytes as f64 / ranks / c.read_bytes_per_us)
        .fold(0.0, f64::max);
    let trip = cards
        .iter()
        .skip(1)
        .map(|c| 2.0 * (c.link_us + rows as f64 * stack.row_bytes as f64 / c.link_bytes_per_us))
        .fold(0.0, f64::max);
    stack.layers as f64 * (read + 2.0 * trip)
}

/// Predicted microseconds of a token through the stack placed layer by layer: `layers_on[i]`
/// layers read whole by `cards[i]` in turn, and one trip of `rows` rows at each change of card.
pub fn layered_us(
    stack: &DenseStack,
    cards: &[ShardCard],
    layers_on: &[usize],
    rows: usize,
) -> f64 {
    let reads: f64 = cards
        .iter()
        .zip(layers_on)
        .map(|(c, &n)| n as f64 * stack.layer_bytes as f64 / c.read_bytes_per_us)
        .sum();
    let hops: f64 = cards
        .iter()
        .zip(layers_on)
        .skip(1)
        .filter(|(_, &n)| n > 0)
        .map(|(c, _)| c.link_us + rows as f64 * stack.row_bytes as f64 / c.link_bytes_per_us)
        .sum();
    reads + hops
}

/// Whether splitting every layer over `cards` is predicted faster for `rows` rows than the
/// layer by layer placement `layers_on`.
pub fn split_pays(
    stack: &DenseStack,
    cards: &[ShardCard],
    layers_on: &[usize],
    rows: usize,
) -> bool {
    cards.len() > 1 && split_us(stack, cards, rows) < layered_us(stack, cards, layers_on, rows)
}

/// Whether `demand`, a dense stack of rows `row_bytes` wide, runs faster split over `cards` (the
/// first holding the layers) than as the planner places it layer by layer in `layered`, from each
/// card's measured read rate and the measured links from the first card. A layered plan that
/// leaves layers on the host is beaten by any split that holds them on the cards.
pub fn pays_over(
    demand: &super::ladder::Demand,
    layered: &super::layer_executor::HeteroPlan,
    cards: &[usize],
    row_bytes: u64,
) -> bool {
    use super::layer_executor::DeviceKind;
    let Some(&owner) = cards.first() else {
        return false;
    };
    if layered.segments.iter().any(|s| s.kind == DeviceKind::Cpu) {
        return true;
    }
    let read = super::expert_parallel::card_read_rates();
    let mut shard_cards = Vec::with_capacity(cards.len());
    for &c in cards {
        let Some(&rate) = read.get(&c).filter(|r| **r > 0.0) else {
            return false;
        };
        let (link_us, link_bytes_per_us) = if c == owner {
            (0.0, f64::INFINITY)
        } else {
            match super::device_probe::card_pair_link(owner, c) {
                Some(l) => l,
                None => return false,
            }
        };
        shard_cards.push(ShardCard {
            index: c,
            read_bytes_per_us: rate,
            link_us,
            link_bytes_per_us,
            room: 0,
        });
    }
    let layers_on: Vec<usize> = cards
        .iter()
        .map(|&c| {
            layered
                .segments
                .iter()
                .filter(|s| s.kind == DeviceKind::Cuda(c))
                .map(|s| s.layer_end - s.layer_start)
                .sum()
        })
        .collect();
    let stack = DenseStack {
        layers: demand.layers,
        layer_bytes: demand.block_bytes / demand.layers.max(1) as u64,
        row_bytes,
    };
    split_pays(&stack, &shard_cards, &layers_on, 1)
}

/// The plan that splits every layer of `demand` over the fewest of `cards` that hold an even
/// share each, when that is predicted faster than the layer by layer plan; `refusal(ranks)` says
/// why the model cannot split over that many cards, and `row_bytes` is a row of activations.
pub fn plan(
    demand: &super::ladder::Demand,
    cards: &[(usize, u64)],
    row_bytes: u64,
    refusal: impl Fn(usize) -> Option<String>,
) -> Option<super::layer_executor::HeteroPlan> {
    (2..=cards.len()).find_map(|ranks| {
        if let Some(why) = refusal(ranks) {
            tracing::info!("stays off tensor parallelism on {ranks} cards: {why}");
            return None;
        }
        let group = super::llm::tensor_parallel(demand, cards, ranks)?;
        if !pays_over(demand, &demand.plan(cards), &group, row_bytes) {
            tracing::info!(
                "stays layer by layer on {ranks} cards: the links cost more than the split saves"
            );
            return None;
        }
        let mut plan =
            super::layer_executor::HeteroPlan::forced_gpu(demand.layers, demand.layers, group[0]);
        plan.tp_cards = Some(group);
        Some(plan)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(read_gbs: f64, link: Option<(f64, f64)>) -> ShardCard {
        let (link_us, link_bytes_per_us) =
            link.map_or((0.0, f64::INFINITY), |(us, gbs)| (us, gbs * 1e3));
        ShardCard {
            index: 0,
            read_bytes_per_us: read_gbs * 1e3,
            link_us,
            link_bytes_per_us,
            room: 0,
        }
    }

    /// A 20 GB stack of 64 layers on two PCIe cards that reach each other through the host: the
    /// four trips a layer cost more than the half of its read they save.
    #[test]
    fn a_slow_link_keeps_the_layers_whole() {
        let stack = DenseStack {
            layers: 64,
            layer_bytes: 310_000_000,
            row_bytes: 20_480,
        };
        let cards = [card(896.0, None), card(448.0, Some((60.0, 12.0)))];
        assert!(!split_pays(&stack, &cards, &[42, 22], 1));
    }

    /// The same stack on two cards joined by a fast fabric: each reads half of every layer, and
    /// the four short trips a layer leave it at about two thirds of the whole read.
    #[test]
    fn a_fast_fabric_splits_the_layers() {
        let stack = DenseStack {
            layers: 64,
            layer_bytes: 310_000_000,
            row_bytes: 20_480,
        };
        let cards = [card(3350.0, None), card(3350.0, Some((3.0, 450.0)))];
        assert!(split_pays(&stack, &cards, &[32, 32], 1));
        assert!(split_us(&stack, &cards, 1) < 0.7 * layered_us(&stack, &cards, &[32, 32], 1));
    }

    /// One card is no split.
    #[test]
    fn one_card_does_not_split() {
        let stack = DenseStack {
            layers: 4,
            layer_bytes: 1_000_000,
            row_bytes: 1024,
        };
        assert!(!split_pays(&stack, &[card(100.0, None)], &[4], 1));
    }
}
