//! Tensor parallelism: every dense layer's heads and feed-forward split across a group of cards,
//! each card reading its share of every layer while the others read theirs.
//!
//! Whether it pays is predicted from measured quantities only, against the layer by layer split
//! the planner otherwise makes, where a token crosses the cards in turn and each reads its own
//! layers whole. Split, a layer's bytes are read by the group together, in proportion to each
//! card's read rate, and each layer sums its attention and its feed-forward onto every card. The
//! split is chosen when it is predicted faster both for a decoded token and for a block of prompt
//! rows: a group whose sums cost more than the reads and the products they save keeps the layers
//! whole. A card's share of a layer takes the longer of its read and its product of the rows.

use super::expert_parallel::ShardCard;

/// `units` whole units shared in proportion to `weights`, each share at least one unit, the
/// remainders to the largest fractions; `None` when there are fewer units than weights.
pub fn shares(weights: &[f64], units: usize) -> Option<Vec<usize>> {
    let total: f64 = weights.iter().sum();
    if units < weights.len() || total <= 0.0 {
        return None;
    }
    let exact: Vec<f64> = weights.iter().map(|w| units as f64 * w / total).collect();
    let mut out: Vec<usize> = exact.iter().map(|e| (e.floor() as usize).max(1)).collect();
    // Too many after the floor of one: the largest shares give back.
    while out.iter().sum::<usize>() > units {
        let i = (0..out.len()).max_by_key(|&i| out[i]).expect("a share");
        out[i] -= 1;
    }
    // Too few: the largest remainders take one more.
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| (exact[b] - out[b] as f64).total_cmp(&(exact[a] - out[a] as f64)));
    let mut left = units - out.iter().sum::<usize>();
    for i in order.into_iter().cycle() {
        if left == 0 {
            break;
        }
        out[i] += 1;
        left -= 1;
    }
    Some(out)
}

/// A layer's routed experts as the split sees them.
#[derive(Clone, Copy, Debug)]
pub struct RoutedExperts {
    /// Bytes of all of a layer's experts.
    pub bytes: u64,
    pub experts: usize,
    /// Experts a row routes to.
    pub used: usize,
}

/// A stack of layers as the split sees it.
#[derive(Clone, Copy, Debug)]
pub struct DenseStack {
    pub layers: usize,
    /// Bytes of one layer's weights every row reads: all of them, a mixture's experts aside.
    pub layer_bytes: u64,
    /// A mixture layer's experts, of which `rows` rows read the ones they route to.
    pub routed: Option<RoutedExperts>,
    /// Bytes of one row of activations as it crosses a link.
    pub row_bytes: u64,
    /// Floating point operations one row costs through one layer's projections, a mixture's
    /// counted at the experts a row routes to.
    pub row_flops: f64,
}

impl DenseStack {
    /// Bytes `rows` rows read of a layer: its shared weights and the experts they route to, at
    /// most all of them.
    fn read_bytes(&self, rows: usize) -> f64 {
        let routed = self.routed.map_or(0.0, |r| {
            let touched = (rows * r.used) as f64 / r.experts.max(1) as f64;
            r.bytes as f64 * touched.min(1.0)
        });
        self.layer_bytes as f64 + routed
    }
}

/// Microseconds a card spends on `fraction` of a layer for `rows` rows: the longer of reading its
/// part of the weights at `read` bytes and multiplying the rows through it at `flops` operations
/// per microsecond.
fn share_us(stack: &DenseStack, fraction: f64, read: f64, flops: f64, rows: usize) -> f64 {
    let bytes = fraction * stack.read_bytes(rows) / read;
    let product = fraction * rows as f64 * stack.row_flops / flops;
    bytes.max(product)
}

/// What summing a message onto every card of a group costs: microseconds for a small one and
/// bytes per microsecond beyond it.
#[derive(Clone, Copy, Debug)]
pub struct GroupSum {
    pub us: f64,
    pub bytes_per_us: f64,
}

/// Predicted microseconds of `rows` rows through the stack with each layer run on every card of
/// `cards` in proportion to their read rates, `flops[i]` the product rate of `cards[i]`: per
/// layer, the slowest card's share, and two sums of `rows` rows onto every card, each card
/// sending its part `2 (n - 1) / n` times over.
pub fn split_us(
    stack: &DenseStack,
    cards: &[ShardCard],
    flops: &[f64],
    sum: GroupSum,
    rows: usize,
) -> f64 {
    let n = cards.len() as f64;
    let total_read: f64 = cards.iter().map(|c| c.read_bytes_per_us).sum();
    let share = cards
        .iter()
        .zip(flops)
        .map(|(c, &f)| {
            share_us(
                stack,
                c.read_bytes_per_us / total_read,
                c.read_bytes_per_us,
                f,
                rows,
            )
        })
        .fold(0.0, f64::max);
    let bytes = rows as f64 * stack.row_bytes as f64 * 2.0 * (n - 1.0) / n;
    stack.layers as f64 * (share + 2.0 * (sum.us + bytes / sum.bytes_per_us))
}

/// Predicted microseconds of `rows` rows through the stack placed layer by layer: `layers_on[i]`
/// layers run whole by `cards[i]` in turn, at the product rate `flops[i]`, and one trip of the
/// rows at each change of card.
pub fn layered_us(
    stack: &DenseStack,
    cards: &[ShardCard],
    flops: &[f64],
    layers_on: &[usize],
    rows: usize,
) -> f64 {
    let reads: f64 = cards
        .iter()
        .zip(flops)
        .zip(layers_on)
        .map(|((c, &f), &n)| n as f64 * share_us(stack, 1.0, c.read_bytes_per_us, f, rows))
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

/// Whether splitting every layer over `cards` is predicted faster than the layer by layer
/// placement `layers_on`, for a decoded token and for `prompt_rows` rows of a prompt.
pub fn split_pays(
    stack: &DenseStack,
    cards: &[ShardCard],
    flops: &[f64],
    sum: GroupSum,
    layers_on: &[usize],
    prompt_rows: usize,
) -> bool {
    cards.len() > 1
        && [1, prompt_rows].iter().all(|&rows| {
            split_us(stack, cards, flops, sum, rows)
                < layered_us(stack, cards, flops, layers_on, rows)
        })
}

/// Whether `demand`, a dense stack of rows `row_bytes` wide, runs faster split over `cards` (the
/// first holding the layers) than as the planner places it layer by layer in `layered`, for a
/// decoded token and for `prompt_rows` rows of a prompt, from each card's measured read rate, the
/// measured links from the first card and the measured sum across the group. A layered plan that
/// leaves layers on the host is beaten by any split that holds them on the cards.
pub fn pays_over(
    demand: &super::ladder::Demand,
    layered: &super::layer_executor::HeteroPlan,
    cards: &[usize],
    row_bytes: u64,
    row_flops: f64,
    routed: Option<RoutedExperts>,
    prompt_rows: usize,
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
    let Some((us, bytes_per_us)) = super::device_probe::group_sum(cards) else {
        return false;
    };
    let Some(flops) = cards
        .iter()
        .map(|&c| super::device_probe::card_matmul_rate(c))
        .collect::<Option<Vec<f64>>>()
    else {
        return false;
    };
    let stack = DenseStack {
        layers: demand.layers,
        layer_bytes: (demand.block_bytes / demand.layers.max(1) as u64)
            .saturating_sub(routed.map_or(0, |r| r.bytes)),
        routed,
        row_bytes,
        row_flops,
    };
    split_pays(
        &stack,
        &shard_cards,
        &flops,
        GroupSum { us, bytes_per_us },
        &layers_on,
        prompt_rows,
    )
}

/// The plan that splits every layer of `demand` over the fewest of `cards` that hold their shares,
/// when that is predicted faster than the layer by layer plan for a decoded token and for
/// `prompt_rows` rows of a prompt; `refusal(ranks)` says why the model cannot split over that many
/// cards, `row_bytes` is a row of activations, `row_flops` what a row costs through a layer and
/// `routed` a mixture layer's experts.
pub fn plan(
    demand: &super::ladder::Demand,
    cards: &[(usize, u64)],
    row_bytes: u64,
    row_flops: f64,
    routed: Option<RoutedExperts>,
    prompt_rows: usize,
    refusal: impl Fn(usize) -> Option<String>,
) -> Option<super::layer_executor::HeteroPlan> {
    (2..=cards.len()).find_map(|ranks| {
        if let Some(why) = refusal(ranks) {
            tracing::info!("stays off tensor parallelism on {ranks} cards: {why}");
            return None;
        }
        let rates = super::expert_parallel::card_read_rates();
        let weight = |idx: usize| rates.get(&idx).copied().unwrap_or(0.0);
        let group = super::llm::tensor_parallel(demand, cards, ranks, weight)?;
        if !pays_over(demand, &demand.plan(cards), &group, row_bytes, row_flops, routed, prompt_rows) {
            tracing::info!(
                "stays layer by layer on {ranks} cards: the sums cost more than the split saves, for a token or for {prompt_rows} prompt rows"
            );
            return None;
        }
        let mut plan = super::layer_executor::HeteroPlan::forced_gpu(demand.layers, demand.layers, group[0]);
        plan.tp_cards = Some(
            group
                .iter()
                .map(|&index| super::layer_executor::TpCard { index, weight: weight(index) })
                .collect(),
        );
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

    fn sum(us: f64, gbs: f64) -> GroupSum {
        GroupSum {
            us,
            bytes_per_us: gbs * 1e3,
        }
    }

    /// A 20 GB stack of 64 layers of 600 million parameters each.
    const STACK: DenseStack = DenseStack {
        layers: 64,
        layer_bytes: 310_000_000,
        routed: None,
        row_bytes: 20_480,
        row_flops: 1.2e9,
    };

    /// TFLOP/s as operations per microsecond.
    fn tflops(t: f64) -> f64 {
        t * 1e6
    }

    /// The stack on two PCIe cards whose sums go through host memory: a token is faster split,
    /// two short sums a layer against half its read, but a block of prompt rows pays two slow sums
    /// of every row a layer, more than the products it shares out, and keeps the layers whole.
    #[test]
    fn sums_through_the_host_keep_a_prompt_layered() {
        let cards = [card(896.0, None), card(448.0, Some((150.0, 2.0)))];
        let flops = [tflops(80.0), tflops(40.0)];
        let host = sum(40.0, 1.7);
        let on = [42, 22];
        assert!(
            split_us(&STACK, &cards, &flops, host, 1) < layered_us(&STACK, &cards, &flops, &on, 1)
        );
        assert!(
            split_us(&STACK, &cards, &flops, host, 512)
                > layered_us(&STACK, &cards, &flops, &on, 512)
        );
        assert!(!split_pays(&STACK, &cards, &flops, host, &on, 512));
    }

    /// The same stack on two cards joined by a fast fabric: each reads half of every layer and
    /// runs half of every product, so a token, under three quarters of its layered time, and a
    /// block of prompt rows are both faster split.
    #[test]
    fn a_fast_fabric_splits_the_layers() {
        let cards = [card(3350.0, None), card(3350.0, Some((3.0, 450.0)))];
        let flops = [tflops(700.0), tflops(700.0)];
        let fabric = sum(10.0, 450.0);
        let on = [32, 32];
        assert!(split_pays(&STACK, &cards, &flops, fabric, &on, 512));
        assert!(
            split_us(&STACK, &cards, &flops, fabric, 1)
                < 0.75 * layered_us(&STACK, &cards, &flops, &on, 1)
        );
    }

    /// Units go in proportion to the weights, every share at least one unit, summing to all.
    #[test]
    fn shares_follow_the_weights() {
        assert_eq!(shares(&[896.0, 448.0], 4), Some(vec![3, 1]));
        assert_eq!(shares(&[896.0, 448.0], 108), Some(vec![72, 36]));
        assert_eq!(
            shares(&[1.0, 1.0, 1.0], 4).map(|s| s.iter().sum::<usize>()),
            Some(4)
        );
        assert_eq!(shares(&[10.0, 1.0], 2), Some(vec![1, 1]));
        assert_eq!(shares(&[1.0, 1.0], 1), None);
    }

    /// A token reads its routed experts alone, a block of prompt rows every expert.
    #[test]
    fn rows_read_the_experts_they_route_to() {
        let stack = DenseStack {
            layers: 48,
            layer_bytes: 20_000_000,
            routed: Some(RoutedExperts {
                bytes: 1_280_000_000,
                experts: 128,
                used: 8,
            }),
            row_bytes: 8192,
            row_flops: 1e9,
        };
        assert_eq!(stack.read_bytes(1), 20_000_000.0 + 80_000_000.0);
        assert_eq!(stack.read_bytes(512), 1_300_000_000.0);
    }

    /// One card is no split.
    #[test]
    fn one_card_does_not_split() {
        let stack = DenseStack {
            layers: 4,
            layer_bytes: 1_000_000,
            routed: None,
            row_bytes: 1024,
            row_flops: 2e6,
        };
        assert!(!split_pays(
            &stack,
            &[card(100.0, None)],
            &[tflops(1.0)],
            sum(1.0, 1.0),
            &[4],
            1
        ));
    }
}
