//! Expert parallelism: a mixture layer's routed experts split by id range across cards, each
//! card computing the pairs of its own experts while the others compute theirs.
//!
//! Whether a split pays is predicted from measured quantities only: how fast each card reads its
//! memory, what the link from the layer's card to each other card costs for a small message and
//! per byte, and how much room each card has for experts. A token's experts are read once per
//! layer whatever card holds them, so a split shortens a layer when the cards read their shares
//! together faster than one card reads them all, the activation's trip out and the products'
//! trip back included. A machine whose links are slow next to its cards' memory gets no split.

/// Bytes per microsecond each card reads its own memory at, by ordinal.
pub fn card_read_rates() -> std::collections::HashMap<usize, f64> {
    super::device_probe::probe_cuda_gpus()
        .iter()
        .map(|g| (g.index, g.mem_bw_gbs * 1e3))
        .collect()
}

/// A card a layer's experts may be split onto, as the layer's own card sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct ShardCard {
    pub index: usize,
    /// Bytes per microsecond the card reads its own memory at.
    pub read_bytes_per_us: f64,
    /// Microseconds a small message takes from the layer's card to this one; zero for the
    /// layer's own card.
    pub link_us: f64,
    /// Bytes per microsecond the link from the layer's card carries; infinite for its own card.
    pub link_bytes_per_us: f64,
    /// Bytes of experts the card can hold.
    pub room: u64,
}

/// A layer's routed experts as the split sees them.
#[derive(Clone, Copy, Debug)]
pub struct ExpertLayer {
    pub n_experts: usize,
    /// Experts each row routes to.
    pub used: usize,
    /// Bytes of one expert, its gate, up and down together.
    pub expert_bytes: u64,
    /// Bytes of one row of activations as it crosses a link (out, and the product back).
    pub row_bytes: u64,
}

impl ExpertLayer {
    /// Distinct experts `rows` rows read, in expectation when each row routes to `used` of the
    /// `n_experts` uniformly.
    fn touched(&self, rows: usize) -> f64 {
        let n = self.n_experts as f64;
        let miss = 1.0 - self.used as f64 / n;
        n * (1.0 - miss.powi(rows as i32))
    }

    /// Microseconds the experts of `rows` rows take on `card` holding the fraction `share` of
    /// them: its read of its share, and for a card other than the layer's, the rows out and
    /// the products back.
    fn time_on(&self, card: &ShardCard, share: f64, rows: usize) -> f64 {
        let read = share * self.touched(rows) * self.expert_bytes as f64 / card.read_bytes_per_us;
        let trip = if card.link_us > 0.0 {
            2.0 * (card.link_us + rows as f64 * self.row_bytes as f64 / card.link_bytes_per_us)
        } else {
            0.0
        };
        read + trip
    }
}

/// The predicted microseconds of a layer's experts for `rows` rows with the counts `counts`
/// (experts per card, in `cards` order): the slowest card's time.
pub fn predicted_us(
    layer: &ExpertLayer,
    cards: &[ShardCard],
    counts: &[usize],
    rows: usize,
) -> f64 {
    cards
        .iter()
        .zip(counts)
        .filter(|(_, &c)| c > 0)
        .map(|(card, &c)| layer.time_on(card, c as f64 / layer.n_experts as f64, rows))
        .fold(0.0, f64::max)
}

/// How many experts each of `cards` holds, in their order, the first being the layer's own card:
/// the split predicted fastest for `rows` rows, when it is faster than the first card holding
/// them all or when the first card cannot hold them all; `None` when the first card alone is
/// fastest or no split fits. Every card given experts holds at least `used` of them, so a row's
/// foreign pairs find experts of the card it does not route to.
pub fn split_experts(layer: &ExpertLayer, cards: &[ShardCard], rows: usize) -> Option<Vec<usize>> {
    let n = layer.n_experts;
    let all_bytes = n as u64 * layer.expert_bytes;
    let alone = cards
        .first()
        .filter(|c| c.room >= all_bytes)
        .map(|c| layer.time_on(c, 1.0, rows));
    // The other cards by what they would add: the fastest reads first.
    let mut order: Vec<usize> = (1..cards.len()).collect();
    order.sort_by(|&a, &b| {
        let t = |i: usize| layer.time_on(&cards[i], 1.0, rows);
        t(a).total_cmp(&t(b))
    });
    let mut best: Option<(f64, Vec<usize>)> = None;
    for take in 1..=order.len() {
        let group: Vec<usize> = std::iter::once(0)
            .chain(order[..take].iter().copied())
            .collect();
        let Some(counts) = balanced(layer, cards, &group, rows) else {
            continue;
        };
        let t = predicted_us(layer, cards, &counts, rows);
        if best.as_ref().is_none_or(|(b, _)| t < *b) {
            best = Some((t, counts));
        }
    }
    let (t, counts) = best?;
    match alone {
        Some(a) if a <= t => None,
        _ => Some(counts),
    }
}

/// The expert counts of `group` (indices into `cards`) that finish together for `rows` rows,
/// each within its room and at least `used`, summing to the layer's experts; `None` when the
/// group cannot hold them.
fn balanced(
    layer: &ExpertLayer,
    cards: &[ShardCard],
    group: &[usize],
    rows: usize,
) -> Option<Vec<usize>> {
    let n = layer.n_experts;
    let cap = |i: usize| (cards[i].room / layer.expert_bytes.max(1)) as usize;
    if group.iter().any(|&i| cap(i) < layer.used)
        || group.iter().map(|&i| cap(i)).sum::<usize>() < n
    {
        return None;
    }
    // Each card's time is `share * per_share + fixed`; equal finishing times give each card the
    // share `(T - fixed) / per_share`, capped by its room, the rest shared again.
    let per_share =
        |i: usize| layer.time_on(&cards[i], 1.0, rows) - layer.time_on(&cards[i], 0.0, rows);
    let fixed = |i: usize| layer.time_on(&cards[i], 0.0, rows);
    let mut counts = vec![0usize; cards.len()];
    let mut open: Vec<usize> = group.to_vec();
    let mut left = n;
    while !open.is_empty() && left > 0 {
        let inv: f64 = open.iter().map(|&i| 1.0 / per_share(i)).sum();
        let t = (left as f64 / n as f64
            + open.iter().map(|&i| fixed(i) / per_share(i)).sum::<f64>())
            / inv;
        let want: Vec<(usize, f64)> = open
            .iter()
            .map(|&i| (i, ((t - fixed(i)) / per_share(i) * n as f64).max(0.0)))
            .collect();
        // A card asked for more than it holds takes what it holds and leaves the share.
        if let Some(&(i, _)) = want.iter().find(|&&(i, w)| w > cap(i) as f64) {
            counts[i] = cap(i);
            left -= cap(i);
            open.retain(|&j| j != i);
            continue;
        }
        // Whole experts, largest remainders first, summing to what is left.
        let mut floor: Vec<(usize, usize, f64)> = want
            .iter()
            .map(|&(i, w)| (i, w.floor() as usize, w - w.floor()))
            .collect();
        let mut short = left - floor.iter().map(|f| f.1).sum::<usize>().min(left);
        floor.sort_by(|a, b| b.2.total_cmp(&a.2));
        for f in &mut floor {
            if short > 0 && f.1 < cap(f.0) {
                f.1 += 1;
                short -= 1;
            }
        }
        for (i, c, _) in floor {
            counts[i] = c;
        }
        left = 0;
    }
    // A card with experts holds at least a row's worth; one that would hold fewer gives its few
    // to the card with the most room left.
    for &i in group {
        if counts[i] > 0 && counts[i] < layer.used {
            let give = counts[i];
            counts[i] = 0;
            let to = group
                .iter()
                .copied()
                .filter(|&j| j != i)
                .max_by_key(|&j| cap(j).saturating_sub(counts[j]))?;
            if cap(to) < counts[to] + give {
                return None;
            }
            counts[to] += give;
        }
    }
    (counts.iter().sum::<usize>() == n).then_some(counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: f64 = 1e9;
    /// Bytes per microsecond of `gbs` gigabytes per second.
    fn per_us(gbs: f64) -> f64 {
        gbs * GB / 1e6
    }

    fn card(index: usize, read_gbs: f64, link: Option<(f64, f64)>, room_gb: f64) -> ShardCard {
        let (link_us, link_bytes_per_us) =
            link.map_or((0.0, f64::INFINITY), |(us, gbs)| (us, per_us(gbs)));
        ShardCard {
            index,
            read_bytes_per_us: per_us(read_gbs),
            link_us,
            link_bytes_per_us,
            room: (room_gb * GB) as u64,
        }
    }

    /// Small experts (2 MB, 10 of 256 a row) on two PCIe cards without peer access: one token's
    /// experts take about 22 us on the first card, less than two trips through the host.
    #[test]
    fn small_experts_stay_on_one_card_when_the_link_costs_more_than_they_do() {
        let layer = ExpertLayer {
            n_experts: 256,
            used: 10,
            expert_bytes: 2_000_000,
            row_bytes: 10_240,
        };
        let cards = [
            card(0, 896.0, None, 16.0),
            card(1, 448.0, Some((30.0, 12.0)), 16.0),
        ];
        assert_eq!(split_experts(&layer, &cards, 1), None);
    }

    /// Large experts (44 MB, 8 of 256 a row) on eight cards joined by a fast fabric, at a batch of
    /// 64 rows: the first card alone reads about 220 experts; eight read an eighth each.
    #[test]
    fn large_experts_spread_over_a_fast_fabric() {
        let layer = ExpertLayer {
            n_experts: 256,
            used: 8,
            expert_bytes: 44_000_000,
            row_bytes: 14_336,
        };
        let mut cards = vec![card(0, 3350.0, None, 80.0)];
        cards.extend((1..8).map(|i| card(i, 3350.0, Some((3.0, 450.0)), 80.0)));
        let counts = split_experts(&layer, &cards, 64).expect("a split pays");
        assert_eq!(counts.iter().sum::<usize>(), 256);
        assert!(counts.iter().all(|&c| (30..=34).contains(&c)), "{counts:?}");
        let alone = predicted_us(&layer, &cards, &[256, 0, 0, 0, 0, 0, 0, 0], 64);
        assert!(predicted_us(&layer, &cards, &counts, 64) < alone / 6.0);
    }

    /// The same fabric at a decoded token: eight experts on eight cards each pay a trip that the
    /// first card's own read of the eight does not, and the split is smaller or none.
    #[test]
    fn a_decoded_token_splits_less_than_a_batch() {
        let layer = ExpertLayer {
            n_experts: 256,
            used: 8,
            expert_bytes: 44_000_000,
            row_bytes: 14_336,
        };
        let mut cards = vec![card(0, 3350.0, None, 80.0)];
        cards.extend((1..8).map(|i| card(i, 3350.0, Some((3.0, 450.0)), 80.0)));
        let one =
            split_experts(&layer, &cards, 1).map_or(1, |c| c.iter().filter(|&&n| n > 0).count());
        let batch =
            split_experts(&layer, &cards, 64).map_or(1, |c| c.iter().filter(|&&n| n > 0).count());
        assert!(one <= batch, "{one} cards for a token, {batch} for a batch");
    }

    /// A first card that cannot hold every expert splits even across a slow link, within each
    /// card's room, every card with experts holding at least a row's worth.
    #[test]
    fn a_layer_too_large_for_its_card_splits_within_each_room() {
        let layer = ExpertLayer {
            n_experts: 128,
            used: 8,
            expert_bytes: 100_000_000,
            row_bytes: 8_192,
        };
        let cards = [
            card(0, 900.0, None, 6.0),
            card(1, 450.0, Some((30.0, 12.0)), 16.0),
        ];
        let counts = split_experts(&layer, &cards, 1).expect("the only placement that fits");
        assert_eq!(counts.iter().sum::<usize>(), 128);
        assert!(counts[0] as u64 * layer.expert_bytes <= cards[0].room);
        assert!(counts[1] as u64 * layer.expert_bytes <= cards[1].room);
        assert!(counts.iter().all(|&c| c == 0 || c >= layer.used));
    }

    /// Cards that together cannot hold the experts give no split.
    #[test]
    fn no_split_when_the_cards_cannot_hold_the_experts() {
        let layer = ExpertLayer {
            n_experts: 128,
            used: 8,
            expert_bytes: 100_000_000,
            row_bytes: 8_192,
        };
        let cards = [
            card(0, 900.0, None, 4.0),
            card(1, 450.0, Some((30.0, 12.0)), 4.0),
        ];
        assert_eq!(split_experts(&layer, &cards, 1), None);
    }
}
