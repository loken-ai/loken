//! Which stored variant each routed expert is read from, for a model served from several files
//! of one checkpoint, each quantised to its own format.
//!
//! The memory the placement keeps routed experts in is filled in routing-heat order, rank by
//! rank across the layers: the cards' room first, then the host tier. A tier that computes an
//! expert at the same speed whatever its format (a card) gives the experts it keeps the variant
//! with the most bytes, the most precise, while it has room for them. A tier whose compute reads
//! every byte of an expert at each use (the host's cores), and the disk, give the variant with
//! the fewest bytes: a precise expert there costs time at each use and room a hotter expert
//! would take.

/// A tier the plan fills: the bytes it keeps experts in, and whether an expert's format leaves
/// its compute time unchanged there, so the tier takes the precise variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tier {
    pub bytes: u64,
    pub precise: bool,
}

/// How the plan picks a variant for each expert.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VariantChoice {
    /// The precise variant for the experts kept in memory, the small one for those read from disk.
    #[default]
    Plan,
    /// The variant with the fewest bytes for every expert.
    Smallest,
    /// The variant with the most bytes for every expert.
    Largest,
}

/// The plan: per layer, the variant of each expert and how many experts each tier keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantPlan {
    /// Per layer, per expert id, the variant it is read from.
    pub chosen: Vec<Vec<usize>>,
    /// Per tier, per layer, the experts the tier keeps.
    pub kept: Vec<Vec<usize>>,
}

impl VariantPlan {
    /// How many experts are read from each variant, over every layer.
    pub fn per_variant(&self, variants: usize) -> Vec<usize> {
        let mut n = vec![0; variants];
        for &v in self.chosen.iter().flatten() {
            if let Some(c) = n.get_mut(v) {
                *c += 1;
            }
        }
        n
    }
}

/// Layer `layer`'s experts, hottest first: the ranked ones, then every expert the ranking does
/// not name, by id.
fn order(ranks: &[Vec<usize>], layer: usize, experts: usize) -> Vec<usize> {
    let mut seen = vec![false; experts];
    let mut out = Vec::with_capacity(experts);
    for &e in ranks.get(layer).into_iter().flatten() {
        if e < experts && !seen[e] {
            seen[e] = true;
            out.push(e);
        }
    }
    out.extend((0..experts).filter(|&e| !seen[e]));
    out
}

/// Per layer, the experts ranked by the routing counts `counts` (`[layers * experts]`, layer by
/// layer), most routed first, ties by id; the experts never counted follow in the order of
/// `prior`. `None` when nothing was counted.
pub fn ranks_from_counts(
    counts: &[u32],
    layers: usize,
    experts: usize,
    prior: &[Vec<usize>],
) -> Option<Vec<Vec<usize>>> {
    if counts.len() != layers * experts || counts.iter().all(|&c| c == 0) {
        return None;
    }
    Some(
        (0..layers)
            .map(|l| {
                let row = &counts[l * experts..(l + 1) * experts];
                let mut counted: Vec<usize> = (0..experts).filter(|&e| row[e] > 0).collect();
                counted.sort_by_key(|&e| (std::cmp::Reverse(row[e]), e));
                let rest = prior
                    .get(l)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|&e| e < experts && row[e] == 0);
                counted.extend(rest);
                counted
            })
            .collect(),
    )
}

/// The plan for `layers` layers of `experts` routed experts in `variants` variants, `bytes(layer,
/// expert, variant)` each. `ranks` orders each layer's experts by routing heat, hottest first;
/// `tiers` are where the placement keeps experts, in the order it fills them.
pub fn plan(
    layers: usize,
    experts: usize,
    variants: usize,
    bytes: &dyn Fn(usize, usize, usize) -> u64,
    ranks: &[Vec<usize>],
    tiers: &[Tier],
    choice: VariantChoice,
) -> VariantPlan {
    let by_size = |l: usize, e: usize, largest: bool| -> usize {
        let key = |&v: &usize| bytes(l, e, v);
        let vs = 0..variants.max(1);
        if largest {
            // The first of equal sizes, so a file's own order breaks ties.
            vs.rev().max_by_key(key).unwrap_or(0)
        } else {
            vs.min_by_key(key).unwrap_or(0)
        }
    };
    // Whether a tier's experts, and those read from disk, take the precise variant.
    let precise_in = |t: &Tier| match choice {
        VariantChoice::Plan => t.precise,
        VariantChoice::Smallest => false,
        VariantChoice::Largest => true,
    };
    let on_disk = choice == VariantChoice::Largest;
    let mut chosen: Vec<Vec<usize>> = (0..layers)
        .map(|l| (0..experts).map(|e| by_size(l, e, on_disk)).collect())
        .collect();
    let mut kept = vec![vec![0usize; layers]; tiers.len()];
    let orders: Vec<Vec<usize>> = (0..layers).map(|l| order(ranks, l, experts)).collect();
    let mut left: Vec<u64> = tiers.iter().map(|t| t.bytes).collect();
    let mut tier = 0usize;
    'fill: for rank in 0..experts {
        for (l, ids) in orders.iter().enumerate() {
            let e = ids[rank];
            // A tier that cannot take this expert is full: the next one is filled from here on.
            let v = loop {
                let Some(t) = tiers.get(tier) else {
                    break 'fill;
                };
                let v = by_size(l, e, precise_in(t));
                if left[tier] >= bytes(l, e, v) {
                    break v;
                }
                tier += 1;
            };
            left[tier] -= bytes(l, e, v);
            kept[tier][l] += 1;
            chosen[l][e] = v;
        }
    }
    VariantPlan { chosen, kept }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A card tier and a host tier of these bytes, the host taking the precise variant or not.
    fn both(card: u64, host: u64, host_precise: bool) -> [Tier; 2] {
        [
            Tier {
                bytes: card,
                precise: true,
            },
            Tier {
                bytes: host,
                precise: host_precise,
            },
        ]
    }

    #[test]
    fn a_tier_whose_compute_reads_every_byte_keeps_the_small_variant() {
        let ranks = vec![vec![2, 0], vec![3]];
        // Two precise experts on the cards; the host tier keeps small ones, two of them.
        let p = plan(
            2,
            4,
            2,
            &sizes,
            &ranks,
            &both(28, 20, false),
            VariantChoice::Plan,
        );
        assert_eq!(p.kept, vec![vec![1, 1], vec![1, 1]]);
        assert_eq!(p.chosen[0], vec![0, 0, 1, 0]);
        assert_eq!(p.chosen[1], vec![0, 0, 0, 1]);
    }

    /// Two variants, 10 and 14 bytes an expert; two layers of four experts.
    fn sizes(_l: usize, _e: usize, v: usize) -> u64 {
        [10, 14][v]
    }

    #[test]
    fn the_hottest_experts_kept_take_the_precise_variant_and_the_rest_the_small_one() {
        let ranks = vec![vec![2, 0], vec![3]];
        // The cards hold three precise experts, the host tier one more.
        let p = plan(
            2,
            4,
            2,
            &sizes,
            &ranks,
            &both(42, 14, true),
            VariantChoice::Plan,
        );
        // Rank 0: layer 0 expert 2, layer 1 expert 3; rank 1: layer 0 expert 0 on the cards, then
        // layer 1 expert 0 (unranked, first by id) in the host tier.
        assert_eq!(p.kept, vec![vec![2, 1], vec![0, 1]]);
        assert_eq!(p.chosen[0], vec![1, 0, 1, 0]);
        assert_eq!(p.chosen[1], vec![1, 0, 0, 1]);
        assert_eq!(p.per_variant(2), vec![4, 4]);
    }

    #[test]
    fn no_budget_reads_every_expert_small() {
        let p = plan(2, 4, 2, &sizes, &[], &both(9, 0, true), VariantChoice::Plan);
        assert!(p.chosen.iter().flatten().all(|&v| v == 0));
        assert_eq!(p.kept, vec![vec![0, 0], vec![0, 0]]);
    }

    #[test]
    fn a_forced_choice_reads_every_expert_from_one_variant() {
        let ranks = vec![vec![1], vec![2]];
        let small = plan(
            2,
            4,
            2,
            &sizes,
            &ranks,
            &both(30, 30, true),
            VariantChoice::Smallest,
        );
        assert!(small.chosen.iter().flatten().all(|&v| v == 0));
        // Kept counts follow the forced variant's bytes: three a tier.
        assert_eq!(small.kept, vec![vec![2, 1], vec![1, 2]]);
        let large = plan(
            2,
            4,
            2,
            &sizes,
            &ranks,
            &both(30, 30, false),
            VariantChoice::Largest,
        );
        assert!(large.chosen.iter().flatten().all(|&v| v == 1));
        assert_eq!(large.kept, vec![vec![1, 1], vec![1, 1]]);
    }

    #[test]
    fn counts_rank_first_and_the_prior_orders_the_uncounted() {
        let prior = vec![vec![3, 0, 2], vec![1]];
        // A layer with no counts ranks from the prior alone.
        let r = ranks_from_counts(&[0, 5, 9, 5, 0, 0, 0, 0], 2, 4, &prior).unwrap();
        assert_eq!(r[0], vec![2, 1, 3, 0]);
        assert_eq!(r[1], vec![1]);
        let r = ranks_from_counts(&[0, 5, 9, 5, 0, 0, 1, 0], 2, 4, &prior).unwrap();
        assert_eq!(r[1], vec![2, 1]);
        assert_eq!(ranks_from_counts(&[0; 8], 2, 4, &prior), None);
    }

    #[test]
    fn the_largest_variant_is_picked_by_bytes_not_by_file_order() {
        // The precise file listed first.
        let swapped = |_l: usize, _e: usize, v: usize| -> u64 { [14, 10][v] };
        let tiers = [Tier {
            bytes: 14,
            precise: true,
        }];
        let p = plan(1, 2, 2, &swapped, &[vec![1]], &tiers, VariantChoice::Plan);
        assert_eq!(p.chosen[0], vec![1, 0]);
    }
}
