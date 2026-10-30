//! Which experts a host keeps in memory: the experts each token reads from disk, for a policy and
//! a number of slots per layer, on a routing drawn from an expert profile. No model, no disk and
//! no card is touched: the bench replays the routing through each policy and turns the experts
//! it misses into the time the disk takes to read them.
//!
//! The routing is drawn independently for each token from the profile's counts, so the bench
//! says what a policy gains from the heat of the experts and nothing of the order they come in:
//! a recency policy is judged here without the locality a real answer adds.

use super::{Context, Measured};
use crate::inference::generic_transformer::expert_profile::ExpertProfile;
use std::collections::{HashMap, HashSet};

/// Experts a token routes to in a layer.
const ACTIVE: usize = 6;
/// Tokens whose misses make one sample.
const BLOCK: usize = 200;

/// How a layer's slots are filled and emptied.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Policy {
    /// The slots hold the most counted experts of the profile and never change.
    Static,
    /// The least recently routed expert makes room.
    Recency,
    /// The least often routed expert makes room, the least recent among equals.
    Frequency,
}

impl Policy {
    fn name(self) -> &'static str {
        match self {
            Policy::Static => "static, hottest by the profile",
            Policy::Recency => "recency",
            Policy::Frequency => "frequency",
        }
    }
}

/// One layer's tier: the experts kept and what a policy needs to choose a victim.
struct Tier {
    policy: Policy,
    slots: usize,
    kept: HashSet<usize>,
    last: HashMap<usize, u64>,
    seen: HashMap<usize, u64>,
}

impl Tier {
    fn new(policy: Policy, slots: usize, hottest: &[usize]) -> Self {
        let kept = if policy == Policy::Static {
            hottest.iter().copied().take(slots).collect()
        } else {
            HashSet::new()
        };
        Self {
            policy,
            slots,
            kept,
            last: HashMap::new(),
            seen: HashMap::new(),
        }
    }

    /// Route to `expert` at `clock`: true when it was kept, otherwise admitted when the policy
    /// admits.
    fn route(&mut self, expert: usize, clock: u64) -> bool {
        *self.seen.entry(expert).or_insert(0) += 1;
        self.last.insert(expert, clock);
        if self.kept.contains(&expert) {
            return true;
        }
        if self.policy == Policy::Static || self.slots == 0 {
            return false;
        }
        if self.kept.len() >= self.slots {
            let victim = self
                .kept
                .iter()
                .copied()
                .min_by_key(|e| match self.policy {
                    Policy::Recency => (0, self.last[e]),
                    _ => (self.seen[e], self.last[e]),
                })
                .expect("a full tier holds an expert");
            self.kept.remove(&victim);
        }
        self.kept.insert(expert);
        false
    }
}

/// A small generator: the draws only need to be repeatable.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

/// The experts a read-ahead guess names for a token that routes to `chosen`: each is named with
/// the probability `hit`, and a miss names a draw of the layer's heat in its place.
fn guess(chosen: &[usize], hit: f64, weights: &[f64], rng: &mut Rng) -> Vec<usize> {
    let mut named: Vec<usize> = Vec::with_capacity(chosen.len());
    for &e in chosen {
        if rng.next() < hit {
            named.push(e);
        } else {
            let other = draw(weights, rng)
                .into_iter()
                .find(|x| !named.contains(x) && !chosen.contains(x))
                .unwrap_or(e);
            named.push(other);
        }
    }
    named.sort_unstable();
    named.dedup();
    named
}

/// The `ACTIVE` distinct experts a token routes to in a layer whose heat is `weights`: the
/// largest of each expert's log weight plus a Gumbel draw, which samples without replacement in
/// proportion to the weights.
fn draw(weights: &[f64], rng: &mut Rng) -> Vec<usize> {
    let mut keyed: Vec<(f64, usize)> = weights
        .iter()
        .enumerate()
        .map(|(e, w)| (w.ln() - (-rng.next().ln()).ln(), e))
        .collect();
    keyed.select_nth_unstable_by(ACTIVE, |a, b| b.0.total_cmp(&a.0));
    keyed[..ACTIVE].iter().map(|&(_, e)| e).collect()
}

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let path = cx
        .options
        .profile
        .as_ref()
        .ok_or("cache-policy needs --profile, an expert profile")?;
    let profile = ExpertProfile::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let expert_bytes = cx.shapes.expert_bytes();
    if expert_bytes == 0 {
        return Err(
            "cache-policy needs an expert size: --shapes-from or --projection-bytes".into(),
        );
    }
    let card_slots = cx.options.card_slots;
    let disk = cx.options.disk_gbs * 1e9;
    let tokens = cx.options.tokens.max(BLOCK);

    // Per layer: the weights of the experts a host tier may keep, the cards' share left out,
    // and the order the profile ranks them in.
    let mut layers = Vec::new();
    for l in 0..profile.layers {
        let counts = profile.layer(l);
        let mut order: Vec<usize> = (0..profile.experts).collect();
        order.sort_by_key(|&e| std::cmp::Reverse(counts[e]));
        let on_cards: HashSet<usize> = order.iter().copied().take(card_slots).collect();
        let weights: Vec<f64> = counts.iter().map(|&c| f64::from(c).max(0.5)).collect();
        let hosted: Vec<usize> = order
            .into_iter()
            .filter(|e| !on_cards.contains(e))
            .collect();
        layers.push((weights, on_cards, hosted));
    }

    let mut rows = Vec::new();
    let hits: Vec<f64> = if cx.options.guess_hit.is_empty() {
        vec![1.0]
    } else {
        cx.options.guess_hit.clone()
    };
    for &slots in &cx.options.slots {
        for policy in [Policy::Static, Policy::Recency, Policy::Frequency] {
            for &hit in &hits {
                let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
                let mut tiers: Vec<Tier> = layers
                    .iter()
                    .map(|(_, _, hosted)| Tier::new(policy, slots, hosted))
                    .collect();
                let (mut samples, mut in_block, mut routed, mut missed, mut wasted) =
                    (Vec::new(), 0u64, 0u64, 0u64, 0u64);
                for t in 0..tokens {
                    for (l, (weights, on_cards, _)) in layers.iter().enumerate() {
                        let chosen = draw(weights, &mut rng);
                        // The read-ahead reads what it names that no card and no tier holds,
                        // and reads for nothing what the token then does not route to.
                        for e in guess(&chosen, hit, weights, &mut rng) {
                            if !on_cards.contains(&e)
                                && !tiers[l].kept.contains(&e)
                                && !chosen.contains(&e)
                            {
                                wasted += 1;
                                in_block += 1;
                            }
                        }
                        for e in chosen {
                            if on_cards.contains(&e) {
                                continue;
                            }
                            routed += 1;
                            if !tiers[l].route(e, t as u64) {
                                in_block += 1;
                                missed += 1;
                            }
                        }
                    }
                    if (t + 1) % BLOCK == 0 {
                        let bytes = in_block as f64 * expert_bytes as f64 / BLOCK as f64;
                        samples.push(bytes / disk * 1e6);
                        in_block = 0;
                    }
                }
                let per_token = (missed + wasted) as f64 * expert_bytes as f64 / tokens as f64;
                let waste = wasted as f64 * expert_bytes as f64 / tokens as f64;
                rows.push(Measured {
                    variant: format!("{slots} slots, {}, guess hit {hit}", policy.name()),
                    bytes: Some(per_token as u64),
                    samples_us: samples,
                    note: Some(format!(
                        "{:.1} percent of the host's routed experts kept, {:.0} MB read per token of which {:.0} MB for nothing, {:.1} ms per token at {} GB/s",
                        100.0 * (routed - missed) as f64 / routed.max(1) as f64,
                        per_token / 1e6,
                        waste / 1e6,
                        per_token / disk * 1e3,
                        cx.options.disk_gbs
                    )),
                });
            }
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tier of one slot keeps what it last saw under recency and keeps the hot one under
    /// frequency.
    #[test]
    fn the_policies_choose_different_victims() {
        let mut recency = Tier::new(Policy::Recency, 1, &[]);
        let mut frequency = Tier::new(Policy::Frequency, 1, &[]);
        for (clock, e) in [0, 0, 0, 1].into_iter().enumerate() {
            recency.route(e, clock as u64);
            frequency.route(e, clock as u64);
        }
        assert!(recency.kept.contains(&1));
        assert!(
            frequency.kept.contains(&1),
            "the newcomer takes the only slot"
        );
        // With room for the hot expert and one more, the cold newcomer replaces the cold one.
        let mut frequency = Tier::new(Policy::Frequency, 2, &[]);
        for (clock, e) in [0, 0, 0, 1, 2].into_iter().enumerate() {
            frequency.route(e, clock as u64);
        }
        assert!(frequency.kept.contains(&0) && frequency.kept.contains(&2));
    }

    /// A static tier never changes, and keeps the experts it was given in order.
    #[test]
    fn a_static_tier_keeps_the_hottest_it_was_given() {
        let mut t = Tier::new(Policy::Static, 2, &[5, 3, 9]);
        assert!(t.route(5, 0) && t.route(3, 1));
        assert!(!t.route(9, 2));
        assert!(!t.kept.contains(&9));
    }

    /// A draw names distinct experts, and the heavy ones come up more.
    #[test]
    fn a_draw_follows_the_weights() {
        let mut rng = Rng(7);
        let weights: Vec<f64> = (0..32).map(|e| if e < 4 { 1000.0 } else { 1.0 }).collect();
        let mut hot = 0;
        for _ in 0..200 {
            let d = draw(&weights, &mut rng);
            assert_eq!(d.iter().collect::<HashSet<_>>().len(), ACTIVE);
            hot += d.iter().filter(|&&e| e < 4).count();
        }
        assert!(hot > 700, "{hot} of 800 draws were the heavy experts");
    }
}
