//! Choosing a placement by MEASURING the placements, instead of estimating one.
//!
//! THE CIRCULARITY THIS BREAKS. A plan says how many blocks each card holds. What a
//! card must have free is the weights it was given PLUS the high-water mark of one
//! forward on it - and that high-water mark is a property of the plan, because the
//! stream, its rotary tables, its modulation and its attention scratch live on
//! whichever card is running the block. So the peak depends on the plan and the plan
//! depends on the peak. Every placement in this fleet cuts that loop the same way: it
//! replaces the peak with a formula that does not depend on the plan, and the loop
//! disappears because one of its two halves was thrown away.
//!
//! A dry run (see `crate::tensor::dry`) costs a header parse and a walk of the
//! shape algebra - a tenth of a second, no card, no file data. At that price the loop
//! does not need cutting: measure the candidate. Ask each placement what IT would hold,
//! on each of ITS cards, and keep the first one that fits. The peak is then a function
//! of the plan, which is what it always was, and the fixed point is reached by
//! iterating rather than by pretending it is not there.
//!
//! WHAT DECIDES, AND IN WHICH ORDER. Among the placements that fit:
//!
//! 1. NO BLOCK ON THE HOST. A block on the host is minutes per denoise step, and no
//!    error is raised - the request simply never finishes while the client waits. That
//!    is worse than any GPU-side arrangement, so a plan that spills is not considered
//!    against one that does not; the solver never emits a host segment at all and
//!    reports "nothing fits" instead, leaving the caller its existing fallback.
//! 2. THE FEWEST CARDS. Splitting a stack that fits one card is not free: the blocks
//!    form a chain, so the cards run in SEQUENCE and gain nothing, while every segment
//!    boundary pays a device-to-device copy of the stream on the hot path, on every
//!    step. One card that fits beats two that fit.
//! 3. THE FASTEST CARD. Between two placements using the same number of cards, prefer
//!    the ones earlier in the caller's order - which is the probe's order, ranked by
//!    measured compute throughput, never by index and never by free VRAM.
//!
//! The enumeration below is written in that order and stops at the first fit, so the
//! order IS the code rather than a comparator that has to agree with a comment.
//!
//! THE MARGIN IS NOT DECIDED HERE. What a dry run counts is the live set the tensor
//! layer sees: a LOWER bound, silent about the driver's rounding, about fragmentation
//! and about whatever a kernel allocates below that layer. How much to add on top can
//! only be learnt by making renders fail on purpose, which is not something this module
//! may do to a machine somebody is using. So it is a PARAMETER: the caller supplies it,
//! and this file has no opinion about its value.

use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};

/// What a candidate placement would hold on ONE device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceLoad {
    pub kind: DeviceKind,
    /// Resident once the model is built: this device's share of the blocks, plus
    /// whatever else the loader puts on it (the stem rides on the first device).
    pub weights: u64,
    /// The high-water mark of one forward on this device, beyond its weights.
    pub forward: u64,
}

impl DeviceLoad {
    /// What this device must have free for the placement to run, before any margin.
    pub fn peak(&self) -> u64 {
        self.weights.saturating_add(self.forward)
    }
}

/// What a candidate placement would hold, device by device.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PlanLoad {
    pub devices: Vec<DeviceLoad>,
}

impl PlanLoad {
    pub fn on(&self, kind: DeviceKind) -> Option<&DeviceLoad> {
        self.devices.iter().find(|d| d.kind == kind)
    }

    /// A line for the log: what each device holds, in the plan's own order.
    pub fn describe(&self) -> String {
        self.devices
            .iter()
            .map(|d| {
                format!(
                    "{} {:.2}+{:.2}={:.2} GB",
                    d.kind,
                    d.weights as f64 / 1e9,
                    d.forward as f64 / 1e9,
                    d.peak() as f64 / 1e9
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// A placement and what it was measured to hold.
#[derive(Debug, Clone)]
pub struct Solution {
    pub plan: HeteroPlan,
    pub load: PlanLoad,
    /// Dry runs performed to get here - the cost of the decision, so it can be seen.
    pub measurements: usize,
}

/// Segments as `CUDA(0):0-15 CUDA(1):15-30`, for the log line and for comparing two
/// placements by eye.
pub fn describe_plan(plan: &HeteroPlan) -> String {
    plan.segments
        .iter()
        .map(|s| format!("{}:{}-{}", s.kind, s.layer_start, s.layer_end))
        .collect::<Vec<_>>()
        .join(" ")
}

/// All `total_layers` blocks on one card.
pub fn whole_on(total_layers: usize, idx: usize, free: u64) -> HeteroPlan {
    HeteroPlan {
        segments: vec![HeteroSegment {
            kind: DeviceKind::Cuda(idx),
            layer_start: 0,
            layer_end: total_layers,
            free_memory_bytes: free,
        }],
        total_layers,
    }
}

/// A first guess at a split across `cards`: blocks in proportion to free VRAM.
///
/// A GUESS, and only that - it is the placement about to be measured, not the one
/// about to be trusted. Free VRAM is a fair opening bid because it is the only thing
/// known before the first measurement; what the measurement says then supersedes it
/// through [`reapportion`].
pub fn spread(total_layers: usize, cards: &[(usize, u64)]) -> HeteroPlan {
    let counts = share_out(total_layers, cards);
    segments_from(&counts, cards, total_layers)
}

/// Blocks per card, proportional to the second element, every card getting at least
/// one and the last taking the remainder.
fn share_out(total_layers: usize, cards: &[(usize, u64)]) -> Vec<usize> {
    let mut counts = vec![0usize; cards.len()];
    if cards.is_empty() || total_layers == 0 {
        return counts;
    }
    let sum: u128 = cards.iter().map(|(_, f)| *f as u128).sum();
    let mut assigned = 0usize;
    for (n, (_, free)) in cards.iter().enumerate() {
        let left = cards.len() - n - 1;
        let want = if n + 1 == cards.len() || sum == 0 {
            total_layers - assigned
        } else {
            let share = (total_layers as u128 * *free as u128 / sum) as usize;
            share.clamp(1, (total_layers - assigned).saturating_sub(left).max(1))
        };
        counts[n] = want.min(total_layers - assigned);
        assigned += counts[n];
    }
    counts
}

/// Contiguous segments from a per-card block count, dropping cards given nothing -
/// an empty segment is a card the plan does not use, and it must not show up as one.
fn segments_from(counts: &[usize], cards: &[(usize, u64)], total_layers: usize) -> HeteroPlan {
    let mut segments = Vec::new();
    let mut at = 0usize;
    for (n, count) in counts.iter().enumerate() {
        if *count == 0 {
            continue;
        }
        segments.push(HeteroSegment {
            kind: DeviceKind::Cuda(cards[n].0),
            layer_start: at,
            layer_end: at + count,
            free_memory_bytes: cards[n].1,
        });
        at += count;
    }
    HeteroPlan {
        segments,
        total_layers,
    }
}

/// Whether every device in a measured placement stays inside its budget.
pub fn fits(load: &PlanLoad, cards: &[(usize, u64)], margin: u64) -> bool {
    load.devices.iter().all(|d| match d.kind {
        DeviceKind::Cuda(idx) => cards
            .iter()
            .find(|(i, _)| *i == idx)
            .is_some_and(|(_, free)| d.peak().saturating_add(margin) <= *free),
        // A placement that reaches the host is not one this solver proposes, and a
        // measurement that reports one cannot be judged against a VRAM budget.
        _ => false,
    })
}

/// The next placement to try, read off the one just measured.
///
/// The measurement gives, per card, what its blocks weigh and what a forward on it
/// holds. Divide the first by the blocks it was given and the card states its own
/// per-block price; subtract the forward and the margin from its budget and it states
/// how many blocks it can hold. That is the whole step - the plan the peak implies,
/// where the previous round used the plan the FREE MEMORY implied.
///
/// `None` when the cards cannot hold the model however the blocks are arranged: the
/// caller then tries more cards rather than spilling to the host.
fn reapportion(
    total_layers: usize,
    cards: &[(usize, u64)],
    margin: u64,
    load: &PlanLoad,
    previous: &HeteroPlan,
) -> Option<HeteroPlan> {
    let mut counts = vec![0usize; cards.len()];
    let mut left = total_layers;
    for (n, (idx, free)) in cards.iter().enumerate() {
        let d = load.on(DeviceKind::Cuda(*idx))?;
        let held = previous
            .segments
            .iter()
            .filter(|s| s.kind == DeviceKind::Cuda(*idx))
            .map(|s| s.layer_end - s.layer_start)
            .sum::<usize>();
        if held == 0 {
            continue;
        }
        // The stem (embedders, refiners, the way back out) rides on one card and is
        // counted in its weights, so this card's per-block price comes out high and it
        // is offered fewer blocks. Erring that way is the safe one: the alternative
        // over-fills the card that already carries the most.
        let per_block = (d.weights / held as u64).max(1);
        let room = free.saturating_sub(d.forward.saturating_add(margin));
        let cap = (room / per_block) as usize;
        counts[n] = cap.min(left);
        left -= counts[n];
    }
    if left > 0 {
        return None;
    }
    Some(segments_from(&counts, cards, total_layers))
}

/// Two placements that put the same blocks on the same devices.
fn same_plan(a: &HeteroPlan, b: &HeteroPlan) -> bool {
    a.segments.len() == b.segments.len()
        && a.segments.iter().zip(b.segments.iter()).all(|(x, y)| {
            x.kind == y.kind && x.layer_start == y.layer_start && x.layer_end == y.layer_end
        })
}

/// How many times a split may be re-derived from its own measurement before the
/// solver gives up on that set of cards. Two rounds is what a converging sequence
/// needs (guess, correct); the third only ever catches a rounding oscillation, and a
/// placement that has not settled by then is one where the cards are too close to the
/// model's size for the answer to be stable.
const MAX_REAPPORTIONMENTS: usize = 3;

/// The placement to use for `total_layers` blocks, chosen by measuring candidates.
///
/// `cuda_fastest_first` is `(device index, free bytes)` in the probe's order - ranked
/// by measured compute throughput, which is what makes "prefer the earlier card" mean
/// "prefer the faster card". `margin` is what to add to every measured peak; see the
/// module note on why this file does not choose it.
///
/// `measure` runs one dry pass of a candidate and reports what it would hold per
/// device. Returning `None` (a checkpoint the dry pass cannot walk, a forward that
/// errored) makes the candidate unusable rather than making the whole decision fail.
///
/// `None` overall means no arrangement of these cards holds the model within the
/// margin: the caller keeps whatever it would have done, which is the only honest
/// answer a measurement can give about a placement it has ruled out.
pub fn solve<M>(
    total_layers: usize,
    cuda_fastest_first: &[(usize, u64)],
    margin: u64,
    measure: &mut M,
) -> Option<Solution>
where
    M: FnMut(&HeteroPlan) -> Option<PlanLoad>,
{
    if total_layers == 0 || cuda_fastest_first.is_empty() {
        return None;
    }
    let mut measurements = 0usize;

    // ONE card, fastest first. The best outcome there is: nothing on the host, no
    // boundary copy, and the hot stack undivided on the quickest card that holds it.
    for &(idx, free) in cuda_fastest_first {
        let plan = whole_on(total_layers, idx, free);
        let Some(load) = measure(&plan) else { continue };
        measurements += 1;
        if fits(&load, cuda_fastest_first, margin) {
            return Some(Solution {
                plan,
                load,
                measurements,
            });
        }
    }

    // Then two cards, then three: the fastest k of them, with the split re-derived
    // from what each round measured until it settles or the cards run out of room.
    for k in 2..=cuda_fastest_first.len() {
        let cards = &cuda_fastest_first[..k];
        let mut plan = spread(total_layers, cards);
        for _ in 0..MAX_REAPPORTIONMENTS {
            let Some(load) = measure(&plan) else { break };
            measurements += 1;
            if fits(&load, cuda_fastest_first, margin) {
                return Some(Solution {
                    plan,
                    load,
                    measurements,
                });
            }
            let Some(next) = reapportion(total_layers, cards, margin, &load, &plan) else {
                break;
            };
            if same_plan(&next, &plan) {
                break;
            }
            plan = next;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gb(n: f64) -> u64 {
        (n * 1e9) as u64
    }

    /// A checkpoint that answers the way the Z-Image dry run answers: the block
    /// weights follow the blocks, the stem rides on the plan's FIRST device, and every
    /// device running a block holds one forward's transients (they are what the stream
    /// and its tables cost, and the stream visits every device).
    ///
    /// The recorded figures this is set up with come from the dry run against the real
    /// checkpoint: 12.31 GB resident, 1.61 GB of forward at 1024 square and 0.48 GB at
    /// 512 square.
    fn checkpoint(
        blocks: usize,
        weights: u64,
        stem: u64,
        forward: u64,
    ) -> impl FnMut(&HeteroPlan) -> Option<PlanLoad> {
        let per_block = (weights - stem) / blocks as u64;
        move |plan: &HeteroPlan| {
            let devices = plan
                .segments
                .iter()
                .enumerate()
                .map(|(n, s)| DeviceLoad {
                    kind: s.kind,
                    weights: per_block * (s.layer_end - s.layer_start) as u64
                        + if n == 0 { stem } else { 0 },
                    forward,
                })
                .collect();
            Some(PlanLoad { devices })
        }
    }

    /// Z-Image at 1024 square: 12.31 GB of weights and 1.61 GB of forward against two
    /// cards of 16.5 GB. Nothing goes to the host - and nothing goes to the SECOND
    /// card either, because one card holds the whole stack and a split would pay a
    /// copy per boundary per step for nothing.
    #[test]
    fn two_cards_of_16_5_gb_keep_every_block_off_the_host() {
        let cards = [(0usize, gb(16.5)), (1usize, gb(16.5))];
        let mut m = checkpoint(30, gb(12.31), gb(2.0), gb(1.61));
        let s =
            solve(30, &cards, 0, &mut m).expect("no placement for a model that fits twice over");
        assert!(
            !s.plan
                .segments
                .iter()
                .any(|g| matches!(g.kind, DeviceKind::Cpu)),
            "a block went to the host with {} of free VRAM per card: {}",
            "16.5 GB",
            describe_plan(&s.plan)
        );
        assert_eq!(s.plan.segments.len(), 1, "split a stack that fits one card");
        assert_eq!(
            s.plan.segments[0].kind,
            DeviceKind::Cuda(0),
            "not the fastest card"
        );
    }

    /// The fastest card is busy, the second is free. One card still beats two, so the
    /// whole stack goes to the second card rather than straddling both.
    #[test]
    fn a_busy_fastest_card_moves_the_stack_whole_rather_than_splitting_it() {
        let cards = [(0usize, gb(6.0)), (1usize, gb(16.5))];
        let mut m = checkpoint(30, gb(12.31), gb(2.0), gb(1.61));
        let s = solve(30, &cards, 0, &mut m).expect("no placement while a card was free");
        assert_eq!(s.plan.segments.len(), 1, "{}", describe_plan(&s.plan));
        assert_eq!(s.plan.segments[0].kind, DeviceKind::Cuda(1));
    }

    /// No single card holds it and both together do: the blocks are SPLIT, and not one
    /// of them reaches the host.
    #[test]
    fn a_stack_no_card_holds_is_split_across_cards_never_spilled() {
        let cards = [(0usize, gb(8.5)), (1usize, gb(8.5))];
        let mut m = checkpoint(30, gb(12.31), gb(2.0), gb(1.61));
        let s = solve(30, &cards, 0, &mut m).expect("two cards with room found nothing");
        assert_eq!(s.plan.segments.len(), 2, "{}", describe_plan(&s.plan));
        assert!(!s
            .plan
            .segments
            .iter()
            .any(|g| matches!(g.kind, DeviceKind::Cpu)));
        assert_eq!(s.plan.total_layers, 30);
        assert_eq!(
            s.plan
                .segments
                .iter()
                .map(|g| g.layer_end - g.layer_start)
                .sum::<usize>(),
            30,
            "the split lost blocks: {}",
            describe_plan(&s.plan)
        );
    }

    /// The card carrying the stem is offered FEWER blocks, because the measurement
    /// charges the stem to it and the re-apportionment reads that off the measurement
    /// rather than being told about it.
    #[test]
    fn the_card_carrying_the_stem_is_given_fewer_blocks() {
        let cards = [(0usize, gb(8.0)), (1usize, gb(8.0))];
        let mut m = checkpoint(30, gb(12.31), gb(2.5), gb(1.2));
        let s = solve(30, &cards, 0, &mut m).expect("two 8 GB cards found nothing");
        assert_eq!(s.plan.segments.len(), 2, "{}", describe_plan(&s.plan));
        let first = s.plan.segments[0].layer_end - s.plan.segments[0].layer_start;
        let second = s.plan.segments[1].layer_end - s.plan.segments[1].layer_start;
        assert!(
            first < second,
            "the stem card took {first} blocks against {second}: {}",
            describe_plan(&s.plan)
        );
    }

    /// Cards that cannot hold it however it is arranged get NO answer - and in
    /// particular not an answer with blocks on the host, which is the placement this
    /// solver exists to stop being reached by accident.
    #[test]
    fn cards_too_small_report_nothing_rather_than_spilling() {
        let cards = [(0usize, gb(2.0)), (1usize, gb(2.0))];
        let mut m = checkpoint(30, gb(12.31), gb(2.0), gb(1.61));
        assert!(solve(30, &cards, 0, &mut m).is_none());
    }

    /// The margin is the caller's, and it MOVES the answer: the same cards and the
    /// same checkpoint go from one card to two when more is held back per card. That
    /// is why it is a parameter and not a constant in this file.
    #[test]
    fn the_margin_is_a_parameter_and_it_changes_the_answer() {
        let cards = [(0usize, gb(15.0)), (1usize, gb(15.0))];
        let mut m = checkpoint(30, gb(12.31), gb(2.0), gb(1.61));
        let tight = solve(30, &cards, 0, &mut m).expect("nothing fit at no margin");
        assert_eq!(tight.plan.segments.len(), 1);
        let mut m = checkpoint(30, gb(12.31), gb(2.0), gb(1.61));
        let loose = solve(30, &cards, gb(2.0), &mut m).expect("nothing fit at a 2 GB margin");
        assert_eq!(
            loose.plan.segments.len(),
            2,
            "{}",
            describe_plan(&loose.plan)
        );
    }

    /// AT 512 SQUARE THE FIXED POINT AGREES WITH THE LOADER AS IT STANDS.
    ///
    /// This is the gate on switching the decision over: today's rule is "the first
    /// card in probe order whose free VRAM holds the weights plus the request's
    /// reserve, undivided". Measured, the same request picks the same card - so
    /// nothing moves the day the measurement starts deciding.
    #[cfg(feature = "image")]
    #[test]
    fn at_512_square_the_fixed_point_picks_what_the_loader_picks_today() {
        let cards = [(0usize, gb(16.5)), (1usize, gb(16.5))];
        let weights = gb(12.31);
        // What `zimage_runtime_demand(512, 512)` returns, and what the loader adds to
        // the weights before asking whether a card holds the stack.
        let reserve = crate::inference::engine::image_engine::zimage_runtime_demand(512, 512);
        let today = cards
            .iter()
            .find(|(_, free)| *free >= weights + reserve)
            .map(|(idx, _)| DeviceKind::Cuda(*idx))
            .expect("today's rule placed nothing");
        let mut m = checkpoint(30, weights, gb(2.0), gb(0.48));
        let s = solve(30, &cards, 0, &mut m).expect("the fixed point placed nothing");
        assert_eq!(s.plan.segments.len(), 1, "{}", describe_plan(&s.plan));
        assert_eq!(
            s.plan.segments[0].kind, today,
            "the two rules chose different cards"
        );
    }

    /// A card that gets no blocks is not a segment. A plan naming a device it does not
    /// use makes the loader open a context on it - real memory, on a card the
    /// placement decided to leave alone.
    #[test]
    fn a_card_given_nothing_is_not_in_the_plan() {
        let plan = spread(4, &[(0, gb(10.0)), (1, 0)]);
        assert_eq!(
            plan.segments.len(),
            2,
            "a zero-budget card should still get its one block"
        );
        let plan = segments_from(&[4, 0], &[(0, gb(10.0)), (1, 0)], 4);
        assert_eq!(plan.segments.len(), 1);
        assert_eq!(plan.segments[0].kind, DeviceKind::Cuda(0));
    }
}
