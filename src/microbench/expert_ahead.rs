//! Experts of the next layer copied to the card while the current layer runs: a decode of
//! `--layers` layers whose routed experts are all missing from the card, each layer taking its
//! `m` experts through a transit room from page-locked host memory. The variants differ in when
//! the next layer's experts start to move and in what names them:
//!
//! - nothing is fetched before the layer's own router has answered (the copy sits in the token);
//! - the experts the token before routed to are copied from the start of the layer before;
//! - the same set is copied once the layer before has crossed to the host;
//! - the router of the next layer is run early on the card, and its answer rides the crossing.
//!
//! A guess names each routed expert with the probability `--guess-hit`. A wrong name copies an
//! expert nobody uses, and the layer copies the ones it still lacks after its router.
//!
//! Every variant is swept over the experts missing per layer and over the ratio of a layer's card
//! time to the time its copies take, so that the lead a copy has shows as a column of the table.
//! The routed experts are stood in for by a copy within card memory of the bytes a layer reads,
//! the layer's other work by a kernel that holds the compute stream. No expert's arithmetic runs.

use super::card::{copy_stream, device_bytes, Card, Pinned, Spin};
use super::stats::{run_for, timed, Summary};
use super::{Context, Fail, Measured};
use crate::tensor::gpu_runtime::driver::{CudaEvent, CudaSlice, CudaStream};
use std::sync::Arc;
use std::time::Duration;

/// Experts missing from the card in each layer, one sweep step each.
const MISSES: [usize; 4] = [1, 2, 4, 8];

/// A layer's card time over the time to copy its missing experts, one sweep step each.
const LEAD_RATIOS: [f64; 3] = [0.5, 1.0, 2.0];

/// Bytes of the router's answer that cross to the host in a layer.
const CHOICE_BYTES: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Variant {
    /// The copies start once the router of the layer has answered.
    None,
    /// The token before's set is copied from the start of the layer before.
    LastAtStart,
    /// The token before's set is copied once the layer before has crossed.
    LastAfterCrossing,
    /// The next layer's router runs on the card in the crossing of this one.
    EarlyRouter,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::None => "no prefetch",
            Variant::LastAtStart => "last token's set, from the start of the layer before",
            Variant::LastAfterCrossing => {
                "last token's set, after the crossing of the layer before"
            }
            Variant::EarlyRouter => "early router, after the crossing of the layer before",
        }
    }

    fn prefetches(self) -> bool {
        self != Variant::None
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

/// How many of `m` named experts a layer routes to, each named right with probability `hit`.
fn named_right(m: usize, hit: f64, rng: &mut Rng) -> usize {
    (0..m).filter(|_| rng.next() < hit).count()
}

/// What a layer copies after its router when `named` experts were copied before it and `right`
/// of them are routed to: the routed experts the guess missed.
fn copied_after(m: usize, named: usize, right: usize) -> usize {
    debug_assert!(right <= named.min(m));
    m - right
}

/// The expert slots a transit buffer needs: the guess and the experts it missed, each up to `m`.
fn slots_for(m: usize) -> usize {
    2 * m
}

/// The card, its streams, and the buffers one decode of the stack moves experts through.
struct Rig {
    compute: Arc<CudaStream>,
    copy: Arc<CudaStream>,
    /// One expert's 3 projections in page-locked memory, the source of every copy.
    source: Vec<Pinned>,
    /// The sizes of those projections.
    sizes: Vec<usize>,
    /// The transit room: 2 buffers of 3 stacks, consecutive layers taking turns.
    room: [Vec<CudaSlice<u8>>; 2],
    /// Where the card-side stand-in of the experts' work writes.
    sink: Vec<CudaSlice<u8>>,
    choice_card: CudaSlice<u8>,
    choice_host: Pinned,
    /// Recorded on the compute stream when a buffer's last reader is queued.
    free: [CudaEvent; 2],
    /// Recorded on the copy stream when a buffer's copies are queued.
    ready: [CudaEvent; 2],
    layers: usize,
    rng: Rng,
}

impl Rig {
    fn new(cx: &Context, card: &Card) -> Result<Self, String> {
        let compute = card.dev.stream().clone();
        let copy = copy_stream(card)?;
        let sizes: Vec<usize> = cx
            .shapes
            .projections
            .iter()
            .map(|p| p.bytes as usize)
            .collect();
        let most = MISSES.iter().copied().max().unwrap_or(1);
        let slots = slots_for(most);
        let mut source = Vec::new();
        for &n in &sizes {
            let mut p = Pinned::new(card, n)?;
            for (i, b) in p.as_mut_slice().iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
            source.push(p);
        }
        let stacks = |n: usize| -> Result<Vec<CudaSlice<u8>>, String> {
            sizes
                .iter()
                .map(|&s| device_bytes(&compute, s * n))
                .collect()
        };
        let room = [stacks(slots)?, stacks(slots)?];
        let sink = stacks(most)?;
        let event = || card.dev.new_event().map_err(|e| format!("event: {e:?}"));
        let rig = Self {
            compute,
            copy,
            source,
            sizes,
            room,
            sink,
            choice_card: device_bytes(&card.dev.stream().clone(), CHOICE_BYTES)?,
            choice_host: Pinned::new(card, CHOICE_BYTES)?,
            free: [event()?, event()?],
            ready: [event()?, event()?],
            layers: cx.options.layers.max(2),
            rng: Rng(0x9e37_79b9_7f4a_7c15),
        };
        // An event that was never recorded would be a wait on nothing, but one recorded once
        // keeps the first token like the others.
        for e in rig.free.iter().chain(&rig.ready) {
            e.record(&rig.compute)
                .map_err(|e| format!("record: {e:?}"))?;
        }
        rig.compute
            .synchronize()
            .map_err(|e| format!("synchronise: {e:?}"))?;
        Ok(rig)
    }

    /// Queue `count` experts from the source into buffer `b`, from slot `at`, on the copy stream.
    fn copy_experts(&mut self, b: usize, at: usize, count: usize) -> Result<(), String> {
        for j in 0..self.sizes.len() {
            let n = self.sizes[j];
            for s in at..at + count {
                self.copy
                    .memcpy_htod(
                        self.source[j].as_slice(),
                        &mut self.room[b][j].slice_mut(s * n..(s + 1) * n),
                    )
                    .map_err(|e| format!("copy: {e:?}"))?;
            }
        }
        Ok(())
    }

    /// Queue the copy of the `m` experts a guess names for layer `layer`, behind the layer that
    /// read the same buffer before.
    fn prefetch(&mut self, layer: usize, m: usize) -> Result<(), String> {
        let b = layer % 2;
        self.copy
            .wait(&self.free[b])
            .map_err(|e| format!("wait: {e:?}"))?;
        self.copy_experts(b, 0, m)?;
        self.ready[b]
            .record(&self.copy)
            .map_err(|e| format!("record: {e:?}"))
    }

    /// One token through the stack. `hit` is the share of a layer's experts the guess names
    /// right, `lead` the card time of a layer before its router answers.
    fn token(
        &mut self,
        v: Variant,
        m: usize,
        hit: f64,
        lead: &Spin,
        early: &Spin,
    ) -> Result<(), String> {
        let right: Vec<usize> = (0..self.layers)
            .map(|_| named_right(m, hit, &mut self.rng))
            .collect();
        if v.prefetches() {
            self.prefetch(0, m)?;
        }
        for i in 0..self.layers {
            let b = i % 2;
            if v == Variant::LastAtStart && i + 1 < self.layers {
                self.prefetch(i + 1, m)?;
            }
            lead.launch(&self.compute)?;
            if v == Variant::EarlyRouter {
                early.launch(&self.compute)?;
            }
            // The layer's router answers on the host.
            self.compute
                .memcpy_dtoh(
                    &self.choice_card.slice(0..CHOICE_BYTES),
                    &mut self.choice_host.as_mut_slice()[..CHOICE_BYTES],
                )
                .map_err(|e| format!("choice: {e:?}"))?;
            self.compute
                .synchronize()
                .map_err(|e| format!("choice wait: {e:?}"))?;
            // What the guess did not name is copied now, behind what it did.
            let named = if v.prefetches() { m } else { 0 };
            let got = if v.prefetches() { right[i] } else { 0 };
            let after = copied_after(m, named, got);
            if !v.prefetches() {
                self.copy
                    .wait(&self.free[b])
                    .map_err(|e| format!("wait: {e:?}"))?;
            }
            self.copy_experts(b, named, after)?;
            self.ready[b]
                .record(&self.copy)
                .map_err(|e| format!("record: {e:?}"))?;
            if matches!(v, Variant::LastAfterCrossing | Variant::EarlyRouter) && i + 1 < self.layers
            {
                self.prefetch(i + 1, m)?;
            }
            // The experts run once their copies have landed.
            self.compute
                .wait(&self.ready[b])
                .map_err(|e| format!("wait: {e:?}"))?;
            for j in 0..self.sizes.len() {
                let n = self.sizes[j] * m;
                self.compute
                    .memcpy_dtod(
                        &self.room[b][j].slice(0..n),
                        &mut self.sink[j].slice_mut(0..n),
                    )
                    .map_err(|e| format!("experts: {e:?}"))?;
            }
            self.free[b]
                .record(&self.compute)
                .map_err(|e| format!("record: {e:?}"))?;
        }
        self.compute
            .synchronize()
            .map_err(|e| format!("token wait: {e:?}"))
    }
}

pub fn run(cx: &Context) -> Result<Vec<Measured>, String> {
    let card = cx.card.ok_or("expert-ahead needs a card")?;
    let target = Duration::from_secs_f64(cx.options.seconds);
    let mut rig = Rig::new(cx, card)?;
    let expert_bytes = cx.shapes.expert_bytes();
    let early = Spin::lasting(card, 0.0)?;
    let fail = Fail::default();
    let hits = &cx.options.guess_hit;
    if hits.iter().any(|h| !(0.0..=1.0).contains(h)) {
        return Err("--guess-hit takes shares from 0 to 1".to_string());
    }

    let mut rows = Vec::new();
    for &m in &MISSES {
        // The time the card takes to receive `m` experts, idle.
        let landed = run_for(
            target / 10,
            timed(|| {
                fail.ok(rig.prefetch(0, m));
                fail.ok(rig.ready[0].synchronize());
            }),
        );
        fail.check("copy of the experts")?;
        let copy_us = Summary::of(&landed).median_us;
        for &ratio in &LEAD_RATIOS {
            let lead = Spin::lasting(card, ratio * copy_us)?;
            let mut base_us = None;
            for v in [
                Variant::None,
                Variant::LastAtStart,
                Variant::LastAfterCrossing,
                Variant::EarlyRouter,
            ] {
                // A variant without a guess does not read the share.
                let shares: &[f64] = if v.prefetches() { hits } else { &[1.0] };
                for &hit in shares {
                    let samples = run_for(
                        target,
                        timed(|| fail.ok(rig.token(v, m, hit, &lead, &early))),
                    );
                    let name = format!(
                        "{m} missing, layer {:.1}x the copy, {}{}",
                        ratio,
                        v.name(),
                        if v.prefetches() {
                            format!(", {:.0}% named right", hit * 100.0)
                        } else {
                            String::new()
                        }
                    );
                    fail.check(&name)?;
                    let median = Summary::of(&samples).median_us;
                    let note = match (v, base_us) {
                        (Variant::None, _) => {
                            base_us = Some(median);
                            Some(format!("{:.0} us a layer", median / rig.layers as f64))
                        }
                        (_, Some(base)) => Some(format!(
                            "{:.2} of the token time without prefetch",
                            median / base
                        )),
                        _ => None,
                    };
                    let per_layer = if v.prefetches() {
                        m as f64 * (2.0 - hit)
                    } else {
                        m as f64
                    };
                    rows.push(Measured {
                        variant: name,
                        bytes: Some((per_layer * rig.layers as f64 * expert_bytes as f64) as u64),
                        samples_us: samples,
                        note,
                    });
                }
            }
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_guess_right_about_every_expert_leaves_no_copy_after_the_router() {
        assert_eq!(copied_after(4, 4, 4), 0);
        assert_eq!(copied_after(4, 4, 1), 3);
        assert_eq!(copied_after(4, 0, 0), 4);
    }

    #[test]
    fn the_room_holds_a_guess_and_every_expert_it_missed() {
        for m in MISSES {
            assert!(slots_for(m) >= m + copied_after(m, m, 0));
        }
    }

    #[test]
    fn the_share_named_right_follows_the_probability() {
        let mut rng = Rng(7);
        let n = 4000;
        let right: usize = (0..n).map(|_| named_right(8, 0.25, &mut rng)).sum();
        let share = right as f64 / (8 * n) as f64;
        assert!((share - 0.25).abs() < 0.02, "{share}");
        assert_eq!(named_right(8, 1.0, &mut rng), 8);
        assert_eq!(named_right(8, 0.0, &mut rng), 0);
    }
}
