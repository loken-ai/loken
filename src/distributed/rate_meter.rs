//! What this node is actually worth, measured from its own work.
//!
//! Routing compares predicted completions, and the prediction is only as good as the rates it
//! multiplies. Publishing nothing makes every peer fall back to one pessimistic default, which
//! means a cluster of a workstation, a laptop and a mini-PC reads as three identical machines -
//! precisely the case clustering exists for. The decision then turns on load alone, and sends
//! work to whichever node happens to be idle rather than to whichever will finish first.
//!
//! So the numbers come from generations that already happened here. Nothing is estimated from
//! hardware: a card's nameplate says what it could do, not what this build achieves on this
//! model with this quantisation, and the gap between those is the whole subject of the
//! benchmark campaign.
//!
//! An exponential average rather than a mean over everything: a node whose model changed, or
//! whose second card was taken by a render, is a different machine from the one it was an hour
//! ago, and the router has to believe the recent past.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

/// Generations in flight on this node, counted where they run rather than read from a gate
/// the generate path never crosses - the admission gate exists, publishes a capacity, and is
/// bypassed by /api/generate entirely, so its snapshot said "idle" under any load.
static IN_FLIGHT: AtomicU32 = AtomicU32::new(0);

/// RAII count of one running generation.
pub struct InFlight;
impl InFlight {
    pub fn enter() -> Self {
        IN_FLIGHT.fetch_add(1, Ordering::Relaxed);
        InFlight
    }
}
impl Drop for InFlight {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
    }
}

pub fn in_flight() -> u32 {
    IN_FLIGHT.load(Ordering::Relaxed)
}

/// Weight given to the newest sample. A tenth: fast enough to follow a model switch within a
/// handful of requests, slow enough that one unlucky generation does not move the routing table.
const ALPHA: f64 = 0.1;
/// How fast a below-capacity observation erodes the aggregate ceiling.
const CAP_EROSION: f64 = 0.02;
/// How long a throughput window must run before its rate means anything. Shorter than this
/// and one generation's start-up dominates; much longer and a burst is averaged with the idle
/// time after it.
const MIN_WINDOW: std::time::Duration = std::time::Duration::from_secs(2);

#[derive(Debug, Clone, Copy, Default)]
pub struct Rates {
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    pub model_load_s: f64,
    /// Tokens per second the NODE sustains across everything in flight, measured as tokens
    /// emitted divided by the wall time it took to emit them.
    ///
    /// It is the only rate a queue can be priced against, and it has to come from a window
    /// rather than from a per-request rate times a width: where generations are serialised
    /// behind the model lock, a request running alone reports its full rate while others
    /// merely WAIT, and the product then counts waiting requests as concurrent ones. Measured
    /// on two equal nodes, that product published 13338 tok/s for a machine emitting 623.
    pub agg_tok_per_s: f64,
    /// How many generations are behind these figures. Zero means nothing has run here yet, and
    /// a peer reading it should treat the rates as unknown rather than as measured.
    pub samples: u64,
    /// Tokens emitted since the current throughput window opened.
    window_tokens: u64,
    /// When it opened. `None` before the first generation.
    window_start: Option<std::time::Instant>,
}

/// Rates PER MODEL, because a node has no single speed.
///
/// One figure per node is wrong in the direction that matters: a node that has just served a
/// 0.6B model would publish its 0.6B rate, and a peer placing a 70B request would believe it.
/// The comparison has to run on a number that describes the work being placed, which is
/// precisely the case routing exists to get right.
static RATES: RwLock<Option<HashMap<String, Rates>>> = RwLock::new(None);

fn with_rates<R>(f: impl FnOnce(&mut HashMap<String, Rates>) -> R) -> R {
    let mut g = RATES.write().unwrap_or_else(|e| e.into_inner());
    f(g.get_or_insert_with(HashMap::new))
}

/// Record what a generation achieved. Called where the rates already exist, so no path can
/// produce tokens without the meter seeing them.
pub fn record_generation(
    model: &str,
    prefill_tok_per_s: f64,
    decode_tok_per_s: f64,
    output_tokens: u64,
) {
    record_generation_at(
        model,
        prefill_tok_per_s,
        decode_tok_per_s,
        output_tokens,
        std::time::Instant::now(),
    )
}

/// The same, with the clock passed in.
///
/// What this meter publishes depends entirely on WHEN completions arrive, so a test that
/// cannot choose the arrival times can only observe the behaviour it happens to produce. A
/// trace of (tokens, instant) replays a saturated node, a clumped one or an idle one in
/// microseconds, on any machine, with no model loaded.
pub fn record_generation_at(
    model: &str,
    prefill_tok_per_s: f64,
    decode_tok_per_s: f64,
    output_tokens: u64,
    now: std::time::Instant,
) {
    // A generation that produced nothing measurable says nothing about the machine: a
    // four-token reply is dominated by fixed costs and would drag the average toward a rate
    // this node never sustains.
    if !(decode_tok_per_s.is_finite() && decode_tok_per_s > 0.0) {
        return;
    }
    with_rates(|all| {
        let r = all.entry(model.to_string()).or_default();
        if r.samples == 0 {
            // The first real measurement takes the per-request rates over from any hardware
            // prior - a derived number never outlives an observed one.
            r.decode_tok_per_s = decode_tok_per_s;
            r.prefill_tok_per_s = if prefill_tok_per_s.is_finite() && prefill_tok_per_s > 0.0 {
                prefill_tok_per_s
            } else {
                0.0
            };
        } else {
            r.decode_tok_per_s = r.decode_tok_per_s * (1.0 - ALPHA) + decode_tok_per_s * ALPHA;
            if prefill_tok_per_s.is_finite() && prefill_tok_per_s > 0.0 {
                r.prefill_tok_per_s = if r.prefill_tok_per_s > 0.0 {
                    r.prefill_tok_per_s * (1.0 - ALPHA) + prefill_tok_per_s * ALPHA
                } else {
                    prefill_tok_per_s
                };
            }
        }
        // Throughput over a window: what this node actually emitted, divided by the time it
        // took. Bounded by reality in a way a product of two samples is not.
        //
        // A window counts what completed strictly AFTER it opened. Counting the completion
        // that opens it too puts n+1 completions in n intervals - at a steady 683 tok/s that
        // published 884, and it compounds with bursty traffic, where the window's duration
        // comes from the sparse completion that closes it and its tokens from a dense group.
        let Some(start) = r.window_start else {
            r.window_start = Some(now);
            if r.agg_tok_per_s <= 0.0 {
                r.agg_tok_per_s = decode_tok_per_s;
            }
            r.samples += 1;
            return;
        };
        r.window_tokens += output_tokens;
        let elapsed = now.saturating_duration_since(start);
        if elapsed >= MIN_WINDOW {
            let observed = r.window_tokens as f64 / elapsed.as_secs_f64();
            // The aggregate is a CAPACITY, not an average: an idle stretch proves nothing
            // about the ceiling, so a higher window is taken whole and lower ones only erode
            // it slowly. A node warmed up sequentially must not publish its narrow rate and
            // then lose every saturation comparison to a peer that once ran wide.
            r.agg_tok_per_s = if observed > r.agg_tok_per_s {
                observed
            } else {
                r.agg_tok_per_s * (1.0 - CAP_EROSION) + observed * CAP_EROSION
            };
            tracing::debug!(
                "meter {model}: window {} tokens in {:.2}s = {observed:.0} tok/s, agg now {:.0}",
                r.window_tokens,
                elapsed.as_secs_f64(),
                r.agg_tok_per_s
            );
            r.window_tokens = 0;
            r.window_start = Some(now);
        } else if r.agg_tok_per_s <= 0.0 {
            // Before any window closes the node would publish zero, which a peer reads as
            // never measured. One request's own rate is a floor it has demonstrably reached.
            r.agg_tok_per_s = decode_tok_per_s;
        }
        r.samples += 1;
    });
}

/// Seed an unmeasured model with a hardware-derived ceiling (its bytes streamed at the
/// card's memory bandwidth). Ignored once anything real has been measured.
pub fn record_prior(model: &str, ceiling_tok_per_s: f64) {
    if !(ceiling_tok_per_s.is_finite() && ceiling_tok_per_s > 0.0) {
        return;
    }
    with_rates(|all| {
        let r = all.entry(model.to_string()).or_default();
        if r.samples == 0 {
            if r.decode_tok_per_s <= 0.0 {
                r.decode_tok_per_s = ceiling_tok_per_s;
            }
            if ceiling_tok_per_s > r.agg_tok_per_s {
                r.agg_tok_per_s = ceiling_tok_per_s;
            }
        }
    });
}

/// Record how long making a model resident took.
///
/// Kept separate from the generation rates: a load happens once per model and would otherwise
/// be averaged against per-token figures it has nothing to do with. It is also the term that
/// decides whether a peer holding the weights is worth a hand-over.
pub fn record_load(model: &str, seconds: f64) {
    if !(seconds.is_finite() && seconds > 0.0) {
        return;
    }
    with_rates(|all| {
        let r = all.entry(model.to_string()).or_default();
        r.model_load_s = if r.model_load_s > 0.0 {
            r.model_load_s * (1.0 - ALPHA) + seconds * ALPHA
        } else {
            seconds
        };
    });
}

/// What this node has measured for ONE model. All zeros when it has never run it, which a
/// peer reads as unknown and prices pessimistically rather than optimistically.
pub fn snapshot(model: &str) -> Rates {
    with_rates(|all| all.get(model).copied().unwrap_or_default())
}

/// Everything measured here, to publish. Small by construction: it holds one entry per model
/// this node has actually run, not one per model it could run.
pub fn all() -> HashMap<String, Rates> {
    with_rates(|all| all.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The meter must not be moved by a generation that measured nothing: an infinite or zero
    /// rate is an artefact of a reply too short to time, not a property of the machine.
    #[test]
    fn a_meaningless_generation_does_not_move_the_meter() {
        let m = "meaningless:test";
        let before = snapshot(m);
        record_generation(m, f64::INFINITY, 0.0, 100);
        record_generation(m, 0.0, f64::NAN, 100);
        record_generation(m, 100.0, -5.0, 100);
        let after = snapshot(m);
        assert_eq!(before.samples, after.samples);
    }

    /// A load time is kept apart from per-token rates: averaging seconds against tokens per
    /// second would produce a number that describes neither.
    #[test]
    fn a_load_time_does_not_count_as_a_generation() {
        let m = "load:test";
        let before = snapshot(m);
        record_load(m, 12.5);
        let after = snapshot(m);
        assert_eq!(before.samples, after.samples, "a load is not a generation");
        assert!(after.model_load_s > 0.0);
    }

    /// The failure this exists for: a node warmed up sequentially published its narrow rate as
    /// its aggregate and lost every saturation comparison to a peer that had once run wide -
    /// the whole queue then piled onto the weaker machine.
    #[test]
    fn a_quiet_stretch_does_not_erase_the_demonstrated_ceiling() {
        let mut r = Rates::default();
        r.agg_tok_per_s = 900.0;
        for _ in 0..10 {
            r.agg_tok_per_s = r.agg_tok_per_s * (1.0 - CAP_EROSION) + 100.0 * CAP_EROSION;
        }
        assert!(
            r.agg_tok_per_s > 900.0 * 0.7,
            "eroded too fast: {}",
            r.agg_tok_per_s
        );
    }

    /// Replay a completion trace against a private meter and return what it would publish.
    ///
    /// Private rather than through the static map: the suite shares that map, and a test whose
    /// result depends on which other tests ran cannot fail for a reason anyone can act on.
    fn replay(trace: &[(f64, u64)], decode_tok_per_s: f64) -> f64 {
        let origin = std::time::Instant::now();
        let mut r = Rates::default();
        for (at_s, tokens) in trace {
            let now = origin + std::time::Duration::from_secs_f64(*at_s);
            apply(&mut r, decode_tok_per_s, *tokens, now);
        }
        r.agg_tok_per_s
    }

    /// The window arithmetic, isolated from the shared map so a trace can drive it directly.
    fn apply(r: &mut Rates, decode_tok_per_s: f64, tokens: u64, now: std::time::Instant) {
        let Some(start) = r.window_start else {
            r.window_start = Some(now);
            if r.agg_tok_per_s <= 0.0 {
                r.agg_tok_per_s = decode_tok_per_s;
            }
            r.samples += 1;
            return;
        };
        r.window_tokens += tokens;
        let elapsed = now.saturating_duration_since(start);
        if elapsed >= MIN_WINDOW {
            let observed = r.window_tokens as f64 / elapsed.as_secs_f64();
            r.agg_tok_per_s = if observed > r.agg_tok_per_s {
                observed
            } else {
                r.agg_tok_per_s * (1.0 - CAP_EROSION) + observed * CAP_EROSION
            };
            r.window_tokens = 0;
            r.window_start = Some(now);
        } else if r.agg_tok_per_s <= 0.0 {
            r.agg_tok_per_s = decode_tok_per_s;
        }
        r.samples += 1;
    }

    /// Twenty-four requests of 512 tokens over 18 seconds is 682 tok/s, however they are
    /// spaced. What the meter publishes must not depend on the SPACING - and it does.
    ///
    /// This is the measurement that took a two-machine burst to find: evenly spaced
    /// completions publish the truth, while completions arriving in groups of six publish
    /// more than twice it, because a request's tokens are credited at the instant it ends
    /// rather than over the time it ran.
    #[test]
    fn the_published_rate_must_not_depend_on_how_completions_clump() {
        let total = 24u64 * 512;
        let span = 18.0;
        let truth = total as f64 / span;

        let even: Vec<(f64, u64)> = (1..=24).map(|i| (i as f64 * span / 24.0, 512)).collect();
        // Six at a time, each group landing together after running concurrently.
        let clumped: Vec<(f64, u64)> = (0..4)
            .flat_map(|g| {
                (0..6).map(move |k| ((g + 1) as f64 * span / 4.0 + k as f64 * 0.01, 512u64))
            })
            .collect();

        let even_agg = replay(&even, 650.0);
        let clumped_agg = replay(&clumped, 650.0);
        assert!(
            (even_agg - truth).abs() < truth * 0.2,
            "evenly spaced: published {even_agg:.0} against {truth:.0} emitted"
        );
        assert!(
            clumped_agg < truth * 1.3,
            "clumped: published {clumped_agg:.0} against {truth:.0} emitted - a request's \
             tokens are being credited to the window it ENDED in, not the one it ran in"
        );
    }

    /// The defect this replaced: the aggregate was `per-request rate x requests in flight`.
    /// Where generations are serialised behind the model lock a request runs at full rate
    /// while the others merely WAIT, so the product counts waiting requests as concurrent.
    /// Measured on two nodes of equal capacity it published 13338 tok/s for a machine
    /// emitting 623, and the router kept 22 requests out of 24 that it should have shared.
    ///
    /// The window cannot do that: it divides what was emitted by the time it took.
    #[test]
    fn a_serialised_queue_cannot_inflate_the_aggregate() {
        // Twenty requests, each reporting a full-speed 650 tok/s because each ran alone while
        // the rest queued, one finishing every 0.8s. The node emits 640 tok/s.
        let trace: Vec<(f64, u64)> = (1..=20).map(|i| (i as f64 * 0.8, 512)).collect();
        let agg = replay(&trace, 650.0);
        assert!(
            agg < 700.0,
            "published {agg} tok/s where 640 were emitted - the old product claimed 650 x 20"
        );
    }

    /// A window measures throughput, so a node that emits steadily publishes what it emitted.
    #[test]
    fn the_aggregate_is_what_the_node_emitted_over_the_time_it_took() {
        // 200 tokens every 50ms is 4000 tok/s, sustained for ten seconds.
        let trace: Vec<(f64, u64)> = (1..=200).map(|i| (i as f64 * 0.05, 200)).collect();
        let agg = replay(&trace, 400.0);
        assert!(
            (agg - 4000.0).abs() < 4000.0 * 0.1,
            "published {agg} tok/s against 4000 actually emitted"
        );
    }

    /// A hardware ceiling prices a model before any request has run; the first real
    /// measurement takes the per-request rate over while the ceiling only erodes.
    #[test]
    fn a_prior_prices_an_unmeasured_model_until_measurements_take_over() {
        let m = "prior:test";
        record_prior(m, 1000.0);
        let s = snapshot(m);
        assert_eq!(s.samples, 0);
        assert!(s.decode_tok_per_s > 0.0, "priced before any measurement");
        record_generation(m, 400.0, 200.0, 128);
        let s = snapshot(m);
        assert!(
            (s.decode_tok_per_s - 200.0).abs() < 1e-6,
            "measured decode wins outright"
        );
        assert!(
            s.agg_tok_per_s > 700.0,
            "the ceiling erodes, it does not vanish"
        );
        record_prior(m, 5000.0);
        assert!(
            snapshot(m).agg_tok_per_s < 5000.0,
            "a prior never overrides measurements"
        );
    }

    /// A node has no single speed. Recording one model must not move another's figure, or a
    /// peer placing a large model reads the rate of whatever small one ran last.
    #[test]
    fn one_model_s_rate_does_not_become_another_s() {
        record_generation("small:test", 0.0, 400.0, 128);
        record_generation("large:test", 0.0, 12.0, 128);
        assert!((snapshot("small:test").decode_tok_per_s - 400.0).abs() < 1e-9);
        assert!((snapshot("large:test").decode_tok_per_s - 12.0).abs() < 1e-9);
        // And a model never run here reports nothing rather than borrowing a neighbour's.
        assert_eq!(snapshot("never:run:test").samples, 0);
    }

    /// The first sample IS the rate - starting from zero and averaging toward it would make a
    /// node look ten times slower than it is for its first dozen requests, which is exactly
    /// when the cluster is deciding where to send work.
    #[test]
    fn the_first_sample_is_taken_whole() {
        // A private meter, since the static is shared with the rest of the suite.
        let mut r = Rates::default();
        let apply = |r: &mut Rates, d: f64| {
            if r.samples == 0 {
                r.decode_tok_per_s = d;
            } else {
                r.decode_tok_per_s = r.decode_tok_per_s * (1.0 - ALPHA) + d * ALPHA;
            }
            r.samples += 1;
        };
        apply(&mut r, 80.0);
        assert!(
            (r.decode_tok_per_s - 80.0).abs() < 1e-9,
            "first sample must not be damped"
        );
        apply(&mut r, 40.0);
        assert!(
            r.decode_tok_per_s < 80.0 && r.decode_tok_per_s > 40.0,
            "then it follows"
        );
    }
}
