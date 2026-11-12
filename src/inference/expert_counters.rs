//! What the routed experts of every layer did while the stage measure runs.
//!
//! The expert tiers note what they served, and the stage endpoint of the server reports it.
//! Nothing is counted while the measure is off.
//!
//! A layer is keyed by the address of the tier or block that serves it.
//! A pair is 1 token routed to 1 expert.
//! A hit is a pair a tier served from weights it holds: a card's slots or its kept experts.
//! A miss is a pair the host computed from the expert's weights in the file mapping.
//! The host's tier keeps some of those weights in memory: its pairs are counted apart.
//!
//! The host's tier also traces what each layer computed, in order, while the measure runs.
//! A read of the trace empties it.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

/// What the host tier held of an expert a forward computed on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Found {
    /// The tier kept it.
    Kept,
    /// The tier never kept it since the load.
    NeverKept,
    /// A prompt let it go.
    LeftAtPrompt,
    /// A decoded row let it go.
    LeftAtDecode,
}

/// 1 expert the host computed in a traced forward.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct HostExpert {
    pub expert: usize,
    /// The last row of the forward routed to it.
    pub last: usize,
    /// The rows of the forward routed to it.
    pub rows: usize,
    pub found: Found,
    /// Bytes of its pages that were not in memory when the forward asked for it.
    ///
    /// Only a decoded row's experts are looked at, 0 for a prompt's.
    pub absent: u64,
    /// A guess read it ahead.
    pub ahead: bool,
}

/// 1 entry of the host tier's trace.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierEvent {
    /// What a layer kept when its trace started, the least recent first, and its bound.
    Kept {
        layer: usize,
        experts: Vec<usize>,
        capacity: usize,
    },
    /// A forward of `rows` rows in a layer: the experts the cards served, and those the host
    /// computed.
    Forward {
        layer: usize,
        rows: usize,
        card: Vec<usize>,
        host: Vec<HostExpert>,
    },
}

/// The counts of 1 measure.
#[derive(Default)]
pub struct Counters {
    on: AtomicBool,
    /// Per layer, how many pairs routed to each expert.
    working_set: Mutex<HashMap<usize, HashMap<usize, u64>>>,
    /// Per layer, the experts the last decoded token routed to.
    previous: Mutex<HashMap<usize, Vec<usize>>>,
    hits: AtomicU64,
    misses: AtomicU64,
    read_for_compute: AtomicU64,
    read_for_ahead: AtomicU64,
    routed_again: AtomicU64,
    decoded_routed: AtomicU64,
    guess_routed: AtomicU64,
    guess_named: AtomicU64,
    guess_unkept: AtomicU64,
    guess_unkept_named: AtomicU64,
    host_bytes: AtomicU64,
    tier_pairs: AtomicU64,
    /// Decoded rows' host experts the tier did not keep, by what it held of them.
    never_kept: AtomicU64,
    left_at_prompt: AtomicU64,
    left_at_decode: AtomicU64,
    /// Of them, those a guess read ahead.
    missed_ahead: AtomicU64,
    /// Bytes of decoded rows' host experts not in memory when asked for: kept, not kept, and
    /// not kept but read ahead.
    absent_kept: AtomicU64,
    absent_missed: AtomicU64,
    absent_ahead: AtomicU64,
    /// Routed slots a lossy routing gave to an expert outside the exact choice, and slots it
    /// left empty.
    replaced: AtomicU64,
    skipped: AtomicU64,
    /// The host tier's trace since the last read.
    trace: Mutex<Vec<TierEvent>>,
    /// The layers whose trace started since the measure did.
    traced: Mutex<HashSet<usize>>,
    /// Per card ordinal, the pairs its routed experts served.
    cards: Mutex<std::collections::BTreeMap<usize, u64>>,
}

/// What a measure has counted.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Report {
    /// The distinct experts of each layer, largest first.
    pub working_set: Vec<usize>,
    pub hits: u64,
    pub misses: u64,
    pub read_for_compute: u64,
    pub read_for_ahead: u64,
    pub routed_again: u64,
    pub decoded_routed: u64,
    pub guess_routed: u64,
    pub guess_named: u64,
    pub guess_unkept: u64,
    pub guess_unkept_named: u64,
    /// Bytes of the experts the host computed, read from memory or from the file.
    pub host_bytes: u64,
    /// Pairs the host computed from experts its tier keeps.
    pub tier_pairs: u64,
    /// Decoded rows' host experts the tier never kept since the load.
    pub never_kept: u64,
    /// Decoded rows' host experts a prompt let go.
    pub left_at_prompt: u64,
    /// Decoded rows' host experts a decoded row let go.
    pub left_at_decode: u64,
    /// Of the 3 above, the experts a guess read ahead.
    pub missed_ahead: u64,
    /// Bytes of decoded rows' host experts the tier kept that were not in memory when asked for.
    pub absent_kept: u64,
    /// The same for the experts it did not keep.
    pub absent_missed: u64,
    /// Of those, the bytes of experts a guess read ahead.
    pub absent_ahead: u64,
    /// Routed slots a lossy routing gave to an expert outside the exact choice.
    pub replaced: u64,
    /// Routed slots a lossy routing left empty.
    pub skipped: u64,
    /// Per card ordinal, the pairs its routed experts served, by ordinal.
    pub card_pairs: Vec<(usize, u64)>,
}

impl Counters {
    fn atomics(&self) -> [&AtomicU64; 21] {
        [
            &self.never_kept,
            &self.left_at_prompt,
            &self.left_at_decode,
            &self.missed_ahead,
            &self.absent_kept,
            &self.absent_missed,
            &self.absent_ahead,
            &self.hits,
            &self.misses,
            &self.read_for_compute,
            &self.read_for_ahead,
            &self.routed_again,
            &self.decoded_routed,
            &self.guess_routed,
            &self.guess_named,
            &self.guess_unkept,
            &self.guess_unkept_named,
            &self.host_bytes,
            &self.tier_pairs,
            &self.replaced,
            &self.skipped,
        ]
    }

    /// Starts the measure from zero, or stops it.
    pub fn enable(&self, on: bool) {
        if on {
            self.atomics().iter().for_each(|c| c.store(0, Relaxed));
            self.working_set
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
            self.previous
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
            self.trace.lock().unwrap_or_else(|e| e.into_inner()).clear();
            self.traced
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear();
            self.cards.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
        self.on.store(on, Relaxed);
    }

    /// Whether `layer`'s trace starts now: true once per layer while the measure runs.
    pub fn trace_starts(&self, layer: usize) -> bool {
        self.on()
            && self
                .traced
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .insert(layer)
    }

    /// Adds `event` to the trace while the measure runs.
    pub fn trace(&self, event: TierEvent) {
        if self.on() {
            self.trace
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push(event);
        }
    }

    /// The trace since the last read, which it empties.
    pub fn take_trace(&self) -> Vec<TierEvent> {
        std::mem::take(&mut *self.trace.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// A decoded row's host expert of `bytes` bytes, `absent` of them not in memory when asked
    /// for: `found` says what the tier held of it, `ahead` whether a guess read it ahead.
    pub fn found(&self, found: Found, ahead: bool, absent: u64) {
        if !self.on() {
            return;
        }
        let missed = match found {
            Found::Kept => {
                self.absent_kept.fetch_add(absent, Relaxed);
                return;
            }
            Found::NeverKept => &self.never_kept,
            Found::LeftAtPrompt => &self.left_at_prompt,
            Found::LeftAtDecode => &self.left_at_decode,
        };
        missed.fetch_add(1, Relaxed);
        self.absent_missed.fetch_add(absent, Relaxed);
        if ahead {
            self.missed_ahead.fetch_add(1, Relaxed);
            self.absent_ahead.fetch_add(absent, Relaxed);
        }
    }

    /// Whether the measure runs: a caller skips the work of a note while it does not.
    pub fn on(&self) -> bool {
        self.on.load(Relaxed)
    }

    /// `experts` routed in `layer`, 1 entry per pair.
    pub fn routed(&self, layer: usize, experts: impl IntoIterator<Item = usize>) {
        if !self.on() {
            return;
        }
        let mut m = self.working_set.lock().unwrap_or_else(|e| e.into_inner());
        let counts = m.entry(layer).or_default();
        for e in experts {
            *counts.entry(e).or_insert(0) += 1;
        }
    }

    /// Pairs a tier served from weights it holds, and pairs the host computed from the mapping.
    pub fn served(&self, hits: u64, misses: u64) {
        if self.on() {
            self.hits.fetch_add(hits, Relaxed);
            self.misses.fetch_add(misses, Relaxed);
        }
    }

    /// Pairs the routed experts card `ordinal` keeps served.
    pub fn card(&self, ordinal: usize, pairs: u64) {
        if self.on() && pairs > 0 {
            *self
                .cards
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(ordinal)
                .or_insert(0) += pairs;
        }
    }

    /// Bytes of expert weights the host read from the file mapping: by the step computing them,
    /// or by a read queued ahead of the layer.
    pub fn read(&self, ahead: bool, bytes: u64) {
        if self.on() {
            match ahead {
                true => &self.read_for_ahead,
                false => &self.read_for_compute,
            }
            .fetch_add(bytes, Relaxed);
        }
    }

    /// Bytes of the experts the host computed, and the pairs of them its tier keeps.
    pub fn host(&self, bytes: u64, tier_pairs: u64) {
        if self.on() {
            self.host_bytes.fetch_add(bytes, Relaxed);
            self.tier_pairs.fetch_add(tier_pairs, Relaxed);
        }
    }

    /// The experts a decoded token routed to in `layer`, against those of the token before in
    /// the same layer.
    pub fn decoded(&self, layer: usize, now: &[usize]) {
        if !self.on() {
            return;
        }
        let before = self
            .previous
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(layer, now.to_vec())
            .unwrap_or_default();
        let again = now.iter().filter(|e| before.contains(e)).count();
        self.repeat(again as u64, now.len() as u64);
    }

    /// Of `routed` experts of decoded tokens, `again` the token before routed to as well.
    pub fn repeat(&self, again: u64, routed: u64) {
        if self.on() {
            self.routed_again.fetch_add(again, Relaxed);
            self.decoded_routed.fetch_add(routed, Relaxed);
        }
    }

    /// How a guess of a layer's routing compared with the routing: the experts routed, those
    /// the guess named, those no tier kept, and those of them the guess named.
    pub fn guess(&self, routed: u64, named: u64, unkept: u64, unkept_named: u64) {
        if self.on() {
            self.guess_routed.fetch_add(routed, Relaxed);
            self.guess_named.fetch_add(named, Relaxed);
            self.guess_unkept.fetch_add(unkept, Relaxed);
            self.guess_unkept_named.fetch_add(unkept_named, Relaxed);
        }
    }

    /// Routed slots a lossy routing gave to an expert outside the exact choice, and slots it
    /// left empty.
    pub fn truncated(&self, replaced: u64, skipped: u64) {
        if self.on() {
            self.replaced.fetch_add(replaced, Relaxed);
            self.skipped.fetch_add(skipped, Relaxed);
        }
    }

    /// The counts since the measure started.
    pub fn report(&self) -> Report {
        let mut working_set: Vec<usize> = self
            .working_set
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(HashMap::len)
            .collect();
        working_set.sort_unstable_by(|a, b| b.cmp(a));
        Report {
            working_set,
            hits: self.hits.load(Relaxed),
            misses: self.misses.load(Relaxed),
            read_for_compute: self.read_for_compute.load(Relaxed),
            read_for_ahead: self.read_for_ahead.load(Relaxed),
            routed_again: self.routed_again.load(Relaxed),
            decoded_routed: self.decoded_routed.load(Relaxed),
            guess_routed: self.guess_routed.load(Relaxed),
            guess_named: self.guess_named.load(Relaxed),
            guess_unkept: self.guess_unkept.load(Relaxed),
            guess_unkept_named: self.guess_unkept_named.load(Relaxed),
            host_bytes: self.host_bytes.load(Relaxed),
            tier_pairs: self.tier_pairs.load(Relaxed),
            never_kept: self.never_kept.load(Relaxed),
            left_at_prompt: self.left_at_prompt.load(Relaxed),
            left_at_decode: self.left_at_decode.load(Relaxed),
            missed_ahead: self.missed_ahead.load(Relaxed),
            absent_kept: self.absent_kept.load(Relaxed),
            absent_missed: self.absent_missed.load(Relaxed),
            absent_ahead: self.absent_ahead.load(Relaxed),
            replaced: self.replaced.load(Relaxed),
            skipped: self.skipped.load(Relaxed),
            card_pairs: self
                .cards
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|(&o, &p)| (o, p))
                .collect(),
        }
    }

    /// The share of the pairs the `cap` most routed experts of each layer took, averaged over
    /// the layers: what a tier holding the hottest `cap` would have served.
    pub fn coverage(&self, cap: usize) -> f64 {
        let m = self.working_set.lock().unwrap_or_else(|e| e.into_inner());
        if m.is_empty() {
            return 0.0;
        }
        let mut sum = 0.0;
        for counts in m.values() {
            let total: u64 = counts.values().sum();
            if total == 0 {
                continue;
            }
            let mut v: Vec<u64> = counts.values().copied().collect();
            v.sort_unstable_by(|a, b| b.cmp(a));
            let top: u64 = v.iter().take(cap).sum();
            sum += top as f64 / total as f64;
        }
        sum / m.len() as f64
    }
}

/// The measure of the process, which the expert tiers feed and the stage endpoint reads.
pub fn global() -> &'static Counters {
    static GLOBAL: std::sync::OnceLock<Counters> = std::sync::OnceLock::new();
    GLOBAL.get_or_init(Counters::default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_is_counted_while_the_measure_is_off() {
        let c = Counters::default();
        c.enable(true);
        c.enable(false);
        c.routed(1, [3, 4]);
        c.served(2, 1);
        c.read(true, 64);
        c.decoded(1, &[3, 4]);
        c.guess(2, 1, 1, 1);
        c.host(64, 2);
        c.found(Found::NeverKept, true, 64);
        c.card(1, 3);
        c.trace(TierEvent::Kept {
            layer: 0,
            experts: vec![1],
            capacity: 1,
        });
        assert!(!c.trace_starts(0));
        assert_eq!(c.report(), Report::default());
        assert!(c.take_trace().is_empty());
    }

    #[test]
    fn card_pairs_count_per_card_and_restart_with_the_measure() {
        let c = Counters::default();
        c.enable(true);
        c.card(1, 3);
        c.card(0, 2);
        c.card(1, 4);
        c.card(2, 0);
        assert_eq!(c.report().card_pairs, vec![(0, 2), (1, 7)]);
        c.enable(true);
        assert!(c.report().card_pairs.is_empty());
    }

    #[test]
    fn misses_count_by_cause_and_absent_bytes_by_what_was_kept() {
        let c = Counters::default();
        c.enable(true);
        c.found(Found::Kept, false, 10);
        c.found(Found::NeverKept, false, 20);
        c.found(Found::LeftAtPrompt, true, 30);
        c.found(Found::LeftAtPrompt, false, 0);
        c.found(Found::LeftAtDecode, true, 5);
        let r = c.report();
        assert_eq!(
            (
                r.never_kept,
                r.left_at_prompt,
                r.left_at_decode,
                r.missed_ahead
            ),
            (1, 2, 1, 2)
        );
        assert_eq!(
            (r.absent_kept, r.absent_missed, r.absent_ahead),
            (10, 55, 35)
        );
    }

    #[test]
    fn a_layer_starts_its_trace_once_and_a_read_empties_the_trace() {
        let c = Counters::default();
        c.enable(true);
        assert!(c.trace_starts(3));
        assert!(!c.trace_starts(3));
        let e = TierEvent::Forward {
            layer: 3,
            rows: 1,
            card: vec![2],
            host: vec![HostExpert {
                expert: 5,
                last: 0,
                rows: 1,
                found: Found::LeftAtDecode,
                absent: 4096,
                ahead: false,
            }],
        };
        c.trace(e.clone());
        assert_eq!(c.take_trace(), vec![e]);
        assert!(c.take_trace().is_empty());
        c.enable(true);
        assert!(c.trace_starts(3), "a new measure starts the trace again");
    }

    #[test]
    fn the_measure_counts_pairs_reads_repeats_and_guesses_from_zero() {
        let c = Counters::default();
        c.enable(true);
        c.served(5, 3);
        c.enable(true);
        c.routed(1, [3, 4, 3, 3]);
        c.routed(2, [7]);
        c.served(3, 2);
        c.read(false, 100);
        c.read(true, 40);
        c.decoded(1, &[3, 4]);
        c.decoded(1, &[4, 5]);
        c.guess(4, 2, 3, 1);
        c.host(300, 2);
        c.host(200, 1);
        let r = c.report();
        assert_eq!((r.host_bytes, r.tier_pairs), (500, 3));
        assert_eq!(r.working_set, vec![2, 1]);
        assert_eq!(
            (r.hits, r.misses),
            (3, 2),
            "the second start zeroed the first"
        );
        assert_eq!((r.read_for_compute, r.read_for_ahead), (100, 40));
        assert_eq!((r.routed_again, r.decoded_routed), (1, 4));
        assert_eq!(
            (
                r.guess_routed,
                r.guess_named,
                r.guess_unkept,
                r.guess_unkept_named
            ),
            (4, 2, 3, 1)
        );
    }

    #[test]
    fn coverage_is_the_share_of_the_hottest_experts_averaged_over_the_layers() {
        let c = Counters::default();
        c.enable(true);
        c.routed(1, [3, 3, 3, 4]);
        c.routed(2, [7, 8]);
        assert!((c.coverage(1) - (0.75 + 0.5) / 2.0).abs() < 1e-12);
        assert!((c.coverage(2) - 1.0).abs() < 1e-12);
    }
}
