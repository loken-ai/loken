//! What the routed experts of every layer did while the stage measure runs.
//!
//! The expert tiers note what they served, and the stage endpoint of the server reports it.
//! Nothing is counted while the measure is off.
//!
//! A layer is keyed by the address of the tier or block that serves it.
//! A pair is 1 token routed to 1 expert.
//! A hit is a pair a tier served from weights it holds: a card's slots or its kept experts.
//! A miss is a pair the host computed from the expert's weights in the file mapping.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

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
}

impl Counters {
    fn atomics(&self) -> [&AtomicU64; 10] {
        [
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
        }
        self.on.store(on, Relaxed);
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
        assert_eq!(c.report(), Report::default());
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
        let r = c.report();
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
