//! What one engine measured about reading a prompt: its own read and, per peer, the whole
//! hand-over. The decision to ask a peer for a prompt reads only these.
//!
//! Each way to read a prompt keeps its latest measure.
//! A measure goes stale once the other way ran, since it was taken, as long as this way's
//! measures took since it last won.
//! A stale way is measured again by the next long prompt.
//! A way that keeps losing is measured again less and less often.

use std::sync::Mutex;

/// The latest measure of one way to read a prompt.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Measure {
    /// Prompt tokens per second.
    tok_s: f64,
    /// Seconds this way's measures took since it last won, this one included.
    losing_s: f64,
    /// Seconds the other way ran since this measure.
    other_s: f64,
}

impl Measure {
    /// A measure of `tokens` in `secs`, after `before`, against the other way's speed.
    /// It extends the losing time of `before` when it is slower than `other`.
    fn new(tokens: usize, secs: f64, before: Option<Measure>, other: Option<f64>) -> Self {
        let tok_s = tokens as f64 / secs;
        let lost = other.is_some_and(|o| o > tok_s);
        let losing_s = match before {
            Some(b) if lost => b.losing_s + secs,
            _ => secs,
        };
        Self {
            tok_s,
            losing_s,
            other_s: 0.0,
        }
    }

    /// The speed while the measure is fresh.
    fn fresh(&self) -> Option<f64> {
        (self.other_s < self.losing_s).then_some(self.tok_s)
    }
}

/// The latest hand-over from one peer.
#[derive(Debug)]
struct PeerRate {
    endpoint: String,
    measure: Measure,
}

#[derive(Debug, Default)]
struct Rates {
    /// The last local read of a prefill chunk or more.
    local: Option<Measure>,
    peers: Vec<PeerRate>,
}

fn usable(tokens: usize, secs: f64) -> bool {
    tokens > 0 && secs > 0.0 && secs.is_finite()
}

impl Rates {
    /// A local read of `tokens` in `secs`.
    /// It is a measure when it read at least `chunk` tokens.
    /// A read this engine chose for its own request ages every peer measure.
    fn note_local(&mut self, tokens: usize, chunk: usize, secs: f64, own: bool) {
        if !usable(tokens, secs) {
            return;
        }
        if own {
            for p in &mut self.peers {
                p.measure.other_s += secs;
            }
        }
        if tokens >= chunk {
            let best_peer = self.peers.iter().map(|p| p.measure.tok_s).reduce(f64::max);
            self.local = Some(Measure::new(tokens, secs, self.local, best_peer));
        }
    }

    /// A hand-over of `tokens` from `endpoint` that took `secs`, from the ask to the end of the
    /// local read after it.
    /// It replaces the previous one of that endpoint and ages the local measure.
    fn note_peer(&mut self, endpoint: &str, tokens: usize, secs: f64) {
        if !usable(tokens, secs) {
            return;
        }
        let local = self.local.map(|l| l.tok_s);
        if let Some(l) = &mut self.local {
            l.other_s += secs;
        }
        match self.peers.iter_mut().find(|p| p.endpoint == endpoint) {
            Some(p) => p.measure = Measure::new(tokens, secs, Some(p.measure), local),
            None => self.peers.push(PeerRate {
                endpoint: endpoint.to_string(),
                measure: Measure::new(tokens, secs, None, local),
            }),
        }
    }

    /// The speed of `endpoint`, or `None` when it was never measured or its measure is stale.
    fn peer_tok_s(&self, endpoint: &str) -> Option<f64> {
        self.peers
            .iter()
            .find(|p| p.endpoint == endpoint)
            .and_then(|p| p.measure.fresh())
    }

    /// The local speed, or `None` when it was never measured or its measure is stale.
    fn local_tok_s(&self) -> Option<f64> {
        self.local.and_then(|l| l.fresh())
    }
}

/// The measures of one engine.
#[derive(Debug, Default)]
pub(super) struct HandoverRates(Mutex<Rates>);

impl HandoverRates {
    fn rates(&self) -> std::sync::MutexGuard<'_, Rates> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Records a local read of `tokens` that took `secs`, on a model whose prefill chunk is
    /// `chunk`.
    /// `own` is false for a read done for a peer.
    pub(super) fn note_local_prefill(&self, tokens: usize, chunk: usize, secs: f64, own: bool) {
        self.rates().note_local(tokens, chunk, secs, own);
    }

    /// Records a hand-over of `tokens` from `endpoint` that took `secs`, from the ask to the end
    /// of the local read after it.
    pub(super) fn note_handover(&self, endpoint: &str, tokens: usize, secs: f64) {
        self.rates().note_peer(endpoint, tokens, secs);
    }

    /// Whether `endpoint` should read the prompt of `prompt_len` tokens, of which
    /// `tokens_to_read` are missing here, from what this engine measured.
    pub(super) fn peer_reads(
        &self,
        endpoint: &str,
        prompt_len: usize,
        tokens_to_read: usize,
    ) -> bool {
        let r = self.rates();
        use_peer(
            r.local_tok_s(),
            r.peer_tok_s(endpoint),
            prompt_len,
            tokens_to_read,
        )
    }
}

/// Prompt tokens a local prefill read.
/// A prefill that was skipped by restoring a state read none.
/// Otherwise it read what the cache, a restored state or a hand-over did not cover.
pub(super) fn tokens_read(prompt_len: usize, cached: usize, skipped: bool) -> usize {
    if skipped {
        0
    } else {
        prompt_len.saturating_sub(cached)
    }
}

/// Decides between a hand-over and reading the prompt here.
///
/// Nothing to read: the prompt stays here.
/// No fresh measure of the peer: ask it, the hand-over is how its cost is learned.
/// A fresh measure of the peer and none of the local read: read here, to learn it.
/// Both fresh: ask the peer when handing over the whole prompt takes no longer than reading
/// the missing tokens here.
/// The peer hands over the whole prompt, the local prefill reads only what is missing.
///
/// A hand-over is measured from the ask to the end of the local read after it.
/// It counts the peer's read, the transfer, the import and the restore.
///
/// Each instance compares its own measures, so two instances with the roles swapped agree on
/// which of them reads.
fn use_peer(
    local_tok_s: Option<f64>,
    peer_tok_s: Option<f64>,
    prompt_len: usize,
    tokens_to_read: usize,
) -> bool {
    if tokens_to_read == 0 {
        return false;
    }
    match (local_tok_s, peer_tok_s) {
        (_, None) => true,
        (None, Some(_)) => false,
        (Some(local), Some(peer)) => prompt_len as f64 / peer <= tokens_to_read as f64 / local,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decide(r: &Rates, peer: &str, prompt_len: usize, to_read: usize) -> bool {
        use_peer(r.local_tok_s(), r.peer_tok_s(peer), prompt_len, to_read)
    }

    #[test]
    fn nothing_to_read_stays_here() {
        assert!(!use_peer(None, None, 0, 0));
        assert!(!use_peer(Some(1.0), Some(9.0), 100, 0));
    }

    #[test]
    fn unmeasured_peer_is_asked_first() {
        assert!(use_peer(None, None, 100, 100));
        assert!(use_peer(Some(1e6), None, 100, 100));
    }

    #[test]
    fn measured_peer_without_local_measure_reads_here() {
        assert!(!use_peer(None, Some(1e6), 100, 100));
    }

    #[test]
    fn faster_way_wins_and_a_tie_goes_to_the_peer() {
        assert!(use_peer(Some(100.0), Some(200.0), 100, 100));
        assert!(use_peer(Some(100.0), Some(100.0), 100, 100));
        assert!(!use_peer(Some(200.0), Some(100.0), 100, 100));
    }

    #[test]
    fn peer_hands_over_the_whole_prompt_against_the_missing_part() {
        // The peer delivers 200 tokens/s, the local prefill reads 100 tokens/s.
        // 1000 tokens from the peer take 5 s, so the peer wins from 500 missing tokens.
        assert!(use_peer(Some(100.0), Some(200.0), 1000, 1000));
        assert!(use_peer(Some(100.0), Some(200.0), 1000, 500));
        assert!(!use_peer(Some(100.0), Some(200.0), 1000, 499));
    }

    #[test]
    fn local_measure_needs_a_chunk() {
        let mut r = Rates::default();
        r.note_local(10, 512, 1.0, true);
        assert_eq!(r.local_tok_s(), None);
        r.note_local(1024, 512, 2.0, true);
        assert_eq!(r.local_tok_s(), Some(512.0));
        r.note_local(512, 512, 4.0, true);
        assert_eq!(r.local_tok_s(), Some(128.0));
    }

    #[test]
    fn degenerate_timings_change_nothing() {
        let mut r = Rates::default();
        r.note_local(100, 10, 0.0, true);
        r.note_local(0, 10, 1.0, true);
        r.note_local(100, 10, f64::NAN, true);
        r.note_peer("a", 0, 1.0);
        r.note_peer("a", 10, 0.0);
        assert_eq!(r.local, None);
        assert!(r.peers.is_empty());
    }

    #[test]
    fn peer_measure_is_kept_per_endpoint_and_replaced() {
        let mut r = Rates::default();
        r.note_peer("a", 100, 2.0);
        r.note_peer("b", 100, 1.0);
        assert_eq!(r.peer_tok_s("a"), Some(50.0));
        assert_eq!(r.peer_tok_s("b"), Some(100.0));
        r.note_peer("a", 300, 1.0);
        assert_eq!(r.peer_tok_s("a"), Some(300.0));
        assert_eq!(r.peers.len(), 2);
        assert_eq!(r.peer_tok_s("c"), None);
    }

    /// The asker of the measured case: its card reads 9500 tokens in 0.8 s, a hand-over of them
    /// takes 1.1 s from the ask to the first token.
    #[test]
    fn slower_hand_over_stops_once_both_are_measured() {
        let (prompt, local_s, peer_s) = (9500, 0.8, 1.1);
        let mut r = Rates::default();
        let mut handed = Vec::new();
        for _ in 0..6 {
            let ask = decide(&r, "b", prompt, prompt);
            handed.push(ask);
            if ask {
                r.note_peer("b", prompt, peer_s);
            } else {
                r.note_local(prompt, 512, local_s, true);
            }
        }
        // The first prompt measures the peer, the second the local read.
        // The peer is then measured again once, and its losing time doubles.
        assert_eq!(handed, [true, false, false, true, false, false]);
        let asked_late = (0..20)
            .filter(|_| {
                let ask = decide(&r, "b", prompt, prompt);
                if ask {
                    r.note_peer("b", prompt, peer_s);
                } else {
                    r.note_local(prompt, 512, local_s, true);
                }
                ask
            })
            .count();
        // Each measure that loses lengthens the wait before the next one.
        assert!(
            asked_late <= 4,
            "the slower peer was asked {asked_late} times of 20"
        );
    }

    #[test]
    fn faster_hand_over_is_kept_and_the_local_read_is_measured_again() {
        let (prompt, local_s, peer_s) = (9500, 2.0, 0.5);
        let mut r = Rates::default();
        let mut handed = Vec::new();
        for _ in 0..12 {
            let ask = decide(&r, "a", prompt, prompt);
            handed.push(ask);
            if ask {
                r.note_peer("a", prompt, peer_s);
            } else {
                r.note_local(prompt, 512, local_s, true);
            }
        }
        assert_eq!(&handed[..2], [true, false]);
        let local_reads = handed.iter().filter(|h| !**h).count();
        assert!(
            local_reads <= 3,
            "the slower local read ran {local_reads} times of 12"
        );
        assert!(handed[2..].iter().filter(|h| **h).count() >= 8);
    }

    #[test]
    fn a_change_of_speed_is_followed() {
        let prompt = 1000;
        let mut r = Rates::default();
        let mut step = |r: &mut Rates, local_s: f64, peer_s: f64| {
            let ask = decide(r, "b", prompt, prompt);
            if ask {
                r.note_peer("b", prompt, peer_s);
            } else {
                r.note_local(prompt, 10, local_s, true);
            }
            ask
        };
        // The local read wins at first.
        let asked = (0..30).filter(|_| step(&mut r, 1.0, 2.0)).count();
        assert!(asked <= 6, "the slower peer was asked {asked} times of 30");
        // The local card slows down: the peer is asked again and kept.
        let asked = (0..30).filter(|_| step(&mut r, 5.0, 2.0)).count();
        assert!(asked >= 20, "the peer was asked {asked} times of 30");
    }

    #[test]
    fn a_read_for_the_peer_measures_without_ageing_it() {
        let mut r = Rates::default();
        r.note_peer("a", 100, 2.0);
        r.note_local(1000, 10, 100.0, false);
        assert_eq!(r.peer_tok_s("a"), Some(50.0));
        assert_eq!(r.local_tok_s(), Some(10.0));
        r.note_local(1000, 10, 2.0, true);
        assert_eq!(r.peer_tok_s("a"), None);
    }

    #[test]
    fn both_instances_decide_on_their_own_table() {
        // A reads 100 tokens/s and takes B's hand-over at 300, B reads 400 and takes A's at 50.
        let mut a = Rates::default();
        a.note_peer("b", 1000, 1000.0 / 300.0);
        a.note_local(1000, 10, 1000.0 / 100.0, false);
        let mut b = Rates::default();
        b.note_peer("a", 1000, 1000.0 / 50.0);
        b.note_local(1000, 10, 1000.0 / 400.0, false);
        assert!(decide(&a, "b", 800, 800));
        assert!(!decide(&b, "a", 800, 800));
    }

    #[test]
    fn restored_state_read_nothing() {
        assert_eq!(tokens_read(1000, 1000, true), 0);
        assert_eq!(tokens_read(1000, 0, true), 0);
    }

    #[test]
    fn partly_covered_prompt_reads_the_rest() {
        assert_eq!(tokens_read(1000, 600, false), 400);
    }

    #[test]
    fn uncached_prompt_reads_all() {
        assert_eq!(tokens_read(1000, 0, false), 1000);
    }

    #[test]
    fn fully_covered_prompt_reads_nothing() {
        assert_eq!(tokens_read(1000, 1000, false), 0);
        assert_eq!(tokens_read(1000, 1200, false), 0);
    }

    #[test]
    fn table_decides() {
        let t = HandoverRates::default();
        assert!(t.peer_reads("http://x", 100, 100));
        t.note_handover("http://x", 100, 1.0);
        assert!(!t.peer_reads("http://x", 100, 100));
        t.note_local_prefill(1000, 10, 100.0, true);
        assert!(t.peer_reads("http://x", 100, 100) || t.rates().peer_tok_s("http://x").is_none());
    }
}
