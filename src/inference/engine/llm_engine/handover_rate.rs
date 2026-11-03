//! What this process measured about reading a prompt: the speed of its own prefill and, per peer,
//! the speed of a hand-over. The decision to ask a peer for a prompt reads only these.

use std::sync::Mutex;

/// The last hand-over from one peer.
struct PeerRate {
    endpoint: String,
    /// Prompt tokens delivered per second, from the request to the end of the import.
    tok_s: f64,
    /// Seconds the hand-over took.
    took_s: f64,
    /// Seconds the local prefill ran since that hand-over.
    local_s_since: f64,
}

struct Rates {
    /// Prompt tokens per second of the last local prefill that read a chunk.
    local_tok_s: Option<f64>,
    peers: Vec<PeerRate>,
}

static RATES: Mutex<Rates> = Mutex::new(Rates {
    local_tok_s: None,
    peers: Vec::new(),
});

fn rates() -> std::sync::MutexGuard<'static, Rates> {
    RATES.lock().unwrap_or_else(|e| e.into_inner())
}

fn usable(tokens: usize, secs: f64) -> bool {
    tokens > 0 && secs > 0.0 && secs.is_finite()
}

impl Rates {
    /// A local prefill of `tokens` in `secs`.
    /// It sets the local speed when it read at least `chunk` tokens.
    /// Its time counts toward the age of every peer measurement.
    fn note_local(&mut self, tokens: usize, chunk: usize, secs: f64) {
        if !usable(tokens, secs) {
            return;
        }
        if tokens >= chunk {
            self.local_tok_s = Some(tokens as f64 / secs);
        }
        for p in &mut self.peers {
            p.local_s_since += secs;
        }
    }

    /// A hand-over of `tokens` from `endpoint` that took `secs`.
    /// It replaces the previous one of that endpoint.
    fn note_peer(&mut self, endpoint: &str, tokens: usize, secs: f64) {
        if !usable(tokens, secs) {
            return;
        }
        let fresh = PeerRate {
            endpoint: endpoint.to_string(),
            tok_s: tokens as f64 / secs,
            took_s: secs,
            local_s_since: 0.0,
        };
        match self.peers.iter_mut().find(|p| p.endpoint == endpoint) {
            Some(p) => *p = fresh,
            None => self.peers.push(fresh),
        }
    }

    /// The speed of `endpoint`, or `None` when it was never measured or its measurement is stale.
    /// A measurement is stale once the local prefill ran as long as the hand-over took.
    fn peer_tok_s(&self, endpoint: &str) -> Option<f64> {
        self.peers
            .iter()
            .find(|p| p.endpoint == endpoint)
            .filter(|p| p.local_s_since < p.took_s)
            .map(|p| p.tok_s)
    }
}

/// Records a local prefill of `tokens` that took `secs`, on a model whose prefill chunk is `chunk`.
pub(super) fn note_local_prefill(tokens: usize, chunk: usize, secs: f64) {
    rates().note_local(tokens, chunk, secs);
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

/// Records a hand-over of `tokens` from `endpoint` that took `secs`.
pub(super) fn note_handover(endpoint: &str, tokens: usize, secs: f64) {
    rates().note_peer(endpoint, tokens, secs);
}

/// Whether `endpoint` should read the prompt of `prompt_len` tokens, of which `tokens_to_read`
/// are missing here, from what this process measured.
pub(super) fn peer_reads(endpoint: &str, prompt_len: usize, tokens_to_read: usize) -> bool {
    let r = rates();
    use_peer(
        r.local_tok_s,
        r.peer_tok_s(endpoint),
        prompt_len,
        tokens_to_read,
    )
}

/// Decides between a hand-over and reading the prompt here.
///
/// Nothing to read: the prompt stays here.
/// No measurement of the peer: ask it, the hand-over is how its speed is learned.
/// A measurement of the peer and none of the local speed: ask the peer.
/// Both measured: ask the peer when handing over the whole prompt takes no longer than reading
/// the missing tokens here.
/// The peer hands over the whole prompt, the local prefill reads only what is missing.
///
/// The caller passes `None` for a peer whose measurement is stale.
/// A peer measured slow or cold is asked again once the local prefill ran as long as its
/// hand-over took.
/// The local speed is the latest one, so a slower local prefill moves the decision at once.
///
/// Each instance compares its own local speed with the speed measured for the other,
/// so two instances with the roles swapped agree on which of them reads.
pub(super) fn use_peer(
    local_tok_s: Option<f64>,
    peer_tok_s: Option<f64>,
    prompt_len: usize,
    tokens_to_read: usize,
) -> bool {
    if tokens_to_read == 0 {
        return false;
    }
    match (local_tok_s, peer_tok_s) {
        (_, None) | (None, Some(_)) => true,
        (Some(local), Some(peer)) => prompt_len as f64 / peer <= tokens_to_read as f64 / local,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty() -> Rates {
        Rates {
            local_tok_s: None,
            peers: Vec::new(),
        }
    }

    #[test]
    fn nothing_to_read_stays_here() {
        assert!(!use_peer(None, None, 0, 0));
        assert!(!use_peer(Some(1.0), Some(9.0), 100, 0));
    }

    #[test]
    fn unmeasured_peer_is_asked_once() {
        assert!(use_peer(None, None, 100, 100));
        assert!(use_peer(Some(1e6), None, 100, 100));
    }

    #[test]
    fn measured_peer_without_local_speed_is_asked() {
        assert!(use_peer(None, Some(1.0), 100, 100));
    }

    #[test]
    fn faster_peer_wins_and_a_tie_goes_to_the_peer() {
        assert!(use_peer(Some(100.0), Some(200.0), 100, 100));
        assert!(use_peer(Some(100.0), Some(100.0), 100, 100));
        assert!(!use_peer(Some(200.0), Some(100.0), 100, 100));
    }

    #[test]
    fn swapped_roles_decide_consistently() {
        // The slow instance asks the fast one, the fast instance reads for itself.
        let (slow, fast) = (100.0, 300.0);
        assert!(use_peer(Some(slow), Some(fast), 10, 10));
        assert!(!use_peer(Some(fast), Some(slow), 10, 10));
    }

    #[test]
    fn local_speed_needs_a_chunk() {
        let mut r = empty();
        r.note_local(10, 512, 1.0);
        assert_eq!(r.local_tok_s, None);
        r.note_local(1024, 512, 2.0);
        assert_eq!(r.local_tok_s, Some(512.0));
        r.note_local(512, 512, 4.0);
        assert_eq!(r.local_tok_s, Some(128.0));
    }

    #[test]
    fn degenerate_timings_change_nothing() {
        let mut r = empty();
        r.note_local(100, 10, 0.0);
        r.note_local(0, 10, 1.0);
        r.note_local(100, 10, f64::NAN);
        r.note_peer("a", 0, 1.0);
        r.note_peer("a", 10, 0.0);
        assert_eq!(r.local_tok_s, None);
        assert!(r.peers.is_empty());
    }

    #[test]
    fn peer_speed_is_kept_per_endpoint_and_replaced() {
        let mut r = empty();
        r.note_peer("a", 100, 2.0);
        r.note_peer("b", 100, 1.0);
        assert_eq!(r.peer_tok_s("a"), Some(50.0));
        assert_eq!(r.peer_tok_s("b"), Some(100.0));
        r.note_peer("a", 300, 1.0);
        assert_eq!(r.peer_tok_s("a"), Some(300.0));
        assert_eq!(r.peers.len(), 2);
        assert_eq!(r.peer_tok_s("c"), None);
    }

    #[test]
    fn stale_peer_is_measured_again() {
        let mut r = empty();
        r.note_peer("a", 100, 2.0);
        r.note_local(100, 10, 1.0);
        assert_eq!(r.peer_tok_s("a"), Some(50.0));
        r.note_local(100, 10, 1.0);
        assert_eq!(r.peer_tok_s("a"), None);
        r.note_peer("a", 100, 2.0);
        assert_eq!(r.peer_tok_s("a"), Some(50.0));
    }

    #[test]
    fn slow_peer_is_asked_again_after_local_reads() {
        let mut r = empty();
        r.note_local(1000, 10, 1.0);
        r.note_peer("a", 100, 10.0);
        let decide = |r: &Rates| use_peer(r.local_tok_s, r.peer_tok_s("a"), 100, 100);
        assert!(!decide(&r));
        for _ in 0..10 {
            r.note_local(1000, 10, 1.0);
        }
        assert!(decide(&r));
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
    fn table_is_shared_and_decides() {
        note_local_prefill(1000, 10, 1.0);
        note_handover("http://table-test", 100, 1.0);
        assert!(!peer_reads("http://table-test", 100, 100));
        assert!(peer_reads("http://table-test-unknown", 100, 100));
    }

    #[test]
    fn peer_hands_over_the_whole_prompt_against_the_missing_part() {
        // The peer delivers 200 tokens/s, the local prefill reads 100 tokens/s.
        // 1000 tokens from the peer take 5 s, so the peer wins from 500 missing tokens.
        assert!(use_peer(Some(100.0), Some(200.0), 1000, 1000));
        assert!(use_peer(Some(100.0), Some(200.0), 1000, 500));
        assert!(!use_peer(Some(100.0), Some(200.0), 1000, 499));
    }

    /// One instance's table: its own prefill speed and the speed of the peer it asks.
    fn view(local: f64, peer: Option<(&str, f64)>) -> Rates {
        let mut r = empty();
        r.note_local(1000, 10, 1000.0 / local);
        if let Some((name, tok_s)) = peer {
            r.note_peer(name, 1000, 1000.0 / tok_s);
        }
        r
    }

    #[test]
    fn both_instances_decide_on_their_own_table() {
        // A reads 100 tokens/s and B reads 400.
        // A measured B's hand-over at 300 tokens/s, B measured A's at 50.
        let a = view(100.0, Some(("b", 300.0)));
        let b = view(400.0, Some(("a", 50.0)));
        assert!(use_peer(a.local_tok_s, a.peer_tok_s("b"), 800, 800));
        assert!(!use_peer(b.local_tok_s, b.peer_tok_s("a"), 800, 800));
    }

    #[test]
    fn first_request_asks_then_the_measure_decides_on_both_sides() {
        let (mut a, mut b) = (empty(), empty());
        // Nothing measured: each side asks its peer, no speed is assumed.
        assert!(use_peer(a.local_tok_s, a.peer_tok_s("b"), 800, 800));
        assert!(use_peer(b.local_tok_s, b.peer_tok_s("a"), 800, 800));
        // A read 1000 tokens in 10 s, then took 800 tokens from B in 2 s.
        a.note_local(1000, 10, 10.0);
        a.note_peer("b", 800, 2.0);
        // B read 1000 tokens in 2.5 s, then took 800 tokens from A in 20 s.
        b.note_local(1000, 10, 2.5);
        b.note_peer("a", 800, 20.0);
        assert!(use_peer(a.local_tok_s, a.peer_tok_s("b"), 800, 800));
        assert!(!use_peer(b.local_tok_s, b.peer_tok_s("a"), 800, 800));
    }

    #[test]
    fn stale_speed_asks_again_on_either_side() {
        let mut b = view(400.0, Some(("a", 50.0)));
        assert!(!use_peer(b.local_tok_s, b.peer_tok_s("a"), 800, 800));
        // The hand-over took 20 s, so 20 s of local prefill make it stale.
        b.note_local(8000, 10, 20.0);
        assert_eq!(b.peer_tok_s("a"), None);
        assert!(use_peer(b.local_tok_s, b.peer_tok_s("a"), 800, 800));
    }
}
