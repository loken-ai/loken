//! Who is in the cluster, and how confident we are that a silent node is gone.
//!
//! A fixed timeout has to choose between two failures. Short, and a node that paused for a
//! garbage collection or a long prefill is declared dead, its work re-routed, its caches
//! abandoned - under load, exactly when the cluster can least afford it. Long, and a node that
//! really died keeps receiving requests until the timeout expires. There is no setting that is
//! right for both, because the question "is this silence abnormal?" depends on what this
//! node's silences usually look like.
//!
//! So the detector learns that. It keeps the recent intervals between a node's heartbeats and
//! reports phi, a suspicion level: the negative log of the probability that a node this regular
//! would still be silent after this long. A node that beats every 200 ms and has been quiet for
//! 250 ms is unremarkable; the same silence from a node that has never missed 200 ms by more
//! than a millisecond is not. One threshold then means the same thing for a node on a quiet
//! link and a node on a congested one, which is what a single timeout can never do.
//!
//! Time is a parameter here, never read from the clock. A failure detector tested against the
//! real clock is a test that passes on an idle machine and fails on a busy one, which is the
//! same false positive it exists to prevent.

use std::collections::{HashMap, VecDeque};

/// How a node is named across the cluster.
pub type NodeId = String;

/// What a node publishes about itself. Gossip carries this; the detector only cares that it
/// arrived, but the routing table needs the contents.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize)]
pub struct NodeState {
    /// Devices it can place work on, as the local probe describes them.
    pub devices: Vec<String>,
    /// Recent load, 0.0 idle to 1.0 saturated.
    pub load: f32,
    /// Requests currently held - in flight plus queued. The raw count, because a fraction
    /// cannot express a QUEUE: two nodes at load 1.0 with 2 and 40 waiting are not equals.
    pub busy: u32,
    /// How many requests this node serves at once (its admission gate's width). Zero means
    /// the peer never said - an older build - and the estimate falls back to the load share.
    pub lanes: u32,
    /// Models it currently holds resident.
    pub models: Vec<String>,
    /// Models it could load - its catalogue on disk, which is a different question from what
    /// it holds. Residency prices a load; the catalogue decides candidacy.
    ///
    /// `None` and `Some(empty)` are DIFFERENT facts and the distinction is load-bearing.
    /// None means the node never said - an older build, or a directory it could not read -
    /// and it must not be struck off on silence. Some(empty) means it said, and the answer
    /// was nothing: that node cannot serve any model and routing must skip it. Collapsing
    /// the two makes a weightless node look like an unknown one, and it keeps the requests
    /// it cannot answer.
    pub serves: Option<Vec<String>>,
    /// This node's row of the link cost matrix: what it measures to each peer, GB/s.
    pub link_gbps: HashMap<NodeId, f64>,
    /// Prefix block hashes it holds, so a request can be routed to where its prefix already is.
    pub prefix_blocks: Vec<u64>,
    /// What each of its cards admits as a whole load, in bytes, under its own memory
    /// fraction. Empty means the node never said - an older build - and a hand-over
    /// then cannot tell whether the peer holds a render, which reads as "it may".
    pub cards: Vec<u64>,
}

impl NodeState {
    /// Whether one card of this node admits `bytes` whole, or it never said.
    pub fn card_may_hold(&self, bytes: u64) -> bool {
        self.cards.is_empty() || self.cards.iter().any(|c| *c >= bytes)
    }
}

/// Suspicion that one node has failed, learned from its own rhythm.
#[derive(Debug, Clone)]
pub struct PhiAccrual {
    intervals: VecDeque<f64>,
    window: usize,
    last_seen_ms: u64,
    /// Until enough intervals are known, a node is judged against this instead of against a
    /// distribution estimated from one or two samples - which would be confident nonsense.
    bootstrap_interval_ms: f64,
    /// How long a silence stays ordinary, beyond the rhythm itself.
    ///
    /// This cannot be learned, and that is the point. A node's history says how regular it has
    /// been, never how long it is ALLOWED to pause - a long prefill or a page-cache stall is
    /// legitimate and has simply not happened yet. Without it, a node whose beats are very
    /// regular has almost no observed variance, so the first hiccup reads as impossibly
    /// abnormal: the detector was rejecting a three-beat pause at phi 300, which would have
    /// evicted the steadiest node in the cluster on its first stall.
    acceptable_pause_ms: f64,
}

impl PhiAccrual {
    pub fn new(now_ms: u64, expected_interval_ms: f64, acceptable_pause_ms: f64) -> Self {
        Self {
            intervals: VecDeque::new(),
            window: 64,
            last_seen_ms: now_ms,
            bootstrap_interval_ms: expected_interval_ms,
            acceptable_pause_ms,
        }
    }

    /// A heartbeat arrived.
    pub fn heartbeat(&mut self, now_ms: u64) {
        let delta = now_ms.saturating_sub(self.last_seen_ms) as f64;
        self.last_seen_ms = now_ms;
        if delta > 0.0 {
            if self.intervals.len() == self.window {
                self.intervals.pop_front();
            }
            self.intervals.push_back(delta);
        }
    }

    fn mean(&self) -> f64 {
        if self.intervals.is_empty() {
            return self.bootstrap_interval_ms;
        }
        self.intervals.iter().sum::<f64>() / self.intervals.len() as f64
    }

    /// Spread of the observed intervals, floored against the tolerated pause.
    ///
    /// The floor is what keeps a metronomic node from being fragile: with a near-zero observed
    /// variance, suspicion would otherwise rise by hundreds for a delay of one interval.
    fn stddev(&self) -> f64 {
        let floor = (self.acceptable_pause_ms / 4.0).max(1.0);
        if self.intervals.len() < 2 {
            return (self.mean() * 0.1).max(floor);
        }
        let m = self.mean();
        let var = self.intervals.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            / (self.intervals.len() - 1) as f64;
        var.sqrt().max(floor)
    }

    /// How abnormal this silence is. 1 means "roughly one chance in ten this is normal",
    /// 8 means one in a hundred million.
    pub fn phi(&self, now_ms: u64) -> f64 {
        let elapsed = now_ms.saturating_sub(self.last_seen_ms) as f64;
        if elapsed <= 0.0 {
            return 0.0;
        }
        // Suspicion is measured against the rhythm PLUS what a node is allowed to pause for.
        let normal = self.mean() + self.acceptable_pause_ms;
        // The probability that an interval exceeds `elapsed`, under a normal fitted to the
        // observed intervals. The logistic approximation of the normal tail keeps this to
        // arithmetic - no error function, no table.
        let y = (elapsed - normal) / self.stddev();
        let e = (-y * (1.5976 + 0.070566 * y * y)).exp();
        let p_later = if elapsed > normal {
            e / (1.0 + e)
        } else {
            1.0 - 1.0 / (1.0 + e)
        };
        -p_later.max(1e-300).log10()
    }
}

/// The cluster as this node currently believes it to be.
#[derive(Debug, Clone)]
pub struct Membership {
    detectors: HashMap<NodeId, PhiAccrual>,
    states: HashMap<NodeId, NodeState>,
    /// Suspicion above which a node stops being routed to. 8 is roughly one false eviction in
    /// a hundred million judgements, which on a per-second gossip is centuries.
    pub threshold: f64,
    expected_interval_ms: f64,
    /// Silence this long past a node's rhythm is still ordinary. Defaults to three beats:
    /// long enough for a prefill or a stall, short enough that a dead node is caught in
    /// single-digit rounds.
    pub acceptable_pause_ms: f64,
}

impl Membership {
    pub fn new(expected_interval_ms: f64) -> Self {
        Self {
            detectors: HashMap::new(),
            states: HashMap::new(),
            threshold: 8.0,
            expected_interval_ms,
            acceptable_pause_ms: expected_interval_ms * 3.0,
        }
    }

    /// A gossip round brought news of a node.
    pub fn observe(&mut self, node: &NodeId, now_ms: u64, state: NodeState) {
        self.detectors
            .entry(node.clone())
            .or_insert_with(|| {
                PhiAccrual::new(now_ms, self.expected_interval_ms, self.acceptable_pause_ms)
            })
            .heartbeat(now_ms);
        self.states.insert(node.clone(), state);
    }

    /// Suspicion for one node, for a caller that wants to weigh rather than decide.
    pub fn phi(&self, node: &NodeId, now_ms: u64) -> Option<f64> {
        self.detectors.get(node).map(|d| d.phi(now_ms))
    }

    /// Who may be routed to right now.
    pub fn alive(&self, now_ms: u64) -> Vec<NodeId> {
        let mut v: Vec<NodeId> = self
            .detectors
            .iter()
            .filter(|(_, d)| d.phi(now_ms) < self.threshold)
            .map(|(n, _)| n.clone())
            .collect();
        v.sort();
        v
    }

    /// What a live node published.
    /// Every node this table has ever heard from, alive or not. Eviction removes a node from
    /// routing; it stays here until `forget_dead` drops it, which is what lets an observer see
    /// a node go quiet rather than simply vanish.
    pub fn known(&self) -> impl Iterator<Item = (&NodeId, &NodeState)> {
        self.states.iter()
    }

    pub fn state_of(&self, node: &NodeId) -> Option<&NodeState> {
        self.states.get(node)
    }

    /// Forget nodes that have been silent long past the threshold, so their published state
    /// stops taking room. Kept separate from `alive`: eviction from ROUTING must be instant,
    /// eviction from MEMORY can wait, and a node that returns before this is a reconnection
    /// rather than a join.
    pub fn forget_dead(&mut self, now_ms: u64, grace: f64) {
        let doomed: Vec<NodeId> = self
            .detectors
            .iter()
            .filter(|(_, d)| d.phi(now_ms) > self.threshold * grace)
            .map(|(n, _)| n.clone())
            .collect();
        for n in doomed {
            self.detectors.remove(&n);
            self.states.remove(&n);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BEAT: u64 = 200;

    fn steady(m: &mut Membership, node: &str, from_ms: u64, beats: u64) -> u64 {
        let mut t = from_ms;
        for _ in 0..beats {
            m.observe(&node.to_string(), t, NodeState::default());
            t += BEAT;
        }
        t
    }

    /// A node keeping its rhythm is never suspected, however long it goes on.
    #[test]
    fn a_regular_node_is_never_suspected() {
        let mut m = Membership::new(BEAT as f64);
        let mut t = 0;
        for _ in 0..500 {
            m.observe(&"a".to_string(), t, NodeState::default());
            assert!(m.phi(&"a".to_string(), t).unwrap() < m.threshold);
            t += BEAT;
        }
        assert_eq!(m.alive(t), vec!["a".to_string()]);
    }

    /// The invariant's first half: a node that stops is evicted from routing within a bounded
    /// number of rounds - not eventually, and not after a timeout nobody tuned.
    #[test]
    fn a_killed_node_leaves_the_routing_table_within_a_bounded_number_of_rounds() {
        let mut m = Membership::new(BEAT as f64);
        let t = steady(&mut m, "a", 0, 100);
        steady(&mut m, "b", 0, 100);

        // "a" dies here; "b" keeps beating.
        let mut rounds = 0;
        let mut now = t;
        while m.alive(now).contains(&"a".to_string()) {
            now += BEAT;
            rounds += 1;
            m.observe(&"b".to_string(), now, NodeState::default());
            assert!(
                rounds < 20,
                "still routing to a dead node after {rounds} rounds"
            );
        }
        assert!(rounds <= 10, "took {rounds} rounds to notice");
        assert_eq!(
            m.alive(now),
            vec!["b".to_string()],
            "b was not caught in the eviction"
        );
    }

    /// The invariant's second half, and the reason for phi rather than a timeout: a node that
    /// pauses briefly and comes back is NOT evicted. A fixed timeout short enough to catch the
    /// death above would have killed this one.
    #[test]
    fn a_paused_then_resumed_node_is_not_evicted() {
        let mut m = Membership::new(BEAT as f64);
        let t = steady(&mut m, "a", 0, 100);

        // A pause of three beats - a long prefill, a page-cache stall - then it resumes.
        let resumed_at = t + BEAT * 3;
        assert!(
            m.alive(resumed_at).contains(&"a".to_string()),
            "phi {} crossed {} during a pause a live node can have",
            m.phi(&"a".to_string(), resumed_at).unwrap(),
            m.threshold
        );
        m.observe(&"a".to_string(), resumed_at, NodeState::default());
        assert!(m.alive(resumed_at).contains(&"a".to_string()));
        assert!(
            m.phi(&"a".to_string(), resumed_at).unwrap() < 1.0,
            "suspicion cleared on return"
        );
    }

    /// What a single timeout cannot express: two nodes with different rhythms, judged by one
    /// threshold. The slow-but-regular node keeps its place; the fast one that falls silent
    /// for the SAME duration is suspected, because for it that silence is abnormal.
    #[test]
    fn one_threshold_means_the_same_thing_on_two_different_rhythms() {
        let mut m = Membership::new(BEAT as f64);
        // "slow" beats every 2 s, "fast" every 50 ms.
        let mut t = 0;
        for _ in 0..100 {
            m.observe(&"slow".to_string(), t, NodeState::default());
            t += 2000;
        }
        let mut t2 = 0;
        for _ in 0..100 {
            m.observe(&"fast".to_string(), t2, NodeState::default());
            t2 += 50;
        }
        let now = t.max(t2) + 1000; // a second of silence for both

        let phi_slow = m.phi(&"slow".to_string(), now).unwrap();
        let phi_fast = m.phi(&"fast".to_string(), now).unwrap();
        assert!(
            phi_fast > phi_slow,
            "a second is alarming for a 50 ms node and ordinary for a 2 s one \
             (fast {phi_fast:.1}, slow {phi_slow:.1})"
        );
    }

    /// Routing eviction and forgetting are separate decisions: a node stops receiving work
    /// immediately, but what it published survives long enough for a reconnection to be a
    /// reconnection.
    #[test]
    fn a_node_leaves_routing_before_it_leaves_memory() {
        let mut m = Membership::new(BEAT as f64);
        let t = steady(&mut m, "a", 0, 100);
        let state = NodeState {
            load: 0.5,
            models: vec!["qwen3:8b".into()],
            ..Default::default()
        };
        m.observe(&"a".to_string(), t, state);

        let quiet = t + BEAT * 20;
        assert!(
            !m.alive(quiet).contains(&"a".to_string()),
            "should not be routed to"
        );
        assert!(m.state_of(&"a".to_string()).is_some(), "but still known");

        m.forget_dead(quiet, 1.0);
        assert!(
            m.state_of(&"a".to_string()).is_none(),
            "forgotten once well past the threshold"
        );
    }

    /// Gossip carries what routing needs, so the payload has to survive a round trip intact.
    #[test]
    fn a_node_publishes_what_routing_needs() {
        let mut m = Membership::new(BEAT as f64);
        let mut link = HashMap::new();
        link.insert("b".to_string(), 12.5);
        let state = NodeState {
            devices: vec!["cuda:0".into(), "cuda:1".into()],
            load: 0.25,
            busy: 3,
            lanes: 8,
            models: vec!["gemma4:12b".into()],
            serves: Some(vec!["gemma4:12b".into(), "qwen3:8b".into()]),
            link_gbps: link,
            prefix_blocks: vec![0xABCD, 0x1234],
            cards: vec![16 << 30],
        };
        m.observe(&"a".to_string(), 0, state.clone());
        assert_eq!(m.state_of(&"a".to_string()), Some(&state));
    }
}

#[cfg(test)]
mod card_tests {
    use super::NodeState;

    #[test]
    fn a_peer_holds_a_demand_on_one_of_its_cards_or_never_said() {
        let peer = NodeState {
            cards: vec![8 << 30, 16 << 30],
            ..Default::default()
        };
        assert!(peer.card_may_hold(12 << 30));
        assert!(!peer.card_may_hold(20 << 30));
        // An older build publishes no cards; silence is not a refusal.
        let silent = NodeState::default();
        assert!(silent.card_may_hold(20 << 30));
    }
}
