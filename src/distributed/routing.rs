//! Which node should answer a request, decided by prediction rather than by availability.
//!
//! "Forward when the local node is busy" is the obvious rule and the wrong one. It sends work
//! to a node that is free because it is slow, and it keeps work at home when home holds the
//! prompt's prefix in cache and the alternative would have to prefill it from nothing. Both
//! mistakes cost a whole request, and neither is visible from a load average.
//!
//! So the comparison is between predicted completions: forward to B only when
//! `predicted(B) + rtt < predicted(A)`. That single form covers the cases separately - the node
//! that holds the model but is saturated, the node that holds it but is simply slower, and the
//! node that holds nothing at all but can fetch and still finish first.
//!
//! The prefix cache enters the same arithmetic instead of being a special case. A node holding
//! the prompt's blocks skips that much prefill, which is a smaller predicted completion, which
//! is already what the comparison reads. That keeps one rule instead of two that can disagree.
//!
//! Nothing here reads a clock or a socket: it is arithmetic over what gossip published, so a
//! routing decision can be tested exactly rather than observed and hoped for.

use std::collections::HashMap;

use super::membership::{Membership, NodeId, NodeState};

/// What a request needs, in the terms routing can weigh.
#[derive(Debug, Clone)]
pub struct RequestShape {
    pub model: String,
    /// How long the prompt is. An estimate is fine and a count is better; what matters is
    /// that it is the same figure on every node, since it is compared against what each peer
    /// says it already holds.
    pub prompt_tokens: u32,
    /// Tokens to generate.
    pub max_tokens: u32,
}

/// What a node is worth for one request, all of it derived from published state.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeEstimate {
    pub node: NodeId,
    /// Predicted milliseconds until the last token, transport included.
    pub completion_ms: f64,
    /// Prompt TOKENS this node says it already holds - answered by the peer, which is the
    /// only party that can tokenise for its own model and look in its own cache.
    pub prefix_hits: usize,
    /// True when the node must load the model before it can start.
    pub needs_load: bool,
}

/// Rates a node publishes about itself, measured by that node rather than assumed here.
#[derive(Debug, Clone, Copy)]
pub struct NodeRates {
    pub prefill_tok_per_s: f64,
    pub decode_tok_per_s: f64,
    /// Seconds to make the model resident when it is not.
    pub model_load_s: f64,
    /// Tokens per second the node sustains ACROSS everything in flight, measured as
    /// per-request rate times width. Zero = never measured; the estimate then falls back to
    /// the width models below.
    pub agg_tok_per_s: f64,
}

impl Default for NodeRates {
    fn default() -> Self {
        // Only used for a peer that has published nothing yet. Deliberately pessimistic: a
        // node we know nothing about should have to prove itself, not win by default.
        Self {
            prefill_tok_per_s: 50.0,
            decode_tok_per_s: 5.0,
            model_load_s: 60.0,
            agg_tok_per_s: 0.0,
        }
    }
}

/// Can this node run this model at all - not "does it hold it", but "could it".
///
/// The distinction is the whole difference between a cluster that works and one that looks
/// like it does. A node holding nothing resident looks identical to every other node holding
/// nothing resident, so the local one wins the tie and then fails on a model whose weights it
/// has never had. Residency prices a load; the catalogue decides candidacy.
///
/// Silence and an empty catalogue are not the same answer - see `NodeState::serves`.
pub fn can_serve(state: &NodeState, model: &str) -> bool {
    match &state.serves {
        // Never told us. Judge it on residency alone, as before catalogues existed.
        None => true,
        Some(catalogue) => {
            catalogue.iter().any(|m| m == model) || state.models.iter().any(|m| m == model)
        }
    }
}

/// Predict how long `node` needs, given what it published.
///
/// Queueing is modelled as the simplest thing that is true: a node reporting load `l` gives a
/// new request roughly `1 - l` of itself. It is coarse, and it is measured rather than guessed,
/// which is what matters - the alternative is a rule that cannot see a saturated node at all.
pub fn estimate(
    node: &NodeId,
    state: &NodeState,
    rates: NodeRates,
    req: &RequestShape,
    rtt_ms: f64,
    cached_tokens: u32,
) -> NodeEstimate {
    // What the node says it already holds, never more than the prompt itself: a peer that
    // over-reports would win every request and then prefill from scratch anyway.
    let hits = cached_tokens.min(req.prompt_tokens) as usize;
    let to_prefill = f64::from(req.prompt_tokens) - hits as f64;

    // Service time for THIS request at the node's measured per-request rates.
    let service_ms = to_prefill / rates.prefill_tok_per_s * 1000.0
        + f64::from(req.max_tokens) / rates.decode_tok_per_s * 1000.0;
    // The wait is a QUEUE, not a discount. A node runs `lanes` requests at once (its
    // admission width) and each takes a service time, so a new arrival waits
    // busy/lanes rounds - which is what the fractional `1 - load` share could never say:
    // measured at 24 in flight on a 724 tok/s node, per-request rate fell to a third while
    // the share model still called it nearly free. Peers from an older build publish no
    // lanes and keep the share model.
    let (prefill_ms, decode_ms) = if rates.agg_tok_per_s > 0.0 {
        // The batching model, when the node has measured itself: everything in flight shares
        // one sustained throughput, so a new request finishes after (busy + 1) requests'
        // worth of tokens have flowed. This is what the width models below approximate -
        // measured, it needs no lane count at all.
        let decode =
            f64::from(req.max_tokens) * f64::from(state.busy + 1) / rates.agg_tok_per_s * 1000.0;
        (to_prefill / rates.prefill_tok_per_s * 1000.0, decode)
    } else if state.lanes > 0 {
        let rounds_ahead = (state.busy / state.lanes) as f64;
        (rounds_ahead * service_ms, service_ms)
    } else {
        let share = (1.0 - state.load as f64).max(0.05);
        (
            to_prefill / (rates.prefill_tok_per_s * share) * 1000.0,
            f64::from(req.max_tokens) / (rates.decode_tok_per_s * share) * 1000.0,
        )
    };
    let needs_load = !state.models.iter().any(|m| m == &req.model);
    let load_ms = if needs_load {
        rates.model_load_s * 1000.0
    } else {
        0.0
    };

    NodeEstimate {
        node: node.clone(),
        completion_ms: prefill_ms + decode_ms + load_ms + rtt_ms,
        prefix_hits: hits,
        needs_load,
    }
}

/// Pick where a request should run.
///
/// `local` is always considered, and wins ties: moving a request has costs this arithmetic does
/// not model - a second failure domain, a stream to relay - so it must be strictly better, not
/// merely equal.
pub fn choose(
    local: &NodeId,
    members: &Membership,
    now_ms: u64,
    rates: &HashMap<NodeId, NodeRates>,
    rtt_ms: &HashMap<NodeId, f64>,
    req: &RequestShape,
    // What each node answered when asked how much of this prompt it already holds. A node
    // absent from the map answered nothing and is priced as holding none of it - the safe
    // direction: it may be pleasantly surprised, never disappointed.
    cached: &HashMap<NodeId, u32>,
    // Peers serving a penalty for a failed hand-over. Not a ranking input: a node that fails
    // quickly scores WELL, so the only honest treatment is to leave it out of the comparison.
    barred: &std::collections::HashSet<NodeId>,
    // Hand-overs dispatched to each peer since its last report: its published busy cannot
    // include them yet, so they are charged here - otherwise every gossip interval is a
    // window in which the most attractive peer absorbs the whole queue.
    sent_since_report: &HashMap<NodeId, u32>,
) -> Option<NodeEstimate> {
    let mut best: Option<NodeEstimate> = None;
    for node in members.alive(now_ms) {
        // The local node is never barred: there is nowhere else for its own requests to go.
        if &node != local && barred.contains(&node) {
            continue;
        }
        let Some(state) = members.state_of(&node) else {
            continue;
        };
        if !can_serve(state, &req.model) {
            continue;
        }
        let r = rates.get(&node).copied().unwrap_or_default();
        let hop = if &node == local {
            0.0
        } else {
            rtt_ms.get(&node).copied().unwrap_or(1.0)
        };
        let hits = cached.get(&node).copied().unwrap_or(0);
        let in_flight_to = sent_since_report.get(&node).copied().unwrap_or(0);
        let e = if in_flight_to > 0 {
            let mut charged = state.clone();
            charged.busy += in_flight_to;
            estimate(&node, &charged, r, req, hop, hits)
        } else {
            estimate(&node, state, r, req, hop, hits)
        };
        best = match best {
            None => Some(e),
            Some(b) => {
                // Strictly better, and the local node keeps a tie.
                let better = e.completion_ms < b.completion_ms;
                let local_ties = &b.node != local
                    && &e.node == local
                    && (e.completion_ms - b.completion_ms).abs() < f64::EPSILON;
                if better || local_ties {
                    Some(e)
                } else {
                    Some(b)
                }
            }
        };
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(models: &[&str], load: f32, blocks: &[u64]) -> NodeState {
        NodeState {
            models: models.iter().map(|s| s.to_string()).collect(),
            load,
            prefix_blocks: blocks.to_vec(),
            ..Default::default()
        }
    }

    /// A prompt of `prompt_tokens`, which is what routing weighs now.
    fn req(prompt_tokens: u32) -> RequestShape {
        RequestShape {
            model: "qwen3:8b".into(),
            prompt_tokens,
            max_tokens: 128,
        }
    }

    fn none_sent() -> HashMap<NodeId, u32> {
        HashMap::new()
    }

    /// Between two reports a peer's published busy is frozen; the hand-overs already sent
    /// must deepen its queue in the comparison, or a burst lands on it whole.
    #[test]
    fn hand_overs_in_flight_deepen_the_peer_queue_before_it_reports() {
        let mut m = Membership::new(1000.0);
        let mut fast = node(&["qwen3:8b"], 0.0, &[]);
        fast.lanes = 1;
        let mut here = node(&["qwen3:8b"], 0.0, &[]);
        here.busy = 2;
        here.lanes = 1;
        m.observe(&"b".to_string(), 0, fast);
        m.observe(&"a".to_string(), 0, here);
        let rates: HashMap<NodeId, NodeRates> = [
            ("a".to_string(), NodeRates::default()),
            ("b".to_string(), NodeRates::default()),
        ]
        .into();
        let rtt = HashMap::new();
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(
            got.node, "b",
            "idle peer wins while nothing is in flight to it"
        );
        let sent: HashMap<NodeId, u32> = [("b".to_string(), 8)].into();
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &sent,
        )
        .unwrap();
        assert_eq!(
            got.node, "a",
            "eight undelivered hand-overs outweigh a frozen idle report"
        );
    }

    /// What peers answered when asked how much of the prompt they hold.
    fn cached(pairs: &[(&str, u32)]) -> HashMap<NodeId, u32> {
        pairs.iter().map(|(n, v)| (n.to_string(), *v)).collect()
    }

    fn fast() -> NodeRates {
        NodeRates {
            prefill_tok_per_s: 2000.0,
            decode_tok_per_s: 80.0,
            model_load_s: 20.0,
            agg_tok_per_s: 0.0,
        }
    }
    fn slow() -> NodeRates {
        NodeRates {
            prefill_tok_per_s: 400.0,
            decode_tok_per_s: 12.0,
            model_load_s: 20.0,
            agg_tok_per_s: 0.0,
        }
    }

    /// No peer is serving a penalty - the ordinary case.
    fn no_bar() -> std::collections::HashSet<NodeId> {
        std::collections::HashSet::new()
    }

    fn members(entries: &[(&str, NodeState)]) -> Membership {
        let mut m = Membership::new(200.0);
        for (n, s) in entries {
            m.observe(&n.to_string(), 0, s.clone());
        }
        m
    }

    /// The case a load average gets wrong: the idle node is idle because it is slow.
    #[test]
    fn an_idle_but_slow_node_does_not_win() {
        let m = members(&[
            ("a", node(&["qwen3:8b"], 0.5, &[])),
            ("b", node(&["qwen3:8b"], 0.0, &[])),
        ]);
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), slow())]);
        let rtt = HashMap::from([("b".to_string(), 1.0)]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(got.node, "a", "half-loaded and quick beats idle and slow");
    }

    /// And the case that keeps a request at home: a LONG prefix is already here.
    ///
    /// The length matters, and getting it wrong is how this test failed twice - once in
    /// blocks, once again after the contract moved to tokens and the same figure meant
    /// something far smaller. The crossover is arithmetic, not intuition: at these rates a
    /// 40% load costs 1066 ms of decode, and skipping prefill saves half a millisecond per
    /// token, so the cache only wins past roughly 2100 prompt tokens.
    #[test]
    fn a_long_cached_prefix_keeps_the_request_where_the_prefix_is() {
        let m = members(&[
            ("a", node(&["qwen3:8b"], 0.4, &[])),
            ("b", node(&["qwen3:8b"], 0.0, &[])),
        ]);
        // Same hardware on both, so only the cache can decide.
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), fast())]);
        let rtt = HashMap::from([("b".to_string(), 1.0)]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(8192),
            &cached(&[("a", 8192)]),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(got.node, "a");
        assert_eq!(got.prefix_hits, 8192, "the whole prompt was already here");
    }

    /// But not at any price - and the price is lower than it looks. A moderate load is enough
    /// to outweigh a short prefix, so the rule cannot be "prefer the cache". Same prompt as
    /// above, one eighth of it cached: the saving no longer covers the load.
    #[test]
    fn a_short_cached_prefix_loses_to_a_node_that_is_simply_free() {
        let m = members(&[
            ("a", node(&["qwen3:8b"], 0.4, &[])),
            ("b", node(&["qwen3:8b"], 0.0, &[])),
        ]);
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), fast())]);
        let rtt = HashMap::from([("b".to_string(), 1.0)]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(8192),
            &cached(&[("a", 1024)]),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(
            got.node, "b",
            "a short prefix cannot pay for a saturated node"
        );
    }

    /// The case measured on real hardware: a fast node with a deep queue must lose to an
    /// idle slower peer once the wait dominates. The share model never let this happen -
    /// at 24 in flight it still priced the saturated node as nearly free.
    #[test]
    fn a_deep_queue_sends_the_request_to_an_idle_slower_peer() {
        let mut fast_but_buried = node(&["qwen3:8b"], 1.0, &[]);
        // The boundary is arithmetic: at these rates one round of service here costs 1.6 s
        // against 10.8 s on the slow peer, so the hand-over pays only past six rounds of
        // queue. Eight rounds makes the verdict unambiguous.
        fast_but_buried.busy = 64;
        fast_but_buried.lanes = 8;
        let mut idle_and_slow = node(&["qwen3:8b"], 0.0, &[]);
        idle_and_slow.busy = 0;
        idle_and_slow.lanes = 8;
        let m = members(&[("a", fast_but_buried), ("b", idle_and_slow)]);
        // a is 3x faster per token - and 3 full rounds deep.
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), slow())]);
        let rtt = HashMap::from([("b".to_string(), 2.0)]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(
            got.node, "b",
            "eight rounds of queue outweigh a 6.6x rate advantage"
        );
    }

    /// And the converse: a shallow queue on the fast node is still worth waiting for.
    #[test]
    fn a_shallow_queue_keeps_the_request_on_the_faster_node() {
        let mut fast_short_queue = node(&["qwen3:8b"], 0.5, &[]);
        fast_short_queue.busy = 4;
        fast_short_queue.lanes = 8;
        let mut idle_and_slow = node(&["qwen3:8b"], 0.0, &[]);
        idle_and_slow.busy = 0;
        idle_and_slow.lanes = 8;
        let m = members(&[("a", fast_short_queue), ("b", idle_and_slow)]);
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), slow())]);
        let rtt = HashMap::from([("b".to_string(), 2.0)]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(
            got.node, "a",
            "under one round of queue, the 3x rate still wins"
        );
    }

    /// A peer that failed a hand-over must not be ranked, only excluded. It answers fast when
    /// it answers with an error, and a fast answer is exactly what this arithmetic rewards.
    #[test]
    fn a_barred_peer_is_not_a_candidate_however_good_it_looks() {
        let m = members(&[
            ("a", node(&["qwen3:8b"], 0.9, &[])),
            ("b", node(&["qwen3:8b"], 0.0, &[])),
        ]);
        let rates = HashMap::from([("a".to_string(), slow()), ("b".to_string(), fast())]);
        let rtt = HashMap::from([("b".to_string(), 1.0)]);
        let free = HashMap::new();
        // Unbarred, the idle fast peer takes it - which is the whole point of the comparison.
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &free,
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(got.node, "b");
        // Barred, the saturated slow local node keeps it rather than hand it to a peer that
        // just refused one.
        let barred = std::collections::HashSet::from(["b".to_string()]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &free,
            &barred,
            &none_sent(),
        )
        .unwrap();
        assert_eq!(got.node, "a");
    }

    /// Barring the local node would leave the request nowhere to go, so it is never barred.
    #[test]
    fn the_local_node_is_never_barred_from_its_own_requests() {
        let m = members(&[("a", node(&["qwen3:8b"], 0.0, &[]))]);
        let rates = HashMap::from([("a".to_string(), fast())]);
        let barred = std::collections::HashSet::from(["a".to_string()]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &HashMap::new(),
            &req(64),
            &HashMap::new(),
            &barred,
            &none_sent(),
        );
        assert_eq!(got.map(|e| e.node), Some("a".to_string()));
    }

    /// The invariant a two-machine run broke: a node with NO weights on disk kept the request
    /// and failed on it. Both nodes hold nothing resident, so residency alone cannot tell them
    /// apart - and the local one wins ties. Only the catalogue separates them.
    #[test]
    fn a_node_whose_catalogue_is_empty_is_not_a_candidate() {
        let mut empty = node(&[], 0.0, &[]);
        empty.serves = Some(vec![]);
        let mut stocked = node(&[], 0.0, &[]);
        stocked.serves = Some(vec!["qwen3:8b".into()]);
        let m = members(&[("a", empty), ("b", stocked)]);
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), fast())]);
        let rtt = HashMap::from([("b".to_string(), 1.0)]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(
            got.node, "b",
            "a is local and idle, but it has no weights to load"
        );
    }

    /// And the other half: a node that never published a catalogue must not be struck off on
    /// silence. Otherwise one older build in the cluster strands every request it receives.
    #[test]
    fn a_node_that_published_no_catalogue_is_still_a_candidate() {
        let silent = node(&["qwen3:8b"], 0.0, &[]);
        assert!(
            silent.serves.is_none(),
            "the fixture must be the silent case"
        );
        let m = members(&[("a", silent)]);
        let rates = HashMap::from([("a".to_string(), fast())]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &HashMap::new(),
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        );
        assert_eq!(got.map(|e| e.node), Some("a".to_string()));
    }

    /// The plan's invariant: a node holding no weights can still serve anything the cluster
    /// can serve - it forwards. Here the local node has nothing and must not be chosen.
    #[test]
    fn a_node_holding_no_weights_forwards_rather_than_refuses() {
        let m = members(&[
            ("a", node(&[], 0.0, &[])),
            ("b", node(&["qwen3:8b"], 0.3, &[])),
        ]);
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), fast())]);
        let rtt = HashMap::from([("b".to_string(), 2.0)]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(got.node, "b");
        assert!(!got.needs_load, "b already holds it");
    }

    /// The other half of the invariant: a request whose prefix lives on exactly one live node
    /// goes there. And when that node dies, routing must move rather than wait.
    ///
    /// The prefix is the peer's ANSWER, not what it publishes: `prefix_blocks` describes blocks
    /// hashed under the peer's own tokeniser, which no other node can reproduce. Both nodes are
    /// therefore identical here except for what they answered.
    #[test]
    fn routing_follows_the_prefix_and_abandons_a_dead_node() {
        let mut m = Membership::new(200.0);
        for t in (0..2000).step_by(200) {
            m.observe(&"a".to_string(), t, node(&["qwen3:8b"], 0.2, &[]));
            m.observe(&"b".to_string(), t, node(&["qwen3:8b"], 0.2, &[]));
        }
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), fast())]);
        let rtt = HashMap::from([("b".to_string(), 1.0)]);
        // Long enough that skipping the prefill outweighs the hop - a short prompt would not,
        // and that boundary has its own test above.
        let held = cached(&[("b", 4096)]);

        let got = choose(
            &"a".to_string(),
            &m,
            2000,
            &rates,
            &rtt,
            &req(4096),
            &held,
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(got.node, "b", "the prefix is there");
        assert_eq!(got.prefix_hits, 4096);

        // "b" dies; "a" keeps beating. Routing must return home rather than wait for a peer
        // that will not answer.
        let mut now = 2000;
        for _ in 0..12 {
            now += 200;
            m.observe(&"a".to_string(), now, node(&["qwen3:8b"], 0.2, &[]));
        }
        let got = choose(
            &"a".to_string(),
            &m,
            now,
            &rates,
            &rtt,
            &req(4096),
            &held,
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(
            got.node, "a",
            "a dead node keeps no request, however good its cache was"
        );
    }

    /// Moving a request has costs this arithmetic does not model, so an equal peer never wins.
    #[test]
    fn an_equal_peer_does_not_take_the_request() {
        let m = members(&[
            ("a", node(&["qwen3:8b"], 0.3, &[])),
            ("b", node(&["qwen3:8b"], 0.3, &[])),
        ]);
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), fast())]);
        // Zero RTT makes the two strictly equal, which is the only way to test the tie rule.
        let rtt = HashMap::from([("b".to_string(), 0.0)]);
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(got.node, "a", "ties stay home");
    }

    /// A node that would have to fetch the model first is priced for it, so it wins only when
    /// it would still finish sooner - a long generation on a fast free node, not a short one.
    #[test]
    fn a_node_that_must_load_the_model_is_priced_for_it() {
        let m = members(&[
            ("a", node(&["qwen3:8b"], 0.6, &[])),
            ("b", node(&[], 0.0, &[])),
        ]);
        let rates = HashMap::from([("a".to_string(), fast()), ("b".to_string(), fast())]);
        let rtt = HashMap::from([("b".to_string(), 1.0)]);

        let short = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &req(64),
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(
            short.node, "a",
            "a 128-token request cannot pay for a model load"
        );

        let long = RequestShape {
            max_tokens: 100_000,
            ..req(64)
        };
        let got = choose(
            &"a".to_string(),
            &m,
            0,
            &rates,
            &rtt,
            &long,
            &HashMap::new(),
            &no_bar(),
            &none_sent(),
        )
        .unwrap();
        assert_eq!(
            got.node, "b",
            "over a long generation the load pays for itself"
        );
        assert!(got.needs_load);
    }
}
