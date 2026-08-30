//! The running cluster: what this node publishes, what it learns, and when it hands a request over.
//!
//! The pieces beside this file are mechanisms with invariants - a detector that reports
//! suspicion, arithmetic that ranks nodes, a planner that cuts a stack. None of them serves a
//! request. This is where they meet the server: a peer list, a gossip loop that keeps
//! `Membership` fed with something real, and the one decision that turns all of it into
//! behaviour - serve here, or forward whole and relay the stream back.
//!
//! Control traffic stays on HTTP/JSON, as the plan says. A forwarded request IS the request:
//! the same body to the same route on another node, which is why this can work before the
//! binary data plane carries a single activation.
//!
//! Two rules the wiring must not lose:
//!
//! - A request must never be forwarded twice. Without a marker, two nodes that each think the
//!   other is better bounce a request between them until something times out, and the symptom
//!   is a hang rather than an error.
//! - Distributing has to EARN it. `min_speedup` is why: moving a request costs a second
//!   failure domain and a relayed stream, so a peer that is merely equal keeps its hands off.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::membership::{Membership, NodeId, NodeState};
use super::routing::{choose, NodeRates, RequestShape};

/// Policy. The fabric is discovered, so only these are configured.
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Peers to contact on start. Empty means this node runs alone and every hook below is a
    /// no-op - the single-node path must stay exactly what it was.
    pub join: Vec<String>,
    /// How often to ask peers for their state.
    pub gossip_interval_ms: u64,
    /// Refuse to move a request below this predicted speedup over serving it here.
    pub min_speedup: f64,
    /// How this node is named to its peers.
    pub node_id: NodeId,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            join: Vec::new(),
            gossip_interval_ms: 1000,
            min_speedup: 1.15,
            node_id: String::new(),
        }
    }
}

impl ClusterConfig {
    /// A node with no peers is not a cluster, and every path below must behave as before.
    pub fn is_clustered(&self) -> bool {
        !self.join.is_empty()
    }
}

/// One peer as this node sees it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PeerView {
    pub node_id: NodeId,
    /// Where this node would send a hand-over. `None` for a peer heard of but never addressed.
    pub endpoint: Option<String>,
    pub alive: bool,
    /// Suspicion from the failure detector. `None` before enough gossip rounds to judge.
    pub phi: Option<f64>,
    /// Round trip measured on the gossip request itself, not on a separate ping: pricing a
    /// hand-over against a latency nobody pays would describe a different network.
    pub rtt_ms: Option<f64>,
    pub is_self: bool,
    pub state: NodeState,
}

/// What the router decided, and why - the reason belongs in the evidence report.
#[derive(Debug, Clone, PartialEq)]
pub enum Decision {
    /// Serve it here.
    Local { reason: String },
    /// Hand it to a peer and relay the answer.
    Forward {
        peer: NodeId,
        url: String,
        reason: String,
    },
}

/// Live cluster state for this node.
pub struct Cluster {
    pub config: ClusterConfig,
    members: Mutex<Membership>,
    /// Rates each peer publishes about itself, aggregated over whatever it last ran.
    rates: Mutex<HashMap<NodeId, NodeRates>>,
    /// The same, per model. Consulted first: a peer's speed on one model says nothing about
    /// its speed on another, and placing a 70B request on a 0.6B measurement is how a node
    /// wins a comparison it should lose.
    rates_by_model: Mutex<HashMap<NodeId, HashMap<String, NodeRates>>>,
    /// Round-trip time to each peer, measured by the gossip that just talked to it.
    rtt_ms: Mutex<HashMap<NodeId, f64>>,
    /// Where a peer can be reached.
    urls: Mutex<HashMap<NodeId, String>>,
    /// Hand-overs dispatched to each peer since its last report; that report clears them.
    sent_since_report: Mutex<HashMap<NodeId, u32>>,
    /// What this node publishes about itself.
    local: Mutex<NodeState>,
    /// Peers that just failed a hand-over, and the moment they may be considered again.
    ///
    /// The detector answers "is this node alive?", which a node returning errors quickly
    /// still is - it gossips, it publishes fine rates, and being fast at failing makes it
    /// the MOST attractive candidate. So a refusal is remembered separately from a silence.
    penalised: Mutex<HashMap<NodeId, u64>>,
}

impl Cluster {
    pub fn new(config: ClusterConfig) -> Arc<Self> {
        Arc::new(Self {
            members: Mutex::new(Membership::new(config.gossip_interval_ms as f64)),
            rates: Mutex::new(HashMap::new()),
            rates_by_model: Mutex::new(HashMap::new()),
            penalised: Mutex::new(HashMap::new()),
            rtt_ms: Mutex::new(HashMap::new()),
            urls: Mutex::new(HashMap::new()),
            sent_since_report: Mutex::new(HashMap::new()),
            local: Mutex::new(NodeState::default()),
            config,
        })
    }

    /// Refresh what this node advertises. Called when a model loads or unloads and as load moves.
    pub fn publish_local(&self, state: NodeState) {
        *self.local.lock().unwrap_or_else(|e| e.into_inner()) = state;
    }

    pub fn local_state(&self) -> NodeState {
        self.local.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// Record what a gossip round learned about a peer.
    pub fn observe_peer(
        &self,
        node: &NodeId,
        url: &str,
        now_ms: u64,
        state: NodeState,
        rates: NodeRates,
        per_model: HashMap<String, NodeRates>,
        rtt_ms: f64,
    ) {
        self.rates_by_model
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(node.clone(), per_model);
        self.members
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .observe(node, now_ms, state);
        self.sent_since_report
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(node);
        self.rates
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(node.clone(), rates);
        self.rtt_ms
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(node.clone(), rtt_ms);
        self.urls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(node.clone(), url.to_string());
    }

    /// A hand-over to this peer failed. Keep it out of the running for a while.
    ///
    /// Scaled to the gossip period rather than to a fixed duration: a cluster that talks every
    /// second and one that talks every ten describe "a while" differently, and the point is to
    /// give the peer time to be seen recovering, not to punish it for a number of milliseconds.
    pub fn note_handover_failed(&self, peer: &NodeId, now_ms: u64) {
        const ROUNDS_OUT: u64 = 10;
        let until = now_ms + self.config.gossip_interval_ms * ROUNDS_OUT;
        self.penalised
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(peer.clone(), until);
        tracing::warn!(
            "cluster: {peer} failed a hand-over - not a candidate until +{}ms",
            self.config.gossip_interval_ms * ROUNDS_OUT
        );
    }

    /// Peers currently worth routing to.
    pub fn alive(&self, now_ms: u64) -> Vec<NodeId> {
        self.members
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .alive(now_ms)
    }

    /// Serve here or hand over.
    ///
    /// `already_forwarded` is the loop breaker. A request that arrived from a peer is served
    /// where it landed, whatever the arithmetic now says: two nodes that each prefer the other
    /// would otherwise pass it back and forth until a timeout, and a hang is a far worse
    /// failure than a slightly suboptimal placement.
    /// What this node believes about every peer it knows: where it is, whether it is alive,
    /// how suspicious its silence looks, and what it last said about itself.
    ///
    /// A node publishes its own state on `/api/cluster/state`, which lets an observer see what
    /// it can reach. This is the other half: what each node can reach. A partition where two
    /// nodes each see the observer but not each other is invisible without it.
    pub fn peer_view(&self, now_ms: u64) -> Vec<PeerView> {
        let members = self.members.lock().unwrap_or_else(|e| e.into_inner());
        let urls = self.urls.lock().unwrap_or_else(|e| e.into_inner());
        let rtt = self.rtt_ms.lock().unwrap_or_else(|e| e.into_inner());
        let alive: std::collections::HashSet<NodeId> = members.alive(now_ms).into_iter().collect();
        // This node belongs in its own answer. It only enters the member table when a routing
        // decision puts it there, so a node that has served nothing would describe its peers
        // and omit itself - and an observer cannot tell that from a node that is not in the
        // cluster at all.
        let mut out: Vec<PeerView> = std::iter::once((
            &self.config.node_id,
            &self.local_state(),
        ))
        .filter(|(n, _)| members.state_of(n).is_none())
        .map(|(node, state)| PeerView {
            node_id: node.clone(),
            // The address peers were given lives in the runtime that announces it, not in
            // this table. An observer reaching this endpoint already knows where it is.
            endpoint: None,
            alive: true,
            phi: None,
            rtt_ms: Some(0.0),
            is_self: true,
            state: (*state).clone(),
        })
        .chain(members
            .known()
            .map(|(node, state)| PeerView {
                node_id: node.clone(),
                endpoint: urls.get(node).cloned(),
                alive: alive.contains(node),
                phi: members.phi(node, now_ms),
                rtt_ms: rtt.get(node).copied(),
                is_self: *node == self.config.node_id,
                state: state.clone(),
            }))
            .collect();
        out.sort_by(|a, b| a.node_id.cmp(&b.node_id));
        out
    }

    pub fn decide(
        &self,
        req: &RequestShape,
        now_ms: u64,
        already_forwarded: bool,
        // What each peer answered when asked how much of this prompt it already holds. Asked
        // rather than derived: only the node that owns the model can tokenise for it and look
        // in its own cache, so the question travels instead of a hashing convention.
        cached: &std::collections::HashMap<NodeId, u32>,
    ) -> Decision {
        // Every verdict says why, one line per request: a decision to stay home looks exactly
        // like a healthy cluster until it explains itself. At info, beside the hand-over it is
        // the counterpart of, so the default filter shows both halves of the choice.
        let verdict = self.decide_inner(req, now_ms, already_forwarded, cached);
        if let Decision::Local { reason } = &verdict {
            tracing::info!("cluster: serving locally: {reason}");
        }
        verdict
    }

    /// What each node is worth for one model: its rates for THAT model where it has measured
    /// them, its aggregate otherwise.
    ///
    /// The fallback is not a free pass: a node that has never run the model also reports
    /// `needs_load`, so it is priced for the fetch as well.
    fn rates_for(&self, model: &str) -> HashMap<NodeId, NodeRates> {
        let flat = self.rates.lock().unwrap_or_else(|e| e.into_inner());
        let per = self
            .rates_by_model
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        flat.iter()
            .map(|(node, agg)| {
                let r = per
                    .get(node)
                    .and_then(|m| m.get(model))
                    .copied()
                    .unwrap_or(*agg);
                (node.clone(), r)
            })
            .collect()
    }


    fn decide_inner(
        &self,
        req: &RequestShape,
        now_ms: u64,
        already_forwarded: bool,
        cached: &std::collections::HashMap<NodeId, u32>,
    ) -> Decision {
        if !self.config.is_clustered() {
            return Decision::Local {
                reason: "single node".into(),
            };
        }
        if already_forwarded {
            return Decision::Local {
                reason: "already forwarded once - a second hop would risk a loop".into(),
            };
        }

        // The local node has to be in the table to be compared against, and its own state is
        // known first-hand rather than through gossip.
        {
            let mut m = self.members.lock().unwrap_or_else(|e| e.into_inner());
            m.observe(&self.config.node_id, now_ms, self.local_state());
        }

        let members = self.members.lock().unwrap_or_else(|e| e.into_inner());
        // Rates for THIS model where a node has measured it, its aggregate otherwise. The
        // fallback is not a free pass: a node that has never run the model also reports
        // needs_load, so it is priced for the fetch as well.
        let rates = self.rates_for(&req.model);
        let rtt = self.rtt_ms.lock().unwrap_or_else(|e| e.into_inner());

        // A peer that just failed a hand-over is not a candidate until its penalty expires.
        // Excluded rather than ranked down: a node that fails FAST looks fast, so any score
        // it keeps argues for sending it more work.
        let barred: std::collections::HashSet<NodeId> = {
            let mut pen = self.penalised.lock().unwrap_or_else(|e| e.into_inner());
            pen.retain(|_, until| *until > now_ms);
            pen.keys().cloned().collect()
        };

        // The local node judges itself by its own meter, not by the pessimistic default: the
        // rates map is filled by gossip, and gossip only ever describes PEERS. Left out, this
        // node priced itself at the unknown-peer floor and handed over work it would have
        // finished eighty times faster - the exact mirror of the bug where it never handed
        // over at all.
        let mut rates = rates;
        if !rates.contains_key(&self.config.node_id) {
            let own = super::rate_meter::snapshot(&req.model);
            if own.decode_tok_per_s <= 0.0 {
                // Bootstrap: with neither a measurement nor a hardware prior this node
                // cannot compare itself to anyone - priced at the unknown floor it would
                // forward everything and never measure itself. The first request stays
                // home and becomes the meter.
                return Decision::Local {
                    reason: "no local price for this model yet".into(),
                };
            }
            {
                let dflt = super::routing::NodeRates::default();
                let or_unknown = |v: f64, fb: f64| if v > 0.0 { v } else { fb };
                rates.insert(
                    self.config.node_id.clone(),
                    super::routing::NodeRates {
                        prefill_tok_per_s: or_unknown(
                            own.prefill_tok_per_s,
                            dflt.prefill_tok_per_s,
                        ),
                        decode_tok_per_s: or_unknown(own.decode_tok_per_s, dflt.decode_tok_per_s),
                        model_load_s: or_unknown(own.model_load_s, dflt.model_load_s),
                        agg_tok_per_s: own.agg_tok_per_s,
                    },
                );
            }
        }

        let Some(best) = choose(
            &self.config.node_id,
            &members,
            now_ms,
            &rates,
            &rtt,
            req,
            cached,
            &barred,
            &self
                .sent_since_report
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
        ) else {
            return Decision::Local {
                reason: "no peer published anything usable".into(),
            };
        };
        // How long serving it here would take. Computed before the winner is known, because
        // it is what every other estimate is worth comparing against - including when the
        // winner IS this node, which is the case a breakdown emitted only on hand-over can
        // never show.
        let here = members
            .state_of(&self.config.node_id)
            .cloned()
            .unwrap_or_default();
        let local_rates = rates.get(&self.config.node_id).copied().unwrap_or_default();
        let local = super::routing::estimate(
            &self.config.node_id,
            &here,
            local_rates,
            req,
            0.0,
            cached.get(&self.config.node_id).copied().unwrap_or(0),
        );
        tracing::debug!(
            "cluster: best={} {:.0}ms vs here {:.0}ms | here {} | best {}",
            best.node,
            best.completion_ms,
            local.completion_ms,
            local.terms,
            best.terms
        );

        if best.node == self.config.node_id {
            return Decision::Local {
                reason: "this node is the best estimate".into(),
            };
        }

        let speedup = if best.completion_ms > 0.0 {
            local.completion_ms / best.completion_ms
        } else {
            f64::INFINITY
        };
        // The margin exists because moving a request costs more than the arithmetic models.
        // It cannot apply when the alternative is not serving the request at all: a node
        // without the weights on disk has no local option to compare against, however
        // favourable the numbers look.
        if speedup < self.config.min_speedup && super::routing::can_serve(&here, &req.model) {
            return Decision::Local {
                reason: format!(
                    "peer {} is only {speedup:.2}x faster, below the {:.2}x a hand-over has to earn",
                    best.node, self.config.min_speedup
                ),
            };
        }

        let url = self
            .urls
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&best.node)
            .cloned();
        if url.is_some() {
            *self
                .sent_since_report
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(best.node.clone())
                .or_insert(0) += 1;
        }
        match url {
            Some(url) => Decision::Forward {
                peer: best.node.clone(),
                url,
                // A node without the weights has no local time to be slower than, so quoting a
                // ratio against one would describe a race that was never run.
                reason: if super::routing::can_serve(&here, &req.model) {
                    format!(
                        "{speedup:.2}x faster there ({} prompt tokens already cached, {} ms \
                         predicted against {} ms here)",
                        best.prefix_hits,
                        best.completion_ms.round(),
                        local.completion_ms.round()
                    )
                } else {
                    format!(
                        "{} is not in this node's catalogue ({} prompt tokens already cached \
                         there, {} ms predicted)",
                        req.model,
                        best.prefix_hits,
                        best.completion_ms.round()
                    )
                },
            },
            // Known through gossip but not reachable: serving here is the honest fallback.
            None => Decision::Local {
                reason: format!("no address known for {}", best.node),
            },
        }
    }
}

/// Header marking a request that has already been handed over once - the loop guard.
pub const FORWARDED_HEADER: &str = "x-loken-forwarded";

/// The spelling used before the project was renamed. Still recognised on the way IN, so a
/// cluster whose nodes are upgraded one at a time does not lose its loop guard halfway
/// through: an old node forwarding to a new one would otherwise look like a fresh request
/// and could be handed straight back.
pub const FORWARDED_HEADER_LEGACY: &str = "x-llmuse-forwarded";

#[cfg(test)]
mod tests {
    use super::*;

    fn shape() -> RequestShape {
        RequestShape {
            model: "qwen3:8b".into(),
            prompt_tokens: 64,
            max_tokens: 128,
        }
    }

    /// Nobody answered - every node priced as holding none of the prompt.
    fn nothing_cached() -> std::collections::HashMap<NodeId, u32> {
        std::collections::HashMap::new()
    }

    fn cfg(peers: &[&str]) -> ClusterConfig {
        ClusterConfig {
            join: peers.iter().map(|s| s.to_string()).collect(),
            node_id: "self".into(),
            ..Default::default()
        }
    }

    fn busy() -> NodeState {
        NodeState {
            models: vec!["qwen3:8b".into()],
            load: 0.95,
            ..Default::default()
        }
    }
    fn free() -> NodeState {
        NodeState {
            models: vec!["qwen3:8b".into()],
            load: 0.0,
            ..Default::default()
        }
    }
    fn quick() -> NodeRates {
        NodeRates {
            prefill_tok_per_s: 2000.0,
            decode_tok_per_s: 80.0,
            model_load_s: 20.0,
            agg_tok_per_s: 0.0,
        }
    }

    /// A node with no peers must behave exactly as it did before any of this existed. The
    /// clustering code has to be invisible on the machine that is not clustered.
    #[test]
    fn a_node_with_no_peers_always_serves_locally() {
        let c = Cluster::new(ClusterConfig {
            node_id: "self".into(),
            ..Default::default()
        });
        c.publish_local(busy());
        assert!(matches!(
            c.decide(&shape(), 0, false, &nothing_cached()),
            Decision::Local { .. }
        ));
    }

    /// The hand-over: this node is saturated, a peer is free, and the request goes there.
    #[test]
    fn a_saturated_node_hands_the_request_to_a_free_peer() {
        let c = Cluster::new(cfg(&["http://b:11435"]));
        c.publish_local(busy());
        c.observe_peer(
            &"b".into(),
            "http://b:11435",
            0,
            free(),
            quick(),
            HashMap::new(),
            1.0,
        );
        {
            let mut r = c.rates.lock().unwrap();
            r.insert("self".into(), quick());
        }
        match c.decide(&shape(), 0, false, &nothing_cached()) {
            Decision::Forward { peer, url, .. } => {
                assert_eq!(peer, "b");
                assert_eq!(url, "http://b:11435");
            }
            d => panic!("expected a hand-over, got {d:?}"),
        }
    }

    /// The loop breaker. A request that already came from a peer is served where it landed,
    /// whatever the arithmetic says - two nodes each preferring the other would otherwise
    /// bounce it until a timeout, and the client sees a hang rather than an error.
    #[test]
    fn a_request_that_already_hopped_once_is_never_forwarded_again() {
        let c = Cluster::new(cfg(&["http://b:11435"]));
        c.publish_local(busy());
        c.observe_peer(
            &"b".into(),
            "http://b:11435",
            0,
            free(),
            quick(),
            HashMap::new(),
            1.0,
        );
        {
            let mut r = c.rates.lock().unwrap();
            r.insert("self".into(), quick());
        }
        // Same state that produced a hand-over above.
        assert!(matches!(
            c.decide(&shape(), 0, false, &nothing_cached()),
            Decision::Forward { .. }
        ));
        match c.decide(&shape(), 0, true, &nothing_cached()) {
            Decision::Local { reason } => assert!(reason.contains("already forwarded")),
            d => panic!("a second hop was allowed: {d:?}"),
        }
    }

    /// Distributing must earn it. A peer that is barely better keeps its hands off, because
    /// moving a request costs a second failure domain and a relayed stream that the
    /// arithmetic does not model.
    #[test]
    fn a_marginally_better_peer_does_not_take_the_request() {
        let c = Cluster::new(cfg(&["http://b:11435"]));
        // 0.05 against 0.0 is a real difference and a tiny one.
        c.publish_local(NodeState {
            load: 0.05,
            ..free()
        });
        c.observe_peer(
            &"b".into(),
            "http://b:11435",
            0,
            free(),
            quick(),
            HashMap::new(),
            1.0,
        );
        {
            let mut r = c.rates.lock().unwrap();
            r.insert("self".into(), quick());
        }
        match c.decide(&shape(), 0, false, &nothing_cached()) {
            Decision::Local { reason } => assert!(reason.contains("below the"), "got: {reason}"),
            d => panic!("a marginal gain moved the request: {d:?}"),
        }
    }

    /// A peer known through gossip but with no address must not silently drop the request.
    #[test]
    fn a_peer_without_an_address_falls_back_to_serving_here() {
        let c = Cluster::new(cfg(&["http://b:11435"]));
        c.publish_local(busy());
        {
            let mut m = c.members.lock().unwrap();
            m.observe(&"b".into(), 0, free());
            let mut r = c.rates.lock().unwrap();
            r.insert("b".into(), quick());
            r.insert("self".into(), quick());
        }
        match c.decide(&shape(), 0, false, &nothing_cached()) {
            Decision::Local { reason } => assert!(reason.contains("no address"), "got: {reason}"),
            d => panic!("forwarded to an unreachable peer: {d:?}"),
        }
    }

    /// A peer that stops gossiping leaves the table, and the request comes home rather than
    /// waiting for a node that will not answer.
    #[test]
    fn a_silent_peer_stops_receiving_requests() {
        let c = Cluster::new(cfg(&["http://b:11435"]));
        c.publish_local(busy());
        for t in (0..2000).step_by(1000) {
            c.observe_peer(
                &"b".into(),
                "http://b:11435",
                t,
                free(),
                quick(),
                HashMap::new(),
                1.0,
            );
        }
        {
            let mut r = c.rates.lock().unwrap();
            r.insert("self".into(), quick());
        }
        assert!(matches!(
            c.decide(&shape(), 2000, false, &nothing_cached()),
            Decision::Forward { .. }
        ));
        // Long silence: it must leave the routing table.
        assert!(matches!(
            c.decide(&shape(), 60_000, false, &nothing_cached()),
            Decision::Local { .. }
        ));
    }
}

impl Cluster {
    /// Ask every live peer how much of this prompt it already holds.
    ///
    /// Asked rather than derived. A node that does not own the model has no tokeniser for it,
    /// so it cannot compute the prompt's block hashes at all - and a shared hashing convention
    /// over raw text would describe a cache neither side actually keys on. Sending the question
    /// puts the answer where the knowledge is, and works whatever the architecture: the peer
    /// tokenises with its own tokeniser and looks in its own cache.
    ///
    /// Peers are asked in parallel and a slow one cannot hold the request: whatever has not
    /// answered by the deadline counts as holding nothing, which only ever costs an
    /// opportunity, never a wrong hand-over.
    pub async fn ask_peers_what_they_hold(
        &self,
        model: &str,
        prompt: &str,
        now_ms: u64,
    ) -> HashMap<NodeId, u32> {
        let mut out = HashMap::new();
        if !self.config.is_clustered() {
            return out;
        }
        let targets: Vec<(NodeId, String)> = {
            let urls = self.urls.lock().unwrap_or_else(|e| e.into_inner());
            let members = self.members.lock().unwrap_or_else(|e| e.into_inner());
            members
                .alive(now_ms)
                .into_iter()
                .filter(|n| n != &self.config.node_id)
                // Only nodes that hold the model can have cached anything of this prompt.
                .filter(|n| {
                    members
                        .state_of(n)
                        .is_some_and(|s| s.models.iter().any(|m| m == model))
                })
                .filter_map(|n| urls.get(&n).map(|u| (n, u.clone())))
                .collect()
        };
        if targets.is_empty() {
            return out;
        }

        let client = reqwest::Client::new();
        let asks = targets.into_iter().map(|(node, url)| {
            let client = client.clone();
            let model = model.to_string();
            let prompt = prompt.to_string();
            async move {
                let r = client
                    .post(format!("{}/api/cluster/prefix", url.trim_end_matches('/')))
                    .timeout(std::time::Duration::from_millis(250))
                    .json(&serde_json::json!({ "model": model, "prompt": prompt }))
                    .send()
                    .await
                    .ok()?;
                let v: serde_json::Value = r.json().await.ok()?;
                Some((node, v.get("cached_tokens")?.as_u64()? as u32))
            }
        });
        for answer in futures::future::join_all(asks).await.into_iter().flatten() {
            out.insert(answer.0, answer.1);
        }
        out
    }
}
