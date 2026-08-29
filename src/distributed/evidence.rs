//! What a cluster run has to be able to state afterwards.
//!
//! Two things are missing from every distributed-inference benchmark I can find, and they are
//! the two this project already answers on one machine.
//!
//! The first is energy. A per-node figure is not the answer: a request that touched three nodes
//! cost what all three spent on it, including the node that only forwarded. Reporting the
//! serving node's joules would make a cluster look more efficient the more it distributes,
//! which is exactly backwards - the coordination is real and somebody paid for it.
//!
//! The second is what was measured. A number without its topology is not reproducible: the
//! same request on the same hardware costs differently depending on where the weights were and
//! which link the pipeline crossed, so a report that omits the arrangement cannot be compared
//! to anything, including itself a week later.
//!
//! And a chaos run needs more than a pass mark. Killing a node produces four numbers that
//! matter separately - the latency the survivors saw, the requests that failed outright, the
//! tokens already streamed to a client that then had to be replayed, and how long the cluster
//! took to agree on who was left.

use std::collections::BTreeMap;

/// What one node spent serving part of a request.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeCost {
    pub node: String,
    pub joules: f64,
    /// Tokens this node itself produced. A forwarder produces none and still costs.
    pub tokens: u32,
    pub role: NodeRole,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRole {
    /// Took the request and answered it.
    Served,
    /// Passed it on and relayed the stream back.
    Forwarded,
    /// Ran part of a pipeline.
    Stage,
}

/// Everything one request cost, wherever it went.
#[derive(Debug, Clone, PartialEq)]
pub struct RequestEvidence {
    pub request_id: String,
    pub costs: Vec<NodeCost>,
    /// The arrangement this was measured on. Without it the numbers cannot be compared.
    pub topology: String,
    /// Why the router chose that arrangement.
    pub reason: String,
}

impl RequestEvidence {
    /// Joules for the request, summed over every node that touched it - forwarders included.
    pub fn total_joules(&self) -> f64 {
        self.costs.iter().map(|c| c.joules).sum()
    }

    pub fn total_tokens(&self) -> u32 {
        self.costs.iter().map(|c| c.tokens).sum()
    }

    /// The number that survives comparison across topologies.
    pub fn joules_per_token(&self) -> Option<f64> {
        let t = self.total_tokens();
        (t > 0).then(|| self.total_joules() / f64::from(t))
    }

    /// What coordination cost: the share spent by nodes that produced no tokens.
    ///
    /// Reported separately because it is the honest price of distributing. A cluster that
    /// hides it looks more efficient the more it spreads work around.
    pub fn coordination_joules(&self) -> f64 {
        self.costs
            .iter()
            .filter(|c| c.tokens == 0)
            .map(|c| c.joules)
            .sum()
    }

    /// Refuse to publish a figure that cannot be interpreted.
    pub fn publishable(&self) -> Result<(), String> {
        if self.costs.is_empty() {
            return Err("no node reported: the request cost something and nobody said what".into());
        }
        if self.topology.trim().is_empty() {
            return Err("no topology: a rate without its arrangement is not reproducible".into());
        }
        if self.total_tokens() == 0 {
            return Err("no tokens produced: there is no rate to publish".into());
        }
        Ok(())
    }
}

/// What killing a node during a run produced.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChaosReport {
    pub killed: String,
    /// Latency the surviving requests saw, milliseconds.
    pub p99_ms: f64,
    /// Requests that ended in an error rather than a pause.
    pub failed: u32,
    pub total: u32,
    /// Tokens already streamed to a client that had to be produced again on another node.
    pub tokens_replayed: u32,
    /// How long until every survivor agreed on the membership.
    pub reconvergence_ms: u64,
    pub topology: String,
}

impl ChaosReport {
    pub fn error_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        f64::from(self.failed) / f64::from(self.total)
    }

    /// A run is only worth publishing if it says what it broke and what that cost. A pass mark
    /// alone hides the difference between a cluster that paused and one that lost work.
    pub fn publishable(&self) -> Result<(), String> {
        if self.killed.trim().is_empty() {
            return Err("a chaos run that does not say what it killed measured nothing".into());
        }
        if self.total == 0 {
            return Err("no requests in flight: killing an idle node proves nothing".into());
        }
        if self.topology.trim().is_empty() {
            return Err("no topology: the result cannot be compared to another run".into());
        }
        Ok(())
    }
}

/// Rows for the report, one line per request, aggregated by topology so two arrangements can
/// be set against each other rather than averaged into one meaningless figure.
pub fn by_topology(runs: &[RequestEvidence]) -> BTreeMap<String, (f64, u32)> {
    let mut out: BTreeMap<String, (f64, u32)> = BTreeMap::new();
    for r in runs {
        let e = out.entry(r.topology.clone()).or_insert((0.0, 0));
        e.0 += r.total_joules();
        e.1 += r.total_tokens();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cost(node: &str, joules: f64, tokens: u32, role: NodeRole) -> NodeCost {
        NodeCost {
            node: node.into(),
            joules,
            tokens,
            role,
        }
    }

    /// The figure that matters: a forwarding node costs joules and produces no tokens, and
    /// both facts have to reach the total.
    #[test]
    fn a_forwarder_costs_joules_and_produces_no_tokens() {
        let e = RequestEvidence {
            request_id: "r1".into(),
            costs: vec![
                cost("edge", 3.0, 0, NodeRole::Forwarded),
                cost("gpu-a", 40.0, 128, NodeRole::Served),
            ],
            topology: "2 nodes, request offloaded".into(),
            reason: "local node holds no weights".into(),
        };
        assert_eq!(e.total_joules(), 43.0);
        assert_eq!(e.total_tokens(), 128);
        assert_eq!(
            e.coordination_joules(),
            3.0,
            "the forwarder's share is not free"
        );
        assert!((e.joules_per_token().unwrap() - 43.0 / 128.0).abs() < 1e-12);
    }

    /// Reporting only the serving node would make distributing look free. It is not.
    #[test]
    fn the_serving_nodes_figure_alone_understates_what_the_request_cost() {
        let e = RequestEvidence {
            request_id: "r2".into(),
            costs: vec![
                cost("edge", 5.0, 0, NodeRole::Forwarded),
                cost("stage-1", 30.0, 0, NodeRole::Stage),
                cost("stage-2", 30.0, 64, NodeRole::Served),
            ],
            topology: "3 nodes, pipeline of 2 stages".into(),
            reason: "no single node holds the model".into(),
        };
        let served_only = 30.0;
        assert!(
            e.total_joules() > served_only * 2.0,
            "coordination is most of this"
        );
        assert_eq!(e.coordination_joules(), 35.0);
    }

    /// A number without its arrangement is not reproducible, so it is refused rather than
    /// published with a caveat nobody reads.
    #[test]
    fn a_figure_without_its_topology_is_refused() {
        let mut e = RequestEvidence {
            request_id: "r3".into(),
            costs: vec![cost("a", 1.0, 8, NodeRole::Served)],
            topology: "  ".into(),
            reason: String::new(),
        };
        assert!(e.publishable().unwrap_err().contains("topology"));
        e.topology = "single node".into();
        assert!(e.publishable().is_ok());
    }

    /// And a request that produced nothing has no rate, whatever it spent - the same rule the
    /// single-machine bench already applies when an answer is degenerate.
    #[test]
    fn a_request_that_produced_no_tokens_has_no_rate_to_publish() {
        let e = RequestEvidence {
            request_id: "r4".into(),
            costs: vec![cost("a", 12.0, 0, NodeRole::Served)],
            topology: "single node".into(),
            reason: String::new(),
        };
        assert!(e.joules_per_token().is_none());
        assert!(e.publishable().unwrap_err().contains("no tokens"));
    }

    /// A chaos run has to distinguish a cluster that paused from one that lost work, so the
    /// replayed tokens are a separate number from the failures.
    #[test]
    fn a_chaos_run_separates_a_pause_from_a_loss() {
        let paused = ChaosReport {
            killed: "gpu-b".into(),
            p99_ms: 900.0,
            failed: 0,
            total: 50,
            tokens_replayed: 320,
            reconvergence_ms: 1200,
            topology: "3 nodes".into(),
        };
        let lost = ChaosReport {
            failed: 7,
            tokens_replayed: 0,
            ..paused.clone()
        };
        assert_eq!(paused.error_rate(), 0.0);
        assert!(paused.tokens_replayed > 0, "work was redone, not lost");
        assert!(lost.error_rate() > 0.0);
        assert!(paused.publishable().is_ok());
    }

    /// Killing an idle node proves nothing and must not read as a pass.
    #[test]
    fn a_chaos_run_with_nothing_in_flight_is_refused() {
        let idle = ChaosReport {
            killed: "gpu-b".into(),
            total: 0,
            topology: "3 nodes".into(),
            ..Default::default()
        };
        assert!(idle.publishable().unwrap_err().contains("idle"));
    }

    /// Topologies are compared, never averaged: one number over two arrangements describes
    /// neither.
    #[test]
    fn topologies_are_kept_apart_rather_than_averaged() {
        let runs = vec![
            RequestEvidence {
                request_id: "a".into(),
                costs: vec![cost("x", 10.0, 100, NodeRole::Served)],
                topology: "single node".into(),
                reason: String::new(),
            },
            RequestEvidence {
                request_id: "b".into(),
                costs: vec![
                    cost("x", 8.0, 50, NodeRole::Stage),
                    cost("y", 9.0, 50, NodeRole::Served),
                ],
                topology: "pipeline of 2".into(),
                reason: String::new(),
            },
        ];
        let agg = by_topology(&runs);
        assert_eq!(agg.len(), 2);
        assert_eq!(agg["single node"], (10.0, 100));
        assert_eq!(agg["pipeline of 2"], (17.0, 100));
    }
}
