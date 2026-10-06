//! A node measuring its own cost profile for the cut planner, so the cluster tunes itself with no
//! operator input. Each node knows how fast its cores compute a missed expert, how fast its cards
//! run a layer, and how much VRAM it has; the planner reads these, never a flag.

use super::cut_plan::NodeProfile;
use crate::tensor::quantized::GgmlDType;

/// A reference point for turning a node's GPU throughput into a per-layer on-card cost: one layer's
/// measured cost at a known throughput. A node with fewer, slower cores reads as slower per layer,
/// which is what keeps a weak node from being handed work it cannot carry.
const LAYER_MS_AT_REF: f64 = 1.75;
/// The reference node's summed cores x clock (a 2-card desktop), the denominator of the scaling.
const REF_PERF: f64 = 41_925_120.0;

/// This node's cost profile, measured now: VRAM and GPU throughput from the device probe, the miss
/// cost from a kernel bench on the cores over the model's own expert dims and quant formats. The
/// per-layer cost is scaled from the reference by this node's throughput. No operator input.
///
/// `None` when the miss cost cannot be measured on this node's kernels: a node whose expert cost is
/// unknown stays out of the split rather than being planned with a guessed number. Safe to call on
/// the model-load path - it measures or declines, it never crashes.
pub fn local_profile(
    dim: usize,
    inter: usize,
    gate_dt: GgmlDType,
    down_dt: GgmlDType,
) -> Option<NodeProfile> {
    let cards = crate::inference::place::room::card_rooms();
    let vram: u64 = cards.iter().map(|c| c.available).sum();
    let perf: f64 = cards.iter().map(|c| c.perf as f64).sum();
    let layer_ms = if perf > 0.0 {
        LAYER_MS_AT_REF * REF_PERF / perf
    } else {
        LAYER_MS_AT_REF
    };
    let miss_ms =
        crate::inference::offload::cpu_experts::bench_miss_ms(dim, inter, gate_dt, down_dt, 200)?;
    Some(NodeProfile {
        vram,
        layer_ms,
        miss_ms,
    })
}

/// The distribution the cluster runs, decided from what the nodes measured about themselves and the
/// links between them - no operator input. Each `(id, profile)` is a node's self-measured cost; the
/// `rtt_ms` are the round-trips the cluster already tracks, the crossing cost between nodes. Returns
/// which node runs which layers, dropping any that would only slow the pipeline, or a single node
/// when no split pays. This is the N-node planner over the whole cluster, not a pair.
#[allow(clippy::too_many_arguments)]
pub fn auto_distribution(
    attn_per_layer: &[u64],
    expert_bytes: u64,
    working_set: u64,
    activated: u64,
    nodes: &[(String, NodeProfile)],
    rtt_ms: &[(&str, &str, f64)],
    is_clean: impl Fn(usize) -> bool,
) -> Option<super::cut_plan::TimePlan> {
    use super::cut_plan::{plan_by_time, CutCosts, TimedNode};
    let timed: Vec<TimedNode> = nodes
        .iter()
        .map(|(id, profile)| TimedNode {
            id: id.clone(),
            profile: *profile,
        })
        .collect();
    let costs = CutCosts::from_rtt_ms(rtt_ms);
    plan_by_time(
        attn_per_layer,
        expert_bytes,
        working_set,
        activated,
        &timed,
        &costs,
        is_clean,
    )
}

/// The distribution of a dense stack, decided from what the nodes measured about themselves:
/// the time of a layer on each node's cards and host, the room its cards leave for weights, and
/// the round-trips the cluster tracks between nodes. `first` is the node that holds the request
/// and starts the stack. `layer_bytes` is each layer's weights with its cache. Returns which node
/// runs which layers, or a single node when no split pays.
pub fn auto_dense_distribution(
    layer_bytes: &[u64],
    head_bytes: u64,
    nodes: &[(String, NodeProfile)],
    first: usize,
    rtt_ms: &[(&str, &str, f64)],
) -> Option<super::cut_plan::TimePlan> {
    use super::cut_plan::{plan_dense_by_time, CutCosts, TimedNode};
    let timed: Vec<TimedNode> = nodes
        .iter()
        .map(|(id, profile)| TimedNode {
            id: id.clone(),
            profile: *profile,
        })
        .collect();
    plan_dense_by_time(
        layer_bytes,
        head_bytes,
        &timed,
        first,
        &CutCosts::from_rtt_ms(rtt_ms),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The profile is self-measured and usable: a real miss cost, a finite layer cost. Without cards
    /// (a CPU host) the VRAM is zero and the layer cost falls back to the reference - the node then
    /// holds nothing resident, which the planner reads correctly.
    #[test]
    fn a_node_profiles_itself() {
        let p = local_profile(5120, 2304, GgmlDType::Iq2Xxs, GgmlDType::Q2K)
            .expect("real expert dims measure on the cores");
        assert!(p.miss_ms > 0.0, "the miss cost is measured on the cores");
        assert!(p.layer_ms.is_finite() && p.layer_ms > 0.0);
    }

    /// The whole cluster decides from measured profiles and round-trips: a fast pair keeps a slow
    /// third node out, and every layer is still covered.
    #[test]
    fn the_cluster_decides_from_measured_profiles() {
        let gb = 1u64 << 30;
        let attn = vec![100u64 << 20; 40];
        let fast = NodeProfile {
            vram: 32 * gb,
            layer_ms: 1.0,
            miss_ms: 0.5,
        };
        let slow = NodeProfile {
            vram: 8 * gb,
            layer_ms: 100.0,
            miss_ms: 50.0,
        };
        let nodes = vec![
            ("a".to_string(), fast),
            ("b".to_string(), fast),
            ("slow".to_string(), slow),
        ];
        let rtt = [("a", "b", 0.2), ("a", "slow", 0.2), ("b", "slow", 0.2)];
        let plan = auto_distribution(&attn, 10 << 20, 180, 6, &nodes, &rtt, |_| true).unwrap();
        assert!(
            !plan.segments.iter().any(|s| s.node == "slow"),
            "a node slow at everything is left out: {:?}",
            plan.segments
        );
        let mut next = 0;
        for s in &plan.segments {
            assert_eq!(s.start, next);
            next = s.end;
        }
        assert_eq!(next, 40, "every layer covered");
    }
}
