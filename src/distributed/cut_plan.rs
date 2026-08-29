//! Where to cut a layer stack across nodes, priced by what each link can carry.
//!
//! The single-node planner packs layers onto cards until one is full and moves to the next. On
//! one host that is nearly right, because the boundaries all cost about the same. Across a
//! fabric it is wrong in a way nothing reports: a cut placed on a narrow link is paid on every
//! token for the life of the model, and the plan that made it looks exactly like the plan that
//! did not.
//!
//! A pipeline is a chain, so the optimum is computable rather than approachable. Each node
//! takes one contiguous run of layers; the cost of a plan is the sum over its boundaries of
//! the activation crossing them divided by what that link carries. Dynamic programming over
//! (node, layer) gives the exact minimum under the memory each node reports.
//!
//! Two properties this deliberately keeps:
//!
//! - Capacity is a HARD constraint, never a penalty. A plan that overflows a node is not a
//!   worse plan, it is a plan that fails at load, so it must not be representable in the
//!   output at all.
//! - No answer is better than a wrong one. When nothing fits, the planner says so instead of
//!   returning its least-bad arrangement, because a caller that receives a plan will run it.

use std::collections::HashMap;

/// A node the planner may use.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeCapacity {
    pub id: String,
    /// Bytes of weights it can hold.
    pub bytes: u64,
}

/// What crossing a boundary costs, per token, in milliseconds.
///
/// Keyed by ordered pair so an asymmetric fabric - a fast downlink and a slow uplink - is
/// expressible. Absent pairs are unreachable rather than free.
#[derive(Debug, Clone, Default)]
pub struct CutCosts {
    ms: HashMap<(String, String), f64>,
}

impl CutCosts {
    /// Derive the per-token cost of moving one activation between two nodes.
    pub fn from_link_gbps(pairs: &[(&str, &str, f64)], activation_bytes: u64) -> Self {
        let mut ms = HashMap::new();
        for (a, b, gbps) in pairs {
            let t = if *gbps <= 0.0 {
                f64::INFINITY
            } else {
                (activation_bytes as f64 / 1e9) / gbps * 1000.0
            };
            ms.insert((a.to_string(), b.to_string()), t);
            ms.insert((b.to_string(), a.to_string()), t);
        }
        Self { ms }
    }

    pub fn cost(&self, from: &str, to: &str) -> f64 {
        if from == to {
            return 0.0;
        }
        self.ms
            .get(&(from.to_string(), to.to_string()))
            .copied()
            .unwrap_or(f64::INFINITY)
    }
}

/// One node's share of the stack.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub node: String,
    /// Half-open layer range.
    pub start: usize,
    pub end: usize,
}

impl Segment {
    pub fn len(&self) -> usize {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }
}

/// A plan and what it is predicted to cost.
#[derive(Debug, Clone, PartialEq)]
pub struct CutPlan {
    pub segments: Vec<Segment>,
    /// Milliseconds per token spent crossing boundaries.
    pub crossing_ms: f64,
}

impl CutPlan {
    /// Every layer covered exactly once, in order, and no empty segment. The property a caller
    /// is entitled to assume before running it.
    pub fn covers(&self, layers: usize) -> bool {
        let mut next = 0;
        for s in &self.segments {
            if s.start != next || s.is_empty() {
                return false;
            }
            next = s.end;
        }
        next == layers
    }
}

/// Above this many nodes the exact search over orderings is abandoned for a fixed order.
/// Factorial growth: six is 720 orderings and instant, ten is three and a half million.
const EXACT_ORDER_LIMIT: usize = 6;

/// Cut `layer_bytes` across `nodes`, minimising what the boundaries cost per token.
///
/// Returns `None` when no arrangement fits - the caller must fall back (fewer layers, another
/// quantisation, a single node) rather than run something that will not load.
pub fn plan(layer_bytes: &[u64], nodes: &[NodeCapacity], costs: &CutCosts) -> Option<CutPlan> {
    if layer_bytes.is_empty() || nodes.is_empty() {
        return None;
    }
    let orders: Vec<Vec<usize>> = if nodes.len() <= EXACT_ORDER_LIMIT {
        permutations((0..nodes.len()).collect())
    } else {
        // Widest first: the biggest node absorbs the longest run, which is where a heuristic
        // is least likely to be badly wrong. Stated rather than silent - a caller at this
        // scale should know the answer is no longer exact.
        let mut idx: Vec<usize> = (0..nodes.len()).collect();
        idx.sort_by_key(|&i| std::cmp::Reverse(nodes[i].bytes));
        vec![idx]
    };

    let mut best: Option<CutPlan> = None;
    for order in orders {
        if let Some(p) = plan_in_order(layer_bytes, nodes, costs, &order) {
            best = match best {
                Some(b) if b.crossing_ms <= p.crossing_ms => Some(b),
                _ => Some(p),
            };
        }
    }
    best
}

/// Exact DP for one node ordering: `dp[j][i]` is the cheapest way to cover layers `0..i`
/// using the first `j` nodes of the order.
fn plan_in_order(
    layer_bytes: &[u64],
    nodes: &[NodeCapacity],
    costs: &CutCosts,
    order: &[usize],
) -> Option<CutPlan> {
    let l = layer_bytes.len();
    let n = order.len();
    let mut prefix = vec![0u64; l + 1];
    for i in 0..l {
        prefix[i + 1] = prefix[i] + layer_bytes[i];
    }

    // dp[j][i] = (cost, split point k) for layers 0..i on the first j nodes.
    let inf = f64::INFINITY;
    let mut dp = vec![vec![(inf, 0usize); l + 1]; n + 1];
    dp[0][0] = (0.0, 0);

    for j in 1..=n {
        let node = &nodes[order[j - 1]];
        for i in 0..=l {
            for k in 0..=i {
                let (prev, _) = dp[j - 1][k];
                if !prev.is_finite() {
                    continue;
                }
                // A node may take nothing, so a plan can use fewer nodes than are offered.
                if k < i && prefix[i] - prefix[k] > node.bytes {
                    continue;
                }
                let cut = if k == 0 || k == i {
                    0.0
                } else {
                    costs.cost(&nodes[order[j - 2]].id, &node.id)
                };
                if !cut.is_finite() {
                    continue;
                }
                let total = prev + cut;
                if total < dp[j][i].0 {
                    dp[j][i] = (total, k);
                }
            }
        }
    }

    let (cost, _) = dp[n][l];
    if !cost.is_finite() {
        return None;
    }

    // Walk the choices back into segments, dropping the nodes that took nothing.
    let mut segments = Vec::new();
    let mut i = l;
    for j in (1..=n).rev() {
        let (_, k) = dp[j][i];
        if k < i {
            segments.push(Segment {
                node: nodes[order[j - 1]].id.clone(),
                start: k,
                end: i,
            });
        }
        i = k;
    }
    segments.reverse();
    Some(CutPlan {
        segments,
        crossing_ms: cost,
    })
}

fn permutations(items: Vec<usize>) -> Vec<Vec<usize>> {
    if items.len() <= 1 {
        return vec![items];
    }
    let mut out = Vec::new();
    for i in 0..items.len() {
        let mut rest = items.clone();
        let head = rest.remove(i);
        for mut p in permutations(rest) {
            p.insert(0, head);
            out.push(p);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes(spec: &[(&str, u64)]) -> Vec<NodeCapacity> {
        spec.iter()
            .map(|(id, b)| NodeCapacity {
                id: id.to_string(),
                bytes: *b,
            })
            .collect()
    }

    /// One node with room takes everything, and a plan with no boundary costs nothing.
    #[test]
    fn a_stack_that_fits_one_node_is_not_cut() {
        let layers = vec![1_000u64; 10];
        let p = plan(&layers, &nodes(&[("a", 100_000)]), &CutCosts::default()).unwrap();
        assert_eq!(p.segments.len(), 1);
        assert_eq!(p.crossing_ms, 0.0);
        assert!(p.covers(10));
    }

    /// The point of the module: with three nodes in a line, the boundary lands on the fast
    /// link. A planner that packs greedily would cut wherever the first node filled up.
    #[test]
    fn the_boundary_lands_on_the_wide_link_not_the_narrow_one() {
        let layers = vec![1_000u64; 12];
        // Every node can hold at most eight layers, so exactly one cut is unavoidable.
        let ns = nodes(&[("a", 8_000), ("b", 8_000), ("c", 8_000)]);
        // a<->b is narrow, a<->c and b<->c are wide.
        let costs = CutCosts::from_link_gbps(
            &[("a", "b", 1.0), ("a", "c", 32.0), ("b", "c", 32.0)],
            8 * 1024,
        );
        let p = plan(&layers, &ns, &costs).unwrap();
        assert!(p.covers(12));
        let pair: Vec<&str> = p.segments.iter().map(|s| s.node.as_str()).collect();
        assert!(
            !(pair
                .windows(2)
                .any(|w| (w[0] == "a" && w[1] == "b") || (w[0] == "b" && w[1] == "a"))),
            "the plan crossed the narrow link: {pair:?}"
        );
    }

    /// The link matrix has to change the answer, not decorate it: the same cut over four
    /// lanes costs four times what it costs over sixteen.
    #[test]
    fn a_narrow_link_makes_the_same_cut_cost_more() {
        let layers = vec![1_000u64; 10];
        let ns = nodes(&[("a", 5_000), ("b", 5_000)]);
        let wide = CutCosts::from_link_gbps(&[("a", "b", 63.0)], 8 * 1024);
        let narrow = CutCosts::from_link_gbps(&[("a", "b", 15.75)], 8 * 1024);
        let pw = plan(&layers, &ns, &wide).unwrap();
        let pn = plan(&layers, &ns, &narrow).unwrap();
        assert!(pw.crossing_ms > 0.0);
        assert!(
            (pn.crossing_ms / pw.crossing_ms - 4.0).abs() < 1e-6,
            "wide {} narrow {}",
            pw.crossing_ms,
            pn.crossing_ms
        );
    }

    /// Capacity is a constraint, not a preference. No segment may exceed what its node holds.
    #[test]
    fn no_segment_exceeds_the_node_that_holds_it() {
        let layers = vec![1_000u64; 20];
        let ns = nodes(&[("a", 6_000), ("b", 9_000), ("c", 9_000)]);
        let costs = CutCosts::from_link_gbps(
            &[("a", "b", 8.0), ("a", "c", 8.0), ("b", "c", 8.0)],
            8 * 1024,
        );
        let p = plan(&layers, &ns, &costs).unwrap();
        assert!(p.covers(20));
        for s in &p.segments {
            let cap = ns.iter().find(|n| n.id == s.node).unwrap().bytes;
            let used: u64 = layers[s.start..s.end].iter().sum();
            assert!(
                used <= cap,
                "segment {s:?} uses {used} on a node holding {cap}"
            );
        }
    }

    /// When nothing fits, say so. Returning the least-bad arrangement would hand the caller a
    /// plan that fails at load, after it has already committed to running it.
    #[test]
    fn an_impossible_stack_returns_no_plan_rather_than_a_bad_one() {
        let layers = vec![10_000u64; 10]; // 100 000 bytes
        let ns = nodes(&[("a", 20_000), ("b", 20_000)]); // 40 000 between them
        let costs = CutCosts::from_link_gbps(&[("a", "b", 8.0)], 8 * 1024);
        assert!(plan(&layers, &ns, &costs).is_none());
    }

    /// A pair with no measured link is unreachable, not free - otherwise the cheapest plan is
    /// always the one crossing the link we know least about.
    #[test]
    fn an_unmeasured_link_is_not_a_free_one() {
        let layers = vec![1_000u64; 10];
        let ns = nodes(&[("a", 5_000), ("b", 5_000)]);
        // Nothing published about a<->b at all.
        assert!(plan(&layers, &ns, &CutCosts::default()).is_none());
    }

    /// Nodes that are not needed are left out, so a two-node plan does not carry a third
    /// empty segment that a consumer would have to filter.
    #[test]
    fn unused_nodes_do_not_appear_in_the_plan() {
        let layers = vec![1_000u64; 4];
        let ns = nodes(&[("a", 100_000), ("b", 100_000), ("c", 100_000)]);
        let costs = CutCosts::from_link_gbps(
            &[("a", "b", 8.0), ("a", "c", 8.0), ("b", "c", 8.0)],
            8 * 1024,
        );
        let p = plan(&layers, &ns, &costs).unwrap();
        assert_eq!(p.segments.len(), 1, "one node had room for all of it");
        assert!(p.segments.iter().all(|s| !s.is_empty()));
    }

    /// Layers are not uniform - an embedding or a head weighs more than a block - and the cut
    /// has to respect that rather than count layers.
    #[test]
    fn uneven_layers_are_cut_by_weight_not_by_count() {
        // A heavy head: five light layers then one that is worth ten of them.
        let layers = vec![1_000, 1_000, 1_000, 1_000, 1_000, 10_000];
        let ns = nodes(&[("a", 5_000), ("b", 11_000)]);
        let costs = CutCosts::from_link_gbps(&[("a", "b", 8.0)], 8 * 1024);
        let p = plan(&layers, &ns, &costs).unwrap();
        assert!(p.covers(6));
        let heavy = p.segments.iter().find(|s| s.end == 6).unwrap();
        assert_eq!(
            heavy.node, "b",
            "the heavy tail can only be on the larger node"
        );
    }
}
