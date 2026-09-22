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
    /// A fixed cost added to every crossing beyond its transfer time: the link's latency and the
    /// receiving node's per-token setup, paid on every hop whatever the boundary's size.
    base_ms: f64,
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
        Self { ms, base_ms: 0.0 }
    }

    /// Crossings priced straight from measured round-trip times between nodes, in milliseconds -
    /// what the cluster already measures, so the planner needs no bandwidth guess. Absent pairs are
    /// unreachable.
    pub fn from_rtt_ms(pairs: &[(&str, &str, f64)]) -> Self {
        let mut ms = HashMap::new();
        for (a, b, t) in pairs {
            ms.insert((a.to_string(), b.to_string()), *t);
            ms.insert((b.to_string(), a.to_string()), *t);
        }
        Self { ms, base_ms: 0.0 }
    }

    /// Add a fixed per-hop latency to every reachable crossing - the RTT and setup a boundary pays
    /// regardless of its size, which is what makes a many-hop plan cost more than the bytes alone.
    pub fn with_base_ms(mut self, base_ms: f64) -> Self {
        self.base_ms = base_ms;
        self
    }

    pub fn cost(&self, from: &str, to: &str) -> f64 {
        if from == to {
            return 0.0;
        }
        match self.ms.get(&(from.to_string(), to.to_string())) {
            Some(t) => self.base_ms + t,
            None => f64::INFINITY,
        }
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

/// A residency-first cut for the streaming case, where the weights do not all fit resident and
/// `plan` above would refuse every arrangement.
///
/// When a stage streams its weights, a node's memory no longer bounds which layers it may run - it
/// bounds how many of those layers' routed experts stay resident rather than missing to CPU or
/// disk. Fewer layers on a node leaves more room for its experts. So the cut is chosen to maximise
/// the routed experts held resident across both nodes: for each candidate boundary, the room each
/// node has left after its layers' always-resident weights is filled with experts, capped at the
/// working set a layer actually routes, and the boundary with the most resident experts wins.
///
/// `resident_per_layer` is the always-read bytes of each layer (attention, shared expert, gate) -
/// not the routed experts, which stream. `node_vram` is `[head, tail]`. `is_clean` says whether a
/// boundary carries no open band state, the only place a pipeline may cut. Returns the head's layer
/// count (the tail runs the rest), or `None` when no clean boundary leaves either node any room.
pub fn residency_cut(
    resident_per_layer: &[u64],
    expert_bytes: u64,
    working_set: u64,
    node_vram: [u64; 2],
    is_clean: impl Fn(usize) -> bool,
) -> Option<usize> {
    let n = resident_per_layer.len();
    if n == 0 || expert_bytes == 0 {
        return None;
    }
    // Experts a contiguous run keeps resident: the room left after its attention, filled with
    // experts, but never more than the run actually routes over its layers.
    let resident_experts = |layers: &[u64], vram: u64| -> u64 {
        if layers.is_empty() {
            return 0;
        }
        let attn: u64 = layers.iter().sum();
        let room = vram.saturating_sub(attn);
        let held = room / expert_bytes;
        held.min(working_set * layers.len() as u64)
    };
    let mut best: Option<(u64, usize)> = None;
    for cut in 1..n {
        if !is_clean(cut) {
            continue;
        }
        let held = resident_experts(&resident_per_layer[..cut], node_vram[0])
            + resident_experts(&resident_per_layer[cut..], node_vram[1]);
        // Most resident experts; on a tie, the larger head cut, so the tail - the smaller, slower
        // node - runs the fewest layers of the two arrangements the pipeline cannot tell apart.
        let key = (held, cut);
        if best.map(|b| key > b).unwrap_or(true) {
            best = Some(key);
        }
    }
    best.map(|(_, cut)| cut)
}

/// A node's per-token decode costs, in milliseconds, for the time model `best_cut` minimises.
/// A node is defined by what it can hold and how fast it computes what it cannot.
#[derive(Clone, Copy, Debug)]
pub struct NodeProfile {
    /// Bytes of weights the node keeps resident on its cards.
    pub vram: u64,
    /// Time for one layer's always-read path and the experts the node holds resident: the on-card
    /// work, flat in the number of misses.
    pub layer_ms: f64,
    /// Time to compute one routed expert the node does NOT hold resident, on its cores. The slow
    /// path; a weak node's high `miss_ms` is exactly what a residency-count cut ignores and this
    /// model does not.
    pub miss_ms: f64,
}

/// How to place a stack on two nodes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CutChoice {
    /// Run every layer on one node - `0` (head) or `1` (tail) - because splitting was not worth its
    /// crossing on this pair. The answer a residency-count cut can never give.
    Whole(usize),
    /// Head runs `[0, cut)`, tail runs `[cut, n)`.
    Split(usize),
}

/// The routed experts a run of `attn` layers holds resident on `vram`: the room left after the
/// layers' always-read bytes, filled with experts, capped at what the layers actually route.
fn resident_held(attn: &[u64], vram: u64, expert_bytes: u64, working_set: u64) -> u64 {
    if attn.is_empty() || expert_bytes == 0 {
        return 0;
    }
    let need: u64 = attn.iter().sum();
    let room = vram.saturating_sub(need);
    (room / expert_bytes).min(working_set * attn.len() as u64)
}

/// Predicted milliseconds per token for a node running the layers whose always-read bytes are
/// `attn`. Each token routes `activated` experts per layer; the fraction its resident set covers
/// run on the cards (folded into `layer_ms`), the rest miss to the cores at `miss_ms` each. Coverage
/// is the resident share of the working set - a floor, since routing is concentrated and the hottest
/// experts a card holds cover more than their count. Additive rather than `max(card, core)` on
/// purpose: it never rewards a cut for overlap it may not get, so the choice it makes is safe.
fn node_ms(
    attn: &[u64],
    p: &NodeProfile,
    expert_bytes: u64,
    working_set: u64,
    activated: u64,
) -> f64 {
    let layers = attn.len();
    if layers == 0 || working_set == 0 {
        return 0.0;
    }
    let resident = resident_held(attn, p.vram, expert_bytes, working_set);
    let resident_per_layer = resident as f64 / layers as f64;
    let coverage = (resident_per_layer / working_set as f64).min(1.0);
    let misses_per_token = activated as f64 * layers as f64 * (1.0 - coverage);
    layers as f64 * p.layer_ms + misses_per_token * p.miss_ms
}

/// The fastest way to place `n` layers on two nodes: split at a clean boundary, or run the whole
/// stack on one node when splitting is predicted slower. Minimises predicted per-token latency -
/// each node's compute plus, for a split, the one boundary crossing - under each node's own costs.
///
/// This is what a residency-count cut gets wrong. A miss on a slow node costs its `miss_ms`, not a
/// unit, so the model hands a weak node fewer layers, or none at all, rather than the arrangement
/// that merely holds the most experts somewhere. When no split beats a single node, it says so.
///
/// `attn_per_layer` is each layer's always-read bytes; `expert_bytes` one routed expert's; the
/// `working_set` the experts a layer routes over the calibration; `activated` how many a token
/// routes per layer. `is_clean(cut)` gates the boundaries a pipeline may cut. `crossing_ms` is the
/// per-token cost of one boundary on the link.
#[allow(clippy::too_many_arguments)]
pub fn best_cut(
    attn_per_layer: &[u64],
    expert_bytes: u64,
    working_set: u64,
    activated: u64,
    profiles: [NodeProfile; 2],
    crossing_ms: f64,
    is_clean: impl Fn(usize) -> bool,
) -> Option<CutChoice> {
    let n = attn_per_layer.len();
    if n == 0 || expert_bytes == 0 {
        return None;
    }
    let mut best: Option<(f64, CutChoice)> = None;
    let mut consider = |ms: f64, choice: CutChoice, best: &mut Option<(f64, CutChoice)>| {
        if best.map(|(b, _)| ms < b).unwrap_or(true) {
            *best = Some((ms, choice));
        }
    };
    // The whole stack on one node, no crossing: the baseline a split must beat.
    for node in 0..2 {
        let ms = node_ms(
            attn_per_layer,
            &profiles[node],
            expert_bytes,
            working_set,
            activated,
        );
        consider(ms, CutChoice::Whole(node), &mut best);
    }
    // Every clean split, its two nodes' compute plus the one crossing.
    for cut in 1..n {
        if !is_clean(cut) {
            continue;
        }
        let head = node_ms(
            &attn_per_layer[..cut],
            &profiles[0],
            expert_bytes,
            working_set,
            activated,
        );
        let tail = node_ms(
            &attn_per_layer[cut..],
            &profiles[1],
            expert_bytes,
            working_set,
            activated,
        );
        consider(head + tail + crossing_ms, CutChoice::Split(cut), &mut best);
    }
    best.map(|(_, c)| c)
}

/// A node with an id, for pricing the links between nodes, and its decode-cost profile.
#[derive(Clone, Debug)]
pub struct TimedNode {
    pub id: String,
    pub profile: NodeProfile,
}

/// A time-model placement across many nodes: which node runs which layers, and the predicted
/// per-token milliseconds. Nodes given no layers are absent from `segments` - dropped from the
/// distribution.
#[derive(Debug, Clone, PartialEq)]
pub struct TimePlan {
    pub segments: Vec<Segment>,
    pub total_ms: f64,
}

/// Place `attn_per_layer` layers across `nodes` to minimise predicted per-token time: each used
/// node's compute - its layers plus its misses at its own cost - and each boundary's crossing on
/// the link. The N-node generalisation of `best_cut`: it chooses the subset of nodes, their order,
/// and the cuts together, and a node that would only slow the pipeline is left out.
#[allow(clippy::too_many_arguments)]
pub fn plan_by_time(
    attn_per_layer: &[u64],
    expert_bytes: u64,
    working_set: u64,
    activated: u64,
    nodes: &[TimedNode],
    costs: &CutCosts,
    is_clean: impl Fn(usize) -> bool,
) -> Option<TimePlan> {
    if attn_per_layer.is_empty() || expert_bytes == 0 || nodes.is_empty() {
        return None;
    }
    let orders: Vec<Vec<usize>> = if nodes.len() <= EXACT_ORDER_LIMIT {
        permutations((0..nodes.len()).collect())
    } else {
        // Widest first once the exact search is too costly - the biggest node anchors the plan.
        let mut idx: Vec<usize> = (0..nodes.len()).collect();
        idx.sort_by_key(|&i| std::cmp::Reverse(nodes[i].profile.vram));
        vec![idx]
    };
    let mut best: Option<TimePlan> = None;
    for order in &orders {
        if let Some(p) = plan_time_in_order(
            attn_per_layer,
            expert_bytes,
            working_set,
            activated,
            nodes,
            costs,
            order,
            &is_clean,
        ) {
            if best
                .as_ref()
                .map(|b| p.total_ms < b.total_ms)
                .unwrap_or(true)
            {
                best = Some(p);
            }
        }
    }
    best
}

/// Exact DP for one node ordering: `dp[j][i]` is the least predicted time to cover layers `0..i`
/// with the first `j` nodes of the order, each taking a NON-EMPTY contiguous run. The answer is the
/// cheapest `dp[j][l]` over `j`, so using fewer nodes - dropping the rest - is one of the options.
#[allow(clippy::too_many_arguments)]
fn plan_time_in_order(
    attn: &[u64],
    eb: u64,
    ws: u64,
    activated: u64,
    nodes: &[TimedNode],
    costs: &CutCosts,
    order: &[usize],
    is_clean: &impl Fn(usize) -> bool,
) -> Option<TimePlan> {
    let l = attn.len();
    let m = order.len();
    let inf = f64::INFINITY;
    let mut dp = vec![vec![inf; l + 1]; m + 1];
    let mut back = vec![vec![0usize; l + 1]; m + 1];
    dp[0][0] = 0.0;
    for j in 1..=m {
        let node = &nodes[order[j - 1]];
        for i in 1..=l {
            for k in 0..i {
                let prev = dp[j - 1][k];
                if !prev.is_finite() {
                    continue;
                }
                // An interior boundary must be clean; the ends (0 and l) are not crossings.
                if k > 0 && !is_clean(k) {
                    continue;
                }
                let cross = if j == 1 {
                    0.0
                } else {
                    costs.cost(&nodes[order[j - 2]].id, &node.id)
                };
                if !cross.is_finite() {
                    continue;
                }
                let seg = node_ms(&attn[k..i], &node.profile, eb, ws, activated);
                let total = prev + cross + seg;
                if total < dp[j][i] {
                    dp[j][i] = total;
                    back[j][i] = k;
                }
            }
        }
    }
    let mut best_j = 0;
    let mut best_ms = inf;
    for j in 1..=m {
        if dp[j][l] < best_ms {
            best_ms = dp[j][l];
            best_j = j;
        }
    }
    if !best_ms.is_finite() {
        return None;
    }
    let mut segments = Vec::new();
    let mut i = l;
    for j in (1..=best_j).rev() {
        let k = back[j][i];
        segments.push(Segment {
            node: nodes[order[j - 1]].id.clone(),
            start: k,
            end: i,
        });
        i = k;
    }
    segments.reverse();
    Some(TimePlan {
        segments,
        total_ms: best_ms,
    })
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

    /// The streaming cut gives the smaller node fewer layers: with less room for attention it
    /// keeps fewer experts resident per layer, so it should run fewer of them.
    #[test]
    fn the_streaming_cut_hands_the_smaller_node_fewer_layers() {
        // Eight layers, one gibibyte of attention each, one gibibyte per expert, a layer routes two.
        let g = 1u64 << 30;
        let resident = vec![g; 8];
        let cut = residency_cut(&resident, g, 2, [10 * g, 4 * g], |_| true).unwrap();
        // The head (10 GiB) takes the bulk; the tail (4 GiB) runs fewer than half.
        assert!(
            cut >= 4,
            "the larger node runs at least half the layers, got head={cut}"
        );
        assert!(
            8 - cut < cut,
            "the smaller node runs fewer layers than the larger"
        );
    }

    /// A boundary inside an open band is not offered: the cut lands only where the crossing is
    /// clean, whatever the residency arithmetic would prefer.
    #[test]
    fn the_streaming_cut_only_lands_on_a_clean_boundary() {
        let g = 1u64 << 30;
        let resident = vec![g; 8];
        // Only layer 6 is a clean boundary.
        let cut = residency_cut(&resident, g, 4, [8 * g, 8 * g], |c| c == 6).unwrap();
        assert_eq!(cut, 6, "the only clean boundary is the only cut");
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

#[cfg(test)]
mod time_model_tests {
    use super::*;

    // 40 layers, each 100 MB always-read (attention fits with room to spare); an expert is 10 MB; a
    // layer routes 180 experts over the calibration.
    fn stack() -> Vec<u64> {
        vec![100 << 20; 40]
    }
    const EB: u64 = 10 << 20;
    const WS: u64 = 180;
    const GB: u64 = 1 << 30;
    const ACT: u64 = 6;

    /// Two equal nodes beat one: two cards hold more experts resident than one, so total misses
    /// fall. For a single stream the exact split is indifferent (total work is the same wherever it
    /// lands), so the test asserts only that splitting is chosen over running whole.
    #[test]
    fn two_nodes_beat_one_by_holding_more_resident() {
        let p = NodeProfile {
            vram: 16 * GB,
            layer_ms: 1.0,
            miss_ms: 0.5,
        };
        let c = best_cut(&stack(), EB, WS, ACT, [p, p], 1.0, |_| true).unwrap();
        assert!(
            matches!(c, CutChoice::Split(_)),
            "two cards hold more experts than one, so a split should win, got {c:?}"
        );
    }

    /// A node slow at everything - its on-card path a hundred times the head's - earns nothing:
    /// even one layer on it costs more than it saves, so the stack runs whole on the head and the
    /// slow node is eliminated from the distribution.
    #[test]
    fn a_node_slow_at_everything_is_eliminated() {
        let head = NodeProfile {
            vram: 32 * GB,
            layer_ms: 1.0,
            miss_ms: 0.5,
        };
        let slow = NodeProfile {
            vram: 8 * GB,
            layer_ms: 100.0,
            miss_ms: 20.0,
        };
        let c = best_cut(&stack(), EB, WS, ACT, [head, slow], 1.0, |_| true).unwrap();
        assert_eq!(
            c,
            CutChoice::Whole(0),
            "a node slow at everything must be dropped, not given layers"
        );
    }

    /// A weak node with room, not slow on-card, still profits from a SMALL fully-resident share it
    /// takes without a single miss - so it is kept, given few layers, not dropped. The smart middle
    /// ground a residency-count cut and a naive drop both miss.
    #[test]
    fn a_small_but_useful_node_is_kept_with_a_small_share() {
        let head = NodeProfile {
            vram: 32 * GB,
            layer_ms: 1.0,
            miss_ms: 0.5,
        };
        let small = NodeProfile {
            vram: 8 * GB,
            layer_ms: 2.0,
            miss_ms: 15.0,
        };
        let c = best_cut(&stack(), EB, WS, ACT, [head, small], 1.0, |_| true).unwrap();
        match c {
            CutChoice::Split(cut) => assert!(
                cut >= 30,
                "the weak node should take only a small tail share, head got {cut}"
            ),
            other => panic!("a node that can take a free resident layer should be kept: {other:?}"),
        }
    }

    /// A crossing dear enough outweighs any residency a split buys: run whole on the best node.
    #[test]
    fn a_dear_crossing_keeps_the_stack_whole() {
        let p = NodeProfile {
            vram: 16 * GB,
            layer_ms: 1.0,
            miss_ms: 0.5,
        };
        let c = best_cut(&stack(), EB, WS, ACT, [p, p], 1.0e9, |_| true).unwrap();
        assert!(
            matches!(c, CutChoice::Whole(_)),
            "an unaffordable crossing must keep the stack on one node, got {c:?}"
        );
    }

    /// Only clean boundaries are cut; if none is clean the stack stays whole.
    #[test]
    fn no_clean_boundary_keeps_the_stack_whole() {
        let p = NodeProfile {
            vram: 16 * GB,
            layer_ms: 1.0,
            miss_ms: 0.5,
        };
        let c = best_cut(&stack(), EB, WS, ACT, [p, p], 0.1, |_| false).unwrap();
        assert!(
            matches!(c, CutChoice::Whole(_)),
            "no clean cut means no split, got {c:?}"
        );
    }
}

#[cfg(test)]
mod nnode_tests {
    use super::*;

    fn stack() -> Vec<u64> {
        vec![100 << 20; 40]
    }
    const EB: u64 = 10 << 20;
    const WS: u64 = 180;
    const GB: u64 = 1 << 30;
    const ACT: u64 = 6;

    fn node(id: &str, vram_gb: u64, layer_ms: f64, miss_ms: f64) -> TimedNode {
        TimedNode {
            id: id.to_string(),
            profile: NodeProfile {
                vram: vram_gb * GB,
                layer_ms,
                miss_ms,
            },
        }
    }

    fn covers(p: &TimePlan, n: usize) -> bool {
        let mut next = 0;
        for s in &p.segments {
            if s.start != next || s.is_empty() {
                return false;
            }
            next = s.end;
        }
        next == n
    }

    /// Three equal nodes on a cheap fabric split three ways: each third holds more experts resident,
    /// so total misses fall enough to pay the two crossings.
    #[test]
    fn three_equal_nodes_split_three_ways() {
        let ns = [
            node("a", 12, 1.0, 0.5),
            node("b", 12, 1.0, 0.5),
            node("c", 12, 1.0, 0.5),
        ];
        let costs = CutCosts::from_link_gbps(
            &[("a", "b", 32.0), ("a", "c", 32.0), ("b", "c", 32.0)],
            256 << 10,
        );
        let p = plan_by_time(&stack(), EB, WS, ACT, &ns, &costs, |_| true).unwrap();
        assert_eq!(
            p.segments.len(),
            3,
            "three equal nodes on a cheap link should all be used: {:?}",
            p.segments
        );
        assert!(
            covers(&p, 40),
            "the plan must cover every layer: {:?}",
            p.segments
        );
    }

    /// A node slow at everything earns nothing on top of two that pay: it is left out of the plan.
    #[test]
    fn a_slow_node_is_left_out() {
        let ns = [
            node("a", 32, 1.0, 0.5),
            node("b", 16, 1.0, 0.7),
            node("slow", 8, 100.0, 50.0),
        ];
        let costs = CutCosts::from_link_gbps(
            &[("a", "b", 32.0), ("a", "slow", 32.0), ("b", "slow", 32.0)],
            256 << 10,
        );
        let p = plan_by_time(&stack(), EB, WS, ACT, &ns, &costs, |_| true).unwrap();
        assert!(
            !p.segments.iter().any(|s| s.node == "slow"),
            "a node slow at everything must be dropped: {:?}",
            p.segments
        );
        assert!(covers(&p, 40));
    }

    /// A high per-hop latency, even over a fast link with a tiny boundary, thins the plan: each
    /// extra node costs its hop, so the DP stops adding nodes once the hop outweighs the residency.
    #[test]
    fn a_high_hop_latency_uses_fewer_nodes() {
        let ns = [
            node("a", 16, 1.0, 0.5),
            node("b", 16, 1.0, 0.5),
            node("c", 16, 1.0, 0.5),
        ];
        let costs = CutCosts::from_link_gbps(
            &[("a", "b", 64.0), ("a", "c", 64.0), ("b", "c", 64.0)],
            16 << 10,
        )
        .with_base_ms(1.0e6);
        let p = plan_by_time(&stack(), EB, WS, ACT, &ns, &costs, |_| true).unwrap();
        assert_eq!(
            p.segments.len(),
            1,
            "an enormous hop latency must keep the stack on one node: {:?}",
            p.segments
        );
        assert!(covers(&p, 40));
    }

    /// A dear fabric keeps the stack whole on the best node - splitting never pays its crossing.
    #[test]
    fn a_dear_fabric_keeps_one_segment() {
        let ns = [node("a", 16, 1.0, 0.5), node("b", 16, 1.0, 0.5)];
        let costs = CutCosts::from_link_gbps(&[("a", "b", 0.0001)], 256 << 10);
        let p = plan_by_time(&stack(), EB, WS, ACT, &ns, &costs, |_| true).unwrap();
        assert_eq!(
            p.segments.len(),
            1,
            "a dear link keeps the stack on one node: {:?}",
            p.segments
        );
        assert!(covers(&p, 40));
    }

    /// On two nodes it agrees with `best_cut`: where best_cut splits, plan_by_time gives two
    /// segments cut at the same layer; where best_cut runs whole, it gives one.
    #[test]
    fn two_nodes_agree_with_best_cut() {
        let head = NodeProfile {
            vram: 32 * GB,
            layer_ms: 1.0,
            miss_ms: 0.5,
        };
        let tail = NodeProfile {
            vram: 14 * GB,
            layer_ms: 2.0,
            miss_ms: 4.0,
        };
        let ns = [
            TimedNode {
                id: "h".into(),
                profile: head,
            },
            TimedNode {
                id: "t".into(),
                profile: tail,
            },
        ];
        let costs = CutCosts::from_link_gbps(&[("h", "t", 32.0)], 256 << 10);
        let crossing = costs.cost("h", "t");
        let bc = best_cut(&stack(), EB, WS, ACT, [head, tail], crossing, |_| true).unwrap();
        let pt = plan_by_time(&stack(), EB, WS, ACT, &ns, &costs, |_| true).unwrap();
        match bc {
            CutChoice::Split(cut) => {
                assert_eq!(
                    pt.segments.len(),
                    2,
                    "best_cut split, plan_by_time did not: {:?}",
                    pt.segments
                );
                assert_eq!(pt.segments[0].end, cut, "cut points differ");
            }
            CutChoice::Whole(_) => assert_eq!(pt.segments.len(), 1),
        }
    }
}
