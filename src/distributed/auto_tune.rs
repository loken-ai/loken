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
    let miss_ms = bench_miss_ms(dim, inter, gate_dt, down_dt, 200)?;
    Some(NodeProfile {
        vram,
        layer_ms,
        miss_ms,
    })
}

/// How long this node's cores take to compute one routed expert it does not hold resident: gate and
/// up in `gate_dt`, down in `down_dt`, over `dim` x `inter`, averaged over `iters`. This is the cut
/// planner's `miss_ms`, measured on the node rather than guessed. The weight bytes are a fixed
/// pattern - a GEMV's time is its size, not its values - and the activation drawn once.
///
/// `None` when a kernel cannot run these dims and formats (an unaligned dim, a quant the kernel does
/// not carry): a node whose miss cost cannot be measured yields no number rather than crashing the
/// caller, which is what lets this run on the model-load path. The caller reads `None` as unknown.
pub fn bench_miss_ms(
    dim: usize,
    inter: usize,
    gate_dt: crate::tensor::quantized::GgmlDType,
    down_dt: crate::tensor::quantized::GgmlDType,
    iters: usize,
) -> Option<f64> {
    use crate::tensor::quant_cpu::matmul_bytes_multi;
    let iters = iters.max(1);
    let gate_bytes = vec![0x11u8; inter * (dim / gate_dt.block_size()) * gate_dt.type_size()];
    let down_bytes = vec![0x11u8; dim * (inter / down_dt.block_size()) * down_dt.type_size()];
    let x = vec![0.05f32; dim];
    let h = vec![0.05f32; inter];
    let mut gout = vec![0f32; inter];
    let mut uout = vec![0f32; inter];
    let mut dout = vec![0f32; dim];
    let run = |gout: &mut [f32], uout: &mut [f32], dout: &mut [f32]| -> bool {
        matmul_bytes_multi(gate_dt, dim, inter, &[&gate_bytes], &[&x], &mut [gout]).is_ok()
            && matmul_bytes_multi(gate_dt, dim, inter, &[&gate_bytes], &[&x], &mut [uout]).is_ok()
            && matmul_bytes_multi(down_dt, inter, dim, &[&down_bytes], &[&h], &mut [dout]).is_ok()
    };
    // Warm the caches and the pool, then time. A kernel failure at either phase yields no number.
    if !run(&mut gout, &mut uout, &mut dout) {
        return None;
    }
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        if !run(&mut gout, &mut uout, &mut dout) {
            return None;
        }
    }
    Some(t0.elapsed().as_secs_f64() * 1000.0 / iters as f64)
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

#[cfg(test)]
mod bench_tests {
    use super::bench_miss_ms;
    use crate::tensor::quantized::GgmlDType;

    /// The bench measures a positive per-miss cost over real, block-aligned expert dims.
    #[test]
    fn bench_measures_a_miss() {
        let ms = bench_miss_ms(5120, 2304, GgmlDType::Iq2Xxs, GgmlDType::Q2K, 8)
            .expect("aligned dims measure");
        assert!(ms > 0.0);
    }

    /// A dim the kernel cannot run (not a multiple of the quant's block) yields no number rather
    /// than a panic - the guard that lets this run on the model-load path.
    #[test]
    fn an_unrunnable_dim_yields_none() {
        assert!(bench_miss_ms(100, 2304, GgmlDType::Iq2Xxs, GgmlDType::Q2K, 4).is_none());
    }
}
