//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Context length above which prompt-lookup (PLD) speculative decode is gated
/// OFF - on BOTH CPU and GPU. PLD drafts n-gram matches from the context; at long
/// context it drafts more but acceptance collapses (~15% @2.5K), so the per-position
/// verify weight-GEMVs cost more than the drafts save (qwen3/deepcoder GPU 2.5K were
/// a loss with PLD on, a tie with it gated). PLD is exact spec-decode (verify
/// re-samples each position) -> gating changes only speed, never the greedy tokens.
pub(super) fn pld_max_ctx_threshold() -> usize {
    2048
}

/// Apply repetition penalty to logits (matching llama.cpp / Ollama behavior).
/// For each recently-used token, if its logit is positive, divide by penalty;
/// if negative, multiply by penalty. This reduces the probability of repeating tokens.
pub(crate) fn apply_repeat_penalty(
    logits: &Tensor,
    recent_tokens: &[u32],
    penalty: f32,
    last_n: usize,
) -> crate::tensor::Result<Tensor> {
    if penalty <= 1.0 || last_n == 0 || recent_tokens.is_empty() {
        return Ok(logits.clone());
    }
    let mut logits_v: Vec<f32> = logits.to_vec1()?;
    penalise_recent(&mut logits_v, recent_tokens, penalty, last_n);
    Tensor::from_vec(logits_v, logits.shape(), &logits.device())
}

/// Divides (or, for a negative logit, multiplies) by `penalty` the logit of every distinct
/// token among the last `last_n`: a token repeated in the window is penalised once, as the
/// device kernel does.
fn penalise_recent(logits: &mut [f32], recent_tokens: &[u32], penalty: f32, last_n: usize) {
    if penalty <= 1.0 || last_n == 0 {
        return;
    }
    let start = recent_tokens.len().saturating_sub(last_n);
    let mut seen: Vec<u32> = Vec::with_capacity(recent_tokens.len() - start);
    for &tok in &recent_tokens[start..] {
        if seen.contains(&tok) {
            continue;
        }
        seen.push(tok);
        if let Some(l) = logits.get_mut(tok as usize) {
            *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
        }
    }
}

/// Token sampling for a row that may live on a card.
///
/// A greedy draw with no bias or log-probabilities runs as one fused penalty + argmax kernel
/// on the card (4 bytes come back instead of the row). Anything else draws on the host.
pub(crate) fn gpu_sample(
    logits: &Tensor,
    recent_tokens: &[u32],
    repeat_penalty: f32,
    repeat_last_n: usize,
    logits_processor: &mut crate::inference::sample::token_sampling::LogitsProcessor,
) -> crate::tensor::Result<u32> {
    // A top-k draw runs on the card and returns only the token.
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        if let Some((k, temperature, top_p, u)) =
            logits_processor.device_draw(crate::inference::kernel::fused::SAMPLE_TOPK_MAX)
        {
            let start = recent_tokens.len().saturating_sub(repeat_last_n);
            let mut unique: Vec<u32> = Vec::with_capacity(recent_tokens.len() - start);
            if repeat_penalty > 1.0 {
                for &t in &recent_tokens[start..] {
                    if !unique.contains(&t) {
                        unique.push(t);
                    }
                }
            }
            return crate::inference::kernel::fused::sample_topk(
                logits,
                &unique,
                repeat_penalty,
                temperature,
                k,
                top_p,
                u,
            );
        }
    }
    // Any other sampled draw, a bias or a log-probability request is served on the host,
    // where the sampler reads the whole row; the device argmax knows none of them.
    if logits_processor.needs_host() {
        let mut v = logits
            .flatten_all()?
            .to_dtype(crate::tensor::DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?;
        penalise_recent(&mut v, recent_tokens, repeat_penalty, repeat_last_n);
        return logits_processor.sample_logits(v);
    }
    // Fused repeat penalty + argmax in a single CUDA kernel (1 launch vs 6+)
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        let start = recent_tokens.len().saturating_sub(repeat_last_n);
        let window = &recent_tokens[start..];
        // Dedup with a fixed-size stack buffer instead of HashSet - for a
        // typical repeat_last_n of 64 the quadratic scan is faster than
        // HashSet alloc + hashing. Pre-allocates a Vec of capacity=window.len
        // so the only allocation is the final result; HashSet adds an extra.
        let mut unique: Vec<u32> = Vec::with_capacity(window.len());
        for &t in window {
            if !unique.contains(&t) {
                unique.push(t);
            }
        }
        return crate::inference::kernel::fused::fused_penalty_argmax(
            logits,
            &unique,
            repeat_penalty,
        );
    }

    // CPU fallback
    let logits_cpu = logits.to_device(&Device::Cpu)?;
    let logits_cpu =
        apply_repeat_penalty(&logits_cpu, recent_tokens, repeat_penalty, repeat_last_n)?;
    logits_cpu
        .argmax(crate::tensor::D::Minus1)?
        .to_scalar::<u32>()
}

/// Path B step 3 entry point. Sister of `gpu_sample` that also
/// returns the sampled token wrapped as a device-resident U32 Tensor of
/// shape `[1, 1]` - exactly what `state.model.forward(&x, pos)` expects
/// as its input - so the next decode iteration can skip the
/// `Tensor::new(&[host_tok], device)?.unsqueeze(0)?` H->D round-trip.
///
/// Only the CUDA + greedy (temperature=0) path produces the device
/// tensor; the CPU / temperature>=1 fallbacks return `None` for the
/// device tensor and the caller must build `x` from the host token
/// (existing behavior). Temperature > 0 needs full sampling on host
/// logits and isn't on the Path B hot path.
#[cfg(feature = "cuda")]
pub(super) fn gpu_sample_returning_tensor(
    logits: &Tensor,
    recent_tokens: &[u32],
    repeat_penalty: f32,
    repeat_last_n: usize,
) -> crate::tensor::Result<(u32, Option<Tensor>)> {
    if !logits.device().is_cuda() {
        // CPU fallback - no device tensor.
        let logits_cpu = logits.to_device(&Device::Cpu)?;
        let logits_cpu =
            apply_repeat_penalty(&logits_cpu, recent_tokens, repeat_penalty, repeat_last_n)?;
        let tok: u32 = logits_cpu
            .argmax(crate::tensor::D::Minus1)?
            .to_scalar::<u32>()?;
        return Ok((tok, None));
    }

    let start = recent_tokens.len().saturating_sub(repeat_last_n);
    let window = &recent_tokens[start..];
    let mut unique: Vec<u32> = Vec::with_capacity(window.len());
    for &t in window {
        if !unique.contains(&t) {
            unique.push(t);
        }
    }
    let (tok, dev_slice) = crate::inference::kernel::fused::fused_penalty_argmax_u32_with_device(
        logits,
        &unique,
        repeat_penalty,
    )?;
    let dev_tensor =
        crate::tensor::cuda_ext::tensor_from_u32_slice(dev_slice, (1, 1), &logits.device())?;
    Ok((tok, Some(dev_tensor)))
}

/// Capture one decode-token forward into CUDA graphs (shared by the
/// non-streaming and streaming decode loops).
///
/// Runs the full gptoss/lfm2 capture recipe: capture on each card's stream, 0-node defense,
/// node-type histogram, and resource pre-upload. The step is a sequence of ops: each captured
/// one is recorded into a graph of its own, each eager one (a layer that reads or writes the
/// host between kernels, the hop between cards) is kept as a step to run at every replay.
/// Returns `Some((graphs, logits_ref))` on success -
/// note a capture only RECORDS: the logits buffer has NOT been written
/// for this token; the caller must `graph.launch()` once and run
/// probation validation before sampling.
///
/// On ANY failure returns `None` with the stream left out of capture
/// mode; the caller should compute the token eagerly, call
/// `invalidate_graph_state()`. Re-running the captured forward is safe - the
/// KV write at `pos` is idempotent (dev_slot kernels write the same slot).
#[cfg(feature = "cuda")]
pub(super) fn capture_decode_graph(
    model: &mut BoxedModelBackend,
    stream: &std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
    pos: usize,
    needs_split: bool,
) -> Option<(DecodeGraph, Tensor)> {
    // Drop the transients a previous request's capture left in the model.
    model.clear_graph_transients();
    // The split path records on the model's stream alone.
    let ops = match needs_split {
        true => vec![(true, stream.clone())],
        false => model.graph_step_ops(),
    };
    if ops.is_empty() {
        warn!("Graph capture found no card stream at pos={pos}; eager decode");
        return None;
    }
    let mut steps = Vec::with_capacity(ops.len());
    let mut logits = None;
    let mut nodes = NodeCounts::default();
    for (op, (captured, op_stream)) in ops.iter().enumerate() {
        if !captured {
            steps.push(DecodeStep::Eager(op));
            continue;
        }
        let (g, out) = capture_op(model, op_stream, op, pos, needs_split)?;
        nodes.add(&g);
        steps.push(DecodeStep::Graph(
            g,
            op_stream.clone(),
            crate::inference::offload::take_capture_marks(),
        ));
        if out.is_some() {
            logits = out;
        }
    }
    // A captured op with no node recorded nothing: the replay would be a no-op and the logits
    // would stay as they were.
    if nodes.total == 0 {
        warn!("CUDA graph captured 0 nodes at pos={pos} - discarding and falling back to eager decode");
        return None;
    }
    let graphs = steps
        .iter()
        .filter(|s| matches!(s, DecodeStep::Graph(..)))
        .count();
    info!(
        "CUDA graph captured at pos={pos}: {graphs} graphs, {} eager ops, {} nodes",
        steps.len() - graphs,
        nodes.total
    );
    {
        let mut by_count: Vec<_> = nodes.kernels.iter().collect();
        by_count.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let top: Vec<String> = by_count
            .iter()
            .take(40)
            .map(|(name, n)| format!("{n} {name}"))
            .collect();
        info!("kernels in the step's graphs: {}", top.join("; "));
        tracing::debug!("first kernels of the step: {}", nodes.sequence.join(" | "));
    }
    info!(
        "node histogram: kernel={} memcpy={} memset={} empty={} MEM_ALLOC={} MEM_FREE={} other={}",
        nodes.kernel, nodes.memcpy, nodes.memset, nodes.empty, nodes.alloc, nodes.free, nodes.other
    );
    if crate::inference::offload::stage_prof_enabled() {
        for step in &steps {
            if let DecodeStep::Graph(g, s, marks) = step {
                info!("stage marks recorded in the graph: {}", marks.len());
                report_kernel_costs(g, s);
            }
        }
    }
    // The eager pass before the capture left its transients cached in the pool; the graphs'
    // own allocations need that memory when they are uploaded and run.
    let mut trimmed = Vec::new();
    for step in &steps {
        if let DecodeStep::Graph(_, s, _) = step {
            let card = s.context().ordinal();
            if !trimmed.contains(&card) {
                trimmed.push(card);
                let _ = s.synchronize();
                crate::tensor::cuda::trim_pool(card);
            }
        }
    }
    // Pre-upload graph resources to device memory before first launch - eliminates the lazy
    // upload window on cuGraphLaunch.
    for step in &steps {
        if let DecodeStep::Graph(g, _, _) = step {
            if let Err(e) = g.upload() {
                warn!("Graph upload failed (will fall back to lazy upload): {e}");
            }
        }
    }
    Some((DecodeGraph { steps }, logits?))
}

/// One part of a decode step.
#[cfg(feature = "cuda")]
enum DecodeStep {
    /// A recorded graph and the stream of the card it runs on.
    Graph(
        crate::tensor::cuda_ext::CapturedGraph,
        std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
        Vec<crate::inference::offload::StageMark>,
    ),
    /// An op the model runs eagerly at every replay.
    Eager(usize),
}

/// The decode step: its graphs and the eager ops between them, in order.
#[cfg(feature = "cuda")]
pub(super) struct DecodeGraph {
    steps: Vec<DecodeStep>,
}

#[cfg(feature = "cuda")]
impl DecodeGraph {
    /// Runs every part in order. The host waits for the last card's stream, which holds the
    /// logits.
    pub(super) fn launch(&self, model: &mut BoxedModelBackend) -> crate::tensor::Result<()> {
        let err = |e| crate::tensor::Error::msg(format!("graph launch: {e:?}"));
        use crate::inference::offload::{prof_record, stage_prof_enabled};
        let prof = stage_prof_enabled();
        let started = prof.then(std::time::Instant::now);
        let mut last = None;
        let mut timed: Vec<(usize, cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)> =
            Vec::new();
        for (at, step) in self.steps.iter().enumerate() {
            let t = prof.then(std::time::Instant::now);
            match step {
                DecodeStep::Graph(g, s, _) => {
                    let marks = if prof {
                        let ctx = s.context();
                        let new = || {
                            ctx.new_event(Some(
                                cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT,
                            ))
                        };
                        match (new(), new()) {
                            (Ok(a), Ok(b)) if a.record(s).is_ok() => Some((a, b)),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    g.launch().map_err(err)?;
                    if let Some((a, b)) = marks {
                        if b.record(s).is_ok() {
                            timed.push((at, a, b));
                        }
                    }
                    last = Some(s);
                    if let Some(t) = t {
                        prof_record("step: graph launch", t.elapsed().as_nanos() as u64);
                    }
                }
                DecodeStep::Eager(op) => {
                    model.run_graph_op(*op)?;
                    if let Some(t) = t {
                        prof_record("step: eager op", t.elapsed().as_nanos() as u64);
                    }
                }
            }
        }
        let t = prof.then(std::time::Instant::now);
        if let Some(s) = last {
            s.synchronize().map_err(err)?;
        }
        for (_, a, b) in &timed {
            if let Ok(ms) = a.elapsed_ms(b) {
                prof_record("step: graph device time", (ms as f64 * 1e6) as u64);
            }
        }
        if prof {
            for step in &self.steps {
                if let DecodeStep::Graph(_, _, marks) = step {
                    crate::inference::offload::record_stage_marks(marks);
                }
            }
        }
        if let (Some(t), Some(started)) = (t, started) {
            prof_record(
                "step: wait for the last card",
                t.elapsed().as_nanos() as u64,
            );
            prof_record("step: total", started.elapsed().as_nanos() as u64);
        }
        Ok(())
    }

    /// The single-card graph, which the per-token re-capture path updates in place.
    pub(super) fn single_mut(&mut self) -> Option<&mut crate::tensor::cuda_ext::CapturedGraph> {
        match self.steps.as_mut_slice() {
            [DecodeStep::Graph(g, _, _)] => Some(g),
            _ => None,
        }
    }
}

/// Node counts of the graphs of a step, by type.
#[cfg(feature = "cuda")]
#[derive(Default)]
struct NodeCounts {
    /// Kernel launches by kernel name.
    kernels: std::collections::BTreeMap<String, usize>,
    /// The first kernels of the step, in the order the driver lists the nodes.
    sequence: Vec<String>,
    total: usize,
    kernel: usize,
    memcpy: usize,
    memset: usize,
    empty: usize,
    alloc: usize,
    free: usize,
    other: usize,
}

#[cfg(feature = "cuda")]
impl NodeCounts {
    fn add(&mut self, g: &crate::tensor::cuda_ext::CapturedGraph) {
        use crate::tensor::cuda_ext::CUgraphNodeType;
        let count = g.num_nodes().unwrap_or(0);
        self.total += count;
        let Ok(nodes) = g.nodes(count) else { return };
        if let Ok(names) = g.kernel_names(&nodes) {
            for name in names.into_iter().flatten() {
                if self.sequence.len() < 700 {
                    self.sequence.push(name.chars().take(48).collect());
                }
                *self.kernels.entry(name).or_insert(0) += 1;
            }
        }
        let Ok(types) = g.node_types(&nodes) else {
            return;
        };
        for t in &types {
            match *t {
                CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => self.kernel += 1,
                CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => self.memcpy += 1,
                CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => self.memset += 1,
                CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY => self.empty += 1,
                CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => self.alloc += 1,
                CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => self.free += 1,
                _ => self.other += 1,
            }
        }
    }
}

/// Records op `op` of the decode step on `stream`. Only the head's output is kept: the
/// hidden state of the other ops stays in the step's buffers.
#[cfg(feature = "cuda")]
fn capture_op(
    model: &mut BoxedModelBackend,
    stream: &std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
    op: usize,
    pos: usize,
    needs_split: bool,
) -> Option<(crate::tensor::cuda_ext::CapturedGraph, Option<Tensor>)> {
    let cap: crate::tensor::Result<(
        Option<crate::tensor::cuda_ext::CapturedGraph>,
        Option<Tensor>,
    )> = (|| {
        crate::tensor::cuda_ext::begin_capture(stream)?;
        let logits = if needs_split {
            // Compute-only half is captured. Reads Q from
            // graph_q_buffer + KV from kv_cache (both refreshed by
            // prepare_all_kv, run by the caller before this).
            Some(model.compute_all_from_kv_captured()?)
        } else {
            model.run_graph_op(op)?
        };
        let g = crate::tensor::cuda_ext::end_capture(stream)?;
        Ok((g, logits))
    })();
    // A capture records kernels without running them, while the host side of each append
    // still counted its token. The cache holds `pos` tokens, so the eager forward that
    // follows (probation, or the fallback when the capture is dropped) writes slot `pos` and
    // attends over the same positions the replay will.
    model.sync_kv_len_for_graph(pos);
    match cap {
        Err(e) => {
            // A failed mid-capture forward can leave the stream
            // capturing - terminate the capture before anything else
            // touches the stream.
            let _ = crate::tensor::cuda_ext::end_capture(stream);
            let _ = stream.synchronize();
            warn!("Graph capture failed at pos={pos} (op {op}): {e}; eager decode for rest of request");
            None
        }
        Ok((None, _)) => {
            warn!("Graph capture returned no graph at pos={pos}; eager decode");
            None
        }
        Ok((Some(g), logits)) => Some((g, logits)),
    }
}

/// With the stage profile on, say what each kernel of a graph costs the card: the graph is
/// launched with every node of one kernel switched off, and the fall in its time is that
/// kernel's. The launches write into the model's state, so the request is not one to keep.
#[cfg(feature = "cuda")]
fn report_kernel_costs(
    g: &crate::tensor::cuda_ext::CapturedGraph,
    s: &std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
) {
    static DONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    let ctx = s.context();
    let time = |g: &crate::tensor::cuda_ext::CapturedGraph| -> Option<f64> {
        let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let (a, b) = (
            ctx.new_event(Some(flags)).ok()?,
            ctx.new_event(Some(flags)).ok()?,
        );
        let mut best = f64::MAX;
        for _ in 0..4 {
            a.record(s).ok()?;
            g.launch().ok()?;
            b.record(s).ok()?;
            b.synchronize().ok()?;
            best = best.min(a.elapsed_ms(&b).ok()? as f64 * 1e3);
        }
        Some(best)
    };
    let Ok(count) = g.num_nodes() else { return };
    let Ok(nodes) = g.nodes(count) else { return };
    let Ok(names) = g.kernel_names(&nodes) else {
        return;
    };
    let Some(whole) = time(g) else { return };
    let mut by_name: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for (at, name) in names.iter().enumerate() {
        if let Some(name) = name {
            by_name
                .entry(name.chars().take(60).collect())
                .or_default()
                .push(at);
        }
    }
    let mut costs = Vec::new();
    for (name, at) in &by_name {
        for &i in at {
            let _ = g.set_node_enabled(&nodes, i, false);
        }
        let without = time(g);
        for &i in at {
            let _ = g.set_node_enabled(&nodes, i, true);
        }
        if let Some(without) = without {
            costs.push((whole - without, at.len(), name.clone()));
        }
    }
    costs.sort_by(|a, b| b.0.total_cmp(&a.0));
    if !DONE.swap(true, std::sync::atomic::Ordering::Relaxed) {
        let top: Vec<String> = costs
            .iter()
            .take(30)
            .map(|(us, n, name)| format!("{us:.0} us x{n} {name}"))
            .collect();
        info!(
            "graph {whole:.0} us; cost of each kernel: {}",
            top.join("; ")
        );
    }
}
