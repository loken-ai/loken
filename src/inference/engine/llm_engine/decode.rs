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

/// Capture one decode-token forward into a CUDA graph (shared by the
/// non-streaming and streaming decode loops).
///
/// Runs the full gptoss/lfm2 capture recipe: capture on the MODEL's stream,
/// 0-node defense, node-type histogram, and resource pre-upload. Returns `Some((graph, logits_ref))` on success  -
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
    // One graph per card the step runs on, each recorded on that card's stream; the split
    // path records on the model's stream alone.
    let streams = match needs_split {
        true => vec![stream.clone()],
        false => model.graph_segment_streams(),
    };
    if streams.is_empty() {
        warn!("Graph capture found no card stream at pos={pos}; eager decode");
        return None;
    }
    let mut parts = Vec::with_capacity(streams.len());
    let mut logits = None;
    let last = streams.len() - 1;
    for (seg, seg_stream) in streams.iter().enumerate() {
        let (g, out) = capture_segment(model, seg_stream, seg, seg == last, pos, needs_split)?;
        parts.push((g, seg_stream.clone()));
        logits = out;
    }
    Some((DecodeGraph { parts }, logits?))
}

/// The decode step's graphs, one per card in layer order. A launch runs them in order,
/// each card's stream waiting on the previous one's work; the hidden state hops between
/// cards through pinned host memory inside the graphs.
#[cfg(feature = "cuda")]
pub(super) struct DecodeGraph {
    parts: Vec<(
        crate::tensor::cuda_ext::CapturedGraph,
        std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
    )>,
}

#[cfg(feature = "cuda")]
impl DecodeGraph {
    /// Launches every part. With more than one card the host waits for the last card, which
    /// holds the logits: the caller synchronises only the first card's stream.
    pub(super) fn launch(&self) -> crate::tensor::Result<()> {
        let err = |e| crate::tensor::Error::msg(format!("graph launch: {e:?}"));
        for (i, (g, s)) in self.parts.iter().enumerate() {
            g.launch().map_err(err)?;
            if let Some((_, next)) = self.parts.get(i + 1) {
                let ev = crate::tensor::cuda_ext::record_event(s)?;
                crate::tensor::cuda_ext::wait_event(next, &ev)?;
            }
        }
        if self.parts.len() > 1 {
            if let Some((_, last)) = self.parts.last() {
                last.synchronize().map_err(err)?;
            }
        }
        Ok(())
    }

    /// The single-card graph, which the per-token re-capture path updates in place.
    pub(super) fn single_mut(&mut self) -> Option<&mut crate::tensor::cuda_ext::CapturedGraph> {
        match self.parts.as_mut_slice() {
            [(g, _)] => Some(g),
            _ => None,
        }
    }
}

/// Records segment `seg` of the decode step on `stream`. Only the last segment's output is
/// kept: an earlier one reaches the next card through pinned host memory inside the graphs, and
/// is freed inside its capture so the graph allocates nothing it keeps.
#[cfg(feature = "cuda")]
fn capture_segment(
    model: &mut BoxedModelBackend,
    stream: &std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
    seg: usize,
    last: bool,
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
            model.compute_all_from_kv_captured()?
        } else {
            model.forward_graph_segment(seg)?
        };
        let kept = last.then_some(logits);
        let g = crate::tensor::cuda_ext::end_capture(stream)?;
        Ok((g, kept))
    })();
    // A capture records kernels without running them, while the host side of each append
    // still counted its token. The cache holds `pos` tokens, so the eager forward that
    // follows (probation, or the fallback when the capture is dropped) writes slot `pos`
    // and attends over the same positions the replay will.
    model.sync_kv_len_for_graph(pos);
    match cap {
        Err(e) => {
            // A failed mid-capture forward can leave the stream
            // capturing - terminate the capture before anything else
            // touches the stream.
            let _ = crate::tensor::cuda_ext::end_capture(stream);
            let _ = stream.synchronize();
            warn!("Graph capture failed at pos={pos} (segment {seg}): {e}; eager decode for rest of request");
            None
        }
        Ok((None, _)) => {
            warn!("Graph capture returned no graph at pos={pos}; eager decode");
            None
        }
        Ok((Some(g), logits)) => {
            let nc = g.num_nodes().unwrap_or(0);
            // DEFENSE IN DEPTH: a 0-node graph means the captured
            // forward recorded NO work (e.g. every layer bailed on an
            // inconsistent KV state). Storing it would replay a no-op
            // every token -> frozen logits -> constant-token output.
            if nc == 0 {
                warn!("CUDA graph captured 0 nodes at pos={pos} - discarding and falling back to eager decode");
                drop(g);
                return None;
            }
            info!(
                "CUDA graph captured at pos={} segment {} ({} nodes)",
                pos, seg, nc
            );
            // Node-type histogram of the capture.
            if let Ok(nodes) = g.nodes(nc) {
                if let Ok(types) = g.node_types(&nodes) {
                    use crate::tensor::cuda_ext::CUgraphNodeType;
                    let mut kernel = 0;
                    let mut memcpy = 0;
                    let mut memset = 0;
                    let mut empty = 0;
                    let mut alloc = 0;
                    let mut free = 0;
                    let mut other = 0;
                    for t in &types {
                        match *t {
                            CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => kernel += 1,
                            CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => memcpy += 1,
                            CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => memset += 1,
                            CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY => empty += 1,
                            CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => alloc += 1,
                            CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => free += 1,
                            _ => other += 1,
                        }
                    }
                    info!("node histogram: kernel={kernel} memcpy={memcpy} memset={memset} empty={empty} MEM_ALLOC={alloc} MEM_FREE={free} other={other}");
                }
            }
            // The eager pass before the capture left its transients cached in the pool;
            // the graph's own allocations need that memory when it is uploaded and run.
            let _ = stream.synchronize();
            crate::tensor::cuda::trim_pool(stream.context().ordinal());
            // Pre-upload graph resources to device memory before first
            // launch - eliminates the lazy upload window on cuGraphLaunch.
            if let Err(e) = g.upload() {
                warn!("Graph upload failed (will fall back to lazy upload): {e}");
            }
            Some((g, logits))
        }
    }
}
