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
    let start = recent_tokens.len().saturating_sub(last_n);
    let window = &recent_tokens[start..];
    let mut logits_v: Vec<f32> = logits.to_vec1()?;
    for &tok in window {
        let idx = tok as usize;
        if idx < logits_v.len() {
            if logits_v[idx] > 0.0 {
                logits_v[idx] /= penalty;
            } else {
                logits_v[idx] *= penalty;
            }
        }
    }
    Tensor::from_vec(logits_v, logits.shape(), &logits.device())
}

/// GPU-accelerated token sampling.
///
/// For temperature < 1.0: uses argmax on GPU (transfers 4 bytes instead of 500KB).
/// For temperature >= 1.0: falls back to full CPU sampling with repeat penalty.
pub(crate) fn gpu_sample(
    logits: &Tensor,
    recent_tokens: &[u32],
    repeat_penalty: f32,
    repeat_last_n: usize,
    _temperature: f32,
    _top_k: usize,
    logits_processor: &mut crate::inference::sample::token_sampling::LogitsProcessor,
) -> crate::tensor::Result<u32> {
    // A bias or a log-probability request is served on the host, where the sampler
    // reads the whole row; the device argmax knows neither.
    if logits_processor.needs_host() {
        let cpu = logits.to_device(&Device::Cpu)?;
        let cpu = apply_repeat_penalty(&cpu, recent_tokens, repeat_penalty, repeat_last_n)?;
        return logits_processor.sample(&cpu);
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
/// Runs the full gptoss/lfm2 capture recipe: adaptive capture arena
/// (seed 16 MB, double on overflow, ceiling = free VRAM - 1/8), capture
/// on the MODEL's stream, 0-node defense, node-type histogram, and
/// resource pre-upload. Returns `Some((graph, logits_ref))` on success  -
/// note a capture only RECORDS: the logits buffer has NOT been written
/// for this token; the caller must `graph.launch()` once and run
/// probation validation before sampling.
///
/// On ANY failure returns `None` with the stream left out of capture
/// mode; the caller should compute the token eagerly, call
/// `invalidate_graph_state()` and free the capture arena. Re-running the
/// captured forward across arena retries is safe - the KV write at `pos`
/// is idempotent (dev_slot kernels write the same slot).
#[cfg(feature = "cuda")]
pub(super) fn capture_decode_graph(
    model: &mut BoxedModelBackend,
    stream: &std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
    pos: usize,
    needs_split: bool,
) -> Option<(crate::tensor::cuda_ext::CudaGraph, Tensor)> {
    let ctx = stream.context();
    // Drop stale captured-region transients from a previous request's
    // capture BEFORE the arena they live in is freed by
    // begin_capture_arena (arena-range drops are no-ops only while the
    // arena is alive).
    model.clear_graph_transients();
    let free_vram = ctx.mem_get_info().map(|(f, _)| f).unwrap_or(0);
    let ceiling = free_vram.saturating_sub(free_vram / 8).max(1 << 20);
    let mut arena_bytes = (16usize << 20).min(ceiling);
    loop {
        if let Err(e) = ctx.begin_capture_arena(arena_bytes) {
            warn!(
                "graph capture arena alloc failed ({} MB): {e:?}; eager decode",
                arena_bytes >> 20
            );
            return None;
        }
        let cap: crate::tensor::Result<(Option<crate::tensor::cuda_ext::CudaGraph>, Tensor)> =
            (|| {
                crate::tensor::cuda_ext::begin_capture(stream)?;
                let logits = if needs_split {
                    // Compute-only half is captured. Reads Q from
                    // graph_q_buffer + KV from kv_cache (both refreshed by
                    // prepare_all_kv, run by the caller before this).
                    model.compute_all_from_kv_captured()?
                } else {
                    model.forward_from_hidden()?
                };
                // cuda_ext::end_capture instantiates with flags=0: no
                // AUTO_FREE (keeps stream-ordered allocs persistent across
                // replays). AUTO_FREE_ON_LAUNCH=1 measured DESTROYING qwen3
                // (-95%) - it sweeps the cached stable buffers too.
                let g = crate::tensor::cuda_ext::end_capture(stream)?;
                Ok((g, logits))
            })();
        let (_arena_peak, overflow) = ctx.end_capture_arena();
        match cap {
            Err(e) => {
                // A failed mid-capture forward can leave the stream
                // capturing - terminate the capture before anything else
                // touches the stream.
                let _ = crate::tensor::cuda_ext::end_capture(stream);
                let _ = stream.synchronize();
                warn!("Graph capture failed at pos={pos}: {e}; eager decode for rest of request");
                return None;
            }
            Ok((None, _)) => {
                warn!("Graph capture returned no graph at pos={pos}; eager decode");
                return None;
            }
            Ok((Some(g), logits)) => {
                if overflow > 0 {
                    // The forward spilled past the arena into real
                    // allocations (-> MEM_ALLOC nodes that relocate on
                    // replay). Discard this graph, grow, recapture  -
                    // unless already at the VRAM ceiling.
                    drop(g);
                    model.clear_graph_transients();
                    ctx.free_capture_arena();
                    if arena_bytes >= ceiling {
                        warn!(
                            "graph capture arena overflow at VRAM ceiling ({} MB); eager decode",
                            ceiling >> 20
                        );
                        return None;
                    }
                    arena_bytes = (arena_bytes * 2).min(ceiling);
                    continue;
                }
                info!(
                    "CUDA graph arena sized to {} MB (free VRAM {} MB)",
                    arena_bytes >> 20,
                    free_vram >> 20
                );
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
                info!("CUDA graph captured at pos={} ({} nodes)", pos, nc);
                // Node-type histogram: confirms
                // MEM_ALLOC/MEM_FREE stay at 0 with the capture arena.
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
                // Pre-upload graph resources to device memory before first
                // launch - eliminates the lazy upload window on cuGraphLaunch.
                if let Err(e) = g.upload() {
                    warn!("Graph upload failed (will fall back to lazy upload): {e}");
                }
                return Some((g, logits));
            }
        }
    }
}
