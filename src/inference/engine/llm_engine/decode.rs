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
        // At debug level, the casts and host bounces by call site up to this recording.
        crate::tensor::bounce::report();
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

/// Replays of a recorded decode step held against an eager run of the same step before the
/// recording is trusted: the first ones after a capture, the first one in a request that
/// replays a step kept from an earlier one.
pub(super) const GRAPH_VALIDATIONS: usize = 3;

/// Recorded graphs a loaded model keeps between requests, with the model's
/// `ModelBackend::graph_generation` they were recorded at: they are replayed again only
/// while the model reports the same, and dropped otherwise.
pub(crate) struct KeptGraphs<T> {
    held: Option<(T, (u64, u64))>,
}

impl<T> Default for KeptGraphs<T> {
    fn default() -> Self {
        Self { held: None }
    }
}

impl<T> KeptGraphs<T> {
    /// Keeps `graphs` recorded at `at`; a model that reports no generation keeps none.
    pub(crate) fn keep(&mut self, graphs: T, at: Option<(u64, u64)>) {
        self.held = at.map(|at| (graphs, at));
    }

    /// Whether graphs are kept and the model still reports what they were recorded at.
    pub(crate) fn current(&self, now: Option<(u64, u64)>) -> bool {
        matches!((&self.held, now), (Some((_, at)), Some(now)) if *at == now)
    }

    /// The kept graphs and their generation while they are current; stale ones are dropped.
    pub(crate) fn take_current(&mut self, now: Option<(u64, u64)>) -> Option<(T, (u64, u64))> {
        let current = self.current(now);
        if let Some((_, at)) = &self.held {
            tracing::debug!(
                "kept graphs recorded at {at:?}, the model at {now:?}: {}",
                if current {
                    "replayed"
                } else {
                    "recorded again"
                }
            );
        }
        let held = self.held.take();
        held.filter(|_| current)
    }

    /// Drops the kept graphs. Done before the buffers they read are dropped.
    pub(crate) fn clear(&mut self) {
        self.held = None;
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.held.is_none()
    }
}

/// What a request's decode leaves recorded: the step's graph with the logits buffer it writes,
/// and the verify windows.
#[cfg(feature = "cuda")]
#[derive(Default)]
pub(crate) struct DecodeGraphs {
    pub(super) step: Option<(DecodeGraph, Tensor)>,
    pub(super) verify: VerifyGraphs,
}

// The graphs are reached only through the loaded model's state, behind its mutex, and run on
// the model's stream from whichever thread holds it, as the model itself is.
#[cfg(feature = "cuda")]
unsafe impl Send for DecodeGraphs {}

/// Where the graphs of a request were recorded: the generation of the first recording, and
/// whether a later one saw another. Graphs of one generation only are kept past the request.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RecordedAt {
    at: Option<(u64, u64)>,
    mixed: bool,
}

impl RecordedAt {
    /// Graphs kept from an earlier request, recorded at `at`.
    pub(super) fn kept(at: (u64, u64)) -> Self {
        Self {
            at: Some(at),
            mixed: false,
        }
    }

    /// A graph was recorded with the model at `now`.
    pub(super) fn note(&mut self, now: Option<(u64, u64)>) {
        match (self.at, now) {
            (None, Some(now)) if !self.mixed => self.at = Some(now),
            (Some(at), Some(now)) if at == now => {}
            _ => self.mixed = true,
        }
    }

    /// The generation the graphs may be kept at once the request ends with the model at `now`:
    /// all of them recorded at it, and nothing moved since.
    pub(super) fn keepable(&self, now: Option<(u64, u64)>) -> Option<(u64, u64)> {
        tracing::debug!(
            "graphs recorded at {:?} (mixed {}), the model at {now:?}",
            self.at,
            self.mixed
        );
        match (self.at, now) {
            (Some(at), Some(now)) if !self.mixed && at == now => Some(at),
            _ => None,
        }
    }
}

/// The recorded draft verifies of a request, one graph per window size, and whether they were
/// given up on.
#[cfg(feature = "cuda")]
pub(super) struct VerifyGraphs {
    by_rows: std::collections::HashMap<usize, (VerifyGraph, Tensor)>,
    /// Window sizes kept from an earlier request whose replay has not been held against an
    /// eager run in this one.
    unheld: std::collections::HashSet<usize>,
    pub(super) disabled: bool,
    /// Replays that named other tokens than their eager rows, this request.
    mismatches: usize,
    /// The step's buffers were dropped with a window: the step's graph is recorded again.
    pub(super) step_stale: bool,
    /// Seconds spent readying the replays' buffers, and in the replays themselves, summed.
    pub(super) prepare_secs: f64,
    pub(super) launch_secs: f64,
}

#[cfg(feature = "cuda")]
impl Default for VerifyGraphs {
    fn default() -> Self {
        Self {
            by_rows: std::collections::HashMap::new(),
            unheld: std::collections::HashSet::new(),
            disabled: false,
            mismatches: 0,
            step_stale: false,
            prepare_secs: 0.0,
            launch_secs: 0.0,
        }
    }
}

#[cfg(feature = "cuda")]
impl VerifyGraphs {
    /// Forget the graphs: the buffers they read are gone with the step's.
    pub(super) fn clear(&mut self) {
        self.by_rows.clear();
        self.unheld.clear();
    }

    /// The windows kept for another request: its tallies start again, and each window's first
    /// replay in it is held against an eager run.
    pub(super) fn for_next_request(&mut self) {
        self.unheld = self.by_rows.keys().copied().collect();
        self.disabled = false;
        self.mismatches = 0;
        self.step_stale = false;
        self.prepare_secs = 0.0;
        self.launch_secs = 0.0;
    }

    pub(super) fn is_empty(&self) -> bool {
        self.by_rows.is_empty()
    }
}

/// The copy a lookup offers for a window: a match of the longest n-gram the cache keeps
/// copies as far as it reaches, one n-gram shorter copies two tokens, a shorter one none.
pub(super) fn copy_draft(
    cache: &crate::inference::serve::prompt_lookup::NgramDraftCache,
) -> Vec<u32> {
    let (mut draft, matched) = cache.lookup_matched();
    let reach = match matched {
        0 => 0,
        n if n >= cache.max_ngram() => draft.len(),
        n if n + 1 >= cache.max_ngram() => 2,
        _ => 0,
    };
    draft.truncate(reach);
    if draft.len() >= 2 {
        draft
    } else {
        Vec::new()
    }
}

/// Whether every row of `logits` (`[.., rows, vocab]`) is finite: a row's sum is not finite
/// when any of its values is not.
pub(super) fn rows_finite(logits: &Tensor) -> crate::tensor::Result<bool> {
    let sums = logits
        .sum_keepdim(crate::tensor::D::Minus1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    Ok(sums.iter().all(|v| v.is_finite()))
}

/// Each row's sum and the token it names, for a report.
#[cfg(feature = "cuda")]
fn rows_told(logits: &Tensor) -> crate::tensor::Result<String> {
    let sums = logits
        .sum_keepdim(crate::tensor::D::Minus1)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let rows = logits.dim(crate::tensor::D::Minus2)?;
    let mut out = Vec::with_capacity(rows);
    for (r, sum) in sums.iter().enumerate() {
        let row = logits.flatten_to(1)?.narrow(0, r, 1)?.flatten_all()?;
        let top = row.argmax(0)?.to_scalar::<u32>()?;
        let peak = row.max_keepdim(0)?.flatten_all()?.to_vec1::<f32>()?[0];
        out.push(format!("row {r}: sum {sum}, top {top} at {peak}"));
    }
    Ok(out.join("; "))
}

/// Wait for the card without keeping a core busy; a wait that lasts says what every expert
/// cache's mailbox holds, once, since a step recorded against them waits on the host there.
#[cfg(feature = "cuda")]
fn wait_telling(
    model: &BoxedModelBackend,
    stream: &std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
) -> crate::tensor::Result<()> {
    let event = stream
        .record_event(None)
        .map_err(|e| crate::tensor::Error::msg(format!("record_event: {e:?}")))?;
    let started = std::time::Instant::now();
    let mut told = false;
    // A fault on the card is an error of the query, not a wait that never ends.
    while !event
        .try_is_complete()
        .map_err(|e| crate::tensor::Error::msg(format!("the card's work: {e:?}")))?
    {
        std::thread::sleep(std::time::Duration::from_micros(20));
        if !told && started.elapsed().as_secs_f64() > WAIT_TELLS_AFTER_SECS {
            told = true;
            for line in model.tier_mailboxes() {
                warn!("a recorded verify waits on the card: {line}");
            }
        }
    }
    tier_trusted(model)
}

/// After a recorded step has run: an error when an expert cache's worker released the card's
/// waits meanwhile, the step's output then holding no host answer.
#[cfg(feature = "cuda")]
fn tier_trusted(model: &BoxedModelBackend) -> crate::tensor::Result<()> {
    match model.tier_failure() {
        Some(why) => crate::tensor::bail!("expert tier: {why}"),
        None => Ok(()),
    }
}

/// Seconds a wait on the card lasts before it says what the mailboxes hold.
#[cfg(feature = "cuda")]
const WAIT_TELLS_AFTER_SECS: f64 = 3.0;

/// A draft verify of `ids` (the token carried forward, then the draft) at `pos` through a
/// recorded window: its logits `[1, rows, vocab]` and, when the window was recorded for this
/// verify, the seconds its replay alone took (what a verify costs from now on, which the
/// recording's own cost would hide); `None` when the verify has to run eagerly, the model's
/// state as it was. The first verify of each window size runs eagerly first, is recorded,
/// replayed, and the replay has to name the eager tokens, or recorded verifies are given up on
/// for the request.
#[cfg(feature = "cuda")]
pub(super) fn verify_recorded(
    model: &mut BoxedModelBackend,
    stream: &std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
    graphs: &mut VerifyGraphs,
    pos: usize,
    ids: &[u32],
) -> Option<(Tensor, Option<f64>)> {
    let rows = ids.len();
    if graphs.disabled || rows < 2 || rows > model.verify_window_rows() {
        return None;
    }
    // The buffers the step's graph reads go with the state, so the step is recorded again:
    // a graph replayed over dropped buffers reads freed memory.
    let give_up = |model: &mut BoxedModelBackend, graphs: &mut VerifyGraphs, why: String| {
        warn!("recorded verify given up at pos={pos}: {why}; eager verifies for the rest of the request");
        graphs.disabled = true;
        graphs.clear();
        if let Err(e) = model.verify_window_settle(0) {
            warn!("the model did not return to the rows before the verify: {e}");
        }
        model.invalidate_graph_state();
        graphs.step_stale = true;
    };
    // The window's first row stands for the plain step of the carried token: before a
    // recording, that step is run eagerly from a checkpoint of the state, and the state
    // returned, so that the window's row can be held against it.
    // A size that passed every check once with this model is held against its eager run alone.
    let proven = model.graph_proven(rows);
    let plain_row: Option<Vec<f32>> = if graphs.by_rows.contains_key(&rows) || proven {
        None
    } else {
        match model.plain_step_held(ids[0], pos) {
            Ok(row) => Some(row),
            Err(e) => {
                warn!("the plain step the window's first row is held against did not run: {e}");
                None
            }
        }
    };
    let preparing = std::time::Instant::now();
    let prepared = model.verify_window_prepare(pos, ids);
    graphs.prepare_secs += preparing.elapsed().as_secs_f64();
    if let Err(e) = prepared {
        give_up(model, graphs, format!("prepare: {e}"));
        return None;
    }
    // A window kept from an earlier request is held at its first replay in this one against
    // its eager run from the same state, as a recording is.
    let held_rows: Option<Vec<Vec<f32>>> = if graphs.unheld.contains(&rows) {
        let eager = match model.forward_verify_window(rows) {
            Ok(l) => l,
            Err(e) => {
                give_up(model, graphs, format!("eager run of a kept window: {e}"));
                return None;
            }
        };
        let _ = stream.synchronize();
        let want: Option<Vec<Vec<f32>>> = (0..rows)
            .map(|r| {
                eager
                    .narrow(1, r, 1)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .ok()
            })
            .collect();
        if let Err(e) = model
            .verify_window_settle(0)
            .and_then(|()| model.verify_window_prepare(pos, ids))
        {
            give_up(
                model,
                graphs,
                format!("return before a kept window's replay: {e}"),
            );
            return None;
        }
        let _ = stream.synchronize();
        match want {
            Some(w) => Some(w),
            None => {
                give_up(
                    model,
                    graphs,
                    "the eager rows of a kept window were not read".into(),
                );
                return None;
            }
        }
    } else {
        None
    };
    if let Some((g, logits)) = graphs.by_rows.get(&rows) {
        let launching = std::time::Instant::now();
        if let Err(e) = g.launch(model, rows) {
            give_up(model, graphs, format!("launch: {e}"));
            return None;
        }
        model.verify_window_replayed();
        if let Err(e) = wait_telling(model, stream) {
            give_up(model, graphs, format!("replay: {e}"));
            return None;
        }
        graphs.launch_secs += launching.elapsed().as_secs_f64();
        // A row that is not finite names token zero and, once committed, leaves the recurrent
        // state not finite for the rest of the request: the window is given up instead.
        // A replay that is not finite is run once more from the state before it: the fault
        // is a transient of the cards, the same graph on the same rows is finite again.
        let finite = match rows_finite(logits) {
            Ok(true) => true,
            Ok(false) => {
                let told = rows_told(logits).unwrap_or_default();
                let hops = model.verify_hops_report(rows);
                warn!("a replay of {rows} rows at pos={pos} is not finite: {hops}");
                let again = model
                    .verify_window_settle(0)
                    .and_then(|()| model.verify_window_prepare(pos, ids))
                    .and_then(|()| g.launch(model, rows))
                    .and_then(|()| {
                        model.verify_window_replayed();
                        wait_telling(model, stream)
                    })
                    .and_then(|()| rows_finite(logits));
                match again {
                    Ok(true) => {
                        warn!("a replay of {rows} rows at pos={pos} was not finite ({told}); the same replay is finite again");
                        for line in model.tier_mailboxes() {
                            warn!("tier at the time: {line}");
                        }
                        true
                    }
                    Ok(false) => false,
                    Err(e) => {
                        give_up(model, graphs, format!("the replay run again: {e}"));
                        return None;
                    }
                }
            }
            Err(e) => {
                give_up(
                    model,
                    graphs,
                    format!("the replay's rows were not read: {e}"),
                );
                return None;
            }
        };
        if !finite {
            let told = rows_told(logits).unwrap_or_default();
            let states = model
                .verify_window_settle(0)
                .map(|()| model.verify_states_report())
                .unwrap_or_else(|e| format!("not settled: {e}"));
            give_up(
                model,
                graphs,
                format!(
                    "a row of the replay is not finite twice: {told}; after the return: {states}"
                ),
            );
            return None;
        }
        let out = logits.clone();
        if let Some(want) = held_rows.as_ref() {
            let agrees = (0..rows).all(|r| {
                out.narrow(1, r, 1)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .is_ok_and(|got| {
                        crate::inference::engine::decode_step::replay_agrees(&got, &want[r])
                    })
            });
            if !agrees {
                // The window, the step and their buffers are recorded again from here.
                warn!("a kept verify window of {rows} rows at pos={pos} named other tokens than its eager rows; recorded again");
                graphs.clear();
                if let Err(e) = model.verify_window_settle(0) {
                    warn!("the model did not return to the rows before the verify: {e}");
                }
                model.invalidate_graph_state();
                graphs.step_stale = true;
                return None;
            }
            graphs.unheld.remove(&rows);
        }
        return Some((out, None));
    }
    // The eager run of the window: what the replay has to reproduce.
    let eager = match model.forward_verify_window(rows) {
        Ok(l) => l,
        Err(e) => {
            give_up(model, graphs, format!("eager window: {e}"));
            return None;
        }
    };
    let _ = stream.synchronize();
    let eager_rows: Option<Vec<Vec<f32>>> = (0..rows)
        .map(|r| {
            eager
                .narrow(1, r, 1)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .ok()
        })
        .collect();
    if let (Some(plain), Some(first)) = (
        plain_row.as_ref(),
        eager_rows.as_ref().and_then(|r| r.first()),
    ) {
        let distance = crate::inference::engine::decode_step::replay_distance(first, plain);
        info!("recorded verify of {rows} rows at pos={pos}: the first row is {distance:.4} of the plain step's norm away");
        if !crate::inference::engine::decode_step::replay_agrees(first, plain) {
            let _ = model.verify_window_settle(0);
            give_up(
                model,
                graphs,
                format!(
                    "the window's first row is not the plain step ({distance:.4} of its norm away)"
                ),
            );
            return None;
        }
    }
    if let Err(e) = model
        .verify_window_settle(0)
        .and_then(|()| model.verify_window_prepare(pos, ids))
    {
        give_up(model, graphs, format!("return before the capture: {e}"));
        return None;
    }
    let _ = stream.synchronize();
    // The rows run again from the returned state have to give the same logits: a return that
    // leaves the state changed would make every replay of the window wrong.
    if !proven {
        let again = match model.forward_verify_window(rows) {
            Ok(l) => l,
            Err(e) => {
                give_up(model, graphs, format!("eager window again: {e}"));
                return None;
            }
        };
        let _ = stream.synchronize();
        if let Some(want) = eager_rows.as_ref() {
            let distances: Vec<String> = (0..rows)
                .map(|r| {
                    again
                        .narrow(1, r, 1)
                        .and_then(|t| t.flatten_all())
                        .and_then(|t| t.to_vec1::<f32>())
                        .map(|got| {
                            crate::inference::engine::decode_step::replay_distance(&got, &want[r])
                        })
                        .map_or("?".to_string(), |d| format!("{d:.4}"))
                })
                .collect();
            let returned = (0..rows).all(|r| {
                again
                    .narrow(1, r, 1)
                    .and_then(|t| t.flatten_all())
                    .and_then(|t| t.to_vec1::<f32>())
                    .is_ok_and(|got| {
                        crate::inference::engine::decode_step::replay_agrees(&got, &want[r])
                    })
            });
            if !returned {
                give_up(
                    model,
                    graphs,
                    format!("the rows run again from the returned state differ (relative distance per row: {})", distances.join(" ")),
                );
                return None;
            }
        }
        if let Err(e) = model
            .verify_window_settle(0)
            .and_then(|()| model.verify_window_prepare(pos, ids))
        {
            give_up(model, graphs, format!("return before the capture: {e}"));
            return None;
        }
        let _ = stream.synchronize();
    }
    let g = match VerifyGraph::capture(model, rows) {
        Ok(g) => {
            model.note_graphs_recorded();
            g
        }
        Err(e) => {
            let _ = stream.synchronize();
            give_up(model, graphs, format!("capture: {e}"));
            return None;
        }
    };
    // The eager run and the recording wrote the same buffer: the replay fills it again.
    let logits = eager;
    let replay = std::time::Instant::now();
    if let Err(e) = g.launch(model, rows) {
        give_up(model, graphs, format!("first launch: {e}"));
        return None;
    }
    let _ = stream.synchronize();
    let replay_secs = replay.elapsed().as_secs_f64();
    // Every row of the replay against the eager row; every row's distance is named when one fails.
    let mut distances = Vec::with_capacity(rows);
    let agrees = eager_rows.is_some_and(|want| {
        (0..rows).all(|r| {
            logits
                .narrow(1, r, 1)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .ok()
                .is_some_and(|got: Vec<f32>| {
                    distances.push(format!(
                        "{:.4}",
                        crate::inference::engine::decode_step::replay_distance(&got, &want[r])
                    ));
                    crate::inference::engine::decode_step::replay_agrees(&got, &want[r])
                })
        })
    });
    if !agrees && graphs.mismatches + 1 < REPLAY_MISMATCHES_TO_GIVE_UP {
        // A near tie flips on the last bits of an expert computed on the host in one run and
        // on the card in the other; a wrong replay disagrees again. This verify runs eagerly
        // and a later window is recorded afresh.
        graphs.mismatches += 1;
        warn!(
            "recorded verify of {rows} rows at pos={pos} named other tokens than its eager rows (relative distance per row: {}); recorded again at a later window",
            distances.join(" ")
        );
        graphs.clear();
        if let Err(e) = model.verify_window_settle(0) {
            warn!("the model did not return to the rows before the verify: {e}");
        }
        model.invalidate_graph_state();
        graphs.step_stale = true;
        return None;
    }
    if !agrees {
        give_up(
            model,
            graphs,
            format!(
                "the replay of {rows} rows does not name the eager rows' tokens (relative distance per row: {})",
                distances.join(" ")
            ),
        );
        return None;
    }
    info!(
        "recorded verify of {rows} rows captured at pos={pos}: {} nodes, replayed in {:.1} ms",
        g.num_nodes(),
        replay_secs * 1e3
    );
    {
        let nodes = g.nodes();
        let mut by_count: Vec<_> = nodes.kernels.iter().collect();
        by_count.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let top: Vec<String> = by_count
            .iter()
            .take(40)
            .map(|(name, n)| format!("{n} {name}"))
            .collect();
        info!(
            "kernels in the verify's graph of {rows} rows ({} kernels, {} memcpy, {} memset, {} alloc): {}",
            nodes.kernel,
            nodes.memcpy,
            nodes.memset,
            nodes.alloc,
            top.join("; ")
        );
    }
    graphs.by_rows.insert(rows, (g, logits.clone()));
    model.mark_graph_proven(rows);
    Some((logits, Some(replay_secs)))
}

/// Replays that name other tokens than their eager rows after which a request verifies eagerly.
#[cfg(feature = "cuda")]
pub(super) const REPLAY_MISMATCHES_TO_GIVE_UP: usize = 3;

/// One part of a recorded verify.
#[cfg(feature = "cuda")]
enum VerifyStep {
    /// A recorded graph and the stream of the card it runs on.
    Graph(
        crate::tensor::cuda_ext::CapturedGraph,
        std::sync::Arc<crate::tensor::cuda_ext::CudaStream>,
    ),
    /// An op the model runs eagerly at every replay: the rows' hop between two cards.
    Eager(usize),
}

/// A recorded verify of one window size: its graphs and the eager ops between them.
#[cfg(feature = "cuda")]
pub(super) struct VerifyGraph {
    steps: Vec<VerifyStep>,
}

#[cfg(feature = "cuda")]
impl VerifyGraph {
    /// Records the prepared verify of `rows` rows, op by op on its card's stream.
    fn capture(model: &mut BoxedModelBackend, rows: usize) -> crate::tensor::Result<Self> {
        let ops = model.verify_step_ops();
        if ops.is_empty() {
            crate::tensor::bail!("the verify has no op to record");
        }
        let mut steps = Vec::with_capacity(ops.len());
        for (op, (captured, s)) in ops.iter().enumerate() {
            if !captured {
                steps.push(VerifyStep::Eager(op));
                continue;
            }
            crate::tensor::cuda_ext::begin_capture(s)?;
            let ran = model.run_verify_op(op, rows);
            // A failed forward can leave the stream capturing: the capture ends before
            // anything else touches the stream.
            let ended = crate::tensor::cuda_ext::end_capture(s);
            ran?;
            match ended? {
                Some(g) => {
                    if let Err(e) = g.upload() {
                        warn!("verify graph upload failed (will fall back to lazy upload): {e}");
                    }
                    steps.push(VerifyStep::Graph(g, s.clone()));
                }
                None => crate::tensor::bail!("the capture of op {op} recorded nothing"),
            }
        }
        Ok(Self { steps })
    }

    /// Runs every part in order and waits for the last card's stream, which holds the logits.
    fn launch(&self, model: &mut BoxedModelBackend, rows: usize) -> crate::tensor::Result<()> {
        let err = |e| crate::tensor::Error::msg(format!("graph launch: {e:?}"));
        model.tier_check()?;
        let mut last = None;
        for step in &self.steps {
            match step {
                VerifyStep::Graph(g, s) => {
                    g.launch().map_err(err)?;
                    last = Some(s);
                }
                VerifyStep::Eager(op) => {
                    model.run_verify_op(*op, rows)?;
                }
            }
        }
        if let Some(s) = last {
            s.synchronize().map_err(err)?;
        }
        tier_trusted(model)
    }

    /// The nodes of every graph of the verify.
    fn nodes(&self) -> NodeCounts {
        let mut nodes = NodeCounts::default();
        for step in &self.steps {
            if let VerifyStep::Graph(g, _) = step {
                nodes.add(g);
            }
        }
        nodes
    }

    /// Nodes in every graph of the verify.
    fn num_nodes(&self) -> usize {
        self.steps
            .iter()
            .map(|step| match step {
                VerifyStep::Graph(g, _) => g.num_nodes().unwrap_or(0),
                VerifyStep::Eager(_) => 0,
            })
            .sum()
    }
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
        model.tier_check()?;
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
        tier_trusted(model)?;
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

#[cfg(test)]
mod kept_graphs_tests {
    use super::{KeptGraphs, RecordedAt};

    #[test]
    fn graphs_are_kept_while_the_generation_holds() {
        let mut kept = KeptGraphs::default();
        kept.keep(7u32, Some((1, 5)));
        assert!(kept.current(Some((1, 5))));
        assert_eq!(kept.take_current(Some((1, 5))), Some((7, (1, 5))));
        assert!(kept.is_empty());
    }

    #[test]
    fn graphs_are_dropped_once_a_buffer_moved_or_the_model_changed() {
        let mut kept = KeptGraphs::default();
        kept.keep(7u32, Some((1, 5)));
        assert!(!kept.current(Some((1, 6))));
        assert!(!kept.current(Some((2, 5))));
        assert!(!kept.current(None));
        assert_eq!(kept.take_current(Some((1, 6))), None);
        assert!(kept.is_empty(), "stale graphs are dropped");
        kept.keep(8, None);
        assert!(
            kept.is_empty(),
            "a model without a generation keeps nothing"
        );
    }

    #[test]
    fn only_graphs_of_one_generation_are_kept_past_the_request() {
        let mut at = RecordedAt::default();
        assert_eq!(at.keepable(Some((1, 5))), None, "nothing recorded");
        at.note(Some((1, 5)));
        at.note(Some((1, 5)));
        assert_eq!(at.keepable(Some((1, 5))), Some((1, 5)));
        assert_eq!(
            at.keepable(Some((1, 6))),
            None,
            "a buffer moved after the recording"
        );
        at.note(Some((1, 6)));
        assert_eq!(
            at.keepable(Some((1, 6))),
            None,
            "recordings of two generations"
        );
        let mut kept = RecordedAt::kept((1, 5));
        kept.note(None);
        assert_eq!(kept.keepable(Some((1, 5))), None);
    }
}
