//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

thread_local! {
    static GRAPH_STEP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive, the forward on this thread is the graph's decode step: it reads its position
/// from the per-step buffers `update_graph_state` fills. Every other forward, an eager step
/// between replays or a verified draft row, reads the position it is given.
pub(crate) struct GraphStep {
    previous: bool,
}

impl GraphStep {
    pub(crate) fn enter() -> Self {
        Self {
            previous: GRAPH_STEP.with(|s| s.replace(true)),
        }
    }

    pub(crate) fn active() -> bool {
        GRAPH_STEP.with(std::cell::Cell::get)
    }
}

impl Drop for GraphStep {
    fn drop(&mut self) {
        GRAPH_STEP.with(|s| s.set(self.previous));
    }
}

/// One operation of a decode step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepOp {
    /// Layers of card segment `seg`, recorded in one graph; the head follows in it when `head`.
    /// A run that follows a hop starts with the copy that lands the hidden state on its card
    /// (`recv`), and one that precedes a hop ends with the copy that sends it (`send`).
    Run {
        seg: usize,
        layers: std::ops::Range<usize>,
        head: bool,
        recv: bool,
        send: bool,
    },
    /// The head alone, after a last layer that runs eagerly.
    Head { seg: usize },
    /// One layer that runs eagerly at every replay.
    Layer { seg: usize, layer: usize },
    /// The hidden state crossing from card segment `seg` to the next: the next card's stream
    /// waits for the send, and the copies a neighbouring graph does not hold run here.
    Hop { seg: usize, send: bool, recv: bool },
}

impl StepOp {
    /// Whether the op is recorded in a graph; the others run eagerly at every replay.
    pub fn captured(&self) -> bool {
        matches!(self, StepOp::Run { .. } | StepOp::Head { .. })
    }
}

/// The ops of a decode step over the layer ranges `segs` of its cards, in order, given which
/// layers are recorded: each run of recorded layers on a card is one op, each other layer an op
/// of its own, a hop stands between two cards, and the head ends the step, in the last run when
/// the last layer is recorded and as an op of its own otherwise. Empty for no segment.
pub(crate) fn plan_step(
    segs: &[std::ops::Range<usize>],
    captured: impl Fn(usize) -> bool,
) -> Vec<StepOp> {
    let mut ops = Vec::new();
    for (seg, range) in segs.iter().enumerate() {
        if seg > 0 {
            ops.push(StepOp::Hop {
                seg: seg - 1,
                send: true,
                recv: true,
            });
        }
        let mut run: Option<usize> = None;
        for layer in range.clone() {
            if captured(layer) {
                run.get_or_insert(layer);
                continue;
            }
            if let Some(first) = run.take() {
                ops.push(StepOp::Run {
                    seg,
                    layers: first..layer,
                    head: false,
                    recv: false,
                    send: false,
                });
            }
            ops.push(StepOp::Layer { seg, layer });
        }
        if let Some(first) = run {
            ops.push(StepOp::Run {
                seg,
                layers: first..range.end,
                head: false,
                recv: false,
                send: false,
            });
        }
    }
    match ops.last_mut() {
        Some(StepOp::Run { head, .. }) => *head = true,
        Some(_) => ops.push(StepOp::Head {
            seg: segs.len() - 1,
        }),
        None => {}
    }
    // The copies of a hop belong to the graph beside it, where there is one: the hidden state
    // then crosses between two recordings with no call of the host in between but the event.
    for at in 0..ops.len() {
        if !matches!(ops[at], StepOp::Hop { .. }) {
            continue;
        }
        let before_is_run = at > 0 && matches!(ops[at - 1], StepOp::Run { .. });
        let after_is_run = matches!(ops.get(at + 1), Some(StepOp::Run { .. }));
        if before_is_run {
            if let StepOp::Run { send, .. } = &mut ops[at - 1] {
                *send = true;
            }
        }
        if after_is_run {
            if let StepOp::Run { recv, .. } = &mut ops[at + 1] {
                *recv = true;
            }
        }
        if let StepOp::Hop { send, recv, .. } = &mut ops[at] {
            *send = !before_is_run;
            *recv = !after_is_run;
        }
    }
    ops
}

/// The hidden state of one decode step crossing from one card to the next without peer
/// access: the source stream copies it into page-locked host memory, the destination stream
/// copies it from there into a buffer on its card. Both copies are graph nodes; the order
/// between the two streams is the launcher's.
#[cfg(feature = "gpu")]
pub(crate) struct GraphHop {
    host: *mut std::ffi::c_void,
    bytes: usize,
    /// `[1, 1, width]` F32 on the destination card.
    dst: Tensor,
}

// SAFETY: `host` is a page-locked allocation owned by this hop alone, freed once on drop;
// the copies that touch it are ordered on the card streams.
#[cfg(feature = "gpu")]
unsafe impl Send for GraphHop {}
#[cfg(feature = "gpu")]
unsafe impl Sync for GraphHop {}

#[cfg(feature = "gpu")]
impl GraphHop {
    fn new(width: usize, dst_dev: &Device) -> Result<Self> {
        Self::with_rows(width, 1, dst_dev)
    }

    /// A hop of up to `rows` rows of `width`: its landing buffer is `[1, rows, width]`.
    pub(crate) fn with_rows(width: usize, rows: usize, dst_dev: &Device) -> Result<Self> {
        let bytes = rows * width * std::mem::size_of::<f32>();
        let host =
            crate::tensor::cuda_ext::malloc_host(bytes).map_err(crate::tensor::Error::msg)?;
        let dst = Tensor::zeros_on((1, rows, width), crate::tensor::DType::F32, dst_dev)?;
        Ok(Self { host, bytes, dst })
    }

    /// The landing buffer's first `rows` rows, `[1, rows, width]`.
    pub(crate) fn landing(&self, rows: usize) -> Result<Tensor> {
        self.dst.narrow(1, 0, rows)
    }

    /// Copies `hidden`, `[1, rows, width]` up to the hop's rows, into host memory on its
    /// card's stream.
    pub(crate) fn send(&self, hidden: &Tensor) -> Result<()> {
        let src = hidden.to_dtype(crate::tensor::DType::F32)?.contiguous()?;
        let bytes = src.elem_count() * std::mem::size_of::<f32>();
        if bytes > self.bytes || src.dims().last() != self.dst.dims().last() {
            crate::tensor::bail!("graph hop: hidden {:?} does not fit the hop", src.dims());
        }
        let stream = crate::tensor::cuda_ext::stream_of(&src.device())?;
        let ptr = f32_device_ptr(&src)?;
        // A raw driver copy runs in the thread's current context: the stream's own.
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("graph hop: bind context: {e:?}")))?;
        let r = unsafe {
            crate::tensor::gpu_runtime::driver::sys::cuMemcpyDtoHAsync_v2(
                self.host,
                ptr,
                bytes,
                stream.cu_stream(),
            )
        };
        if r != crate::tensor::gpu_runtime::driver::sys::cudaError_enum::CUDA_SUCCESS {
            crate::tensor::bail!("graph hop: device to host copy failed: {r:?}");
        }
        Ok(())
    }

    /// Copies the host memory into the destination buffer on its card's stream.
    /// How many values of the first `rows` rows are not finite in the host buffer the card
    /// sent to, and in the landing the next card received: `(host, landing)`.
    pub(crate) fn finite_report(&self, rows: usize) -> (usize, usize) {
        let width = self.dst.dims().last().copied().unwrap_or(0);
        let n = (rows * width).min(self.bytes / 4);
        let host = unsafe { std::slice::from_raw_parts(self.host as *const f32, n) }
            .iter()
            .filter(|v| !v.is_finite())
            .count();
        let landing = self
            .dst
            .narrow(1, 0, rows.min(self.dst.dims()[1]))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map(|v| v.iter().filter(|x| !x.is_finite()).count())
            .unwrap_or(usize::MAX);
        (host, landing)
    }

    pub(crate) fn receive(&self) -> Result<Tensor> {
        let stream = crate::tensor::cuda_ext::stream_of(&self.dst.device())?;
        let ptr = f32_device_ptr(&self.dst)?;
        // A raw driver copy runs in the thread's current context: the stream's own.
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| crate::tensor::Error::msg(format!("graph hop: bind context: {e:?}")))?;
        let r = unsafe {
            crate::tensor::gpu_runtime::driver::sys::cuMemcpyHtoDAsync_v2(
                ptr,
                self.host,
                self.bytes,
                stream.cu_stream(),
            )
        };
        if r != crate::tensor::gpu_runtime::driver::sys::cudaError_enum::CUDA_SUCCESS {
            crate::tensor::bail!("graph hop: host to device copy failed: {r:?}");
        }
        Ok(self.dst.clone())
    }
}

#[cfg(feature = "gpu")]
impl Drop for GraphHop {
    fn drop(&mut self) {
        // SAFETY: allocated by `malloc_host` in `new` and freed only here.
        unsafe { crate::tensor::cuda_ext::free_host(self.host) }
    }
}

/// The device address of an F32 card tensor's first element.
#[cfg(feature = "gpu")]
fn f32_device_ptr(t: &Tensor) -> Result<u64> {
    use crate::tensor::cuda_ext::DevicePtr;
    let (st, layout) = t.storage_and_layout();
    match &*st {
        crate::tensor::StorageView::Cuda(c) => {
            let s = c.as_cuda_slice::<f32>()?;
            Ok(s.device_ptr(s.stream()).0
                + (layout.start_offset() * std::mem::size_of::<f32>()) as u64)
        }
        _ => crate::tensor::bail!("graph hop: tensor is not on a card"),
    }
}

impl GenericHeteroTransformer {
    /// Tokens the KV cache accepts: the ceiling it grows to, or was allocated at.
    pub fn kv_capacity(&self) -> usize {
        self.config.effective_max_context()
    }

    /// Why layer `li` runs eagerly between the captured parts of a decode step, `None` when
    /// it is captured: its step reads or writes the host between two kernels.
    pub(super) fn layer_eager_reason(&self, li: usize) -> Option<&'static str> {
        let l = &self.layers[li];
        if !l.tp_shards.is_empty() && l.tp_comms.is_none() {
            return Some("its shares move their activations through copies to the host");
        }
        if l.context.is_some() {
            return Some("its cache is cut over several cards, whose parts it combines");
        }
        // A recurrent mixer is recorded only where its step advances the state in place: a
        // recorded step writes the addresses it read.
        if !l.mixer.decodes_in_place(&l.cos.device()) {
            return Some("recurrent mixer: its decode step does not advance its state in place");
        }
        // The recorded layer runs a mixture that reads the normed hidden alone; one beside a
        // dense feed-forward, with its own norms, runs only in the eager forward.
        if l.moe.as_ref().is_some_and(|m| !m.router_prenormed) {
            return Some(
                "a mixture beside a dense feed-forward: the captured layer has no such branch",
            );
        }
        if l.sparse.is_some() {
            // The indexer and its attention run on the card where its kernels serve the layer's
            // shapes and its cache is the card's half-width one.
            #[cfg(feature = "gpu")]
            let on_card = l.indexed_on_card();
            #[cfg(not(feature = "gpu"))]
            let on_card = false;
            if !on_card {
                return Some("indexed attention selects its tokens on the host");
            }
        }
        // A cached layer whose host share is served through a mailbox records like any other.
        if l.moe.as_ref().is_some_and(|m| {
            m.decodes_through_host() && !m.tier.as_ref().is_some_and(|t| t.graph_ready())
        }) {
            return Some("some routed experts are in host memory");
        }
        None
    }

    /// Whether layer `li` is recorded in the decode step's graphs.
    pub(super) fn layer_captured(&self, li: usize) -> bool {
        self.layer_eager_reason(li).is_none()
    }

    /// The layers the decode step records, in order.
    fn captured_layers(&self) -> impl Iterator<Item = &GenericTransformerLayer> {
        (0..self.layers.len())
            .filter(|&i| self.layer_captured(i))
            .map(|i| &self.layers[i])
    }

    /// Whether this model's KV-cache state is safe under CUDA graph
    /// capture/replay. The graph_ops branch in `forward_attn` writes K/V
    /// into the F-dtype `kv_cache` (k_buffer / v_buffer) using
    /// `scatter_set(graph_kv_pos, ...)` - a device-side position tensor
    /// updated outside the captured graph. For this to work the F-dtype
    /// cache must hold the prefill history.
    ///
    /// Layers that route prefill into Q8/Q4 caches (and DON'T also
    /// populate the F-dtype side via `populate_dual_kv`) leave the
    /// F-dtype buffer empty - graph replay then reads garbage and CUDA
    /// crashes with ILLEGAL_ADDRESS. The fix is structural (Q4/Q8 caches
    /// would need a device-side position tensor too), so for now we
    /// gate auto-on on F-dtype-only or dual-populate layers.
    /// Why graph capture is refused, or `None` when nothing refuses it.
    ///
    /// The reason names the branch that fired. One canned string used to stand for all of
    /// them and blamed an HD=512 F-dtype layer for every refusal - including models whose
    /// head_dim is 128, where it sent a root-cause after a layer that does not exist.
    pub fn kv_state_graph_block(&self) -> Option<&'static str> {
        self.captured_layers().find_map(|l| {
            // A layer that does not attend keeps no KV.
            if !l.attends() {
                return None;
            }
            // An indexed layer writes its half-width cache by device position, with kernels of
            // its own.
            if l.sparse.is_some() {
                return None;
            }
            // Q4 KV cache: append/flush/attention all run through
            // device-pos kernels in graph mode. `update_graph_state`
            // primes `cur_pos_dev` before each token, so this is safe.
            if l.has_q4_cache() {
                return None;
            }
            // Dual-populate donors keep F-dtype populated alongside the
            // quantized cache. The graph_ops branch in `forward_attn`
            // takes the F-dtype path with scatter_set onto `graph_kv_pos`,
            // which is graph-safe by construction.
            if l.populate_dual_kv {
                return None;
            }
            // Q8 KV cache: graph wiring is in tree and coherent
            // (commit 2c7fa3d, fresh-alloc dev_pos pattern matching Q4)
            // but gated OFF by a bench: deepcoder regressed
            // -8.7pp (+5.6% -> -3.1% vs Ollama) at medium prompt. Root
            // cause: dev_pos score kernel writes -INF for every position
            // in `[seq_kv, max_seq_padded=2048)` per token. At seq_kv≈300
            // (medium prompt + ~100 decode), wasted writes dominate the
            // launch-overhead savings - 1700 x 24 heads x 32 layers ≈
            // 1.3M extra writes/token. The host-int non-graph Q8 path
            // (try_q8_decode below the graph branch in forward_attn)
            // skips these and wins on short-medium decode.
            //
            // Re-enable when EITHER:
            //   1. dev_pos kernel skips the -INF write (the prior
            //      attempt did this with a pre-init'd persistent buffer
            //      but raced clone_dtod; needs a facade CudaView API
            //      first), OR
            //   2. dispatch adds a `seq_kv > N` guard so only long
            //      decodes take the graph path, OR
            //   3. max_seq_padded shrinks dynamically with the cache
            //      occupancy.
            if l.has_q8_cache() {
                // A Q8 layer is graph-safe: its device-position kernels read the sequence
                // length from the device and write at the padded stride, the host KV length
                // is synced per replayed token, the capture-frozen ceiling is enforced per
                // token, and probation replays catch anything else. A routed layer groups its
                // experts on the device, so its launches are the same on every token.
                return None;
            }
            // gemma4 HD=512 still blocked.
            // F-dtype-only path: graph-safe ONLY when head_dim <= 256.
            // The HD=512 F-dtype path (gemma4:latest) replays into
            // CUDA_ERROR_ILLEGAL_ADDRESS at pos=31 first replay.
            // A bisection ruled out: cuBLAS workspace growth,
            // matmul-output allocs, strided V, strided K^T, GQA fast
            // path itself. CUDA_LAUNCH_BLOCKING=1 confirms the error
            // is at cuGraphLaunch but doesn't pinpoint which captured
            // kernel - needs Nsight --cuda-graph-trace or cuda-gdb.
            // Remaining candidates: scatter_set for KV writes,
            // QKV projection cuBLAS GEMM at HD=512 output shape,
            // or RoPE apply at HD=512.
            // The capture arena eliminates the HD=512 fresh-alloc crash, but
            // the split-path numerical divergence (see Q8 branch above)
            // blocks shipping. DEFINITIVELY DEAD: graph-on is 5.3x SLOWER
            // than non-graph decode for gemma4 (measured). Do NOT re-attempt.
            (l.head_dim > 256).then_some("F-dtype KV cache with head_dim above 256")
        })
    }

    pub fn kv_state_graph_safe(&self) -> bool {
        self.kv_state_graph_block().is_none()
    }

    /// Combined gate the engine reads to decide whether to auto-on
    /// graph mode. Conditions:
    /// 1. KV cache state is graph-safe (`kv_state_graph_safe()`).
    /// 2. Model lives on a single CUDA device - multi-GPU graph capture
    ///    is not validated yet (per-device capture + cross-device peer
    ///    copies need a per-device graph orchestration that doesn't
    ///    exist).
    /// 3. Architecture is on the bench-validated whitelist. As of
    ///    the perimeter that ran 200/200 tokens coherently
    ///    under env=1 + commits 04cfd3e/cce9912/be7ba76: gemma4 (any
    ///    size), qwen2 (deepcoder), qwen3, mistral (devstral). The
    ///    bench delta per arch:
    ///      - gemma4:latest: +62% (largest win)
    ///      - gemma4:26b:     +7%
    ///      - deepcoder:      +2%
    ///      - qwen3:          ~tied
    ///      - devstral:       ~tied
    ///
    ///    Adding new arches needs the same 200-token + Ollama-compare
    ///    cycle before flipping their auto-on bit on here.
    /// True when this model has F-dtype-only KV layers that would hit
    /// the scatter_set fresh-K-pointer bug under captured forward_from_hidden.
    /// Such models must use the split-path: `prepare_all_kv` (outside
    /// capture) + `compute_all_from_kv` (inside capture). Currently true
    /// for phi2 (moondream) and gemma4:latest HD=512 Global layers.
    pub fn needs_split_graph_path(&self) -> bool {
        self.captured_layers().any(|l| {
            // Any layer that has no Q4/Q8 KV cache AND is not a
            // dual-populate donor falls back to F-dtype scatter_set,
            // which is the unstable path. A sink layer attends over its
            // buffers by the device length and reads no padded mask.
            l.attends()
                && l.sparse.is_none()
                && l.attn_sinks.is_none()
                && !l.has_q4_cache()
                && !l.has_q8_cache()
                && !l.populate_dual_kv
        })
    }

    /// Drop the F16 lm_head when nothing can use it.
    ///
    /// Whether decode captures a CUDA graph, and when it does not, WHY.
    ///
    /// "no capture" and "capture that logs nothing" used to be indistinguishable from the
    /// outside, so a model could sit in the allow-list and never be captured without anyone
    /// noticing - which is exactly what happened to olmoe. The reason is returned rather
    /// than logged here so the caller decides where it surfaces; `graph_capture_auto_on`
    /// keeps the boolean contract for existing callers.
    pub fn graph_capture_decision(&self) -> std::result::Result<(), &'static str> {
        // A sink layer's step is not captured: its replay, which reads the length on the
        // device, costs more than the eager step, whose attention is GPU-bound.
        if self.captured_layers().any(|l| l.attn_sinks.is_some()) {
            return Err("attention sinks: the replayed step costs more than the eager one");
        }
        if let Some(why) = self.kv_state_graph_block() {
            return Err(why);
        }
        // Engine-site capture MECHANICS restored (gptoss
        // treatment, proven end-to-end on the Q4-KV dev-pos path  -
        // deepcoder KV=q4 captures 1636 nodes, 0 MEM_ALLOC/MEM_FREE,
        // replays coherently and token-matches the Q8 eager reference):
        //  1. capture happens on the MODEL's stream (`model_cuda_stream`,
        //     this file's `cuda_stream()`), not state.device's - every
        //     `Device::new_cuda` has its own stream on the native
        //     substrate, so the old capture recorded 0 nodes while the
        //     forward ran eagerly on the model stream;
        //  2. per-slice CudaEvent tracking is disabled on the hetero
        //     loader's device handles BEFORE tensor loads (llm_engine
        //     hetero_cuda_devs creation) - with it ON the capture is
        //     INVALIDATED by in-capture record/wait of outside-capture
        //     events (reproduced; also a large eager decode
        //     win on its own: qwen3:0.6b Q8 ~350->~555 tok/s);
        //  3. cuBLAS workspace pinned + mmvq workspace pre-grown (the
        //     one-time block in update_graph_state) so no in-capture
        //     workspace alloc;
        //  4. transient allocations route through the context capture
        //     arena (adaptively sized by the engine: double on overflow)
        //     so the graph has 0 MEM_ALLOC/MEM_FREE nodes;
        //  5. the capture token launches the recorded graph once (a
        //     capture only RECORDS - the logits buffer was never
        //     written), and the first replays run gptoss-style probation
        //     (argmax vs an eager forward; mismatch -> discard + eager).
        //
        // F-dtype (needs_split) models stay GATED OFF - root-caused
        // to MODEL-side bugs, NOT capture mechanics:
        //  a. the split path is ARCHITECTURALLY WRONG: `prepare_all_kv`
        //     computes EVERY layer's Q/K/V from the EMBEDDING hidden
        //     (batch prepare, then batch compute), but layer i's QKV
        //     must come from layer i-1's output. Verified on qwen3:0.6b
        //     KV=off: L0 output matches the normal forward to ~1e-3
        //     (its input IS the embedding) while L1+ diverge hard ->
        //     garbage from the WARMUP token on, BEFORE any capture. No
        //     capture-mechanics work can fix this; the batched split
        //     design cannot be correct for a serial transformer. (The
        //     engine probation is blind to it because the eager
        //     comparator `compute_all_from_kv_captured` reads the same
        //     wrong inputs.)
        //  b. the NON-split F-dtype graph branch (forward_from_hidden ->
        //     forward_graph scatter_set path) also fails: warmup emits
        //     argmax=0 (flat/NaN logits) on qwen3:0.6b KV=off even
        //     though the padded k_buffer/v_buffer DOES hold the prefill
        //     history (append/append_padded share storage) - the bug is
        //     in the graph_ops attention/mask wiring, unreached since
        //     the substrate flip.
        // Re-enabling F-dtype needs (b) fixed model-side (then the
        // capture arena makes the split workaround unnecessary - retire
        // it). Until then eager decode is correct and this gate makes
        // it explicit.
        if self.needs_split_graph_path() {
            return Err(
                "F-dtype (F16) KV: both graph branches are known-broken, see the note above",
            );
        }
        // Single-GPU requirement (except gemma4:26b which has its own
        // multi-GPU graph code path). Without this, multi-GPU + CPU
        // placement causes `embed_for_graph` to fail with
        // "embed_for_graph needs CUDA" when the first cuda_device
        // entry isn't on the path the embeddings buffer was placed.
        // Verified: devstral loaded across 2 GPUs + CPU
        // under VRAM contention crashes on the first decode token.
        //
        // Count GPUs that ACTUALLY have layers placed, not the
        // engine-populated `cuda_devices` map (which has every detected
        // GPU regardless of placement). A single-segment placement on
        // GPU 0 of a 2-GPU host is graph-safe - the old check returned
        // false here and wrongly disabled graph mode.
        let mut layer_gpus = std::collections::HashSet::<usize>::new();
        let mut has_cpu_layer = false;
        for d in &self.layer_devs {
            match d {
                LayerDevice::Cuda(idx) => {
                    layer_gpus.insert(*idx);
                }
                LayerDevice::Cpu => {
                    has_cpu_layer = true;
                }
            }
        }
        // Spanning cards is no longer a disqualification. It was one only because the
        // segment boundary established its ordering by stopping the host, and a stream
        // synchronize is illegal inside a capture; the boundary now records an event on
        // the source stream and has the destination wait on it, and both of those
        // capture. The exception previously carved out for a single model by its layer
        // count is what showed the rest of the machinery already worked across devices -
        // and it was the wrong model to single out, since anything larger than one card
        // is precisely what most needs its launches folded away.
        let _ = &layer_gpus;
        // CPU-spill disqualifies graph mode unconditionally. CUDA graph
        // capture uses a device-side `graph_kv_pos` index and per-layer
        // CUDA KV buffers; layers on CPU can't honour either, so the
        // scatter_set into their CPU KV buffer with a CUDA `pos_idx`
        // (or CPU `pos_idx` against CUDA-projected k_new in the mixed
        // case) fails with `device mismatch in scatter-set`. Verified
        // qwen3 forced to CUDA(0):0-14, CPU:14-36 under
        // VRAM contention crashed every decode call until this guard.
        if has_cpu_layer {
            return Err("some layers are on the CPU");
        }
        // A layer that reads or writes the host between two kernels runs eagerly between the
        // captured parts of the step, with the layer's own inputs. A model whose layers take
        // inputs the step does not carry (shared caches, per-layer embeddings) is captured
        // whole or not at all.
        let eager: Vec<usize> = (0..self.layers.len())
            .filter(|&i| !self.layer_captured(i))
            .collect();
        if let Some(&first) = eager.first() {
            if eager.len() == self.layers.len()
                || self.config.shared_kv_layers > 0
                || self.ple_dim > 0
            {
                return Err(self
                    .layer_eager_reason(first)
                    .expect("an eager layer has its reason"));
            }
        }
        // The captured decode computes the logits on the card of the last layer. An output
        // projection left elsewhere, on another card or on the host because that card had no
        // room, takes the eager path, which moves the hidden state to it.
        if let Some(LayerDevice::Cuda(last)) = self.layer_devs.last() {
            let head_card = self
                .output_proj_dev
                .as_ref()
                .and_then(|d| match d.location() {
                    crate::tensor::DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
                    _ => None,
                });
            // A wide residual's final mixer stands for the output norm.
            if head_card != Some(*last)
                || (self.output_norm_cuda_weight.is_none() && self.output_hyper.is_none())
            {
                return Err("the output projection is not on the card of the last layer");
            }
        }
        // phi2 (moondream): option-E split path (commits 3c78c9a +
        // 787986a + the engine wiring) moves scatter_set OUTSIDE the
        // captured region (capture succeeds at pos=14 with new path,
        // up from pos=51 before, no scatter_set ILLEGAL_ADDRESS).
        // BUT replay still ILLEGAL_ADDRESS - fresh intermediate
        // tensor allocations inside compute_from_kv (matmul outputs,
        // attn_output, FFN intermediates) ALL go through the captured
        // graph with CAPTURE-time addresses. scatter_set was just one
        // example. The proper fix is per-layer persistent intermediate
        // buffers (graph_k_proj_buffer, graph_v_proj_buffer,
        // graph_attn_out_buffer, graph_ffn_up_buffer, ...). Larger
        // refactor; gated off for now while the infra stays in tree.
        // Parallel-attention layers (phi2) change the graph's structure from one token to
        // the next, which a replay cannot patch, and their capture faulted on an illegal
        // address. Every other layout captures, checked by the probation replays.
        if self.config.flags.parallel_attn {
            return Err("parallel-attention layers change the graph between tokens");
        }
        Ok(())
    }

    /// Boolean form, for callers that only need the verdict.
    pub fn graph_capture_auto_on(&self) -> bool {
        self.graph_capture_decision().is_ok()
    }

    /// Say out loud whether decode will capture, and why not when it will not. Without a
    /// model can sit in the allow-list and be refused for an unrelated reason - a KV dtype,
    /// a second card - with nothing in the log to show for it, which is how the cost of
    /// running uncaptured stayed invisible: 717 kernel launches per token, the GPU busy 7.5%
    /// of the decode window.
    pub fn log_graph_capture_decision(&self) {
        match self.graph_capture_decision() {
            Ok(()) => tracing::info!("🎞️  CUDA graph capture: ON (arch={})", self.config.arch),
            Err(why) => tracing::info!(
                "🎞️  CUDA graph capture: OFF (arch={}) - {why}",
                self.config.arch
            ),
        }
    }

    /// Whether the engine should re-capture the decode graph on every
    /// replay (vs the cheaper capture-once-replay-many default).
    ///
    /// the recapture-forward's output
    /// tensor becomes the new logits_ref (was: stale warmup logits).
    /// Replay loop now:
    ///   1. begin_capture
    ///   2. forward -> records kernels into graph; output tensor =
    ///      new logits_ref binding to the new capture's output
    ///   3. end_capture_and_update -> cheap pointer patch
    ///   4. launch -> actually executes
    ///   5. sync + read logits_ref (now valid)
    ///
    /// Pattern from llama.cpp ggml-cuda.cu:4400-4430.
    pub fn recapture_each_token(&self) -> bool {
        // A bisect: with the engine-side update_graph_state +
        // embed_for_graph running BEFORE begin_capture, the same
        // captured graph can be replayed each token - the input buffer
        // (graph_hidden_buffer) and per-layer state (cur_pos_dev,
        // padded_mask) live at stable addresses; only their VALUES
        // change per token. Recapture-every-token was needed when those
        // updates happened inline inside the captured forward. Now
        // they're hoisted out. Per-token recapture costs ~50-200 µs of
        // instantiate overhead per the new end_capture_or_reinstantiate
        // path - wasted if the initial capture works.
        //
        // Was: self.config.arch == "phi2" (per / history).
        // Probe with all 17 buffers wired: 2493 captured nodes still
        // ILLEGAL_ADDRESS. Attention chain has MORE fresh-alloc sites
        // than initially mapped - possibly reshape/transpose views or
        // workspace allocations internal to the substrate. Each probe reveals
        // another site. Gate kept off pending sustained focus session.
        //
        // A final probe: even with recapture, graph.launch
        // crashes - captured-time pointers go stale between end_capture and
        // launch due to Rust Tensor drops freeing memory the pool doesn't
        // know to hold. Need Option F (cuGraphExecKernelNodeSetParams)
        // pre-replay pointer sweep. Multi-day, not for cron-tick scope.
        false
    }

    /// Compute layer-consolidation groups for graph-mode state buffers.
    /// Derived from immutable layer config (device, n_kv_head, head_dim,
    /// max_seq_len_padded, rope tensor storage identity) so the result
    /// can be cached on the model and reused every token.
    pub(super) fn compute_graph_state_groups(&self) -> GraphStateGroups {
        use std::collections::HashMap;
        let n_layers = self.layers.len();
        let device_id = |i: usize| -> usize {
            match self.layers[i].cos.device().location() {
                crate::tensor::DeviceLocation::Cpu => 0usize,
                crate::tensor::DeviceLocation::Cuda { gpu_id } => 1 + gpu_id,
            }
        };
        let mask: Vec<usize> = {
            let mut leaders: HashMap<(usize, usize), usize> = HashMap::new();
            (0..n_layers)
                .map(|i| {
                    if !self.layer_captured(i) {
                        return i;
                    }
                    let key = (device_id(i), self.layers[i].kv_cache.max_seq_len_padded());
                    *leaders.entry(key).or_insert(i)
                })
                .collect()
        };
        let kv_pos: Vec<usize> = {
            let mut leaders: HashMap<(usize, usize, usize), usize> = HashMap::new();
            (0..n_layers)
                .map(|i| {
                    if !self.layer_captured(i) {
                        return i;
                    }
                    let l = &self.layers[i];
                    let key = (device_id(i), l.n_kv_head, l.head_dim);
                    *leaders.entry(key).or_insert(i)
                })
                .collect()
        };
        let rope: Vec<usize> = {
            let mut leaders: Vec<usize> = Vec::with_capacity(n_layers);
            for i in 0..n_layers {
                let mine = &self.layers[i];
                let mut leader = i;
                for &j in leaders.iter() {
                    // An eager layer shares its buffers with no captured one.
                    if self.layer_captured(i) != self.layer_captured(j) {
                        continue;
                    }
                    let other = &self.layers[j];
                    if mine.cos.shares_storage(&other.cos) && mine.sin.shares_storage(&other.sin) {
                        leader = j;
                        break;
                    }
                }
                leaders.push(leader);
            }
            leaders
        };
        GraphStateGroups { mask, kv_pos, rope }
    }

    /// Drop all per-layer + model graph state buffers. After this call,
    /// `forward_attn` sees `graph_rope_cos.is_none()` and routes through
    /// the standard non-graph path. The engine calls this when it
    /// observes a stale captured graph (e.g. F-dtype kv_cache outgrew
    /// the captured mask cap) so subsequent decode steps don't crash on
    /// orphaned device pointers.
    /// Whether the recording `what` (0 the decode step, `r` the verify of `r` rows) passed
    /// every check once with this model.
    pub fn graph_proven(&self, what: usize) -> bool {
        what < 64 && self.graph_proven & (1u64 << what) != 0
    }

    /// The recording `what` passed every check.
    pub fn mark_graph_proven(&mut self, what: usize) {
        if what < 64 {
            self.graph_proven |= 1u64 << what;
        }
    }

    /// Where the buffers the recorded graphs read stand: the explicit drops of the model's
    /// graph state and moves of its expert slots, with every time a layer's cache, indexer or
    /// carried state was made anew. Equal values mean no buffer a graph recorded at the first
    /// could read has moved since.
    pub fn graph_generation(&self) -> u64 {
        self.layers.iter().fold(self.graph_generation, |g, l| {
            g.wrapping_add(l.buffer_moves())
        })
    }

    /// This model's [`GenericHeteroTransformer::graph_instance`].
    pub fn graph_instance(&self) -> u64 {
        self.graph_instance
    }

    /// Releases the quantised caches' capture bookkeeping while graphs kept from an earlier
    /// request wait for their next replay: a prefill may grow the caches, which moves the
    /// generation and drops the graphs.
    pub fn suspend_graph_captured(&mut self) {
        #[cfg(feature = "gpu")]
        for l in &mut self.layers {
            let shards = l.tp_shards.iter_mut().filter_map(|s| s.quant_kv.as_mut());
            for c in l.quant_kv.as_mut().into_iter().chain(shards) {
                c.suspend_graph_captured();
            }
        }
    }

    /// Graphs kept from an earlier request are live again over the quantised caches, with the
    /// ceilings they were recorded with.
    #[cfg(feature = "gpu")]
    pub fn resume_graph_captured(&mut self) {
        let captured: Vec<bool> = (0..self.layers.len())
            .map(|i| self.layer_captured(i))
            .collect();
        for (l, captured) in self.layers.iter_mut().zip(captured) {
            if !captured {
                continue;
            }
            let shards = l.tp_shards.iter_mut().filter_map(|s| s.quant_kv.as_mut());
            for c in l.quant_kv.as_mut().into_iter().chain(shards) {
                c.resume_graph_captured();
            }
        }
    }

    pub fn invalidate_graph_state(&mut self) {
        self.graph_generation = self.graph_generation.wrapping_add(1);
        for l in &mut self.layers {
            clear_layer_graph_state(l);
        }
        self.graph_logits_buffer = None;
        self.graph_hidden_buffer = None;
        self.graph_work_buffer = None;
        self.graph_feature = None;
        #[cfg(feature = "gpu")]
        self.clear_verify_state();
        // ROOT CAUSE FIX: drop model-level captured-region tensor handles
        // so cuMemFree can reclaim their memory now that graph is torn down.
        self.graph_alive_tensors_model.clear();
        // Release the quantised caches' capture bookkeeping: with the graph gone the
        // buffers may grow again and the frozen seq ceiling no longer applies.
        #[cfg(feature = "gpu")]
        for l in &mut self.layers {
            let shards = l.tp_shards.iter_mut().filter_map(|s| s.quant_kv.as_mut());
            for c in l.quant_kv.as_mut().into_iter().chain(shards) {
                c.clear_graph_captured();
            }
        }
        // The memory the dropped graphs reserved goes back to the card: a request's graphs
        // are recorded again, and the reserve of every request before would stay otherwise.
        #[cfg(feature = "gpu")]
        for dev in self.cuda_devices.values() {
            if let Err(e) = crate::tensor::cuda_ext::stream_of(dev)
                .and_then(|s| crate::tensor::cuda_ext::graph_mem_trim(&s))
            {
                tracing::warn!("graph memory not trimmed: {e}");
            }
        }
    }

    /// Engine hook: a CUDA graph referencing the per-layer KV buffers is
    /// now live. Freezes each quantised cache's growth + seq ceiling (see
    /// `KvCore::mark_captured`). Paired with `invalidate_graph_state`
    /// which clears it on teardown.
    #[cfg(feature = "gpu")]
    pub fn mark_graph_captured(&mut self) {
        let captured: Vec<bool> = (0..self.layers.len())
            .map(|i| self.layer_captured(i))
            .collect();
        for (l, captured) in self.layers.iter_mut().zip(captured) {
            if !captured {
                continue;
            }
            if let Some(c) = l.quant_kv.as_mut() {
                c.mark_graph_captured();
            }
            for c in l.tp_shards.iter_mut().filter_map(|s| s.quant_kv.as_mut()) {
                c.mark_graph_captured();
            }
        }
    }

    /// Engine hook, called after every replayed decode token with the
    /// post-token KV length (`pos` after the engine's increment). Captured
    /// graphs advance the quantized KV caches purely device-side (the
    /// dev_slot append kernel reads `cur_pos_dev`), so without this the
    /// host `current_seq_len` staled for the whole replayed span and the
    /// NEXT request's trim/prefix-reuse/prefill operated on wrong lengths
    /// (server repro: request 2+ broadcast-shape 500s or silent
    /// garbage). O(n_layers) host-only work.
    #[cfg(feature = "gpu")]
    pub fn sync_kv_len_for_graph(&mut self, len: usize) {
        let captured: Vec<bool> = (0..self.layers.len())
            .map(|i| self.layer_captured(i))
            .collect();
        for (l, captured) in self.layers.iter_mut().zip(captured) {
            for c in l.tp_shards.iter_mut().filter_map(|s| s.quant_kv.as_mut()) {
                c.set_seq_len_for_graph(len);
            }
            match l.quant_kv.as_mut() {
                Some(c) => c.set_seq_len_for_graph(len),
                // An F-dtype cache the graph writes by device position (a sink layer's, an
                // indexed layer's, which keeps its indexer on the card as well).
                None if l.attn_sinks.is_some() || (captured && l.sparse.is_some()) => {
                    l.kv_cache.sync_len(len);
                    if let Some(sparse) = l.sparse.as_mut() {
                        sparse.card_confirm(len);
                    }
                }
                None => {}
            }
        }
    }

    /// What the recorded step of each indexed layer reads at `pos`: the indexer's cache on the
    /// card holds the tokens before it, the cache has room for the token and keeps the buffers
    /// the step was recorded against. A cache that outgrew those ends the step.
    #[cfg(feature = "gpu")]
    fn prepare_indexed_layers(&mut self, pos: usize) -> Result<()> {
        for i in 0..self.layers.len() {
            if !self.layer_captured(i) || self.layers[i].sparse.is_none() {
                continue;
            }
            let layer = &mut self.layers[i];
            let cap = layer.kv_cache.max_seq_len_padded();
            if let Some(recorded) = layer.graph_kv_cap {
                if recorded != cap || pos >= recorded {
                    return Err(crate::tensor::Error::msg(format!(
                        "graph_state: pos {pos} outgrew the cache a recorded step reads, {recorded} \
                         tokens (layer {i}). Caller should invalidate the captured graph."
                    )));
                }
            } else if pos >= cap {
                layer.kv_cache.reserve(pos + 1)?;
            }
            let room = layer.kv_cache.max_seq_len_padded();
            if let Some(sparse) = layer.sparse.as_mut() {
                sparse.card_room_for(room)?;
                sparse.card_hold(pos)?;
            }
        }
        Ok(())
    }

    /// The workspaces a recorded forward reads, grown once before any capture: a workspace that
    /// grew inside a capture would move under the kernels recorded against it.
    pub(super) fn ensure_graph_workspaces(&mut self) {
        if !self.workspace_pre_grown {
            self.workspace_pre_grown = true;
            #[cfg(feature = "gpu")]
            for cuda_dev in self.cuda_devices.values() {
                if let Ok(cd) = cuda_dev.as_cuda_device() {
                    let max_intermediate = self
                        .layers
                        .iter()
                        .map(|l| l.flags.intermediate_size)
                        .max()
                        .unwrap_or(self.config.embedding_length);
                    let max_k = std::cmp::max(self.stream_width(), max_intermediate);
                    let max_scratch =
                        crate::tensor::quantized::fast_mmvq::max_scratch_bytes_for_k(max_k);
                    if let Err(e) = crate::tensor::quantized::fast_mmvq::ensure_workspace_capacity(
                        &cd,
                        max_scratch,
                    ) {
                        tracing::warn!("fast_mmvq workspace pre-allocation failed: {e}");
                    } else {
                        tracing::info!(
                            "fast_mmvq workspace pre-allocated: {max_scratch} bytes (max_k={max_k})"
                        );
                    }
                }
                // Pin a dedicated cuBLAS workspace on the MODEL's handle so
                // cuBLAS stops allocating per-call workspaces - an
                // in-capture cuBLAS alloc invalidates the graph capture
                // (same reason gptoss pins before its capture). Once per
                // model (the workspace leaks by design, see
                // pin_cublas_workspace). update_graph_state only runs on
                // the graph path, so non-graph models never pay this.
                if let Err(e) = crate::tensor::cuda_ext::pin_cublas_workspace(cuda_dev, 64 << 20) {
                    tracing::warn!(
                        "graph: cuBLAS workspace pin failed (capture may be invalidated): {e}"
                    );
                }
            }
        }
    }

    pub fn update_graph_state(&mut self, pos: usize) -> Result<()> {
        self.spec_clear();
        // Captured-graph seq ceiling (Q8 path): the live graph was
        // recorded against a frozen buffer capacity and (if the
        // fixed-stride score chain was captured) a frozen max_seq_padded
        // stride. Once `pos + 1` outgrows it, error out so the engine
        // invalidates the graph - the request then continues on the
        // eager path (or re-captures at the wider state).
        self.graph_pos = pos;
        #[cfg(feature = "gpu")]
        for (i, l) in self.layers.iter().enumerate() {
            if !self.layer_captured(i) {
                continue;
            }
            let limit = l.quant_kv.as_ref().and_then(|c| c.graph_seq_limit());
            if let Some(limit) = limit {
                if pos + 1 > limit {
                    return Err(crate::tensor::Error::msg(format!(
                        "graph_state: pos {} outgrew captured KV seq ceiling {} (layer {}). \
                         Caller should invalidate the captured graph.",
                        pos, limit, i
                    )));
                }
            }
        }

        // Fix: pre-grow fast_mmvq's Q8_1 scratch
        // workspace UNCONDITIONALLY on the first update_graph_state call.
        // Without this, the first captured matmul fixes a small workspace,
        // then a LATER captured matmul with bigger k triggers
        // workspace_ensure to realloc, invalidating earlier captured
        // kernels' scratch pointers -> ILLEGAL_ADDRESS on replay.
        #[cfg(feature = "gpu")]
        self.ensure_graph_hops()?;
        #[cfg(feature = "gpu")]
        self.prepare_indexed_layers(pos)?;
        // A recorded n-gram layer reads and writes its convolution's rows where they lie.
        if let Some(layer) = self.ngram.as_ref().map(|n| n.meta.layer) {
            if self.layer_captured(layer) {
                let dev = self.layers[layer].cos.device();
                if let Some(ple) = self.layers[layer].ngram.as_mut() {
                    ple.ensure_state(&dev)?;
                }
            }
        }
        self.ensure_graph_workspaces();

        // Buffer consolidation: under graph capture, ~150 redundant
        // copy2d_f32 ops/token came from per-layer slice_sets in
        // update_graph_state writing IDENTICAL data into per-layer buffers.
        // For homogeneous models (deepcoder: 48 uniform layers, qwen3,
        // etc.) we share ONE backing tensor per group via Arc-clone, so
        // the per-token slice_set fires once per GROUP, not per layer.
        //
        // Group key for padded_mask:  (device_id, max_kv)
        // Group key for graph_kv_pos: (device_id, n_kv_head, head_dim)
        // Group key for rope cos/sin: source-tensor storage identity
        //   (`Tensor::shares_storage`), conservative - different RoPE
        //   periods (gemma4:latest SWA vs global) get separate buffers.
        //
        // First pass: compute group -> leader-layer-index maps. Cached on
        // the model - layer config is immutable after load, so we only
        // rebuild on first call.
        let n_layers = self.layers.len();
        if self.graph_state_groups.is_none() {
            self.graph_state_groups = Some(self.compute_graph_state_groups());
        }
        // Borrow-check: clone the small Vec<usize>s so we can mutate layers
        // below. ~3 x n_layers usize allocations per token (small, hot in
        // cache; cheaper than the prior HashMap construction).
        let groups = self.graph_state_groups.as_ref().unwrap().clone();
        let mask_group = groups.mask;
        let kv_pos_group = groups.kv_pos;
        let rope_group = groups.rope;

        // Quantized-KV layers never read `padded_mask` or `graph_kv_pos`
        // on the graph path (their append + attention run through the
        // cur_pos_dev kernels), so skip those per-token updates when the
        // WHOLE model is quantized-KV without dual-populate donors - for
        // an all-Q8 model (qwen3:0.6b) this drops a mask slice_set launch
        // and an ~8 KB graph_kv_pos H2D per token. All-or-nothing so a
        // mixed placement (one layer's Q8 alloc failed -> F-dtype) never
        // sees a skipped group leader.
        let skip_fdtype_state = self.captured_layers().all(|l| {
            !l.attends()
                || l.sparse.is_some()
                || ((l.has_q8_cache() || l.has_q4_cache()) && !l.populate_dual_kv)
        });

        // The step's position for the sink layers' device-length attention: written once
        // per card, shared by that card's sink layers.
        let mut pos_by_card: std::collections::HashMap<usize, Tensor> =
            std::collections::HashMap::new();
        // Second pass: update each leader, then assign Arc-clones to followers.
        for i in 0..n_layers {
            // An eager layer reads none of the step's buffers.
            if !self.layer_captured(i) {
                continue;
            }
            let leaders = GraphStateLeaders {
                rope: rope_group[i] == i,
                mask: mask_group[i] == i,
                kv_pos: kv_pos_group[i] == i,
            };
            update_layer_graph_state(
                &mut self.layers[i],
                i,
                pos,
                leaders,
                skip_fdtype_state,
                &mut pos_by_card,
            )?;
            // The shares of a split layer each read buffers of their own on their card.
            let mut shards = std::mem::take(&mut self.layers[i].tp_shards);
            let updated = shards.iter_mut().try_for_each(|shard| {
                update_layer_graph_state(
                    shard,
                    i,
                    pos,
                    GraphStateLeaders::OWN,
                    skip_fdtype_state,
                    &mut pos_by_card,
                )
            });
            self.layers[i].tp_shards = shards;
            updated?;
        }

        // Third pass: followers Arc-clone the group leader's tensors. This
        // makes each layer's `padded_mask` / `graph_kv_pos` / rope buffers
        // point at the SAME underlying CudaSlice as the leader, so subsequent
        // slice_sets by the leader are visible to all followers and the
        // captured graph reads from one shared device pointer per group.
        for i in 0..n_layers {
            if !self.layer_captured(i) {
                continue;
            }
            if !skip_fdtype_state && mask_group[i] != i {
                let leader_mask = self.layers[mask_group[i]].padded_mask.clone();
                self.layers[i].padded_mask = leader_mask;
            }
            if !skip_fdtype_state && kv_pos_group[i] != i {
                let leader_kv = self.layers[kv_pos_group[i]].graph_kv_pos.clone();
                self.layers[i].graph_kv_pos = leader_kv;
            }
            if rope_group[i] != i {
                let leader = rope_group[i];
                let lcos = self.layers[leader].graph_rope_cos.clone();
                let lsin = self.layers[leader].graph_rope_sin.clone();
                self.layers[i].graph_rope_cos = lcos;
                self.layers[i].graph_rope_sin = lsin;
            }
        }

        Ok(())
    }

    /// Compute the token embedding and write it into graph_hidden_buffer at
    /// a stable device pointer. Call OUTSIDE graph capture so the embedding
    /// lookup + H2D copy are not part of the captured graph.
    pub fn embed_for_graph(&mut self, input_ids: &Tensor) -> Result<()> {
        self.spec_clear();
        let emb_device = self.embeddings.embeddings().device().clone();
        let input_local;
        let input_ids = if input_ids.device().same_device(&emb_device) {
            input_ids
        } else {
            input_local = input_ids.to_device(&emb_device)?;
            &input_local
        };
        let first_card = self
            .layers
            .first()
            .map(|l| l.cos.device().clone())
            .filter(|d| d.is_cuda());
        let mut hidden = match first_card.as_ref().and_then(|d| self.embed_table_on(d)) {
            Some(table) => self.embed_rows_on_card(table, input_ids)?,
            None => self.embeddings.forward(input_ids)?,
        };
        if let Some(scale) = self.config.embed_scale {
            hidden = (hidden * scale)?;
        }
        if let Some(h) = self.config.hyper {
            hidden = super::wide::widen_embedding(&hidden, h.streams)?;
        }
        // Land hidden on the device that owns layer 0 - that's where
        // forward_from_hidden starts iterating and any earlier device
        // (e.g. `cuda_devices.values().next()` returning a non-first
        // GPU under HashMap iteration order) would cost an extra
        // cross-device copy on the very first attention. layer 0's
        // `cos` tensor lives on the layer's compute device and is the
        // canonical placement source-of-truth elsewhere in the engine.
        let target_dev = self
            .layers
            .first()
            .map(|l| l.cos.device().clone())
            .filter(|d| d.is_cuda())
            .or_else(|| self.cuda_devices.values().next().cloned())
            .ok_or_else(|| crate::tensor::Error::msg("embed_for_graph needs CUDA"))?;
        let hidden_gpu = hidden.to_device(&target_dev)?;
        let input = self.ensure_hidden_buffer(&hidden_gpu)?;
        if self.graph_work_buffer.is_none() {
            self.graph_work_buffer = Some(Tensor::zeros_on(
                input.dims(),
                input.dtype(),
                &input.device(),
            )?);
        }
        self.place_ngram_rows(input_ids)?;
        Ok(())
    }

    /// The n-gram rows of `input_ids`, the step's token, in the buffer of the layer that adds
    /// them when that layer is recorded: its replay reads them from there.
    /// The embedding table on `dev`, when the card holds one.
    pub(super) fn embed_table_on(
        &self,
        dev: &Device,
    ) -> Option<&crate::tensor::quantized::QTensor> {
        self.embed_cards
            .iter()
            .map(|t| t.as_ref())
            .find(|t| t.device().same_device(dev))
    }

    /// The rows of `input_ids` (`[1, n]`) decoded from the table on the card, `[1, n, dim]` F32.
    #[cfg(feature = "gpu")]
    pub(super) fn embed_rows_on_card(
        &self,
        table: &crate::tensor::quantized::QTensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let dev = table.device().clone();
        let n = input_ids.elem_count();
        let dim = table.shape().dims()[1];
        let ids = input_ids
            .flatten_all()?
            .to_dtype(crate::tensor::DType::U32)?
            .to_device(&dev)?;
        let out = Tensor::zeros_on((1, n, dim), crate::tensor::DType::F32, &dev)?;
        crate::inference::moe_cuda::embed_rows_gguf(table, &ids, &out)?;
        Ok(out)
    }

    #[cfg(not(feature = "gpu"))]
    pub(super) fn embed_rows_on_card(
        &self,
        _table: &crate::tensor::quantized::QTensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        self.embeddings.forward(input_ids)
    }

    fn place_ngram_rows(&mut self, input_ids: &Tensor) -> Result<()> {
        let Some(layer) = self.ngram.as_ref().map(|n| n.meta.layer) else {
            return Ok(());
        };
        self.gather_ngram_rows(input_ids, self.graph_pos)?;
        if !self.layer_captured(layer) {
            return Ok(());
        }
        let rows = self
            .ngram
            .as_ref()
            .and_then(|n| n.pending.clone())
            .ok_or_else(|| crate::tensor::Error::msg("the n-gram rows were not gathered"))?;
        let held = &mut self.layers[layer];
        let rows = rows.to_device(&held.cos.device())?;
        match held.graph_ngram_rows.as_ref() {
            Some(buffer) => buffer.slice_set(&rows, 0, 0)?,
            None => held.graph_ngram_rows = Some(rows.force_contiguous()?),
        }
        Ok(())
    }

    /// Populate per-layer `graph_ple_input_buffer` slots from a given
    /// `input_ids` tensor (typically `[1]` for single-token decode in
    /// graph mode). Must be called BEFORE `begin_capture` so the
    /// PLE-table lookup + projection + per-layer split runs OUTSIDE the
    /// captured graph; the captured graph then reads from the stable
    /// per-layer buffers.
    ///
    /// No-op for non-gemma4 models (ple_dim == 0). Reads model-level
    /// ple_token_embd / ple_model_proj / ple_proj_norm; writes a
    /// `[1, 1, ple_dim]` F32 tensor into each layer's graph_ple_input_buffer.
    ///
    /// Mirrors the per-layer PLE computation in
    /// `forward_inner` at lines 5450-5538.
    pub fn populate_ple_input_buffers(&mut self, input_ids: &Tensor) -> Result<()> {
        if self.ple_dim == 0 {
            return Ok(());
        }
        let model_proj = match &self.ple_model_proj {
            Some(p) => p,
            None => return Ok(()),
        };
        let proj_norm = match &self.ple_proj_norm {
            Some(n) => n,
            None => return Ok(()),
        };
        let n_layers = self.layers.len();
        let ple_scale = (self.ple_dim as f64).sqrt();

        // Token embeddings via PLE table.
        let tok_ple = if let Some(embd) = &self.ple_token_embd {
            let ple_device = embd.embeddings().device().clone();
            let input_local;
            let input_ids_ple = if input_ids.device().same_device(&ple_device) {
                input_ids
            } else {
                input_local = input_ids.to_device(&ple_device)?;
                &input_local
            };
            let t = embd.forward(input_ids_ple)?;
            let t = if t.dtype() != crate::tensor::DType::F32 {
                t.to_dtype(crate::tensor::DType::F32)?
            } else {
                t
            };
            Some((t * ple_scale)?)
        } else {
            None
        };

        // Model projection: hidden (graph_hidden_buffer) -> per-layer PLE.
        let hidden = self
            .graph_hidden_buffer
            .as_ref()
            .ok_or_else(|| {
                crate::tensor::Error::msg(
                    "populate_ple_input_buffers: graph_hidden_buffer not populated",
                )
            })?
            .clone();
        let (b, seq_h, _) = hidden.dims3()?;
        let proj_dev = model_proj.device();
        let hidden_on_proj = if hidden.device().same_device(&proj_dev) {
            hidden.clone()
        } else {
            hidden.to_device(&proj_dev)?
        };
        let hidden_2d = hidden_on_proj.reshape((b * seq_h, self.config.embedding_length))?;
        let model_ple = model_proj.forward(&hidden_2d)?;
        let model_ple = (model_ple / (self.config.embedding_length as f64).sqrt())?;
        let model_ple = model_ple.reshape((b * seq_h, n_layers, self.ple_dim))?;
        let model_ple = proj_norm.forward(&model_ple)?;

        let ple = if let Some(tok) = tok_ple {
            let tok = tok.reshape((b * seq_h, n_layers, self.ple_dim))?;
            let tok = if tok.device().same_device(&model_ple.device()) {
                tok
            } else {
                tok.to_device(&model_ple.device())?
            };
            ((model_ple + tok)? * (1.0 / 2f64.sqrt()))?
        } else {
            model_ple
        };

        // Split per-layer + slice_set into each layer's stable buffer.
        for i in 0..n_layers {
            if let Some(buf) = self.layers[i].graph_ple_input_buffer.as_ref() {
                let layer_ple = ple.i((.., i, ..))?.reshape((b, seq_h, self.ple_dim))?;
                // Move to buffer's device (might differ from proj device
                // under multi-GPU placement).
                let layer_ple = if layer_ple.device().same_device(&buf.device()) {
                    layer_ple
                } else {
                    layer_ple.to_device(&buf.device())?
                };
                buf.slice_set(&layer_ple, 0, 0)?;
            }
        }
        Ok(())
    }

    /// Run layers + output projection from graph_hidden_buffer. The buffer
    /// must have been populated by embed_for_graph before calling this.
    /// Every op of the step runs in order, eagerly: what the capture records, one graph per
    /// captured op, is this same sequence.
    pub fn forward_from_hidden(&mut self) -> Result<Tensor> {
        self.rewind_eager_layers();
        let n = self.step_plan().len();
        let mut logits = None;
        for op in 0..n {
            if let Some(out) = self.run_step_op(op)? {
                logits = Some(out);
            }
        }
        logits.ok_or_else(|| crate::tensor::Error::msg("forward_from_hidden: no op ran"))
    }

    /// The decode step as a sequence of ops: the captured layers of each card in runs, the
    /// layers that read the host between kernels one by one, the hop where the step crosses
    /// from one card to the next, and the head. Empty when a layer is on the host.
    pub fn step_plan(&self) -> Vec<StepOp> {
        let segs: Vec<std::ops::Range<usize>> =
            self.graph_segments().into_iter().map(|(r, _)| r).collect();
        plan_step(&segs, |layer| self.layer_captured(layer))
    }

    /// For each op of the step: whether it is recorded in a graph, and the stream of the card
    /// it runs on.
    #[cfg(feature = "gpu")]
    pub fn step_streams(&self) -> Vec<(bool, std::sync::Arc<crate::tensor::cuda_ext::CudaStream>)> {
        let streams = self.graph_segment_streams();
        self.step_plan()
            .iter()
            .filter_map(|op| {
                let seg = match op {
                    StepOp::Hop { seg, .. } => seg + 1,
                    StepOp::Run { seg, .. } | StepOp::Head { seg } | StepOp::Layer { seg, .. } => {
                        *seg
                    }
                };
                streams.get(seg).map(|s| (op.captured(), s.clone()))
            })
            .collect()
    }

    /// The hidden state a card segment's ops read and write in place: the first card's working
    /// copy, the hop's landing buffer for the others.
    fn step_buffer(&self, seg: usize) -> Result<Tensor> {
        if seg == 0 {
            return self.graph_work_buffer.clone().ok_or_else(|| {
                crate::tensor::Error::msg("forward_from_hidden: call embed_for_graph first")
            });
        }
        #[cfg(feature = "gpu")]
        {
            self.graph_hops
                .get(seg - 1)
                .map(|hop| hop.dst.clone())
                .ok_or_else(|| crate::tensor::Error::msg("graph hop not prepared"))
        }
        #[cfg(not(feature = "gpu"))]
        crate::tensor::bail!("a card hop needs CUDA")
    }

    /// Op `op` of the step. A captured op is what a recording runs, and runs as it is when
    /// the step is eager; an eager op runs at every replay. The head's op returns the logits.
    pub fn run_step_op(&mut self, op: usize) -> Result<Option<Tensor>> {
        let Some(this) = self.step_plan().into_iter().nth(op) else {
            crate::tensor::bail!("run_step_op: no op {op}");
        };
        match this {
            StepOp::Run {
                seg,
                layers,
                head,
                recv,
                send,
            } => {
                let _step = GraphStep::enter();
                let buffer = self.step_buffer(seg)?;
                #[cfg(feature = "gpu")]
                if recv {
                    self.graph_hops
                        .get(seg - 1)
                        .ok_or_else(|| crate::tensor::Error::msg("graph hop not prepared"))?
                        .receive()?;
                }
                let mut hidden = self.step_input(op, seg)?;
                // A wide layer followed by another in the run hands it the norm its token mixer
                // starts from, taken in its last write-back.
                #[cfg(feature = "gpu")]
                let mut xn: Option<Tensor> = None;
                // See the timer in the eager decode loop: this measures issuing the layer,
                // which is what tells a dispatching layer from one running on the host.
                for li in layers.clone() {
                    let _timer = crate::inference::place::layer_perf::LayerTimer::start(li);
                    #[cfg(feature = "gpu")]
                    if self.layers[li].is_wide() {
                        let next = (li + 1 < layers.end)
                            .then(|| self.layers[li + 1].chained_attn_norm())
                            .flatten()
                            .filter(|_| self.layers[li + 1].is_wide());
                        let (h, x) =
                            self.layers[li].forward_wide_graph_chained(&hidden, xn.take(), next)?;
                        hidden = h;
                        xn = x;
                        continue;
                    }
                    hidden = self.layers[li].forward_graph(&hidden)?;
                }
                self.tp_forget()?;
                if head {
                    return self.graph_head(hidden).map(Some);
                }
                #[cfg(feature = "gpu")]
                if send {
                    self.graph_hops
                        .get(seg)
                        .ok_or_else(|| crate::tensor::Error::msg("graph hop not prepared"))?
                        .send(&hidden)?;
                    return Ok(None);
                }
                let _ = (recv, send);
                Self::store_step_hidden(&buffer, hidden)?;
                Ok(None)
            }
            StepOp::Head { seg } => {
                let _step = GraphStep::enter();
                let hidden = self.step_input(op, seg)?;
                self.graph_head(hidden).map(Some)
            }
            StepOp::Layer { seg, layer } => {
                let buffer = self.step_buffer(seg)?;
                let input = self.step_input(op, seg)?;
                // The gathered n-gram rows go to the one layer that adds them.
                let ngram = self.layers[layer]
                    .ngram
                    .is_some()
                    .then(|| self.ngram.as_ref().and_then(|n| n.pending.clone()))
                    .flatten();
                let pos = self.graph_pos;
                let _timer = crate::inference::place::layer_perf::LayerTimer::start(layer);
                let hidden = self.layers[layer].forward(
                    &input,
                    None,
                    pos,
                    ngram.as_ref(),
                    None,
                    None,
                    #[cfg(feature = "gpu")]
                    None,
                )?;
                self.tp_forget()?;
                Self::store_step_hidden(&buffer, hidden)?;
                Ok(None)
            }
            StepOp::Hop { seg, send, recv } => {
                #[cfg(feature = "gpu")]
                {
                    let hop = self
                        .graph_hops
                        .get(seg)
                        .ok_or_else(|| crate::tensor::Error::msg("graph hop not prepared"))?;
                    if send {
                        hop.send(&self.step_buffer(seg)?)?;
                    }
                    // The next card's stream waits for the copy to the host, then fills its
                    // own buffer from it.
                    let streams = self.graph_segment_streams();
                    if let (Some(a), Some(b)) = (streams.get(seg), streams.get(seg + 1)) {
                        let ev = crate::tensor::cuda_ext::record_event(a)?;
                        crate::tensor::cuda_ext::wait_event(b, &ev)?;
                    }
                    if recv {
                        hop.receive()?;
                    }
                    Ok(None)
                }
                #[cfg(not(feature = "gpu"))]
                {
                    let _ = (seg, send, recv);
                    crate::tensor::bail!("a card hop needs CUDA")
                }
            }
        }
    }

    /// What op `op` of the step reads: the embedding for the first, else the segment's buffer.
    fn step_input(&self, op: usize, seg: usize) -> Result<Tensor> {
        match (op, &self.graph_hidden_buffer) {
            (0, Some(embedding)) => Ok(embedding.clone()),
            _ => self.step_buffer(seg),
        }
    }

    /// `hidden` into the segment's buffer, which the next op reads.
    fn store_step_hidden(buffer: &Tensor, hidden: Tensor) -> Result<()> {
        let hidden = if hidden.dtype() == buffer.dtype() {
            hidden
        } else {
            hidden.to_dtype(buffer.dtype())?
        };
        buffer.slice_set(&hidden.contiguous()?, 0, 0)
    }

    /// Cut the caches of the layers the step runs eagerly back to the step's position: an
    /// eager step that follows a replayed one runs the layers whose host length the first
    /// advanced, which a captured layer's device position does not need.
    fn rewind_eager_layers(&mut self) {
        let pos = self.graph_pos;
        if pos == 0 {
            return;
        }
        for li in 0..self.layers.len() {
            if self.layer_captured(li) {
                continue;
            }
            let layer = &mut self.layers[li];
            layer.kv_cache.trim_to(pos);
            #[cfg(feature = "gpu")]
            if let Some(c) = layer.quant_kv.as_mut() {
                c.trim_to(pos);
            }
        }
    }

    /// Drops the copies of the hidden state the split layers kept on their other cards: the
    /// next pass starts from its own input.
    pub(super) fn tp_forget(&self) -> Result<()> {
        #[cfg(feature = "gpu")]
        if let Some(comms) = self.layers.iter().find_map(|l| l.tp_comms.as_ref()) {
            comms.forget()?;
        }
        Ok(())
    }

    /// The graph step's card segments in layer order: each run of layers on one card and
    /// that card. Empty when a layer is on the host.
    pub fn graph_segments(&self) -> Vec<(std::ops::Range<usize>, Device)> {
        let mut segs: Vec<(std::ops::Range<usize>, Device)> = Vec::new();
        let mut last: Option<usize> = None;
        for (i, d) in self.layer_devs.iter().enumerate() {
            let LayerDevice::Cuda(idx) = d else {
                return Vec::new();
            };
            match (last == Some(*idx), segs.last_mut()) {
                (true, Some((range, _))) => range.end = i + 1,
                _ => {
                    let Some(dev) = self.cuda_devices.get(idx) else {
                        return Vec::new();
                    };
                    segs.push((i..i + 1, dev.clone()));
                }
            }
            last = Some(*idx);
        }
        segs
    }

    /// The stream of each segment's card.
    #[cfg(feature = "gpu")]
    pub fn graph_segment_streams(
        &self,
    ) -> Vec<std::sync::Arc<crate::tensor::cuda_ext::CudaStream>> {
        let segs = self.graph_segments();
        if segs.len() <= 1 {
            return self.cuda_stream().into_iter().collect();
        }
        segs.iter()
            .filter_map(|(_, d)| crate::tensor::cuda_ext::stream_of(d).ok())
            .collect()
    }

    /// Allocates the hop buffers between consecutive cards, outside any capture.
    #[cfg(feature = "gpu")]
    fn ensure_graph_hops(&mut self) -> Result<()> {
        let segs = self.graph_segments();
        let boundaries = segs.len().saturating_sub(1);
        if self.graph_hops.len() == boundaries {
            return Ok(());
        }
        let width = self.stream_width();
        self.graph_hops = segs
            .iter()
            .skip(1)
            .map(|(_, dev)| GraphHop::new(width, dev))
            .collect::<Result<_>>()?;
        Ok(())
    }

    /// Hand the draft head the last layer's row of the step that just ran or replayed.
    pub fn step_feature(&mut self) {
        if self.capture_feature {
            self.last_feature = self.graph_feature.clone();
        }
    }

    /// Final norm, output projection and logit transforms of the graph's step, copied into
    /// the stable logits buffer.
    fn graph_head(&mut self, hidden: Tensor) -> Result<Tensor> {
        // The last layer's row, in a buffer of its own: what the draft head reads after a replay.
        // A row already in F32 is the layer's own buffer, so the copy gets a storage of its own.
        if self.graph_feature.is_none() {
            let row = hidden.to_dtype(crate::tensor::DType::F32)?;
            self.graph_feature = Some(row.zeros_like()?);
        }
        if let Some(buffer) = self.graph_feature.as_ref() {
            buffer.slice_set(
                &hidden.to_dtype(crate::tensor::DType::F32)?.contiguous()?,
                0,
                0,
            )?;
        }
        self.step_feature();
        let wide_normed: Option<Tensor> = match self.output_hyper.as_ref() {
            #[cfg(feature = "gpu")]
            Some(oh) => {
                let xn = oh.normed_on_card(&hidden)?;
                let (mixed, _) = oh.mix_on_card(&xn, false)?;
                Some(mixed.reshape((1, self.config.embedding_length))?)
            }
            #[cfg(not(feature = "gpu"))]
            Some(_) => crate::tensor::bail!("a wide residual's head runs on a card"),
            None => None,
        };
        let norm_w = match (&wide_normed, self.output_norm_cuda_weight.as_ref()) {
            (_, Some(w)) => Some(w),
            (Some(_), None) => None,
            (None, None) => {
                crate::tensor::bail!("forward_from_hidden needs output_norm_cuda_weight")
            }
        };
        let last = hidden.i((.., 0, ..))?;
        let last_f32 = last.to_dtype(crate::tensor::DType::F32)?;
        // Prefer the quantized output proj (Q6_K mvq, F32-out, no F16
        // cast) when available - same fast path the non-graph forward
        // takes. nsys traces showed the F16 cutlass fallback adds
        // ~1ms/token on deepcoder (152Kx5120 GEMM); the Q-quant mvq is
        // ~2.4x faster on the same shape.
        let mut logits = if self.output_proj_cuda_qmm.is_some() {
            let qmm = self.output_proj_cuda_qmm.as_ref().unwrap();
            // The norm and the head apart, as the eager decode and the verify rows run them.
            let normed = match (&wide_normed, norm_w) {
                (Some(n), _) => n.clone(),
                (None, Some(w)) => {
                    crate::tensor::ops::rms_norm(&last_f32, w, self.config.rms_norm_eps as f32)?
                }
                (None, None) => unreachable!("a norm was found above"),
            };
            qmm.forward(&normed)?
        } else {
            let proj = self.output_proj_cuda.as_ref().ok_or_else(|| {
                crate::tensor::Error::msg(
                    "forward_from_hidden needs output_proj_cuda or output_proj_cuda_qmm",
                )
            })?;
            let normed = match (&wide_normed, norm_w) {
                (Some(n), _) => n.clone(),
                (None, Some(w)) => {
                    crate::tensor::ops::rms_norm(&last_f32, w, self.config.rms_norm_eps as f32)?
                }
                (None, None) => unreachable!("a norm was found above"),
            };
            let normed_f16 = normed.to_dtype(crate::tensor::DType::F16)?;
            normed_f16.matmul_t(proj)?
        };
        logits = self.apply_output_bias(logits)?;
        logits = logits.to_dtype(crate::tensor::DType::F32)?;
        if let Some(scale) = self.config.logit_scale {
            logits = (logits / scale)?;
        }
        let logits = match self.config.final_logit_softcapping {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }?;

        // Copy logits into a stable-pointer buffer. CUDA graph instantiation
        // remaps internal allocations, so the `logits` tensor's address may
        // differ between capture and replay. By copying into a pre-allocated
        // buffer (outside the graph's memory pool), the engine can always
        // read the output from a fixed address after graph.launch().
        if self.graph_logits_buffer.is_none() {
            self.graph_logits_buffer = Some(logits.force_contiguous()?);
        } else {
            self.graph_logits_buffer
                .as_ref()
                .unwrap()
                .slice_set(&logits, 0, 0)?;
        }
        Ok(self.graph_logits_buffer.as_ref().unwrap().clone())
    }

    /// Context shift: drops positions `[from - discard, from)` of every layer's cache and
    /// pulls the tail down by `discard`, re-phasing the stored keys for their new
    /// positions. Refused before touching anything on a layer whose cache cannot move in
    /// place: a Q4 cache (a block groups 32 positions of one channel) or a sliding
    /// window (after a slide the buffer index is no longer the position). An `Err` from
    /// a cache leaves the layers in no common state; the caller resets.
    pub fn shift_kv_tail(&mut self, from: usize, discard: usize) -> crate::tensor::Result<()> {
        if self.is_recurrent() {
            return Err(crate::tensor::Error::msg(
                "context shift: a recurrent state cannot drop positions".to_string(),
            ));
        }
        for layer in &self.layers {
            if layer.sliding_window.is_some() {
                return Err(crate::tensor::Error::msg(
                    "context shift: sliding-window layer".to_string(),
                ));
            }
            #[cfg(feature = "gpu")]
            if layer.q4_kv().is_some() {
                return Err(crate::tensor::Error::msg(
                    "context shift: Q4 KV cache".to_string(),
                ));
            }
        }
        let delta = -(discard as i64);
        for layer in &mut self.layers {
            layer.shift_kv_tail(from, discard, delta)?;
        }
        Ok(())
    }

    /// Whether a layer's heads and feed-forward are split across cards.
    pub fn has_split_layers(&self) -> bool {
        self.layers.iter().any(|l| !l.tp_shards.is_empty())
    }

    /// Snapshots cover the F-dtype device cache and the host f16 cache. A quantised cache
    /// has no copy constructor, so a model built with one keeps a single resident KV.
    pub fn supports_kv_snapshots(&self) -> bool {
        // A quantised layer (Q8/Q4 on the card) is captured by dequantising to F32 and
        // restored by re-appending, which stays on the same quant grid. The host q8 cache
        // has no such round-trip and still bars snapshots. A recurrent state cannot be cut
        // to the prefix a KV snapshot is reused at; it is kept by `snapshot_prefix`.
        // A cache cut over cards has no snapshot form either: its prefix is read again. Nor
        // has the cache of a layer on a Vulkan device.
        self.layers
            .iter()
            .all(|l| l.cpu_q8_kv.is_none() && l.context.is_none())
            && !self.is_recurrent()
            && self.vulkan.is_empty()
            && self.opencl.is_empty()
    }

    /// Whether some layer's cache is cut over several cards.
    pub fn has_context_shards(&self) -> bool {
        self.layers.iter().any(|l| l.context.is_some())
    }

    /// Copies the resident KV aside under `tokens`, keeping at most `cap` snapshots: the
    /// one this sequence extends is replaced, else at capacity the least recently used.
    /// A copy that fails, for lack of memory, drops every snapshot and returns the error;
    /// the request that produced the KV is not at stake.
    pub fn snapshot_kv(&mut self, tokens: Vec<u32>, kv_len: usize) -> crate::tensor::Result<()> {
        // A recurrent model's table holds its prefix snapshots, which this leaves alone.
        if self.is_recurrent() {
            return Ok(());
        }
        if !self.supports_kv_snapshots() {
            self.kv_snapshots.clear();
            return Ok(());
        }
        if !self.room_for_snapshot(&tokens) {
            return Ok(());
        }
        let layers = match self.capture_layers() {
            Ok(layers) => layers,
            Err(e) => {
                self.kv_snapshots.clear();
                return Err(e);
            }
        };
        self.store_snapshot(tokens, kv_len, layers, None);
        Ok(())
    }

    /// Makes room on the cards for a snapshot of what the layers hold now, before it is
    /// copied: while the copy does not fit in the free memory less what the snapshot budget
    /// keeps aside, drops the snapshot `tokens` would replace, then the least recently used.
    /// False, with nothing copied, when it still does not fit once none is left.
    fn room_for_snapshot(&mut self, tokens: &[u32]) -> bool {
        let need: u64 = self.layers.iter().map(layer_snapshot_bytes).sum();
        let reserve = self.snapshot_reserve();
        loop {
            let free = self.free_device_memory();
            if snapshot_fits(need, free, reserve) {
                return true;
            }
            let existing: Vec<(&[u32], u64)> = self
                .kv_snapshots
                .iter()
                .map(|s| (s.tokens.as_slice(), s.rank()))
                .collect();
            let Some(victim) = snapshot_slot(&existing, tokens, existing.len()) else {
                tracing::debug!(
                    "kv snapshot skipped: {need} bytes do not fit in {} free bytes less {reserve} kept aside",
                    free.unwrap_or(0)
                );
                return false;
            };
            self.kv_snapshots.swap_remove(victim);
            crate::inference::place::ladder::release_pools();
        }
    }

    /// Every layer's KV, and every recurrent mixer's state, copied aside.
    fn capture_layers(&self) -> crate::tensor::Result<Vec<super::LayerKvSnapshot>> {
        self.layers.iter().map(capture_layer).collect()
    }

    /// Enters a snapshot in the table: the one this sequence extends is replaced, and the
    /// table is held to its memory budget.
    pub(super) fn store_snapshot(
        &mut self,
        tokens: Vec<u32>,
        kv_len: usize,
        layers: Vec<super::LayerKvSnapshot>,
        logits: Option<crate::tensor::Tensor>,
    ) {
        self.expire_snapshots();
        self.kv_snapshot_tick += 1;
        let draft = match self.capture_draft(kv_len) {
            Ok(d) => d,
            Err(e) => {
                tracing::debug!("kv snapshot: the draft head's state is not kept: {e}");
                None
            }
        };
        let mut snap = super::KvSnapshot {
            draft,
            tokens,
            kv_len,
            layers,
            tick: self.kv_snapshot_tick,
            logits,
            used: std::time::Instant::now(),
            hits: 0,
        };
        let existing: Vec<(&[u32], u64)> = self
            .kv_snapshots
            .iter()
            .map(|s| (s.tokens.as_slice(), s.rank()))
            .collect();
        match snapshot_slot(&existing, &snap.tokens, SNAPSHOT_COUNT_UNBOUNDED) {
            Some(i) => {
                // The conversation goes on in the snapshot that replaces its earlier turn.
                snap.hits = self.kv_snapshots[i].hits;
                self.kv_snapshots[i] = snap;
            }
            None => self.kv_snapshots.push(snap),
        }
        self.enforce_kv_snapshot_budget();
    }

    /// Every recurrent layer's state, copied, for `recurrent_restore` to put back: what lets
    /// a step be run twice from the same state. Empty for a model with no recurrent layer.
    pub fn recurrent_checkpoint(&self) -> Result<Vec<Option<super::mixer::MixerState>>> {
        if !self.is_recurrent() {
            return Ok(Vec::new());
        }
        self.layers.iter().map(|l| l.state_checkpoint()).collect()
    }

    /// Writes a checkpoint back into the buffers the states live in.
    pub fn recurrent_restore(
        &mut self,
        checkpoint: &[Option<super::mixer::MixerState>],
    ) -> Result<()> {
        for (l, saved) in self.layers.iter_mut().zip(checkpoint) {
            if let Some(state) = saved {
                l.state_restore_in_place(state)?;
            }
        }
        Ok(())
    }

    /// Makes a checkpoint the recurrent states again, wherever they live: on a card or on the
    /// host, in new buffers when the old ones differ.
    pub fn recurrent_rewind(
        &mut self,
        checkpoint: &[Option<super::mixer::MixerState>],
    ) -> Result<()> {
        for (l, saved) in self.layers.iter_mut().zip(checkpoint) {
            if let Some(state) = saved {
                l.state_restore(state)?;
            }
        }
        Ok(())
    }

    /// True when a layer carries a recurrent state beside its KV cache.
    pub fn is_recurrent(&self) -> bool {
        self.layers.iter().any(|l| l.mixer.is_recurrent())
    }

    /// A recurrent model's state after prefilling `prompt`, entered in the snapshot table
    /// with the prefill's last logits. A model without a recurrent layer keeps none: its KV
    /// is reused through `snapshot_kv` and `trim_kv`.
    pub fn snapshot_prefix(
        &mut self,
        prompt: &[u32],
        logits: &crate::tensor::Tensor,
    ) -> crate::tensor::Result<()> {
        self.keep_prefix_state(prompt, Some(logits)).map(|_| ())
    }

    /// A recurrent model's state after `prefix`, kept without logits for a later prompt that
    /// begins with `prefix` and goes on: the next turn of the conversation. True when it was
    /// kept; false for a model with no recurrent layer or when no card has the room.
    pub fn snapshot_boundary(&mut self, prefix: &[u32]) -> crate::tensor::Result<bool> {
        self.keep_prefix_state(prefix, None)
    }

    fn keep_prefix_state(
        &mut self,
        prompt: &[u32],
        logits: Option<&crate::tensor::Tensor>,
    ) -> crate::tensor::Result<bool> {
        if !self.is_recurrent() {
            return Ok(false);
        }
        self.expire_snapshots();
        if !self.room_for_snapshot(prompt) {
            return Ok(false);
        }
        let layers = match self.capture_layers() {
            Ok(layers) => layers,
            Err(e) => {
                self.kv_snapshots.clear();
                return Err(e);
            }
        };
        let logits = logits.map(|l| l.affine(1.0, 0.0)).transpose()?;
        self.store_snapshot(prompt.to_vec(), prompt.len(), layers, logits);
        Ok(true)
    }

    /// Sets the time a snapshot is kept after its last use, from the configuration.
    pub fn set_kv_lease(&mut self, lease: Option<std::time::Duration>) {
        self.kv_lease = lease;
    }

    /// Drops the snapshots whose lease ran out, handing their memory back.
    pub fn expire_snapshots(&mut self) {
        let Some(lease) = self.kv_lease else { return };
        let before = self.kv_snapshots.len();
        self.kv_snapshots.retain(|s| s.used.elapsed() < lease);
        if self.kv_snapshots.len() < before {
            tracing::debug!(
                "kv snapshots: {} past their lease dropped",
                before - self.kv_snapshots.len()
            );
            crate::inference::place::ladder::release_pools();
        }
    }

    /// Names the position up to which the prompt about to be read is shared with the next
    /// turn of its conversation: the state there is kept while the prompt is read.
    pub fn set_prefix_boundary(&mut self, boundary: Option<usize>) {
        self.prefix_boundary = boundary;
        self.prefix_covered = 0;
    }

    pub fn prefix_boundary(&self) -> Option<usize> {
        self.prefix_boundary
    }

    /// Tokens the last prefix restore covered, 0 after a prompt read whole.
    pub fn prefix_covered(&self) -> usize {
        self.prefix_covered
    }

    /// Records the tokens a prompt's read began after, for the reply to report.
    pub fn set_prefix_covered(&mut self, covered: usize) {
        self.prefix_covered = covered;
    }

    /// Device memory the kept conversation states hold.
    pub fn kv_snapshot_bytes(&self) -> u64 {
        self.kv_snapshots.iter().map(KvSnapshot::device_bytes).sum()
    }

    /// Restores the longest prefix snapshot whose tokens begin `prompt` and returns how
    /// many tokens it covers with the logits it was taken with. A recurrent state cannot be
    /// cut back to a shorter prefix, so only a snapshot `prompt` extends whole qualifies.
    pub fn try_restore_prefix(
        &mut self,
        prompt: &[u32],
    ) -> crate::tensor::Result<Option<(usize, crate::tensor::Tensor)>> {
        Ok(self
            .restore_prefix(prompt, true)?
            .and_then(|(covered, logits)| logits.map(|l| (covered, l))))
    }

    /// [`Self::try_restore_prefix`] that also takes a state kept without logits (a
    /// conversation's boundary) when it is shorter than `prompt`; the logits come back only
    /// for a state that has them.
    pub fn restore_prefix(
        &mut self,
        prompt: &[u32],
        logits_only: bool,
    ) -> crate::tensor::Result<Option<(usize, Option<crate::tensor::Tensor>)>> {
        self.prefix_covered = 0;
        if !self.is_recurrent() {
            return Ok(None);
        }
        self.expire_snapshots();
        let Some(index) = self
            .kv_snapshots
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                prompt.starts_with(&s.tokens)
                    && (s.logits.is_some() || (!logits_only && s.tokens.len() < prompt.len()))
            })
            .max_by_key(|(_, s)| s.tokens.len())
            .map(|(i, _)| i)
        else {
            return Ok(None);
        };
        let Self {
            layers,
            kv_snapshots,
            ..
        } = self;
        restore_layers(layers, &kv_snapshots[index].layers)?;
        self.rebuild_host_stores();
        let draft = self.kv_snapshots[index].draft.clone();
        self.restore_draft(draft.as_ref())?;
        let snap = &self.kv_snapshots[index];
        let covered = snap.kv_len;
        let logits = snap
            .logits
            .as_ref()
            .map(|l| l.affine(1.0, 0.0))
            .transpose()?;
        self.kv_snapshot_tick += 1;
        let used = &mut self.kv_snapshots[index];
        used.tick = self.kv_snapshot_tick;
        used.used = std::time::Instant::now();
        used.hits = used.hits.saturating_add(1);
        self.prefix_covered = covered;
        Ok(Some((covered, logits)))
    }

    /// Keeps the snapshots within a device-memory budget, dropping the least recently
    /// used until they fit. The budget is the free device memory plus what the snapshots
    /// already hold, less the working window the cache may still grow by and the memory
    /// the next prefill chunk works in, for a model that declares its chunk.
    /// Drop the least recently used snapshots until `bytes` of the cards are free of them, or
    /// none is left; what they held goes back to the cards.
    pub(crate) fn drop_kv_snapshots_for(&mut self, bytes: u64) {
        let mut freed = 0u64;
        while freed < bytes && !self.kv_snapshots.is_empty() {
            let i = self
                .kv_snapshots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.rank())
                .map(|(i, _)| i)
                .unwrap();
            freed += self.kv_snapshots[i].device_bytes();
            self.kv_snapshots.swap_remove(i);
        }
        if freed > 0 {
            tracing::debug!("kv snapshots: {freed} bytes dropped for a prefill chunk");
            crate::inference::place::ladder::release_pools();
        }
    }

    /// Device memory the snapshots leave free: the working window the cache may still grow by
    /// and what a decode was seen to take. The next prefill's chunk is not kept aside: it takes
    /// its room from the snapshots when it starts, oldest first.
    fn snapshot_reserve(&self) -> u64 {
        let (layers, n_kv, hd) = self.kv_layout();
        let window_bytes = (layers
            * 2
            * n_kv
            * hd
            * crate::inference::cache::KV_WORKING_WINDOW_TOKENS
            * crate::tensor::DType::F16.size_in_bytes()) as u64;
        window_bytes.saturating_add(self.decode_reserve)
    }

    fn enforce_kv_snapshot_budget(&mut self) {
        let held: u64 = self.kv_snapshots.iter().map(KvSnapshot::device_bytes).sum();
        let budget = match self.free_device_memory() {
            Some(free) => (free + held).saturating_sub(self.snapshot_reserve()),
            // No card: the snapshots live in host memory, bounded by what the host can spare.
            None => held
                .saturating_add(crate::inference::place::layer_executor::host_spill_budget_bytes()),
        };
        let mut held = held;
        while held > budget && !self.kv_snapshots.is_empty() {
            let i = self
                .kv_snapshots
                .iter()
                .enumerate()
                .min_by_key(|(_, s)| s.rank())
                .map(|(i, _)| i)
                .unwrap();
            held -= self.kv_snapshots[i].device_bytes();
            self.kv_snapshots.swap_remove(i);
        }
        if held > budget {
            tracing::debug!(
                "kv snapshot dropped: {held} bytes exceed the {budget} byte budget alone"
            );
            self.kv_snapshots.clear();
        }
    }

    /// Free memory on the device holding the first layer, when it is a card.
    /// The least free memory over the model's cards: what every card can still give.
    pub fn free_device_memory(&self) -> Option<u64> {
        self.cuda_devices
            .values()
            .filter_map(crate::inference::place::room::free_now)
            .min()
    }

    /// A request's decode took `bytes` of a card beyond what it had at its start, the
    /// recorded step, windows and head among them: the snapshots leave that much room.
    pub fn note_decode_reserve(&mut self, bytes: u64) {
        self.decode_reserve = self.decode_reserve.max(bytes);
    }

    /// The snapshot sharing the longest prefix with `prompt`, as `(index, common)`,
    /// `common` never past what the snapshot holds in its KV.
    pub fn best_kv_snapshot(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        self.kv_snapshots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.logits.is_none())
            .map(|(i, s)| (i, common_prefix(&s.tokens, prompt).min(s.kv_len)))
            .filter(|(_, c)| *c > 0)
            .max_by_key(|(_, c)| *c)
    }

    /// Makes snapshot `index` the resident KV; returns its tokens and KV length. The
    /// captured decode graph is dropped, as after any change of resident sequence.
    pub fn restore_kv_snapshot(
        &mut self,
        index: usize,
    ) -> crate::tensor::Result<(Vec<u32>, usize)> {
        let Self {
            layers,
            kv_snapshots,
            ..
        } = self;
        let snap = &kv_snapshots[index];
        restore_layers(layers, &snap.layers)?;
        let out = (snap.tokens.clone(), snap.kv_len);
        self.rebuild_host_stores();
        self.kv_snapshot_tick += 1;
        let used = &mut self.kv_snapshots[index];
        used.tick = self.kv_snapshot_tick;
        used.used = std::time::Instant::now();
        used.hits = used.hits.saturating_add(1);
        self.invalidate_graph_state();
        Ok(out)
    }

    /// The snapshot whose cache covers the longest run of `prompt` from its start: the index of
    /// the one holding most positions that are all `prompt`'s own tokens, whether or not it
    /// carries logits.
    pub fn state_snapshot_covering(&self, prompt: &[u32]) -> Option<usize> {
        self.kv_snapshots
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.kv_len > 0
                    && s.tokens.len() >= s.kv_len
                    && prompt.starts_with(&s.tokens[..s.kv_len])
            })
            .max_by_key(|(_, s)| s.kv_len)
            .map(|(i, _)| i)
    }

    /// `(layers, n_kv, head_dim)`: what a KV row of this model is made of.
    pub fn kv_layout(&self) -> (usize, usize, usize) {
        let l = self
            .layers
            .iter()
            .find(|l| matches!(l.mixer, super::mixer::Mixer::Attention))
            .unwrap_or(&self.layers[0]);
        (self.layers.len(), l.n_kv_head, l.head_dim)
    }

    /// Tokens and KV length of snapshot `index`.
    pub fn kv_snapshot_ref(&self, index: usize) -> Option<(&[u32], usize)> {
        self.kv_snapshots
            .get(index)
            .map(|s| (s.tokens.as_slice(), s.kv_len))
    }

    /// Every layer's rows for tokens `[from, to)` of snapshot `index`, host f32, or
    /// `None` for a layer that keeps no KV of its own.
    /// The KV element dtype of a snapshot (0 = F16, 1 = F32, 2 = BF16), read from its
    /// first layer. The disk manifest records it so a cold import rebuilds tensors the
    /// cache accepts.
    pub fn kv_snapshot_dtype(&self, index: usize) -> Option<u8> {
        let snap = self.kv_snapshots.get(index)?;
        let dt = snap
            .layers
            .iter()
            .find_map(|l| l.spec.as_ref().map(|s| s.dtype()))
            .or_else(|| {
                // A quantised layer's rows come out dequantised to F32, which holds them exactly.
                snap.layers
                    .iter()
                    .any(|l| l.quant.is_some())
                    .then_some(crate::tensor::DType::F32)
            })?;
        Some(match dt {
            crate::tensor::DType::F16 => 0,
            crate::tensor::DType::F32 => 1,
            crate::tensor::DType::BF16 => 2,
            _ => 0,
        })
    }

    /// The prefix length every layer of a snapshot holds in full, when they agree.
    /// A windowed (SWA) layer keeps only its recent window, not the whole prefix, so
    /// its captured length is shorter; the disk tier, which stores blocks of a token
    /// prefix, cannot represent that and must skip such a snapshot.
    pub fn kv_snapshot_uniform_len(&self, index: usize) -> Option<usize> {
        let snap = self.kv_snapshots.get(index)?;
        let mut len: Option<usize> = None;
        for l in &snap.layers {
            let n = l.kv_len()?;
            match len {
                None => len = Some(n),
                Some(prev) if prev == n => {}
                Some(_) => return None,
            }
        }
        len.filter(|&n| n == snap.kv_len)
    }

    pub fn export_kv_rows(
        &self,
        index: usize,
        from: usize,
        to: usize,
    ) -> crate::tensor::Result<crate::inference::engine::model_backend::KvRows> {
        let snap = self
            .kv_snapshots
            .get(index)
            .ok_or_else(|| crate::tensor::Error::msg(format!("kv snapshot {index} gone")))?;
        let kv_len = snap.kv_len;
        snap.layers
            .iter()
            .map(|l| {
                // Only layers that hold the whole [0, kv_len] prefix go in the shared,
                // content-addressed block chain. A windowed layer holds a suffix whose
                // extent depends on the full sequence, not on this block's tokens, so it
                // could not share a block hash without corrupting it; it travels in the
                // manifest's own window blob instead.
                match l.kv_len() {
                    Some(n) if n == kv_len => l.host_rows(from, to),
                    _ => Ok(None),
                }
            })
            .collect()
    }

    /// The windowed layers' rows for a snapshot, trimmed to end at `covered` (the
    /// block-aligned prefix the global layers share), so both describe the same length.
    /// Global layers return `None`; they live in the block chain.
    pub fn export_window_rows(
        &self,
        index: usize,
        covered: usize,
    ) -> crate::tensor::Result<crate::inference::engine::model_backend::KvRows> {
        let snap = self
            .kv_snapshots
            .get(index)
            .ok_or_else(|| crate::tensor::Error::msg(format!("kv snapshot {index} gone")))?;
        let kv_len = snap.kv_len;
        let drop = kv_len.saturating_sub(covered);
        snap.layers
            .iter()
            .map(|l| match l.kv_len() {
                Some(n) if n < kv_len => l.host_rows(0, n.saturating_sub(drop)),
                _ => Ok(None),
            })
            .collect()
    }

    /// A snapshot rebuilt from host rows, each layer's placed where that layer keeps
    /// its KV today, entered in the table under `tokens`.
    pub fn import_kv_snapshot(
        &mut self,
        tokens: Vec<u32>,
        kv_len: usize,
        rows: Vec<Option<(Vec<f32>, Vec<f32>)>>,
        dtype_code: u8,
    ) -> crate::tensor::Result<()> {
        let kv_dtype = match dtype_code {
            1 => crate::tensor::DType::F32,
            2 => crate::tensor::DType::BF16,
            _ => crate::tensor::DType::F16,
        };
        // Rows on disk are a whole layer's: a layer split across cards has no place to put them.
        if !self.supports_kv_snapshots() || self.has_split_layers() {
            return Ok(());
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for (l, r) in self.layers.iter().zip(rows) {
            let Some((k, v)) = r else {
                layers.push(super::LayerKvSnapshot {
                    spec: None,
                    quant: None,
                    cpu_f16: None,
                    mixer: None,
                    shards: Vec::new(),
                });
                continue;
            };
            layers.push(layer_snapshot_from_rows(l, &k, &v, kv_dtype)?);
        }
        self.store_snapshot(tokens, kv_len, layers, None);
        Ok(())
    }

    /// Trim every layer's KV cache to `new_len` valid positions.
    ///
    /// Used by speculative decoding to rewind after partial acceptance: the
    /// verification forward writes K+1 new entries into the KV cache, but
    /// if only M of K drafts were accepted we must discard the rejected
    /// ones so subsequent decode steps see the correct context.
    ///
    /// O(num_layers) - each layer's cache only updates its internal
    /// `current_seq_len` counter (see `SpecKvCache::trim_to`).
    pub fn trim_kv(&mut self, new_len: usize) {
        self.spec_settle(new_len);
        for layer in &mut self.layers {
            // A recurrent state cannot be cut to a shorter length: it is dropped only when
            // nothing is kept.
            if new_len == 0 {
                layer.state_reset();
                // Nothing is mid-pass once the cache starts again: the expert transit of a
                // pass that was cut short is forgotten.
                #[cfg(feature = "gpu")]
                if let Some(tier) = layer.moe.as_ref().and_then(|m| m.tier.as_ref()) {
                    if let Err(e) = tier.reset_stage() {
                        tracing::warn!("expert transit not reset: {e}");
                    }
                }
            }
            layer.trim_kv_to(new_len);
        }
    }

    /// Get the CUDA stream used by this model's layers (for graph capture).
    /// Returns None if the model doesn't use CUDA. CUDA-only (the return type
    /// is a CUDA handle).
    ///
    /// Prefers layer 0's compute device - the canonical placement
    /// source-of-truth (`embed_for_graph`/`prepare_all_kv` use the same
    /// rule); `cuda_devices.values().next()` is HashMap-ordered and can
    /// name an unused GPU. Graph mode is single-GPU so they coincide, but
    /// the capture stream MUST be the one the kernels enqueue on.
    #[cfg(feature = "gpu")]
    pub fn cuda_stream(&self) -> Option<std::sync::Arc<crate::tensor::cuda_ext::CudaStream>> {
        self.layers
            .first()
            .map(|l| l.cos.device().clone())
            .filter(|d| d.is_cuda())
            .or_else(|| self.cuda_devices.values().next().cloned())
            .and_then(|d| crate::tensor::cuda_ext::stream_of(&d).ok())
    }

    /// Drop captured-region transient lifetime-extension handles WITHOUT
    /// touching the stable graph buffers. The engine calls this right
    /// before (re)allocating the capture arena: those handles hold
    /// arena-backed tensors from a previous capture, and they must be
    /// dropped while their arena backing is still alive (arena-range drops
    /// are no-ops) - never after the arena is freed/reallocated.
    pub fn clear_graph_transients(&mut self) {
        for l in &mut self.layers {
            l.graph_alive_tensors.clear();
            for shard in &mut l.tp_shards {
                shard.graph_alive_tensors.clear();
            }
        }
        self.graph_alive_tensors_model.clear();
    }

    /// Pre-allocated hidden state buffer for graph replay (fixed device pointer).
    /// Uses force_contiguous on first call to guarantee an INDEPENDENT
    /// allocation (not a shared-storage clone).
    fn ensure_hidden_buffer(&mut self, hidden: &Tensor) -> Result<Tensor> {
        if self.graph_hidden_buffer.is_none() {
            self.graph_hidden_buffer = Some(hidden.force_contiguous()?);
        } else {
            self.graph_hidden_buffer
                .as_ref()
                .unwrap()
                .slice_set(hidden, 0, 0)?;
        }
        Ok(self.graph_hidden_buffer.as_ref().unwrap().clone())
    }

    /// Split forward step 1: QKV + RoPE + KV write for ALL layers.
    /// Runs OUTSIDE graph. Returns (hidden_buffer, vec_of_Q_buffers).
    /// All returned tensors are at fixed device pointers for graph replay.
    pub fn prepare_all_kv(
        &mut self,
        input_ids: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let _step = GraphStep::enter();
        let (_, seq_len) = input_ids.dims2()?;
        assert_eq!(seq_len, 1);

        let emb_device = self.embeddings.embeddings().device().clone();
        let input_local;
        let input_ids = if input_ids.device().same_device(&emb_device) {
            input_ids
        } else {
            input_local = input_ids.to_device(&emb_device)?;
            &input_local
        };
        let mut hidden = self.embeddings.forward(input_ids)?;
        // FIX: apply embed_scale (gemma4: x√2560 ≈ 50.6).
        // forward_inner (line 5964) and embed_for_graph (line 7278) both
        // scale; prepare_all_kv was MISSING it -> split-path hidden was 50.6x
        // too small -> garbage. This was bug #1 of the split-path divergence.
        if let Some(scale) = self.config.embed_scale {
            hidden = (hidden * scale)?;
        }
        // Land hidden on the device that owns layer 0 - same canonical
        // placement as embed_for_graph (commit f86ee5c). HashMap iteration
        // order of cuda_devices is non-deterministic, so the previous
        // `.values().next()` could land on the wrong GPU and trigger
        // "device mismatch in layer-norm" at the first layer.
        let target_dev = self
            .layers
            .first()
            .map(|l| l.cos.device().clone())
            .filter(|d| d.is_cuda())
            .or_else(|| self.cuda_devices.values().next().cloned())
            .ok_or_else(|| crate::tensor::Error::msg("prepare_all_kv needs CUDA"))?;
        if hidden.device().location() != target_dev.location() {
            hidden = hidden.to_device(&target_dev)?;
        }

        // Write hidden into fixed buffer (same pointer for graph replay)
        let hidden = self.ensure_hidden_buffer(&hidden)?;

        // FIX: populate per-layer PLE input buffers.
        // populate_ple_input_buffers was DEFINED but NEVER CALLED -> the
        // graph_ple_input_buffer slots stayed zero -> the PLE block in
        // compute_from_kv added zero per-layer embeddings -> garbage. This
        // was bug #2. Runs OUTSIDE capture each token (reads the current
        // graph_hidden_buffer just written above); the captured compute
        // then reads the stable per-layer PLE buffers. No-op for ple_dim==0.
        self.populate_ple_input_buffers(input_ids)?;

        let mut all_q = Vec::with_capacity(self.layers.len());
        for layer in &mut self.layers {
            let q = layer.prepare_kv(&hidden, index_pos)?;
            all_q.push(q);
        }
        Ok((hidden, all_q))
    }

    /// Split forward step 2: attention + FFN for ALL layers + output projection.
    /// Runs INSIDE graph. Uses pre-computed Q tensors (with correct RoPE position).
    pub fn compute_all_from_kv(&mut self, hidden: &Tensor, all_q: &[Tensor]) -> Result<Tensor> {
        let mut hidden = hidden.clone();
        // FIX: shared-KV layers. gemma4 8B's last
        // `shared_kv_layers` layers reuse K/V from a type-matched reference
        // layer (mirrors forward_layers_and_output:6163-6208). The split path
        // previously ignored this -> 18 layers attended to their own (wrong)
        // K/V. Resolve the donor and pass its padded KV buffer.
        let n_layers = self.layers.len();
        let shared_kv_n = self.config.shared_kv_layers;
        let first_kv_shared = if shared_kv_n > 0 {
            n_layers - shared_kv_n
        } else {
            n_layers
        };
        for layer_idx in 0..n_layers {
            if first_kv_shared > 0 && layer_idx >= first_kv_shared {
                // Donor = nearest earlier non-shared layer with matching head_dim.
                let layer_hd = self.layers[layer_idx].head_dim;
                let ref_idx = (0..first_kv_shared)
                    .rev()
                    .find(|&i| self.layers[i].head_dim == layer_hd)
                    .unwrap_or(first_kv_shared.saturating_sub(1));
                let target_dev = self.layers[layer_idx].cos.device().clone();
                // Clone donor's padded K/V (owned Tensors -> borrow ends here).
                let (k, v) = {
                    let ref_cache = &self.layers[ref_idx].kv_cache;
                    let k = ref_cache.k_buffer().ok_or_else(|| {
                        crate::tensor::Error::msg("shared-KV: donor k_buffer missing")
                    })?;
                    let v = ref_cache.v_buffer().ok_or_else(|| {
                        crate::tensor::Error::msg("shared-KV: donor v_buffer missing")
                    })?;
                    let k = if k.device().same_device(&target_dev) {
                        k
                    } else {
                        k.to_device(&target_dev)?
                    };
                    let v = if v.device().same_device(&target_dev) {
                        v
                    } else {
                        v.to_device(&target_dev)?
                    };
                    (k, v)
                };
                let layer = &mut self.layers[layer_idx];
                hidden = layer.compute_from_kv_shared(&hidden, &all_q[layer_idx], Some((k, v)))?;
            } else {
                let layer = &mut self.layers[layer_idx];
                hidden = layer.compute_from_kv(&hidden, &all_q[layer_idx])?;
            }
        }

        // Output projection (same as forward_padded)
        // prefer the Q-quant path (mvq, no cuBLAS GEMM) when
        // available. forward_from_hidden uses this path; compute_all_from_kv
        // previously used cuBLAS GEMM which the captured graph couldn't
        // safely replay (nvjet kernel writes 128B before its output buffer
        // - cuBLAS workspace state isn't tracked by the reference Tensor
        // lifetime extension, see compute-sanitizer evidence).
        let mut logits = if self.output_proj_cuda_qmm.is_some() {
            let qmm = self.output_proj_cuda_qmm.as_ref().unwrap();
            let last = hidden.i((.., 0, ..))?;
            self.graph_alive_tensors_model.push(last.clone());
            let norm_w = self.output_norm_cuda_weight.as_ref().unwrap();
            let last_f32 = last.to_dtype(crate::tensor::DType::F32)?;
            self.graph_alive_tensors_model.push(last_f32.clone());
            let normed =
                crate::tensor::ops::rms_norm(&last_f32, norm_w, self.config.rms_norm_eps as f32)?;
            self.graph_alive_tensors_model.push(normed.clone());
            let mm_out = qmm.forward(&normed)?;
            self.graph_alive_tensors_model.push(mm_out.clone());
            mm_out
        } else if self.output_proj_cuda.is_some() {
            let last = hidden.i((.., 0, ..))?;
            self.graph_alive_tensors_model.push(last.clone());
            let norm_w = self.output_norm_cuda_weight.as_ref().unwrap();
            let last_f32 = last.to_dtype(crate::tensor::DType::F32)?;
            self.graph_alive_tensors_model.push(last_f32.clone());
            let normed =
                crate::tensor::ops::rms_norm(&last_f32, norm_w, self.config.rms_norm_eps as f32)?;
            self.graph_alive_tensors_model.push(normed.clone());
            let proj = self.output_proj_cuda.as_ref().unwrap();
            let normed_f16 = normed.to_dtype(crate::tensor::DType::F16)?;
            self.graph_alive_tensors_model.push(normed_f16.clone());
            let mm_out = normed_f16.matmul_t(proj)?;
            self.graph_alive_tensors_model.push(mm_out.clone());
            mm_out
        } else {
            let hidden_cpu = hidden.to_device(&Device::Cpu)?;
            let normed = self.output_norm.forward(&hidden_cpu)?;
            let last = normed.i((.., 0, ..))?;
            self.output_proj.forward(&last)?
        };
        logits = self.apply_output_bias(logits)?;

        let pre_cast = logits.clone();
        logits = logits.to_dtype(crate::tensor::DType::F32)?;
        self.graph_alive_tensors_model.push(pre_cast);
        self.graph_alive_tensors_model.push(logits.clone());
        if let Some(scale) = self.config.logit_scale {
            logits = (logits / scale)?;
        }
        let logits = match self.config.final_logit_softcapping {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }?;

        // Route through stable graph_logits_buffer - mirror
        // forward_from_hidden (line 5183). Was missing here, causing
        // moondream graph replay to read a freed pointer (compute-sanitizer
        // pinned this to fused_penalty_argmax in gpu_sample).
        if self.graph_logits_buffer.is_none() {
            self.graph_logits_buffer = Some(logits.force_contiguous()?);
        } else {
            self.graph_logits_buffer
                .as_ref()
                .unwrap()
                .slice_set(&logits, 0, 0)?;
        }
        Ok(self.graph_logits_buffer.as_ref().unwrap().clone())
    }

    /// Forward pass with padded attention (CUDA graph mode).
    /// All operations have fixed dimensions for graph capture/replay.
    /// Only supports single-token inference (seq_len=1) on all-CUDA models.
    pub fn forward_padded(&mut self, input_ids: &Tensor, index_pos: usize) -> Result<Tensor> {
        // Just use the proven correct forward path
        self.forward(input_ids, index_pos)
    }

    /// Split-path forward for F-dtype graph mode (option-E fix for
    /// +). Does the prepare-then-compute sequence in one call so
    /// engine dispatch is a single method swap. The QKV+RoPE+KV-append
    /// half runs uncaptured each call (writes Q/K/V to stable buffers);
    /// only the compute-from-kv half is meant to be captured.
    ///
    /// CALLER CONTRACT: this method itself runs both halves. Engine
    /// uses it for warmup; for capture, engine calls
    /// `prepare_all_kv` separately, begins capture, then calls
    /// `compute_all_from_kv_captured` inside capture.
    pub fn forward_from_hidden_split(&mut self, input_ids: &Tensor, pos: usize) -> Result<Tensor> {
        let (hidden, all_q) = self.prepare_all_kv(input_ids, pos)?;
        self.compute_all_from_kv(&hidden, &all_q)
    }

    /// Captured-side half of the split-path forward. Reads from
    /// `graph_hidden_buffer` + each layer's `graph_q_buffer` (refreshed
    /// outside capture by `prepare_all_kv`). Does NOT take args - uses
    /// internal state - because captured graphs can't bind per-replay
    /// arguments.
    pub fn compute_all_from_kv_captured(&mut self) -> Result<Tensor> {
        let _step = GraphStep::enter();
        let hidden = self
            .graph_hidden_buffer
            .as_ref()
            .ok_or_else(|| {
                crate::tensor::Error::msg("compute_all_from_kv_captured: call prepare_all_kv first")
            })?
            .clone();
        // Build a vec of clones that wrap each layer's graph_q_buffer
        // storage. `compute_from_kv` ignores the arg and reads from the
        // buffer directly, but the indexing must be valid.
        let all_q: Vec<Tensor> = self
            .layers
            .iter()
            .map(|l| {
                l.graph_q_buffer
                    .as_ref()
                    .ok_or_else(|| {
                        crate::tensor::Error::msg(
                            "compute_all_from_kv_captured: layer has no graph_q_buffer",
                        )
                    })
                    .cloned()
            })
            .collect::<Result<_>>()?;
        self.compute_all_from_kv(&hidden, &all_q)
    }
}

/// Snapshots are kept by the device-memory budget alone (`enforce_kv_snapshot_budget`), not
/// by a count.
const SNAPSHOT_COUNT_UNBOUNDED: usize = usize::MAX;

/// Which snapshot a new one replaces: the one whose tokens it extends or shortens (the
/// same conversation, one turn on), else, at capacity, the least recently used; `None`
/// appends.
pub(crate) fn snapshot_slot(
    existing: &[(&[u32], u64)],
    tokens: &[u32],
    cap: usize,
) -> Option<usize> {
    if let Some((i, _)) = existing
        .iter()
        .enumerate()
        .find(|(_, (t, _))| tokens.starts_with(t) || t.starts_with(tokens))
    {
        return Some(i);
    }
    if existing.len() < cap {
        return None;
    }
    existing
        .iter()
        .enumerate()
        .min_by_key(|(_, (_, tick))| *tick)
        .map(|(i, _)| i)
}

fn common_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// A number no model built before in this process was given.
pub(super) fn next_graph_instance() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl GenericTransformerLayer {
    /// Times any buffer of this layer that a recorded step reads was made anew: the caches,
    /// the indexer's copy on the card and the carried states, the shares of a split layer's
    /// included.
    pub(super) fn buffer_moves(&self) -> u64 {
        let mut n = self
            .kv_cache
            .buffer_moves()
            .wrapping_add(self.mixer.buffer_moves());
        if let Some(p) = self.ngram.as_ref() {
            n = n.wrapping_add(p.buffer_moves());
        }
        #[cfg(feature = "gpu")]
        {
            if let Some(c) = self.quant_kv.as_ref() {
                n = n.wrapping_add(c.buffer_moves());
            }
            if let Some(sp) = self.sparse.as_ref() {
                n = n.wrapping_add(sp.card_births());
            }
        }
        self.tp_shards
            .iter()
            .fold(n, |n, s| n.wrapping_add(s.buffer_moves()))
    }

    /// Drop `discard` positions from `from` on in every cache this layer and its shares keep, the
    /// keys behind them rotated back by `delta` positions.
    pub(super) fn shift_kv_tail(
        &mut self,
        from: usize,
        discard: usize,
        delta: i64,
    ) -> crate::tensor::Result<()> {
        for shard in &mut self.tp_shards {
            shard.shift_kv_tail(from, discard, delta)?;
        }
        let table = self.rope_table();
        let host = table.delta_host(delta)?;
        let mut rotate = |k: &crate::tensor::Tensor| table.rotate_keys_by(k, delta);
        self.kv_cache.shift_tail(from, discard, &mut rotate)?;
        #[cfg(feature = "gpu")]
        if let Some(c) = self.q8_kv_mut() {
            c.shift_tail(from, discard, &mut rotate)
                .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
        }
        if let Some(c) = self.cpu_q8_kv.as_mut() {
            c.shift_tail(from, discard, host.as_ref())?;
        }
        if let Some(c) = self.cpu_f16_kv.as_mut() {
            c.shift_tail(from, discard, host.as_ref())?;
        }
        Ok(())
    }

    /// Cut every cache this layer keeps back to `new_len` positions.
    pub(super) fn trim_kv_to(&mut self, new_len: usize) {
        for shard in &mut self.tp_shards {
            shard.trim_kv_to(new_len);
        }
        if let Some(c) = self.context.as_mut() {
            if let Err(e) = c.trim_to(new_len) {
                tracing::warn!("context ranges not cut to {new_len}: {e}");
            }
        }
        self.kv_cache.trim_to(new_len);
        // Keep the Q8 mirror in sync with the F-dtype cache; otherwise
        // speculative-decoding rollbacks leave Q8 with stale draft
        // positions and subsequent decode reads garbage.
        //
        // The quantised cache moves length, device position and graph state together
        // (see `KvCore::trim_to`).
        #[cfg(feature = "gpu")]
        if let Some(c) = self.quant_kv.as_mut() {
            c.trim_to(new_len);
        }
        // CPU KV stores (dense Q8 / gemma4 F16). Multi-token verify
        // (PLD/EAGLE) appends drafts here too; without trimming them the
        // rejected draft positions accumulate and corrupt later attention
        // (greedy decode diverges a few tokens in). Mirrors the F-dtype trim.
        if let Some(c) = self.cpu_q8_kv.as_mut() {
            if new_len == 0 {
                c.reset();
            } else {
                c.trim_to(new_len);
            }
        }
        if let Some(c) = self.cpu_f16_kv.as_mut() {
            if new_len == 0 {
                c.reset();
            } else {
                c.trim_to(new_len);
            }
        }
    }
}

#[cfg(test)]
mod kv_snapshot_tests {
    use super::snapshot_slot;

    #[test]
    fn a_grown_conversation_replaces_its_own_snapshot() {
        let a = [1u32, 2, 3];
        let b = [9u32, 8];
        let existing = [(&a[..], 1u64), (&b[..], 2u64)];
        assert_eq!(snapshot_slot(&existing, &[1, 2, 3, 4, 5], 2), Some(0));
        assert_eq!(snapshot_slot(&existing, &[9], 2), Some(1));
    }

    #[test]
    fn a_snapshot_larger_than_the_free_room_is_skipped() {
        use super::snapshot_fits;
        let (free, reserve) = (10_000u64, 3_000u64);
        assert!(snapshot_fits(7_000, Some(free), reserve));
        assert!(!snapshot_fits(7_001, Some(free), reserve));
        assert!(!snapshot_fits(1, Some(reserve - 1), reserve));
        assert!(snapshot_fits(0, Some(0), reserve));
        assert!(snapshot_fits(u64::MAX, None, reserve));
    }

    #[test]
    fn a_new_conversation_appends_then_evicts_the_least_recent() {
        let a = [1u32, 2, 3];
        let b = [9u32, 8];
        let existing = [(&a[..], 5u64), (&b[..], 2u64)];
        assert_eq!(snapshot_slot(&existing, &[7, 7], 3), None);
        assert_eq!(snapshot_slot(&existing, &[7, 7], 2), Some(1));
    }
}

/// Capture a quantised layer's KV as a snapshot: a device-to-device copy of the cache's
/// packed codes and scales for the positions it holds, restored bit for bit. Returns `None`
/// for a non-quantised layer, whose spec cache the caller snapshots directly.
fn quantised_layer_snapshot(
    l: &GenericTransformerLayer,
) -> crate::tensor::Result<Option<super::LayerKvSnapshot>> {
    #[cfg(feature = "gpu")]
    {
        if let Some(q) = l.quant_kv.as_ref() {
            let quant = q
                .snapshot_raw()
                .map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
            return Ok(Some(super::LayerKvSnapshot {
                spec: None,
                quant,
                cpu_f16: None,
                mixer: None,
                shards: Vec::new(),
            }));
        }
    }
    let _ = l;
    Ok(None)
}

/// Device bytes [`capture_layer`] would copy from `l` and its shares as they hold now: each
/// KV cache's bytes for its length in the form its snapshot takes, and each recurrent state.
fn layer_snapshot_bytes(l: &GenericTransformerLayer) -> u64 {
    #[cfg(feature = "gpu")]
    let kv = match l.quant_kv.as_ref() {
        Some(q) => q.raw_bytes(q.current_seq_len()) as u64,
        None => l.kv_cache.snapshot_bytes(),
    };
    #[cfg(not(feature = "gpu"))]
    let kv = l.kv_cache.snapshot_bytes();
    kv + l.mixer.state_device_bytes() + l.tp_shards.iter().map(layer_snapshot_bytes).sum::<u64>()
}

/// Whether a snapshot copy of `need` bytes fits in `free` device bytes once `reserve` is
/// kept aside. An unknown free figure (no card) never refuses.
pub(super) fn snapshot_fits(need: u64, free: Option<u64>, reserve: u64) -> bool {
    free.is_none_or(|free| need <= free.saturating_sub(reserve))
}

/// A layer's KV snapshot rebuilt from host rows `k` and `v` (token-major, `[n, n_kv, head_dim]`),
/// placed where the layer keeps its KV today. A GPU layer keeps its KV in the spec cache; on
/// a cold server the buffer is not allocated yet, so a like tensor is built on the layer's
/// own device in `kv_dtype`.
pub(super) fn layer_snapshot_from_rows(
    l: &GenericTransformerLayer,
    k: &[f32],
    v: &[f32],
    kv_dtype: crate::tensor::DType,
) -> crate::tensor::Result<super::LayerKvSnapshot> {
    let (n_kv, hd) = (l.n_kv_head, l.head_dim);
    let n = k.len() / (n_kv * hd);
    let spec_like = if let Some(buf) = l.kv_cache.k_buffer() {
        Some(buf)
    } else if l.cpu_f16_kv.is_none() {
        Some(crate::tensor::Tensor::zeros_on(
            (1, 1, 1, 1),
            kv_dtype,
            &l.cos.device(),
        )?)
    } else {
        None
    };
    if let Some(buf) = spec_like {
        let spec = crate::inference::serve::spec_kv_cache::SpecKvSnapshot::from_host_rows(
            &buf, k, v, n, n_kv, hd,
        )?;
        return Ok(super::LayerKvSnapshot {
            spec: Some(spec),
            quant: None,
            cpu_f16: None,
            mixer: None,
            shards: Vec::new(),
        });
    }
    if let Some(c) = &l.cpu_f16_kv {
        let mut fresh = c.like();
        let row = n_kv * hd;
        for t in 0..n {
            fresh.append(&k[t * row..(t + 1) * row], &v[t * row..(t + 1) * row])?;
        }
        return Ok(super::LayerKvSnapshot {
            spec: None,
            quant: None,
            cpu_f16: Some(fresh),
            mixer: None,
            shards: Vec::new(),
        });
    }
    Err(crate::tensor::Error::msg(
        "kv snapshot import: a layer has no cache to place rows in".to_string(),
    ))
}

/// Makes a snapshot's layers the resident ones: every KV cache, and every recurrent
/// mixer's state, a mixer without a captured state starting fresh.
pub(super) fn restore_layers(
    layers: &mut [GenericTransformerLayer],
    snap: &[super::LayerKvSnapshot],
) -> crate::tensor::Result<()> {
    for (l, s) in layers.iter_mut().zip(snap) {
        restore_layer(l, s)?;
    }
    Ok(())
}

/// A layer's caches and state, and its shares', as `l` holds them now.
fn capture_layer(l: &GenericTransformerLayer) -> crate::tensor::Result<super::LayerKvSnapshot> {
    let mixer = l.state_snapshot()?;
    let mut snap = match quantised_layer_snapshot(l)? {
        Some(snap) => snap,
        None => super::LayerKvSnapshot {
            spec: l.kv_cache.snapshot()?,
            quant: None,
            cpu_f16: l.cpu_f16_kv.clone(),
            mixer: None,
            shards: Vec::new(),
        },
    };
    snap.mixer = mixer;
    snap.shards = l
        .tp_shards
        .iter()
        .map(capture_layer)
        .collect::<crate::tensor::Result<_>>()?;
    Ok(snap)
}

/// [`capture_layer`]'s snapshot `s` made the layer's and its shares' caches and state again.
fn restore_layer(
    l: &mut GenericTransformerLayer,
    s: &super::LayerKvSnapshot,
) -> crate::tensor::Result<()> {
    for (shard, saved) in l.tp_shards.iter_mut().zip(&s.shards) {
        restore_layer(shard, saved)?;
    }
    match &s.mixer {
        Some(state) => l.state_restore(state)?,
        None => l.state_reset(),
    }
    if restore_quantised_layer(l, s)? {
        return Ok(());
    }
    match &s.spec {
        Some(spec) => l.kv_cache.restore(spec)?,
        None => l.kv_cache.trim_to(0),
    }
    l.cpu_f16_kv = s.cpu_f16.clone();
    Ok(())
}

/// Restore a snapshot layer into a quantised cache: its raw codes and scales copied back, or,
/// for a snapshot rebuilt from host rows, its K/V re-appended. Returns `false` for a
/// non-quantised layer, left to the spec path.
fn restore_quantised_layer(
    l: &mut GenericTransformerLayer,
    s: &super::LayerKvSnapshot,
) -> crate::tensor::Result<bool> {
    #[cfg(feature = "gpu")]
    {
        if let Some(q) = l.quant_kv.as_mut() {
            let restored = match (&s.quant, &s.spec) {
                (Some(raw), _) => q.restore_raw(raw),
                (None, Some(spec)) => {
                    q.reset();
                    let (k, v) = spec.kv();
                    q.append(k, v)
                }
                (None, None) => {
                    q.reset();
                    Ok(())
                }
            };
            restored.map_err(|e| crate::tensor::Error::msg(e.to_string()))?;
            return Ok(true);
        }
    }
    let _ = (l, s);
    Ok(false)
}

/// Drops every per-step buffer a recorded step read from `l` and its shares.
fn clear_layer_graph_state(l: &mut GenericTransformerLayer) {
    l.padded_mask = None;
    l.graph_rope_cos = None;
    l.graph_rope_sin = None;
    l.graph_kv_pos = None;
    l.graph_pos_i32 = None;
    l.graph_attn_qk_buffer = None;
    l.graph_attn_out_buffer = None;
    l.graph_attn_proj_buffer = None;
    l.graph_ffn_up_buffer = None;
    l.graph_ffn_activated_buffer = None;
    l.graph_ffn_down_buffer = None;
    l.graph_phi2_merge_buffer = None;
    // Reset the FULL set of per-layer stable buffers. A partial
    // reset left buffers like graph_ffn_gate_out (and the PLE /
    // norm / scale buffers) pointing at memory that the
    // graph_alive_tensors.clear() below reclaims, so the next decode
    // step read a freed pointer through gate = graph_ffn_gate_out
    // (wild-pointer crash in fused_gelu_mul once the F-dtype KV cache
    // outgrew the captured mask cap and triggered this teardown).
    // Clearing every buffer makes the decode path fall back to fresh
    // tensors after invalidation instead of stale stable buffers.
    l.graph_q_buffer = None;
    l.graph_ffn_up_concat_buffer = None;
    l.graph_post_ffn_norm_buffer = None;
    l.graph_ple_gate_buffer = None;
    l.graph_ple_gelu_buffer = None;
    l.graph_ple_proj_buffer = None;
    l.graph_ple_final_buffer = None;
    l.graph_post_attn_norm_buffer = None;
    l.graph_x_norm_ffn_buffer = None;
    l.graph_x_norm_attn_buffer = None;
    l.graph_ple_input_buffer = None;
    l.graph_ngram_rows = None;
    l.graph_kv_cap = None;
    l.graph_attn_mask_added_buffer = None;
    l.graph_attn_softmax_buffer = None;
    l.graph_ffn_gate_out = None;
    l.graph_x_out_scaled = None;
    l.graph_attn_scaled_buffer = None;
    // ROOT CAUSE FIX: drop captured-region tensor
    // lifetime-extension handles so cuMemFree can reclaim
    // their memory now that the graph is being torn down.
    l.graph_alive_tensors.clear();
    for shard in &mut l.tp_shards {
        clear_layer_graph_state(shard);
    }
}

/// Which of a layer's per-step graph buffers it writes itself rather than sharing a group
/// leader's.
#[derive(Clone, Copy)]
struct GraphStateLeaders {
    rope: bool,
    mask: bool,
    kv_pos: bool,
}

impl GraphStateLeaders {
    /// A layer that keeps every buffer of its own.
    const OWN: Self = Self {
        rope: true,
        mask: true,
        kv_pos: true,
    };
}

/// The per-step state layer `i` reads in a recorded step at `pos`: the buffers it leads, its
/// quantised cache's slot and the attention buffers it fills in place, on its own card. The
/// card's position for sink and indexed layers is written once into `pos_by_card`.
fn update_layer_graph_state(
    layer: &mut GenericTransformerLayer,
    i: usize,
    pos: usize,
    leaders: GraphStateLeaders,
    skip_fdtype_state: bool,
    pos_by_card: &mut std::collections::HashMap<usize, Tensor>,
) -> Result<()> {
    // A layer's buffers live on the card it runs on.
    let dev = layer.cos.device().clone();
    if layer.attn_sinks.is_some() || layer.sparse.is_some() {
        if let crate::tensor::DeviceLocation::Cuda { gpu_id } = dev.location() {
            match (pos_by_card.get(&gpu_id), layer.graph_pos_i32.as_ref()) {
                (Some(written), None) => layer.graph_pos_i32 = Some(written.clone()),
                (Some(_), Some(_)) => {}
                (None, held) => {
                    let row = Tensor::new(&[pos as i32], &dev)?;
                    let t = match held {
                        Some(t) => {
                            t.slice_set(&row, 0, 0)?;
                            t.clone()
                        }
                        None => row.force_contiguous()?,
                    };
                    layer.graph_pos_i32 = Some(t.clone());
                    pos_by_card.insert(gpu_id, t);
                }
            }
        }
    }
    // RoPE: only update if this layer is the rope-group leader.
    if leaders.rope {
        layer.update_rope_buffers(pos).map_err(|e| {
            crate::tensor::Error::msg(format!("update_rope_buffers layer {i}: {e}"))
        })?;
    }

    // Padded mask: only the mask-group leader does alloc/slice_set.
    if !skip_fdtype_state && leaders.mask {
        let max_kv = layer.kv_cache.max_seq_len_padded();
        // `pos` writes into mask[..., pos] via slice_set. If the
        // existing mask is narrower than `pos+1`, slice_set fails.
        // (kv_cache.max_seq_len_padded() can auto-grow inside
        // append_padded, but padded_mask doesn't follow - its
        // captured device pointer is frozen at the original cap.)
        // Either condition means the captured graph state is
        // stale; bail so the engine can invalidate the graph and
        // fall through to the non-graph forward path for the
        // rest of the request.
        let mask_width = layer
            .padded_mask
            .as_ref()
            .and_then(|m| m.dim(3).ok())
            .unwrap_or(max_kv);
        // A cache that grew since the mask was made moved its buffers, which a
        // recorded graph still names.
        if pos >= mask_width || mask_width != max_kv {
            return Err(crate::tensor::Error::msg(format!(
                "graph_state: pos {} >= padded_mask width {} (kv_cache cap {}, layer {}). \
                 Caller should invalidate the captured graph.",
                pos, mask_width, max_kv, i
            )));
        }
        if layer.padded_mask.is_none() {
            let mut data = vec![f32::NEG_INFINITY; max_kv];
            // First (pos + 1) entries: visible (mask = 0); rest stays -INF.
            data[..=pos].fill(0.0);
            layer.padded_mask = Some(
                Tensor::new(&data[..], &dev)?
                    .reshape((1, 1, 1, max_kv))?
                    .force_contiguous()?,
            );
        } else {
            let mask = layer.padded_mask.as_ref().unwrap();
            let zero = Tensor::zeros_on((1, 1, 1, 1), crate::tensor::DType::F32, &mask.device())?;
            mask.slice_set(&zero, 3, pos).map_err(|e| {
                crate::tensor::Error::msg(format!("padded_mask slice_set layer {i} pos={pos}: {e}"))
            })?;
        }
    }

    // Graph KV pos: same leader-only pattern.
    if !skip_fdtype_state && leaders.kv_pos {
        let kv_shape = (1usize, layer.n_kv_head, 1usize, layer.head_dim);
        let numel = layer.n_kv_head * layer.head_dim;
        let filled_cpu = Tensor::from_vec(vec![pos as i64; numel], kv_shape, &Device::Cpu)?;
        if layer.graph_kv_pos.is_none() {
            layer.graph_kv_pos = Some(filled_cpu.to_device(&dev)?);
        } else {
            let filled_gpu = filled_cpu.to_device(&dev)?;
            layer
                .graph_kv_pos
                .as_ref()
                .unwrap()
                .slice_set(&filled_gpu, 0, 0)?;
        }
    }

    // Each quantised cache names the slot this replay writes (its own device
    // position; the copy is skipped when the slot has not changed).
    #[cfg(feature = "gpu")]
    if let Some(c) = layer.quant_kv.as_mut() {
        c.update_graph_state(pos).map_err(|e| {
            crate::tensor::Error::msg(format!("KV update_graph_state layer {i} pos={pos}: {e}"))
        })?;
    }

    // Persistent attention intermediates for the GQA fast path
    // under graph mode. Allocate once on first call; subsequent
    // calls leave them in place (slice_set wiring in
    // padded_standard_attention writes into them).
    //
    // Two shape variants:
    //   1. GQA fast path (n_rep > 1, q[2]==1):
    //      qk:  [1, n_kv_head, n_rep, max_kv]
    //      out: [1, n_kv_head, n_rep, head_dim]
    //   2. Non-GQA fallback (n_rep == 1, q[2]==1):
    //      qk:  [1, n_head, 1, max_kv]
    //      out: [1, n_head, 1, head_dim]
    //
    // Previously only allocated for variant 1 when HD>256
    // (gemma4:latest Global). Extended to also
    // allocate for variant 2 (moondream/phi2-style models with
    // no GQA) - those were silently allocating fresh per call
    // and causing CUDA_ERROR_ILLEGAL_ADDRESS at first replay.
    //
    // For moondream (n_head=24, n_kv_head=24, HD=64, max_kv=2048):
    //   qk:  1 x 24 x 1 x 2048 x 4 = 192 KB / layer
    //   out: 1 x 24 x 1 x 64 x 4   =   6 KB / layer
    // x 24 layers ≈ 4.8 MB total.
    let layer_ref = &*layer;
    let n_rep = layer_ref.n_head / layer_ref.n_kv_head;
    let needs_attn_buffers = (n_rep > 1 && layer_ref.head_dim > 256) || n_rep == 1;
    if needs_attn_buffers {
        let max_kv = layer.kv_cache.max_seq_len_padded();
        let hd = layer.head_dim;
        let (qk_shape, out_shape) = if n_rep > 1 {
            let n_kv = layer.n_kv_head;
            ((1, n_kv, n_rep, max_kv), (1, n_kv, n_rep, hd))
        } else {
            // Non-GQA: padded_standard_attention falls into the
            // repeat_kv path (which is a no-op at n_rep=1) and
            // produces att/out at shape [1, n_head, 1, *].
            let n_h = layer.n_head;
            ((1, n_h, 1, max_kv), (1, n_h, 1, hd))
        };
        if layer.graph_attn_qk_buffer.is_none() {
            layer.graph_attn_qk_buffer =
                Some(Tensor::zeros_on(qk_shape, crate::tensor::DType::F32, &dev)?);
        }
        // Attention-internal wiring: broadcast_add + softmax
        // outputs same shape as graph_attn_qk_buffer.
        if layer.graph_attn_mask_added_buffer.is_none() {
            layer.graph_attn_mask_added_buffer =
                Some(Tensor::zeros_on(qk_shape, crate::tensor::DType::F32, &dev)?);
        }
        if layer.graph_attn_softmax_buffer.is_none() {
            layer.graph_attn_softmax_buffer =
                Some(Tensor::zeros_on(qk_shape, crate::tensor::DType::F32, &dev)?);
        }
        if layer.graph_attn_out_buffer.is_none() {
            layer.graph_attn_out_buffer = Some(Tensor::zeros_on(
                out_shape,
                crate::tensor::DType::F32,
                &dev,
            )?);
        }
        // Per-layer post-attention intermediate buffers.
        // Allocated only for non-GQA F-dtype graph mode (phi2-style)
        // for now - Q4/Q8 paths bypass these via integrated kernels.
        // Routed in compute_from_kv parallel_attn branch.
        // extend buffer allocation to gemma4-class
        // (GQA n_rep>1 with serial-attn + load-time-concat'd
        // gate||up). The graph_attn_proj_buffer must use the
        // ACTUAL attn_output weight row dim (hidden_size), NOT
        // n_head*head_dim - for gemma4 these differ
        // (attn_out_dim=4096, hidden_size=2560). Bug bit the
        // A probe; corrected here.
        let gemma4_class = n_rep > 1 && !layer.flags.parallel_attn && layer.ffn_gate.is_none();
        if n_rep == 1 || gemma4_class {
            let hidden = if gemma4_class {
                // Use attn_output weight's row dim for the
                // projection output shape.
                layer
                    .attn_output
                    .qtensor()
                    .map(|qt| qt.shape().dims()[0])
                    .unwrap_or(layer.n_head * layer.head_dim)
            } else {
                layer.n_head * layer.head_dim
            };
            let intermediate = layer.flags.intermediate_size.max(1);
            if layer.graph_attn_proj_buffer.is_none() {
                layer.graph_attn_proj_buffer = Some(Tensor::zeros_on(
                    (1, 1, hidden),
                    crate::tensor::DType::F32,
                    &dev,
                )?);
            }
            // gemma4-specific concat ffn_up output: [1, 1, 2N]
            if gemma4_class && layer.graph_ffn_up_concat_buffer.is_none() {
                layer.graph_ffn_up_concat_buffer = Some(Tensor::zeros_on(
                    (1, 1, 2 * intermediate),
                    crate::tensor::DType::F32,
                    &dev,
                )?);
            }
            // gemma4 post_ffn_norm + add output: [1, 1, hidden]
            if gemma4_class && layer.graph_post_ffn_norm_buffer.is_none() {
                layer.graph_post_ffn_norm_buffer = Some(Tensor::zeros_on(
                    (1, 1, hidden),
                    crate::tensor::DType::F32,
                    &dev,
                )?);
            }
            // gemma4 PLE block buffers - only allocate when the
            // PLE block exists for this layer (ple_inp_gate set).
            if gemma4_class && layer.ple_inp_gate.is_some() {
                // ple_dim from the gate weight's row dim: weight
                // shape is [ple_dim, hidden].
                let ple_dim = layer
                    .ple_inp_gate
                    .as_ref()
                    .and_then(|m| m.qtensor().map(|qt| qt.shape().dims()[0]))
                    .unwrap_or(0);
                if ple_dim > 0 {
                    if layer.graph_ple_gate_buffer.is_none() {
                        layer.graph_ple_gate_buffer = Some(Tensor::zeros_on(
                            (1, 1, ple_dim),
                            crate::tensor::DType::F32,
                            &dev,
                        )?);
                    }
                    if layer.graph_ple_gelu_buffer.is_none() {
                        layer.graph_ple_gelu_buffer = Some(Tensor::zeros_on(
                            (1, 1, ple_dim),
                            crate::tensor::DType::F32,
                            &dev,
                        )?);
                    }
                    // per-layer PLE input slot for
                    // graph mode. Engine populates via
                    // populate_ple_input_buffers(input_ids) BEFORE
                    // begin_capture; captured graph reads through.
                    if layer.graph_ple_input_buffer.is_none() {
                        layer.graph_ple_input_buffer = Some(Tensor::zeros_on(
                            (1, 1, ple_dim),
                            crate::tensor::DType::F32,
                            &dev,
                        )?);
                    }
                }
                if layer.graph_ple_proj_buffer.is_none() {
                    layer.graph_ple_proj_buffer = Some(Tensor::zeros_on(
                        (1, 1, hidden),
                        crate::tensor::DType::F32,
                        &dev,
                    )?);
                }
                if layer.graph_ple_final_buffer.is_none() {
                    layer.graph_ple_final_buffer = Some(Tensor::zeros_on(
                        (1, 1, hidden),
                        crate::tensor::DType::F32,
                        &dev,
                    )?);
                }
            }
            // post_attn_norm + ffn_norm + attn_norm buffers
            // for gemma4. All same shape [1, 1, hidden].
            if gemma4_class {
                if layer.graph_post_attn_norm_buffer.is_none() {
                    layer.graph_post_attn_norm_buffer = Some(Tensor::zeros_on(
                        (1, 1, hidden),
                        crate::tensor::DType::F32,
                        &dev,
                    )?);
                }
                if layer.graph_x_norm_ffn_buffer.is_none() {
                    layer.graph_x_norm_ffn_buffer = Some(Tensor::zeros_on(
                        (1, 1, hidden),
                        crate::tensor::DType::F32,
                        &dev,
                    )?);
                }
                if layer.graph_x_norm_attn_buffer.is_none() {
                    layer.graph_x_norm_attn_buffer = Some(Tensor::zeros_on(
                        (1, 1, hidden),
                        crate::tensor::DType::F32,
                        &dev,
                    )?);
                }
            }
            if layer.graph_ffn_up_buffer.is_none() {
                layer.graph_ffn_up_buffer = Some(Tensor::zeros_on(
                    (1, 1, intermediate),
                    crate::tensor::DType::F32,
                    &dev,
                )?);
            }
            if layer.graph_ffn_activated_buffer.is_none() {
                layer.graph_ffn_activated_buffer = Some(Tensor::zeros_on(
                    (1, 1, intermediate),
                    crate::tensor::DType::F32,
                    &dev,
                )?);
            }
            if layer.graph_ffn_down_buffer.is_none() {
                layer.graph_ffn_down_buffer = Some(Tensor::zeros_on(
                    (1, 1, hidden),
                    crate::tensor::DType::F32,
                    &dev,
                )?);
            }
            if layer.graph_phi2_merge_buffer.is_none() {
                layer.graph_phi2_merge_buffer = Some(Tensor::zeros_on(
                    (1, 1, hidden),
                    crate::tensor::DType::F32,
                    &dev,
                )?);
            }
            // Path B (gemma4 graph-mode unblock):
            // 3 additional buffers for compute_from_kv serial path.
            if gemma4_class {
                if layer.graph_ffn_gate_out.is_none() {
                    layer.graph_ffn_gate_out = Some(Tensor::zeros_on(
                        (1, 1, intermediate),
                        crate::tensor::DType::F32,
                        &dev,
                    )?);
                }
                if layer.graph_x_out_scaled.is_none() {
                    layer.graph_x_out_scaled = Some(Tensor::zeros_on(
                        (1, 1, hidden),
                        crate::tensor::DType::F32,
                        &dev,
                    )?);
                }
                // graph_attn_scaled_buffer only for arches with residual_scale.
                // gemma4:latest doesn't have rs but gemma4:26b does. Allocate
                // unconditionally for gemma4_class - small (10 KB x 42 layers ≈
                // 420 KB) - harmless if unused.
                if layer.graph_attn_scaled_buffer.is_none() {
                    layer.graph_attn_scaled_buffer = Some(Tensor::zeros_on(
                        (1, 1, hidden),
                        crate::tensor::DType::F32,
                        &dev,
                    )?);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(from: usize, to: usize) -> std::ops::Range<usize> {
        from..to
    }

    fn run(seg: usize, layers: std::ops::Range<usize>, head: bool) -> StepOp {
        StepOp::Run {
            seg,
            layers,
            head,
            recv: false,
            send: false,
        }
    }

    fn hop(seg: usize, send: bool, recv: bool) -> StepOp {
        StepOp::Hop { seg, send, recv }
    }

    #[test]
    fn a_step_of_recorded_layers_is_one_run_ending_in_the_head() {
        let ops = plan_step(&[span(0, 6)], |_| true);
        assert_eq!(ops, vec![run(0, 0..6, true)]);
    }

    #[test]
    fn a_layer_that_is_not_recorded_splits_the_run() {
        let ops = plan_step(&[span(0, 8)], |l| l != 3 && l != 4);
        assert_eq!(
            ops,
            vec![
                run(0, 0..3, false),
                StepOp::Layer { seg: 0, layer: 3 },
                StepOp::Layer { seg: 0, layer: 4 },
                run(0, 5..8, true),
            ]
        );
    }

    #[test]
    fn a_last_layer_that_is_not_recorded_leaves_the_head_alone() {
        let ops = plan_step(&[span(0, 4)], |l| l != 3);
        assert_eq!(
            ops,
            vec![
                run(0, 0..3, false),
                StepOp::Layer { seg: 0, layer: 3 },
                StepOp::Head { seg: 0 },
            ]
        );
        assert!(ops.last().is_some_and(StepOp::captured));
        assert!(!ops[1].captured());
    }

    #[test]
    fn a_first_layer_that_is_not_recorded_starts_the_step() {
        let ops = plan_step(&[span(0, 3)], |l| l != 0);
        assert_eq!(
            ops,
            vec![StepOp::Layer { seg: 0, layer: 0 }, run(0, 1..3, true)]
        );
    }

    #[test]
    fn a_hop_stands_between_two_cards() {
        let ops = plan_step(&[span(0, 3), span(3, 6)], |l| l != 2 && l != 3);
        assert_eq!(
            ops,
            vec![
                run(0, 0..2, false),
                StepOp::Layer { seg: 0, layer: 2 },
                hop(0, true, true),
                StepOp::Layer { seg: 1, layer: 3 },
                run(1, 4..6, true),
            ]
        );
    }

    /// Where a graph sits beside a hop it holds the copy: the hidden state leaves one recording
    /// and enters the next with only the event between them.
    #[test]
    fn a_hop_between_two_runs_is_held_by_their_graphs() {
        let ops = plan_step(&[span(0, 3), span(3, 6)], |_| true);
        assert_eq!(
            ops,
            vec![
                StepOp::Run {
                    seg: 0,
                    layers: 0..3,
                    head: false,
                    recv: false,
                    send: true
                },
                hop(0, false, false),
                StepOp::Run {
                    seg: 1,
                    layers: 3..6,
                    head: true,
                    recv: true,
                    send: false
                },
            ]
        );
        let ops = plan_step(&[span(0, 3), span(3, 6)], |l| l != 3);
        assert!(matches!(
            ops[..],
            [
                StepOp::Run { send: true, .. },
                StepOp::Hop {
                    send: false,
                    recv: true,
                    ..
                },
                StepOp::Layer { .. },
                StepOp::Run { .. },
            ]
        ));
    }

    #[test]
    fn a_step_of_no_recorded_layer_is_every_layer_then_the_head() {
        let ops = plan_step(&[span(0, 2)], |_| false);
        assert_eq!(
            ops,
            vec![
                StepOp::Layer { seg: 0, layer: 0 },
                StepOp::Layer { seg: 0, layer: 1 },
                StepOp::Head { seg: 0 },
            ]
        );
        assert!(plan_step(&[], |_| true).is_empty());
    }
}
