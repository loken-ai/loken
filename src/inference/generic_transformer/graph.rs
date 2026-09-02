//! Split out of `inference/generic_transformer/` (move-only refactor).

#[allow(unused_imports)]
use super::*;

impl GenericHeteroTransformer {
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
        self.layers.iter().find_map(|l| {
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
                // gemma4 still blocked - see recapture_each_token.
                // tested flipping to true unconditionally.
                // deepcoder regressed -3.2 % medium / -3.5 % long: graph
                // capture overhead per replay (2744 nodes for 48-layer
                // model) exceeded the launch-savings on long-seq Q8.
                //
                // re-enabled SPECIFICALLY for phi2-class
                // (parallel_attn + layer_norm_with_bias). For phi2 the
                // non-graph dev_pos kernel chain still does 4 launches
                // per layer (post-Phase-2 fused softmax+output is 2
                // launches, but the engine wraps several more around
                // it). CUDA graphs eliminate per-launch overhead, and
                // the dev_pos kernels are designed graph-safe (read
                // seq_kv from cur_pos_dev, write at fixed
                // max_seq_padded stride). Other Q8 arches still bail
                // because their dispatch is gated to `use_graph_ops`
                // and the deepcoder regression analysis still holds
                // for them.
                //
                // re-tested with dynamic max_seq_padded now
                // in tree (commit 8019b06). Capture succeeded (2360 nodes
                // for deepcoder qwen2 40-layer), but bench showed ZERO
                // perf delta (75.1 vs 75.8 tok/s, ±0.3 std). Launch
                // overhead in warm-stream decode is already ~0.5-1µs/launch,
                // so eliminating it via graph capture doesn't move the
                // needle. The deepcoder -10% gap is some other cost
                // (Q8 K cache read bandwidth or matmul scheduling).
                // Leaving non-phi2 gated to avoid graph-capture surface
                // area without any upside.
                //
                // re-opened for SMALL dense Q8 models
                // (`q8_graph_small_dense_ok`, embedding <= 2048 on the
                // qwen2/qwen3/llama whitelist). Those decodes are
                // LAUNCH-BOUND (qwen3:0.6b eager loses ~6-9 % to
                // ollama's graph replay); the historical "zero delta"
                // above was measured on 14B where kernels dominate, and
                // pre-dates the capture work (model-
                // stream capture + event-tracking-off + capture arena).
                // Correctness prerequisites now in tree:
                //   - host KV bookkeeping synced per replayed token
                //     (`sync_kv_len_for_graph`, engine replay loop);
                //   - capture-frozen seq ceiling enforced per token
                //     (`Q8KvCache::graph_seq_limit` checked in
                //     update_graph_state -> engine invalidates and
                //     re-captures/eager-falls-back at the wider state);
                //   - probation (replay argmax vs eager) + 0-node
                //     defense at the engine catch anything else.
                let phi2_dev_pos =
                    self.config.flags.parallel_attn && self.config.flags.layer_norm_with_bias;
                if self.q8_graph_small_dense_ok() {
                    return None;
                }
                // gemma4 graph mode: the capture crash is solved (capture
                // arena), but the split path (prepare_kv/compute_from_kv)
                // reads the F-dtype kv_cache while the normal forward uses
                // Q8+GH_APQ - a numerical mismatch (append vs append_padded
                // + Q8-vs-F-dtype) that diverges at layer-0 attention.
                // Gated OFF until the normal forward is captured directly
                // (the arena makes the split-path workaround unnecessary).
                // Details + reframe in memory: gemma4 graph split-path notes.
                // DEFINITIVELY DEAD: measured gemma4 graph-on decode at
                // 26.6 tok/s vs 142.4 graph-off = 5.3x SLOWER (split-path
                // capture/replay overhead). Even with correct output it would
                // never be enabled. Do NOT re-attempt gemma4 graph mode.
                return (!phi2_dev_pos).then_some(
                    "Q8 KV cache on an arch that measured slower captured than eager",
                );
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
        self.layers.iter().any(|l| {
            // Any layer that has no Q4/Q8 KV cache AND is not a
            // dual-populate donor falls back to F-dtype scatter_set,
            // which is the unstable path.
            !l.has_q4_cache() && !l.has_q8_cache() && !l.populate_dual_kv
        })
    }

    /// Small dense Q8-KV models whose decode is LAUNCH-BOUND - the class
    /// where CUDA-graph replay pays (qwen3:0.6b measured 581-605 tok/s
    /// eager vs ollama ~640 graph-replayed). Larger Q8 models (deepcoder
    /// 14B, qwen3:8b) measured ~zero graph delta historically (kernels
    /// dominate) and stay on the proven eager path until re-benched.
    /// gemma4 is excluded by arch (its graph mode is measured 5.3x
    /// SLOWER - see kv_state_graph_safe comments; do not re-attempt).
    /// `granite` joins the set on the same reasoning that admitted the others:
    /// it is genuinely dense, its embedding is inside the bound, and the main
    /// capture allow-list already carries it - only this list's name check kept
    /// it out, and it decoded well below the reference in both streaming modes.
    /// The MoE members of that family stay out: this path is for dense decode,
    /// and a routed expert set allocates per token, which a capture cannot
    /// contain. Each name here is a measured admission, never a merge of the
    /// two lists - gemma4 sits in the allow-list and belongs OUT of this one.
    pub fn q8_graph_small_dense_ok(&self) -> bool {
        matches!(
            self.config.arch.as_str(),
            "qwen2" | "qwen3" | "llama" | "granite"
        ) && self.config.embedding_length <= 2048
    }

    /// Drop the F16 lm_head when nothing can use it.
    ///
    /// The loader materializes the output projection twice on the primary card:
    /// the quantized `QMatMul` that decode prefers, and an F16 dequant of the
    /// same weight. Only `forward_graph` requires the F16 form - every other
    /// reader falls back to the quantized one - so when capture is refused the
    /// F16 copy is resident for a path that never runs, and a wide vocabulary
    /// makes that over a gigabyte of the card that decides whether the model
    /// fits without being split.
    ///
    /// The refusal is read from `graph_capture_decision`, never re-derived: a
    /// second copy of that rule would drift from the first and silently free a
    /// weight the graph path still needs.
    pub(crate) fn release_graph_only_lm_head(mut self) -> Self {
        if self.output_proj_cuda.is_some()
            && self.output_proj_cuda_qmm.is_some()
            && self.graph_capture_decision().is_err()
        {
            self.output_proj_cuda = None;
        }
        self
    }

    /// Whether decode captures a CUDA graph, and when it does not, WHY.
    ///
    /// "no capture" and "capture that logs nothing" used to be indistinguishable from the
    /// outside, so a model could sit in the allow-list and never be captured without anyone
    /// noticing - which is exactly what happened to olmoe. The reason is returned rather
    /// than logged here so the caller decides where it surfaces; `graph_capture_auto_on`
    /// keeps the boolean contract for existing callers.
    pub fn graph_capture_decision(&self) -> std::result::Result<(), &'static str> {
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
        // Q4-KV dev-pos models - the ONE path whose engine-site capture +
        // replay is PROVEN correct within a request (deepcoder
        // KV=q4: 1636-node graph, 0 MEM_ALLOC, coherent 128-token decode
        // token-matching the Q8 eager reference, probation green) - are
        // still gated OFF because replayed tokens advance the KV cache
        // only DEVICE-side (the captured append kernel reads cur_pos_dev,
        // refreshed by update_graph_state): the Q4 cache's HOST state
        // (current_seq_len + block/residual bookkeeping) goes stale for
        // the entire replayed span, so the NEXT request's trim/reset/
        // prefill operates on wrong lengths. Server repro (// LOKEN_KV_QUANT=q4 deepcoder): request 2+ with fresh prompts
        // 500s with `broadcast_as: cannot broadcast [10, 11] to
        // [1, 40, 10, 30]`; with an identical prompt (prefix reuse) it
        // silently decodes garbage. Note `trim_kv` also never trims
        // q4_kv_cache. Re-enabling needs a host/device KV-accounting sync
        // for replayed tokens (host current_seq_len/block state advanced
        // per replay, or derived from cur_pos_dev at request end) plus a
        // q4 trim in trim_kv - model-side work, not capture mechanics.
        // Single-request verification harness: llm_render deepcoder:14b
        // LOKEN_KV_QUANT=q4 with this gate removed.
        #[cfg(feature = "cuda")]
        if self.layers.iter().any(|l| l.has_q4_cache()) {
            return Err("Q4 KV cache");
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
        // Whitelist by arch. `kv_state_graph_safe()` already filters
        // out HD=512 F-dtype-only layers (which crash on graph replay
        // - gemma4:latest is the known case), so the arch check here
        // just selects bench-validated arches.
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
        let arch = self.config.arch.as_str();
        // EXPERIMENTAL: include phi2 to test whether the per-layer
        // stable intermediate buffers close the
        // ILLEGAL_ADDRESS replay crash for arches with capture-once
        // buffers. phi2 is excluded - both the recapture path AND
        // the non-graph fallback have bugs:
        //   - cudaGraphExecUpdate returns FAILURE (phi2 changes graph
        //     STRUCTURE each token, not just params; update can only
        //     patch params)
        //   - Post-clear fallback hits "cannot broadcast [6,32] to
        //     [1,32,6,33]" - separate shape-tracking bug in phi2's
        //     non-graph path
        // Both need multi-day fixes. Leave phi2 in non-graph mode
        // (its current -49% LOSS state).
        // phi2 BACK OFF the whitelist after measuring that
        // F-dtype KV (now default after the auto-promote revert) is
        // significantly faster than Q8 KV (326 tok/s vs 262 tok/s on
        // moondream short - bench above). Graph mode for F-dtype phi2
        // is blocked by the original ILLEGAL_ADDRESS bug
        // which the graph-capture
        // dev_slot work only addressed for Q8. Leaving phi2 OUT of
        // auto-on so F-dtype runs on the unbroken non-graph path.
        // Re-enable phi2 once either (a) F-dtype graph capture is
        // fixed at the engine level, or (b) Q8 becomes faster than
        // F-dtype on this shape via further kernel work.
        // The allow-list is the performance dividing line, not a property of the
        // architectures: every model measured against ollama that LOSES sits outside it
        // (olmoe -21%, granitemoe -18%, granite -18%, olmo2 -9%) and every large win sits
        // inside it (qwen3moe +181%, llama +22%, qwen3 +27%). Without capture, decode
        // submits 717 kernels per token for 170ms of GPU work in a 2272ms window - the
        // device computes 7.5% of the time and the host burns a core issuing launches.
        // qwen3moe already proves a mixture captures correctly, so expert routing is not
        // the obstacle it was assumed to be.
        if !matches!(
            arch,
            "gemma4"
                | "qwen2"
                | "qwen3"
                | "qwen3moe"
                | "llama"
                | "granitemoe"
                | "granite"
                | "olmoe"
                | "olmo2"
        ) {
            return Err("architecture not in the capture allow-list");
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
    fn compute_graph_state_groups(&self) -> GraphStateGroups {
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
                    let key = (device_id(i), self.layers[i].kv_cache.max_seq_len_padded());
                    *leaders.entry(key).or_insert(i)
                })
                .collect()
        };
        let kv_pos: Vec<usize> = {
            let mut leaders: HashMap<(usize, usize, usize), usize> = HashMap::new();
            (0..n_layers)
                .map(|i| {
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
    pub fn invalidate_graph_state(&mut self) {
        for l in &mut self.layers {
            l.padded_mask = None;
            l.graph_rope_cos = None;
            l.graph_rope_sin = None;
            l.graph_kv_pos = None;
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
            l.graph_attn_mask_added_buffer = None;
            l.graph_attn_softmax_buffer = None;
            l.graph_ffn_gate_out = None;
            l.graph_x_out_scaled = None;
            l.graph_attn_scaled_buffer = None;
            // ROOT CAUSE FIX: drop captured-region tensor
            // lifetime-extension handles so cuMemFree can reclaim
            // their memory now that the graph is being torn down.
            l.graph_alive_tensors.clear();
        }
        self.graph_logits_buffer = None;
        self.graph_hidden_buffer = None;
        // ROOT CAUSE FIX: drop model-level captured-region tensor handles
        // so cuMemFree can reclaim their memory now that graph is torn down.
        self.graph_alive_tensors_model.clear();
        // Release the Q8 capture bookkeeping: with the graph gone the
        // buffers may grow again and the frozen seq ceiling no longer
        // applies (`update_graph_state` would otherwise keep erroring).
        #[cfg(feature = "cuda")]
        for l in &mut self.layers {
            if let Some(c) = l.q8_kv_cache.as_mut() {
                c.clear_graph_captured();
            }
        }
    }

    /// Engine hook: a CUDA graph referencing the per-layer KV buffers is
    /// now live. Freezes each Q8 cache's growth + seq ceiling (see
    /// `Q8KvCache::mark_graph_captured`). Paired with
    /// `invalidate_graph_state` which clears it on teardown.
    #[cfg(feature = "cuda")]
    pub fn mark_graph_captured(&mut self) {
        for l in &mut self.layers {
            if let Some(c) = l.q8_kv_cache.as_mut() {
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
    #[cfg(feature = "cuda")]
    pub fn sync_kv_len_for_graph(&mut self, len: usize) {
        for l in &mut self.layers {
            if let Some(c) = l.q8_kv_cache.as_mut() {
                c.set_seq_len_for_graph(len);
            }
        }
    }

    pub fn update_graph_state(&mut self, pos: usize) -> Result<()> {
        let dev = self
            .layers
            .first()
            .map(|l| l.cos.device().clone())
            .unwrap_or(Device::Cpu);

        // Captured-graph seq ceiling (Q8 path): the live graph was
        // recorded against a frozen buffer capacity and (if the
        // fixed-stride score chain was captured) a frozen max_seq_padded
        // stride. Once `pos + 1` outgrows it, error out so the engine
        // invalidates the graph - the request then continues on the
        // eager path (or re-captures at the wider state).
        #[cfg(feature = "cuda")]
        for (i, l) in self.layers.iter().enumerate() {
            if let Some(limit) = l.q8_kv_cache.as_ref().and_then(|c| c.graph_seq_limit()) {
                if pos + 1 > limit {
                    return Err(crate::tensor::Error::msg(format!(
                        "graph_state: pos {} outgrew captured Q8 seq ceiling {} (layer {}). \
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
        if !self.workspace_pre_grown {
            self.workspace_pre_grown = true;
            #[cfg(feature = "cuda")]
            if let Some(cuda_dev) = self.cuda_devices.values().next() {
                if let Ok(cd) = cuda_dev.as_cuda_device() {
                    let max_intermediate = self
                        .layers
                        .iter()
                        .map(|l| l.flags.intermediate_size)
                        .max()
                        .unwrap_or(self.config.embedding_length);
                    let max_k = std::cmp::max(self.config.embedding_length, max_intermediate);
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
        let skip_fdtype_state = self
            .layers
            .iter()
            .all(|l| (l.has_q8_cache() || l.has_q4_cache()) && !l.populate_dual_kv);

        // Second pass: update each leader, then assign Arc-clones to followers.
        for i in 0..n_layers {
            // RoPE: only update if this layer is the rope-group leader.
            if rope_group[i] == i {
                let layer = &mut self.layers[i];
                layer.update_rope_buffers(pos).map_err(|e| {
                    crate::tensor::Error::msg(format!("update_rope_buffers layer {i}: {e}"))
                })?;
            }

            // Padded mask: only the mask-group leader does alloc/slice_set.
            if !skip_fdtype_state && mask_group[i] == i {
                let layer = &mut self.layers[i];
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
                if pos >= mask_width {
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
                    let zero =
                        Tensor::zeros_on((1, 1, 1, 1), crate::tensor::DType::F32, &mask.device())?;
                    mask.slice_set(&zero, 3, pos).map_err(|e| {
                        crate::tensor::Error::msg(format!(
                            "padded_mask slice_set layer {i} pos={pos}: {e}"
                        ))
                    })?;
                }
            }

            // Graph KV pos: same leader-only pattern.
            if !skip_fdtype_state && kv_pos_group[i] == i {
                let layer = &mut self.layers[i];
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

            // Q4 KV cache device-pos prime is per-layer (each cache has its
            // own cur_pos_dev; the value is small (i32) and the htod copy
            // is ~free; not worth deduping).
            #[cfg(feature = "cuda")]
            if let Some(q4) = self.layers[i].q4_kv_cache.as_mut() {
                q4.update_graph_state(pos).map_err(|e| {
                    crate::tensor::Error::msg(format!(
                        "Q4KvCache::update_graph_state layer {i} pos={pos}: {e}"
                    ))
                })?;
            }

            // Q8 KV cache mirror - same per-layer pattern as Q4 above.
            // Without this, `try_q8_graph_decode` in forward_attn never
            // engages (cur_pos_dev stays None) and graph mode falls back
            // to the F-dtype path which OOMs on full-context KV alloc.
            #[cfg(feature = "cuda")]
            if let Some(q8) = self.layers[i].q8_kv_cache.as_mut() {
                q8.update_graph_state(pos).map_err(|e| {
                    crate::tensor::Error::msg(format!(
                        "Q8KvCache::update_graph_state layer {i} pos={pos}: {e}"
                    ))
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
            let layer_ref = &self.layers[i];
            let n_rep = layer_ref.n_head / layer_ref.n_kv_head;
            let needs_attn_buffers = (n_rep > 1 && layer_ref.head_dim > 256) || n_rep == 1;
            if needs_attn_buffers {
                let layer = &mut self.layers[i];
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
                let gemma4_class =
                    n_rep > 1 && !layer.flags.parallel_attn && layer.ffn_gate.is_none();
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
        }

        // Third pass: followers Arc-clone the group leader's tensors. This
        // makes each layer's `padded_mask` / `graph_kv_pos` / rope buffers
        // point at the SAME underlying CudaSlice as the leader, so subsequent
        // slice_sets by the leader are visible to all followers and the
        // captured graph reads from one shared device pointer per group.
        for i in 0..n_layers {
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

    /// Graph-compatible decode forward for CUDA graph capture/replay.
    ///
    /// Single-token (seq_len == 1), single-CUDA-device model only. Caller
    /// must have invoked `update_graph_state(pos)` immediately prior.
    ///
    /// Returns logits at the captured output buffer. On replay this is
    /// stored at a stable device pointer (the captured output of the last
    /// op), so the engine can sample from it after `graph.launch()`.
    pub fn forward_graph(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, seq_len) = input_ids.dims2()?;
        debug_assert_eq!(seq_len, 1, "forward_graph is decode-only");

        // Embedding lookup on the table's device (CPU today, Path A may
        // move small tables to GPU). The result lands on the same device;
        // a subsequent .to_device() places it on the layer-0 GPU.
        let emb_device = self.embeddings.embeddings().device().clone();
        let input_local;
        let input_ids = if input_ids.device().same_device(&emb_device) {
            input_ids
        } else {
            input_local = input_ids.to_device(&emb_device)?;
            &input_local
        };
        let hidden_cpu = self.embeddings.forward(input_ids)?;
        let hidden_cpu = if let Some(scale) = self.config.embed_scale {
            (hidden_cpu * scale)?
        } else {
            hidden_cpu
        };

        // All layers live on one CUDA device in graph mode.
        let target_dev = self
            .cuda_devices
            .values()
            .next()
            .ok_or_else(|| crate::tensor::Error::msg("forward_graph requires a CUDA device"))?
            .clone();
        let mut hidden = hidden_cpu.to_device(&target_dev)?;

        // See the timer in the eager decode loop: this measures issuing the layer,
        // which is what tells a dispatching layer from one running on the host.
        // Layer-by-layer graph-compatible path.
        for (li, layer) in self.layers.iter_mut().enumerate() {
            let _timer = crate::inference::place::layer_perf::LayerTimer::start(li);
            hidden = layer.forward_graph(&hidden)?;
        }

        // Final norm + output projection on CUDA (fast path only - no CPU
        // fallback for graph mode; the engine rejects before capture if
        // output_proj_cuda is absent).
        let norm_w = self.output_norm_cuda_weight.as_ref().ok_or_else(|| {
            crate::tensor::Error::msg("forward_graph needs output_norm_cuda_weight")
        })?;
        let proj = self
            .output_proj_cuda
            .as_ref()
            .ok_or_else(|| crate::tensor::Error::msg("forward_graph needs output_proj_cuda"))?;
        let last = hidden.i((.., 0, ..))?;
        let last_f32 = last.to_dtype(crate::tensor::DType::F32)?;
        let normed =
            crate::tensor::ops::rms_norm(&last_f32, norm_w, self.config.rms_norm_eps as f32)?;
        let normed_f16 = normed.to_dtype(crate::tensor::DType::F16)?;
        let mut logits = normed_f16.matmul_t(&proj)?;
        logits = logits.to_dtype(crate::tensor::DType::F32)?;
        if let Some(scale) = self.config.logit_scale {
            logits = (logits / scale)?;
        }
        match self.config.final_logit_softcapping {
            Some(cap) => (logits / cap)?.tanh()? * cap,
            None => Ok(logits),
        }
    }

    /// Compute the token embedding and write it into graph_hidden_buffer at
    /// a stable device pointer. Call OUTSIDE graph capture so the embedding
    /// lookup + H2D copy are not part of the captured graph.
    pub fn embed_for_graph(&mut self, input_ids: &Tensor) -> Result<()> {
        let emb_device = self.embeddings.embeddings().device().clone();
        let input_local;
        let input_ids = if input_ids.device().same_device(&emb_device) {
            input_ids
        } else {
            input_local = input_ids.to_device(&emb_device)?;
            &input_local
        };
        let mut hidden = self.embeddings.forward(input_ids)?;
        if let Some(scale) = self.config.embed_scale {
            hidden = (hidden * scale)?;
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
        let _ = self.ensure_hidden_buffer(&hidden_gpu)?;
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
        } else if let Some((ref bf16_tensor, _)) = self.ple_token_embd_bf16 {
            // bf16_tensor lives on CPU (large vocabxple_dimxn_layers table).
            // Move input_ids to its device before index_select.
            let bf16_dev = bf16_tensor.device().clone();
            let flat = input_ids.flatten_all()?;
            let flat = if flat.device().same_device(&bf16_dev) {
                flat
            } else {
                flat.to_device(&bf16_dev)?
            };
            let t = bf16_tensor.index_select(&flat, 0)?;
            let t = t.to_dtype(crate::tensor::DType::F32)?;
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
        let proj_dev = match model_proj {
            crate::tensor::quantized::QMatMul::QTensor(qt) => qt.device().clone(),
            crate::tensor::quantized::QMatMul::Tensor(t) => t.device().clone(),
            crate::tensor::quantized::QMatMul::TensorF16(t) => t.device().clone(),
        };
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
    /// This is the body that gets captured into a CUDA graph.
    pub fn forward_from_hidden(&mut self) -> Result<Tensor> {
        let mut hidden = self
            .graph_hidden_buffer
            .as_ref()
            .ok_or_else(|| {
                crate::tensor::Error::msg("forward_from_hidden: call embed_for_graph first")
            })?
            .clone();
        // See the timer in the eager decode loop: this measures issuing the layer,
        // which is what tells a dispatching layer from one running on the host.

        for (li, layer) in self.layers.iter_mut().enumerate() {
            let _timer = crate::inference::place::layer_perf::LayerTimer::start(li);
            hidden = layer.forward_graph(&hidden)?;
        }

        let norm_w = self.output_norm_cuda_weight.as_ref().ok_or_else(|| {
            crate::tensor::Error::msg("forward_from_hidden needs output_norm_cuda_weight")
        })?;
        let last = hidden.i((.., 0, ..))?;
        let last_f32 = last.to_dtype(crate::tensor::DType::F32)?;
        // Prefer the quantized output proj (Q6_K mvq, F32-out, no F16
        // cast) when available - same fast path the non-graph forward
        // takes. nsys traces showed the F16 cutlass fallback adds
        // ~1ms/token on deepcoder (152Kx5120 GEMM); the Q-quant mvq is
        // ~2.4x faster on the same shape.
        let mut logits = if self.output_proj_cuda_qmm.is_some() {
            let qmm = self.output_proj_cuda_qmm.as_ref().unwrap();
            // Try the fused rms_norm+qmatmul kernel first.
            #[cfg(feature = "cuda")]
            let fused = {
                use crate::tensor::quantized::QMatMul;
                if last_f32.device().is_cuda() && last_f32.dim(crate::tensor::D::Minus1)? <= 16384 {
                    if let QMatMul::QTensor(ref qt) = qmm {
                        crate::inference::moe_cuda::rms_norm_then_qmatmul(
                            &last_f32,
                            norm_w,
                            qt.as_ref(),
                            self.config.rms_norm_eps as f32,
                        )
                        .ok()
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            #[cfg(not(feature = "cuda"))]
            let fused: Option<Tensor> = None;
            if let Some(o) = fused {
                let dims = o.dims();
                if dims.len() == 3 && dims[0] == 1 && dims[1] == 1 {
                    o.squeeze(0)?
                } else {
                    o
                }
            } else {
                let normed = crate::tensor::ops::rms_norm(
                    &last_f32,
                    norm_w,
                    self.config.rms_norm_eps as f32,
                )?;
                qmm.forward(&normed)?
            }
        } else {
            let proj = self.output_proj_cuda.as_ref().ok_or_else(|| {
                crate::tensor::Error::msg(
                    "forward_from_hidden needs output_proj_cuda or output_proj_cuda_qmm",
                )
            })?;
            let normed =
                crate::tensor::ops::rms_norm(&last_f32, norm_w, self.config.rms_norm_eps as f32)?;
            let normed_f16 = normed.to_dtype(crate::tensor::DType::F16)?;
            normed_f16.matmul_t(&proj)?
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
        for layer in &mut self.layers {
            layer.kv_cache.trim_to(new_len);
            // Keep the Q8 mirror in sync with the F-dtype cache; otherwise
            // speculative-decoding rollbacks leave Q8 with stale draft
            // positions and subsequent decode reads garbage.
            //
            // When fully resetting (new_len == 0), call Q8 `reset()`
            // instead of trim_to(0). reset() additionally clears
            // `cur_pos_dev` + `cur_pos_dev_value`, which the dev_pos
            // attention kernels read to derive seq_kv. trim_to only sets
            // current_seq_len -> stale device-side position survives across
            // requests -> cyclic-N non-determinism at temperature=0.
            #[cfg(feature = "cuda")]
            if let Some(c) = layer.q8_kv_cache.as_mut() {
                if new_len == 0 {
                    c.reset();
                } else {
                    c.trim_to(new_len);
                }
            }
            // Q4 KIVI mirror: was MISSING - spec-decode
            // rollbacks and cross-request prefix reuse left the Q4 cache
            // at a stale length, so the next prefill appended at the
            // wrong slots (server repro: request 2+ broadcast-shape 500s
            // with LOKEN_KV_QUANT=q4). trim_to also restores the
            // partial-block K residual; if that fails, fall back to a
            // full reset (correct, just loses the reuse win).
            #[cfg(feature = "cuda")]
            if let Some(c) = layer.q4_kv_cache.as_mut() {
                if new_len == 0 {
                    c.reset();
                } else if let Err(e) = c.trim_to(new_len) {
                    tracing::warn!("Q4 KV trim_to({new_len}) failed: {e}; resetting cache");
                    c.reset();
                }
            }
            // CPU KV stores (dense Q8 / gemma4 F16). Multi-token verify
            // (PLD/EAGLE) appends drafts here too; without trimming them the
            // rejected draft positions accumulate and corrupt later attention
            // (greedy decode diverges a few tokens in). Mirrors the F-dtype trim.
            if let Some(c) = layer.cpu_q8_kv.as_mut() {
                if new_len == 0 {
                    c.reset();
                } else {
                    c.trim_to(new_len);
                }
            }
            if let Some(c) = layer.cpu_f16_kv.as_mut() {
                if new_len == 0 {
                    c.reset();
                } else {
                    c.trim_to(new_len);
                }
            }
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
    #[cfg(feature = "cuda")]
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
            let proj_t = proj.t()?;
            self.graph_alive_tensors_model.push(proj_t.clone());
            let mm_out = normed_f16.matmul(&proj_t)?;
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
                    .map(|t| t.clone())
            })
            .collect::<Result<_>>()?;
        self.compute_all_from_kv(&hidden, &all_q)
    }
}
