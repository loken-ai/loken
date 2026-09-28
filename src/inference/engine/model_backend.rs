//! `ModelBackend`: the per-architecture dispatch trait for the LLM engine.
//!
//! Replaces the old `enum ModelVariant` + ~27 separate `match self { ... }`
//! methods in `inference/engine/llm_engine/mod.rs`. Each supported architecture wraps its concrete
//! model type in a small backend struct implementing this trait; the engine
//! stores a `Box<dyn ModelBackend>` and calls trait methods directly. Adding a
//! model = one new struct + one `impl ModelBackend` (only overriding what the
//! arch actually supports) instead of touching 27 match statements.
//!
//! The trait carries default implementations for every optional capability
//! (graph mode, vision, KV trim, PLD, ...) so most backends only implement
//! `forward` + `take_generic`.

use std::collections::HashSet;

use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::inference::model::moondream::quantized as moondream;
use crate::tensor::{Device, Tensor};

use crate::inference::engine::llm_engine::{
    VisionTower, ADAPTIVE_PREFILL_CHUNK, PREFILL_CHUNK_TOKENS, PREFILL_CHUNK_TOKENS_MIN,
};

/// Boxed backend as stored in `LoadedModelState`.
pub(crate) type BoxedModelBackend = Box<dyn ModelBackend>;

/// Fold a per-layer device list into the `device_layer_distribution` contract:
/// one `(device_label, ordinal, layer_start, layer_end)` entry per contiguous
/// run of layers on one device, both ends INCLUSIVE. The labels are "CPU" and
/// "CUDA" - the exact spellings `get_gpu_portion_size` filters on, so a model
/// that reports through here is counted as resident on the card.
pub(crate) fn group_layers_by_device(
    locations: impl IntoIterator<Item = crate::tensor::DeviceLocation>,
) -> Vec<(String, usize, u32, u32)> {
    use crate::tensor::DeviceLocation;
    let mut out: Vec<(String, usize, u32, u32)> = Vec::new();
    for (i, loc) in locations.into_iter().enumerate() {
        let (label, ordinal) = match loc {
            DeviceLocation::Cuda { gpu_id } => ("CUDA", gpu_id),
            DeviceLocation::Cpu => ("CPU", 0usize),
        };
        let i = i as u32;
        match out.last_mut() {
            Some(last) if last.0 == label && last.1 == ordinal => last.3 = i,
            _ => out.push((label.to_string(), ordinal, i, i)),
        }
    }
    out
}

/// Enum-free dispatch surface over supported quantized model architectures.
///
/// Safety contract carried over from the old `unsafe impl Send for
/// ModelVariant`: the engine's `Mutex` provides exclusive access, preventing
/// concurrent `RefCell` mutation inside the model types - each backend struct
/// declares `unsafe impl Send` on that basis.
/// Starting prefill chunk sized so the per-forward activation working set
/// (~`chunk x widest_ffn x f32 x 2 live buffers`) stays within an L3-fit budget.
/// Calibrated so the widest-FFN dense model (mistral-nemo, ffn=14336) lands at
/// 1024 (measured safe); narrower / MoE models get up to 2048. The only measured
/// win is a prompt whose length lands just past a chunk boundary (a tiny final
/// chunk re-streams the whole weight set for a few tokens) - a bigger chunk folds
/// that remainder into one pass; larger remainders amortise so it is neutral.
/// Bit-identical (chunk size never changes the logits). `widest_ffn == 0`
/// (unknown) keeps the 512 default. The OOM ladder still halves from here.
/// Memoised against the residency epoch. The answer is a four-rung ladder over the free
/// VRAM, so it cannot change between two requests that load nothing - but it was recomputed
/// per request, and each recomputation probed both cards with a stability loop AND trimmed
/// the memory pools. Measured at four device probes per `/api/generate`, on every model.
static CHUNK_MEMO: std::sync::Mutex<Option<(u64, usize, usize)>> = std::sync::Mutex::new(None);

pub(crate) fn adaptive_prefill_chunk(widest_ffn: usize) -> usize {
    let epoch = crate::inference::place::vram_manager::residency_epoch();
    if let Ok(memo) = CHUNK_MEMO.lock() {
        if let Some((e, ffn, chunk)) = *memo {
            if e == epoch && ffn == widest_ffn {
                return chunk;
            }
        }
    }
    let chunk = adaptive_prefill_chunk_uncached(widest_ffn);
    if let Ok(mut memo) = CHUNK_MEMO.lock() {
        *memo = Some((epoch, widest_ffn, chunk));
    }
    chunk
}

fn adaptive_prefill_chunk_uncached(widest_ffn: usize) -> usize {
    if widest_ffn == 0 {
        return PREFILL_CHUNK_TOKENS;
    }
    // One chunk's activation peak is roughly `chunk x widest_ffn x 4 bytes`, twice over for
    // the gate and up projections a SwiGLU holds at once.
    let per_token = widest_ffn * 4 * 2;

    // Derived from what the card actually has free, not from a constant. The budget here used
    // to be `1024 * 14336 * 4 * 2` - a figure calibrated so one particular FFN width got 1024
    // tokens - which meant a near-full card was handed the same chunk as an empty one and
    // discovered the difference by running out of memory.
    //
    // An eighth of the free VRAM, because the activation peak is not the only thing a forward
    // allocates: the attention scores and the KV growth transient come out of the same pool,
    // and a chunk sized to take all of it would leave nothing for either.
    let free = crate::inference::place::vram_manager::free_total() as usize;
    let budget = if free == 0 {
        // No CUDA, or nothing to probe: the host path is bounded by RAM, which this is not
        // measuring, so keep the conservative default rather than inventing a number.
        return PREFILL_CHUNK_TOKENS;
    } else {
        free / 8
    };

    let raw = budget / per_token;
    if raw >= 2048 {
        2048
    } else if raw >= 1024 {
        1024
    } else if raw >= PREFILL_CHUNK_TOKENS {
        PREFILL_CHUNK_TOKENS
    } else {
        // Below the default the ladder in `forward_prefill_chunked` takes over, but starting
        // it lower saves the passes that would only have failed.
        raw.max(PREFILL_CHUNK_TOKENS_MIN)
    }
}

/// One layer of exported KV rows per entry: `(k, v)` as f32, or `None` where the
/// layer keeps no cache.
pub(crate) type KvRows = Vec<Option<(Vec<f32>, Vec<f32>)>>;

/// What a backend can do beyond a plain forward. `Default` claims nothing, so a
/// backend reports only the capabilities it sets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Capabilities {
    /// `trim_kv` cuts the KV cache to an arbitrary length, not only back to
    /// position 0. Session-persistent KV reuse activates only when set.
    pub trim_kv: bool,
    /// `shift_kv_tail` drops a span of positions and re-phases the tail, so a
    /// full context can shift instead of resetting.
    pub shift_kv: bool,
    /// Prompt-lookup speculative decoding is offered: `forward_all` verifies a
    /// draft and `trim_kv` discards its rejected tail.
    pub pld: bool,
    /// The engine's CUDA-graph decode dispatch applies to this backend.
    pub graph_mode: bool,
    /// The backend drives a draft/verify split of its own.
    pub speculative: bool,
}

pub(crate) trait ModelBackend: Send {
    /// The capabilities this backend reports. The default claims none.
    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    /// Tokens the KV cache accepts, when the backend allocates one up front.
    fn kv_capacity(&self) -> Option<usize> {
        None
    }

    // --- Core forwards --------------------------------------------------
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor>;

    /// The shape a node measures its cost of computing one missed routed expert against:
    /// `(dim, inter, gate_up_dtype, down_dtype)`. `None` (default) for a model with no streamed
    /// experts to miss; a MoE whose experts stream overrides it so the cluster can price it.
    fn expert_cost_shape(
        &self,
    ) -> Option<(
        usize,
        usize,
        crate::tensor::quantized::GgmlDType,
        crate::tensor::quantized::GgmlDType,
    )> {
        None
    }

    /// The model-side inputs the cluster's cut planner needs to place this model across nodes:
    /// `(attn_bytes_per_layer, expert_bytes, working_set, activated)`. `None` (default) for a model
    /// that does not stream experts, or when a piece is not measurable; a streamed MoE overrides it.
    fn placement_model_inputs(&self) -> Option<(Vec<u64>, u64, u64, u64)> {
        None
    }

    /// Make this backend the head of a cross-node pipeline: run layers `[0, cut)` here and hand the
    /// rest to the tail at `addr`, which serves `[cut, n)`. The default declines, as a model that
    /// cannot express a layer range cannot be a pipeline head; a pipelineable model overrides it.
    fn connect_pipeline_tail(
        &mut self,
        _addr: std::net::SocketAddr,
        _cut: usize,
    ) -> crate::tensor::Result<()> {
        Err(crate::tensor::Error::msg(
            "this model cannot be a pipeline head",
        ))
    }

    /// Widest FFN intermediate dim across the model's layers (the activation-peak
    /// driver used to size the prefill chunk). `0` (default) keeps the 512 chunk.
    fn widest_ffn(&self) -> usize {
        0
    }

    /// Chunked prefill: forward `x` (shape `[1, seq]`) in
    /// `PREFILL_CHUNK_TOKENS`-token slices at increasing `index_pos`,
    /// returning only the LAST chunk's logits - the next-token prediction,
    /// which is the only thing prefill needs. Each chunk appends its full
    /// K/V to the cache and is masked causally against all prior positions;
    /// the model already supports a multi-token forward at arbitrary
    /// `index_pos` (the same path speculative-decode verify uses), so this is
    /// numerically a faithful prefill, just split.
    ///
    /// The point is to BOUND the per-forward activation peak to
    /// `chunk x ffn` regardless of prompt length - the same role
    /// llama.cpp/ollama's `num_batch` plays - so GPU placement no longer has
    /// to reserve for a worst-case full-context prefill. A prompt of `seq <=
    /// chunk` (the common case) takes a single `forward`, byte-identical to
    /// the unchunked path; only long prompts are split, and only their final
    /// chunk's logits are kept.
    fn forward_prefill_chunked(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> crate::tensor::Result<Tensor> {
        let (_, seq_len) = x.dims2()?;
        let chunk_start = adaptive_prefill_chunk(self.widest_ffn());
        // A prompt that fits one chunk runs as one forward; a longer one starts from the last
        // chunk that fit this model, clamped into [MIN, default]. Each out-of-memory failure
        // halves the chunk down to the floor and replays the prefill from `index_pos`, which
        // shrinks the activation peak, the attention scores and the KV growth transient
        // without changing the final chunk's logits. At the floor the card is full of
        // something else, so room is asked for once before the failure surfaces.
        use std::sync::atomic::Ordering;
        let widest = self.widest_ffn();
        let first = if seq_len <= chunk_start {
            seq_len
        } else {
            match ADAPTIVE_PREFILL_CHUNK.load(Ordering::Relaxed) {
                0 => chunk_start,
                remembered => remembered.clamp(PREFILL_CHUNK_TOKENS_MIN, chunk_start),
            }
        };
        let mut replay = false;
        let (logits, chunk_tokens) = crate::inference::place::recover::shrink_on_oom(
            "prefill",
            first,
            |chunk| {
                (chunk > PREFILL_CHUNK_TOKENS_MIN)
                    .then(|| (chunk / 2).max(PREFILL_CHUNK_TOKENS_MIN))
            },
            |chunk| {
                // One chunk's activation peak: `chunk x widest_ffn` f32 values for each of the
                // gate and up projections.
                let want = (chunk * widest.max(1) * 4 * 2) as u64;
                crate::inference::place::vram_manager::ensure_gpu_headroom_blocking("llm", want)
            },
            |chunk| {
                // A failed pass leaves partial KV behind; the session prefix [0, index_pos)
                // stays.
                if replay {
                    self.reset_kv_from(index_pos);
                }
                replay = true;
                self.forward_prefill_with_chunk(x, index_pos, seq_len, chunk)
            },
        )?;
        // Remember a chunk that had to shrink, so later requests start at it.
        if chunk_tokens < chunk_start && chunk_tokens < seq_len {
            ADAPTIVE_PREFILL_CHUNK.store(chunk_tokens, Ordering::Relaxed);
        }
        Ok(logits)
    }

    /// One chunked-prefill pass over `x` (shape `[1, seq_len]`) using a fixed
    /// `chunk_tokens` compute batch. Factored out of `forward_prefill_chunked`
    /// so the OOM-adaptive wrapper can replay it with a smaller chunk.
    fn forward_prefill_with_chunk(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        seq_len: usize,
        chunk_tokens: usize,
    ) -> crate::tensor::Result<Tensor> {
        let chunk_tokens = chunk_tokens.max(1);
        let mut off = 0usize;
        let mut last: Option<Tensor> = None;
        while off < seq_len {
            let n = chunk_tokens.min(seq_len - off);
            let chunk = x.narrow(1, off, n)?;
            last = Some(self.forward(&chunk, index_pos + off)?);
            off += n;
        }
        last.ok_or_else(|| crate::tensor::Error::msg("empty prefill".to_string()))
    }

    /// Multi-position forward: returns logits for every input position
    /// (shape `[batch, seq_len, vocab]`). Used by prompt-lookup speculative
    /// decoding. Implemented by GenericHetero.
    fn forward_all(&mut self, _x: &Tensor, _index_pos: usize) -> crate::tensor::Result<Tensor> {
        crate::tensor::bail!("forward_all not supported for this variant")
    }

    /// Embedding: last-token final-hidden of a prompt `[1, seq]` -> `[hidden]` (caller
    /// L2-normalizes). Routes through the loaded generic model (like rerank), so any
    /// llama-arch GGUF works - no separate encoder.
    fn embed_last_hidden(&mut self, _x: &Tensor) -> crate::tensor::Result<Vec<f32>> {
        crate::tensor::bail!("embeddings require a dense llama-arch (GenericHetero) model")
    }

    // --- KV-cache management --------------------------------------------

    /// Trim every layer's KV cache to `new_len` valid positions. Used by
    /// speculative decoding to discard rejected draft K/V on partial
    /// acceptance. Variants without trim support no-op (session reuse no-ops).
    /// Adapters attached to this model right now, in the order they were applied.
    fn adapters(&self) -> Vec<String> {
        Vec::new()
    }

    /// Replace the attached adapter set without reloading the checkpoint. An empty list
    /// detaches everything.
    ///
    /// The default refuses rather than succeeding silently: a family that cannot carry an
    /// adapter and answers "done" would leave a caller believing its fine-tune was live.
    fn set_adapters(
        &mut self,
        _wanted: &[(String, f32)],
    ) -> crate::tensor::Result<crate::inference::load::lora::AdapterReport> {
        Err(crate::tensor::Error(
            "this model family cannot carry adapters".into(),
        ))
    }

    fn trim_kv(&mut self, _new_len: usize) {}

    /// Roll every per-layer KV cache back to `keep` valid positions, discarding
    /// any K/V appended beyond it. Used by the OOM-adaptive prefill to drop the
    /// partial KV a failed pass appended before replaying from `keep` with a
    /// smaller chunk - `keep` is the prefill's `index_pos`, so session-reuse
    /// prefixes ([0, session_start)) are preserved. For trim-capable variants
    /// this is `trim_kv(keep)`; for TP (which only ever prefills at pos 0 and
    /// resets there) and the recurrent-MoE variants (lfm2/qwen35/nemotron/gptoss,
    /// which self-reset KV + conv state on the next forward at `input_pos == 0`)
    /// it is a no-op when `keep == 0`.
    fn reset_kv_from(&mut self, _keep: usize) {}

    /// Context shift: drop positions `[from - discard, from)` and pull the tail down by
    /// `discard`, keys re-phased (`GenericHeteroTransformer::shift_kv_tail`). On `Err`
    /// the caches are in no defined state and the caller must `trim_kv(0)`.
    fn shift_kv_tail(&mut self, _from: usize, _discard: usize) -> crate::tensor::Result<()> {
        Err(crate::tensor::Error::msg(
            "context shift: unsupported backend".to_string(),
        ))
    }

    /// Snapshots of the resident KV (`GenericHeteroTransformer::snapshot_kv`): copy the
    /// resident sequence aside, find the snapshot sharing the most of a prompt, make one
    /// resident again. Defaults: none kept, none found.
    fn snapshot_kv(&mut self, _tokens: Vec<u32>, _kv_len: usize) -> crate::tensor::Result<()> {
        Ok(())
    }
    fn kv_snapshot_dtype(&self, _index: usize) -> Option<u8> {
        None
    }
    fn best_kv_snapshot(&self, _prompt: &[u32]) -> Option<(usize, usize)> {
        None
    }
    fn restore_kv_snapshot(&mut self, _index: usize) -> crate::tensor::Result<(Vec<u32>, usize)> {
        Err(crate::tensor::Error::msg(
            "kv snapshots: unsupported backend".to_string(),
        ))
    }

    /// The disk tier's view of the snapshots: geometry, a snapshot's tokens, its rows out
    /// as host f32, and a snapshot rebuilt from such rows.
    fn kv_layout(&self) -> Option<(usize, usize, usize)> {
        None
    }
    fn kv_snapshot_ref(&self, _index: usize) -> Option<(&[u32], usize)> {
        None
    }
    fn export_kv_rows(
        &self,
        _index: usize,
        _from: usize,
        _to: usize,
    ) -> crate::tensor::Result<KvRows> {
        Err(crate::tensor::Error::msg(
            "kv snapshots: unsupported backend".to_string(),
        ))
    }
    fn export_window_rows(&self, _index: usize, _covered: usize) -> crate::tensor::Result<KvRows> {
        Ok(Vec::new())
    }
    fn import_kv_snapshot(
        &mut self,
        _tokens: Vec<u32>,
        _kv_len: usize,
        _rows: Vec<Option<(Vec<f32>, Vec<f32>)>>,
        _dtype_code: u8,
    ) -> crate::tensor::Result<()> {
        Ok(())
    }

    /// Tokens the KV cache actually holds, when the backend can say.
    ///
    /// A caller that needs a POSITION must ask here rather than track one
    /// alongside: the decode loop's own counter is for sampling and lags the
    /// cache whenever a sampled token is carried forward before being pushed.
    /// The reference implementation derives its position from the token
    /// sequence for the same reason, with one call shared by the speculative
    /// and non-speculative paths.
    fn kv_len(&self) -> Option<usize> {
        None
    }

    /// Where the model's layers run, when the backend can say.
    fn placement(&self) -> Option<crate::inference::place::Placement> {
        None
    }

    // --- Exact-prompt state snapshot (recurrent-hybrid prefix reuse) -----
    //
    // `trim_kv` reuse assumes the state can return to an arbitrary token
    // position. Hybrids carrying a rolling recurrent state (short-conv / SSM)
    // cannot: that state is advanced per token and can never be rewound, nor
    // rebuilt from the KV - which is why `Capabilities::trim_kv` is correctly false
    // for them, and why they re-prefill in full today. An exact-prompt snapshot
    // is the one reuse shape their state does allow: capture everything after a
    // prefill, restore it verbatim when the very same prompt comes back.
    //
    // Defaults are no-ops, so variants that reuse via `trim_kv` are untouched.

    /// Capture per-layer state at position `prompt.len()` plus the prefill
    /// logits. Call AFTER prefilling `[0, prompt.len())`.
    fn snapshot_prefix(&mut self, _prompt: &[u32], _logits: &Tensor) -> crate::tensor::Result<()> {
        Ok(())
    }

    /// On an EXACT prompt match, restore every layer's state and return the
    /// prefill logits, so decode resumes at `prompt.len()` with no prefill.
    /// An inexact prompt MUST miss: a partial match cannot be replayed.
    fn try_restore_prefix(&mut self, _prompt: &[u32]) -> crate::tensor::Result<Option<Tensor>> {
        Ok(None)
    }

    // --- CUDA-graph decode path -----------------------------------------

    /// Split step 1: QKV + RoPE + KV write (outside graph).
    fn prepare_all_kv(
        &mut self,
        _x: &Tensor,
        _index_pos: usize,
    ) -> crate::tensor::Result<(Tensor, Vec<Tensor>)> {
        crate::tensor::bail!("prepare_all_kv only for GenericHetero / GenericHeteroVision")
    }

    /// Refresh graph-compatible state (rope buffers + padded mask) for `pos`.
    fn update_graph_state(&mut self, _pos: usize) -> crate::tensor::Result<()> {
        crate::tensor::bail!("update_graph_state only for GenericHetero / GenericHeteroVision")
    }

    /// Compute embedding and write into stable GPU hidden buffer (outside graph).
    fn embed_for_graph(&mut self, _x: &Tensor) -> crate::tensor::Result<()> {
        crate::tensor::bail!("embed_for_graph only for GenericHetero / GenericHeteroVision")
    }

    /// Get the model's actual CUDA stream (for graph capture on the right stream).
    #[cfg(feature = "cuda")]
    fn model_cuda_stream(&self) -> Option<std::sync::Arc<crate::tensor::cuda_ext::CudaStream>> {
        None
    }

    /// Run layers + output proj from stable hidden buffer (inside graph).
    fn forward_from_hidden(&mut self) -> crate::tensor::Result<Tensor> {
        crate::tensor::bail!("forward_from_hidden only for GenericHetero / GenericHeteroVision")
    }

    /// True when this model needs the split graph path (F-dtype-only KV
    /// layers present) to avoid scatter_set ILLEGAL_ADDRESS at replay.
    fn needs_split_graph_path(&self) -> bool {
        false
    }

    /// Split-path forward: prepare_all_kv (uncaptured) +
    /// compute_all_from_kv (captured). Engine uses this for both warmup
    /// and capture so the capture region contains only attention/FFN
    /// (no scatter_set with fresh K pointer).
    fn forward_from_hidden_split(
        &mut self,
        _input_ids: &Tensor,
        _pos: usize,
    ) -> crate::tensor::Result<Tensor> {
        crate::tensor::bail!(
            "forward_from_hidden_split only for GenericHetero / GenericHeteroVision"
        )
    }

    /// Captured-side half of the split path. Used inside begin_capture.
    fn compute_all_from_kv_captured(&mut self) -> crate::tensor::Result<Tensor> {
        crate::tensor::bail!(
            "compute_all_from_kv_captured only for GenericHetero / GenericHeteroVision"
        )
    }

    /// Arch-level auto-enable gate for graph capture.
    fn graph_capture_auto_on(&self) -> bool {
        false
    }

    /// Say whether decode will capture a CUDA graph, and why not when it will not.
    ///
    /// The default covers the backends that never capture; the generic transformer
    /// overrides it with the actual refusal cause. Running uncaptured is not free -
    /// ~717 kernel launches per token, the GPU busy 7.5% of the decode window - so the
    /// decision belongs in the log rather than being inferred from a rate.
    fn log_graph_capture_decision(&self) {
        tracing::info!("🎞️  CUDA graph capture: OFF - this backend does not implement it");
    }

    /// Whether the KV-cache state is safe to capture in a CUDA graph.
    fn kv_state_graph_safe(&self) -> bool {
        false
    }

    /// Drop per-layer graph buffers so the next forward takes the non-graph
    /// path (`forward_attn` keys on `graph_rope_cos.is_some()`).
    fn invalidate_graph_state(&mut self) {}

    /// Engine hook after a successful graph capture: a live CUDA graph now
    /// references the per-layer KV buffers. Freezes quantized-KV growth +
    /// the capture-time seq ceiling. Default no-op for non-graph backends.
    fn mark_graph_captured(&mut self) {}

    /// Engine hook after each graph-replayed decode token: sync host-side
    /// KV bookkeeping (`current_seq_len`) to the post-token length `_len`.
    /// Replays advance the quantized caches device-side only; without this
    /// the next request's trim/prefix-reuse operates on stale lengths.
    fn sync_kv_len_for_graph(&mut self, _len: usize) {}

    /// Drop captured-region transient tensors (lifetime-extension handles)
    /// WITHOUT dropping the stable graph buffers. Engine calls this before
    /// (re)allocating the capture arena those transients may live in.
    fn clear_graph_transients(&mut self) {}

    /// per-token re-capture for archs whose captured graphs hold
    /// position-dependent pointers (phi2). Auto-on for phi2 only.
    fn recapture_each_token(&self) -> bool {
        false
    }

    /// CUDA device ordinals in this model's device map (for PTX prewarm).
    #[cfg(feature = "cuda")]
    fn cuda_device_ordinals(&self) -> HashSet<usize> {
        HashSet::new()
    }

    /// Per-layer device map segments `(device_label, ordinal, layer_start,
    /// layer_end)` for the GUI topology view. Empty = caller falls back to a
    /// single-entry heuristic on the primary device.
    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        Vec::new()
    }

    /// For a placement that streams its weights from the host and keeps a resident set on the
    /// cards, the bytes resident on each card: `(ordinal, bytes)`. `None` for a model whose
    /// layers have fixed device homes, which `device_layer_distribution` already reports. This
    /// is what keeps a card-resident streamed model from being drawn as living on the CPU.
    fn card_residency(&self) -> Option<Vec<(usize, u64)>> {
        None
    }

    // --- Vision ---------------------------------------------------------

    /// Whether this model is a vision model (accepts images)
    fn is_vision_model(&self) -> bool {
        false
    }

    /// True iff this is a qwen35moe model that carries the vision ViT. qwen35
    /// vision uses a dedicated prefill path (`forward_with_image`), NOT the
    /// moondream `encode_image`/`forward_with_img` prepend path.
    fn is_qwen35_vision(&self) -> bool {
        false
    }

    /// qwen35 vision image-token sentinel id (the single prompt token that
    /// `forward_qwen35_image` expands in place to `n_merged` copies). The
    /// engine's prompt clamp must preserve it - the generic BOS+tail clamp
    /// would drop a front-positioned sentinel and fail the request.
    fn qwen35_image_token(&self) -> Option<u32> {
        None
    }

    /// Encode an image through the vision encoder (Moondream only).
    /// Input: preprocessed image tensor (1, 3, 378, 378)
    /// Output: image embeddings tensor
    fn encode_image(&self, _image: &Tensor) -> crate::tensor::Result<Tensor> {
        Err(crate::tensor::Error::msg(
            "encode_image only supported on vision models".to_string(),
        ))
    }

    /// Pixtral tower? (variable-resolution: encodes from the RAW image and
    /// splices as [prefix, embeds, [IMG_END]+suffix] instead of a BOS-prepend).
    fn is_pixtral_vision(&self) -> bool {
        false
    }

    /// Pixtral: raw RGB image -> decoder-ready embeds [1, n, 5120] (with
    /// [IMG_BREAK] rows). HF-exact preprocessing at native 1024 longest-edge.
    fn encode_raw_image(&self, _img: &image::RgbImage) -> crate::tensor::Result<Tensor> {
        Err(crate::tensor::Error::msg(
            "encode_raw_image is pixtral-only".to_string(),
        ))
    }

    /// Pixtral vision prefill: splice the image embeds INSIDE the prompt as
    /// `[<s>[INST], embeds, [IMG_END] rest-of-prompt]` (transformers/llama.cpp
    /// region layout - the image must sit right after [INST] and be closed by
    /// the [IMG_END] token). `prompt_ids` = the tokenized prompt; a leading
    /// BOS/[INST] pair is absorbed into the prefix (prepended if absent).
    /// Returns (prefill logits, TOTAL sequence length) - the caller decodes
    /// from that position.
    fn forward_pixtral_spliced(
        &mut self,
        _prompt_ids: &[u32],
        _embeds: &Tensor,
        _device: &Device,
    ) -> crate::tensor::Result<(Tensor, usize)> {
        Err(crate::tensor::Error::msg(
            "forward_pixtral_spliced is pixtral-only".to_string(),
        ))
    }

    /// Vision-aware prefill: forward with image embeddings (Moondream only).
    /// bos_token: BOS token tensor (1, 1)
    /// text_input: text token tensor (1, seq_len)
    /// image_embeds: encoded image embeddings from encode_image()
    fn forward_with_img(
        &mut self,
        _bos_token: &Tensor,
        _text_input: &Tensor,
        _image_embeds: &Tensor,
    ) -> crate::tensor::Result<Tensor> {
        Err(crate::tensor::Error::msg(
            "forward_with_img only supported on vision models".to_string(),
        ))
    }

    /// qwen35moe (Qwen3-VL) vision prefill. `prompt_tokens` carries ONE
    /// `image_token` sentinel (handler-injected via `<|image_pad|>`); expand it
    /// to `n_merged=(gh/2)*(gw/2)` copies, then run `forward_with_image` (ViT +
    /// splice + mRoPE-2D). Returns `(last_logits, next_decode_pos)` - the decode
    /// loop must use `next_decode_pos` (the continuing logical mRoPE position),
    /// not the sequence length.
    fn forward_qwen35_image(
        &mut self,
        _prompt_tokens: &[u32],
        _px: &Tensor,
        _gh: usize,
        _gw: usize,
        _device: &Device,
    ) -> crate::tensor::Result<(Tensor, usize)> {
        Err(crate::tensor::Error::msg(
            "forward_qwen35_image on non-qwen35 (or non-CUDA build)".to_string(),
        ))
    }

    // --- Misc capabilities ----------------------------------------------

    fn set_early_exit_threshold(&mut self, _val: Option<f32>) {}

    /// Speculative-decode auto-tune calibrator. None keeps the opencl speculative driver
    /// inert, which is what every variant wants today.
    #[cfg(feature = "opencl")]
    fn create_calibrator(
        &self,
    ) -> Option<crate::inference::serve::speculative_config::SpeculativeCalibrator> {
        None
    }

    /// Draft forward (legacy CUDA+OpenCL spec path - removed).
    #[cfg(feature = "opencl")]
    fn forward_draft(
        &mut self,
        _x: &Tensor,
        _index_pos: usize,
    ) -> crate::tensor::Result<(Tensor, Tensor)> {
        Err(crate::tensor::Error::msg(
            "speculative draft path removed".to_string(),
        ))
    }

    /// Verify forward (legacy CUDA+OpenCL spec path - removed).
    #[cfg(feature = "opencl")]
    fn forward_verify_batch(
        &mut self,
        _states: &[(Tensor, usize)],
        _base_pos: usize,
    ) -> crate::tensor::Result<Vec<Tensor>> {
        Err(crate::tensor::Error::msg(
            "speculative verify path removed".to_string(),
        ))
    }

    /// Trim CUDA KV caches only (legacy spec rollback - no-op now).
    #[cfg(feature = "opencl")]
    fn trim_cuda_kv(&mut self, _seq_len: usize) {}

    /// Trim OpenCL KV caches only (legacy spec rollback - no-op now).
    #[cfg(feature = "opencl")]
    fn trim_opencl_kv(&mut self, _seq_len: usize) {}

    // --- Engine-side downcast hooks -------------------------------------

    /// Moondream? (drives the engine's moondream CUDA-graph decode path).
    fn is_moondream(&self) -> bool {
        false
    }

    /// Concrete moondream model access for the engine's graph capture path.
    fn moondream_mut(&mut self) -> Option<&mut moondream::Model> {
        None
    }

    /// The continuous-batch worker, when this model is CB-served. The serial
    /// decode methods are NEVER called on a CB-served model - requests are
    /// delegated to the `ContinuousServer`.
    /// The batched-decode server this model is served through, when it is. The generic
    /// continuous-batch server and the DeepSeek batch worker both provide one; the serial decode
    /// methods are never called on a batch-served model - requests go to the server.
    fn batched_server(
        &self,
    ) -> Option<std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>> {
        None
    }

    /// True when this is a `cb_eligible` GPU GenericHetero model that
    /// `cb_maybe_wrap` may move into a ContinuousServer worker.
    fn cb_eligible_gpu(&self) -> bool {
        false
    }

    /// Extract the inner `GenericHeteroTransformer` (CB-wrap stepping stone);
    /// every other backend returns itself unchanged.
    fn take_generic(self: Box<Self>) -> Result<GenericHeteroTransformer, BoxedModelBackend>;

    /// Move a batch-eligible model into its batch worker (the CB-wrap stepping stone); a backend
    /// with nothing to batch, or that does not batch, returns itself unchanged.
    fn into_batched(
        self: Box<Self>,
        eos: u32,
        max_batch: usize,
    ) -> Result<
        std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>,
        BoxedModelBackend,
    >;

    /// Take a batch-served model back to serial serving; a backend that is not batch-served,
    /// or cannot hand its model back, returns itself unchanged.
    fn into_serial(self: Box<Self>) -> Result<BoxedModelBackend, BoxedModelBackend>;
}

/// `take_generic` passthrough for every backend that is NOT GenericHetero.
macro_rules! take_generic_passthrough {
    () => {
        fn take_generic(self: Box<Self>) -> Result<GenericHeteroTransformer, BoxedModelBackend> {
            Err(self)
        }
        fn into_batched(
            self: Box<Self>,
            _eos: u32,
            _max_batch: usize,
        ) -> Result<
            std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>,
            BoxedModelBackend,
        > {
            Err(self)
        }
        fn into_serial(self: Box<Self>) -> Result<BoxedModelBackend, BoxedModelBackend> {
            Err(self)
        }
    };
}

// ------------------------------------------------------------
// Backend wrapper structs (one per former ModelVariant arm)
// ------------------------------------------------------------

/// Moondream legacy path (original PyTorch state-dict naming GGUF).
pub(crate) struct MoondreamBackend(pub moondream::Model);

/// Generic hetero transformer covering Qwen2/3, Gemma3, Phi3, GLM4, StableLM, etc.
pub(crate) struct GenericBackend(pub GenericHeteroTransformer);

/// Vision-capable wrapper around GenericHetero: phi2 LM + separate
/// Ollama-style CLIP vision tower (moondream), or llama LM + Pixtral-ViT.
/// Lets vision models shipped as dual-blob Ollama packages use the generic
/// LM hot path (Q8 KV unblock, fused phi2 kernels, shared-Q8_1 QKV) instead
/// of the kernel-poor legacy QuantizedMoondream branch.
pub(crate) struct GenericVisionBackend {
    pub text: GenericHeteroTransformer,
    pub vision: VisionTower,
}

/// gpt-oss (OPENAI_MOE): MXFP4 MoE transformer with attention sinks +
/// sliding-window + NEOX RoPE. Single-device for now. (CUDA-only: MoE GEMM.)
pub(crate) struct GptOssBackend(pub crate::inference::model::gptoss::GptOssModel);

/// deepseek_v41: latent MQA, CSA2 bands, hyper-connections, and a MoE whose routed experts stream
/// from the mmap through a hot cache. CPU, one sequence at a time. The engine's `forward` is
/// decode-driven: every token of `x` is fed through the per-layer caches at its position, and the
/// last token's next-token logits come back. Position 0 starts a fresh sequence; any other
/// position must be exactly where the cache stands, since the caches only append.
pub(crate) struct DeepseekV41Backend {
    model: crate::inference::model::deepseek_v41::model::DeepseekV41Model,
    state: crate::inference::model::deepseek_v41::cache::DecodeState,
    widest_ffn: usize,
    /// The engine's streamed placement, opened once: the cards keep what they are given across
    /// prompts and tokens, so a placement opened per prompt would give that away at every
    /// prompt. `None` on the host alone.
    streamed: Option<crate::inference::offload::streamed::Streamed>,
    /// The state as it stood before the last `forward_all`, the tokens it verified, and the
    /// position it ran at. A partial acceptance rolls back to this mark and replays the accepted
    /// prefix, since the sliding-window cache cannot truncate by length.
    verify: Option<(
        crate::inference::model::deepseek_v41::cache::DecodeState,
        Vec<u32>,
        usize,
    )>,
    /// The native DSpark draft head, loaded from the released checkpoint when it is on the machine.
    /// `None` leaves speculative decode to the n-gram fallback.
    dspark: Option<crate::inference::model::deepseek_v41::dspark::Dspark>,
    /// The draft's running window state, beside `state`.
    dspark_state: Option<crate::inference::model::deepseek_v41::dspark::DsparkState>,
    /// When set, this node is the HEAD of a cross-node pipeline: it runs layers `[0, cut)` and
    /// hands the boundary to a tail over the link, which runs `[cut, n)` and returns the logits.
    /// Formed when the cluster's auto-tune decides a split pays; `None` keeps the whole-stack
    /// single-node path exactly as it was.
    pipeline: Option<PipelineHead>,
}

/// The head end of a cross-node pipeline held by a backend: the link to the tail, the boundary
/// layer, and a per-sequence id so a new prompt starts the tail's decode state clean.
struct PipelineHead {
    link: crate::distributed::wire_link::WireLink,
    cut: usize,
    seq: u32,
}

impl DeepseekV41Backend {
    /// The backend with the draft the hub checkpoint holds, when it is on the machine.
    pub(crate) fn new(
        model: crate::inference::model::deepseek_v41::model::DeepseekV41Model,
        widest_ffn: usize,
        context_length: usize,
    ) -> Self {
        let dspark = crate::inference::model::deepseek_v41::dspark::Dspark::open();
        Self::with_draft(model, widest_ffn, context_length, dspark)
    }

    /// The backend with `dspark` as its draft, `None` for none.
    pub(crate) fn with_draft(
        model: crate::inference::model::deepseek_v41::model::DeepseekV41Model,
        widest_ffn: usize,
        context_length: usize,
        dspark: Option<crate::inference::model::deepseek_v41::dspark::Dspark>,
    ) -> Self {
        use crate::inference::offload::streamed::Streamed;
        // Decode needs only a small transient (windowed attention bounds KV growth); the
        // full-context reservation would leave no card room for resident experts. A longer
        // prompt is chunked to this width at prefill.
        const PREFILL_TRANSIENT_TOKENS: usize = 64;
        let state = model.new_decode_state();
        // The engine opens the card placement through the generic streamed-model seam: the model
        // states its resident footprint, routing and experts, and no architecture is named here.
        let streamed = Streamed::open_for(&model, context_length.min(PREFILL_TRANSIENT_TOKENS));
        let mut state = state;
        let dspark_state = match &dspark {
            Some(d) => {
                state.capture_layers = d.cfg.target_layers.clone();
                Some(
                    crate::inference::model::deepseek_v41::dspark::DsparkState::new(
                        d.stages.len(),
                        d.cfg.window_size,
                        d.cfg.head_dim,
                    ),
                )
            }
            None => None,
        };
        Self {
            model,
            state,
            widest_ffn,
            streamed,
            verify: None,
            dspark,
            dspark_state,
            pipeline: None,
        }
    }

    /// Serve one forward over the pipeline: run `ids` through layers `[0, cut)` here token by token,
    /// shipping each boundary to the tail and reading its logits, and return the last position's
    /// logits `[1, vocab]`. A prompt is prefilled by feeding its tokens in order - each advances the
    /// caches on both nodes - so no separate prefill path crosses the wire. A new sequence
    /// (`index_pos == 0`) takes a fresh sequence id so the tail resets its decode state for it.
    fn pipeline_forward(&mut self, ids: &[u32], index_pos: usize) -> crate::tensor::Result<Tensor> {
        if index_pos == 0 {
            self.state = self.model.new_decode_state();
            let p = self.pipeline.as_mut().expect("pipeline set");
            p.seq = p.seq.wrapping_add(1);
        } else if self.state.pos != index_pos {
            return Err(crate::tensor::Error::msg(format!(
                "deepseek_v41 pipeline: forward at {index_pos} but the cache stands at {}",
                self.state.pos
            )));
        }
        let Self {
            model,
            state,
            streamed,
            pipeline,
            ..
        } = self;
        let p = pipeline.as_mut().expect("pipeline set");
        let (cut, seq) = (p.cut, p.seq);
        let link = &mut p.link;
        let logits = match streamed {
            Some(s) => {
                model.offload_experts(Some(s.lanes.clone()));
                crate::inference::offload::with_offload(s.offload.clone(), || {
                    let mut last = Vec::new();
                    for &t in ids {
                        last = model.pipeline_head_step_wire(t, cut, seq, state, link)?;
                    }
                    crate::tensor::Result::Ok(last)
                })?
            }
            None => {
                let mut last = Vec::new();
                for &t in ids {
                    last = model.pipeline_head_step_wire(t, cut, seq, state, link)?;
                }
                last
            }
        };
        let vocab = logits.len();
        crate::tensor::Tensor::from_vec(logits, (1, vocab), &crate::tensor::Device::Cpu)
    }

    /// One batched verify over `ids` from the current position: each layer's experts are read
    /// once for the whole block, on the cards when there are any. Advances `self.state` by the
    /// inputs and returns each position's logits `[seq, vocab]`.
    fn verify_batch(&mut self, ids: &[u32]) -> crate::tensor::Result<Tensor> {
        let Self {
            model,
            state,
            streamed,
            ..
        } = self;
        match streamed {
            Some(s) => {
                model.offload_experts(Some(s.lanes.clone()));
                crate::inference::offload::with_offload(s.offload.clone(), || {
                    model.forward_verify_batch(ids, state)
                })
            }
            None => model.forward_verify_batch(ids, state),
        }
    }

    /// The whole prompt in one prefill, its heavy steps on the cards when there are any.
    fn prefill(&mut self, ids: &[u32]) -> crate::tensor::Result<Tensor> {
        let Self {
            model,
            state,
            streamed,
            ..
        } = self;
        match streamed {
            Some(s) => {
                // The routed experts of a layer spread over the lanes, for a prompt's batch and
                // for the tokens that follow alike: the lanes keep what they are given, and a
                // decode reads the same experts the prompt just routed to.
                model.offload_experts(Some(s.lanes.clone()));
                crate::inference::offload::with_offload(s.offload.clone(), || {
                    model.prefill_into(ids, state)
                })
            }
            None => model.prefill_into(ids, state),
        }
    }
}

/// nemotron_h_moe: hybrid Mamba2 + attention + non-gated MoE, multi-device.
pub(crate) struct NemotronHBackend(pub crate::inference::model::nemotron_h::NemotronHModel);

/// lfm2moe: hybrid short-conv + attention + SwiGLU MoE, multi-device (+ CPU).
pub(crate) struct Lfm2MoeBackend(pub crate::inference::model::lfm2_moe::Lfm2MoeModel);

/// qwen35moe: hybrid gated-DeltaNet + attention + softmax MoE, multi-device.
/// Also runs on CPU (text): the DeltaNet recurrence uses the pure tensor-op
/// `forward_pertoken` path and the MoE GEMM the fallback CPU path. Vision is
/// CUDA-only.
pub(crate) struct Qwen35MoeBackend(pub crate::inference::model::qwen35::moe::Qwen35MoeModel);

/// TpQwen2: tensor-parallel (TP=2) decoder across 2 GPUs. Both cards run EVERY layer -
/// column/row-parallel weights joined by an all-reduce - against the pipeline's one card
/// per token with the other idle. Covers qwen2 (deepseek-r1, with QKV bias) and the
/// no-bias llama/mistral family. CUDA-only.
#[cfg(feature = "cuda")]
pub(crate) struct TpQwen2Backend(pub crate::inference::serve::tp_model::TpQwen2);

/// Continuous-batch serving: a `cb_eligible` dense model (Qwen2/3) moved into
/// a background `ContinuousServer` worker that multiplexes concurrent requests
/// into batched paged decode (the throughput-vs-vLLM path). The serial decode
/// methods are NEVER called on this variant - requests are delegated to the
/// worker via `ContinuousServer::submit_sampled`.
pub(crate) struct ContinuousBackend(
    pub std::sync::Arc<crate::inference::serve::continuous_serve::ContinuousServer>,
);

/// DeepSeek V4.1 served through its batch worker: the model moved into a background
/// A batched-decode server that batches concurrent requests' decode steps. The serial decode
/// methods are never called on this variant - requests are delegated to the server.
pub(crate) struct BatchedBackend(
    pub std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>,
);

/// Transient sentinel used ONLY while swapping a loaded GenericHetero model
/// into a Continuous worker (`std::mem::replace` placeholder) - never stored
/// or used; any method call on it is a bug.
pub(crate) struct TakenBackend;

// Safety: Mutex<> provides exclusive access, preventing concurrent RefCell
// mutation (contract inherited from the old `unsafe impl Send for ModelVariant`).
unsafe impl Send for MoondreamBackend {}
unsafe impl Send for GenericBackend {}
unsafe impl Send for GenericVisionBackend {}
unsafe impl Send for GptOssBackend {}
unsafe impl Send for NemotronHBackend {}
unsafe impl Send for Lfm2MoeBackend {}
unsafe impl Send for Qwen35MoeBackend {}
#[cfg(feature = "cuda")]
unsafe impl Send for TpQwen2Backend {}
unsafe impl Send for ContinuousBackend {}
unsafe impl Send for BatchedBackend {}

// ------------------------------------------------------------
// Trait impls
// ------------------------------------------------------------

impl ModelBackend for MoondreamBackend {
    fn forward(&mut self, x: &Tensor, _index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.text_model().forward(x)
    }

    fn is_vision_model(&self) -> bool {
        true
    }

    fn encode_image(&self, image: &Tensor) -> crate::tensor::Result<Tensor> {
        #[allow(unused_imports)]
        use crate::tensor::Module;
        image.apply(self.0.vision_encoder())
    }

    fn forward_with_img(
        &mut self,
        bos_token: &Tensor,
        text_input: &Tensor,
        image_embeds: &Tensor,
    ) -> crate::tensor::Result<Tensor> {
        self.0
            .text_model()
            .forward_with_img(bos_token, text_input, image_embeds)
    }

    fn is_moondream(&self) -> bool {
        true
    }

    fn moondream_mut(&mut self) -> Option<&mut moondream::Model> {
        Some(&mut self.0)
    }

    take_generic_passthrough!();
}

impl ModelBackend for GenericBackend {
    fn kv_capacity(&self) -> Option<usize> {
        Some(self.0.kv_capacity())
    }

    fn adapters(&self) -> Vec<String> {
        self.0.adapters().to_vec()
    }

    fn set_adapters(
        &mut self,
        wanted: &[(String, f32)],
    ) -> crate::tensor::Result<crate::inference::load::lora::AdapterReport> {
        self.0.set_adapters(wanted)
    }

    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward(x, index_pos)
    }

    fn widest_ffn(&self) -> usize {
        self.0.widest_ffn()
    }

    fn forward_all(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward_all(x, index_pos)
    }

    fn embed_last_hidden(&mut self, x: &Tensor) -> crate::tensor::Result<Vec<f32>> {
        self.0.embed_last_hidden(x)
    }

    fn trim_kv(&mut self, new_len: usize) {
        self.0.trim_kv(new_len)
    }

    fn reset_kv_from(&mut self, keep: usize) {
        self.0.trim_kv(keep)
    }

    fn kv_len(&self) -> Option<usize> {
        self.0.kv_len()
    }

    fn placement(&self) -> Option<crate::inference::place::Placement> {
        Some(self.0.placement())
    }

    fn shift_kv_tail(&mut self, from: usize, discard: usize) -> crate::tensor::Result<()> {
        self.0.shift_kv_tail(from, discard)
    }

    fn snapshot_kv(&mut self, tokens: Vec<u32>, kv_len: usize) -> crate::tensor::Result<()> {
        self.0.snapshot_kv(tokens, kv_len)
    }
    fn best_kv_snapshot(&self, prompt: &[u32]) -> Option<(usize, usize)> {
        self.0.best_kv_snapshot(prompt)
    }
    fn restore_kv_snapshot(&mut self, index: usize) -> crate::tensor::Result<(Vec<u32>, usize)> {
        self.0.restore_kv_snapshot(index)
    }

    fn kv_layout(&self) -> Option<(usize, usize, usize)> {
        Some(self.0.kv_layout())
    }
    fn kv_snapshot_ref(&self, index: usize) -> Option<(&[u32], usize)> {
        self.0.kv_snapshot_ref(index)
    }
    fn export_kv_rows(
        &self,
        index: usize,
        from: usize,
        to: usize,
    ) -> crate::tensor::Result<KvRows> {
        self.0.export_kv_rows(index, from, to)
    }
    fn export_window_rows(&self, index: usize, covered: usize) -> crate::tensor::Result<KvRows> {
        self.0.export_window_rows(index, covered)
    }
    fn import_kv_snapshot(
        &mut self,
        tokens: Vec<u32>,
        kv_len: usize,
        rows: Vec<Option<(Vec<f32>, Vec<f32>)>>,
        dtype_code: u8,
    ) -> crate::tensor::Result<()> {
        self.0.import_kv_snapshot(tokens, kv_len, rows, dtype_code)
    }
    fn kv_snapshot_dtype(&self, index: usize) -> Option<u8> {
        self.0.kv_snapshot_dtype(index)
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities {
            // Routed mixtures were excluded because the top-k choice follows the GEMM
            // shape, and a warm re-prefill used shapes a cold run never had - measured
            // divergent on GPU and host alike. Since reuse snaps to the cold run's chunk
            // grid the warm tail runs the exact cold shapes, expert choice included, so
            // the exclusion is lifted; the cold/warm probe on a routed model is the gate
            // that must hold.
            trim_kv: true,
            shift_kv: true,
            pld: true,
            graph_mode: true,
            ..Capabilities::default()
        }
    }

    fn prepare_all_kv(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> crate::tensor::Result<(Tensor, Vec<Tensor>)> {
        self.0.prepare_all_kv(x, index_pos)
    }

    fn update_graph_state(&mut self, pos: usize) -> crate::tensor::Result<()> {
        self.0.update_graph_state(pos)
    }

    fn embed_for_graph(&mut self, x: &Tensor) -> crate::tensor::Result<()> {
        self.0.embed_for_graph(x)
    }

    #[cfg(feature = "cuda")]
    fn model_cuda_stream(&self) -> Option<std::sync::Arc<crate::tensor::cuda_ext::CudaStream>> {
        self.0.cuda_stream()
    }

    fn forward_from_hidden(&mut self) -> crate::tensor::Result<Tensor> {
        self.0.forward_from_hidden()
    }

    fn needs_split_graph_path(&self) -> bool {
        self.0.needs_split_graph_path()
    }

    fn forward_from_hidden_split(
        &mut self,
        input_ids: &Tensor,
        pos: usize,
    ) -> crate::tensor::Result<Tensor> {
        self.0.forward_from_hidden_split(input_ids, pos)
    }

    fn compute_all_from_kv_captured(&mut self) -> crate::tensor::Result<Tensor> {
        self.0.compute_all_from_kv_captured()
    }

    fn graph_capture_auto_on(&self) -> bool {
        self.0.graph_capture_auto_on()
    }

    fn log_graph_capture_decision(&self) {
        self.0.log_graph_capture_decision()
    }

    fn kv_state_graph_safe(&self) -> bool {
        self.0.kv_state_graph_safe()
    }

    fn invalidate_graph_state(&mut self) {
        self.0.invalidate_graph_state()
    }

    #[cfg(feature = "cuda")]
    fn mark_graph_captured(&mut self) {
        self.0.mark_graph_captured()
    }

    #[cfg(feature = "cuda")]
    fn sync_kv_len_for_graph(&mut self, len: usize) {
        self.0.sync_kv_len_for_graph(len)
    }

    fn clear_graph_transients(&mut self) {
        self.0.clear_graph_transients()
    }

    fn recapture_each_token(&self) -> bool {
        self.0.recapture_each_token()
    }

    #[cfg(feature = "cuda")]
    fn cuda_device_ordinals(&self) -> HashSet<usize> {
        self.0.cuda_device_ordinals()
    }

    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        self.0.device_layer_distribution()
    }

    fn cb_eligible_gpu(&self) -> bool {
        self.0.cb_eligible() && !self.0.compute_device().is_cpu()
    }

    fn take_generic(self: Box<Self>) -> Result<GenericHeteroTransformer, BoxedModelBackend> {
        Ok(self.0)
    }
    fn into_serial(self: Box<Self>) -> Result<BoxedModelBackend, BoxedModelBackend> {
        Err(self)
    }
    fn into_batched(
        self: Box<Self>,
        _eos: u32,
        _max_batch: usize,
    ) -> Result<
        std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>,
        BoxedModelBackend,
    > {
        Err(self)
    }
}

impl ModelBackend for GenericVisionBackend {
    fn kv_capacity(&self) -> Option<usize> {
        Some(self.text.kv_capacity())
    }

    fn placement(&self) -> Option<crate::inference::place::Placement> {
        Some(self.text.placement())
    }

    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.text.forward(x, index_pos)
    }

    fn forward_all(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.text.forward_all(x, index_pos)
    }

    fn embed_last_hidden(&mut self, x: &Tensor) -> crate::tensor::Result<Vec<f32>> {
        self.text.embed_last_hidden(x)
    }

    fn trim_kv(&mut self, new_len: usize) {
        self.text.trim_kv(new_len)
    }

    fn reset_kv_from(&mut self, keep: usize) {
        self.text.trim_kv(keep)
    }

    fn shift_kv_tail(&mut self, from: usize, discard: usize) -> crate::tensor::Result<()> {
        self.text.shift_kv_tail(from, discard)
    }

    fn capabilities(&self) -> Capabilities {
        // Vision-conditioned generation produces image captions
        // (moondream's primary use): low n-gram repetition, so PLD
        // acceptance is 0-10 % and the multi-position verify cost
        // dominates the rare-accept gain. Measured:
        // PLD on vision adds ~5-10 % decode overhead net.
        // Re-enable per-arch when a vision use case with bursty
        // repeating tokens emerges (e.g. structured-output VLM).
        Capabilities {
            trim_kv: true,
            shift_kv: true,
            pld: false,
            graph_mode: true,
            ..Capabilities::default()
        }
    }

    fn prepare_all_kv(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> crate::tensor::Result<(Tensor, Vec<Tensor>)> {
        self.text.prepare_all_kv(x, index_pos)
    }

    fn update_graph_state(&mut self, pos: usize) -> crate::tensor::Result<()> {
        self.text.update_graph_state(pos)
    }

    fn embed_for_graph(&mut self, x: &Tensor) -> crate::tensor::Result<()> {
        self.text.embed_for_graph(x)
    }

    #[cfg(feature = "cuda")]
    fn model_cuda_stream(&self) -> Option<std::sync::Arc<crate::tensor::cuda_ext::CudaStream>> {
        self.text.cuda_stream()
    }

    fn forward_from_hidden(&mut self) -> crate::tensor::Result<Tensor> {
        self.text.forward_from_hidden()
    }

    fn needs_split_graph_path(&self) -> bool {
        self.text.needs_split_graph_path()
    }

    fn forward_from_hidden_split(
        &mut self,
        input_ids: &Tensor,
        pos: usize,
    ) -> crate::tensor::Result<Tensor> {
        self.text.forward_from_hidden_split(input_ids, pos)
    }

    fn compute_all_from_kv_captured(&mut self) -> crate::tensor::Result<Tensor> {
        self.text.compute_all_from_kv_captured()
    }

    fn graph_capture_auto_on(&self) -> bool {
        self.text.graph_capture_auto_on()
    }

    fn log_graph_capture_decision(&self) {
        self.text.log_graph_capture_decision()
    }

    fn kv_state_graph_safe(&self) -> bool {
        self.text.kv_state_graph_safe()
    }

    fn invalidate_graph_state(&mut self) {
        self.text.invalidate_graph_state()
    }

    #[cfg(feature = "cuda")]
    fn mark_graph_captured(&mut self) {
        self.text.mark_graph_captured()
    }

    #[cfg(feature = "cuda")]
    fn sync_kv_len_for_graph(&mut self, len: usize) {
        self.text.sync_kv_len_for_graph(len)
    }

    fn clear_graph_transients(&mut self) {
        self.text.clear_graph_transients()
    }

    fn recapture_each_token(&self) -> bool {
        self.text.recapture_each_token()
    }

    #[cfg(feature = "cuda")]
    fn cuda_device_ordinals(&self) -> HashSet<usize> {
        self.text.cuda_device_ordinals()
    }

    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        self.text.device_layer_distribution()
    }

    fn is_vision_model(&self) -> bool {
        true
    }

    fn encode_image(&self, image: &Tensor) -> crate::tensor::Result<Tensor> {
        match &self.vision {
            VisionTower::Clip(v) => v.forward(image),
            VisionTower::Pixtral(_) => Err(crate::tensor::Error::msg(
                "pixtral encodes from the raw image (encode_raw_image), not a fixed tensor"
                    .to_string(),
            )),
        }
    }

    fn is_pixtral_vision(&self) -> bool {
        matches!(self.vision, VisionTower::Pixtral(_))
    }

    fn encode_raw_image(&self, img: &image::RgbImage) -> crate::tensor::Result<Tensor> {
        match &self.vision {
            VisionTower::Pixtral(v) => v
                .encode_image_with_breaks(img)
                .map_err(|e| crate::tensor::Error::msg(format!("pixtral encode: {e}"))),
            _ => Err(crate::tensor::Error::msg(
                "encode_raw_image is pixtral-only".to_string(),
            )),
        }
    }

    fn forward_pixtral_spliced(
        &mut self,
        prompt_ids: &[u32],
        embeds: &Tensor,
        device: &Device,
    ) -> crate::tensor::Result<(Tensor, usize)> {
        // Mistral-tekken fixed control ids (pixtral-12b): 1=<s> 3=[INST] 4=[/INST] 13=[IMG_END].
        const BOS: u32 = 1;
        const INST: u32 = 3;
        const INST_END: u32 = 4;
        const IMG_END: u32 = 13;
        if !self.is_pixtral_vision() {
            return Err(crate::tensor::Error::msg(
                "forward_pixtral_spliced is pixtral-only".to_string(),
            ));
        }
        let mut rest = prompt_ids;
        let mut prefix: Vec<u32> = vec![BOS];
        if rest.first() == Some(&BOS) {
            rest = &rest[1..];
        }
        if rest.first() == Some(&INST) {
            rest = &rest[1..];
        }
        prefix.push(INST);
        let mut suffix: Vec<u32> = Vec::with_capacity(rest.len() + 2);
        suffix.push(IMG_END);
        suffix.extend_from_slice(rest);
        // Raw (untemplated) prompts don't close the turn - without [/INST] the
        // model continues the user text instead of answering. Templated chat
        // prompts already end with it; only append when missing.
        if suffix.last() != Some(&INST_END) {
            suffix.push(INST_END);
        }
        let total = prefix.len() + embeds.dim(1)? + suffix.len();
        let prefix_t = Tensor::from_vec(prefix, &[1usize, 2][..], device)?;
        let suffix_len = suffix.len();
        let suffix_t = Tensor::from_vec(suffix, &[1usize, suffix_len][..], device)?;
        let logits = self
            .text
            .forward_with_audio_embeds(&prefix_t, embeds, &suffix_t)?;
        Ok((logits, total))
    }

    fn forward_with_img(
        &mut self,
        bos_token: &Tensor,
        text_input: &Tensor,
        image_embeds: &Tensor,
    ) -> crate::tensor::Result<Tensor> {
        self.text
            .forward_with_image_embeds(bos_token, text_input, image_embeds)
    }

    take_generic_passthrough!();
}

impl ModelBackend for GptOssBackend {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward(x, index_pos)
    }

    fn snapshot_prefix(&mut self, prompt: &[u32], logits: &Tensor) -> crate::tensor::Result<()> {
        self.0.snapshot_prefix(prompt, logits)
    }

    fn try_restore_prefix(&mut self, prompt: &[u32]) -> crate::tensor::Result<Option<Tensor>> {
        self.0.try_restore_prefix(prompt)
    }

    take_generic_passthrough!();
}

impl ModelBackend for NemotronHBackend {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward(x, index_pos)
    }

    fn snapshot_prefix(&mut self, prompt: &[u32], logits: &Tensor) -> crate::tensor::Result<()> {
        self.0.snapshot_prefix(prompt, logits)
    }

    fn try_restore_prefix(&mut self, prompt: &[u32]) -> crate::tensor::Result<Option<Tensor>> {
        self.0.try_restore_prefix(prompt)
    }

    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        group_layers_by_device(self.0.layer_device_locations())
    }

    take_generic_passthrough!();
}

impl ModelBackend for Lfm2MoeBackend {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward(x, index_pos)
    }

    fn widest_ffn(&self) -> usize {
        self.0.widest_ffn()
    }

    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        group_layers_by_device(self.0.layer_device_locations())
    }

    fn snapshot_prefix(&mut self, prompt: &[u32], logits: &Tensor) -> crate::tensor::Result<()> {
        self.0.snapshot_prefix(prompt, logits)
    }

    fn try_restore_prefix(&mut self, prompt: &[u32]) -> crate::tensor::Result<Option<Tensor>> {
        self.0.try_restore_prefix(prompt)
    }

    take_generic_passthrough!();
}

impl ModelBackend for DeepseekV41Backend {
    fn expert_cost_shape(
        &self,
    ) -> Option<(
        usize,
        usize,
        crate::tensor::quantized::GgmlDType,
        crate::tensor::quantized::GgmlDType,
    )> {
        self.model.expert_cost_shape()
    }

    fn placement_model_inputs(&self) -> Option<(Vec<u64>, u64, u64, u64)> {
        self.model.placement_model_inputs()
    }

    fn connect_pipeline_tail(
        &mut self,
        addr: std::net::SocketAddr,
        cut: usize,
    ) -> crate::tensor::Result<()> {
        let n = self.model.n_layers();
        if cut == 0 || cut >= n {
            return Err(crate::tensor::Error::msg(format!(
                "pipeline head: cut {cut} must be in 1..{n}"
            )));
        }
        let link = crate::distributed::wire_link::WireLink::connect(addr).map_err(|e| {
            crate::tensor::Error::msg(format!("pipeline head: connect {addr}: {e}"))
        })?;
        self.pipeline = Some(PipelineHead { link, cut, seq: 0 });
        Ok(())
    }

    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        let ids = x.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
        // A cross-node pipeline head runs the whole forward through the tail: the local path below
        // never sees a token, so it stays exactly the single-node path when no split is active.
        if self.pipeline.is_some() {
            return self.pipeline_forward(&ids, index_pos);
        }
        // A plain decode moves past any verify mark: a full acceptance never trims, so the mark
        // is stale here and would otherwise hold a whole state clone across the decode.
        self.verify = None;
        if index_pos == 0 {
            self.state = self.model.new_decode_state();
            if let Some(d) = &self.dspark {
                self.state.capture_layers = d.cfg.target_layers.clone();
            }
        }
        if self.state.pos != index_pos {
            return Err(crate::tensor::Error::msg(format!(
                "deepseek_v41: forward at position {index_pos} but the cache stands at {}",
                self.state.pos
            )));
        }
        // A prompt that starts a sequence runs as one prefill over all its tokens, which leaves the
        // caches where feeding them one at a time would; the tokens that follow are decoded.
        if index_pos == 0 && ids.len() > 1 {
            return self.prefill(&ids);
        }
        // The always-read weights answer a token from the card they were left on: the host reads
        // them at its memory's speed, the card at its own, and a decode reads every one of them
        // at every token. The routed experts stay here - there are hundreds of gigabytes of
        // them and each is read once.
        let mut last = None;
        let Self {
            model,
            state,
            streamed,
            dspark,
            dspark_state,
            ..
        } = self;
        match streamed {
            Some(s) => {
                model.offload_experts(Some(s.lanes.clone()));
                // PROBE (diagnostic, off): measures a batched verify against K decodes on the real,
                // warm next tokens - the continuation a draft would propose - so the amortisation is
                // read on representative experts. Gated off; it perturbs decode timing when live.
                if false && ids.len() == 1 && state.pos > 24 {
                    use std::sync::atomic::{AtomicU64, Ordering};
                    static PN: AtomicU64 = AtomicU64::new(0);
                    if PN.fetch_add(1, Ordering::Relaxed).is_multiple_of(20) {
                        const K: usize = 5;
                        let p0 = state.pos;
                        let first = ids[0];
                        let mark = state.checkpoint();
                        let mut toks_in: Vec<u32> = Vec::with_capacity(K);
                        let mut real: Vec<u32> = Vec::with_capacity(K);
                        let mut main_hidden: Vec<f32> = Vec::new();
                        let mut cur = first;
                        let mut dsum = 0u128;
                        for i in 0..K {
                            toks_in.push(cur);
                            let t0 = std::time::Instant::now();
                            let lg =
                                crate::inference::offload::with_offload(s.offload.clone(), || {
                                    model.forward_decode(cur, state)
                                })?;
                            dsum += t0.elapsed().as_micros();
                            if i == 0 {
                                main_hidden = state.captured.concat();
                            }
                            let v = lg.flatten_all()?.to_vec1::<f32>()?;
                            cur = v
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                .map(|(i, _)| i as u32)
                                .unwrap_or(0);
                            real.push(cur);
                        }
                        state.rewind(&mark);
                        let t0 = std::time::Instant::now();
                        let _ = crate::inference::offload::with_offload(s.offload.clone(), || {
                            model.forward_verify_batch(&toks_in, state)
                        })?;
                        let vtime = t0.elapsed().as_micros();
                        state.rewind(&mark);
                        // The DSpark draft from the same token, against the greedy continuation.
                        if let (Some(d), Some(ds)) = (dspark.as_ref(), dspark_state.as_mut()) {
                            if !main_hidden.is_empty() {
                                let t0 = std::time::Instant::now();
                                let drafts = crate::inference::offload::with_offload(
                                    s.offload.clone(),
                                    || {
                                        d.forward_draft(
                                            ds,
                                            &main_hidden,
                                            first,
                                            p0,
                                            model.embed_ref(),
                                            model.head_ref(),
                                        )
                                    },
                                );
                                let dtime = t0.elapsed().as_micros();
                                match drafts {
                                    Ok(drafts) => {
                                        let acc = drafts
                                            .iter()
                                            .zip(&real)
                                            .take_while(|(a, b)| a == b)
                                            .count();
                                        tracing::info!(
                                            "DSPARK-DRAFT accept {}/{} draft_us={} drafts={:?} real={:?}",
                                            acc, K, dtime, drafts, real
                                        );
                                    }
                                    Err(e) => tracing::warn!("DSpark forward_draft failed: {e}"),
                                }
                            }
                        }
                        tracing::info!(
                            "PROBE verify K={K}: decode_sum={dsum}us verify={vtime}us ratio={:.2} (warm, real tokens)",
                            vtime as f64 / dsum.max(1) as f64
                        );
                    }
                }
                for &t in &ids {
                    last = Some(crate::inference::offload::with_offload(
                        s.offload.clone(),
                        || model.forward_decode(t, state),
                    )?);
                    // Maintain the DSpark window from this committed position's backbone hidden.
                    if let (Some(d), Some(ds)) = (dspark.as_ref(), dspark_state.as_mut()) {
                        let mh = state.captured.concat();
                        if !mh.is_empty() && state.pos > 0 {
                            let _ = d.update_window_from_hidden(ds, &mh, state.pos - 1);
                        }
                    }
                }
            }
            None => {
                for &t in &ids {
                    last = Some(model.forward_decode(t, state)?);
                }
            }
        }
        last.ok_or_else(|| crate::tensor::Error::msg("deepseek_v41: forward of an empty input"))
    }

    /// Logits at every input position, for the speculative verify step: the k draft tokens
    /// appended at `index_pos`, each position's next-token distribution kept rather than only
    /// the last. One batched pass reads each layer's experts once for the whole block, which is
    /// what turns the verify into a speed win. The cache is advanced by every input; the caller
    /// then keeps the accepted prefix through `trim_kv`.
    fn forward_all(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        let ids = x.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<u32>()?;
        if index_pos == 0 {
            self.state = self.model.new_decode_state();
            if let Some(d) = &self.dspark {
                self.state.capture_layers = d.cfg.target_layers.clone();
            }
        }
        if self.state.pos != index_pos {
            return Err(crate::tensor::Error::msg(format!(
                "deepseek_v41: forward_all at position {index_pos} but the cache stands at {}",
                self.state.pos
            )));
        }
        // The mark to return to if the caller keeps only a prefix: the window cannot be cut by
        // length, so a rejection replays the accepted tokens from here.
        self.verify = Some((self.state.checkpoint(), ids.clone(), index_pos));
        let logits = self.verify_batch(&ids)?;
        // [seq, vocab] -> [1, seq, vocab].
        logits.unsqueeze(0)
    }

    /// Keep only the first `new_len - index_pos` tokens of the last `forward_all`: roll the
    /// cache back to the mark that forward took and replay that prefix, one more block read.
    /// A full acceptance never calls here, so the common case pays nothing.
    fn trim_kv(&mut self, new_len: usize) {
        let Some((mark, ids, base)) = self.verify.take() else {
            return;
        };
        self.state.rewind(&mark);
        let keep = new_len.saturating_sub(base).min(ids.len());
        if keep > 0 {
            // The replay reads experts again but leaves the cache exactly where a decode of the
            // accepted tokens would; a failure leaves it at the mark, which the next forward's
            // position check reports.
            let _ = self.verify_batch(&ids[..keep]);
        }
    }

    fn capabilities(&self) -> Capabilities {
        // A batched verify pays only when the block's experts are already in memory: it reads
        // their union once instead of per token. With the experts streamed from disk the union
        // barely overlaps across a block, so a verify reads about as many expert bytes as
        // decoding the tokens one at a time and a rejected tail replays a whole block on top -
        // measured a net loss at 82% acceptance (2.5 vs 3.5 tok/s). Resident, the same verify
        // saves the repeated reads, so the draft path is offered only then.
        Capabilities {
            pld: self.streamed.is_none(),
            ..Capabilities::default()
        }
    }

    fn widest_ffn(&self) -> usize {
        self.widest_ffn
    }

    fn reset_kv_from(&mut self, keep: usize) {
        // The caches only append, so the one prefix that can be kept is the empty one; anything
        // else is left in place and the next forward's position check reports it.
        self.verify = None;
        if keep == 0 {
            self.state = self.model.new_decode_state();
            if let Some(d) = &self.dspark {
                self.state.capture_layers = d.cfg.target_layers.clone();
            }
        }
    }

    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        let n = self.model.n_layers() as u32;
        vec![("cpu".to_string(), 0, 0, n.saturating_sub(1))]
    }

    fn card_residency(&self) -> Option<Vec<(usize, u64)>> {
        // Only a streamed placement keeps weights on the cards; run on the host alone (--cpu,
        // or past the pressure ladder) and there is nothing resident to report, so the layer
        // map stands.
        self.streamed.as_ref()?;
        Some(
            crate::inference::offload::room::residency()
                .into_iter()
                .map(|(ordinal, bytes)| (ordinal, bytes as u64))
                .collect(),
        )
    }

    fn into_serial(self: Box<Self>) -> Result<BoxedModelBackend, BoxedModelBackend> {
        Err(self)
    }
    fn into_batched(
        self: Box<Self>,
        eos: u32,
        max_batch: usize,
    ) -> Result<
        std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>,
        BoxedModelBackend,
    > {
        use crate::inference::model::deepseek_v41::batch_server::V41BatchServer;
        let mut this = *self;
        // Only a card-placed model joins the worker: it runs the same streamed decode the serial
        // path does. A host-only load has no lanes to batch onto and stays serial.
        let Some(streamed) = this.streamed.take() else {
            return Err(Box::new(this));
        };
        // The routed experts run on the cards through the streamed lanes, set once for the worker's
        // lifetime; the card offload runs the attention and the products.
        this.model.offload_experts(Some(streamed.lanes.clone()));
        let card: std::sync::Arc<dyn crate::inference::offload::Offload> = streamed.offload.clone();
        let server = V41BatchServer::spawn(
            std::sync::Arc::new(this.model),
            Some(card),
            Some(eos),
            max_batch,
        );
        let server: std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer> =
            std::sync::Arc::new(server);
        Ok(server)
    }

    fn take_generic(self: Box<Self>) -> Result<GenericHeteroTransformer, BoxedModelBackend> {
        Err(self)
    }
}

impl ModelBackend for Qwen35MoeBackend {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward(x, index_pos)
    }

    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        group_layers_by_device(self.0.layer_device_locations())
    }

    fn snapshot_prefix(&mut self, prompt: &[u32], logits: &Tensor) -> crate::tensor::Result<()> {
        self.0.snapshot_prefix(prompt, logits)
    }

    fn try_restore_prefix(&mut self, prompt: &[u32]) -> crate::tensor::Result<Option<Tensor>> {
        // A snapshot of an earlier prompt that this one extends - the previous turn of a
        // conversation - is restored and advanced over the new tokens.
        let Some((covered, logits)) = self.0.try_restore_prefix(prompt)? else {
            return Ok(None);
        };
        if covered == prompt.len() {
            return Ok(Some(logits));
        }
        let rest = Tensor::new(&prompt[covered..], &crate::tensor::Device::Cpu)?.unsqueeze(0)?;
        // A recurrent state that failed part way through the new tokens cannot be rewound:
        // the whole prompt is prefilled from position 0 instead, which starts a fresh state.
        let logits = match self.forward_prefill_chunked(&rest, covered) {
            Ok(logits) => logits,
            Err(e) => {
                tracing::warn!("prefix snapshot advance failed ({e}); prefilling the whole prompt");
                return Ok(None);
            }
        };
        self.0.snapshot_prefix(prompt, &logits)?;
        Ok(Some(logits))
    }

    fn is_vision_model(&self) -> bool {
        self.is_qwen35_vision()
    }

    fn is_qwen35_vision(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.0.has_vision()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn qwen35_image_token(&self) -> Option<u32> {
        #[cfg(feature = "cuda")]
        {
            if self.0.has_vision() {
                Some(self.0.image_token())
            } else {
                None
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    #[cfg(feature = "cuda")]
    fn forward_qwen35_image(
        &mut self,
        prompt_tokens: &[u32],
        px: &Tensor,
        gh: usize,
        gw: usize,
        device: &Device,
    ) -> crate::tensor::Result<(Tensor, usize)> {
        let m = &mut self.0;
        let img_tok = m.image_token();
        let n_merged = (gh / 2) * (gw / 2);
        let mut ids = Vec::with_capacity(prompt_tokens.len() + n_merged);
        let mut expanded = false;
        for &t in prompt_tokens {
            if t == img_tok && !expanded {
                ids.extend(std::iter::repeat_n(img_tok, n_merged));
                expanded = true;
            } else {
                ids.push(t);
            }
        }
        if !expanded {
            return Err(crate::tensor::Error::msg(
                "qwen35 vision: image_token sentinel missing from prompt".to_string(),
            ));
        }
        let input_ids = Tensor::new(ids.as_slice(), device)?.unsqueeze(0)?;
        m.forward_with_image(&input_ids, px, gh, gw)
    }

    take_generic_passthrough!();
}

#[cfg(feature = "cuda")]
impl ModelBackend for TpQwen2Backend {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        // The engine passes [1, seq] token ids; TpQwen2 takes a flat slice and returns the
        // last token's logits [vocab], so wrap them back to [1, vocab].
        let toks: Vec<u32> = x.flatten_all()?.to_vec1::<u32>()?;
        self.0
            .forward(&toks, index_pos)
            .map_err(crate::tensor::Error::msg)?
            .unsqueeze(0)
    }

    fn reset_kv_from(&mut self, keep: usize) {
        // TP only ever prefills at position 0 and resets there.
        if keep == 0 {
            self.0.reset_kv();
        }
    }

    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        // Tensor-parallel: every layer is on both cards, which is the whole point.
        self.0.device_layer_distribution().1
    }

    take_generic_passthrough!();
}

impl ModelBackend for ContinuousBackend {
    fn forward(&mut self, _x: &Tensor, _index_pos: usize) -> crate::tensor::Result<Tensor> {
        // CB-served models are driven by the worker, never the serial forward.
        Err(crate::tensor::Error::msg(
            "Continuous (CB-served) model: serial forward is not supported; \
             requests must be delegated to the ContinuousServer worker"
                .to_string(),
        ))
    }

    fn batched_server(
        &self,
    ) -> Option<std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>> {
        Some(self.0.clone())
    }

    fn take_generic(self: Box<Self>) -> Result<GenericHeteroTransformer, BoxedModelBackend> {
        Err(self)
    }
    fn into_batched(
        self: Box<Self>,
        _eos: u32,
        _max_batch: usize,
    ) -> Result<
        std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>,
        BoxedModelBackend,
    > {
        Err(self)
    }
    fn into_serial(self: Box<Self>) -> Result<BoxedModelBackend, BoxedModelBackend> {
        let server = match std::sync::Arc::try_unwrap(self.0) {
            Ok(server) => server,
            Err(shared) => return Err(Box::new(ContinuousBackend(shared))),
        };
        match server.stop() {
            Ok(model) => Ok(Box::new(GenericBackend(model))),
            Err(e) => {
                tracing::warn!("continuous batching could not hand its model back: {e}");
                Err(Box::new(TakenBackend))
            }
        }
    }
}

impl ModelBackend for BatchedBackend {
    fn forward(&mut self, _x: &Tensor, _index_pos: usize) -> crate::tensor::Result<Tensor> {
        // Batch-served: the worker is the sole caller of the model forward, never this.
        Err(crate::tensor::Error::msg(
            "V4.1 batch-served model: serial forward is not supported; \
             requests are delegated to the batch worker"
                .to_string(),
        ))
    }

    fn batched_server(
        &self,
    ) -> Option<std::sync::Arc<dyn crate::inference::serve::continuous_serve::BatchedServer>> {
        Some(self.0.clone())
    }

    take_generic_passthrough!();
}

impl ModelBackend for TakenBackend {
    fn forward(&mut self, _x: &Tensor, _index_pos: usize) -> crate::tensor::Result<Tensor> {
        Err(crate::tensor::Error::msg(
            "TakenBackend used (model swap placeholder)".to_string(),
        ))
    }
    take_generic_passthrough!();
}

#[cfg(test)]
mod prefill_chunk_memo_tests {
    /// A residency change must discard the memo, or a chunk sized against an empty card
    /// survives into a full one - which is the failure the adaptive sizing exists to prevent.
    /// Judged on the epoch alone, so it needs no GPU: the memo key is (epoch, widest_ffn).
    #[test]
    fn a_residency_change_moves_the_epoch() {
        let before = crate::inference::place::vram_manager::residency_epoch();
        crate::inference::place::vram_manager::residency_changed();
        let after = crate::inference::place::vram_manager::residency_epoch();
        assert_ne!(
            before, after,
            "the memo key never changes, so it never invalidates"
        );
    }

    /// Two calls with no residency change in between agree. Without the memo they would too,
    /// but each would have re-probed both cards and trimmed the pools to say so.
    #[test]
    fn the_answer_is_stable_while_residency_is() {
        let a = super::adaptive_prefill_chunk(14336);
        let b = super::adaptive_prefill_chunk(14336);
        assert_eq!(a, b);
    }
}

#[cfg(test)]
mod capabilities_tests {
    use super::Capabilities;

    /// A backend that overrides nothing claims nothing.
    #[test]
    fn the_default_claims_nothing() {
        assert_eq!(
            Capabilities::default(),
            Capabilities {
                trim_kv: false,
                shift_kv: false,
                pld: false,
                graph_mode: false,
                speculative: false,
            }
        );
    }
}
