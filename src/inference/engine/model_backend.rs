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

use tracing::warn;

use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::inference::model::moondream::quantized as moondream;
use crate::tensor::{Device, Tensor};

use crate::inference::engine::llm_engine::{
    is_cuda_oom, VisionTower, ADAPTIVE_PREFILL_CHUNK, PREFILL_CHUNK_TOKENS,
    PREFILL_CHUNK_TOKENS_MIN,
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

pub(crate) trait ModelBackend: Send {
    // --- Core forwards --------------------------------------------------
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor>;

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
        if seq_len <= chunk_start {
            // Single-forward fast path. Even here, a genuinely huge single
            // chunk on a near-full GPU can OOM; if it does and the chunk is
            // splittable, fall through to the adaptive chunked path rather
            // than surfacing the OOM to the user.
            match self.forward(x, index_pos) {
                Ok(t) => return Ok(t),
                Err(e) if is_cuda_oom(&e) && seq_len > PREFILL_CHUNK_TOKENS_MIN => {
                    warn!("prefill OOM on a single {seq_len}-token forward ({e}); retrying chunk-adaptive");
                    // Drop any partial KV the failed forward may have appended,
                    // preserving the session prefix [0, index_pos).
                    self.reset_kv_from(index_pos);
                }
                Err(e) => return Err(e),
            }
        }
        // Adaptive chunked prefill. Start at the default compute batch and, on
        // a CUDA OOM, halve the chunk (down to PREFILL_CHUNK_TOKENS_MIN) and
        // restart the whole prefill from `index_pos`. A smaller chunk shrinks
        // the per-forward activation peak (`chunk x widest_ffn x f32`), the
        // attention scores tensor (`heads x chunk x total`), and the KV-grow
        // transient - exactly the runtime allocations that OOM a near-full
        // 2-GPU / co-resident model at long context. The final chunk's logits
        // (the next-token prediction) are identical regardless of chunk size,
        // so this degrades speed gracefully without changing the result.
        use std::sync::atomic::Ordering;
        // Seed from the last known-good chunk for this loaded model (if any),
        // so a model that already needed shrinking doesn't re-pay the full
        // ladder on every request. Clamp into [MIN, default].
        // Asked once per prefill: a second eviction round would only be evicting things this
        // request just made room for.
        let mut asked_for_room = false;
        let widest = self.widest_ffn();
        let remembered = ADAPTIVE_PREFILL_CHUNK.load(Ordering::Relaxed);
        let mut chunk_tokens = if remembered == 0 {
            chunk_start
        } else {
            remembered.clamp(PREFILL_CHUNK_TOKENS_MIN, chunk_start)
        };
        loop {
            match self.forward_prefill_with_chunk(x, index_pos, seq_len, chunk_tokens) {
                Ok(t) => {
                    // Remember the chunk that fit (only when it differs from the
                    // default - the common single-forward case never writes).
                    if chunk_tokens < chunk_start {
                        ADAPTIVE_PREFILL_CHUNK.store(chunk_tokens, Ordering::Relaxed);
                    }
                    return Ok(t);
                }
                Err(e) if is_cuda_oom(&e) && chunk_tokens > PREFILL_CHUNK_TOKENS_MIN => {
                    let next = (chunk_tokens / 2).max(PREFILL_CHUNK_TOKENS_MIN);
                    warn!(
                        "prefill OOM at chunk={chunk_tokens} ({e}); shrinking prefill chunk to {next} and retrying"
                    );
                    // Discard the KV appended by the partial failed pass so the
                    // retry replays cleanly from `index_pos` (preserving any
                    // session-reuse prefix in [0, index_pos)).
                    self.reset_kv_from(index_pos);
                    chunk_tokens = next;
                }
                Err(e) if is_cuda_oom(&e) && !asked_for_room => {
                    // The ladder has bottomed out: at the minimum chunk the activation peak is
                    // small, so a card that still cannot serve it is full of something else  -
                    // an idle model from another engine, most often. Every other engine asks
                    // the pressure protocol for room before giving up; this path never did,
                    // and surfaced the OOM to the caller instead while gigabytes sat reclaimable
                    // beside it.
                    asked_for_room = true;
                    let want = (chunk_tokens * widest.max(1) * 4 * 2) as u64;
                    if crate::inference::place::vram_manager::ensure_gpu_headroom_blocking(
                        "llm", want,
                    ) {
                        warn!("prefill OOM at the minimum chunk ({e}); freed room, retrying");
                        self.reset_kv_from(index_pos);
                    } else {
                        warn!("prefill OOM at the minimum chunk ({e}); nothing else holds VRAM");
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            }
        }
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

    /// Whether this variant can trim its KV cache to an arbitrary length
    /// (vs only reset-from-offset-0). Session-persistent KV only activates
    /// when true.
    fn supports_trim_kv(&self) -> bool {
        false
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

    /// Whether this variant supports prompt-lookup speculative decoding.
    fn supports_pld(&self) -> bool {
        false
    }

    // --- Exact-prompt state snapshot (recurrent-hybrid prefix reuse) -----
    //
    // `trim_kv` reuse assumes the state can return to an arbitrary token
    // position. Hybrids carrying a rolling recurrent state (short-conv / SSM)
    // cannot: that state is advanced per token and can never be rewound, nor
    // rebuilt from the KV - which is why `supports_trim_kv` is correctly false
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

    /// Padded forward for CUDA graph mode (fixed-size attention buffers).
    fn forward_padded(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.forward(x, index_pos)
    }

    /// Split step 1: QKV + RoPE + KV write (outside graph).
    fn prepare_all_kv(
        &mut self,
        _x: &Tensor,
        _index_pos: usize,
    ) -> crate::tensor::Result<(Tensor, Vec<Tensor>)> {
        crate::tensor::bail!("prepare_all_kv only for GenericHetero / GenericHeteroVision")
    }

    /// Split step 2: attention + FFN + output (inside graph).
    fn compute_all_from_kv(
        &mut self,
        _hidden: &Tensor,
        _all_q: &[Tensor],
    ) -> crate::tensor::Result<Tensor> {
        crate::tensor::bail!("compute_all_from_kv only for GenericHetero / GenericHeteroVision")
    }

    /// Graph-compatible decode forward (single-token, single CUDA device).
    /// Caller must have called `update_graph_state(pos)` immediately prior.
    /// Only implemented for `GenericHetero`; other variants fall back.
    fn forward_graph(&mut self, _x: &Tensor) -> crate::tensor::Result<Tensor> {
        crate::tensor::bail!("forward_graph only for GenericHetero / GenericHeteroVision")
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

    /// Whether the engine's CUDA-graph decode dispatch applies to this arch
    /// at all (the old `matches!(GenericHetero | GenericHeteroVision)` gate).
    fn supports_graph_mode(&self) -> bool {
        false
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

    /// Whether this model drives a draft/verify split. No variant does today.
    fn supports_speculative(&self) -> bool {
        false
    }

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
    fn continuous_server(
        &self,
    ) -> Option<&std::sync::Arc<crate::inference::serve::continuous_serve::ContinuousServer>> {
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
}

/// `take_generic` passthrough for every backend that is NOT GenericHetero.
macro_rules! take_generic_passthrough {
    () => {
        fn take_generic(self: Box<Self>) -> Result<GenericHeteroTransformer, BoxedModelBackend> {
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

/// Multi-GPU Qwen3-MoE - layer-split across CUDA devices. Required for
/// 30B+ MoE models that don't fit on a single GPU (multi-GPU layer split).
/// Also runs on CPU: the MoE GEMM has a fallback CPU path and the attention
/// uses the plain F-dtype KV cache (the Q8/Q4 KV fast paths are CUDA-only).
pub(crate) struct QwenMoEMultiBackend(
    pub crate::inference::model::qwen3::moe_multi::MultiDeviceQwen3MoE,
);

/// gpt-oss (OPENAI_MOE): MXFP4 MoE transformer with attention sinks +
/// sliding-window + NEOX RoPE. Single-device for now. (CUDA-only: MoE GEMM.)
pub(crate) struct GptOssBackend(pub crate::inference::model::gptoss::GptOssModel);

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

/// Transient sentinel used ONLY while swapping a loaded GenericHetero model
/// into a Continuous worker (`std::mem::replace` placeholder) - never stored
/// or used; any method call on it is a bug.
pub(crate) struct TakenBackend;

// Safety: Mutex<> provides exclusive access, preventing concurrent RefCell
// mutation (contract inherited from the old `unsafe impl Send for ModelVariant`).
unsafe impl Send for MoondreamBackend {}
unsafe impl Send for GenericBackend {}
unsafe impl Send for GenericVisionBackend {}
unsafe impl Send for QwenMoEMultiBackend {}
unsafe impl Send for GptOssBackend {}
unsafe impl Send for NemotronHBackend {}
unsafe impl Send for Lfm2MoeBackend {}
unsafe impl Send for Qwen35MoeBackend {}
#[cfg(feature = "cuda")]
unsafe impl Send for TpQwen2Backend {}
unsafe impl Send for ContinuousBackend {}

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

    fn supports_trim_kv(&self) -> bool {
        // Routed mixtures were excluded because the top-k choice follows the GEMM
        // shape, and a warm re-prefill used shapes a cold run never had - measured
        // divergent on GPU and host alike. Since reuse snaps to the cold run's chunk
        // grid the warm tail runs the exact cold shapes, expert choice included, so
        // the exclusion is lifted; the cold/warm probe on a routed model is the gate
        // that must hold.
        true
    }

    fn supports_pld(&self) -> bool {
        // Per-model PLD opt-out: gemma4 family (dense + MoE) shows
        // ~13-30% n-gram acceptance - straddling or below break-even.
        // Gemma4's chat template + tokenizer don't repeat n-grams the
        // way PLD relies on; gemma4:latest measured 20x speedup with
        // PLD disabled (4.6 -> 115 tok/s on a 64-token medium gen).
        if self.0.is_gemma4_arch() {
            return false;
        }
        // Small-model PLD opt-out, measured.
        // For the sub-~1B dense tier a decode step is so cheap (~1.5 ms,
        // ~300 MB weight read) that PLD's multi-position verify forward +
        // KV rollback + n-gram bookkeeping costs MORE than the tokens its
        // accepts save - even at moderate acceptance. The verify amortizes
        // a weight read across K positions, which is the whole PLD win, but
        // a 0.6B model's weight read is already tiny so there is little to
        // amortize while the fixed per-step overhead is unchanged.
        // Measured qwen3:0.6b: PLD OFF 641 vs ON 597 tok/s short (+7.4%),
        // 630 vs 605 medium at 38% accept (+4%) - PLD never pays and leaves
        // us -8% vs ollama's 645; OFF ties ollama. qwen2.5:0.5b PLD-neutral.
        // Larger models keep PLD (deepcoder 14B / qwen3:8b / mistral-nemo
        // 12B all ≫ 1B). PLD is output-exact (drafts verified against the
        // target's greedy sample), so this only changes speed, never text.
        // Threshold sits above the measured 0.6B tier (~0.6B params) and
        // well below the smallest model where PLD is retained (~1.1B).
        if self.0.decode_weight_params() < 750_000_000 {
            return false;
        }
        true
    }

    fn forward_padded(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward_padded(x, index_pos)
    }

    fn prepare_all_kv(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> crate::tensor::Result<(Tensor, Vec<Tensor>)> {
        self.0.prepare_all_kv(x, index_pos)
    }

    fn compute_all_from_kv(
        &mut self,
        hidden: &Tensor,
        all_q: &[Tensor],
    ) -> crate::tensor::Result<Tensor> {
        self.0.compute_all_from_kv(hidden, all_q)
    }

    fn forward_graph(&mut self, x: &Tensor) -> crate::tensor::Result<Tensor> {
        self.0.forward_graph(x)
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

    fn supports_graph_mode(&self) -> bool {
        true
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
}

impl ModelBackend for GenericVisionBackend {
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

    fn supports_trim_kv(&self) -> bool {
        true
    }

    // Vision-conditioned generation produces image captions
    // (moondream's primary use): low n-gram repetition, so PLD
    // acceptance is 0-10 % and the multi-position verify cost
    // dominates the rare-accept gain. Measured:
    // PLD on vision adds ~5-10 % decode overhead net.
    // Re-enable per-arch when a vision use case with bursty
    // repeating tokens emerges (e.g. structured-output VLM).
    // (supports_pld stays at the default `false`.)

    fn forward_padded(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.text.forward_padded(x, index_pos)
    }

    fn prepare_all_kv(
        &mut self,
        x: &Tensor,
        index_pos: usize,
    ) -> crate::tensor::Result<(Tensor, Vec<Tensor>)> {
        self.text.prepare_all_kv(x, index_pos)
    }

    fn compute_all_from_kv(
        &mut self,
        hidden: &Tensor,
        all_q: &[Tensor],
    ) -> crate::tensor::Result<Tensor> {
        self.text.compute_all_from_kv(hidden, all_q)
    }

    fn forward_graph(&mut self, x: &Tensor) -> crate::tensor::Result<Tensor> {
        self.text.forward_graph(x)
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

    fn supports_graph_mode(&self) -> bool {
        true
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

impl ModelBackend for QwenMoEMultiBackend {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward(x, index_pos)
    }

    fn forward_all(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward_all(x, index_pos)
    }

    fn trim_kv(&mut self, new_len: usize) {
        self.0.trim_kv(new_len)
    }

    fn reset_kv_from(&mut self, keep: usize) {
        self.0.trim_kv(keep)
    }

    fn supports_trim_kv(&self) -> bool {
        true
    }

    // MoE PLD: OFF - verify cost > accept gain on GGUF MoE. RE-MEASURED
    // Marlin: qwen3-coder code-gen reaches a high
    // 48-62% n-gram acceptance, yet enabling PLD dropped it 166 -> 108 tok/s
    // (-23.7% vs ollama, high variance). 's cheap small-M verify is the
    // AWQ/Marlin path; GGUF MoE verify still runs the slow `moe_gemm_gguf`
    // small-batch forward, so each multi-position verify costs more than the
    // accepted tokens save. Needs a cheap GGUF-MoE small-batch verify kernel
    // (or AWQ weights) before PLD pays on MoE.
    // (supports_pld stays at the default `false`.)

    fn device_layer_distribution(&self) -> Vec<(String, usize, u32, u32)> {
        group_layers_by_device(self.0.layer_device_locations())
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

impl ModelBackend for Qwen35MoeBackend {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> crate::tensor::Result<Tensor> {
        self.0.forward(x, index_pos)
    }

    fn snapshot_prefix(&mut self, prompt: &[u32], logits: &Tensor) -> crate::tensor::Result<()> {
        self.0.snapshot_prefix(prompt, logits)
    }

    fn try_restore_prefix(&mut self, prompt: &[u32]) -> crate::tensor::Result<Option<Tensor>> {
        self.0.try_restore_prefix(prompt)
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
                ids.extend(std::iter::repeat(img_tok).take(n_merged));
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

    fn continuous_server(
        &self,
    ) -> Option<&std::sync::Arc<crate::inference::serve::continuous_serve::ContinuousServer>> {
        Some(&self.0)
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
