//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl LlmEngine {
    /// Unload the current model
    pub async fn unload(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut state = self.model_state.lock().await;
        if let Some(model) = state.take() {
            info!("🔌 Unloading model: {}", model.name);
        }
        // THE PROMPT CACHE DESCRIBES A KV THAT NO LONGER EXISTS.
        //
        // Each entry records how many rows the resident cache held when it was written.
        // Dropping the model empties that cache, and leaving the entries behind lets the
        // next request resume from a position the fresh cache never had: the attention
        // mask is built for the remembered length, the KV tensor has the real one, and
        // the request dies on `broadcast_as: cannot broadcast [1589, 4547] to
        // [1, 16, 1589, 2639]`. Nothing upstream can catch it - the entry names the right
        // model and a genuinely matching prefix of tokens; only the LENGTH is a memory of
        // something that has been freed.
        //
        // It surfaces under exactly the load this server is meant to take: an image
        // render asks for VRAM, the LLM is evicted to make room, and the next chat
        // request resumes into the gap. Clearing every entry is right rather than just
        // the global one - the cache is one resident KV, and it is gone for all of them.
        self.sessions.lock().await.clear();
        // A speculative-decode DRAFT engine holds a whole second model. It was not
        // dropped here, so an engine that had ever drafted kept its drafter's
        // weights resident for the process lifetime - VRAM that no /api/ps entry
        // accounts for and that later loads have to plan around.
        if let Some(draft) = self.draft_engine.lock().await.take() {
            // Drop its model state directly rather than recursing into unload()
            // (an async fn cannot call itself without boxing, and the drafter has
            // no drafter of its own): taking the state releases the weights, and
            // the pool trim + cache clears below cover both engines.
            let _ = draft.model_state.lock().await.take();
        }
        // Clear cached model size when unloading
        let mut size = self.cached_model_size.lock().await;
        *size = 0;
        // After all tensors drop, the cudarc stream-ordered allocator
        // keeps freed blocks in a per-device memory pool. nvidia-smi
        // continues to report the pool as in-use, so subsequent
        // HeteroPlan calculations see less free memory than is actually
        // available and may unnecessarily push layers to CPU. Force-trim
        // the default mempool on every CUDA device to return all unused
        // bytes to the system. Safe because no tensors reference these
        // allocations anymore (we just dropped them above).
        #[cfg(feature = "cuda")]
        {
            release_cuda_pools();
        }
        // The MoE expert caches are keyed by weight identity, which no reload reuses, so
        // their entries - full copies of every expert tensor - would otherwise stay resident
        // for the process lifetime: one model-sized leak per reload.
        crate::inference::moe_cuda::clear_expert_cache();
        // CUDA build also caches CPU expert QMatMuls in the moe_cpu twin (used by
        // lfm2/nemotron MoE under `--cpu`); clear it too to avoid the same leak.
        #[cfg(feature = "cuda")]
        crate::inference::moe_cpu::clear_expert_cache();
        // Return freed HEAP to the OS too. Dropping a CPU model frees its
        // buffers to glibc, but the arenas retain them - repeated
        // load/unload cycles accumulate retained RSS until the OOM-killer
        // steps in. malloc_trim releases the retained arena space (no-op on
        // non-glibc).
        #[cfg(target_os = "linux")]
        unsafe {
            libc::malloc_trim(0);
        }
        Ok(())
    }

    /// Generate text (non-streaming) with real model inference
    /// Runs a generation, and gives an out-of-memory one more chance before it
    /// can reach the caller.
    ///
    /// The load path re-plans a placement that does not fit and the prefill path
    /// halves its chunk, but the activation allocations of the decode loop were
    /// covered by neither: a model whose weights fit while its KV cache at full
    /// context does not returned a 500. The pools hold blocks that belong to no
    /// live tensor, so handing them back is often enough; when it is not, the
    /// second failure is reported, and it is the honest one.
    ///
    /// The error is inspected and DROPPED before the retry is awaited. The error
    /// type is `Box<dyn Error>` with no `+ Send`, so holding it across that await
    /// makes this future non-Send - which axum rejects, and not here but at the
    /// three routes that share this call.
    pub async fn generate(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationResult, Box<dyn std::error::Error>> {
        // Scoped so the error - `Box<dyn Error>`, with no `+ Send` - is gone
        // before the retry is awaited. Anything alive across that await must be
        // Send or the three routes sharing this call stop compiling.
        {
            match self.generate_once(prompt, params.clone()).await {
                Ok(r) => return Ok(r),
                Err(e) => {
                    if !is_cuda_oom(&e) {
                        return Err(e);
                    }
                    warn!("OOM during generation ({e}); returning the pools and retrying once");
                }
            }
        }
        release_cuda_pools();
        // Second attempt on the pools alone. If the shortfall was fragmentation
        // this is where it ends; the scope above already dropped the error.
        {
            match self.generate_once(prompt, params.clone()).await {
                Ok(r) => return Ok(r),
                Err(e) => {
                    if !is_cuda_oom(&e) {
                        return Err(e);
                    }
                    warn!(
                        "still out of memory after returning the pools; re-planning the placement"
                    );
                }
            }
        }
        // A third stage that dropped the resident model was tried and removed:
        // `generate_once` requires one to be loaded and does not reload, so the
        // retry answered "Model not loaded" - a worse failure than the one it
        // was meant to repair. Re-planning belongs where the load happens, not
        // here, and the budget now charges the KV cache so the FIRST plan is
        // the right one.
        release_cuda_pools();
        self.generate_once(prompt, params).await
    }

    pub(super) async fn generate_once(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<GenerationResult, Box<dyn std::error::Error>> {
        // Grammar-constrained decoding takes a dedicated, simpler path that
        // bypasses PLD/graph branches.
        if params.grammar.is_some() {
            return self.generate_with_grammar(prompt.to_string(), params).await;
        }
        // CB-served models: drive the non-stream path through the same worker
        // delegation, collecting the streamed chunks into one result.
        if self.is_continuous().await {
            let t0 = std::time::Instant::now();
            // Word-based prompt-token estimate (≈ the handler's estimate_token_count;
            // the precise count lives behind the tokenizer the worker owns).
            let prompt_eval_count =
                ((prompt.split_whitespace().count() as f64) * 1.3).ceil() as u64;
            let mut rx = self.generate_stream(prompt, params).await?;
            let mut text = String::new();
            let mut chunks: u64 = 0;
            let mut first: Option<std::time::Duration> = None;
            while let Some(msg) = rx.recv().await {
                match msg {
                    Ok(chunk) => {
                        if first.is_none() {
                            first = Some(t0.elapsed());
                        }
                        chunks += 1; // one emitted chunk ≈ one generated token
                        text.push_str(&chunk);
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            let ttft = first.unwrap_or_default();
            return Ok(GenerationResult {
                text,
                tokens: Vec::new(),
                prompt_eval_count,
                prompt_eval_duration: ttft.as_nanos() as u64,
                eval_count: chunks,
                eval_duration: t0.elapsed().saturating_sub(ttft).as_nanos() as u64,
            });
        }
        let model_state = self.model_state.clone();
        let config = self.config.clone();
        let prompt = prompt.to_string();
        // Spec-decode: load the drafter, when one is configured, before the
        // blocking decode; capture a cheap Arc-clone engine for use inside. Output
        // correctness is guaranteed by the UNCHANGED target verify (it samples every
        // position) - the drafter only proposes, so a drafter bug costs speed, never
        // correctness.
        // Spec-decode: capture a cheap Arc-clone engine; the drafter is loaded
        // lazily INSIDE the blocking decode closure via block_on (the drafter load is
        // async + !Send, so it can't be awaited in this Send-required handler future  - 
        // block_on on the spawn_blocking thread sidesteps that). Output correctness is
        // guaranteed by the UNCHANGED target verify; the drafter only proposes.
        let spec_engine = self.clone_for_spec();
        // In-memory global prompt cache (captured into the blocking closure).
        let sessions = self.sessions.clone();
        // Owned: the meter is fed from inside the blocking closure, which cannot borrow self.
        let metered_model = self.config.model_id.clone();

        let inner_result = tokio::task::spawn_blocking(move || -> AnyResult<GenerationResult> {
            let mut guard = model_state.blocking_lock();
            let state = guard.as_mut()
                .ok_or_else(|| anyhow!("Model not loaded. Load a model first."))?;

            info!("🤖 Generating with model: {}", state.name);

            // Determinism barrier: when per-thread streams are in use,
            // the tokio blocking-pool thread that runs *this* request can
            // differ from the one that ran the previous request. Per-thread
            // streams are per-CALLING-thread, so without an explicit barrier
            // here the new thread can start submitting work before the
            // previous thread's kernels have physically completed on the
            // GPU - non-deterministic output across calls. Also binds the
            // primary CUDA context to the calling thread (mirrors llama.cpp's
            // `cudaSetDevice` discipline before each backend op).
            #[cfg(feature = "cuda")]
            if state.device.is_cuda() {
                if let Ok(cd) = state.device.as_cuda_device() {
                    let _ = cd.cuda_stream().context().bind_to_thread();
                    let _ = cd.cuda_stream().synchronize();
                }
            }

            // Fix: reset KV cache before prefill. This path does
            // not perform session lookup / prefix-KV reuse (only the
            // streaming path does), but the underlying KV cache is shared
            // across all `generate` calls. Without an explicit trim_kv(0),
            // the cache retains the prior request's K/V; forward(&x, 0)
            // then appends rather than resetting, leaving stale K/V at
            // positions [prompt_len, prior_seq_len). Attention reads those
            // stale positions -> cyclic-3 non-determinism across identical
            // sequential requests at temperature=0 (verified via
            // scripts/repro_task26_determinism.py).
            //
            // trim_kv(0) is a pointer reset (no buffer wipe), ~O(num_layers)
            // and negligible - safe to call even when the cache is empty.
            // NOTE: the unconditional reset moved below - after tokenization the
            // GLOBAL prompt cache may instead trim to a reused prefix length.

            // Tokenize prompt. Its TEXT is user content - log the size, never the text.
            debug!("Tokenizing prompt: {} chars", prompt.len());
            let encoding = state.tokenizer.encode(prompt.as_str(), true)
                .map_err(|e| anyhow!("Tokenization error: {}", e))?;
            // Robustness: clamp an over-long prompt to the KV window (else prefill
            // overflows the KV cache and the attention mask breaks -> crash).
            // Vision requests splice image embeds into the prefill AFTER
            // tokenization, so reserve their KV positions here - and reject
            // outright when the image alone can't fit the window.
            let kv_window = effective_kv_window(state.context_length, config.context_length, params.context_length);
            let vision_extra = vision_extra_kv(state.image_embeds.as_ref())
                + qwen35_vision_extra_kv(state.qwen35_image.as_ref());
            if vision_extra >= kv_window {
                return Err(anyhow!(
                    "image occupies {vision_extra} KV positions but the context window is only \
                     {kv_window}; increase context_length / num_ctx or use a smaller image"
                ));
            }
            // qwen35-VL keeps its image as a single in-prompt sentinel token  - 
            // clamp around it (the generic BOS+tail clamp would drop it).
            let qwen35_sentinel = if state.qwen35_image.is_some() {
                state.model.qwen35_image_token()
            } else { None };
            let prompt_tokens: Vec<u32> = match qwen35_sentinel {
                Some(sent) => clamp_qwen35_prompt_to_window(
                    with_prefix_tokens(&params, encoding.get_ids()),
                    kv_window - vision_extra,
                    params.max_tokens.unwrap_or(config.max_tokens), sent),
                None => clamp_prompt_to_window(
                    with_prefix_tokens(&params, encoding.get_ids()),
                    kv_window - vision_extra,
                    params.max_tokens.unwrap_or(config.max_tokens)),
            };

            // Request-start graph hygiene: a prior request's graph state
            // (per-layer rope/mask buffers, quantized-KV capture flags) must
            // not leak into this request. The captured graph itself is
            // request-local (dropped at the end of the generation loop), but
            // the layer buffers persist on the model - a stale
            // `graph_rope_cos` would make a single-token prefill delta take
            // `use_graph_ops` with FROZEN rope values -> silent garbage, and
            // a stale `graph_captured` flag would fail the new prefill's KV
            // growth. Cheap (~n_layers Option resets); a graph-mode request
            // re-primes everything on its first decode token. Does NOT touch
            // KV contents, so prefix reuse below is unaffected.
            state.model.invalidate_graph_state();

            // -- GLOBAL prompt cache (ollama parity, non-streaming) -----------
            // A sessionless text request reuses the common prefix of the single
            // resident KV (mirrored under GLOBAL_PROMPT_CACHE_KEY, in-memory only).
            // Same proven trim_kv logic as generate_stream -> greedy-identical. Vision
            // requests fall back to a full reset.
            let cache_session_start: usize = {
                let is_vision_req = (state.model.is_vision_model() && state.image_embeds.is_some())
                    || state.qwen35_image.is_some();
                let disabled = is_vision_req
                    || false;
                kv_reuse_start(state.model.as_mut(), &sessions, &state.name, &prompt_tokens, disabled)
            };

            // MAKE THE CACHE ENTRY HONEST NOW, NOT WHEN THE REQUEST FINISHES.
            //
            // The entry is only rewritten after a generation completes. A request that
            // trims the KV and then never finishes - the client disconnects, the request
            // times out, a decode errors - leaves the resident cache SHORTER than the
            // entry claims. The next request that shares a prefix then resumes from a
            // position those rows were trimmed out of, builds its attention mask for the
            // remembered length, and dies on `broadcast_as: cannot broadcast
            // [1589, 4547] to [1, 16, 1589, 2639]` - the two lengths side by side.
            //
            // Recording the post-trim length immediately can only UNDERSTATE what the
            // cache holds (this request is about to prefill more onto it), and
            // understating costs a re-prefill where overstating costs the request.
            if cache_session_start > 0 {
                let mut g = sessions.blocking_lock();
                if let Some(sess) = g.get_mut(GLOBAL_PROMPT_CACHE_KEY) {
                    sess.tokens.truncate(cache_session_start);
                    sess.kv_len = cache_session_start;
                }
            }

            // Token ids decode back to the prompt, so only the count is logged.
            debug!("Prompt tokens: {}", prompt_tokens.len());

            // Resolve generation parameters: per-request params override config defaults
            // Per-request num_ctx is clamped to the model's loaded context window.
            // Effective context = the actual KV window (config cap ∧ model ∧ request),
            // not the model's advertised max - so max_tokens can't be sized past the
            // KV cache and overrun it during decode.
            let effective_context = effective_kv_window(state.context_length, config.context_length, params.context_length);
            // The KV a vision request consumes = text prompt + spliced image
            // embeds; count both so max_tokens can't push decode past the
            // allocated KV cap (Q8 append fails there -> degenerate output).
            let resolved = resolve_gen_params(&params, &config, effective_context, prompt_tokens.len() + vision_extra);
            let ResolvedGenParams {
                max_tokens, temperature, top_k, repeat_penalty, repeat_last_n, ..
            } = resolved;
            // Clone the device so subsequent state.forward (mutable borrow)
            // doesn't conflict with the previous immutable borrow.
            let device_owned = state.device.clone();
            let device = &device_owned;

            // Build logits processor with resolved parameters
            let mut logits_processor = resolved.make_logits_processor();

            // Track recent tokens for repetition penalty
            let mut recent_tokens: Vec<u32> = prompt_tokens.clone();

            // Set inference modes before forward passes
            state.model.set_early_exit_threshold(params.early_exit_threshold);


            // Batched prefill (all prompt tokens in one forward pass)
            let prompt_token_count = prompt_tokens.len() as u64;
            let prefill_start = std::time::Instant::now();
            // Prefill only the divergent suffix when the global cache reused a
            // prefix (cache_session_start>0, text path). Vision/no-cache -> start=0 ->
            // full prompt (vision branches below rely on the full sequence).
            let x = Tensor::new(&prompt_tokens[cache_session_start..], device)?.unsqueeze(0)?;
            // qwen35moe (Qwen3-VL) vision takes a DEDICATED prefill (ViT +
            // splice at the image_token sentinel + mRoPE-2D); `pos` becomes the
            // continuing logical mRoPE position. Otherwise the moondream prepend
            // path (or plain text).
            let qwen35_img = state.qwen35_image.take();
            let (vision_prefix_len, mut logits, qwen35_pos) = if let Some((ref px, (gh, gw))) = qwen35_img {
                let (lg, next_pos) = state.model.forward_qwen35_image(&prompt_tokens, px, gh, gw, device)?;
                (None, lg, Some(next_pos))
            } else if state.model.is_pixtral_vision() && state.image_embeds.is_some() {
                // Pixtral: embeds spliced INSIDE the prompt ([INST] img [IMG_END] text);
                // the method returns the true total length -> decode position.
                let image_embeds = state.image_embeds.as_ref().unwrap().clone();
                let (lg, total) = state.model.forward_pixtral_spliced(&prompt_tokens, &image_embeds, device)?;
                (None, lg, Some(total))
            } else {
                let vpl = if let (true, Some(ref image_embeds)) = (state.model.is_vision_model(), &state.image_embeds) {
                    // BOS(1) + image embeddings (typically 729 patches for
                    // moondream) get spliced ahead of the text tokens.
                    Some(1 + image_embeds.dim(1)?)
                } else { None };
                let lg = if vpl.is_some() {
                    let image_embeds = state.image_embeds.as_ref().unwrap();
                    let bos_token = Tensor::new(&[state.eos_token_id], device)?.unsqueeze(0)?;
                    state.model.forward_with_img(&bos_token, &x, image_embeds)?
                } else if state.qwen35_image.is_some() {
                    // Image-conditioned: the snapshot keys on text tokens alone,
                    // so it would collide across different images. Always prefill.
                    state.model.forward_prefill_chunked(&x, cache_session_start)?
                } else if let Some(lg) = state.model.try_restore_prefix(&prompt_tokens)? {
                    // Exact-prompt hit on a recurrent hybrid: the whole prefill is
                    // skipped. Variants that reuse via trim_kv never reach this  - 
                    // their `try_restore_prefix` is the default miss.
                    lg
                } else {
                    // RESUMING IS AN OPTIMISATION; SERVING THE REQUEST IS NOT.
                    //
                    // If the resident KV turns out to be shorter than the cache said, the
                    // mask is built for rows that are not there and the forward fails on
                    // the shape. That is a 500 for a request the machine can perfectly
                    // well answer - so drop the cache, prefill the whole prompt, and
                    // carry on. The cost is the prefill that was being skipped.
                    let lg = match state.model.forward_prefill_chunked(&x, cache_session_start) {
                        Ok(lg) => lg,
                        Err(e) if cache_session_start > 0 && is_resume_length_mismatch(&e) => {
                            warn!(
                                "prompt cache: resuming at {cache_session_start} did not fit the \
                                 resident KV ({e}) - prefilling the whole prompt instead"
                            );
                            {
                                let mut g = sessions.blocking_lock();
                                g.remove(GLOBAL_PROMPT_CACHE_KEY);
                            }
                            state.model.trim_kv(0);
                            let full = Tensor::new(&prompt_tokens[..], device)?.unsqueeze(0)?;
                            state.model.forward_prefill_chunked(&full, 0)?
                        }
                        Err(e) => return Err(e.into()),
                    };
                    state.model.snapshot_prefix(&prompt_tokens, &lg)?;
                    lg
                };
                (vpl, lg, None)
            };
            let prompt_eval_duration = prefill_start.elapsed().as_nanos() as u64;
            // Decode position must account for the vision prefix written into
            // KV alongside the text tokens (moondream: +image_patches), or the
            // qwen35 mRoPE next-position (image compresses positions).
            let mut pos = qwen35_pos.unwrap_or_else(|| prompt_tokens.len() + vision_prefix_len.unwrap_or(0));
            // Explicit GPU->CPU transfer for logits before sampling
            logits = match logits.device() {
                Device::Cpu => logits,
                _ => logits.to_device(&Device::Cpu)?,
            };
            // Diagnostic: fingerprint prefill logits + KV cache state.
            // Goal: localize the cyclic-3 non-determinism to prefill vs. decode.
            // If iter 1's prefill-logits fingerprint matches iter 2's, prefill
            // is deterministic and the bug is in decode. If they differ, the
            // bug is in prefill (kernel non-determinism, mempool reuse, etc).
            //
            // Hash ALL logits + the argmax index - the first 8 weren't enough
            // because the argmax can be anywhere in the vocab.
            {
                let all: Vec<f32> = logits.squeeze(0).ok()
                    .and_then(|t| t.to_vec1::<f32>().ok())
                    .unwrap_or_default();
                let mut h: u64 = 0xcbf29ce484222325; // FNV-1a 64
                for &v in &all {
                    h ^= v.to_bits() as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                let argmax_idx = all.iter().enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                        if v > bv { (i, v) } else { (bi, bv) }
                    }).0;
                let argmax_val = all.get(argmax_idx).copied().unwrap_or(0.0);
                info!(
                    "🧬 diag: prefill_logits vocab={} fnv1a={:016x} argmax_idx={} argmax_bits={:08x}",
                    all.len(), h, argmax_idx, argmax_val.to_bits()
                );
                // A non-finite logit is a defect upstream, never a property of the prompt - and
                // sampling turns it into an answer. Every position then picks index 0 and the
                // reply is a wall of one token, which reads as a poor model rather than a
                // broken compute path: it took a second machine to notice. Refusing costs one
                // request and names the fault; continuing costs every measurement after it.
                if !all.is_empty() && !argmax_val.is_finite() {
                    error!(
                        "prefill produced non-finite logits (argmax bits {:08x}) - refusing to \
                         sample from them. Neither the model nor the prompt is at fault; this \
                         is the compute path on this device.",
                        argmax_val.to_bits()
                    );
                    // Arm the per-layer scan: this request is already lost, and the next one
                    // will say WHERE the state went bad instead of only that it did.
                    crate::inference::generic_transformer::hetero::SCAN_FOR_NON_FINITE
                        .store(true, std::sync::atomic::Ordering::Relaxed);
                    return Err(anyhow::anyhow!("prefill produced non-finite logits on this device"));
                }
            }
            let logits = apply_repeat_penalty(&logits.squeeze(0)?, &recent_tokens, repeat_penalty, repeat_last_n)?;
            let mut next_token = logits_processor.sample(&logits)?;

            let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);

            // Prompt-lookup / n-gram speculative decoding. The draft cache is
            // seeded with the full prompt + the first sampled token so even
            // the very first decode step can propose drafts that repeat
            // earlier content (common for code: identifiers, closing
            // brackets, repeated API calls).
            //
            // Enabled only when the model is GenericHetero + CUDA and the
            // model's forward supports multi-token inputs. Not supported for
            // Mistral3/Moondream variants yet.
            use crate::inference::serve::prompt_lookup::NgramDraftCache;
            // PLD is enabled for any model variant that supports both
            // forward_all (multi-position logits) and trim_kv. Currently
            // GenericHetero (qwen3, deepcoder, deepseek-r1, etc.) and
            // MultiDeviceMistral3 (devstral). The `device.is_cuda()` check
            // is intentionally NOT applied - multi-device variants set the
            // engine-level `device` to CPU but route most layers to GPU
            // internally; speculative decoding still pays off there.
            let pld_enabled = state.model.supports_pld();
            // Cascading n-gram lookup: try 4-gram first (precise matches
            // for code-style repetition), fall back to 3-gram and 2-gram
            // for shorter patterns. Drafts up to 5 tokens per match.
            // tested max_draft=7. gemma4:latest short
            // +54.5 % (vs +44 % baseline, +10 pp), but medium -9.4 % and
            // long +0.3 % (-15 pp from +15.0 %). Other models also drift.
            // max_draft=5 is the best perimeter-wide setting.
            // max_ngram=5 tested, net-neutral (gemma4 medium
            // drift -1 pp, short/long within noise). Reverted to 4.
            let mut pld_cache = NgramDraftCache::new_cascade(
                /* min_ngram */ 2,
                /* max_ngram */ 4,
                /* max_draft */ 5,
                /* buf       */ 4096,
            );
            if pld_enabled {
                // Seed with the prompt. The initial `next_token` (from
                // prefill sampling) is NOT pushed here - the outer loop
                // pushes it at iteration 0 start (same as `generated` /
                // `recent_tokens`) to keep pld_cache in sync.
                pld_cache.push_many(&prompt_tokens);
            }
            // Spec-decode: lazily load the drafter (block_on - we're on a
            // spawn_blocking thread, so this drives the async !Send load without
            // requiring Send), then prefill it with the prompt so its KV is in lockstep
            // with the target. `spec_on` ⟺ drafter loaded AND prefill succeeded.
            let spec_drafter_on = tokio::runtime::Handle::current()
                .block_on(spec_engine.ensure_draft_loaded());
            let mut spec_on = spec_drafter_on && spec_engine.draft_prefill(&prompt_tokens);
            if spec_on { info!("🜂 spec-decode active: drafter proposing for '{}'", state.name); }
            let (mut spec_drafted, mut spec_accepted): (u64, u64) = (0, 0);
            let mut pld_stats_drafted: u64 = 0;
            let mut pld_stats_accepted: u64 = 0;
            // PLD adaptive cooldown - mirrors the forward_padded path
            // (line 3802). When acceptance in the recent window drops
            // below break-even (~35%), skip PLD for next N iters so the
            // multi-position-forward draft overhead doesn't dominate.
            // Helps long-context regressions where the prompt's n-grams
            // stop predicting the model's diverging completion.
            let mut pld_cooldown: u32 = 0;
            let mut pld_window_drafted: u32 = 0;
            let mut pld_window_accepted: u32 = 0;

            // Stop sequence suffix buffer: only decode last few tokens instead of all (O(1) vs O(n))
            let mut stop_tracker = StopTracker::new(&params.stop_sequences);

            // Auto-tune: create calibrator if model supports speculative decoding
            #[cfg(feature = "opencl")]
            let mut calibrator = state.model.create_calibrator();
            #[cfg(not(feature = "opencl"))]
            let mut calibrator: Option<crate::inference::serve::speculative_config::SpeculativeCalibrator> = None;

            #[cfg(feature = "opencl")]
            if calibrator.is_some() {
                tracing::info!("🔬 Speculative auto-tune: calibrating ({} baseline + {} spec tokens)...", 4, 4);
            }

            // Spec decode state: None = not decided yet or disabled, Some((k, monitor)) = active
            #[cfg(feature = "opencl")]
            let mut spec_active: Option<(usize, crate::inference::serve::speculative_config::SpeculativeMonitor)> = None;
            #[cfg(feature = "opencl")]
            let mut total_drafted = 0usize;
            #[cfg(feature = "opencl")]
            let mut total_accepted = 0usize;
            // Pre-allocate draft buffers to avoid per-iteration allocation
            #[cfg(feature = "opencl")]
            let mut draft_tokens: Vec<u32> = Vec::with_capacity(8);
            #[cfg(feature = "opencl")]
            let mut draft_hidden_states: Vec<(Tensor, usize)> = Vec::with_capacity(8);

            // CUDA graph: capture forward pass after warmup, replay for subsequent tokens
            #[cfg(feature = "cuda")]
            let mut cuda_graph: Option<crate::tensor::cuda_ext::CudaGraph> = None;
            // Reference to the output logits tensor from graph capture (same addresses on replay)
            let mut graph_logits: Option<Tensor> = None;
            // Sticky flag: once graph mode trips a fallback (e.g. seq exceeds
            // the captured kv_cache cap), don't keep calling
            // update_graph_state - the layers stay invalidated for the rest
            // of the request.
            let mut graph_disabled_this_request: bool = false;
            // PLD-resilient graph-mode warmup/capture triggers.
            // Set after warmup forward succeeds; checked in capture branch.
            let mut graph_warmed: bool = false;
            let mut graph_capture_attempted: bool = false;
            // Probation counter (gptoss-style): the captured graph stays on
            // probation - each of the first replays re-validated against an
            // eager forward - until GRAPH_VALIDATIONS consecutive matches.
            #[cfg(feature = "cuda")]
            let mut graph_validations: usize = 0;
            // Capture MUST happen on the MODEL's compute stream - the one its
            // kernels actually enqueue on. On the native substrate every
            // `Device::new_cuda` owns a fresh stream, so `state.device`'s
            // stream is NOT the model's (capturing it recorded 0 nodes while
            // the forward ran eagerly on the model stream - the fleet-wide
            // dead-graph root cause). Fall back to the engine
            // device's stream for backends that don't expose theirs.
            #[cfg(feature = "cuda")]
            let cuda_stream = if device.is_cuda() {
                Some(state.model.model_cuda_stream()
                    .or_else(|| crate::tensor::cuda_ext::stream_of(device).ok()))
            } else { None };

            // Graph-mode user-facing toggle removed: auto-detect uses
            // model arch + KV-cache safety as the gate. No manual override.
            // Graph-mode gates depend only on immutable model config (arch,
            // n_layers, kv cache types). Hoist out of the per-token loop  - 
            // kv_state_graph_safe iterates all layers.
            let (graph_auto_on_cached, kv_safe_cached) = if state.model.supports_graph_mode() {
                let auto = state.model.graph_capture_auto_on();
                let kv_safe = state.model.kv_state_graph_safe();
                let split = state.model.needs_split_graph_path();
                // Say the verdict AND its reason, at a level that is actually on. Running
                // uncaptured costs ~717 kernel launches per token and leaves the GPU busy
                // 7.5% of the decode window, and until now the only trace of it was a
                // debug-level `auto_on=false` with no cause - so a model could sit in the
                // capture allow-list, be refused for its KV dtype, and nothing would say so.
                state.model.log_graph_capture_decision();
                tracing::debug!("graph dispatch: auto_on={} kv_safe={} needs_split={} cuda={}",
                    auto, kv_safe, split, device.is_cuda());
                (auto, kv_safe)
            } else {
                (false, false)
            };
            // use_graph_mode is immutable for a given (model backend, device).
            // Hoist so the per-token loop doesn't redo the capability check
            // + `device.is_cuda()` on every iteration.
            let use_graph_mode_cached = state.model.supports_graph_mode() && device.is_cuda();

            // --- Unified Generation Loop ------------------------------
            let gen_start = std::time::Instant::now();
            let mut decode_diag_first_call = true;
            while next_token != state.eos_token_id
                && !state.eos_token_ids_extra.contains(&next_token)
                && generated.len() < max_tokens {
                generated.push(next_token);
                recent_tokens.push(next_token);
                if pld_enabled { pld_cache.push(next_token); }
                // diag: fingerprint state at the start of the first
                // decode iteration. If next_token + pos are identical across
                // iters but the FORWARD output differs, the bug is in the
                // forward kernels themselves on the FIRST decode shape.
                if decode_diag_first_call {
                    decode_diag_first_call = false;
                    // A token id is content; the position is not.
                    info!("diag: entering first decode iter at pos {pos}");
                }

                if stop_tracker.is_active() {
                    let decoded = state.tokenizer.decode(&[next_token], false).unwrap_or_default();
                    if stop_tracker.push_and_check(&decoded, &params.stop_sequences) {
                        break;
                    }
                }

                if generated.len() >= max_tokens { break; }

                // Determine mode for this token
                #[cfg(feature = "opencl")]
                let use_spec_this_iter = {
                    use crate::inference::serve::speculative_config::CalibrationType;
                    if let Some(ref cal) = calibrator {
                        // During calibration: baseline phase -> normal, spec phase -> spec
                        matches!(cal.phase(), CalibrationType::Speculative)
                    } else if let Some((_, ref monitor)) = spec_active {
                        !monitor.is_disabled()
                    } else {
                        false
                    }
                };
                #[cfg(not(feature = "opencl"))]
                let use_spec_this_iter = false;

                if use_spec_this_iter {
                    #[cfg(feature = "opencl")]
                    {
                        let spec_k = if let Some((k, _)) = &spec_active { *k } else { 2 };
                        let t_spec = Instant::now();

                        // Draft phase (reuse pre-allocated vectors)
                        let t_draft = Instant::now();
                        draft_tokens.clear();
                        draft_hidden_states.clear();
                        let mut draft_token = next_token;
                        let cuda_kv_pos_before = pos;

                        for k in 0..spec_k {
                            let x = Tensor::new(&[draft_token], device)?.unsqueeze(0)?;
                            let (draft_logits, cuda_hidden) = state.model.forward_draft(&x, pos + k)?;
                            let draft_logits = match draft_logits.device() {
                                Device::Cpu => draft_logits,
                                _ => draft_logits.to_device(&Device::Cpu)?,
                            };
                            let draft_logits = apply_repeat_penalty(&draft_logits.squeeze(0)?, &recent_tokens, repeat_penalty, repeat_last_n)?;
                            draft_token = logits_processor.sample(&draft_logits)?;
                            draft_hidden_states.push((cuda_hidden, pos + k));
                            draft_tokens.push(draft_token);
                            recent_tokens.push(draft_token);
                        }
                        let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;
                        total_drafted += spec_k;

                        // Verify phase
                        let t_verify = Instant::now();
                        let verify_logits_batch = state.model.forward_verify_batch(&draft_hidden_states, cuda_kv_pos_before)?;
                        let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;

                        // Accept/reject
                        let mut accepted = 0usize;
                        for k in 0..spec_k {
                            let vl = &verify_logits_batch[k];
                            let vl = match vl.device() {
                                Device::Cpu => vl.clone(),
                                _ => vl.to_device(&Device::Cpu)?,
                            };
                            let penalty_tokens = &recent_tokens[..recent_tokens.len() - (spec_k - k)];
                            let vl = apply_repeat_penalty(&vl.squeeze(0)?, penalty_tokens, repeat_penalty, repeat_last_n)?;
                            let verify_token = logits_processor.sample(&vl)?;

                            if verify_token == draft_tokens[k] {
                                accepted += 1;
                                generated.push(draft_tokens[k]);
                            } else {
                                generated.push(verify_token);
                                for _ in 0..(spec_k - k) { recent_tokens.pop(); }
                                recent_tokens.push(verify_token);
                                break;
                            }
                        }
                        total_accepted += accepted;

                        let new_pos = cuda_kv_pos_before + accepted + 1;
                        pos = new_pos;
                        if accepted < spec_k {
                            state.model.trim_cuda_kv(new_pos);
                            state.model.trim_opencl_kv(new_pos);
                        }

                        // Full forward for next token
                        let last_accepted = *generated.last().unwrap();
                        let x = Tensor::new(&[last_accepted], device)?.unsqueeze(0)?;
                        let mut logits = state.model.forward(&x, pos)?;
                        pos += 1;
                        logits = match logits.device() {
                            Device::Cpu => logits,
                            _ => logits.to_device(&Device::Cpu)?,
                        };
                        let logits = apply_repeat_penalty(&logits.squeeze(0)?, &recent_tokens, repeat_penalty, repeat_last_n)?;
                        next_token = logits_processor.sample(&logits)?;

                        let spec_ms = t_spec.elapsed().as_secs_f64() * 1000.0;
                        let tokens_produced = accepted + 1; // accepted drafts + 1 verify

                        // Feed calibrator during calibration phase
                        if let Some(ref mut cal) = calibrator {
                            cal.record_draft(draft_ms / spec_k as f64);
                            cal.record_verify(verify_ms / spec_k as f64);
                            cal.record_transfer(2.0); // approximate per-token transfer
                            cal.record_acceptance(spec_k, accepted);
                        }

                        // Feed runtime monitor after calibration
                        if let Some((_, ref mut monitor)) = spec_active {
                            if !monitor.record_cycle(spec_ms, tokens_produced) {
                                // Short-circuited! Continue with normal forward from here
                            }
                        }

                        // Finalize calibration if spec phase is done
                        if let Some(ref mut cal) = calibrator {
                            use crate::inference::serve::speculative_config::CalibrationType;
                            if cal.phase() == CalibrationType::Done && spec_active.is_none() {
                                // Extract baseline_ms before finalize borrows mutably
                                let baseline_ms = {
                                    let bts = cal.baseline_timings_ref();
                                    let mut sorted = bts.to_vec();
                                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                                    sorted[sorted.len() / 2]
                                };
                                let decision = cal.finalize();
                                if decision.enabled {
                                    spec_active = Some((decision.k, crate::inference::serve::speculative_config::SpeculativeMonitor::new(baseline_ms, decision.k)));
                                    tracing::info!("🎯 Auto-tune: speculative decoding ENABLED (K={}, predicted {:.2}x speedup)", decision.k, decision.predicted_speedup);
                                } else {
                                    tracing::info!("🎯 Auto-tune: speculative decoding DISABLED ({})", decision.reason);
                                }
                                calibrator = None;
                            }
                        }
                    }
                } else {
                    // --- CUDA forward with graph capture/replay -----
                    // Token 3:  warmup - prime graph buffers (allocates stable pointers)
                    // Token 4:  capture - begin_capture, forward, end_capture
                    // Token 5+: replay  - update state, graph.launch()
                    let token_idx = generated.len();
                    let use_graph_mode = use_graph_mode_cached;

                    // Graph-mode activation: auto-on whenever the loaded model
                    // is on a known-safe arch + ctx - currently gemma4 26B-MoE
                    // (arch=gemma4, 30 layers) - AND the KV state is
                    // graph-safe. KV safety means F-dtype kv_cache holds the
                    // prefill history (Q4/Q8 caches use CPU-side
                    // current_seq_len for offsets, which captures wrong).
                    // `graph_capture_auto_on()` already folds `kv_state_graph_safe()`
                    // in for the GenericHetero backends, but AND the explicit
                    // gate anyway: a backend whose auto-on doesn't imply KV
                    // safety must never reach capture (capturing over an
                    // inconsistent KV records an empty/garbage graph that
                    // REPLAYS silently - constant-token output).
                    let graph_enabled = graph_auto_on_cached && kv_safe_cached;
                    // Option-E split path (commit 787986a): F-dtype-only
                    // models route through prepare_all_kv (uncaptured) +
                    // compute_all_from_kv (captured) to avoid the
                    // scatter_set ILLEGAL_ADDRESS bug at first replay.
                    // Q4/Q8 arches keep the legacy single-pass path.
                    let needs_split = state.model.needs_split_graph_path();
                    let mut split_token_ids: Option<Tensor> = None;
                    if graph_enabled && use_graph_mode && !graph_disabled_this_request {
                        let x = Tensor::new(&[next_token], &Device::Cpu)?.unsqueeze(0)?;
                        // If the F-dtype kv_cache outgrew the captured graph
                        // state buffers (e.g. crossing the 512-cap doubling
                        // boundary mid-decode), update_graph_state errors so
                        // we can invalidate the graph and fall through to the
                        // non-graph forward path for the rest of the request.
                        match state.model.update_graph_state(pos) {
                            Ok(()) => {
                                if needs_split {
                                    // Split path: prepare_all_kv (called below
                                    // before capture/replay) writes the embed,
                                    // QKV, RoPE, KV append. Stash input_ids for
                                    // those call sites; skip embed_for_graph.
                                    split_token_ids = Some(x);
                                } else {
                                    state.model.embed_for_graph(&x)?;
                                }
                            }
                            Err(e) => {
                                warn!("update_graph_state failed at pos={pos}: {e}. Disabling graph for rest of request.");
                                #[cfg(feature = "cuda")]
                                { cuda_graph = None; }
                                graph_logits = None;
                                graph_disabled_this_request = true;
                                // Drop per-layer graph buffers so the next
                                // forward takes the non-graph path
                                // (`forward_attn` keys on
                                // `graph_rope_cos.is_some()`).
                                // Order: invalidate (drops arena-backed
                                // transients as no-ops) BEFORE freeing the
                                // capture arena.
                                state.model.invalidate_graph_state();
                                #[cfg(feature = "cuda")]
                                if let Some(Some(ref stream)) = cuda_stream {
                                    stream.context().free_capture_arena();
                                }
                            }
                        }
                    }

                    // Graph REPLAY (tokens 5+).
                    // Split path: prepare_all_kv must run uncaptured BEFORE
                    // borrowing cuda_graph for the replay launch. Do it
                    // here so failure can disable graph cleanly.
                    #[cfg(feature = "cuda")]
                    if cuda_graph.is_some() {
                        if let Some(ref tok_ids) = split_token_ids {
                            if let Err(e) = state.model.prepare_all_kv(tok_ids, pos) {
                                warn!("prepare_all_kv failed at pos={pos}: {e}. Disabling graph for rest of request.");
                                cuda_graph = None;
                                graph_logits = None;
                                graph_disabled_this_request = true;
                                // Transients first, then the arena (see
                                // update_graph_state failure path above).
                                state.model.invalidate_graph_state();
                                if let Some(Some(ref stream)) = cuda_stream {
                                    stream.context().free_capture_arena();
                                }
                            }
                        }
                    }
                    // per-token re-capture path for archs whose
                    // captured graphs hold position-dependent pointers
                    // (phi2's KV scatter/logits store/sampling-dst go
                    // stale on replay -> ILLEGAL_ADDRESS without this).
                    // Auto-on for phi2 only - other archs use the cheaper
                    // capture-once-replay-many path.
                    let recapture = state.model.recapture_each_token();
                    // Recapture path: each replay re-records forward into the
                    // graph (capturing CURRENT pointers), then ExecUpdate
                    // patches the exec graph, then launch executes with the
                    // new pointers. The recapture forward's OUTPUT tensor
                    // becomes the new logits_ref - the prior warmup-time
                    // logits buffer is stale because the new forward's
                    // sub-tensors are different fresh-alloc pointers.
                    //
                    // Per llama.cpp ggml-cuda.cu:4400-4430 pattern. Capture
                    // mode RECORDS kernel submissions; execution waits
                    // until graph.launch() below.
                    #[cfg(feature = "cuda")]
                    if recapture {
                        if let (Some(graph), Some(Some(stream))) =
                            (cuda_graph.as_mut(), cuda_stream.as_ref())
                        {
                            let _ = stream.synchronize();
                            // populate disabled with gate revert. Code stays
                            // accessible by un-commenting when gemma4 graph
                            // mode is unblocked.
                            if let Err(e) = crate::tensor::cuda_ext::begin_capture(stream) {
                                warn!("recapture begin_capture failed: {e:?}; falling back");
                            } else {
                                let new_logits_res: Result<crate::tensor::Tensor, crate::tensor::Error> =
                                    if let Some(ref _tok_ids) = split_token_ids {
                                        state.model.compute_all_from_kv_captured()
                                    } else {
                                        state.model.forward_from_hidden()
                                    };
                                match new_logits_res {
                                    Ok(new_logits) => {
                                        // Try cheap in-place update first; if
                                        // the per-token graph differs
                                        // structurally (phi2: cuBLAS-selected
                                        // matmul function-handle drift),
                                        // fall back to a full re-instantiate
                                        // automatically. llama.cpp's
                                        // ggml_cuda_graph_update_executable
                                        // pattern. Re-instantiate is ~50-200
                                        // µs for a 2-3K-node graph - still
                                        // a win versus paying per-launch
                                        // overhead on the non-graph path
                                        // (~3 ms/token saved on phi2).
                                        use crate::tensor::cuda_ext::UpdateOutcome;
                                        match graph.end_capture_or_reinstantiate() {
                                            Ok(outcome) => {
                                                if matches!(outcome, UpdateOutcome::Reinstantiated) {
                                                    tracing::debug!("graph re-instantiated at pos={pos} (structural diff handled)");
                                                }
                                                // DEFENSE IN DEPTH: a re-record that
                                                // produced 0 nodes would replay as a
                                                // no-op -> frozen logits -> constant
                                                // tokens. Discard and go eager.
                                                if graph.num_nodes().unwrap_or(0) == 0 {
                                                    warn!("re-captured CUDA graph has 0 nodes at pos={pos}; discarding and falling back to eager decode");
                                                    cuda_graph = None;
                                                    graph_logits = None;
                                                    graph_disabled_this_request = true;
                                                    state.model.invalidate_graph_state();
                                                } else {
                                                    graph_logits = Some(new_logits);
                                                }
                                            }
                                            Err(e) => {
                                                warn!("end_capture_or_reinstantiate failed: {e:?}; clearing graph");
                                                cuda_graph = None;
                                                graph_logits = None;
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        // forward failed mid-capture; clean
                                        // up via end_capture and reset.
                                        let _ = crate::tensor::cuda_ext::end_capture(stream);
                                        cuda_graph = None;
                                        graph_logits = None;
                                    }
                                }
                            }
                        }
                    }
                    #[cfg(feature = "cuda")]
                    let graph_pair = match (cuda_graph.as_mut(), graph_logits.as_ref()) {
                        (Some(g), Some(l)) => Some((g, l)),
                        _ => None,
                    };
                    #[cfg(feature = "cuda")]
                    if let Some((graph, logits_ref)) = graph_pair {
                        // Sync to ensure all state updates (embed_for_graph,
                        // update_graph_state) are visible to the graph launch.
                        if let Some(Some(ref stream)) = cuda_stream {
                            let _ = stream.synchronize();
                        }
                        match graph.launch() {
                            Ok(()) => {
                                // Sync AFTER launch so subsequent gpu_sample
                                // (which reads logits_ref) waits for the
                                // graph to finish writing them. The graph
                                // captured ops use cudaMallocAsync internally;
                                // without sync, the next kernel's read may
                                // race with graph ops or hit freed memory.
                                if let Some(Some(ref stream)) = cuda_stream {
                                    let _ = stream.synchronize();
                                }
                                // Probation self-validation (gptoss-style):
                                // each of the first GRAPH_VALIDATIONS replays
                                // must match an eager forward at the same
                                // (token, pos). Snapshot the replay argmax
                                // FIRST (the eager re-run writes the same
                                // stable logits buffer), then re-run eagerly
                                // (KV write at pos is idempotent) and compare.
                                // On mismatch: sample the CORRECT eager
                                // logits, discard the graph, decode the rest
                                // of the request eagerly - a resurrected
                                // graph path can never emit replay garbage.
                                const GRAPH_VALIDATIONS: usize = 3;
                                if graph_validations < GRAPH_VALIDATIONS {
                                    let rep_argmax: Option<u32> = logits_ref
                                        .flatten_all().ok()
                                        .and_then(|t| t.argmax(0).ok())
                                        .and_then(|t| t.to_scalar::<u32>().ok());
                                    let eager = if needs_split {
                                        state.model.compute_all_from_kv_captured()
                                    } else {
                                        state.model.forward_from_hidden()
                                    };
                                    match eager {
                                        Ok(eag) => {
                                            if let Some(Some(ref stream)) = cuda_stream {
                                                let _ = stream.synchronize();
                                            }
                                            let eag_argmax = eag.flatten_all()?
                                                .argmax(0)?.to_scalar::<u32>()?;
                                            if rep_argmax == Some(eag_argmax) {
                                                graph_validations += 1;
                                            } else {
                                                warn!("CUDA graph replay!=eager at pos={pos} (argmax {:?} vs {}); discarding graph, eager decode for rest of request",
                                                    rep_argmax, eag_argmax);
                                                cuda_graph = None;
                                                graph_logits = None;
                                                graph_disabled_this_request = true;
                                                // Drop arena-backed transients
                                                // BEFORE freeing the arena
                                                // (arena-range drops are no-ops
                                                // only while it is alive).
                                                state.model.invalidate_graph_state();
                                                if let Some(Some(ref stream)) = cuda_stream {
                                                    stream.context().free_capture_arena();
                                                }
                                                pos += 1;
                                                let logits_1d = eag.squeeze(0)?;
                                                next_token = gpu_sample(
                                                    &logits_1d, &recent_tokens, repeat_penalty, repeat_last_n,
                                                    temperature, top_k, &mut logits_processor,
                                                )?;
                                                continue;
                                            }
                                        }
                                        Err(e) => {
                                            warn!("graph probation eager forward failed at pos={pos}: {e}; trusting replay");
                                            graph_validations += 1;
                                        }
                                    }
                                }
                                pos += 1;
                                // Host KV bookkeeping: the replay advanced the
                                // quantized caches device-side only; mirror the
                                // post-token length into the host counters so
                                // the NEXT request's trim/prefix-reuse sees the
                                // true occupancy.
                                state.model.sync_kv_len_for_graph(pos);
                                let logits_1d = logits_ref.squeeze(0)?;
                                next_token = gpu_sample(
                                    &logits_1d, &recent_tokens, repeat_penalty, repeat_last_n,
                                    temperature, top_k, &mut logits_processor,
                                )?;
                                continue;
                            }
                            Err(e) => {
                                warn!("Graph replay failed: {e}, falling back to normal forward");
                                cuda_graph = None;
                                graph_logits = None;
                                graph_disabled_this_request = true;
                                // Same teardown order as the mismatch path:
                                // transients first, then the arena.
                                state.model.invalidate_graph_state();
                                if let Some(Some(ref stream)) = cuda_stream {
                                    stream.context().free_capture_arena();
                                }
                            }
                        }
                    }

                    // Graph-mode warmup: prime stable buffers.
                    // changed `== 3` to `>= 3 && !graph_warmed`.
                    // lowered threshold 3 -> 1 after sweep:
                    // medium gap drops -3.3 % -> -2.7 % (t=2) -> -2.3 % (t=1)
                    // -> -2.7 % (t=0). t=1 is the sweet spot - earlier
                    // captures (t=0) miss some KV-state stabilization on
                    // first decode iter.
                    #[cfg(feature = "cuda")]
                    if graph_enabled && cuda_graph.is_none() && token_idx >= 1 && !graph_warmed && use_graph_mode {
                        let logits = if let Some(ref tok_ids) = split_token_ids {
                            // Split path: prepare_all_kv + compute_all_from_kv
                            // both run uncaptured at warmup. Primes Q+KV buffers
                            // AND attention/FFN paths for first replay.
                            state.model.forward_from_hidden_split(tok_ids, pos)?
                        } else {
                            state.model.forward_from_hidden()?
                        };

                        //  isolation probe REMOVED - it
                        // proved arena==non-arena (correct); residual bug is in
                        // split-path layer compute, isolated via per-layer
                        // numerical diff in compute_all_from_kv vs
                        // forward_layers_and_output. The probe's extra
                        // compute_all_from_kv_captured runs polluted the
                        // per-layer once-counters, so it's gone.

                        pos += 1;
                        graph_warmed = true;
                        let logits_1d = logits.squeeze(0)?;
                        next_token = gpu_sample(
                            &logits_1d, &recent_tokens, repeat_penalty, repeat_last_n,
                            temperature, top_k, &mut logits_processor,
                        )?;
                        continue;
                    }

                    // Graph capture (once per request, on the iteration after
                    // warmup).
                    //
                    // Restored (was silently dead fleet-wide
                    // since the substrate flip): the capture runs on the
                    // MODEL's stream (see `cuda_stream` above) with the
                    // gptoss/lfm2 treatment - per-slice event tracking off at
                    // load (hetero_cuda_devs creation), cuBLAS + mmvq
                    // workspaces pinned (update_graph_state one-time block),
                    // transients routed through an adaptively-sized capture
                    // arena (double on overflow -> 0 MEM_ALLOC nodes), the
                    // capture token launched once (a capture only RECORDS  - 
                    // the logits buffer is not written until the graph runs),
                    // and the first replays validated on probation against an
                    // eager forward. Rationale + history in
                    // generic_transformer/graph.rs::graph_capture_auto_on.
                    // was `== 4`; now `graph_warmed && !attempted`
                    // so it fires on the next iter after warmup regardless of
                    // exact token_idx (see warmup comment above for PLD context).
                    #[cfg(feature = "cuda")]
                    let should_capture = graph_enabled && cuda_graph.is_none() && graph_warmed && !graph_capture_attempted && use_graph_mode;
                    #[cfg(feature = "cuda")]
                    if should_capture {
                        graph_capture_attempted = true;
                        // Fresh gate re-check at the moment of capture:
                        // `kv_safe_cached` was hoisted at request start, but KV
                        // safety is DYNAMIC - a mid-request KV drop (e.g. a
                        // failed Q8 append past the cache cap) flips
                        // `kv_state_graph_safe()` to false. Capturing over that
                        // state records an empty/incoherent graph whose replay
                        // is silent garbage. Capture happens once per request,
                        // so the per-layer scan is off the hot path.
                        if !state.model.kv_state_graph_safe() {
                            warn!("KV state no longer graph-safe at pos={pos}; skipping graph capture (eager decode for rest of request)");
                            graph_disabled_this_request = true;
                            state.model.invalidate_graph_state();
                        } else if let Some(Some(ref stream)) = cuda_stream {
                            stream.synchronize()
                                .map_err(|e| crate::tensor::Error::msg(format!("sync before capture: {e}")))?;

                            // Split path: prepare_all_kv must run OUTSIDE the
                            // capture region so its scatter_set (with fresh K
                            // pointer) doesn't get baked into the graph.
                            if let Some(ref tok_ids) = split_token_ids {
                                state.model.prepare_all_kv(tok_ids, pos)?;
                            }

                            // Capture arena + record + 0-node defense +
                            // histogram + upload - shared with the streaming
                            // decode loop (see `capture_decode_graph`).
                            let ctx = stream.context();
                            match capture_decode_graph(&mut state.model, stream, pos, needs_split) {
                                Some((g, logits)) => {
                                    // A capture only RECORDS - the logits buffer
                                    // was never written for THIS token (sampling
                                    // it now would re-emit the warmup token).
                                    // Launch the recorded graph once so the
                                    // capture token's forward actually executes,
                                    // then run probation validation #1 against an
                                    // eager forward at the same (token, pos).
                                    match g.launch() {
                                        Ok(()) => {
                                            let _ = stream.synchronize();
                                            let rep_argmax: Option<u32> = logits
                                                .flatten_all().ok()
                                                .and_then(|t| t.argmax(0).ok())
                                                .and_then(|t| t.to_scalar::<u32>().ok());
                                            let eager = if needs_split {
                                                state.model.compute_all_from_kv_captured()
                                            } else {
                                                state.model.forward_from_hidden()
                                            };
                                            match eager {
                                                Ok(eag) => {
                                                    let _ = stream.synchronize();
                                                    let eag_argmax = eag.flatten_all()?
                                                        .argmax(0)?.to_scalar::<u32>()?;
                                                    if rep_argmax == Some(eag_argmax) {
                                                        graph_validations += 1;
                                                        pos += 1;
                                                        // Host KV sync + freeze quantized-KV
                                                        // growth/seq-ceiling for the lifetime
                                                        // of the captured graph. (The capture
                                                        // token's probation eager forward
                                                        // appended one extra host-side slot;
                                                        // the sync corrects it.)
                                                        state.model.sync_kv_len_for_graph(pos);
                                                        state.model.mark_graph_captured();
                                                        let logits_1d = logits.squeeze(0)?;
                                                        next_token = gpu_sample(
                                                            &logits_1d, &recent_tokens, repeat_penalty, repeat_last_n,
                                                            temperature, top_k, &mut logits_processor,
                                                        )?;
                                                        graph_logits = Some(logits);
                                                        cuda_graph = Some(g);
                                                        continue;
                                                    }
                                                    warn!("CUDA graph capture-launch!=eager at pos={pos} (argmax {:?} vs {}); discarding graph, eager decode for rest of request",
                                                        rep_argmax, eag_argmax);
                                                    drop(g);
                                                    graph_disabled_this_request = true;
                                                    pos += 1;
                                                    let logits_1d = eag.squeeze(0)?;
                                                    next_token = gpu_sample(
                                                        &logits_1d, &recent_tokens, repeat_penalty, repeat_last_n,
                                                        temperature, top_k, &mut logits_processor,
                                                    )?;
                                                    state.model.invalidate_graph_state();
                                                    ctx.free_capture_arena();
                                                    continue;
                                                }
                                                Err(e) => {
                                                    warn!("graph probation eager forward failed at pos={pos}: {e}; trusting captured graph");
                                                    graph_validations += 1;
                                                    pos += 1;
                                                    state.model.sync_kv_len_for_graph(pos);
                                                    state.model.mark_graph_captured();
                                                    let logits_1d = logits.squeeze(0)?;
                                                    next_token = gpu_sample(
                                                        &logits_1d, &recent_tokens, repeat_penalty, repeat_last_n,
                                                        temperature, top_k, &mut logits_processor,
                                                    )?;
                                                    graph_logits = Some(logits);
                                                    cuda_graph = Some(g);
                                                    continue;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            warn!("Captured graph first launch failed at pos={pos}: {e}; eager decode for rest of request");
                                            drop(g);
                                            graph_disabled_this_request = true;
                                            let logits = if needs_split {
                                                state.model.compute_all_from_kv_captured()?
                                            } else {
                                                state.model.forward_from_hidden()?
                                            };
                                            pos += 1;
                                            let logits_1d = logits.squeeze(0)?;
                                            next_token = gpu_sample(
                                                &logits_1d, &recent_tokens, repeat_penalty, repeat_last_n,
                                                temperature, top_k, &mut logits_processor,
                                            )?;
                                            state.model.invalidate_graph_state();
                                            ctx.free_capture_arena();
                                            continue;
                                        }
                                    }
                                }
                                _ => {
                                    // Capture failed - a capture only RECORDS, so
                                    // the forward never executed and the logits
                                    // were never written. Compute this token
                                    // eagerly and decode the rest of the request
                                    // on the plain (non-graph) path.
                                    graph_disabled_this_request = true;
                                    let logits = if needs_split {
                                        state.model.compute_all_from_kv_captured()?
                                    } else {
                                        state.model.forward_from_hidden()?
                                    };
                                    pos += 1;
                                    let logits_1d = logits.squeeze(0)?;
                                    next_token = gpu_sample(
                                        &logits_1d, &recent_tokens, repeat_penalty, repeat_last_n,
                                        temperature, top_k, &mut logits_processor,
                                    )?;
                                    state.model.invalidate_graph_state();
                                    ctx.free_capture_arena();
                                    continue;
                                }
                            }
                        }
                    }

                    // Prompt-lookup / n-gram speculative decoding.
                    //
                    // Try to draft K tokens from the rolling token buffer.
                    // If lookup returns a non-empty draft, verify it in one
                    // batched forward with seq_len = K+1 (current token +
                    // drafts). Sample each output position, accept the
                    // longest prefix that matches the draft, and trim the
                    // KV cache back to the accepted length on partial match.
                    // Per-prompt-length PLD skip for deepseek-r1: this
                    // reasoning-model family shows -3 % decode at SHORT
                    // prompts with PLD on (multi-position verify
                    // overhead exceeds rare n-gram accepts), but
                    // benefits +8 % at medium/long where acceptance
                    // compensates. Tuning notes:
                    //   < 80: matched medium -> regressed it -10 pp
                    //   < 40: short -0.9 % but medium/long both
                    //         regressed ~7 pp (medium tokenizes < 40
                    //         in deepseek-r1's vocab).
                    //   < 25: tightest gate that only hits assay's
                    //         "short" prompt (~20 tokens after tokeniser).
                    let pld_short_skip = pld_enabled
                        && prompt_tokens.len() < 25
                        && state.name.starts_with("deepseek-r1");
                    // tried gemma4:latest first-30-decode-tokens
                    // PLD skip (hypothesis: cold-start window has 0%
                    // acceptance). Bench result: gemma4:latest short +44% ->
                    // +3.2%, medium -3.3% -> -10.3%, long +15% -> -10.1%.
                    // Hypothesis WRONG - PLD acceptance is high from token 1
                    // for gemma4:latest. The "first-tokens-slow" finding has
                    // a different root cause (likely mempool warmup or graph
                    // capture overhead). Not retrying.
                    // PLD long-ctx gate (CPU-only): the multi-token verify bails the
                    // single-token zero-alloc executor -> slow Tensor-path attention whose
                    // cost grows with context. Measured (qwen3 CPU): PLD is neutral at
                    // <=~1500 ctx but CATASTROPHIC at 2.5K (decode 4.7->2.8 tok/s, -40%)  - 
                    // the slow verify-attention dwarfs the n-gram accept benefit, turning a
                    // +18%-vs-ollama win into a -30% loss. Disable PLD past a context
                    // threshold - on BOTH CPU and GPU. The earlier "GPU verify is fast ->
                    // CPU-only" gate was wrong: on GPU too, PLD's long-ctx accept collapse
                    // makes the per-position verify GEMVs a net loss (qwen3/deepcoder 2.5K
                    // loss->tie once gated). (Proper fix = cpu_q8_kv::attention_multi.)
                    let pld_ctx_ok = pos <= pld_max_ctx_threshold();
                    let pld_active = pld_enabled && pld_cooldown == 0 && !pld_short_skip && pld_ctx_ok;
                    pld_cooldown = pld_cooldown.saturating_sub(1);
                    // Spec-decode: when a drafter is active, draft K tokens from
                    // it (its KV is at `pos`, in lockstep with the target). Otherwise
                    // fall back to the n-gram PLD draft. Either way the verify below
                    // re-samples every position -> output is the target's greedy.
                    const SPEC_K: usize = 4;
                    let mut draft: Vec<u32> = if spec_on {
                        let d = spec_engine.draft_k(next_token, SPEC_K, pos, &recent_tokens, repeat_penalty, repeat_last_n);
                        if d.is_empty() { spec_on = false; } // drafter error -> disable spec
                        d
                    } else if pld_active {
                        pld_cache.lookup()
                    } else {
                        Vec::new()
                    };
                    // Per-model draft-length cap. deepseek-r1 has natural low
                    // PLD acceptance at short prompts (post-pld_short_skip
                    // gating). For non-short prompts that still run PLD,
                    // capping K at 2 reduces per-cycle wasted forward work
                    // when verification ultimately rejects. -0.9% TIE target.
                    if state.name.starts_with("deepseek-r1") && draft.len() > 2 {
                        draft.truncate(2);
                    }
                    // deepcoder cap=3 tested, regressed all 3
                    // prompts (medium -4 pp, short -1 pp, long -1 pp). Higher
                    // PLD acceptance benefits more from larger drafts in
                    // deepcoder's repetitive code-completion patterns.
                    // Not retrying.

                    if !draft.is_empty() && generated.len() + 1 + draft.len() <= max_tokens {
                        let k = draft.len();
                        pld_stats_drafted += k as u64;

                        // Build [current_token, draft_0, ..., draft_{k-1}]
                        let mut input = Vec::with_capacity(k + 1);
                        input.push(next_token);
                        input.extend_from_slice(&draft);
                        let x = Tensor::new(input.as_slice(), device)?.unsqueeze(0)?;
                        // Multi-position forward: logits for every input pos
                        // (shape [1, k+1, vocab]). Required for speculative
                        // verification - the standard `forward` returns only
                        // the last position's logits.
                        // Position from the CACHE, not from `pos`. `pos` counts
                        // sampled tokens and lags the cache by one whenever the
                        // token just sampled is carried forward before being
                        // pushed - and by zero right after a verify has already
                        // resynchronised it, so no constant offset works. A
                        // single-position forward never noticed, because
                        // `make_mask(1, pos)` yields `pos + 1` and that happened
                        // to equal the cache; this forward covers k+1 positions
                        // and its mask came out one key short.
                        let verify_pos = state.model.kv_len().unwrap_or(pos);
                        let logits = state.model.forward_all(&x, verify_pos)?;

                        // Commit-delayed verification loop (shared with the
                        // streaming path - see decode_step::pld_verify_commit
                        // for the position/commit semantics).
                        let outcome = pld_verify_commit(
                            &logits, &draft, &mut generated, &mut recent_tokens,
                            &mut pld_cache, pld_enabled,
                            state.eos_token_id, &state.eos_token_ids_extra,
                            /* committed_base */ 0, max_tokens,
                            &mut |row, recent| {
                                sample_row_sync(
                                    row, recent, repeat_penalty, repeat_last_n,
                                    temperature, top_k, &mut logits_processor,
                                ).map(|t| (t, None))
                            },
                        )?;
                        let accepted_drafts = outcome.accepted_drafts;

                        pld_stats_accepted += accepted_drafts as u64;
                        // Window-rate cooldown (shared: decode_step::pld_window_update).
                        pld_window_update(
                            &mut pld_window_drafted, &mut pld_window_accepted,
                            &mut pld_cooldown, k, accepted_drafts,
                        );
                        // Total committed this step = 1 (S_0) + accepted_drafts
                        let committed = 1 + accepted_drafts;
                        pos += committed;

                        // The verification forward wrote k+1 rows into the KV
                        // cache. Only `committed` of them correspond to
                        // accepted input tokens - trim the rest.
                        if committed < k + 1 {
                            // The verify wrote k+1 rows from `verify_pos`; keep
                            // the `committed` that were accepted.
                            state.model.trim_kv(verify_pos + committed);
                        }

                        // Spec-decode: drafter KV lockstep (shared:
                        // decode_step::spec_draft_lockstep).
                        if spec_on {
                            spec_drafted += k as u64;
                            spec_accepted += accepted_drafts as u64;
                            spec_draft_lockstep(&spec_engine, &draft, accepted_drafts, pos);
                        }

                        // Set next_token to the last committed sample - the
                        // outer loop will push it at the start of the next
                        // iteration (completing the commit-delayed pattern).
                        next_token = outcome.next_token;
                    } else {
                        // No draft available (empty lookup) or not enough
                        // headroom in max_tokens - run the regular single-
                        // token forward.
                        let x = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
                        // stepd: state.forward for moondream graph capture
                        // (device cloned earlier so this works).
                        let logits = state.forward(&x, pos)?;
                        pos += 1;

                        let logits_1d = logits.squeeze(0)?;
                        next_token = sample_row_sync(
                            &logits_1d, &recent_tokens, repeat_penalty, repeat_last_n,
                            temperature, top_k, &mut logits_processor,
                        )?;
                        // next_token will be pushed to pld_cache at the start
                        // of the next outer-loop iteration (symmetric with
                        // generated / recent_tokens).
                    }
                }
            }

            // End-of-request graph teardown. The captured graph + its
            // arena-backed transients must not outlive the request: the next
            // request re-captures with a FRESH arena (begin_capture_arena
            // frees any previous one), and stale arena tensors dropped after
            // that free would real-cuMemFree garbage pointers. Order: graph
            // first, then transients/buffers (arena-range drops are no-ops
            // while the arena is alive), then the arena itself - which also
            // returns its VRAM between requests.
            #[cfg(feature = "cuda")]
            if graph_capture_attempted {
                let captured = cuda_graph.is_some();
                drop(cuda_graph.take());
                drop(graph_logits.take());
                if captured {
                    state.model.invalidate_graph_state();
                    if let Some(Some(ref stream)) = cuda_stream {
                        stream.context().free_capture_arena();
                    }
                }
            }

            #[cfg(feature = "opencl")]
            if total_drafted > 0 {
                tracing::info!(
                    "📊 Speculative stats: {}/{} accepted ({:.1}%), {} total tokens",
                    total_accepted, total_drafted,
                    total_accepted as f64 / total_drafted as f64 * 100.0,
                    generated.len()
                );
            }

            // Prompt-lookup speculative decoding stats.
            if pld_stats_drafted > 0 {
                let accept_rate = pld_stats_accepted as f64 / pld_stats_drafted as f64;
                info!("📊 PLD stats: {}/{} drafts accepted ({:.1}%)",
                    pld_stats_accepted, pld_stats_drafted, accept_rate * 100.0);
            }

            let eval_duration = gen_start.elapsed().as_nanos() as u64;
            let eval_count = generated.len() as u64;

            // Mirror the resident KV (prompt + generated) under the GLOBAL key so
            // the next sessionless request can prefix-reuse it (in-memory only, never
            // persisted). Skip when disabled or for vision (image positions aren't in
            // the token stream -> can't be matched by a text request).
            if true
                && vision_prefix_len.is_none() && qwen35_pos.is_none() {
                let mut full = prompt_tokens.clone();
                full.extend_from_slice(&generated);
                let mut g = sessions.blocking_lock();
                g.insert(GLOBAL_PROMPT_CACHE_KEY.to_string(), SessionState {
                    tokens: full,
                    model_name: state.name.clone(),
                    image_hash: None,
                    image_prefix_len: 0,
                    kv_len: pos,
                });
            }

            // Decode output tokens
            // Token ids decode straight back to the text, so they are content too.
            debug!("Generated {} tokens", generated.len());
            let text = state.tokenizer.decode(&generated, true)
                .map_err(|e| anyhow!("Decode error: {}", e))?;

            debug!("Decoded {} chars", text.len());
            {
                let (macs, calls) = crate::tensor::quant_cpu::take_cpu_mac_counters();
                if calls > 0 && eval_count > 0 {
                    let bytes = crate::tensor::quant_cpu::take_cpu_bytes();
                    info!("🧮 cpu work: {:.2} GMAC/token, {:.1} MB/token, {} calls/token",
                        macs as f64 / 1e9 / eval_count as f64,
                        bytes as f64 / 1e6 / eval_count as f64,
                        calls / eval_count.max(1) as u64);
                }
            }
            let decode_tok_s = if eval_duration > 0 {
                eval_count as f64 / (eval_duration as f64 / 1e9)
            } else {
                0.0
            };
            info!("✅ Generated {} tokens ({:.1} tok/s, prefill {:.1}ms)",
                eval_count, decode_tok_s, prompt_eval_duration as f64 / 1e6,
            );
            // Recorded HERE because this is where the figures already exist: a meter fed from
            // a handler would miss whichever path did not go through it, and a node that
            // under-reports its own speed is one the cluster stops sending work to.
            // Prefill is recorded too: on short replies it is the dominating fixed cost,
            // and a router that only knows decode prices every hand-over as free to start.
            let prefill_tok_s = if prompt_eval_duration > 0 {
                prompt_tokens.len() as f64 / (prompt_eval_duration as f64 / 1e9)
            } else {
                0.0
            };
            crate::distributed::rate_meter::record_generation(
                &metered_model, prefill_tok_s, decode_tok_s, eval_count as u64);
            // The acceptance rate is the only number that says whether speculation is
            // paying for itself. This path was counting it and throwing it away, so a
            // drafter that had stopped helping would have looked exactly like one that
            // was - the streaming path has reported it all along.
            if spec_drafted > 0 {
                info!(
                    "Spec decode done: drafted={spec_drafted} accepted={spec_accepted} ({}%)",
                    (spec_accepted * 100) / spec_drafted,
                );
            }

            Ok(GenerationResult {
                text,
                tokens: {
                    let mut full = prompt_tokens.clone();
                    full.extend_from_slice(&generated);
                    full
                },
                prompt_eval_count: prompt_token_count,
                prompt_eval_duration,
                eval_count,
                eval_duration,
            })
        }).await
            .map_err(|e| format!("spawn_blocking error: {e}").into());

        // Handle nested Result types
        match inner_result {
            Ok(Ok(text)) => Ok(text),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e),
        }
    }

    /// Grammar-constrained sequential decode path. Routed to from `generate()`
    /// when `params.grammar` is set. Plain forward + sample loop with a
    /// per-token logit mask from llguidance - no PLD, no graph capture.
    /// Streaming variant is not implemented yet.
    pub(super) async fn generate_with_grammar(
        &self,
        prompt: String,
        params: GenerationParams,
    ) -> Result<GenerationResult, Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let config = self.config.clone();

        let inner = tokio::task::spawn_blocking(move || -> AnyResult<GenerationResult> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Model not loaded. Load a model first."))?;

            let factory = grammar_factory_for(state)?;
            let grammar_str = params.grammar.as_deref().unwrap_or("json_object");
            let top_grammar = parse_grammar_spec(grammar_str)?;
            let parser = factory
                .create_parser(top_grammar)
                .map_err(|e| anyhow!("create_parser: {e}"))?;
            let mut constraint = llguidance::Constraint::new(parser);
            info!("🎯 Grammar-constrained decode: spec={}", grammar_str);

            let encoding = state
                .tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow!("Tokenization error: {}", e))?;
            let prompt_tokens: Vec<u32> = clamp_prompt_to_window(
                with_prefix_tokens(&params, encoding.get_ids()),
                effective_kv_window(
                    state.context_length,
                    config.context_length,
                    params.context_length,
                ),
                params.max_tokens.unwrap_or(config.max_tokens),
            );

            let effective_context = params
                .context_length
                .map(|c| c.min(state.context_length))
                .unwrap_or(state.context_length);
            let max_tokens = params
                .max_tokens
                .unwrap_or(config.max_tokens)
                .min(effective_context.saturating_sub(prompt_tokens.len()));
            let temperature = params.temperature.unwrap_or(config.temperature);
            let top_p = params.top_p.unwrap_or(config.top_p);
            let top_k = params.top_k.unwrap_or(config.top_k);
            let seed = params.seed.unwrap_or(config.seed);
            let repeat_penalty = params.repeat_penalty.unwrap_or(config.repeat_penalty);
            let repeat_last_n = params.repeat_last_n.unwrap_or(config.repeat_last_n);
            let device = state.device.clone();

            let sampling = build_sampling_from(temperature, top_p, top_k);
            let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

            state
                .model
                .set_early_exit_threshold(params.early_exit_threshold);

            // Request-start graph hygiene: a prior graph-mode request's
            // per-layer buffers (graph_rope_cos etc.) would make this
            // path's single-token decode forwards read frozen rope values.
            state.model.invalidate_graph_state();

            let prefill_start = std::time::Instant::now();
            let x = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
            let mut logits = state.model.forward_prefill_chunked(&x, 0)?;
            let prompt_eval_duration = prefill_start.elapsed().as_nanos() as u64;
            let mut pos = prompt_tokens.len();

            let mut recent_tokens: Vec<u32> = prompt_tokens.clone();
            let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
            let decode_start = std::time::Instant::now();

            for step in 0..max_tokens {
                let logits_cpu = match logits.device() {
                    Device::Cpu => logits.clone(),
                    _ => logits.to_device(&Device::Cpu)?,
                };
                let mut logits_vec = apply_repeat_penalty(
                    &logits_cpu.squeeze(0)?,
                    &recent_tokens,
                    repeat_penalty,
                    repeat_last_n,
                )?
                .to_dtype(crate::tensor::DType::F32)?
                .to_vec1::<f32>()?;

                let res = constraint
                    .compute_mask()
                    .map_err(|e| anyhow!("compute_mask: {e}"))?;
                if res.is_stop() {
                    debug!("Grammar reached stop state at step {}", step);
                    break;
                }
                if let Some(mask) = res.sample_mask.as_ref() {
                    for (i, l) in logits_vec.iter_mut().enumerate() {
                        if !mask.is_allowed(i as u32) {
                            *l = f32::NEG_INFINITY;
                        }
                    }
                }

                let logits_t = Tensor::from_vec(logits_vec, (state.vocab_size,), &Device::Cpu)?;
                let next_token = logits_processor.sample(&logits_t)?;

                let commit = constraint
                    .commit_token(Some(next_token))
                    .map_err(|e| anyhow!("commit_token: {e}"))?;

                generated_tokens.push(next_token);
                recent_tokens.push(next_token);

                if commit.stop {
                    debug!("Grammar committed final token; stopping");
                    break;
                }
                if next_token == state.eos_token_id
                    || state.eos_token_ids_extra.contains(&next_token)
                {
                    break;
                }

                let x = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                // stepd (non-stream decode): use state.forward
                // so moondream gets graph capture.
                logits = state.forward(&x, pos)?;
                pos += 1;
            }

            let eval_duration = decode_start.elapsed().as_nanos() as u64;
            let text = state
                .tokenizer
                .decode(&generated_tokens, true)
                .map_err(|e| anyhow!("decode: {e}"))?;

            Ok(GenerationResult {
                text,
                tokens: {
                    let mut full = prompt_tokens.clone();
                    full.extend_from_slice(&generated_tokens);
                    full
                },
                prompt_eval_count: prompt_tokens.len() as u64,
                prompt_eval_duration,
                eval_count: generated_tokens.len() as u64,
                eval_duration,
            })
        })
        .await;

        match inner {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }

    /// Streaming counterpart of `generate_with_grammar`. Same simple sequential
    /// loop but pushes each decoded token's text onto the returned channel as
    /// soon as it's sampled. Cancellation: a dropped receiver makes the next
    /// `tx.blocking_send` return Err and the loop bails (mirrors the behaviour
    /// of the unconstrained streaming path).
    pub(super) async fn generate_stream_with_grammar(
        &self,
        prompt: String,
        params: GenerationParams,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String, String>>, Box<dyn std::error::Error>>
    {
        let guard = self.model_state.lock().await;
        if guard.is_none() {
            return Err("Model not loaded.".into());
        }
        drop(guard);

        let model_state = self.model_state.clone();
        let config = self.config.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, String>>(64);

        std::thread::spawn(move || {
            let mut guard = model_state.blocking_lock();
            let state = match guard.as_mut() {
                Some(s) => s,
                None => {
                    let _ = tx.blocking_send(Err("Model not loaded".to_string()));
                    return;
                }
            };

            let factory = match grammar_factory_for(state) {
                Ok(f) => f,
                Err(e) => {
                    let _ = tx.blocking_send(Err(format!("grammar factory: {e}")));
                    return;
                }
            };
            let grammar_str = params.grammar.as_deref().unwrap_or("json_object");
            let top_grammar = match parse_grammar_spec(grammar_str) {
                Ok(g) => g,
                Err(e) => {
                    let _ = tx.blocking_send(Err(format!("parse_grammar_spec: {e}")));
                    return;
                }
            };
            let parser = match factory.create_parser(top_grammar) {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.blocking_send(Err(format!("create_parser: {e}")));
                    return;
                }
            };
            let mut constraint = llguidance::Constraint::new(parser);
            info!("🎯 Grammar-constrained streaming: spec={}", grammar_str);

            let encoding = match state.tokenizer.encode(prompt.as_str(), true) {
                Ok(e) => e,
                Err(e) => {
                    let _ = tx.blocking_send(Err(format!("Tokenization: {e}")));
                    return;
                }
            };
            let prompt_tokens: Vec<u32> = clamp_prompt_to_window(
                with_prefix_tokens(&params, encoding.get_ids()),
                effective_kv_window(
                    state.context_length,
                    config.context_length,
                    params.context_length,
                ),
                params.max_tokens.unwrap_or(config.max_tokens),
            );

            let effective_context = params
                .context_length
                .map(|c| c.min(state.context_length))
                .unwrap_or(state.context_length);
            let max_tokens = params
                .max_tokens
                .unwrap_or(config.max_tokens)
                .min(effective_context.saturating_sub(prompt_tokens.len()));
            let temperature = params.temperature.unwrap_or(config.temperature);
            let top_p = params.top_p.unwrap_or(config.top_p);
            let top_k = params.top_k.unwrap_or(config.top_k);
            let seed = params.seed.unwrap_or(config.seed);
            let repeat_penalty = params.repeat_penalty.unwrap_or(config.repeat_penalty);
            let repeat_last_n = params.repeat_last_n.unwrap_or(config.repeat_last_n);
            let device = state.device.clone();

            let sampling = build_sampling_from(temperature, top_p, top_k);
            let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

            state
                .model
                .set_early_exit_threshold(params.early_exit_threshold);

            // Request-start graph hygiene (see generate_with_grammar).
            state.model.invalidate_graph_state();

            let x =
                match Tensor::new(prompt_tokens.as_slice(), &device).and_then(|t| t.unsqueeze(0)) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("Prompt tensor: {e}")));
                        return;
                    }
                };
            let mut logits = match state.model.forward_prefill_chunked(&x, 0) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.blocking_send(Err(format!("Prefill: {e}")));
                    return;
                }
            };
            let mut pos = prompt_tokens.len();

            let mut recent_tokens: Vec<u32> = prompt_tokens.clone();
            // Cumulative-decode bookkeeping for grammar streaming  - 
            // see stream_token in the unified loop for rationale.
            let mut generated_tokens: Vec<u32> = Vec::with_capacity(max_tokens);
            let mut sent_text_len: usize = 0;

            for _step in 0..max_tokens {
                let logits_cpu = match logits.device() {
                    Device::Cpu => logits.clone(),
                    _ => match logits.to_device(&Device::Cpu) {
                        Ok(l) => l,
                        Err(e) => {
                            let _ = tx.blocking_send(Err(format!("Logits to CPU: {e}")));
                            return;
                        }
                    },
                };
                let mut logits_vec = match logits_cpu
                    .squeeze(0)
                    .and_then(|l| {
                        apply_repeat_penalty(&l, &recent_tokens, repeat_penalty, repeat_last_n)
                    })
                    .and_then(|l| l.to_dtype(crate::tensor::DType::F32))
                    .and_then(|l| l.to_vec1::<f32>())
                {
                    Ok(v) => v,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("Logits prep: {e}")));
                        return;
                    }
                };

                let res = match constraint.compute_mask() {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("compute_mask: {e}")));
                        return;
                    }
                };
                if res.is_stop() {
                    debug!("Grammar reached stop state");
                    break;
                }
                if let Some(mask) = res.sample_mask.as_ref() {
                    for (i, l) in logits_vec.iter_mut().enumerate() {
                        if !mask.is_allowed(i as u32) {
                            *l = f32::NEG_INFINITY;
                        }
                    }
                }

                let logits_t = match Tensor::from_vec(logits_vec, (state.vocab_size,), &Device::Cpu)
                {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("Tensor::from_vec: {e}")));
                        return;
                    }
                };
                let next_token = match logits_processor.sample(&logits_t) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("Sample: {e}")));
                        return;
                    }
                };

                let commit = match constraint.commit_token(Some(next_token)) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("commit_token: {e}")));
                        return;
                    }
                };

                generated_tokens.push(next_token);
                let cumulative = state
                    .tokenizer
                    .decode(&generated_tokens, true)
                    .unwrap_or_default();
                let chunk = if cumulative.len() >= sent_text_len {
                    char_safe_suffix(&cumulative, sent_text_len)
                } else {
                    // rewound - `sent_text_len = cumulative.len()` below covers it
                    cumulative.clone()
                };
                sent_text_len = cumulative.len();
                if tx.blocking_send(Ok(chunk)).is_err() {
                    debug!("🛑 Grammar streaming stopped - client disconnected");
                    return;
                }

                recent_tokens.push(next_token);

                if commit.stop
                    || next_token == state.eos_token_id
                    || state.eos_token_ids_extra.contains(&next_token)
                {
                    break;
                }

                let x = match Tensor::new(&[next_token], &device).and_then(|t| t.unsqueeze(0)) {
                    Ok(t) => t,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("Token tensor: {e}")));
                        return;
                    }
                };
                // stepd: use LoadedModelState::forward so
                // moondream gets graph capture/launch when warmed up.
                // Other variants delegate to model.forward unchanged.
                logits = match state.forward(&x, pos) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("Decode forward: {e}")));
                        return;
                    }
                };
                pos += 1;
            }
        });

        Ok(rx)
    }
}
