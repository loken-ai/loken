//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::params::{FinishReason, TokenLogprob};
use super::*;
use crate::inference::engine::decode_step::record_logprobs;

impl LlmEngine {
    /// Generate stream - returns channel receiver for streaming tokens with real model inference
    /// True if the loaded model is served through the continuous-batch worker
    /// (so the HTTP layer can skip the single-request gate -> requests batch).
    pub async fn is_continuous(&self) -> bool {
        matches!(self.model_state.lock().await.as_ref(),
            Some(s) if s.model.continuous_server().is_some())
    }

    /// Stream a request through the continuous-batch worker: tokenize, submit
    /// with the resolved sampling controls, and pump the worker's token channel
    /// back as incrementally-detokenized text chunks (same NDJSON shape the
    /// serial path emits). Stops on any EOS token, a stop sequence, or max_tokens.
    pub(super) fn stream_via_cb(
        &self,
        server: std::sync::Arc<crate::inference::serve::continuous_serve::ContinuousServer>,
        tokenizer: Tokenizer,
        eos_set: Vec<u32>,
        ctx_len: usize,
        prompt: &str,
        params: GenerationParams,
    ) -> tokio::sync::mpsc::Receiver<Result<String, String>> {
        use crate::inference::serve::continuous_batch::SamplingParams;
        use crate::inference::serve::continuous_serve::CbToken;
        let config = self.config.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, String>>(4096);
        let tokens: Vec<u32> = match tokenizer.encode(prompt, true) {
            Ok(e) => with_prefix_tokens(&params, e.get_ids()),
            Err(e) => {
                let _ = tx.try_send(Err(format!("Tokenization: {e}")));
                return rx;
            }
        };
        let max_new = params
            .max_tokens
            .unwrap_or(config.max_tokens)
            .min(ctx_len.saturating_sub(tokens.len()))
            .max(1);
        // Mirror build_sampling_from: top_k>0 gates TopK; top_p<1.0 gates TopP;
        // temperature==0 (penalty<=1) ⇒ exact argmax in the worker's sampler.
        let temperature = params.temperature.unwrap_or(config.temperature);
        let top_p = params.top_p.unwrap_or(config.top_p);
        let top_k = params.top_k.unwrap_or(config.top_k);
        let sampling = SamplingParams {
            temperature: temperature as f64,
            top_k: if top_k > 0 { Some(top_k) } else { None },
            top_p: if top_p < 1.0 {
                Some(top_p as f64)
            } else {
                None
            },
            repeat_penalty: params.repeat_penalty.unwrap_or(config.repeat_penalty),
            repeat_last_n: params.repeat_last_n.unwrap_or(config.repeat_last_n),
            seed: params.seed.unwrap_or(config.seed),
        };
        let stops = params.stop_sequences.clone();
        let cb_rx = server.submit_sampled(tokens, max_new, sampling);
        std::thread::spawn(move || {
            let mut gen: Vec<u32> = Vec::new();
            let mut emitted = String::new();
            // A Harmony vocabulary keeps its channel tokens in the text, for the API to
            // split the analysis from the answer; any other drops its special tokens.
            let skip_special = tokenizer.token_to_id("<|channel|>").is_none();
            while let Ok(CbToken::Tok(t)) = cb_rx.recv() {
                {
                    {
                        if eos_set.contains(&t) {
                            break;
                        }
                        gen.push(t);
                        let full = match tokenizer.decode(&gen, skip_special) {
                            Ok(s) => s,
                            Err(_) => continue,
                        };
                        if full.len() <= emitted.len() || !full.starts_with(&emitted) {
                            continue;
                        }
                        // Stop sequence: emit text up to it, then finish.
                        if let Some(pos) = stops
                            .iter()
                            .filter(|s| !s.is_empty())
                            .filter_map(|s| full.find(s.as_str()))
                            .min()
                        {
                            if pos > emitted.len() {
                                let _ = tx.blocking_send(Ok(full[emitted.len()..pos].to_string()));
                            }
                            break;
                        }
                        let delta = full[emitted.len()..].to_string();
                        emitted = full;
                        if tx.blocking_send(Ok(delta)).is_err() {
                            break;
                        }
                    }
                }
            }
        });
        rx
    }

    pub async fn generate_stream(
        &self,
        prompt: &str,
        params: GenerationParams,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String, String>>, Box<dyn std::error::Error>>
    {
        // Grammar-constrained streaming path. Same fast path as
        // generate_with_grammar, but emits tokens to a channel as they're
        // sampled. Bypasses PLD/graph branches.
        if params.grammar.is_some() {
            return self
                .generate_stream_with_grammar(prompt.to_string(), params)
                .await;
        }
        // Continuous-batch delegation: a CB-served model bypasses the serial
        // lock+forward decode entirely - submit to the background worker and
        // stream its tokens back. Greedy is bit-identical; sampling mirrors
        // build_sampling_from for parity with the serial path.
        {
            let guard = self.model_state.lock().await;
            match guard.as_ref() {
                None => return Err("Model not loaded.".into()),
                Some(state) => {
                    if let Some(server) = state.model.continuous_server() {
                        let server = std::sync::Arc::clone(server);
                        let tokenizer = state.tokenizer.clone();
                        let mut eos_set = state.eos_token_ids_extra.clone();
                        eos_set.push(state.eos_token_id);
                        let ctx_len = state.context_length;
                        drop(guard);
                        return Ok(
                            self.stream_via_cb(server, tokenizer, eos_set, ctx_len, prompt, params)
                        );
                    }
                }
            }
        }

        let model_state = self.model_state.clone();
        let logprob_queue = self.logprob_queue.clone();
        let sessions = self.sessions.clone();
        let session_id = params.session_id.clone();
        let config = self.config.clone();
        let kv_shift_reuse = config.kv_shift_reuse;
        let kv_snapshots = config.kv_snapshots;
        let kv_snapshot_budget =
            (config.kv_snapshot_budget_gb.max(0.0) * (1u64 << 30) as f64) as u64;
        let kv_disk = self.kv_disk.clone();
        let prompt = prompt.to_string();
        // Buffer-size 4096 (vs prior 64): with the prior buffer, after
        // 64 chunks the engine's `blocking_send` blocks waiting for the
        // axum-side async_stream to drain. The wait adds ~50-80ms of
        // engine-side wall to streaming over long generations. With a
        // generous buffer the engine produces freely; the consumer
        // drains in parallel. Memory cost is tiny (a few KB of String
        // pointers per slot).
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, String>>(4096);

        // Bench-fairness fix: capture engine-pure compute time into
        // a shared slot the HTTP handler reads for the stream final
        // chunk. compute_ns measures forward+sample work only,
        // EXCLUDING stream_token's tokenizer decode and the blocking_send
        // back-pressure - matches Ollama's eval_duration semantics so
        // assay computes comparable tok/s in stream mode.
        let stream_stats_slot = self.stream_stats_slot();
        // Spec-decode: capture the drafter engine + a runtime Handle. The stream
        // decode runs on a raw std::thread with NO tokio context, so the lazy drafter
        // load (async, !Send) needs an explicit Handle::block_on. Off unless
        // a drafter is configured; output correctness is structural (target verify
        // unchanged - the drafter only proposes).
        let spec_engine = self.clone_for_spec();
        let spec_rt = tokio::runtime::Handle::current();

        std::thread::spawn(move || {
            let stream_start = std::time::Instant::now();
            #[allow(unused_assignments)]
            let mut prefill_end: Option<std::time::Instant> = None;
            // Accumulates compute (forward + sample) between stream_token
            // calls. iter_compute_start is reset after each stream_token
            // returns; before the NEXT stream_token call we add the
            // elapsed compute work.
            let mut compute_ns: u64 = 0;
            #[allow(unused_assignments)]
            let mut iter_compute_start: Option<std::time::Instant> = None;
            // Phase 1: Setup + tokenize + batched prefill (with lock)
            #[cfg_attr(feature = "cuda", allow(unused_variables))]
            // The resolved sampling parameters travel as ONE value rather than as five
            // slots of this tuple. Spelled out here, spelled out again where the setup
            // block builds it and a third time where the block resolves it, the same list
            // said the same thing three times - and a knob added to it reached the decode
            // loop only if all three were remembered.
            // Log-probabilities gathered between two chunks, and the prompt tokens the
            // session cache already held: both outlive the prefill block.
            let mut logprob_buf: Vec<TokenLogprob> = Vec::new();
            let reused_prompt_tokens: usize;
            let (
                mut next_token,
                mut pos,
                eos_token_id,
                resolved,
                device_clone,
                mut logits_processor,
                prompt_tokens,
                _use_speculative,
                mut calibrator,
            ) = {
                let mut guard = model_state.blocking_lock();
                let state = match guard.as_mut() {
                    Some(s) => s,
                    None => {
                        let _ = tx.blocking_send(Err("Model not loaded".to_string()));
                        return;
                    }
                };

                debug!("🎬 Starting streaming generation for model: {}", state.name);

                // Tokenize. The prompt text is user content; log its size only.
                debug!("Tokenizing prompt: {} chars", prompt.len());
                let encoding = match state.tokenizer.encode(prompt.as_str(), true) {
                    Ok(e) => e,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("Tokenization: {e}")));
                        return;
                    }
                };
                // Vision requests splice image embeds into the prefill AFTER
                // tokenization - reserve their KV positions (see vision_extra_kv),
                // and reject outright when the image alone can't fit the window.
                let kv_window = effective_kv_window(
                    state.context_length,
                    config.context_length,
                    params.context_length,
                );
                let vision_extra = vision_extra_kv(state.image_embeds.as_ref())
                    + qwen35_vision_extra_kv(state.qwen35_image.as_ref());
                if vision_extra >= kv_window {
                    let _ = tx.blocking_send(Err(format!(
                        "image occupies {vision_extra} KV positions but the context window is only \
                         {kv_window}; increase context_length / num_ctx or use a smaller image"
                    )));
                    return;
                }
                // qwen35-VL keeps its image as a single in-prompt sentinel token  -
                // clamp around it (the generic BOS+tail clamp would drop it).
                let qwen35_sentinel = if state.qwen35_image.is_some() {
                    state.model.qwen35_image_token()
                } else {
                    None
                };
                let prompt_tokens: Vec<u32> = match qwen35_sentinel {
                    Some(sent) => clamp_qwen35_prompt_to_window(
                        with_prefix_tokens(&params, encoding.get_ids()),
                        kv_window - vision_extra,
                        params.max_tokens.unwrap_or(config.max_tokens),
                        sent,
                    ),
                    None => clamp_prompt_to_window(
                        with_prefix_tokens(&params, encoding.get_ids()),
                        kv_window - vision_extra,
                        params.max_tokens.unwrap_or(config.max_tokens),
                    ),
                };

                // Token ids decode back to the prompt, so only the count is logged.
                debug!("Prompt tokens: {}", prompt_tokens.len());

                // Resolve generation parameters
                // Per-request num_ctx is clamped to the model's loaded context window.
                let effective_context = params
                    .context_length
                    .map(|c| c.min(state.context_length))
                    .unwrap_or(state.context_length);
                // Count the spliced image embeds toward the consumed KV so
                // max_tokens can't push decode past the allocated KV cap.
                let resolved = resolve_gen_params(
                    &params,
                    &config,
                    effective_context,
                    prompt_tokens.len() + vision_extra,
                );
                let device = state.device.clone();

                // Set inference modes before forward passes
                state
                    .model
                    .set_early_exit_threshold(params.early_exit_threshold);

                // -- Session-persistent KV cache -------------------------
                // If this request carries a session id whose cached tokens
                // are a strict prefix of the new prompt, skip re-prefilling
                // the common prefix: forward only the delta at offset = |prefix|.
                //
                // Vision case: cache positions [0, image_prefix_len) hold
                // the BOS + image-embed K/V from a prior request. Same
                // image (matching image_hash) + same image_prefix_len -> those
                // 730+ positions are automatically common. Then extend with
                // common text-tokens prefix from the post-image positions.
                let is_vision = state.model.is_vision_model() && state.image_embeds.is_some();
                let cur_image_hash: Option<u64> = if is_vision {
                    state.image_embed_cache.as_ref().map(|(h, _)| *h)
                } else {
                    None
                };
                let cur_image_prefix_len: usize = if is_vision {
                    state
                        .image_embeds
                        .as_ref()
                        .and_then(|t| t.dim(1).ok())
                        .map(|n| 1 + n)
                        .unwrap_or(0)
                } else {
                    0
                };
                // Find the longest common prefix between any cached session
                // and the new prompt. Extends the
                // previous "same session_id only" matching to ALSO consider
                // OTHER sessions on the same model, so a fresh session that
                // happens to share a prompt prefix with a recently-finished
                // one can reuse that KV state. Single resident KV per
                // model means we can only match against ONE other session's
                // tokens (whichever the engine currently holds), but the
                // sessions table tells us which.
                let session_start: usize = if is_vision
                    && session_id.is_some()
                    && cur_image_hash.is_some()
                {
                    // Vision prefix-KV reuse: match by image_hash + common text.
                    let sess_guard = sessions.blocking_lock();
                    let mut best: Option<(&str, usize, usize)> = None;
                    for (sid, sess) in sess_guard
                        .iter()
                        .filter(|(sid, _)| sid.as_str() == GLOBAL_PROMPT_CACHE_KEY)
                    {
                        if sess.model_name != state.name {
                            continue;
                        }
                        if sess.image_hash != cur_image_hash {
                            continue;
                        }
                        if sess.image_prefix_len != cur_image_prefix_len {
                            continue;
                        }
                        // Image positions are common by hash match. Now extend
                        // with text tokens prefix.
                        // `kv_len` counts EVERY row written, image prefix included, so it
                        // is the cache length outright - and the ceiling on what any
                        // amount of text agreement can reuse. Deriving that length from
                        // the token list instead is the same defect the text path had:
                        // it names tokens the cache never received.
                        let total_in_cache = sess.kv_len;
                        let common_text = sess
                            .tokens
                            .iter()
                            .zip(prompt_tokens.iter())
                            .take_while(|(a, b)| a == b)
                            .count();
                        let total_common =
                            (sess.image_prefix_len + common_text).min(total_in_cache);
                        let request_match = session_id.as_deref() == Some(sid.as_str());
                        let take = match best {
                            None => true,
                            Some((_, prev_common, _)) => {
                                total_common > prev_common
                                    || (total_common == prev_common && request_match)
                            }
                        };
                        if take {
                            best = Some((sid.as_str(), total_common, total_in_cache));
                        }
                    }
                    match best {
                        None => 0,
                        Some((sid, total_common, total_in_cache)) => {
                            // If everything matches, leave the LAST token to
                            // prefill so we get fresh logits for decode start.
                            let new_full_len = cur_image_prefix_len + prompt_tokens.len();
                            let total_common = if total_common >= new_full_len {
                                new_full_len.saturating_sub(1)
                            } else {
                                total_common
                            };
                            if total_common < total_in_cache {
                                if state.model.supports_trim_kv() {
                                    state.model.trim_kv(total_common);
                                    debug!(
                                        "♻️  Vision prefix reuse from session '{}': image+{} text tokens, trimmed to pos {}",
                                        sid, total_common - cur_image_prefix_len, total_common,
                                    );
                                    total_common
                                } else {
                                    debug!("VISION: no trim_kv support, falling back to cold");
                                    0
                                }
                            } else {
                                debug!("♻️  Vision prefix reuse from session '{}': strict extension at pos {}", sid, total_common);
                                total_common
                            }
                        }
                    }
                } else if is_vision {
                    0 // no session_id or no image_hash -> cold path
                } else if session_id.is_none() {
                    // ollama-parity GLOBAL prompt cache: a request without a session_id
                    // reuses the common prefix of the single RESIDENT KV (the last
                    // request's sequence, mirrored under GLOBAL_PROMPT_CACHE_KEY,
                    // in-memory only - never persisted). The SAME proven trim logic as
                    // the session path guarantees greedy-identical output (keep only the
                    // common prefix; re-prefill the divergent suffix + leave the last
                    // token on a full match). Disable for multi-tenant privacy
                    // (cross-tenant timing side-channel) with
                    // LOKEN_NO_GLOBAL_PROMPT_CACHE=1.
                    kv_reuse_start(
                        state.model.as_mut(),
                        &sessions,
                        &state.name,
                        &prompt_tokens,
                        false,
                        kv_shift_reuse,
                        kv_disk.as_deref().map(|d| (d, kv_snapshots)),
                    )
                } else {
                    let sess_guard = sessions.blocking_lock();
                    // One KV is resident, and the entry under GLOBAL_PROMPT_CACHE_KEY
                    // is the sequence it holds. Every other session entry is token
                    // bookkeeping only: matching the prompt against it and trimming
                    // the resident KV to that length would serve another session's
                    // KV under this prompt's tokens. So the match is made against the
                    // resident entry alone, capped by what is actually in the cache.
                    let mut best: Option<(&str, usize, usize)> = None; // (sid, common, t_len)
                    for (sid, sess) in sess_guard
                        .iter()
                        .filter(|(sid, _)| sid.as_str() == GLOBAL_PROMPT_CACHE_KEY)
                    {
                        if sess.model_name != state.name {
                            continue;
                        }
                        let t = &sess.tokens;
                        let common = t
                            .iter()
                            .zip(prompt_tokens.iter())
                            .take_while(|(a, b)| a == b)
                            .count()
                            .min(sess.kv_len);
                        if common == 0 {
                            continue;
                        }
                        let request_match = session_id.as_deref() == Some(sid.as_str());
                        let take = match best {
                            None => true,
                            Some((_, prev_common, _)) => {
                                common > prev_common || (common == prev_common && request_match)
                            }
                        };
                        if take {
                            best = Some((sid.as_str(), common, t.len()));
                        }
                    }
                    match best {
                        None => 0,
                        Some((sid, common, t_len)) => {
                            // Fix: when the new prompt is a strict
                            // prefix of the cached tokens (common >=
                            // prompt_tokens.len()), we MUST still trim the
                            // cache. Falling back to session_start=0 (the
                            // old "cold path") doesn't reset the cache:
                            // forward(prompt, index_pos=0) appends, so cache
                            // ends up with stale generated tokens from prior
                            // iters at positions [prompt_len, t_len) plus
                            // freshly-written prompt slots shifted by the
                            // cache offset. Attention then reads stale +
                            // misaligned K/V -> cyclic-3 non-determinism.
                            //
                            // Mirror the vision branch's "leave last token
                            // to prefill" policy: trim to prompt_len-1, then
                            // prefill 1 token so we get fresh decode-start
                            // logits. Cache slots [prompt_len-1, t_len) are
                            // now invalidated; only the 1 new token is
                            // appended on top of the trimmed cache.
                            let effective_common = common
                                .min(prompt_tokens.len())
                                .saturating_sub(if common >= prompt_tokens.len() { 1 } else { 0 });
                            if effective_common == 0 && common >= prompt_tokens.len() {
                                // prompt has 1 token (degenerate). Cold path.
                                if state.model.supports_trim_kv() {
                                    state.model.trim_kv(0);
                                }
                                0
                            } else if common >= prompt_tokens.len() {
                                if state.model.supports_trim_kv() {
                                    state.model.trim_kv(effective_common);
                                    debug!(
                                        "♻️  Prefix reuse from session '{}' (full match): stored={} prompt={} -> trim to {} + prefill last token",
                                        sid, t_len, prompt_tokens.len(), effective_common,
                                    );
                                    effective_common
                                } else {
                                    0
                                }
                            } else if common < t_len {
                                if state.model.supports_trim_kv() {
                                    state.model.trim_kv(common);
                                    debug!(
                                        "♻️  Prefix reuse from session '{}': stored={} new={} common={} (trimmed to {})",
                                        sid, t_len, prompt_tokens.len(), common, common,
                                    );
                                    common
                                } else {
                                    debug!("Prefix reuse skipped: variant cannot trim KV");
                                    0
                                }
                            } else {
                                // common == t_len: new is a strict extension
                                debug!("♻️  Prefix reuse from session '{}': strict extension, skipping {} prefill tokens", sid, common);
                                common
                            }
                        }
                    }
                };
                reused_prompt_tokens = session_start;
                if session_start > 0 {
                    debug!(
                        "♻️  Session reuse: skipping prefill of {} tokens, \
                         prefilling only {} new tokens at offset {}",
                        session_start,
                        prompt_tokens.len() - session_start,
                        session_start,
                    );
                    // Same as the non-stream path: record the post-trim length NOW, so a
                    // request that never reaches its write-back cannot leave an entry
                    // claiming rows the trim removed. Understating is safe; the stale
                    // overstatement is what builds a mask for a KV that is not there.
                    // Vision sessions are keyed on their own entry and carry an image
                    // prefix that is not in `tokens`, so only the text cache is adjusted.
                    if !is_vision {
                        let mut g = sessions.blocking_lock();
                        if let Some(sess) = g.get_mut(GLOBAL_PROMPT_CACHE_KEY) {
                            sess.tokens.truncate(session_start);
                            sess.kv_len = session_start;
                        }
                    }
                }

                // Request-start graph hygiene (mirror of the non-stream path):
                // stale per-layer graph buffers from a prior request would make
                // a single-token prefill delta take `use_graph_ops` with frozen
                // rope values, and a stale quantized-KV `graph_captured` flag
                // would block KV growth. Cheap; does not touch KV contents.
                state.model.invalidate_graph_state();

                // Batched prefill - only the suffix beyond session_start.
                // For vision, session_start = image_prefix_len + text_common;
                // prompt_tokens is text-only, so subtract image_prefix_len.
                let text_start = if is_vision && session_start >= cur_image_prefix_len {
                    session_start - cur_image_prefix_len
                } else {
                    session_start
                };
                let prefill_slice = &prompt_tokens[text_start..];
                let x = match Tensor::new(prefill_slice, &device).and_then(|t| t.unsqueeze(0)) {
                    Ok(t) => t,
                    Err(e) => {
                        let err_msg =
                            format!("❌ STREAMING PREFILL: Tensor creation failed: {}", e);
                        error!("{}", err_msg);
                        let _ = tx.blocking_send(Err(err_msg));
                        return;
                    }
                };
                debug!(
                    "🔄 STREAMING PREFILL: Processing {} prompt tokens at offset {}...",
                    prefill_slice.len(),
                    session_start
                );
                let mut qwen35_next_pos: Option<usize> = None;
                let mut logits = if let Some((px, (gh, gw))) = state.qwen35_image.take() {
                    // qwen35moe (Qwen3-VL) cold vision prefill: ViT + splice at
                    // the image_token sentinel + mRoPE-2D. Returns the continuing
                    // logical decode position (image compresses positions).
                    match state
                        .model
                        .forward_qwen35_image(&prompt_tokens, &px, gh, gw, &device)
                    {
                        Ok((l, np)) => {
                            qwen35_next_pos = Some(np);
                            l
                        }
                        Err(e) => {
                            let _ = tx.blocking_send(Err(format!(
                                "STREAMING PREFILL (qwen35 vision): {e}"
                            )));
                            return;
                        }
                    }
                } else if is_vision && state.model.is_pixtral_vision() {
                    // Pixtral cold vision prefill: embeds spliced INSIDE the prompt
                    // ([INST] img [IMG_END] text). Prefix-reuse is N/A (different
                    // region layout); the ViT embed cache still skips re-encoding.
                    let image_embeds = state.image_embeds.as_ref().unwrap().clone();
                    match state.model.forward_pixtral_spliced(
                        &prompt_tokens,
                        &image_embeds,
                        &device,
                    ) {
                        Ok((l, total)) => {
                            qwen35_next_pos = Some(total);
                            l
                        }
                        Err(e) => {
                            let err_msg =
                                format!("STREAMING PREFILL (pixtral): Forward failed: {e}");
                            error!("{}", err_msg);
                            let _ = tx.blocking_send(Err(err_msg));
                            return;
                        }
                    }
                } else if is_vision && session_start >= cur_image_prefix_len {
                    // Vision prefix reuse: cache positions [0, session_start)
                    // already hold BOS + image embeds + previously-prefilled
                    // text tokens. Just prefill the new text-token delta at
                    // index_pos = session_start. NO image-embed re-prefill.
                    debug!("Vision prefix reuse: forward text-only at pos {} (image prefix already in cache)", session_start);
                    match state.model.forward_prefill_chunked(&x, session_start) {
                        Ok(l) => l,
                        Err(e) => {
                            let err_msg =
                                format!("STREAMING PREFILL (vision reuse): Forward failed: {}", e);
                            error!("{}", err_msg);
                            let _ = tx.blocking_send(Err(err_msg));
                            return;
                        }
                    }
                } else if is_vision {
                    // Cold vision prefill: BOS + image embeddings + text tokens.
                    let image_embeds = state.image_embeds.as_ref().unwrap();
                    let bos_token = match Tensor::new(&[state.eos_token_id], &device)
                        .and_then(|t| t.unsqueeze(0))
                    {
                        Ok(t) => t,
                        Err(e) => {
                            let _ = tx.blocking_send(Err(format!("Vision BOS tensor: {e}")));
                            return;
                        }
                    };
                    // PRIVACY-OK: tensor SHAPES, not the embeddings or the text.
                    debug!(
                        "Vision prefill: BOS + image_embeds {:?} + text {:?}",
                        image_embeds.shape(),
                        x.shape()
                    );
                    match state.model.forward_with_img(&bos_token, &x, image_embeds) {
                        Ok(l) => l,
                        Err(e) => {
                            let err_msg =
                                format!("STREAMING PREFILL (vision): Forward failed: {}", e);
                            error!("{}", err_msg);
                            let _ = tx.blocking_send(Err(err_msg));
                            return;
                        }
                    }
                } else {
                    // Exact-prompt snapshot reuse for recurrent hybrids, which the
                    // trim_kv prefix path above cannot serve (rolling state can be
                    // advanced, never rewound). A hit skips the prefill entirely;
                    // trim-capable variants take the default miss and prefill.
                    match state.model.try_restore_prefix(&prompt_tokens) {
                        Ok(Some(l)) => l,
                        Ok(None) => match state.model.forward_prefill_chunked(&x, session_start) {
                            Ok(l) => {
                                if let Err(e) = state.model.snapshot_prefix(&prompt_tokens, &l) {
                                    // A snapshot failure costs only the next reuse.
                                    debug!("prefix snapshot skipped: {e}");
                                }
                                l
                            }
                            Err(e) => {
                                let err_msg = format!("STREAMING PREFILL: Forward failed: {}", e);
                                error!("{}", err_msg);
                                let _ = tx.blocking_send(Err(err_msg));
                                return;
                            }
                        },
                        Err(e) => {
                            let err_msg =
                                format!("STREAMING PREFILL: prefix restore failed: {}", e);
                            error!("{}", err_msg);
                            let _ = tx.blocking_send(Err(err_msg));
                            return;
                        }
                    }
                };
                debug!("STREAMING PREFILL: Forward pass complete");
                // The first compute window is NOT opened here. Sampling the first token and
                // reaching the loop happen before that token reaches the client, so charging
                // them to decode made the published decode rate slower than the one a client
                // observes - impossible, and the reason it was found. That work belongs to
                // prefill, which closes at the top of the loop where its token exists.
                // Vision prefill writes BOS + image_embeds (typically
                // 729 patches for moondream) ahead of the text tokens
                // into the KV cache. Add the prefix length to `pos`
                // so subsequent decode RoPE positions and PLD verify
                // masks match the actual KV length.
                let vision_prefix_len: usize = if is_vision {
                    state
                        .image_embeds
                        .as_ref()
                        .and_then(|t| t.dim(1).ok())
                        .map(|n| 1 + n) // 1 BOS + N image patches
                        .unwrap_or(0)
                } else {
                    0
                };
                let pos = qwen35_next_pos.unwrap_or(prompt_tokens.len() + vision_prefix_len);
                // Explicit GPU->CPU transfer before sampling
                logits = match logits.device() {
                    Device::Cpu => logits,
                    _ => match logits.to_device(&Device::Cpu) {
                        Ok(l) => l,
                        Err(e) => {
                            let err_msg =
                                format!("❌ STREAMING PREFILL: Device transfer failed: {}", e);
                            error!("{}", err_msg);
                            let _ = tx.blocking_send(Err(err_msg));
                            return;
                        }
                    },
                };
                let squeezed = match logits.squeeze(0) {
                    Ok(s) => s,
                    Err(e) => {
                        let err_msg = format!("❌ STREAMING PREFILL: Squeeze failed: {}", e);
                        error!("{}", err_msg);
                        let _ = tx.blocking_send(Err(err_msg));
                        return;
                    }
                };
                let mut logits_processor = resolved.make_logits_processor();
                logits_processor.set_logit_bias(params.logit_bias.clone());
                logits_processor.set_top_logprobs(params.top_logprobs);
                logprob_buf.clear();
                let penalized = match apply_repeat_penalty(
                    &squeezed,
                    &prompt_tokens,
                    resolved.repeat_penalty,
                    resolved.repeat_last_n,
                ) {
                    Ok(l) => l,
                    Err(e) => {
                        let _ = tx.blocking_send(Err(format!("Repeat penalty failed: {}", e)));
                        return;
                    }
                };
                let next_token = match logits_processor.sample(&penalized) {
                    Ok(t) => t,
                    Err(e) => {
                        let err_msg = format!("❌ STREAMING PREFILL: Sampling failed: {}", e);
                        error!("{}", err_msg);
                        let _ = tx.blocking_send(Err(err_msg));
                        return;
                    }
                };
                record_logprobs(&state.tokenizer, &mut logits_processor, &mut logprob_buf);
                // The id itself is content; that prefill produced one is not.
                debug!("STREAMING PREFILL: first token sampled");

                #[cfg(feature = "opencl")]
                let calibrator = state.model.create_calibrator();
                #[cfg(not(feature = "opencl"))]
                let calibrator: Option<
                    crate::inference::serve::speculative_config::SpeculativeCalibrator,
                > = None;

                #[cfg(feature = "opencl")]
                if calibrator.is_some() {
                    tracing::info!("🔬 Streaming: speculative auto-tune calibrating...");
                }

                let use_speculative = state.model.supports_speculative();

                (
                    next_token,
                    pos,
                    state.eos_token_id,
                    resolved,
                    // Moved, not cloned: nothing above reads it again, and a device handle
                    // carries the context and the stream along with it.
                    device,
                    logits_processor,
                    prompt_tokens,
                    use_speculative,
                    calibrator,
                )
            }; // <- lock released after setup
            let ResolvedGenParams {
                max_tokens,
                temperature,
                top_k: _,
                repeat_penalty,
                repeat_last_n,
                ..
            } = resolved;
            let mut token_count = 0usize;
            let mut generated_token_ids = Vec::new();
            let mut recent_tokens: Vec<u32> = prompt_tokens.clone();

            // -- CUDA graph decode (streaming) --------------------------
            // the graph capture/replay loop existed only in the
            // non-streaming `generate` path; fair_bench (--stream) and every
            // streaming client decoded eagerly - the whole per-token launch
            // overhead ollama's graph replay avoids. Wire the same
            // warmup -> capture (shared `capture_decode_graph`) -> replay +
            // probation sequence into this loop. Non-split (Q4/Q8 dev-pos)
            // models only - split-path models are gated off engine-wide.
            #[cfg(feature = "cuda")]
            let (graph_ok, graph_stream) = {
                let guard = model_state.blocking_lock();
                match guard.as_ref() {
                    Some(s) if device_clone.is_cuda() && s.model.supports_graph_mode() => {
                        let ok = s.model.graph_capture_auto_on()
                            && s.model.kv_state_graph_safe()
                            && !s.model.needs_split_graph_path();
                        // Capture MUST target the MODEL's compute stream
                        // (same rule as the non-stream path).
                        let stream = if ok {
                            s.model
                                .model_cuda_stream()
                                .or_else(|| crate::tensor::cuda_ext::stream_of(&device_clone).ok())
                        } else {
                            None
                        };
                        (ok && stream.is_some(), stream)
                    }
                    _ => (false, None),
                }
            };
            #[cfg(feature = "cuda")]
            let mut cuda_graph: Option<crate::tensor::cuda_ext::CudaGraph> = None;
            #[cfg(feature = "cuda")]
            let mut graph_logits: Option<Tensor> = None;
            #[cfg(feature = "cuda")]
            let mut graph_disabled_this_request: bool = false;
            #[cfg(feature = "cuda")]
            let mut graph_warmed: bool = false;
            #[cfg(feature = "cuda")]
            let mut graph_capture_attempted: bool = false;
            #[cfg(feature = "cuda")]
            let mut graph_validations: usize = 0;

            // -- PLD (Prompt-Lookup Decoding) ---------------------------
            // Check supports_pld under a quick lock, then release.
            let pld_enabled = {
                let guard = model_state.blocking_lock();
                guard
                    .as_ref()
                    .map(|s| s.model.supports_pld())
                    .unwrap_or(false)
            };
            let mut pld_cache =
                crate::inference::serve::prompt_lookup::NgramDraftCache::new_cascade(2, 4, 5, 4096);
            if pld_enabled {
                pld_cache.push_many(&prompt_tokens);
                pld_cache.push(next_token);
            }
            // Spec-decode: lazily load the drafter (block_on via the captured
            // runtime Handle - this is a raw thread) + prefill it with the prompt so its
            // KV is in lockstep with the target, ready to draft from `next_token`.
            let mut spec_on = spec_rt.block_on(spec_engine.ensure_draft_loaded())
                && spec_engine.draft_prefill(&prompt_tokens);
            let mut pld_stats_drafted: u64 = 0;
            let mut pld_stats_accepted: u64 = 0;
            // Adaptive PLD gating: if recent accept rate is too low, disable
            // PLD for a cooldown window to avoid wasting compute on rejected
            // multi-position forwards.
            let mut pld_window_drafted: u32 = 0;
            let mut pld_window_accepted: u32 = 0;
            let mut pld_cooldown: u32 = 0;

            // Prepare stop sequence suffix buffer (eliminates O(N) decode per token)
            let mut stop_tracker = StopTracker::new(&params.stop_sequences);

            // Snapshot the tokenizer once so the hot-path stream_token closure
            // can decode each emitted token WITHOUT acquiring model_state's
            // Mutex. Tokenizer is internally Arc'd, so .clone() is cheap.
            let stream_tokenizer: Tokenizer = {
                let guard = model_state.blocking_lock();
                match guard.as_ref() {
                    Some(s) => s.tokenizer.clone(),
                    None => {
                        let _ = tx.blocking_send(Err("Model unloaded".to_string()));
                        return;
                    }
                }
            };
            // A Harmony vocabulary keeps its channel tokens in the text, for the API to
            // split the analysis from the answer; any other drops its special tokens.
            let skip_special_tokens = stream_tokenizer.token_to_id("<|channel|>").is_none();

            // Helper: decode a token to text and send the DIFF to the streaming channel.
            // Returns false if client disconnected or stop sequence matched.
            //
            // Per-token decode loses leading spaces with Metaspace/SentencePiece
            // tokenizers - the decoder treats each token boundary as needing
            // explicit-space, but standalone token decode strips the prefix.
            // Result: joined chunks "Hello" "world" -> "Helloworld" instead of
            // "Hello world". Fix is the standard pattern (llama.cpp, vllm):
            // decode the FULL cumulative token sequence each time, emit only
            // the new suffix beyond what was previously sent. `sent_text_len`
            // tracks how many chars of the cumulative decoded text have been
            // forwarded to the client.
            let stream_token = |token: u32,
                                token_count: &mut usize,
                                generated_token_ids: &mut Vec<u32>,
                                recent_tokens: &mut Vec<u32>,
                                stop_tracker: &mut StopTracker,
                                sent_text_len: &mut usize,
                                tx: &tokio::sync::mpsc::Sender<Result<String, String>>,
                                tokenizer: &Tokenizer,
                                stop_sequences: &[String]|
             -> bool {
                generated_token_ids.push(token);
                recent_tokens.push(token);

                // Incremental decode (2-token sliding window,
                // decode_step::incremental_chunk_text). The original
                // implementation called `tokenizer.decode(ALL)` every iter  -
                // O(N) per call -> O(N²) over the stream. For 64 tokens that's
                // ~2080 token-decode operations; on moondream stream-mode this
                // CPU work added ~80ms of overhead (~50% of total wall),
                // making stream tok/s look ~30pp worse than non-stream tok/s.
                let chunk_text = incremental_chunk_text(tokenizer, generated_token_ids);
                *sent_text_len += chunk_text.len();

                if *token_count < 10 {
                    trace!(
                        "🎯 Token {}: ID={} chunk='{}'",
                        token_count,
                        token,
                        chunk_text
                    );
                }

                if stop_tracker.is_active() {
                    let is_stop = stop_tracker.push_and_check(&chunk_text, stop_sequences);
                    if tx.blocking_send(Ok(chunk_text)).is_err() {
                        debug!("🛑 Streaming stopped - client disconnected");
                        return false;
                    }
                    *token_count += 1;
                    if is_stop {
                        return false;
                    }
                } else {
                    if tx.blocking_send(Ok(chunk_text)).is_err() {
                        debug!("🛑 Streaming stopped - client disconnected");
                        return false;
                    }
                    *token_count += 1;
                }

                true
            };

            let stop_sequences = params.stop_sequences.clone();
            // Cumulative-decode bookkeeping: how many chars of the joined
            // decoded text have already been streamed to the client.
            let mut sent_text_len: usize = 0;

            // Phase 2: Stream tokens - unified loop with auto-tune
            #[cfg(feature = "opencl")]
            let mut spec_active: Option<(
                usize,
                crate::inference::serve::speculative_config::SpeculativeMonitor,
            )> = None;
            // Pre-allocate draft buffers for speculative decoding
            #[cfg(feature = "opencl")]
            let mut draft_tokens_buf: Vec<u32> = Vec::with_capacity(8);
            #[cfg(feature = "opencl")]
            let mut draft_hidden_buf: Vec<(Tensor, usize)> = Vec::with_capacity(8);

            // Path B step 3: when the previous greedy sample
            // produced a device-resident U32 [1,1] tensor (via
            // gpu_sample_returning_tensor), reuse it as the next forward
            // input - skips the host int -> Tensor::new H->D copy.
            // None on first iter (next_token comes from prefill via host)
            // and after any branch that doesn't update the dev tensor
            // (PLD verify/draft, temperature > 0 sampling).
            let mut next_token_dev: Option<Tensor> = None;

            'stream_loop: while next_token != eos_token_id && token_count < max_tokens {
                if !logprob_buf.is_empty() {
                    if let Ok(mut q) = logprob_queue.lock() {
                        q.append(&mut logprob_buf);
                    }
                }
                // Prefill ends where its token exists, not where its forward pass returned.
                if prefill_end.is_none() {
                    prefill_end = Some(std::time::Instant::now());
                }
                // Close the compute window from the previous iter's forward+sample. There is
                // none to close on the first iteration: prefill produced that token.
                if let Some(t) = iter_compute_start.take() {
                    compute_ns += t.elapsed().as_nanos() as u64;
                }
                // Stream the current token (tokenizer decode + blocking_send)
                // - EXCLUDED from compute_ns.
                if !stream_token(
                    next_token,
                    &mut token_count,
                    &mut generated_token_ids,
                    &mut recent_tokens,
                    &mut stop_tracker,
                    &mut sent_text_len,
                    &tx,
                    &stream_tokenizer,
                    &stop_sequences,
                ) {
                    break;
                }
                // Restart compute window for this iter's forward+sample.
                iter_compute_start = Some(std::time::Instant::now());
                // Keep PLD n-gram buffer in sync (stream_token already
                // pushed to recent_tokens; pld_cache needs its own push).
                if pld_enabled {
                    pld_cache.push(next_token);
                }

                if token_count >= max_tokens {
                    break;
                }

                // Determine mode
                #[cfg(feature = "opencl")]
                let use_spec_this_iter = {
                    use crate::inference::serve::speculative_config::CalibrationType;
                    if let Some(ref cal) = calibrator {
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
                        let spec_k = if let Some((k, _)) = &spec_active {
                            *k
                        } else {
                            2
                        };
                        let t_spec = Instant::now();

                        let mut guard = model_state.blocking_lock();
                        let state = match guard.as_mut() {
                            Some(s) => s,
                            None => {
                                let _ = tx.blocking_send(Err("Model unloaded".to_string()));
                                break;
                            }
                        };

                        // Draft (reuse pre-allocated vectors)
                        let t_draft = Instant::now();
                        draft_tokens_buf.clear();
                        draft_hidden_buf.clear();
                        let mut draft_token = next_token;
                        let cuda_kv_pos_before = pos;

                        let draft_ok = (|| -> std::result::Result<(), String> {
                            for k in 0..spec_k {
                                let x = Tensor::new(&[draft_token], &device_clone)
                                    .and_then(|t| t.unsqueeze(0))
                                    .map_err(|e| format!("Draft tensor: {e}"))?;
                                let (dl, ch) = state
                                    .model
                                    .forward_draft(&x, pos + k)
                                    .map_err(|e| format!("Draft forward: {e}"))?;
                                let dl = match dl.device() {
                                    Device::Cpu => dl,
                                    _ => dl
                                        .to_device(&Device::Cpu)
                                        .map_err(|e| format!("Draft xfer: {e}"))?,
                                };
                                let dl = apply_repeat_penalty(
                                    &dl.squeeze(0).map_err(|e| format!("{e}"))?,
                                    &recent_tokens,
                                    repeat_penalty,
                                    repeat_last_n,
                                )
                                .map_err(|e| format!("Draft penalty: {e}"))?;
                                draft_token = logits_processor
                                    .sample(&dl)
                                    .map_err(|e| format!("Draft sample: {e}"))?;
                                draft_hidden_buf.push((ch, pos + k));
                                draft_tokens_buf.push(draft_token);
                                recent_tokens.push(draft_token);
                            }
                            Ok(())
                        })();
                        let draft_ms = t_draft.elapsed().as_secs_f64() * 1000.0;

                        if let Err(e) = draft_ok {
                            let _ = tx.blocking_send(Err(format!("Spec draft failed: {e}")));
                            break;
                        }

                        // Verify
                        let t_verify = Instant::now();
                        let verify_logits_batch = match state
                            .model
                            .forward_verify_batch(&draft_hidden_buf, cuda_kv_pos_before)
                        {
                            Ok(v) => v,
                            Err(e) => {
                                let _ = tx.blocking_send(Err(format!("Spec verify failed: {e}")));
                                break;
                            }
                        };
                        let verify_ms = t_verify.elapsed().as_secs_f64() * 1000.0;

                        // Accept/reject
                        let mut accepted = 0usize;
                        let mut accepted_tokens = Vec::new();
                        for k in 0..spec_k {
                            let vl = match &verify_logits_batch[k] {
                                vl if vl.device().is_cpu() => vl.clone(),
                                vl => match vl.to_device(&Device::Cpu) {
                                    Ok(l) => l,
                                    Err(e) => {
                                        let _ = tx.blocking_send(Err(format!("Verify xfer: {e}")));
                                        break 'stream_loop;
                                    }
                                },
                            };
                            let penalty_tokens =
                                &recent_tokens[..recent_tokens.len() - (spec_k - k)];
                            let vl = match apply_repeat_penalty(
                                &vl.squeeze(0).unwrap_or(vl.clone()),
                                penalty_tokens,
                                repeat_penalty,
                                repeat_last_n,
                            ) {
                                Ok(l) => l,
                                Err(e) => {
                                    let _ = tx.blocking_send(Err(format!("Verify penalty: {e}")));
                                    break 'stream_loop;
                                }
                            };
                            let verify_token = match logits_processor.sample(&vl) {
                                Ok(t) => t,
                                Err(e) => {
                                    let _ = tx.blocking_send(Err(format!("Verify sample: {e}")));
                                    break 'stream_loop;
                                }
                            };

                            if verify_token == draft_tokens_buf[k] {
                                accepted += 1;
                                accepted_tokens.push(draft_tokens_buf[k]);
                            } else {
                                accepted_tokens.push(verify_token);
                                for _ in 0..(spec_k - k) {
                                    recent_tokens.pop();
                                }
                                recent_tokens.push(verify_token);
                                break;
                            }
                        }

                        let new_pos = cuda_kv_pos_before + accepted + 1;
                        pos = new_pos;
                        if accepted < spec_k {
                            state.model.trim_cuda_kv(new_pos);
                            state.model.trim_opencl_kv(new_pos);
                        }

                        // Full forward for next token
                        let next_result = (|| -> std::result::Result<u32, String> {
                            let last = *accepted_tokens.last().unwrap();
                            let x = Tensor::new(&[last], &device_clone)
                                .and_then(|t| t.unsqueeze(0))
                                .map_err(|e| format!("{e}"))?;
                            let mut logits =
                                state.model.forward(&x, pos).map_err(|e| format!("{e}"))?;
                            pos += 1;
                            logits = match logits.device() {
                                Device::Cpu => logits,
                                _ => logits.to_device(&Device::Cpu).map_err(|e| format!("{e}"))?,
                            };
                            let logits = apply_repeat_penalty(
                                &logits.squeeze(0).map_err(|e| format!("{e}"))?,
                                &recent_tokens,
                                repeat_penalty,
                                repeat_last_n,
                            )
                            .map_err(|e| format!("{e}"))?;
                            logits_processor.sample(&logits).map_err(|e| format!("{e}"))
                        })();

                        let next_from_verify = match next_result {
                            Ok(t) => t,
                            Err(e) => {
                                let _ = tx.blocking_send(Err(format!("Post-spec forward: {e}")));
                                break;
                            }
                        };

                        let spec_ms = t_spec.elapsed().as_secs_f64() * 1000.0;
                        let tokens_produced = accepted + 1;

                        // Feed calibrator
                        if let Some(ref mut cal) = calibrator {
                            cal.record_draft(draft_ms / spec_k as f64);
                            cal.record_verify(verify_ms / spec_k as f64);
                            cal.record_transfer(2.0);
                            cal.record_acceptance(spec_k, accepted);

                            use crate::inference::serve::speculative_config::CalibrationType;
                            if cal.phase() == CalibrationType::Done && spec_active.is_none() {
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
                                    tracing::info!(
                                        "🎯 Auto-tune: speculative decoding DISABLED ({})",
                                        decision.reason
                                    );
                                }
                                calibrator = None;
                            }
                        }

                        // Feed runtime monitor
                        if let Some((_, ref mut monitor)) = spec_active {
                            monitor.record_cycle(spec_ms, tokens_produced);
                        }

                        drop(guard); // Release lock before streaming

                        // Close compute window before emitting accepted
                        // tokens (the loop below does tokenizer.decode +
                        // blocking_send - EXCLUDED from compute_ns).
                        if let Some(t) = iter_compute_start.take() {
                            compute_ns += t.elapsed().as_nanos() as u64;
                        }
                        // Stream accepted tokens with incremental
                        // 2-token sliding decode (matches stream_token  -
                        // avoids O(N²) cumulative decode for moondream's
                        // PLD-heavy vision path).
                        for &tok in &accepted_tokens {
                            if tok == eos_token_id || token_count >= max_tokens {
                                break 'stream_loop;
                            }
                            generated_token_ids.push(tok);

                            let chunk_text =
                                incremental_chunk_text(&stream_tokenizer, &generated_token_ids);
                            sent_text_len += chunk_text.len();

                            if stop_tracker.is_active() {
                                let is_stop =
                                    stop_tracker.push_and_check(&chunk_text, &stop_sequences);
                                if tx.blocking_send(Ok(chunk_text)).is_err() {
                                    break 'stream_loop;
                                }
                                token_count += 1;
                                if is_stop {
                                    break 'stream_loop;
                                }
                            } else {
                                if tx.blocking_send(Ok(chunk_text)).is_err() {
                                    break 'stream_loop;
                                }
                                token_count += 1;
                            }
                        }

                        next_token = next_from_verify;
                        // Restart compute window for the next iter.
                        iter_compute_start = Some(std::time::Instant::now());
                    }
                } else {
                    // --- CUDA graph decode step (streaming) ------------
                    // Mirrors the non-stream loop: token 1 warmup (primes
                    // stable buffers), token 2 capture (shared
                    // `capture_decode_graph`) + first probation, tokens 3+
                    // replay with probation for the first replays. Yields
                    // `Some((token, dev_token))` when it produced this
                    // token; `None` falls through to the PLD/eager path
                    // (any failure also disables graph mode for the rest
                    // of the request, so PLD resumes naturally).
                    #[cfg(feature = "cuda")]
                    let graph_step: Option<(u32, Option<Tensor>)> = 'graph_blk: {
                        if !graph_ok || graph_disabled_this_request {
                            break 'graph_blk None;
                        }
                        let Some(stream) = graph_stream.as_ref() else {
                            break 'graph_blk None;
                        };
                        let mut guard = model_state.blocking_lock();
                        let state = match guard.as_mut() {
                            Some(s) => s,
                            None => {
                                let _ = tx.blocking_send(Err("Model unloaded".to_string()));
                                return;
                            }
                        };

                        // Per-token graph state refresh (rope/mask/cur_pos_dev)
                        // + embedding into the stable hidden buffer. An error
                        // here (e.g. seq outgrew the captured Q8 ceiling)
                        // tears the graph down; the request continues eagerly.
                        let x = match next_token_dev.take() {
                            Some(t) => t,
                            None => match Tensor::new(&[next_token], &Device::Cpu)
                                .and_then(|t| t.unsqueeze(0))
                            {
                                Ok(t) => t,
                                Err(e) => {
                                    let _ = tx.blocking_send(Err(format!("Tensor creation: {e}")));
                                    return;
                                }
                            },
                        };
                        if let Err(e) = state.model.update_graph_state(pos) {
                            warn!("update_graph_state failed at pos={pos}: {e}. Disabling graph for rest of request.");
                            cuda_graph = None;
                            graph_logits = None;
                            graph_disabled_this_request = true;
                            state.model.invalidate_graph_state();
                            stream.context().free_capture_arena();
                            break 'graph_blk None;
                        }
                        if let Err(e) = state.model.embed_for_graph(&x) {
                            warn!("embed_for_graph failed at pos={pos}: {e}. Disabling graph for rest of request.");
                            cuda_graph = None;
                            graph_logits = None;
                            graph_disabled_this_request = true;
                            state.model.invalidate_graph_state();
                            stream.context().free_capture_arena();
                            break 'graph_blk None;
                        }

                        // Streaming greedy/temperature sampling from a graph
                        // logits tensor (parity with the eager branch below).
                        macro_rules! sample_graph {
                            ($logits:expr) => {{
                                let logits_1d = match $logits.squeeze(0) {
                                    Ok(s) => s,
                                    Err(e) => {
                                        let _ = tx.blocking_send(Err(format!("Squeeze: {e}")));
                                        return;
                                    }
                                };
                                if temperature == 0.0 && !logits_processor.needs_host() {
                                    match gpu_sample_returning_tensor(
                                        &logits_1d,
                                        &recent_tokens,
                                        repeat_penalty,
                                        repeat_last_n,
                                    ) {
                                        Ok((t, dev_t)) => (t, dev_t),
                                        Err(e) => {
                                            let _ = tx.blocking_send(Err(format!("Sampling: {e}")));
                                            return;
                                        }
                                    }
                                } else {
                                    let logits_cpu = match logits_1d.device() {
                                        Device::Cpu => logits_1d,
                                        _ => match logits_1d.to_device(&Device::Cpu) {
                                            Ok(l) => l,
                                            Err(e) => {
                                                let _ = tx.blocking_send(Err(format!(
                                                    "Device transfer: {e}"
                                                )));
                                                return;
                                            }
                                        },
                                    };
                                    let penalized = match apply_repeat_penalty(
                                        &logits_cpu,
                                        &recent_tokens,
                                        repeat_penalty,
                                        repeat_last_n,
                                    ) {
                                        Ok(l) => l,
                                        Err(e) => {
                                            let _ = tx
                                                .blocking_send(Err(format!("Repeat penalty: {e}")));
                                            return;
                                        }
                                    };
                                    match logits_processor.sample(&penalized) {
                                        Ok(t) => (t, None),
                                        Err(e) => {
                                            let _ = tx.blocking_send(Err(format!("Sampling: {e}")));
                                            return;
                                        }
                                    }
                                }
                            }};
                        }

                        // -- Replay (tokens 3+) -------------------------
                        if let (Some(graph), Some(logits_ref)) =
                            (cuda_graph.as_mut(), graph_logits.as_ref())
                        {
                            let _ = stream.synchronize();
                            match graph.launch() {
                                Ok(()) => {
                                    let _ = stream.synchronize();
                                    const GRAPH_VALIDATIONS: usize = 3;
                                    if graph_validations < GRAPH_VALIDATIONS {
                                        let rep_argmax: Option<u32> = logits_ref
                                            .flatten_all()
                                            .ok()
                                            .and_then(|t| t.argmax(0).ok())
                                            .and_then(|t| t.to_scalar::<u32>().ok());
                                        match state.model.forward_from_hidden() {
                                            Ok(eag) => {
                                                let _ = stream.synchronize();
                                                let eag_argmax: Option<u32> = eag
                                                    .flatten_all()
                                                    .ok()
                                                    .and_then(|t| t.argmax(0).ok())
                                                    .and_then(|t| t.to_scalar::<u32>().ok());
                                                if rep_argmax.is_some() && rep_argmax == eag_argmax
                                                {
                                                    graph_validations += 1;
                                                } else {
                                                    warn!("CUDA graph replay!=eager at pos={pos} (argmax {:?} vs {:?}); discarding graph, eager decode for rest of request",
                                                        rep_argmax, eag_argmax);
                                                    cuda_graph = None;
                                                    graph_logits = None;
                                                    graph_disabled_this_request = true;
                                                    state.model.invalidate_graph_state();
                                                    stream.context().free_capture_arena();
                                                    pos += 1;
                                                    // The eager logits are the trusted ones.
                                                    break 'graph_blk Some(sample_graph!(eag));
                                                }
                                            }
                                            Err(e) => {
                                                warn!("graph probation eager forward failed at pos={pos}: {e}; trusting replay");
                                                graph_validations += 1;
                                            }
                                        }
                                    }
                                    pos += 1;
                                    // Host KV bookkeeping for the device-side append.
                                    state.model.sync_kv_len_for_graph(pos);
                                    break 'graph_blk Some(sample_graph!(logits_ref));
                                }
                                Err(e) => {
                                    warn!(
                                        "Graph replay failed: {e}, falling back to normal forward"
                                    );
                                    cuda_graph = None;
                                    graph_logits = None;
                                    graph_disabled_this_request = true;
                                    state.model.invalidate_graph_state();
                                    stream.context().free_capture_arena();
                                    break 'graph_blk None;
                                }
                            }
                        }

                        // -- Warmup (first decode iteration) ------------
                        if !graph_warmed {
                            match state.model.forward_from_hidden() {
                                Ok(logits) => {
                                    graph_warmed = true;
                                    pos += 1;
                                    break 'graph_blk Some(sample_graph!(logits));
                                }
                                Err(e) => {
                                    warn!("graph warmup forward failed at pos={pos}: {e}; eager decode for rest of request");
                                    graph_disabled_this_request = true;
                                    state.model.invalidate_graph_state();
                                    break 'graph_blk None;
                                }
                            }
                        }

                        // -- Capture (iteration after warmup) -----------
                        if !graph_capture_attempted {
                            graph_capture_attempted = true;
                            // KV safety is DYNAMIC (a failed Q8 append drops
                            // the cache mid-request); re-check at capture.
                            if !state.model.kv_state_graph_safe() {
                                warn!("KV state no longer graph-safe at pos={pos}; skipping graph capture (eager decode for rest of request)");
                                graph_disabled_this_request = true;
                                state.model.invalidate_graph_state();
                                break 'graph_blk None;
                            }
                            if let Err(e) = stream.synchronize() {
                                warn!(
                                    "sync before capture failed at pos={pos}: {e:?}; eager decode"
                                );
                                graph_disabled_this_request = true;
                                state.model.invalidate_graph_state();
                                break 'graph_blk None;
                            }
                            match capture_decode_graph(
                                &mut state.model,
                                stream,
                                pos,
                                /*needs_split=*/ false,
                            ) {
                                Some((g, logits)) => {
                                    // A capture only RECORDS - launch once so
                                    // this token's forward actually executes,
                                    // then probation #1 vs an eager forward.
                                    match g.launch() {
                                        Ok(()) => {
                                            let _ = stream.synchronize();
                                            let rep_argmax: Option<u32> = logits
                                                .flatten_all()
                                                .ok()
                                                .and_then(|t| t.argmax(0).ok())
                                                .and_then(|t| t.to_scalar::<u32>().ok());
                                            match state.model.forward_from_hidden() {
                                                Ok(eag) => {
                                                    let _ = stream.synchronize();
                                                    let eag_argmax: Option<u32> = eag
                                                        .flatten_all()
                                                        .ok()
                                                        .and_then(|t| t.argmax(0).ok())
                                                        .and_then(|t| t.to_scalar::<u32>().ok());
                                                    if rep_argmax.is_some()
                                                        && rep_argmax == eag_argmax
                                                    {
                                                        graph_validations += 1;
                                                        pos += 1;
                                                        state.model.sync_kv_len_for_graph(pos);
                                                        state.model.mark_graph_captured();
                                                        let out = sample_graph!(logits);
                                                        graph_logits = Some(logits);
                                                        cuda_graph = Some(g);
                                                        break 'graph_blk Some(out);
                                                    }
                                                    warn!("CUDA graph capture-launch!=eager at pos={pos} (argmax {:?} vs {:?}); discarding graph, eager decode for rest of request",
                                                        rep_argmax, eag_argmax);
                                                    drop(g);
                                                    graph_disabled_this_request = true;
                                                    state.model.invalidate_graph_state();
                                                    stream.context().free_capture_arena();
                                                    pos += 1;
                                                    break 'graph_blk Some(sample_graph!(eag));
                                                }
                                                Err(e) => {
                                                    warn!("graph probation eager forward failed at pos={pos}: {e}; trusting captured graph");
                                                    graph_validations += 1;
                                                    pos += 1;
                                                    state.model.sync_kv_len_for_graph(pos);
                                                    state.model.mark_graph_captured();
                                                    let out = sample_graph!(logits);
                                                    graph_logits = Some(logits);
                                                    cuda_graph = Some(g);
                                                    break 'graph_blk Some(out);
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            warn!("Captured graph first launch failed at pos={pos}: {e}; eager decode for rest of request");
                                            drop(g);
                                            graph_disabled_this_request = true;
                                            match state.model.forward_from_hidden() {
                                                Ok(l) => {
                                                    pos += 1;
                                                    state.model.invalidate_graph_state();
                                                    stream.context().free_capture_arena();
                                                    break 'graph_blk Some(sample_graph!(l));
                                                }
                                                Err(e2) => {
                                                    warn!("eager forward after failed graph launch also failed: {e2}");
                                                    state.model.invalidate_graph_state();
                                                    stream.context().free_capture_arena();
                                                    break 'graph_blk None;
                                                }
                                            }
                                        }
                                    }
                                }
                                None => {
                                    // Capture failed - a capture only RECORDS,
                                    // so the forward never executed. Compute
                                    // this token eagerly on the primed buffers,
                                    // then decode the rest on the plain path.
                                    graph_disabled_this_request = true;
                                    match state.model.forward_from_hidden() {
                                        Ok(l) => {
                                            pos += 1;
                                            state.model.invalidate_graph_state();
                                            stream.context().free_capture_arena();
                                            break 'graph_blk Some(sample_graph!(l));
                                        }
                                        Err(e) => {
                                            warn!("eager forward after failed capture also failed: {e}");
                                            state.model.invalidate_graph_state();
                                            stream.context().free_capture_arena();
                                            break 'graph_blk None;
                                        }
                                    }
                                }
                            }
                        }

                        // Graph active but nothing produced (shouldn't be
                        // reachable) - fall through to the eager path.
                        break 'graph_blk None;
                    };
                    record_logprobs(&stream_tokenizer, &mut logits_processor, &mut logprob_buf);
                    #[cfg(feature = "cuda")]
                    if let Some((tok, dev_tok)) = graph_step {
                        next_token = tok;
                        next_token_dev = dev_tok;
                        continue 'stream_loop;
                    }

                    // --- PLD or normal streaming forward ---------------
                    // Adaptive PLD: skip if in cooldown or disabled, or if a
                    // prior forward_all error put PLD into a hard-disable
                    // state (e.g. standard_attention mask bug for some
                    // architectures with cached KV).
                    // PLD long-ctx gate (CPU AND GPU): see the non-stream path - past this
                    // ctx the n-gram accept rate collapses so the per-position verify GEMVs
                    // are a net loss on both devices (qwen3/deepcoder 2.5K loss->tie once
                    // gated). (Proper fix = cpu_q8_kv::attention_multi.)
                    let pld_active =
                        pld_enabled && pld_cooldown == 0 && pos <= pld_max_ctx_threshold();
                    pld_cooldown = pld_cooldown.saturating_sub(1);
                    // Spec-decode: draft K from the drafter model when active,
                    // else the n-gram PLD draft. Verify below re-samples every position.
                    const SPEC_K: usize = 4;
                    let draft: Vec<u32> = if spec_on {
                        let d = spec_engine.draft_k(
                            next_token,
                            SPEC_K,
                            pos,
                            &recent_tokens,
                            repeat_penalty,
                            repeat_last_n,
                        );
                        if d.is_empty() {
                            spec_on = false;
                        }
                        d
                    } else if pld_active {
                        pld_cache.lookup()
                    } else {
                        Vec::new()
                    };

                    let pld_result: Option<(Vec<u32>, u32, usize, Option<Tensor>)> = if draft.len()
                        >= 2
                        && token_count + 1 + draft.len() <= max_tokens
                    {
                        // --- PLD speculative verification ---------------
                        let k = draft.len();

                        let mut guard = model_state.blocking_lock();
                        let state = match guard.as_mut() {
                            Some(s) => s,
                            None => {
                                let _ = tx.blocking_send(Err("Model unloaded".into()));
                                break;
                            }
                        };

                        // Build [next_token, draft_0, ..., draft_{k-1}]
                        let mut input = Vec::with_capacity(k + 1);
                        input.push(next_token);
                        input.extend_from_slice(&draft);
                        let x = match Tensor::new(input.as_slice(), &device_clone)
                            .and_then(|t| t.unsqueeze(0))
                        {
                            Ok(t) => t,
                            Err(e) => {
                                let _ = tx.blocking_send(Err(format!("PLD tensor: {e}")));
                                return;
                            }
                        };
                        let logits_res = state.model.forward_all(&x, pos);
                        if let Err(ref e) = logits_res {
                            // Some architectures hit a mask broadcast bug in multi-token
                            // attention with cached KV. Permanently disable PLD for this
                            // stream and fall through to single-token forward.
                            tracing::warn!("PLD forward_all failed, disabling PLD: {}", e);
                            pld_cooldown = u32::MAX;
                        }
                        if let Ok(logits) = logits_res {
                            pld_stats_drafted += k as u64;

                            // Commit-delayed verification loop (shared:
                            // decode_step::pld_verify_commit).
                            // Temperature=0: GPU argmax-with-penalty, also
                            // captures the device-resident U32 [1,1] tensor so
                            // the next-iter forward can skip the Tensor::new
                            // H->D round-trip (Path B step 3 - same bit-exact
                            // contract as the non-PLD greedy site).
                            // Temperature>0: host CPU sampler path.
                            let mut committed: Vec<u32> = Vec::new();
                            let outcome = pld_verify_commit(
                                &logits,
                                &draft,
                                &mut committed,
                                &mut recent_tokens,
                                &mut pld_cache,
                                pld_enabled,
                                eos_token_id,
                                &[],
                                /* committed_base */ token_count,
                                max_tokens,
                                &mut |row, recent| {
                                    if temperature == 0.0 {
                                        #[cfg(feature = "cuda")]
                                        {
                                            gpu_sample_returning_tensor(
                                                row,
                                                recent,
                                                repeat_penalty,
                                                repeat_last_n,
                                            )
                                        }
                                        #[cfg(not(feature = "cuda"))]
                                        {
                                            let penalized = apply_repeat_penalty(
                                                row,
                                                recent,
                                                repeat_penalty,
                                                repeat_last_n,
                                            )?;
                                            logits_processor.sample(&penalized).map(|t| (t, None))
                                        }
                                    } else {
                                        let row_cpu = match row.device() {
                                            Device::Cpu => row.clone(),
                                            _ => row.to_device(&Device::Cpu)?,
                                        };
                                        let penalized = apply_repeat_penalty(
                                            &row_cpu,
                                            recent,
                                            repeat_penalty,
                                            repeat_last_n,
                                        )?;
                                        logits_processor.sample(&penalized).map(|t| (t, None))
                                    }
                                },
                            );
                            let outcome = match outcome {
                                Ok(o) => o,
                                Err(e) => {
                                    let _ = tx.blocking_send(Err(format!("PLD verify: {e}")));
                                    return;
                                }
                            };
                            let accepted_drafts = outcome.accepted_drafts;

                            pld_stats_accepted += accepted_drafts as u64;
                            // Window-rate cooldown synced with /api/generate
                            // path (shared: decode_step::pld_window_update).
                            pld_window_update(
                                &mut pld_window_drafted,
                                &mut pld_window_accepted,
                                &mut pld_cooldown,
                                k,
                                accepted_drafts,
                            );
                            let n_committed = 1 + accepted_drafts;
                            let new_pos = pos + n_committed;

                            if n_committed < k + 1 {
                                state.model.trim_kv(new_pos);
                            }
                            // Spec-decode: drafter KV lockstep (shared:
                            // decode_step::spec_draft_lockstep).
                            if spec_on {
                                spec_draft_lockstep(&spec_engine, &draft, accepted_drafts, new_pos);
                            }

                            // Path B3 carry-through: when the FINAL sampled
                            // token came from gpu_sample_returning_tensor
                            // (temperature=0 path), forward the device-
                            // resident U32 tensor so the next non-PLD iter
                            // can skip Tensor::new H->D round-trip.
                            drop(guard); // lock released
                            Some((
                                committed,
                                outcome.next_token,
                                new_pos,
                                outcome.next_token_dev,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some((committed_tokens, new_next, new_pos, new_next_dev)) = pld_result {
                        // Stream committed tokens (next_token streams at top of next iter).
                        // Cumulative-decode to preserve leading spaces.
                        for &tok in &committed_tokens {
                            if tok == eos_token_id || token_count >= max_tokens {
                                break 'stream_loop;
                            }
                            generated_token_ids.push(tok);
                            let cumulative = stream_tokenizer
                                .decode(&generated_token_ids, skip_special_tokens)
                                .unwrap_or_default();
                            let chunk_text = if cumulative.len() >= sent_text_len {
                                // Char-boundary-safe delta: a multi-token commit (PLD /
                                // spec-decode commits several tokens at once) can COMPLETE a
                                // multi-byte char (e.g. an emoji) whose bytes differ from the
                                // previous step's partial decode, so `sent_text_len` can land
                                // INSIDE a char in the new `cumulative` - slicing there panics
                                // ("byte index N is not a char boundary"). Back off to the
                                // nearest char boundary <= sent_text_len before slicing.
                                let mut start = sent_text_len;
                                while start > 0 && !cumulative.is_char_boundary(start) {
                                    start -= 1;
                                }
                                cumulative[start..].to_string()
                            } else {
                                // rewound - `sent_text_len = cumulative.len()` below covers it
                                cumulative.clone()
                            };
                            sent_text_len = cumulative.len();

                            if stop_tracker.is_active() {
                                let is_stop =
                                    stop_tracker.push_and_check(&chunk_text, &stop_sequences);
                                if tx.blocking_send(Ok(chunk_text)).is_err() {
                                    break 'stream_loop;
                                }
                                token_count += 1;
                                if is_stop {
                                    break 'stream_loop;
                                }
                            } else {
                                if tx.blocking_send(Ok(chunk_text)).is_err() {
                                    break 'stream_loop;
                                }
                                token_count += 1;
                            }
                        }

                        next_token = new_next;
                        pos = new_pos;
                        next_token_dev = new_next_dev;
                    } else {
                        // --- Normal single-token forward ---------------
                        let t_baseline = Instant::now();
                        let (new_next_token, new_pos, new_next_dev) = {
                            let mut guard = model_state.blocking_lock();
                            let state = match guard.as_mut() {
                                Some(s) => s,
                                None => {
                                    let _ = tx.blocking_send(Err("Model unloaded".to_string()));
                                    break;
                                }
                            };

                            // Path B step 3: prefer the prior-iter's
                            // device-resident U32 tensor when available
                            // (skips H->D round-trip of ~50µs/token).
                            // Falls back to host-build if not available
                            // (first iter, after PLD, temperature > 0).
                            let x = match next_token_dev.take() {
                                Some(t) => t,
                                None => match Tensor::new(&[next_token], &device_clone)
                                    .and_then(|t| t.unsqueeze(0))
                                {
                                    Ok(t) => t,
                                    Err(e) => {
                                        let _ =
                                            tx.blocking_send(Err(format!("Tensor creation: {e}")));
                                        return;
                                    }
                                },
                            };
                            let logits = match state.model.forward(&x, pos) {
                                Ok(l) => l,
                                Err(e) => {
                                    let _ = tx.blocking_send(Err(format!(
                                        "Forward failed at pos={}: {e}",
                                        pos
                                    )));
                                    return;
                                }
                            };

                            // Greedy fast path: when temperature is 0 the
                            // sampler is just argmax-with-penalty.
                            // gpu_sample_returning_tensor also produces
                            // the U32 [1,1] device tensor for the next
                            // iter's forward (Path B step 3) - bit-exact
                            // equivalent to Tensor::new(&[tok], dev).
                            let (tok, dev_next) = if temperature == 0.0 {
                                let logits_1d = match logits.squeeze(0) {
                                    Ok(s) => s,
                                    Err(e) => {
                                        let _ = tx.blocking_send(Err(format!("Squeeze: {e}")));
                                        return;
                                    }
                                };
                                #[cfg(feature = "cuda")]
                                {
                                    match gpu_sample_returning_tensor(
                                        &logits_1d,
                                        &recent_tokens,
                                        repeat_penalty,
                                        repeat_last_n,
                                    ) {
                                        Ok((t, dev_t)) => (t, dev_t),
                                        Err(e) => {
                                            let _ = tx.blocking_send(Err(format!("Sampling: {e}")));
                                            return;
                                        }
                                    }
                                }
                                #[cfg(not(feature = "cuda"))]
                                {
                                    match gpu_sample(
                                        &logits_1d,
                                        &recent_tokens,
                                        repeat_penalty,
                                        repeat_last_n,
                                        &mut logits_processor,
                                    ) {
                                        Ok(t) => (t, None),
                                        Err(e) => {
                                            let _ = tx.blocking_send(Err(format!("Sampling: {e}")));
                                            return;
                                        }
                                    }
                                }
                            } else {
                                let logits_cpu = match logits.device() {
                                    Device::Cpu => logits,
                                    _ => match logits.to_device(&Device::Cpu) {
                                        Ok(l) => l,
                                        Err(e) => {
                                            let _ = tx.blocking_send(Err(format!(
                                                "Device transfer: {e}"
                                            )));
                                            return;
                                        }
                                    },
                                };
                                let squeezed = match logits_cpu.squeeze(0) {
                                    Ok(s) => s,
                                    Err(e) => {
                                        let _ = tx.blocking_send(Err(format!("Squeeze: {e}")));
                                        return;
                                    }
                                };
                                let penalized = match apply_repeat_penalty(
                                    &squeezed,
                                    &recent_tokens,
                                    repeat_penalty,
                                    repeat_last_n,
                                ) {
                                    Ok(l) => l,
                                    Err(e) => {
                                        let _ =
                                            tx.blocking_send(Err(format!("Repeat penalty: {e}")));
                                        return;
                                    }
                                };
                                match logits_processor.sample(&penalized) {
                                    Ok(t) => (t, None),
                                    Err(e) => {
                                        let _ = tx.blocking_send(Err(format!("Sampling: {e}")));
                                        return;
                                    }
                                }
                            };

                            if token_count < 3 {
                                trace!("✅ Token {}: id={} pos={}", token_count, tok, pos + 1);
                            }

                            (tok, pos + 1, dev_next)
                        };
                        let baseline_ms = t_baseline.elapsed().as_secs_f64() * 1000.0;

                        // Feed calibrator during baseline phase
                        #[cfg(feature = "opencl")]
                        if let Some(ref mut cal) = calibrator {
                            cal.record_baseline(baseline_ms);
                        }

                        next_token = new_next_token;
                        pos = new_pos;
                        next_token_dev = new_next_dev;
                    }
                }
            }

            if pld_stats_drafted > 0 {
                let accept_rate = pld_stats_accepted as f64 / pld_stats_drafted as f64;
                info!(
                    "📊 PLD stats: {}/{} drafts accepted ({:.1}%)",
                    pld_stats_accepted,
                    pld_stats_drafted,
                    accept_rate * 100.0
                );
            }
            info!("🏁 Streaming generation complete: {} tokens", token_count);
            if !generated_token_ids.is_empty() && generated_token_ids.len() <= 50 {
                debug!("Generated {} token IDs", generated_token_ids.len());
            }

            // Close any still-open compute window (last iter's work that
            // ended via break / EOS before reaching the next stream_token).
            if let Some(t) = iter_compute_start.take() {
                compute_ns += t.elapsed().as_nanos() as u64;
            }
            // Publish compute-only timing for the HTTP handler. The
            // stream final NDJSON chunk uses these so assay computes
            // tok/s symmetrically with Ollama's kernel-only eval_duration.
            let total_duration_ns = stream_start.elapsed().as_nanos() as u64;
            let prompt_eval_duration_ns = match prefill_end {
                Some(end) => end.duration_since(stream_start).as_nanos() as u64,
                None => 0u64,
            };
            if let Ok(mut slot) = stream_stats_slot.lock() {
                if !logprob_buf.is_empty() {
                    if let Ok(mut q) = logprob_queue.lock() {
                        q.append(&mut logprob_buf);
                    }
                }
                let finish_reason = if let Some(hit) = stop_tracker.matched() {
                    FinishReason::StopSequence(hit.to_string())
                } else if next_token == eos_token_id {
                    FinishReason::Eos
                } else if token_count >= max_tokens {
                    FinishReason::MaxTokens
                } else {
                    FinishReason::Disconnect
                };
                *slot = Some(StreamStats {
                    eval_count: token_count as u64,
                    eval_duration_ns: compute_ns,
                    prompt_eval_count: prompt_tokens.len().saturating_sub(reused_prompt_tokens)
                        as u64,
                    prompt_eval_duration_ns,
                    total_duration_ns,
                    cached_prompt_tokens: reused_prompt_tokens as u64,
                    finish_reason,
                    context_tokens: {
                        let mut full = prompt_tokens.clone();
                        full.extend_from_slice(&generated_token_ids);
                        full
                    },
                });
            }

            // -- Session state update + global resident-KV snapshot ----------
            // Snapshot the cache-representative token sequence so the next request
            // can skip redundant prefill: under the explicit session_id when present,
            // AND under GLOBAL_PROMPT_CACHE_KEY (mirroring the single resident KV) so a
            // sessionless request can prefix-reuse it (ollama parity). In-memory only  -
            // never persisted (see project_prompt_cache_privacy). The GLOBAL mirror is
            // skipped when LOKEN_NO_GLOBAL_PROMPT_CACHE=1 (multi-tenant privacy).
            {
                let model_name = {
                    let g = model_state.blocking_lock();
                    g.as_ref().map(|s| s.name.clone())
                };
                if let Some(model_name) = model_name {
                    let mut full = prompt_tokens.clone();
                    full.extend_from_slice(&generated_token_ids);
                    // Capture image-prefix metadata for vision sessions so a
                    // future same-image request can skip the 730-position
                    // image-embed prefill.
                    let (image_hash, image_prefix_len) = {
                        let g = model_state.blocking_lock();
                        match g.as_ref() {
                            Some(s) if s.model.is_vision_model() => {
                                let hash = s.image_embed_cache.as_ref().map(|(h, _)| *h);
                                let prefix = s
                                    .image_embeds
                                    .as_ref()
                                    .and_then(|t| t.dim(1).ok())
                                    .map(|n| 1 + n) // 1 BOS + n image embeds
                                    .unwrap_or(0);
                                (hash, prefix)
                            }
                            _ => (None, 0),
                        }
                    };
                    let mut sess_guard = sessions.blocking_lock();
                    if let Some(sid) = session_id.as_ref() {
                        sess_guard.insert(
                            sid.clone(),
                            SessionState {
                                tokens: full.clone(),
                                model_name: model_name.clone(),
                                kv_len: pos,
                                image_hash,
                                image_prefix_len,
                            },
                        );
                    }
                    {
                        // Mirror the resident KV for sessionless prefix reuse.
                        sess_guard.insert(
                            GLOBAL_PROMPT_CACHE_KEY.to_string(),
                            SessionState {
                                tokens: full.clone(),
                                model_name,
                                kv_len: pos,
                                image_hash,
                                image_prefix_len,
                            },
                        );
                    }
                    // Never two locks at once: the sessions table is released before the
                    // model is taken.
                    drop(sess_guard);
                    if kv_snapshots > 0 && image_hash.is_none() {
                        let mut g = model_state.blocking_lock();
                        if let Some(s) = g.as_mut() {
                            match s.model.snapshot_kv(
                                full.clone(),
                                pos,
                                kv_snapshots,
                                kv_snapshot_budget,
                            ) {
                                Err(e) => tracing::warn!("kv snapshot skipped: {e}"),
                                Ok(()) => {
                                    if let Some(store) = kv_disk.clone() {
                                        spawn_persist(model_state.clone(), store, full.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(rx)
    }

    /// Get reference to config
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Stable fingerprint of the loaded tokenizer's serialized form.
    /// Two engines with the same fingerprint produce identical token IDs
    /// for any given input - required for speculative decoding to be
    /// correct. Returns `None` if the model isn't loaded.
    /// How much of this prompt this node already holds, in tokens.
    ///
    /// Answered HERE because this is the only place that can: the tokeniser belongs to the
    /// loaded model, and the cache is this process's. A peer asking the question gets a figure
    /// derived the same way the local reuse path derives it, so a routing decision and the
    /// prefill that follows it cannot disagree.
    ///
    /// `None` when the model is not resident - the caller must read that as "holds nothing",
    /// never as zero-because-measured.
    pub async fn cached_prompt_tokens(&self, model_name: &str, prompt: &str) -> Option<usize> {
        let guard = self.model_state.lock().await;
        let state = guard.as_ref()?;
        let encoding = state.tokenizer.encode(prompt, true).ok()?;
        let tokens: Vec<u32> = encoding.get_ids().to_vec();
        let sessions = self.sessions.lock().await;
        let Some(s) = sessions.get(GLOBAL_PROMPT_CACHE_KEY) else {
            tracing::debug!("cluster prefix: no resident session to compare against");
            return Some(0);
        };
        // The session records the engine's own name for the model; a peer asks with the name
        // it was given. They are the same string today and a mismatch would silently answer
        // "nothing cached" forever, so it is logged rather than assumed.
        if s.model_name != model_name {
            tracing::debug!(
                "cluster prefix: resident session is for '{}', asked about '{model_name}'",
                s.model_name
            );
            return Some(0);
        }
        if s.image_hash.is_some() {
            return Some(0);
        }
        Some(reusable_prefix(&s.tokens, &tokens, s.kv_len))
    }

    /// What has to agree between a target and its drafter: the id of every ordinary piece.
    /// Nothing else. A drafter never tokenises text - it is fed the target's ids and answers
    /// with ids - so the merges, the pre-tokeniser and the BOS/EOS post-processing are the
    /// target's business alone, and the added tokens are where a distil renames its controls:
    /// deepseek-r1:70b and llama3.2:1b share 128000 of 128256 pieces, and the 256 they do not
    /// are `<|begin_of_text|>`, `<think>` and their kind, which no drafter proposes from prose.
    /// Hashing the whole file refused that pair as a different vocabulary. The caller still
    /// requires equal vocab sizes, so an id that exists on one side only cannot pass.
    pub async fn tokenizer_fingerprint(&self) -> Option<u64> {
        let guard = self.model_state.lock().await;
        let state = guard.as_ref()?;
        let tok = &state.tokenizer;
        let added: std::collections::HashSet<u32> =
            tok.get_added_tokens_decoder().keys().copied().collect();
        let mut pieces: Vec<(u32, String)> = tok
            .get_vocab(false)
            .into_iter()
            .filter(|(_, id)| !added.contains(id))
            .map(|(piece, id)| (id, piece))
            .collect();
        pieces.sort_unstable();
        let mut h = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&pieces, &mut h);
        Some(std::hash::Hasher::finish(&h))
    }
    /// Snapshot of the loaded model's vocab size - used by the spec
    /// decode caller to pre-size logits buffers and verify dim parity.
    pub async fn vocab_size(&self) -> Option<usize> {
        let guard = self.model_state.lock().await;
        guard.as_ref().map(|s| s.vocab_size)
    }
}
