//! Part of `impl LlmEngine`, split out of the parent module.
//!
//! Rust lets one inherent impl live in several modules of a crate, so this is the
//! same impl - only the file changed. Methods that were private are `pub(super)`
//! here, which is the reach they had when they sat beside their callers.

use super::*;

impl LlmEngine {
    /// Run a single-token forward at `pos` and return the logits and
    /// the next token sampled greedily (temperature=0). Used by the
    /// spec-decode path to drive the draft model. Holds the model_state
    /// lock for the duration of the forward - caller must not interleave
    /// with another generate on the same engine.
    pub async fn draft_step(
        &self,
        token: u32,
        pos: usize,
    ) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<(Vec<f32>, u32)> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Draft model not loaded"))?;
            let device = state.device.clone();
            let x = Tensor::new(&[token], &device)?.unsqueeze(0)?;
            let logits = state.model.forward(&x, pos)?;
            let logits_cpu = match logits.device() {
                Device::Cpu => logits,
                _ => logits.to_device(&Device::Cpu)?,
            };
            let logits_1d = logits_cpu
                .squeeze(0)?
                .to_dtype(crate::tensor::DType::F32)?
                .to_vec1::<f32>()?;
            // greedy argmax
            let (best, _) = logits_1d.iter().enumerate().fold(
                (0usize, f32::NEG_INFINITY),
                |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                },
            );
            Ok((logits_1d, best as u32))
        })
        .await;
        match inner {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }

    /// Greedy speculative-decode streaming generate. Drafts K tokens via
    /// `draft`, verifies them via this engine's `forward_all`, accepts
    /// the longest matching greedy prefix, emits accepted tokens plus
    /// one bonus from the target. Trims both KV caches on partial
    /// acceptance to keep them aligned.
    ///
    /// Correctness gate: only valid for greedy decoding (temperature=0).
    /// For temperature>0 we'd need rejection sampling using both draft
    /// and target probabilities - not implemented in this slice.
    pub async fn generate_stream_with_draft(
        &self,
        prompt: String,
        params: GenerationParams,
        draft: Arc<LlmEngine>,
        k: usize,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String, String>>, Box<dyn std::error::Error>>
    {
        let target_state = self.model_state.clone();
        let config = self.config.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, String>>(64);

        // Reset draft KV before we kick off (target's reset happens via
        // forward(_, 0) inside our prefill below).
        draft.draft_reset().await?;

        // Spawn an async task; spec loop must alternate between target and
        // draft engines, each behind its own model_state mutex.
        let target_engine = self.clone_for_spec();
        tokio::spawn(async move {
            let max_tokens = params.max_tokens.unwrap_or(config.max_tokens);
            let eos = {
                let g = target_state.lock().await;
                g.as_ref().map(|s| s.eos_token_id).unwrap_or(0)
            };

            // Tokenize.
            let prompt_tokens: Vec<u32> = {
                let g = target_state.lock().await;
                let state = match g.as_ref() {
                    Some(s) => s,
                    None => {
                        let _ = tx.send(Err("Target model not loaded".into())).await;
                        return;
                    }
                };
                match state.tokenizer.encode(prompt.as_str(), true) {
                    Ok(e) => with_prefix_tokens(&params, e.get_ids()),
                    Err(e) => {
                        let msg = format!("Tokenization: {e}");
                        drop(e);
                        let _ = tx.send(Err(msg)).await;
                        return;
                    }
                }
            };
            let prompt_len = prompt_tokens.len();
            tracing::info!("🎯 Spec decode: prompt={} tokens, k={}", prompt_len, k);

            // Cumulative-decode state for streaming (preserves leading
            // spaces - see stream_token closure in the unified loop).
            let mut generated_token_ids: Vec<u32> = Vec::with_capacity(max_tokens);
            let mut sent_text_len: usize = 0;

            // Prefill target: forward the whole prompt at offset 0, sample
            // greedily from the last position. After prefill, target's KV
            // holds [0, prompt_len). cur_token is sampled but NOT yet in
            // target's KV - it'll be fed back as the first verify input on
            // the next cycle.
            let mut target_pos = prompt_len;
            let cur_token_init = match target_engine
                .spec_prefill(&prompt_tokens)
                .await
                .map_err(|e| format!("Target prefill: {e}"))
            {
                Ok(t) => t,
                Err(msg) => {
                    let _ = tx.send(Err(msg)).await;
                    return;
                }
            };
            // Cumulative decode for chunk extraction.
            generated_token_ids.push(cur_token_init);
            let init_chunk = {
                let cumulative = decode_all(&target_engine, &generated_token_ids).await;
                let c = if cumulative.len() >= sent_text_len {
                    char_safe_suffix(&cumulative, sent_text_len)
                } else {
                    cumulative.clone()
                };
                sent_text_len = cumulative.len();
                c
            };
            let _ = tx.send(Ok(init_chunk)).await;

            // Prefill draft: same prompt batched. Its KV ends at prompt_len
            // too. cur_token is NOT pre-fed to draft - the spec loop's first
            // draft_step does that.
            if let Err(msg) = draft
                .spec_prefill_only(&prompt_tokens)
                .await
                .map_err(|e| format!("Draft prefill: {e}"))
            {
                let _ = tx.send(Err(msg)).await;
                return;
            }
            let mut draft_pos = prompt_len;

            let mut emitted: usize = 1;
            let mut last_tok = cur_token_init;
            let mut spec_drafts: u64 = 0;
            let mut spec_accepted: u64 = 0;
            let max_stop_len = params
                .stop_sequences
                .iter()
                .map(std::string::String::len)
                .max()
                .unwrap_or(0);
            let mut stop_suffix = String::new();
            let stop_seqs = params.stop_sequences.clone();

            'outer: while emitted < max_tokens && last_tok != eos {
                // -- Draft phase: K candidate tokens
                let mut drafts: Vec<u32> = Vec::with_capacity(k);
                let mut feed = last_tok;
                for _ in 0..k {
                    let (_, next) = match draft
                        .draft_step(feed, draft_pos)
                        .await
                        .map_err(|e| format!("Draft step: {e}"))
                    {
                        Ok(r) => r,
                        Err(msg) => {
                            let _ = tx.send(Err(msg)).await;
                            return;
                        }
                    };
                    drafts.push(next);
                    draft_pos += 1;
                    feed = next;
                }

                // -- Verify phase: target.forward_all on [last_tok, drafts...]
                let mut verify: Vec<u32> = Vec::with_capacity(k + 1);
                verify.push(last_tok);
                verify.extend_from_slice(&drafts);
                let (logits_flat, seq, vocab_size) = match target_engine
                    .target_forward_all(verify, target_pos)
                    .await
                    .map_err(|e| format!("Target verify: {e}"))
                {
                    Ok(r) => r,
                    Err(msg) => {
                        let _ = tx.send(Err(msg)).await;
                        return;
                    }
                };
                debug_assert_eq!(seq, k + 1);
                let argmax_at = |i: usize| -> u32 {
                    let row = &logits_flat[i * vocab_size..(i + 1) * vocab_size];
                    row.iter()
                        .enumerate()
                        .fold((0u32, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                            if v > bv {
                                (i as u32, v)
                            } else {
                                (bi, bv)
                            }
                        })
                        .0
                };

                // Greedy accept: target's argmax at position i must equal drafts[i].
                let mut accepted = 0usize;
                for (i, &draft) in drafts.iter().take(k).enumerate() {
                    if argmax_at(i) == draft {
                        accepted += 1;
                    } else {
                        break;
                    }
                }
                let bonus = argmax_at(accepted);
                spec_drafts += k as u64;
                spec_accepted += accepted as u64;

                // -- Emit accepted drafts + bonus, with stop-sequence + EOS checks.
                // Cumulative-decode preserves leading spaces (see init).
                // i is used both for indexing and as a value via drafts[i] further
                // down - keep the index loop here despite needless_range_loop.
                #[allow(clippy::needless_range_loop)]
                for i in 0..accepted {
                    generated_token_ids.push(drafts[i]);
                    let cumulative = decode_all(&target_engine, &generated_token_ids).await;
                    let dec = if cumulative.len() >= sent_text_len {
                        char_safe_suffix(&cumulative, sent_text_len)
                    } else {
                        cumulative.clone()
                    };
                    sent_text_len = cumulative.len();
                    if max_stop_len > 0 {
                        stop_suffix.push_str(&dec);
                        if stop_suffix.len() > max_stop_len * 2 {
                            let cut = stop_suffix.len() - max_stop_len * 2;
                            stop_suffix = stop_suffix[cut..].to_string();
                        }
                        if stop_seqs.iter().any(|s| stop_suffix.ends_with(s.as_str())) {
                            let _ = tx.send(Ok(dec)).await;
                            break 'outer;
                        }
                    }
                    if tx.send(Ok(dec)).await.is_err() {
                        tracing::debug!("🛑 Spec stream stopped - client disconnected");
                        return;
                    }
                    emitted += 1;
                    last_tok = drafts[i];
                    if emitted >= max_tokens {
                        break 'outer;
                    }
                    if last_tok == eos {
                        break 'outer;
                    }
                }

                // KV alignment: target's forward_all advanced its KV by k+1
                // positions. We're keeping the first `accepted + 1` of those
                // (the accepted drafts + the bonus). Trim back to that.
                let new_target_pos = target_pos + accepted + 1;
                if accepted < k {
                    if let Err(msg) = target_engine
                        .trim_kv(new_target_pos)
                        .await
                        .map_err(|e| format!("Target trim: {e}"))
                    {
                        let _ = tx.send(Err(msg)).await;
                        return;
                    }
                }
                target_pos = new_target_pos;

                // Draft KV is at draft_pos. Caller's accepted draft path
                // means the draft produced (drafts[..k]). After verify we
                // keep only the first `accepted + 1` positions of draft KV
                // past the original boundary.
                let new_draft_pos = target_pos; // by construction they should agree
                if new_draft_pos < draft_pos {
                    if let Err(msg) = draft
                        .trim_kv(new_draft_pos)
                        .await
                        .map_err(|e| format!("Draft trim: {e}"))
                    {
                        let _ = tx.send(Err(msg)).await;
                        return;
                    }
                }
                draft_pos = new_draft_pos;

                // Emit the bonus token (cumulative-decode for leading spaces).
                generated_token_ids.push(bonus);
                let cumulative = decode_all(&target_engine, &generated_token_ids).await;
                let dec = if cumulative.len() >= sent_text_len {
                    char_safe_suffix(&cumulative, sent_text_len)
                } else {
                    cumulative.clone()
                };
                sent_text_len = cumulative.len();
                if max_stop_len > 0 {
                    stop_suffix.push_str(&dec);
                    if stop_suffix.len() > max_stop_len * 2 {
                        let cut = stop_suffix.len() - max_stop_len * 2;
                        stop_suffix = stop_suffix[cut..].to_string();
                    }
                    if stop_seqs.iter().any(|s| stop_suffix.ends_with(s.as_str())) {
                        let _ = tx.send(Ok(dec)).await;
                        break 'outer;
                    }
                }
                if tx.send(Ok(dec)).await.is_err() {
                    tracing::debug!("🛑 Spec stream stopped - client disconnected");
                    return;
                }
                emitted += 1;
                last_tok = bonus;
                // Don't pre-feed bonus into draft. Next cycle's first
                // draft_step will feed it naturally.
            }

            tracing::info!(
                "🎯 Spec decode done: drafted={} accepted={} ({}%) emitted={}",
                spec_drafts,
                spec_accepted,
                if spec_drafts > 0 {
                    (spec_accepted * 100) / spec_drafts
                } else {
                    0
                },
                emitted
            );
        });

        Ok(rx)
    }

    /// Internal: run a target prefill and return the first sampled token.
    pub(super) async fn spec_prefill(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<u32, Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let prompt_tokens = prompt_tokens.to_vec();
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<u32> {
            let mut guard = model_state.blocking_lock();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Target model not loaded"))?;
            let device = state.device.clone();
            let x = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
            let logits = state.model.forward_prefill_chunked(&x, 0)?;
            let logits_cpu = match logits.device() {
                Device::Cpu => logits,
                _ => logits.to_device(&Device::Cpu)?,
            };
            let v = logits_cpu
                .squeeze(0)?
                .to_dtype(crate::tensor::DType::F32)?
                .to_vec1::<f32>()?;
            let (best, _) =
                v.iter()
                    .enumerate()
                    .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                        if x > bv {
                            (i, x)
                        } else {
                            (bi, bv)
                        }
                    });
            Ok(best as u32)
        })
        .await;
        match inner {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }

    /// Internal: prefill the model with prompt tokens but discard logits.
    /// Used by the draft model in spec decode to populate its KV cache.
    pub(super) async fn spec_prefill_only(
        &self,
        prompt_tokens: &[u32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let _ = self.spec_prefill(prompt_tokens).await?;
        Ok(())
    }

    /// Cheap clone for spec decode. The LlmEngine struct holds Arc'd
    /// state so .clone() is just bumping reference counts.
    pub(super) fn clone_for_spec(&self) -> Self {
        Self {
            config: self.config.clone(),
            model_state: self.model_state.clone(),
            draft_engine: self.draft_engine.clone(),
            last_error: self.last_error.clone(),
            cached_model_size: self.cached_model_size.clone(),
            sessions: self.sessions.clone(),
            last_stream_stats: self.last_stream_stats.clone(),
            embedding_model: self.embedding_model.clone(),
        }
    }
}
