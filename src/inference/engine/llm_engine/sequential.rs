//! The sequential decode path: one forward per token, every row masked before its draw.
//!
//! A request decodes here when its draw reads more than one row of logits can tell the fast
//! paths: a grammar, the tokens it already wrote (banned words, a minimum length), a mask over
//! the whole row (an allowed list, a probability floor), the special tokens kept in its text,
//! or a prefix it keeps to itself. No prompt-lookup drafts, no captured graphs, no prefix
//! reuse: the prompt is prefilled from its first token, and the records of what the resident
//! KV held are dropped, since this path rewrites it.

use super::params::{FinishReason, TokenLogprob};
use super::*;
use crate::inference::engine::decode_step::{record_logprobs, StopTracker, StreamDetokenizer};
use crate::inference::sample::token_rules::TokenRules;

/// Where a sequential decode sends what it produces as it goes.
struct Outlet<'a> {
    /// Each new piece of text; `false` when the receiver is gone.
    text: &'a mut dyn FnMut(String) -> bool,
    /// Each drawn token's log-probability, when the request asked for them.
    logprob: &'a mut dyn FnMut(TokenLogprob),
}

/// One sequential generation, under the model lock.
fn decode_sequential(
    state: &mut LoadedModelState,
    config: &InferenceConfig,
    prompt: &str,
    params: GenerationParams,
    cancel: &std::sync::atomic::AtomicBool,
    out: Outlet<'_>,
) -> AnyResult<GenerationResult> {
    if state.audio.is_some() {
        return Err(anyhow!(
            "a request with a grammar or token rules cannot carry audio"
        ));
    }
    let mut constraint = match params.grammar.as_deref() {
        Some(spec) => {
            let factory = grammar_factory_for(state)?;
            let parser = factory
                .create_parser(parse_grammar_spec(spec)?)
                .map_err(|e| anyhow!("create_parser: {e}"))?;
            info!("🎯 Grammar-constrained decode: spec={spec}");
            Some(llguidance::Constraint::new(parser))
        }
        None => None,
    };

    let encoding = super::encode_prompt(&state.tokenizer, prompt)
        .map_err(|e| anyhow!("Tokenization error: {e}"))?;
    let kv_window = effective_kv_window(
        state.context_length,
        config.context_length,
        params.context_length,
    );
    let prompt_tokens = prompt_in_window(
        request_prompt_tokens(&params, encoding.get_ids()),
        kv_window,
        params.max_tokens.unwrap_or(0),
        None,
    )
    .map_err(|e| anyhow!(e))?;
    let ResolvedGenParams {
        max_tokens,
        temperature,
        top_p,
        top_k,
        seed,
        repeat_penalty,
        repeat_last_n,
    } = resolve_gen_params(
        &params,
        &state.generation_defaults,
        kv_window,
        prompt_tokens.len(),
    );
    let params = params.with_default_stops(&state.generation_defaults.stop);
    let ends = params
        .rules
        .end_tokens(state.eos_token_id, &state.eos_token_ids_extra);
    let rules = TokenRules::new(&params.rules, &state.tokenizer, &ends);
    let skip_special = !params.rules.keep_special_tokens;
    let device = state.device.clone();

    let mut logits_processor =
        LogitsProcessor::from_sampling(seed, build_sampling_from(temperature, top_p, top_k));
    logits_processor.set_logit_bias(params.logit_bias.clone());
    logits_processor.set_top_logprobs(params.top_logprobs);
    let mut logprob_acc: Vec<TokenLogprob> = Vec::new();
    state
        .model
        .set_early_exit_threshold(params.early_exit_threshold);

    // The graphs kept from the last request go, with the buffers they read: this path's
    // single-token forwards would read their frozen rope values.
    #[cfg(feature = "gpu")]
    state.kept_graphs.clear();
    state.model.note_graphs_kept(None);
    state.model.invalidate_graph_state();
    // The prompt is written from the first row, so nothing the cache held stays usable.
    state.exact_prefill = None;
    state.model.reset_kv_from(0);

    let prefill_start = std::time::Instant::now();
    let x = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
    let mut logits = state.model.forward_prefill_chunked(&x, 0)?;
    let prompt_eval_duration = prefill_start.elapsed().as_nanos() as u64;
    let mut pos = prompt_tokens.len();

    let mut recent_tokens: Vec<u32> = prompt_tokens.clone();
    let mut generated: Vec<u32> = Vec::with_capacity(max_tokens);
    let mut stop_tracker = StopTracker::new(&params.stop_sequences);
    let mut detok = StreamDetokenizer::default();
    let mut finish = FinishReason::MaxTokens;
    let decode_start = std::time::Instant::now();

    while generated.len() < max_tokens {
        crate::request_metrics::note_decoded(generated.len());
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            finish = FinishReason::Disconnect;
            break;
        }
        let row = logits.to_device(&Device::Cpu)?.squeeze(0)?;
        let mut row = apply_repeat_penalty(&row, &recent_tokens, repeat_penalty, repeat_last_n)?
            .to_dtype(crate::tensor::DType::F32)?
            .to_vec1::<f32>()?;
        if let Some(c) = constraint.as_mut() {
            let res = c.compute_mask().map_err(|e| anyhow!("compute_mask: {e}"))?;
            if res.is_stop() {
                debug!("Grammar reached stop state at step {}", generated.len());
                finish = FinishReason::Grammar;
                break;
            }
            if let Some(mask) = res.sample_mask.as_ref() {
                for (i, l) in row.iter_mut().enumerate() {
                    if !mask.is_allowed(i as u32) {
                        *l = f32::NEG_INFINITY;
                    }
                }
            }
        }
        rules
            .apply(&mut row, &generated, temperature)
            .map_err(|e| anyhow!(e))?;
        let next_token = logits_processor.sample_logits(row)?;
        record_logprobs(&state.tokenizer, &mut logits_processor, &mut logprob_acc);
        for lp in logprob_acc.drain(..) {
            (out.logprob)(lp);
        }
        let grammar_done = match constraint.as_mut() {
            Some(c) => {
                c.commit_token(Some(next_token))
                    .map_err(|e| anyhow!("commit_token: {e}"))?
                    .stop
            }
            None => false,
        };
        // An end token ends the generation and is not part of its text.
        if ends.contains(&next_token) {
            finish = FinishReason::Eos;
            break;
        }
        generated.push(next_token);
        recent_tokens.push(next_token);

        let piece = detok.step(&state.tokenizer, &generated, skip_special);
        let stopped = stop_tracker.is_active()
            && stop_tracker.push_and_check(&piece, &params.stop_sequences)
            && rules.may_end(generated.len());
        if !piece.is_empty() && !(out.text)(piece) {
            finish = FinishReason::Disconnect;
            break;
        }
        if stopped {
            finish = FinishReason::StopSequence(stop_tracker.matched().unwrap_or_default().into());
            break;
        }
        if grammar_done {
            debug!("Grammar committed final token; stopping");
            finish = FinishReason::Grammar;
            break;
        }
        if generated.len() >= max_tokens {
            break;
        }
        let x = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        logits = state.forward(&x, pos)?;
        pos += 1;
    }
    let tail = detok.finish(&state.tokenizer, &generated, skip_special);
    if !tail.is_empty() {
        (out.text)(tail);
    }

    let eval_duration = decode_start.elapsed().as_nanos() as u64;
    let text = state
        .tokenizer
        .decode(&generated, skip_special)
        .map_err(|e| anyhow!("decode: {e}"))?;
    let mut tokens = prompt_tokens.clone();
    tokens.extend_from_slice(&generated);
    Ok(GenerationResult {
        text,
        tokens,
        prompt_eval_count: prompt_tokens.len() as u64,
        prompt_eval_duration,
        eval_count: generated.len() as u64,
        eval_duration,
        cached_prompt_tokens: 0,
        finish_reason: finish,
        logprobs: Vec::new(),
    })
}

/// The figures of a finished sequential generation, for the request metrics and the request's
/// probe. This path does not speculate.
fn record_figures(
    model_id: &str,
    queue_s: f64,
    r: &GenerationResult,
    state: &LoadedModelState,
    probe: Option<&crate::request_metrics::RequestProbe>,
) {
    crate::request_metrics::record_outcome(
        model_id,
        probe,
        crate::request_metrics::RequestOutcome {
            figures: crate::request_metrics::RequestFigures {
                queue_s,
                prefill_s: r.prompt_eval_duration as f64 / 1e9,
                decode_s: r.eval_duration as f64 / 1e9,
                prompt_tokens: r.prompt_eval_count,
                cached_prompt_tokens: 0,
                generated_tokens: r.eval_count,
            },
            kv: crate::request_metrics::KvPlace::of(
                state.model.kv_len(),
                state.model.kv_capacity(),
            ),
            ..Default::default()
        },
    );
}

impl LlmEngine {
    /// A whole generation on the sequential path.
    pub(super) async fn generate_sequential(
        &self,
        prompt: String,
        params: GenerationParams,
    ) -> Result<GenerationResult, Box<dyn std::error::Error>> {
        let model_state = self.model_state.clone();
        let sessions = self.sessions.clone();
        let config = self.config.clone();
        let cancel = params.cancel.clone().unwrap_or_default();
        let probe = params.probe.clone();
        let arrived = std::time::Instant::now();
        let inner = tokio::task::spawn_blocking(move || -> AnyResult<GenerationResult> {
            let _prefill = crate::request_metrics::bind_prefill(probe.clone());
            let mut guard = model_state.blocking_lock();
            let queue_s = arrived.elapsed().as_secs_f64();
            let state = guard
                .as_mut()
                .ok_or_else(|| anyhow!("Model not loaded. Load a model first."))?;
            sessions.blocking_lock().clear();
            let mut logprobs = Vec::new();
            let mut result = decode_sequential(
                state,
                &config,
                &prompt,
                params,
                &cancel,
                Outlet {
                    text: &mut |_| true,
                    logprob: &mut |lp| logprobs.push(lp),
                },
            )?;
            result.logprobs = logprobs;
            record_figures(&config.model_id, queue_s, &result, state, probe.as_deref());
            Ok(result)
        })
        .await;
        match inner {
            Ok(Ok(r)) => Ok(r),
            Ok(Err(e)) => Err(e.into()),
            Err(e) => Err(e.into()),
        }
    }

    /// A generation on the sequential path, its text sent as it is written. A dropped receiver
    /// ends it at the next token; the finished generation's figures go to the stream slot.
    pub(super) async fn generate_stream_sequential(
        &self,
        prompt: String,
        params: GenerationParams,
    ) -> Result<tokio::sync::mpsc::Receiver<Result<String, String>>, Box<dyn std::error::Error>>
    {
        if self.model_state.lock().await.is_none() {
            return Err("Model not loaded.".into());
        }
        let model_state = self.model_state.clone();
        let sessions = self.sessions.clone();
        let config = self.config.clone();
        let logprob_queue = self.logprob_queue.clone();
        let stats_slot = self.stream_stats_slot();
        let cancel = params.cancel.clone().unwrap_or_default();
        let probe = params.probe.clone();
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, String>>(64);
        let arrived = std::time::Instant::now();
        std::thread::spawn(move || {
            let _prefill = crate::request_metrics::bind_prefill(probe.clone());
            let mut guard = model_state.blocking_lock();
            let queue_s = arrived.elapsed().as_secs_f64();
            let Some(state) = guard.as_mut() else {
                let _ = tx.blocking_send(Err("Model not loaded".to_string()));
                return;
            };
            sessions.blocking_lock().clear();
            let run = decode_sequential(
                state,
                &config,
                &prompt,
                params,
                &cancel,
                Outlet {
                    text: &mut |piece| tx.blocking_send(Ok(piece)).is_ok(),
                    logprob: &mut |lp| {
                        if let Ok(mut q) = logprob_queue.lock() {
                            q.push(lp);
                        }
                    },
                },
            );
            match run {
                Ok(r) => {
                    record_figures(&config.model_id, queue_s, &r, state, probe.as_deref());
                    if let Ok(mut slot) = stats_slot.lock() {
                        *slot = Some(StreamStats {
                            eval_count: r.eval_count,
                            eval_duration_ns: r.eval_duration,
                            prompt_eval_count: r.prompt_eval_count,
                            prompt_eval_duration_ns: r.prompt_eval_duration,
                            total_duration_ns: arrived.elapsed().as_nanos() as u64,
                            cached_prompt_tokens: 0,
                            finish_reason: r.finish_reason,
                            context_tokens: r.tokens,
                        });
                    }
                }
                Err(e) => {
                    let _ = tx.blocking_send(Err(e.to_string()));
                }
            }
        });
        Ok(rx)
    }
}
