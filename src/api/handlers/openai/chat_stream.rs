//! The chat completions stream: the engine's tokens and failures turned into SSE events.

use super::*;
use crate::api::tool_calls::ToolFormat;
use crate::inference::engine::llm_engine::{DrawnLogprobs, LlmEngine, StreamStats};
use axum::response::sse::Event;
use std::sync::Arc;
use tokio::sync::mpsc::Receiver;

/// What the stream reads from the engine besides its tokens.
#[async_trait::async_trait]
pub(super) trait StreamSource: Send + Sync {
    /// The compute-only stats of the stream that just ended.
    fn take_last_stream_stats(&self) -> Option<StreamStats>;
    /// Tokens the loaded model makes of `text`.
    async fn count_tokens(&self, text: &str) -> Option<usize>;
}

#[async_trait::async_trait]
impl StreamSource for LlmEngine {
    fn take_last_stream_stats(&self) -> Option<StreamStats> {
        LlmEngine::take_last_stream_stats(self)
    }

    async fn count_tokens(&self, text: &str) -> Option<usize> {
        LlmEngine::count_tokens(self, text).await
    }
}

/// What a chat stream needs besides the engine's tokens.
pub(super) struct ChatRun {
    pub completion_id: String,
    pub model: String,
    pub created_at: i64,
    pub service_tier: String,
    pub wants_logprobs: bool,
    /// The log-probabilities the request's stream draws.
    pub drawn_logprobs: DrawnLogprobs,
    pub ids_as_tokens: bool,
    pub include_reasoning: bool,
    pub tools_active: bool,
    pub tool_format: ToolFormat,
    pub thinking_opened: bool,
    pub stops: Vec<String>,
    pub keep_stop: bool,
    pub echo_text: String,
    pub single_tool_call: bool,
    pub max_tokens_cap: Option<usize>,
    pub include_usage: bool,
    pub prompt_tokens: i32,
    pub probe: Arc<crate::request_metrics::RequestProbe>,
    pub energy: Option<crate::energy_report::RequestEnergy>,
    pub stream_started: std::time::Instant,
}

/// Puts the entries drawn since the last chunk into `chunk`.
fn attach_logprobs(chunk: &mut ChatCompletionChunk, drawn: &DrawnLogprobs, ids_as_tokens: bool) {
    let mut entries = drawn.take();
    if ids_as_tokens {
        vllm::tokens_as_ids(&mut entries);
    }
    if entries.is_empty() {
        return;
    }
    if let Some(c) = chunk.choices.first_mut() {
        c.logprobs = Some(serde_json::json!({
            "content": openai_logprobs_content(&entries)
        }));
    }
}

/// The SSE events of an engine's token channel.
/// A failure of the engine ends the stream with an error event.
/// No finish chunk, usage chunk or `[DONE]` follows it.
pub(super) fn chat_events<E>(
    mut rx: Receiver<Result<String, String>>,
    engine_for_stats: Arc<E>,
    run: ChatRun,
) -> impl futures::Stream<Item = Result<Event, axum::Error>>
where
    E: StreamSource + ?Sized + 'static,
{
    let ChatRun {
        completion_id,
        model: model_name_clone,
        created_at,
        service_tier,
        wants_logprobs,
        drawn_logprobs,
        ids_as_tokens,
        include_reasoning,
        tools_active,
        tool_format,
        thinking_opened,
        stops,
        keep_stop,
        echo_text,
        single_tool_call,
        max_tokens_cap,
        include_usage,
        prompt_tokens,
        probe,
        energy,
        stream_started,
    } = run;
    async_stream::stream! {
        let mut token_count: i32 = 0;
        let mut first_token_at: Option<std::time::Instant> = None;
        // When tools are active, route content through a scanner
        // that streams natural text until a tool-call marker,
        // then buffers the tool region for end-of-stream parse.
        let mut tool_scanner = if tools_active {
            Some(crate::api::tool_calls::StreamToolScanner::new(tool_format))
        } else {
            None
        };
        let mut splitter = crate::api::thinking::ThinkSplit::opened(thinking_opened);
        let mut reasoning_acc = String::new();
        let mut holdback = vllm::StopHoldback::new(&stops, keep_stop);
        let mut echo_pending = echo_text;

        while let Some(result) = rx.recv().await {
            match result {
                Ok(chunk_text) => {
                    if first_token_at.is_none() {
                        first_token_at = Some(std::time::Instant::now());
                    }
                    token_count += 1;
                    let chunk_text = format!(
                        "{}{}",
                        std::mem::take(&mut echo_pending),
                        holdback.push(&chunk_text)
                    );
                    for seg in splitter.push(&chunk_text) {
                        let delta = match seg {
                            crate::api::thinking::Segment::Thinking(t) => {
                                reasoning_acc.push_str(&t);
                                if !include_reasoning {
                                    continue;
                                }
                                ChunkDelta::reasoning(t)
                            }
                            crate::api::thinking::Segment::Content(c) => {
                                let emit = match tool_scanner.as_mut() {
                                    Some(sc) => sc.push(&c),
                                    None => c,
                                };
                                if emit.is_empty() {
                                    continue;
                                }
                                ChunkDelta::content(emit)
                            }
                        };
                        let mut chunk = ChatCompletionChunk::new_delta_at(
                            &completion_id,
                            &model_name_clone,
                            created_at,
                            0,
                            delta,
                            None,
                        );
                        chunk.service_tier = service_tier.clone();
                        if wants_logprobs {
                            attach_logprobs(&mut chunk, &drawn_logprobs, ids_as_tokens);
                        }
                        yield Ok(Event::default().data(to_json_string!(&chunk)));
                    }
                }
                Err(e) => {
                    // The failure ends the stream with an error frame.
                    // No finish chunk, usage chunk or terminal marker follows it.
                    yield Ok(Event::default().data(
                        openai_error_body(
                            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                            e.to_string(),
                        )
                        .to_string(),
                    ));
                    // The meter closes with the stream.
                    crate::energy_report::end_share(energy, "text", "[/v1/chat/completions]");
                    return;
                }
            }
        }

        // Final chunk with finish_reason. "length" when we
        // produced the requested max_tokens (model would
        // have continued otherwise), "stop" when the
        // engine emitted an EOS / stop sequence and closed
        // the channel on its own.
        // Parse the buffered tool region (if any) into
        // structured calls and emit them as a single delta
        // before the terminal chunk. finish_reason flips to
        // "tool_calls" when the model invoked functions.
        // The text held back for a stop sequence goes first, without the stop.
        let stats = engine_for_stats.take_last_stream_stats();
        let energy = crate::energy_report::end_share(energy, "text", "[/v1/chat/completions]");
        let held = format!(
            "{echo_pending}{}",
            holdback.finish(stats.as_ref().map(|s| &s.finish_reason))
        );
        let tail: Vec<_> = splitter
            .push(&held)
            .into_iter()
            .chain(splitter.finish())
            .collect();
        for seg in tail {
            let delta = match seg {
                crate::api::thinking::Segment::Thinking(_) if !include_reasoning => {
                    continue
                }
                crate::api::thinking::Segment::Thinking(t) => ChunkDelta::reasoning(t),
                crate::api::thinking::Segment::Content(c) => {
                    let emit = match tool_scanner.as_mut() {
                        Some(sc) => sc.push(&c),
                        None => c,
                    };
                    if emit.is_empty() {
                        continue;
                    }
                    ChunkDelta::content(emit)
                }
            };
            let mut chunk = ChatCompletionChunk::new_delta_at(
                &completion_id,
                &model_name_clone,
                created_at,
                0,
                delta,
                None,
            );
            chunk.service_tier = service_tier.clone();
            yield Ok(Event::default().data(to_json_string!(&chunk)));
        }
        let mut tool_calls = tool_scanner
            .as_ref()
            .map(|sc| sc.finalize())
            .unwrap_or_default();
        if single_tool_call {
            tool_calls.truncate(1);
        }
        // An answer cut at the token limit ends as `length` with its text: a cut call is not a
        // call.
        let cut = match stats.as_ref() {
            Some(s) => s.finish_reason.openai() == "length",
            None => max_tokens_cap.is_some_and(|cap| (token_count as usize) >= cap),
        };
        if cut {
            tool_calls.clear();
        }
        let used_tools = !tool_calls.is_empty();
        if used_tools {
            let deltas: Vec<_> = tool_calls
                .iter()
                .enumerate()
                .map(|(i, c)| crate::api::types::ToolCallDelta::from_tool_call(i, c))
                .collect();
            let mut tc_chunk = ChatCompletionChunk::new_delta_at(
                &completion_id,
                &model_name_clone,
                created_at,
                0,
                ChunkDelta::tool_calls(deltas),
                None,
            );
            tc_chunk.service_tier = service_tier.clone();
            yield Ok(Event::default().data(to_json_string!(&tc_chunk)));
        } else if let Some(tail) =
            tool_scanner.as_ref().map(|sc| sc.unstreamed()).filter(|t| !t.is_empty())
        {
            // Bare-JSON trigger fired on non-tool text:
            // flush the withheld buffer as content so it
            // isn't lost.
            let mut chunk = ChatCompletionChunk::new_delta_at(
                &completion_id,
                &model_name_clone,
                created_at,
                0,
                ChunkDelta::content(tail.to_string()),
                None,
            );
            chunk.service_tier = service_tier.clone();
            yield Ok(Event::default().data(to_json_string!(&chunk)));
        }
        let finish_reason = if used_tools {
            "tool_calls"
        } else if let Some(s) = stats.as_ref() {
            s.finish_reason.openai()
        } else {
            match max_tokens_cap {
                Some(cap) if (token_count as usize) >= cap => "length",
                _ => "stop",
            }
        };
        let mut stop_chunk = ChatCompletionChunk::new_delta_at(
            &completion_id,
            &model_name_clone,
            created_at,
            0,
            ChunkDelta::empty(),
            Some(finish_reason.to_string()),
        );
        stop_chunk.service_tier = service_tier.clone();
        // The entries of the last tokens go with the finish chunk.
        if wants_logprobs {
            attach_logprobs(&mut stop_chunk, &drawn_logprobs, ids_as_tokens);
        }
        yield Ok(Event::default().data(to_json_string!(&stop_chunk)));

        // OpenAI spec: the trailing usage-only chunk
        // (choices=[], usage populated) ships only when the
        // client passes `stream_options.include_usage: true`.
        // Older SDKs treat an unexpected chunk as a hard
        // protocol error, so default-off is the safer
        // behaviour.
        // What the engine actually read, where it said; the word-count
        // estimate only where it did not. Both the usage chunk and the log
        // line below take this figure: reporting the estimate in the log made
        // every streamed request look slower per token than it was, by
        // whatever margin the estimate was wrong - on source code, nearly
        // double.
        // The cached share is reported as the whole answer reports it.
        // The request's own outcome holds its prompt counts on every path.
        let outcome = probe.outcome();
        let mut usage = match (outcome.as_ref(), stats.as_ref()) {
            (Some(o), _) => Usage::from_counts(
                o.figures.prompt_read(),
                o.figures.cached_prompt_tokens,
                o.figures.generated_tokens,
            ),
            (None, Some(s)) => Usage::from_counts(
                s.prompt_eval_count,
                s.cached_prompt_tokens,
                s.eval_count,
            ),
            (None, None) => Usage::new(prompt_tokens, token_count),
        }
        .with_request(probe.outcome().as_ref(), energy.as_ref());
        let (in_tokens, out_tokens) = (usage.prompt_tokens, usage.completion_tokens);
        if include_usage {
            if !reasoning_acc.is_empty() {
                if let (Some(n), Some(d)) = (
                    engine_for_stats.count_tokens(&reasoning_acc).await,
                    usage.completion_tokens_details.as_mut(),
                ) {
                    d.reasoning_tokens = n as i32;
                }
            }
            let mut usage_chunk = ChatCompletionChunk::new_final_at(
                &completion_id,
                &model_name_clone,
                created_at,
                usage,
            );
            usage_chunk.service_tier = service_tier.clone();
            yield Ok(Event::default().data(to_json_string!(&usage_chunk)));
        }

        // Stream summary log - mirrors the non-stream
        // "Chat completion:" line so a single grep covers
        // both paths. `total_ms` is wall time from stream
        // start to terminal marker, including network
        // backpressure on slow clients.
        let total_ms = stream_started.elapsed().as_millis();
        let ttft_ms = time_to_first_token_ms(stream_started, first_token_at);
        let rate = decode_rate_suffix(first_token_at, token_count as u64);
        let cached = match (outcome.as_ref(), stats.as_ref()) {
            (Some(o), _) => o.figures.cached_prompt_tokens,
            (None, s) => s.map_or(0, |s| s.cached_prompt_tokens),
        };
        let prefill_ms = stats
            .as_ref()
            .map_or(0, |s| s.prompt_eval_duration_ns / 1_000_000);
        tracing::info!(
            "Chat completion stream: prompt_tokens={in_tokens} cached={cached} completion_tokens={out_tokens} finish={finish_reason} prefill={prefill_ms}ms ttft={ttft_ms}ms total={total_ms}ms{rate}"
        );
        // Terminal marker
        yield Ok(Event::default().data("[DONE]"));
    }
}

#[cfg(test)]
mod tests;
