//! The chat routes report each answer's own figures: the `loken` object in the final usage of
//! both routes, streamed or not, and the reading of a slow prompt as comment lines on a stream.

use super::model_manager::LoadedModelEntry;
use super::*;
use crate::api::request_figures::PREFILL_REPORT_PERIOD;
use crate::inference::engine::llm_engine::InferenceConfig;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

/// The tiny checkpoint the routes serve.
const CHECKPOINT: &str = "tests/vectors/hf/llama";
/// The name the server knows it by.
pub(super) const MODEL: &str = "test/request-figures";
/// Tokens each answer may write.
const MAX_TOKENS: u64 = 4;
/// Words of a short prompt, read well within a report period.
const SHORT_PROMPT_WORDS: usize = 4;
/// Words of a long prompt, its reading slowed to span several report periods.
const LONG_PROMPT_WORDS: usize = 96;
/// Report periods a slowed prefill pauses for after each report short of the whole prompt.
const SLOW_PREFILL_PERIODS: u32 = 3;
/// What the comment line of a prompt being read starts with.
const PREFILL_COMMENT: &str = ": loken-prefill ";

/// The route tests run one at a time: the slowed prefill is process-wide.
pub(super) use crate::request_metrics::PREFILL_PAUSE_LOCK as ROUTES;

/// A server serving the tiny checkpoint as [`MODEL`], with a word-level vocabulary of its
/// size, and the scratch directory it was copied to.
pub(super) async fn server_with_tiny_model() -> (APIServer, std::path::PathBuf, usize) {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CHECKPOINT);
    let dir = std::env::temp_dir().join(format!(
        "loken-request-figures-{}-{}",
        std::process::id(),
        CALLS.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for f in ["config.json", "model.safetensors"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
    let context_length = config["max_position_embeddings"].as_u64().unwrap() as usize;
    let vocab: ahash::AHashMap<String, u32> = (0..vocab_size as u32)
        .map(|i| (format!("t{i}"), i))
        .collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("t0".into())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer.save(dir.join("tokenizer.json"), false).unwrap();

    let engine = Arc::new(LlmEngine::with_config(InferenceConfig {
        model_id: dir.to_string_lossy().to_string(),
        context_length,
        ..InferenceConfig::default()
    }));
    engine
        .load_model()
        .await
        .expect("the tiny checkpoint loads");
    let state = APIServer::new(
        "/nonexistent-ollama-models".to_string(),
        "/nonexistent-hf-models".to_string(),
    );
    state
        .engines
        .write()
        .await
        .push(LoadedModelEntry::new(MODEL.to_string(), engine, Some(1)));
    (state, dir, vocab_size)
}

/// A prompt of `words` words of the vocabulary, starting at word `first`: two routes given
/// different starts share no prefix, so neither finds the other's prompt in the cache.
pub(super) fn prompt(words: usize, vocab_size: usize, first: usize) -> String {
    (0..words)
        .map(|i| format!("t{}", (i + first) % vocab_size))
        .collect::<Vec<_>>()
        .join(" ")
}

pub(super) async fn body_text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// The JSON of each `data:` line of a stream.
pub(super) fn data_lines(sse: &str) -> Vec<Value> {
    sse.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str(d).ok())
        .collect()
}

/// The `(done, total)` of each prefill comment line of a stream.
fn prefill_comments(sse: &str) -> Vec<(u64, u64)> {
    sse.lines()
        .filter_map(|l| l.strip_prefix(PREFILL_COMMENT))
        .map(|c| {
            let v: Value = serde_json::from_str(c).unwrap();
            (v["done"].as_u64().unwrap(), v["total"].as_u64().unwrap())
        })
        .collect()
}

/// The prefill comment lines of a short prompt's stream that took `took` from its request to
/// its last line.
///
/// A comment is sent only once the reading of the prompt has run for a report period.
/// A stream that took less sends none.
/// A slower stream may send some.
/// Each says what is read of the same whole, short of it, in the order read.
fn assert_short_prompt_comments(sse: &str, took: std::time::Duration) {
    let comments = prefill_comments(sse);
    if took < PREFILL_REPORT_PERIOD {
        assert!(comments.is_empty(), "{sse}");
        return;
    }
    let total = comments.first().map_or(0, |&(_, total)| total);
    assert!(
        comments.iter().all(|&(done, t)| t == total && done < total),
        "{comments:?}"
    );
    assert!(
        comments.windows(2).all(|w| w[0].0 <= w[1].0),
        "{comments:?}"
    );
}

/// The object carries the request's times, none negative, and a cache place that fits the
/// cache when the backend reports one.
fn assert_loken(usage: &Value) {
    let loken = &usage["loken"];
    assert!(loken.is_object(), "no loken object in {usage}");
    for field in ["queue_ms", "prefill_ms", "decode_ms"] {
        let ms = loken[field]
            .as_f64()
            .unwrap_or_else(|| panic!("{field} in {loken}"));
        assert!(ms >= 0.0, "{field} is {ms}");
    }
    if let Some(kv) = loken.get("kv") {
        let used = kv["used_tokens"].as_u64().unwrap();
        let capacity = kv["capacity_tokens"].as_u64().unwrap();
        assert!(used > 0 && used <= capacity, "{kv}");
    }
    if let Some(energy) = loken.get("energy") {
        assert!(energy["joules"].as_f64().unwrap() >= 0.0, "{energy}");
        assert!(matches!(energy["share"].as_str(), Some("alone" | "work")));
    }
}

async fn openai(state: &APIServer, words: usize, vocab_size: usize, stream: bool) -> Response {
    let mut body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt(words, vocab_size, 0)}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
    });
    if stream {
        body["stream"] = json!(true);
        body["stream_options"] = json!({"include_usage": true});
    }
    let request = serde_json::from_value(body).unwrap();
    super::openai::chat_completion_at(
        state.clone(),
        axum::http::HeaderMap::new(),
        request,
        crate::api::gate::Priority::Interactive,
    )
    .await
    .unwrap_or_else(|e| panic!("the chat completion is served: {e:?}"))
}

async fn anthropic(state: &APIServer, words: usize, vocab_size: usize, stream: bool) -> Response {
    let request = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt(words, vocab_size, 1)}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": stream,
    }))
    .unwrap();
    super::anthropic_api::anthropic_messages_at(
        state.clone(),
        axum::http::HeaderMap::new(),
        request,
        crate::api::gate::Priority::Interactive,
    )
    .await
}

/// The OpenAI route puts the object in the usage of a whole answer and in the usage chunk of
/// a stream, with the speculation's counts beside it; a short prompt read within a report
/// period sends no comment line.
#[tokio::test(flavor = "multi_thread")]
async fn the_openai_route_reports_each_answer() {
    let _one = ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;

    let whole: Value = serde_json::from_str(
        &body_text(openai(&state, SHORT_PROMPT_WORDS, vocab, false).await).await,
    )
    .unwrap();
    assert_loken(&whole["usage"]);
    let details = &whole["usage"]["completion_tokens_details"];
    assert!(details["accepted_prediction_tokens"].is_i64(), "{details}");
    assert!(details["rejected_prediction_tokens"].is_i64(), "{details}");

    let asked = std::time::Instant::now();
    let sse = body_text(openai(&state, SHORT_PROMPT_WORDS, vocab, true).await).await;
    let took = asked.elapsed();
    let usage = data_lines(&sse)
        .into_iter()
        .rev()
        .find(|c| c["usage"].is_object())
        .unwrap_or_else(|| panic!("no usage chunk in {sse}"));
    assert_loken(&usage["usage"]);
    assert_short_prompt_comments(&sse, took);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The Anthropic route puts the object in the usage of a whole answer and in the usage of
/// its `message_delta`; a short prompt read within a report period sends no comment line.
#[tokio::test(flavor = "multi_thread")]
async fn the_anthropic_route_reports_each_answer() {
    let _one = ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;

    let whole: Value = serde_json::from_str(
        &body_text(anthropic(&state, SHORT_PROMPT_WORDS, vocab, false).await).await,
    )
    .unwrap();
    assert_loken(&whole["usage"]);

    let asked = std::time::Instant::now();
    let sse = body_text(anthropic(&state, SHORT_PROMPT_WORDS, vocab, true).await).await;
    let took = asked.elapsed();
    let delta = data_lines(&sse)
        .into_iter()
        .find(|e| e["type"] == "message_delta")
        .unwrap_or_else(|| panic!("no message_delta in {sse}"));
    assert_loken(&delta["usage"]);
    assert_short_prompt_comments(&sse, took);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A long prompt read slowly is reported on both routes as it is read: comment lines with
/// what is read of the whole, and the answer's figures in its usage.
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_prompt_is_reported_as_it_is_read() {
    let _one = ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    let pause = PREFILL_REPORT_PERIOD * SLOW_PREFILL_PERIODS;
    crate::request_metrics::PREFILL_PAUSE_MS.store(pause.as_millis() as u64, Ordering::Relaxed);

    let openai_sse = body_text(openai(&state, LONG_PROMPT_WORDS, vocab, true).await).await;
    let anthropic_sse = body_text(anthropic(&state, LONG_PROMPT_WORDS, vocab, true).await).await;
    crate::request_metrics::PREFILL_PAUSE_MS.store(0, Ordering::Relaxed);
    let _ = std::fs::remove_dir_all(&dir);

    for sse in [&openai_sse, &anthropic_sse] {
        let comments = prefill_comments(sse);
        assert!(!comments.is_empty(), "no prefill comment in {sse}");
        assert!(
            comments
                .iter()
                .all(|&(done, total)| total > 0 && done < total),
            "{comments:?}"
        );
        assert!(
            comments.windows(2).all(|w| w[0].0 <= w[1].0),
            "{comments:?}"
        );
    }
    let usage = data_lines(&openai_sse)
        .into_iter()
        .rev()
        .find(|c| c["usage"].is_object())
        .unwrap();
    assert_loken(&usage["usage"]);
    let prefill_ms = usage["usage"]["loken"]["prefill_ms"].as_f64().unwrap();
    assert!(
        prefill_ms >= pause.as_millis() as f64,
        "prefill_ms {prefill_ms}"
    );
}

/// The running models are listed at once while a request holds its model, with the figures
/// the model was listed with before it.
#[tokio::test(flavor = "multi_thread")]
async fn the_running_models_are_listed_during_a_request() {
    let _one = ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    let listed = |state: APIServer| async move {
        let Json(ps) = ollama_ps(State(state)).await.unwrap();
        let m = &ps.models[0];
        (m.size, m.size_vram, m.context_length)
    };
    let idle = listed(state.clone()).await;
    assert!(idle.0 > 0 && idle.2 > 0, "{idle:?}");
    // The request holds its model for this long before its prompt is read.
    let held = PREFILL_REPORT_PERIOD * 8;
    crate::request_metrics::PREFILL_PAUSE_MS.store(held.as_millis() as u64, Ordering::Relaxed);
    let response = openai(&state, LONG_PROMPT_WORDS, vocab, true).await;
    let mut body = response.into_body().into_data_stream();
    let mut seen = String::new();
    while !seen.contains(PREFILL_COMMENT) {
        let frame = futures::StreamExt::next(&mut body).await.unwrap().unwrap();
        seen.push_str(&String::from_utf8_lossy(&frame));
    }
    let busy = tokio::time::timeout(held / 4, listed(state.clone())).await;
    crate::request_metrics::PREFILL_PAUSE_MS.store(0, Ordering::Relaxed);
    drop(body);
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(busy.expect("the listing waited for the request"), idle);
}

/// Reads `response` until its first prefill comment line, then drops it as a client that goes
/// away does.
async fn leave_while_reading(response: Response) {
    let mut body = response.into_body().into_data_stream();
    let mut seen = String::new();
    while let Some(frame) = futures::StreamExt::next(&mut body).await {
        seen.push_str(&String::from_utf8_lossy(&frame.unwrap()));
        if seen.contains(PREFILL_COMMENT) {
            return;
        }
    }
    panic!("the stream ended before a prefill comment: {seen}");
}

/// A client that leaves a stream while its prompt is read stops the request: no answer is
/// written and the request never finishes as one.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_that_leaves_during_the_prompt_stops_the_request() {
    let _one = ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    let engine = state.engines.read().await[0].engine.clone();
    let model_id = dir.to_string_lossy().to_string();
    let finished = || {
        crate::request_metrics::snapshot()
            .get(&model_id)
            .map_or(0, |m| m.requests())
    };
    let pause = PREFILL_REPORT_PERIOD * SLOW_PREFILL_PERIODS;
    crate::request_metrics::PREFILL_PAUSE_MS.store(pause.as_millis() as u64, Ordering::Relaxed);
    leave_while_reading(openai(&state, LONG_PROMPT_WORDS, vocab, true).await).await;
    leave_while_reading(anthropic(&state, LONG_PROMPT_WORDS, vocab, true).await).await;
    crate::request_metrics::PREFILL_PAUSE_MS.store(0, Ordering::Relaxed);
    // The model is taken in turn: this count runs once both left requests have ended.
    let counted = engine.count_tokens("t1").await;
    let _ = std::fs::remove_dir_all(&dir);
    assert!(counted.is_some());
    assert_eq!(
        finished(),
        0,
        "a left request was read to its end and answered"
    );
}

/// A long prompt with token rules, read slowly on the sequential path, is reported as it is read,
/// and the answer's figures are in its usage.
#[tokio::test(flavor = "multi_thread")]
async fn a_slow_prompt_with_token_rules_is_reported_as_it_is_read() {
    let _one = ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    let pause = PREFILL_REPORT_PERIOD * SLOW_PREFILL_PERIODS;
    crate::request_metrics::PREFILL_PAUSE_MS.store(pause.as_millis() as u64, Ordering::Relaxed);
    let request = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt(LONG_PROMPT_WORDS, vocab, 2)}],
        "max_tokens": MAX_TOKENS,
        "min_tokens": 1,
        "temperature": 0,
        "stream": true,
        "stream_options": {"include_usage": true},
    }))
    .unwrap();
    let response = super::openai::chat_completion_at(
        state.clone(),
        axum::http::HeaderMap::new(),
        request,
        crate::api::gate::Priority::Interactive,
    )
    .await
    .unwrap_or_else(|e| panic!("the chat completion is served: {e:?}"));
    let sse = body_text(response).await;
    crate::request_metrics::PREFILL_PAUSE_MS.store(0, Ordering::Relaxed);
    let _ = std::fs::remove_dir_all(&dir);

    let comments = prefill_comments(&sse);
    assert!(!comments.is_empty(), "no prefill comment in {sse}");
    assert!(
        comments
            .iter()
            .all(|&(done, total)| total > 0 && done < total),
        "{comments:?}"
    );
    let usage = data_lines(&sse)
        .into_iter()
        .rev()
        .find(|c| c["usage"].is_object())
        .unwrap_or_else(|| panic!("no usage chunk in {sse}"));
    assert_loken(&usage["usage"]);
}

/// Words of the prompt a request repeats.
const REPEATED_PROMPT_WORDS: usize = 8;

/// The prompt figures of an Ollama generate answer: every prompt position, then the cached
/// ones.
/// The answer states a time reading its prompt, within its whole time.
async fn ollama_counts(state: &APIServer, vocab_size: usize, stream: bool) -> (u64, u64) {
    let mut request = OllamaGenerateRequest::new(
        MODEL.to_string(),
        prompt(REPEATED_PROMPT_WORDS, vocab_size, 2),
    );
    request.stream = stream;
    request.options = Some(json!({"temperature": 0, "num_predict": MAX_TOKENS}));
    let response = super::ollama_generate(
        axum::extract::State(state.clone()),
        axum::http::HeaderMap::new(),
        OllamaJson(request),
    )
    .await
    .unwrap_or_else(|e| panic!("the generate request is served: {e:?}"));
    let text = body_text(response).await;
    let last: Value = text
        .lines()
        .rev()
        .find_map(|l| serde_json::from_str(l).ok())
        .unwrap_or_else(|| panic!("no answer in {text}"));
    let field = |name: &str| {
        last[name]
            .as_u64()
            .unwrap_or_else(|| panic!("{name} in {last}"))
    };
    assert!(field("prompt_eval_duration") > 0, "stream {stream}: {last}");
    assert!(
        field("prompt_eval_duration") + field("eval_duration") <= field("total_duration"),
        "stream {stream}: {last}"
    );
    (
        field("prompt_eval_count"),
        field("prompt_eval_cached_count"),
    )
}

/// The prompt figures of an OpenAI chat answer: every prompt position, then the cached ones.
async fn openai_counts(state: &APIServer, vocab_size: usize, stream: bool) -> (u64, u64) {
    let text = body_text(openai(state, REPEATED_PROMPT_WORDS, vocab_size, stream).await).await;
    let usage = if stream {
        data_lines(&text)
            .into_iter()
            .rev()
            .find(|c| c["usage"].is_object())
            .unwrap_or_else(|| panic!("no usage chunk in {text}"))["usage"]
            .clone()
    } else {
        serde_json::from_str::<Value>(&text).unwrap()["usage"].clone()
    };
    (
        usage["prompt_tokens"].as_u64().unwrap(),
        usage["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap(),
    )
}

/// A prompt asked once then twice again, whole then streamed, on the Ollama generate route
/// and on the OpenAI chat route.
/// The two repeats report the same figures, and every answer counts the whole prompt.
///
/// Returns whether the requests ran in the batch worker.
async fn a_repeated_request_reports_the_same_counts(batched: bool) -> bool {
    let _one = ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;
    let engine = state.engines.read().await[0].engine.clone();
    engine.hold_lone_requests_on(batched);

    let first = ollama_counts(&state, vocab, false).await;
    let whole = ollama_counts(&state, vocab, false).await;
    let streamed = ollama_counts(&state, vocab, true).await;
    let in_worker = engine.is_continuous().await;
    assert_eq!(whole, streamed, "the repeated generate request");
    assert_eq!(whole.0, first.0, "every prompt position is counted");
    assert!(whole.1 <= whole.0, "{whole:?}");

    let first = openai_counts(&state, vocab, false).await;
    let whole = openai_counts(&state, vocab, false).await;
    let streamed = openai_counts(&state, vocab, true).await;
    assert_eq!(whole, streamed, "the repeated chat request");
    assert_eq!(whole.0, first.0, "every prompt position is counted");
    assert!(whole.1 <= whole.0, "{whole:?}");
    assert_eq!(
        engine.is_continuous().await,
        in_worker,
        "the requests kept their path"
    );
    let _ = std::fs::remove_dir_all(&dir);
    in_worker
}

/// [`a_repeated_request_reports_the_same_counts`] on the serial path.
#[tokio::test(flavor = "multi_thread")]
async fn a_repeated_request_reports_the_same_counts_on_the_serial_path() {
    assert!(!a_repeated_request_reports_the_same_counts(false).await);
}

/// [`a_repeated_request_reports_the_same_counts`] through the batch worker of a card.
#[cfg(feature = "gpu")]
#[tokio::test(flavor = "multi_thread")]
async fn a_repeated_request_reports_the_same_counts_through_the_worker() {
    if !a_repeated_request_reports_the_same_counts(true).await {
        eprintln!("no card took the batch worker; the worker's counts are NOT covered by this run");
    }
}
