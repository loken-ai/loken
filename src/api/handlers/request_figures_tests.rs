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
const MODEL: &str = "test/request-figures";
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
static ROUTES: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// A server serving the tiny checkpoint as [`MODEL`], with a word-level vocabulary of its
/// size, and the scratch directory it was copied to.
async fn server_with_tiny_model() -> (APIServer, std::path::PathBuf, usize) {
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
fn prompt(words: usize, vocab_size: usize, first: usize) -> String {
    (0..words)
        .map(|i| format!("t{}", (i + first) % vocab_size))
        .collect::<Vec<_>>()
        .join(" ")
}

async fn body_text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

/// The JSON of each `data:` line of a stream.
fn data_lines(sse: &str) -> Vec<Value> {
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
/// a stream, with the speculation's counts beside it; a short prompt sends no comment line.
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

    let sse = body_text(openai(&state, SHORT_PROMPT_WORDS, vocab, true).await).await;
    let usage = data_lines(&sse)
        .into_iter()
        .rev()
        .find(|c| c["usage"].is_object())
        .unwrap_or_else(|| panic!("no usage chunk in {sse}"));
    assert_loken(&usage["usage"]);
    assert!(prefill_comments(&sse).is_empty(), "{sse}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The Anthropic route puts the object in the usage of a whole answer and in the usage of
/// its `message_delta`; a short prompt sends no comment line.
#[tokio::test(flavor = "multi_thread")]
async fn the_anthropic_route_reports_each_answer() {
    let _one = ROUTES.lock().await;
    let (state, dir, vocab) = server_with_tiny_model().await;

    let whole: Value = serde_json::from_str(
        &body_text(anthropic(&state, SHORT_PROMPT_WORDS, vocab, false).await).await,
    )
    .unwrap();
    assert_loken(&whole["usage"]);

    let sse = body_text(anthropic(&state, SHORT_PROMPT_WORDS, vocab, true).await).await;
    let delta = data_lines(&sse)
        .into_iter()
        .find(|e| e["type"] == "message_delta")
        .unwrap_or_else(|| panic!("no message_delta in {sse}"));
    assert_loken(&delta["usage"]);
    assert!(prefill_comments(&sse).is_empty(), "{sse}");
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
