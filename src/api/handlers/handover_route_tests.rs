//! 2 instances in 1 process, each the prefill peer of the other, joined by the
//! `/api/cluster/prefill` route on loopback ports.
//!
//! A long prompt sent to one instance is read by the other and answered by the first with the
//! tokens an instance without a peer gives.
//! A short prompt is read where it arrives.
//! The peer's reading is relayed to the request as it goes.
//! A peer that answers the handover alone still serves.

use super::model_manager::LoadedModelEntry;
use super::*;
use crate::inference::engine::llm_engine::InferenceConfig;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

/// The tiny checkpoint the instances serve.
const CHECKPOINT: &str = "tests/vectors/hf/llama";
/// The name the instances know it by, a hub name resolved under their models directory.
pub(super) const MODEL: &str = "test/handover";
/// Tokens each answer may write.
const MAX_TOKENS: u64 = 4;
/// Words of a long prompt: more than the largest prefill chunk a model starts at.
const LONG_PROMPT_WORDS: usize = 3000;
/// Words of a short prompt, less than any prefill chunk.
const SHORT_PROMPT_WORDS: usize = 8;
/// Positions the copied checkpoint declares: room for a long prompt and its answer.
const CONTEXT: usize = 4096;

/// The scratch directory of a test, with the checkpoint copied as the hub snapshot of
/// [`MODEL`], its context widened to [`CONTEXT`], and a word-level vocabulary of its size.
pub(super) fn models_dir() -> (std::path::PathBuf, usize) {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CHECKPOINT);
    let root = std::env::temp_dir().join(format!(
        "loken-handover-{}-{}",
        std::process::id(),
        CALLS.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&root);
    let (org, name) = MODEL.split_once('/').unwrap();
    let snapshot = root
        .join(format!("models--{org}--{name}"))
        .join("snapshots")
        .join("main");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::copy(
        src.join("model.safetensors"),
        snapshot.join("model.safetensors"),
    )
    .unwrap();
    let mut config: Value =
        serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    config["max_position_embeddings"] = json!(CONTEXT);
    std::fs::write(snapshot.join("config.json"), config.to_string()).unwrap();
    let vocab_size = config["vocab_size"].as_u64().unwrap() as usize;
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
    tokenizer
        .save(snapshot.join("tokenizer.json"), false)
        .unwrap();
    (root, vocab_size)
}

/// An instance serving [`MODEL`] from `root`, with `peer` as its prefill peer.
/// Lone requests stay on the serial path, the one that asks a peer.
async fn instance(root: &std::path::Path, peer: Option<String>) -> APIServer {
    let engine = Arc::new(LlmEngine::with_config(InferenceConfig {
        model_id: MODEL.to_string(),
        models_dir: Some(root.to_path_buf()),
        context_length: CONTEXT,
        prefill_peer: peer,
        ..InferenceConfig::default()
    }));
    engine
        .load_model()
        .await
        .expect("the tiny checkpoint loads");
    engine.hold_lone_requests_on(false);
    let state = APIServer::new(
        "/nonexistent-ollama-models".to_string(),
        "/nonexistent-hf-models".to_string(),
    );
    state
        .engines
        .write()
        .await
        .push(LoadedModelEntry::new(MODEL.to_string(), engine, Some(1)));
    state
}

/// A prompt of `words` words of the vocabulary, starting at word `first`.
fn prompt(words: usize, vocab_size: usize, first: usize) -> String {
    (0..words)
        .map(|i| format!("t{}", (i * 7 + first) % vocab_size))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The answer of `state` to `prompt` on the Ollama generate route, and the prompt tokens its
/// cache served.
async fn answer(state: &APIServer, prompt: String) -> (String, u64) {
    let mut request = OllamaGenerateRequest::new(MODEL.to_string(), prompt);
    request.stream = false;
    request.options = Some(json!({"temperature": 0, "num_predict": MAX_TOKENS}));
    let response = super::ollama_generate(
        axum::extract::State(state.clone()),
        axum::http::HeaderMap::new(),
        OllamaJson(request),
    )
    .await
    .unwrap_or_else(|e| panic!("the generate request is served: {e:?}"));
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let whole: Value = serde_json::from_slice(&bytes).unwrap();
    (
        whole["response"].as_str().unwrap().to_string(),
        whole["prompt_eval_cached_count"].as_u64().unwrap(),
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_is_handed_over_between_2_instances_both_ways() {
    // A prefill slowed by another test would change the rates the hand-over is decided on.
    let _paused = crate::request_metrics::PREFILL_PAUSE_LOCK.lock().await;
    let (root, vocab) = models_dir();
    let a_port = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let b_port = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = |l: &tokio::net::TcpListener| format!("http://{}", l.local_addr().unwrap());
    let a = instance(&root, Some(url(&b_port))).await;
    let b = instance(&root, Some(url(&a_port))).await;
    let alone = instance(&root, None).await;
    for (listener, state) in [(a_port, &a), (b_port, &b)] {
        let app = state.create_router();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    }

    // Each direction reads a prompt that shares no prefix with the prompts before it.
    for (to, first) in [(&a, 1), (&b, 2)] {
        let long = prompt(LONG_PROMPT_WORDS, vocab, first);
        let (text, cached) = answer(to, long.clone()).await;
        let (want, read_alone) = answer(&alone, long).await;
        assert_eq!(read_alone, 0, "the instance without a peer read the prompt");
        assert!(cached > 0, "the peer did not hand the prompt over");
        assert_eq!(text, want, "the handed prompt gave another answer");
    }
    let (_, cached) = answer(&a, prompt(SHORT_PROMPT_WORDS, vocab, 3)).await;
    assert_eq!(cached, 0, "a short prompt was handed over");
    let _ = std::fs::remove_dir_all(&root);
}

/// What the comment line of a prompt being read starts with.
const PREFILL_COMMENT: &str = ": loken-prefill ";

/// Report periods the prefill pauses for after each report short of the whole prompt.
const SLOW_PREFILL_PERIODS: u32 = 3;

/// A peer that answers the handover alone, as an instance without the streamed answer does,
/// still reads the prompts of an instance that asks for the stream.
#[tokio::test(flavor = "multi_thread")]
async fn a_peer_without_the_streamed_answer_still_hands_a_prompt_over() {
    let _paused = crate::request_metrics::PREFILL_PAUSE_LOCK.lock().await;
    let (root, vocab) = models_dir();
    let peer_port = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", peer_port.local_addr().unwrap());
    let a = instance(&root, Some(url)).await;
    let peer = instance(&root, None).await;
    let handover_alone =
        |axum::extract::State(state): axum::extract::State<APIServer>,
         axum::Json(req): axum::Json<ClusterPrefillRequest>| async move {
            let bytes = state
                .read_prompt_for_peer(&req.model, &req.prompt, None)
                .await
                .unwrap();
            (
                [(axum::http::header::CONTENT_TYPE, "application/octet-stream")],
                bytes,
            )
        };
    let app = axum::Router::new()
        .route("/api/cluster/prefill", axum::routing::post(handover_alone))
        .with_state(peer);
    tokio::spawn(async move { axum::serve(peer_port, app).await.unwrap() });

    let (_, cached) = answer(&a, prompt(LONG_PROMPT_WORDS, vocab, 5)).await;
    let _ = std::fs::remove_dir_all(&root);
    assert!(cached > 0, "the peer did not hand the prompt over");
}

/// A streamed chat answer whose prompt the peer reads reports the reading of the whole prompt
/// while the peer reads it, the peer's progress before its handover lands, and the answer's
/// figures in its usage.
#[tokio::test(flavor = "multi_thread")]
async fn a_prompt_the_peer_reads_is_reported_as_it_is_read() {
    let _paused = crate::request_metrics::PREFILL_PAUSE_LOCK.lock().await;
    let (root, vocab) = models_dir();
    let peer_port = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", peer_port.local_addr().unwrap());
    let a = instance(&root, Some(url)).await;
    let peer = instance(&root, None).await;
    let app = peer.create_router();
    tokio::spawn(async move { axum::serve(peer_port, app).await.unwrap() });

    let pause = crate::api::request_figures::PREFILL_REPORT_PERIOD * SLOW_PREFILL_PERIODS;
    crate::request_metrics::PREFILL_PAUSE_MS.store(pause.as_millis() as u64, Ordering::Relaxed);
    let request = serde_json::from_value(json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt(LONG_PROMPT_WORDS, vocab, 4)}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": true,
        "stream_options": {"include_usage": true},
    }))
    .unwrap();
    let response = super::openai::chat_completion_at(
        a.clone(),
        axum::http::HeaderMap::new(),
        request,
        crate::api::gate::Priority::Interactive,
    )
    .await
    .unwrap_or_else(|e| panic!("the chat completion is served: {e:?}"));
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    crate::request_metrics::PREFILL_PAUSE_MS.store(0, Ordering::Relaxed);
    let _ = std::fs::remove_dir_all(&root);

    let sse = String::from_utf8(bytes.to_vec()).unwrap();
    let comments: Vec<(u64, u64)> = sse
        .lines()
        .filter_map(|l| l.strip_prefix(PREFILL_COMMENT))
        .map(|c| {
            let v: Value = serde_json::from_str(c).unwrap();
            (v["done"].as_u64().unwrap(), v["total"].as_u64().unwrap())
        })
        .collect();
    let usage = sse
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .rev()
        .find(|c| c["usage"].is_object())
        .unwrap_or_else(|| panic!("no usage chunk in {sse}"));
    let usage = &usage["usage"];
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or_else(|| panic!("no cached tokens in {usage}"));
    assert!(cached > 0, "the peer did not hand the prompt over: {usage}");
    let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap();
    assert!(
        comments.contains(&(0, prompt_tokens)),
        "no reading of the whole prompt while the peer read it: {comments:?}"
    );
    // Only the peer reads the prompt's tokens short of the whole: the handover covers them.
    assert!(
        comments
            .iter()
            .any(|&(done, total)| total == prompt_tokens && done > 0 && done < total),
        "the peer's reading was not relayed: {comments:?}"
    );
    assert!(usage["loken"].is_object(), "no loken object in {usage}");
}
