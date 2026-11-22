//! The classify, score and pooling routes on loaded models: a checkpoint with a
//! classification head answers each classification task and scores pairs as a cross-encoder,
//! and a model without a head is refused with the reason.

use super::model_manager::LoadedModelEntry;
use super::request_figures_tests::{body_text, server_with_tiny_model, MODEL};
use super::*;
use crate::inference::engine::llm_engine::InferenceConfig;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

/// The tiny checkpoint the heads are added to.
const CHECKPOINT: &str = "tests/vectors/hf/llama";
/// The name the server knows the checkpoint with a head by.
const HEAD_MODEL: &str = "test/classifier";

/// A server serving the tiny checkpoint as a sequence classifier with one label per name of
/// `labels`, and the scratch directory it was written to.
///
/// The language head is dropped and a `score` head is added, as a classifier checkpoint holds
/// them.
async fn server_with_head(labels: &[&str]) -> (APIServer, std::path::PathBuf) {
    use safetensors::tensor::TensorView;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CHECKPOINT);
    let dir = std::env::temp_dir().join(format!(
        "loken-classifier-{}-{}",
        std::process::id(),
        CALLS.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut config: Value =
        serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    let hidden = config["hidden_size"].as_u64().unwrap() as usize;
    let vocab_size = config["vocab_size"].as_u64().unwrap() as u32;
    let context_length = config["max_position_embeddings"].as_u64().unwrap() as usize;
    config["architectures"] = json!(["LlamaForSequenceClassification"]);
    config["id2label"] = labels
        .iter()
        .enumerate()
        .map(|(i, l)| (i.to_string(), json!(l)))
        .collect::<serde_json::Map<_, _>>()
        .into();
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let bytes = std::fs::read(src.join("model.safetensors")).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    // A fixed pattern of weights, different for each label.
    let score: Vec<u8> = (0..labels.len() * hidden)
        .map(|i| ((i * 7 % 11) as f32 - 5.0) / 5.0)
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut tensors: Vec<(String, TensorView)> = st
        .tensors()
        .into_iter()
        .filter(|(n, _)| n != "lm_head.weight")
        .collect();
    tensors.push((
        "score.weight".into(),
        TensorView::new(safetensors::Dtype::F32, vec![labels.len(), hidden], &score).unwrap(),
    ));
    let out = safetensors::serialize(tensors, None).unwrap();
    std::fs::write(dir.join("model.safetensors"), out).unwrap();

    let vocab: ahash::AHashMap<String, u32> =
        (0..vocab_size).map(|i| (format!("t{i}"), i)).collect();
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
        .expect("the tiny classifier loads");
    let state = APIServer::new(
        "/nonexistent-ollama-models".to_string(),
        "/nonexistent-hf-models".to_string(),
    );
    state.engines.write().await.push(LoadedModelEntry::new(
        HEAD_MODEL.to_string(),
        engine,
        Some(1),
    ));
    (state, dir)
}

/// The status and the JSON of a route's answer.
async fn answer(response: Response) -> (axum::http::StatusCode, Value) {
    let status = response.status();
    let text = body_text(response).await;
    (status, serde_json::from_str(&text).unwrap_or(Value::Null))
}

async fn classify(state: &APIServer, body: Value) -> (axum::http::StatusCode, Value) {
    answer(openai::pooling::classify(State(state.clone()), Json(body)).await).await
}

async fn score(state: &APIServer, body: Value) -> (axum::http::StatusCode, Value) {
    answer(openai::pooling::score(State(state.clone()), Json(body)).await).await
}

async fn pooling(state: &APIServer, body: Value) -> (axum::http::StatusCode, Value) {
    let headers = axum::http::HeaderMap::new();
    answer(openai::vllm::pooling(State(state.clone()), headers, Json(body)).await).await
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5)
}

/// A 2-label head answers `/classify` with its labels, and `/pooling` with the same
/// probabilities, the last token's probabilities among those of `token_classify`.
#[tokio::test(flavor = "multi_thread")]
async fn a_head_answers_classify_and_the_classification_tasks() {
    let (state, dir) = server_with_head(&["negative", "positive"]).await;
    let texts = ["t3 t5 t7", "t9 t1"];
    let (status, v) = classify(&state, json!({"model": HEAD_MODEL, "input": texts})).await;
    assert!(status.is_success(), "{v}");
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    for (i, d) in data.iter().enumerate() {
        assert_eq!(d["index"], i);
        assert_eq!(d["num_classes"], 2);
        let probs = floats(&d["probs"]);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{probs:?}");
        let best = if probs[1] > probs[0] {
            "positive"
        } else {
            "negative"
        };
        assert_eq!(d["label"], best);
    }
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() >= 5);

    let (status, pooled) = pooling(
        &state,
        json!({"model": HEAD_MODEL, "task": "classify", "input": texts}),
    )
    .await;
    assert!(status.is_success(), "{pooled}");
    for (d, p) in data.iter().zip(pooled["data"].as_array().unwrap()) {
        assert!(close(&floats(&d["probs"]), &floats(&p["data"])), "{d} {p}");
    }

    let (status, tokens) = pooling(
        &state,
        json!({"model": HEAD_MODEL, "task": "token_classify", "input": texts[0]}),
    )
    .await;
    assert!(status.is_success(), "{tokens}");
    let rows = tokens["data"][0]["data"].as_array().unwrap();
    assert_eq!(rows.len(), 3, "one row per token");
    assert!(close(
        &floats(rows.last().unwrap()),
        &floats(&data[0]["probs"])
    ));

    let (status, v) = score(
        &state,
        json!({"model": HEAD_MODEL, "text_1": "t1", "text_2": "t2"}),
    )
    .await;
    assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{v}");
    assert!(v.to_string().contains("2 labels"), "{v}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A 1-label head scores a pair as a cross-encoder: the pair read together, the sigmoid of
/// the logit, the same score `/classify` gives the 2 texts written one after the other and
/// the same `/v1/rerank` gives.
#[tokio::test(flavor = "multi_thread")]
async fn a_head_with_one_label_scores_pairs_as_a_cross_encoder() {
    let (state, dir) = server_with_head(&["relevant"]).await;
    let (status, v) = score(
        &state,
        json!({"model": HEAD_MODEL, "text_1": "t4 t6", "text_2": ["t8", "t10 t12"]}),
    )
    .await;
    assert!(status.is_success(), "{v}");
    let scores: Vec<f32> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["score"].as_f64().unwrap() as f32)
        .collect();
    assert_eq!(scores.len(), 2);
    let (_, c) = classify(
        &state,
        json!({"model": HEAD_MODEL, "input": ["t4 t6 t8", "t4 t6 t10 t12"]}),
    )
    .await;
    let classified: Vec<f32> = c["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| floats(&d["probs"])[0])
        .collect();
    assert!(close(&scores, &classified), "{scores:?} {classified:?}");

    let reranked = answer(
        openai::openai_rerank(
            State(state.clone()),
            axum::http::HeaderMap::new(),
            Json(json!({"model": HEAD_MODEL, "query": "t4 t6", "documents": ["t8", "t10 t12"]})),
        )
        .await,
    )
    .await
    .1;
    let mut by_index = [0f32; 2];
    for r in reranked["results"].as_array().unwrap() {
        by_index[r["index"].as_u64().unwrap() as usize] =
            r["relevance_score"].as_f64().unwrap() as f32;
    }
    assert!(close(&by_index, &scores), "{reranked}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A model without a head, that does not pool and has no `yes` and `no` tokens, is refused
/// each classification task and the score, with the reason.
#[tokio::test(flavor = "multi_thread")]
async fn a_model_without_a_head_is_refused_with_the_reason() {
    let (state, dir, _) = server_with_tiny_model().await;
    let bad = axum::http::StatusCode::BAD_REQUEST;
    let (status, v) = classify(&state, json!({"model": MODEL, "input": "t1 t2"})).await;
    assert_eq!(status, bad, "{v}");
    assert!(v.to_string().contains("no classification head"), "{v}");
    for task in ["classify", "token_classify"] {
        let (status, v) =
            pooling(&state, json!({"model": MODEL, "task": task, "input": "t1"})).await;
        assert_eq!(status, bad, "{task}: {v}");
        assert!(v.to_string().contains("no classification head"), "{v}");
    }
    let (status, v) = pooling(
        &state,
        json!({"model": MODEL, "task": "draw", "input": "t1"}),
    )
    .await;
    assert_eq!(status, bad, "{v}");
    let (status, v) = score(
        &state,
        json!({"model": MODEL, "text_1": "t1", "text_2": "t2"}),
    )
    .await;
    assert_eq!(status, bad, "{v}");
    assert!(v.to_string().contains("cannot score"), "{v}");
    let _ = std::fs::remove_dir_all(&dir);
}
