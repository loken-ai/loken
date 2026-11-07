//! The released V4.1 Flash file served through the engine's public API.
//!
//! Ignored by default: it maps a file of hundreds of GB and runs on the host.
//! Run with
//!   cargo test --release --no-default-features --features cpu --lib -- --ignored wide_real_engine --nocapture
//!
//! The file is the store tag of `wide_sparse_real_tests`, resolved from `config.test.toml`.
//! The engine loads it as the server does and answers a greedy request.
//! The ids are the ones both routes gave in the model-level test.

use super::*;
use crate::inference::generic_transformer::wide_sparse_real_tests::{
    blob, prompt_ids, PROMPT, TAG,
};

/// The greedy ids both routes decoded after the prompt.
const GREEDY: [u32; 4] = [11111, 16, 660, 270];

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs the V4.1 Flash file in the store of config.test.toml and 44 GB of memory"]
async fn wide_real_engine_generates_the_greedy_ids_on_the_generic_arm() {
    // The load log names the arm and the plan: it is printed with the result.
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_test_writer()
        .try_init();
    assert!(blob(TAG).exists(), "the store holds the file of {TAG}");
    let store = crate::config::Config::load_test().get_ollama_models_dir();
    let engine = LlmEngine::with_config(InferenceConfig {
        model_id: TAG.to_string(),
        models_dir: Some(store),
        ..InferenceConfig::default()
    });
    let started = std::time::Instant::now();
    engine
        .load_model()
        .await
        .expect("the engine loads the file");
    println!("engine: loaded in {:.1}s", started.elapsed().as_secs_f32());

    // The model-level test prefills the begin token the file names, then the words: the engine
    // takes it as the conversation prefix, since a raw prompt carries none.
    let file = crate::tensor::quantized::gguf_source::SplitGguf::open(blob(TAG)).unwrap();
    let file_ids = prompt_ids(&file.merged_content());
    drop(file);
    let started = std::time::Instant::now();
    let got = engine
        .generate(
            PROMPT,
            GenerationParams {
                max_tokens: Some(GREEDY.len()),
                temperature: Some(0.0),
                repeat_penalty: Some(1.0),
                prefix_tokens: Some(file_ids[..1].to_vec()),
                ..GenerationParams::default()
            },
        )
        .await
        .expect("the engine answers");
    println!(
        "engine: answered in {:.1}s, prompt {} tokens",
        started.elapsed().as_secs_f32(),
        got.prompt_eval_count
    );
    let prompt_len = got.tokens.len() - GREEDY.len();
    println!("engine prompt ids: {:?}", &got.tokens[..prompt_len]);
    assert_eq!(
        &got.tokens[..prompt_len],
        file_ids,
        "the prompt the model-level test prefills"
    );
    let ids = &got.tokens[prompt_len..];
    println!("engine greedy ids: {ids:?}");
    println!("engine text: {:?}", got.text);
    assert_eq!(ids, GREEDY, "the ids the model-level test decoded");

    // The generic arm took the file: its backend gives the model back, a wide residual.
    let state = engine
        .model_state
        .lock()
        .await
        .take()
        .expect("a loaded model");
    let model = state
        .model
        .take_generic()
        .ok()
        .expect("the generic backend holds the file");
    assert!(
        model.sequence_mixers().is_ok(),
        "the generic model is the wide residual"
    );
}
