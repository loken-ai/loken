//! The decode loop's arm for a model that drafts from its own head, run through the engine on a
//! small DeepSeek V4.1 model whose head is replaced by a scripted guess.

use super::*;
use crate::inference::engine::model_backend::DeepseekV41Backend;
use crate::inference::model::deepseek_v41::model::oracle_gate::synthetic;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

const VOCAB: u32 = 48;

/// What the scripted head does with the text the plain decode produced.
#[derive(Clone, Copy)]
enum Guess {
    /// The tokens the plain decode went on to produce.
    Right,
    /// Tokens the plain decode never produces at that place.
    Wrong,
}

fn tokenizer() -> Tokenizer {
    let vocab: ahash::AHashMap<String, u32> = (0..VOCAB).map(|i| (format!("t{i}"), i)).collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("t0".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
}

/// An engine holding the small model, drafting with `script` when there is one.
async fn engine_with(
    script: Option<Box<dyn FnMut(usize, u32, usize) -> Vec<u32> + Send>>,
) -> (LlmEngine, String) {
    let (model, prompt_ids) = synthetic(&[0, 2, 0], &[1], &[1], 0);
    let mut backend = DeepseekV41Backend::new(model, 0, 128);
    if let Some(script) = script {
        backend = backend.with_scripted_draft(script);
    }
    let engine = LlmEngine::with_config(InferenceConfig::default());
    *engine.model_state.lock().await = Some(LoadedModelState {
        name: "small".into(),
        num_layers: 3,
        hidden_size: 0,
        num_heads: 0,
        vocab_size: VOCAB as usize,
        context_length: 128,
        eos_token_id: VOCAB + 1,
        #[cfg(feature = "gpu")]
        moondream_graph: None,
        moondream_decode_count: 0,
        #[cfg(feature = "gpu")]
        kept_graphs: Default::default(),
        #[cfg(feature = "gpu")]
        draft_graph: Default::default(),
        #[cfg(feature = "gpu")]
        draft_step: Default::default(),
        draft_primed: false,
        draft_mark: None,
        exact_prefill: None,
        eos_token_ids_extra: Vec::new(),
        file_size: 0,
        model: Box::new(backend),
        tokenizer: std::sync::Arc::new(tokenizer()),
        device: Device::Cpu,
        image_embeds: None,
        audio: None,
        qwen35_vision: Vec::new(),
        image_embed_cache: None,
        grammar_factory: std::sync::OnceLock::new(),
        generation_defaults: Default::default(),
    });
    let prompt = prompt_ids
        .iter()
        .map(|t| format!("t{t}"))
        .collect::<Vec<_>>()
        .join(" ");
    (engine, prompt)
}

async fn run(engine: &LlmEngine, prompt: &str, new: usize) -> GenerationResult {
    let params = GenerationParams {
        max_tokens: Some(new),
        temperature: Some(0.0),
        repeat_penalty: Some(1.0),
        ..GenerationParams::default()
    };
    engine.generate(prompt, params).await.unwrap()
}

/// With the arm drafting right, drafting wrong, or drafting right then wrong, a greedy answer
/// is the one the plain decode gives, and the arm is the one that drafted.
#[tokio::test(flavor = "multi_thread")]
async fn a_native_draft_gives_the_plain_greedy_answer() {
    let new = 20;
    let (plain_engine, prompt) = engine_with(None).await;
    let plain = run(&plain_engine, &prompt, new).await;
    assert!(plain.tokens.len() > new, "the plain answer has its tokens");
    let truth = plain.tokens.clone();

    for (guess, agree, tag) in [
        (Guess::Right, usize::MAX, "full acceptance"),
        (Guess::Wrong, 0, "full rejection"),
        (Guess::Right, 1, "acceptance then rejection"),
    ] {
        let asked = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = asked.clone();
        let text = truth.clone();
        let script = Box::new(move |pos: usize, next: u32, k: usize| {
            // The cache holds `pos` tokens and `next` is the one at `pos` in the text.
            assert_eq!(text.get(pos), Some(&next), "the draft starts from the text");
            counter.fetch_add(1, Ordering::Relaxed);
            (1..=k)
                .map_while(|i| text.get(pos + i).map(|&t| (i, t)))
                .map(|(i, t)| match guess {
                    Guess::Right if i <= agree => t,
                    _ => (t + 1) % VOCAB,
                })
                .collect()
        });
        let (engine, prompt) = engine_with(Some(script)).await;
        let drafted = run(&engine, &prompt, new).await;
        assert_eq!(drafted.tokens, truth, "{tag}: the tokens");
        assert_eq!(drafted.text, plain.text, "{tag}: the text");
        assert!(asked.load(Ordering::Relaxed) > 0, "{tag}: the head drafted");
    }
}

/// The streamed decode drafts through the same arm and gives the same text.
#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_native_draft_gives_the_plain_greedy_text() {
    let new = 20;
    let params = || GenerationParams {
        max_tokens: Some(new),
        temperature: Some(0.0),
        repeat_penalty: Some(1.0),
        ..GenerationParams::default()
    };
    let collect = |engine: LlmEngine, prompt: String| async move {
        let mut rx = engine.generate_stream(&prompt, params()).await.unwrap();
        let mut text = String::new();
        while let Some(chunk) = rx.recv().await {
            text.push_str(&chunk.unwrap());
        }
        text
    };
    let (plain_engine, prompt) = engine_with(None).await;
    let full = run(&plain_engine, &prompt, new).await;
    let plain = collect(plain_engine, prompt.clone()).await;
    assert!(!plain.is_empty());

    for (agree, tag) in [(usize::MAX, "full acceptance"), (0, "full rejection")] {
        let asked = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = asked.clone();
        let text = full.tokens.clone();
        let script = Box::new(move |pos: usize, _next: u32, k: usize| {
            counter.fetch_add(1, Ordering::Relaxed);
            (1..=k)
                .map_while(|i| text.get(pos + i).map(|&t| (i, t)))
                .map(|(i, t)| if i <= agree { t } else { (t + 1) % VOCAB })
                .collect()
        });
        let (engine, prompt) = engine_with(Some(script)).await;
        let drafted = collect(engine, prompt).await;
        assert_eq!(drafted, plain, "{tag}");
        assert!(asked.load(Ordering::Relaxed) > 0, "{tag}: the head drafted");
    }
}
