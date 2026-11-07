//! The decode loop's arm for a model that drafts from its own head, run through the engine on the
//! generic model of a small Sinkhorn wide-residual file: port of `native_draft_tests` from the
//! V4.1 backend to the generic one.
//!
//! The head is the block draft the file carries, the one a file beside it carries, or a scripted
//! guess in its place.

use super::*;
use crate::inference::engine::model_backend::GenericBackend;
use crate::inference::generic_transformer::block_draft::ScriptedDraft;
use crate::inference::generic_transformer::block_draft_tests::{
    generic_of, prompt, toy_files, toy_with_companion,
};
use crate::inference::model::deepseek_v41::model::tests::VOCAB;
use crate::tensor::quantized::gguf_file::open_mapped;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;

/// The band layout of the V4.1 test's model.
const LAYOUT: (&[u32], &[u32], &[u32]) = (&[0, 2, 0], &[1], &[1]);

/// What the scripted head does with the text the plain decode produced.
#[derive(Clone, Copy)]
enum Guess {
    /// The tokens the plain decode went on to produce.
    Right,
    /// Tokens the plain decode never produces at that place.
    Wrong,
}

/// Which head the model drafts with.
enum Head {
    /// None: the plain decode.
    Plain,
    /// The block draft the file carries.
    File,
    /// A scripted guess.
    Scripted(ScriptedDraft),
}

pub(super) fn tokenizer() -> Tokenizer {
    let vocab: ahash::AHashMap<String, u32> =
        (0..VOCAB as u32).map(|i| (format!("t{i}"), i)).collect();
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("t0".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
}

/// An engine holding the small generic model with `head`.
async fn engine_with(head: Head) -> (LlmEngine, String) {
    let (file, _) = toy_files(LAYOUT, matches!(head, Head::File));
    let mut model = generic_of(file);
    if let Head::Scripted(script) = head {
        model.set_scripted_block_draft(script).unwrap();
    }
    engine_of(model).await
}

/// An engine holding `model`, a generic model of the small file.
pub(super) async fn engine_of(model: GenericHeteroTransformer) -> (LlmEngine, String) {
    let num_layers = model.segment_layers();
    engine_holding(Box::new(GenericBackend(model)), num_layers).await
}

/// An engine holding `backend`, a backend of `num_layers` layers of the small file.
pub(super) async fn engine_holding(
    backend: BoxedModelBackend,
    num_layers: usize,
) -> (LlmEngine, String) {
    let engine = LlmEngine::with_config(InferenceConfig::default());
    *engine.model_state.lock().await = Some(LoadedModelState {
        name: "small".into(),
        num_layers,
        hidden_size: 0,
        num_heads: 0,
        vocab_size: VOCAB,
        context_length: 128,
        eos_token_id: VOCAB as u32 + 1,
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
        model: backend,
        tokenizer: std::sync::Arc::new(tokenizer()),
        device: Device::Cpu,
        image_embeds: None,
        audio: None,
        qwen35_vision: Vec::new(),
        image_embed_cache: None,
        grammar_factory: std::sync::OnceLock::new(),
        generation_defaults: Default::default(),
    });
    let prompt = prompt()
        .iter()
        .map(|t| format!("t{t}"))
        .collect::<Vec<_>>()
        .join(" ");
    (engine, prompt)
}

pub(super) fn params(new: usize) -> GenerationParams {
    GenerationParams {
        max_tokens: Some(new),
        temperature: Some(0.0),
        repeat_penalty: Some(1.0),
        ..GenerationParams::default()
    }
}

pub(super) async fn run(engine: &LlmEngine, prompt: &str, new: usize) -> GenerationResult {
    engine.generate(prompt, params(new)).await.unwrap()
}

pub(super) async fn collect(engine: LlmEngine, prompt: String, new: usize) -> String {
    let mut rx = engine.generate_stream(&prompt, params(new)).await.unwrap();
    let mut text = String::new();
    while let Some(chunk) = rx.recv().await {
        text.push_str(&chunk.unwrap());
    }
    text
}

/// Port of `a_native_draft_gives_the_plain_greedy_answer`: with the generic arm drafting right,
/// drafting wrong, or drafting right then wrong, a greedy answer is the one the plain decode
/// gives, and the arm is the one that drafted.
#[tokio::test(flavor = "multi_thread")]
async fn a_block_draft_gives_the_plain_greedy_answer() {
    let new = 20;
    let (plain_engine, prompt) = engine_with(Head::Plain).await;
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
        let script: ScriptedDraft = Box::new(move |pos: usize, next: u32, k: usize| {
            // The cache holds `pos` tokens and `next` is the one at `pos` in the text.
            assert_eq!(text.get(pos), Some(&next), "the draft starts from the text");
            counter.fetch_add(1, Ordering::Relaxed);
            (1..=k)
                .map_while(|i| text.get(pos + i).map(|&t| (i, t)))
                .map(|(i, t)| match guess {
                    Guess::Right if i <= agree => t,
                    _ => (t + 1) % VOCAB as u32,
                })
                .collect()
        });
        let (engine, prompt) = engine_with(Head::Scripted(script)).await;
        let drafted = run(&engine, &prompt, new).await;
        assert_eq!(drafted.tokens, truth, "{tag}: the tokens");
        assert_eq!(drafted.text, plain.text, "{tag}: the text");
        assert!(asked.load(Ordering::Relaxed) > 0, "{tag}: the head drafted");
    }
}

/// Port of `a_streamed_native_draft_gives_the_plain_greedy_text`: the streamed decode drafts
/// through the same arm and gives the same text.
#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_block_draft_gives_the_plain_greedy_text() {
    let new = 20;
    let (plain_engine, prompt) = engine_with(Head::Plain).await;
    let full = run(&plain_engine, &prompt, new).await;
    let plain = collect(plain_engine, prompt.clone(), new).await;
    assert!(!plain.is_empty());

    for (agree, tag) in [(usize::MAX, "full acceptance"), (0, "full rejection")] {
        let asked = std::sync::Arc::new(AtomicUsize::new(0));
        let counter = asked.clone();
        let text = full.tokens.clone();
        let script: ScriptedDraft = Box::new(move |pos: usize, _next: u32, k: usize| {
            counter.fetch_add(1, Ordering::Relaxed);
            (1..=k)
                .map_while(|i| text.get(pos + i).map(|&t| (i, t)))
                .map(|(i, t)| {
                    if i <= agree {
                        t
                    } else {
                        (t + 1) % VOCAB as u32
                    }
                })
                .collect()
        });
        let (engine, prompt) = engine_with(Head::Scripted(script)).await;
        let drafted = collect(engine, prompt, new).await;
        assert_eq!(drafted, plain, "{tag}");
        assert!(asked.load(Ordering::Relaxed) > 0, "{tag}: the head drafted");
    }
}

/// The block draft the file carries drafts through the same arm, in the plain and the streamed
/// decode, and gives the plain greedy text.
#[tokio::test(flavor = "multi_thread")]
async fn the_block_draft_of_the_file_gives_the_plain_greedy_text() {
    let new = 20;
    let (plain_engine, prompt) = engine_with(Head::Plain).await;
    let plain = run(&plain_engine, &prompt, new).await;
    let (engine, prompt) = engine_with(Head::File).await;
    let drafted = run(&engine, &prompt, new).await;
    assert_eq!(drafted.tokens, plain.tokens, "the tokens");
    assert_eq!(drafted.text, plain.text, "the text");

    let (plain_engine, prompt) = engine_with(Head::Plain).await;
    let plain = collect(plain_engine, prompt.clone(), new).await;
    let (engine, prompt) = engine_with(Head::File).await;
    assert_eq!(
        collect(engine, prompt, new).await,
        plain,
        "the streamed text"
    );
}

/// The block draft of a file beside the model, taken by the generic load arm, drafts through the
/// same arm and gives the plain greedy text.
/// A share of the layers takes none.
#[tokio::test(flavor = "multi_thread")]
async fn the_block_draft_beside_the_file_gives_the_plain_greedy_text() {
    let new = 20;
    let path = toy_with_companion(LAYOUT);
    let open = || generic_of(open_mapped(&path).unwrap());
    let alone = open();
    assert!(!alone.has_block_draft(), "the model file carries no draft");
    let share = with_companion_draft(open(), &path, Some((0, 2)));
    assert!(!share.has_block_draft(), "a share takes no draft");
    let model = with_companion_draft(open(), &path, None);
    assert!(
        model.has_block_draft(),
        "the draft beside the file is taken"
    );

    let (plain_engine, prompt) = engine_of(alone).await;
    let plain = run(&plain_engine, &prompt, new).await;
    let (engine, prompt) = engine_of(model).await;
    let drafted = run(&engine, &prompt, new).await;
    assert_eq!(drafted.tokens, plain.tokens, "the tokens");
    assert_eq!(drafted.text, plain.text, "the text");

    let (plain_engine, prompt) = engine_of(open()).await;
    let plain = collect(plain_engine, prompt.clone(), new).await;
    let (engine, prompt) = engine_of(with_companion_draft(open(), &path, None)).await;
    assert_eq!(
        collect(engine, prompt, new).await,
        plain,
        "the streamed text"
    );
    let _ = std::fs::remove_dir_all(path.parent().unwrap());
}
