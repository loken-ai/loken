//! The placement preview reports the placement the load builds: the same layers on the same
//! devices, and the same window.

use super::{InferenceConfig, LlmEngine};
use std::sync::atomic::{AtomicUsize, Ordering};

/// The tiny checkpoint the tests place.
const CHECKPOINT: &str = "tests/vectors/hf/llama";

/// A scratch directory holding the tiny checkpoint as the hub snapshot of a repository, with a
/// word-level vocabulary of its size.
pub(crate) struct TinyModel {
    /// The directory to remove.
    pub(crate) root: std::path::PathBuf,
    /// The repository's name, as a hub directory under `root` resolves it.
    pub(crate) repo: String,
    /// The snapshot directory, which names the checkpoint by its path.
    pub(crate) snapshot: std::path::PathBuf,
}

impl Drop for TinyModel {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

/// The tiny checkpoint copied for the test `tag`.
pub(crate) fn tiny_model(tag: &str) -> TinyModel {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(CHECKPOINT);
    let root = std::env::temp_dir().join(format!(
        "loken-placement-{}-{tag}-{}",
        std::process::id(),
        CALLS.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&root);
    let repo = format!("test/placement-{tag}");
    let snapshot = root
        .join(format!("models--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join("main");
    std::fs::create_dir_all(&snapshot).unwrap();
    for f in ["config.json", "model.safetensors"] {
        std::fs::copy(src.join(f), snapshot.join(f)).unwrap();
    }
    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(src.join("config.json")).unwrap()).unwrap();
    let vocab_size = config["vocab_size"].as_u64().unwrap() as u32;
    let vocab: ahash::AHashMap<String, u32> =
        (0..vocab_size).map(|i| (format!("t{i}"), i)).collect();
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
    TinyModel {
        root,
        repo,
        snapshot,
    }
}

/// The preview of the engine of `config`, then the placement its load builds: the runs of
/// layers on each device, `(device, ordinal, first, last)`, and the window.
async fn previewed_and_loaded(
    config: InferenceConfig,
) -> (
    Vec<(String, usize, u32, u32)>,
    usize,
    Vec<(String, usize, u32, u32)>,
    usize,
) {
    let engine = LlmEngine::with_config(config);
    let preview = engine
        .placement_preview()
        .await
        .expect("the tiny checkpoint has a plan");
    engine
        .load_model()
        .await
        .expect("the tiny checkpoint loads");
    let mut previewed: Vec<(String, usize, u32, u32)> = preview
        .devices
        .iter()
        .flat_map(|d| {
            d.layers
                .iter()
                .map(|[a, b]| (d.device.clone(), d.index, *a as u32, *b as u32 - 1))
        })
        .collect();
    previewed.sort_by_key(|r| r.2);
    let guard = engine.model_state.lock().await;
    let state = guard.as_ref().expect("a loaded model");
    let mut loaded = state.model.device_layer_distribution();
    loaded.sort_by_key(|r| r.2);
    (
        previewed,
        preview.context_length,
        loaded,
        state.context_length,
    )
}

#[tokio::test(flavor = "multi_thread")]
async fn the_preview_reports_the_layers_and_the_window_the_load_builds() {
    let tiny = tiny_model("agree");
    for context_length in [0, 64] {
        let (previewed, previewed_window, loaded, loaded_window) =
            previewed_and_loaded(InferenceConfig {
                model_id: tiny.snapshot.to_string_lossy().to_string(),
                context_length,
                ..InferenceConfig::default()
            })
            .await;
        assert!(!loaded.is_empty(), "the load reports its layers");
        assert_eq!(previewed, loaded, "layers at window {context_length}");
        assert_eq!(
            previewed_window, loaded_window,
            "window at configured {context_length}"
        );
    }
}
