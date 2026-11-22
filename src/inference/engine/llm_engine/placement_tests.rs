//! The placement preview reports the placement the load builds: the same layers on the same
//! devices, and the same window.

use super::{InferenceConfig, LlmEngine};
use crate::inference::place::preview::{DevicePreview, PlacementPreview};
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
pub(super) async fn previewed_and_loaded(
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

/// A scratch directory for the test `tag`.
fn scratch(tag: &str) -> std::path::PathBuf {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "loken-placement-{}-{tag}-{}",
        std::process::id(),
        CALLS.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The runs of layers of a preview, `(device, ordinal, first, end)`.
fn runs(preview: &PlacementPreview) -> Vec<(String, usize, usize, usize)> {
    preview
        .devices
        .iter()
        .flat_map(|d| {
            d.layers
                .iter()
                .map(|[a, b]| (d.device.clone(), d.index, *a, *b))
        })
        .collect()
}

/// The runs of layers of a plan, as a preview names them.
fn plan_runs(
    plan: &crate::inference::place::layer_executor::HeteroPlan,
) -> Vec<(String, usize, usize, usize)> {
    plan.segments
        .iter()
        .map(|s| {
            let (device, index) = crate::inference::place::preview::device_label(s.kind);
            (device.to_string(), index, s.layer_start, s.layer_end)
        })
        .collect()
}

/// A file its own loader places whole is previewed as the block its load places through the
/// ladder: the file's bytes on the device the load builds on.
#[tokio::test(flavor = "multi_thread")]
async fn a_file_placed_whole_is_previewed_on_the_device_its_load_builds_on() {
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::GgmlDType;
    let dir = scratch("whole");
    let model = dir.join("model.gguf");
    let entry = GgufEntry {
        name: "text_model.transformer.embd.wte.weight".to_string(),
        dims: vec![4, 4],
        dtype: GgmlDType::F32,
        data: vec![0u8; 64],
    };
    write_gguf_with_metadata(&model, &[], &[entry]).unwrap();
    let engine = LlmEngine::with_config(InferenceConfig {
        model_id: "whole".to_string(),
        models_dir: Some(model.clone()),
        ..InferenceConfig::default()
    });
    let preview = engine
        .placement_preview()
        .await
        .expect("a file placed whole has a preview");
    let file_len = std::fs::metadata(&model).unwrap().len();
    let built =
        crate::inference::place::ladder::load_whole("whole", preview.weight_bytes, 0, |on| {
            Ok::<_, String>(on.location())
        })
        .unwrap();
    assert_eq!(preview.refused, None);
    assert_eq!(preview.layers, 1);
    assert_eq!(preview.weight_bytes, file_len);
    assert_eq!(preview.kv_format, "f32");
    assert!(preview.context_length > 0);
    let on = &preview.devices[0];
    assert_eq!(on.layers, vec![[0, 1]]);
    assert_eq!(on.weight_bytes, file_len);
    let device = match on.device.as_str() {
        "CPU" => crate::tensor::Device::Cpu.location(),
        _ => crate::inference::place::ladder::device_of(
            crate::inference::place::layer_executor::DeviceKind::Cuda(on.index),
        )
        .location(),
    };
    assert_eq!(device, built);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A text-image pair is previewed from the placement its load runs: one block of its weights,
/// on the plan the ladder builds it on.
#[tokio::test(flavor = "multi_thread")]
async fn a_text_image_pair_is_previewed_from_the_placement_its_load_runs() {
    let dir = scratch("pair");
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type": "clip", "text_config": {}, "vision_config": {}}"#,
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    std::fs::write(dir.join("preprocessor_config.json"), "{}").unwrap();
    let bytes = vec![0u8; 16];
    let views: Vec<(String, safetensors::tensor::TensorView)> = [
        "text_model.embeddings.token_embedding.weight",
        "vision_model.embeddings.patch_embedding.weight",
        "text_projection.weight",
        "visual_projection.weight",
    ]
    .iter()
    .map(|name| {
        (
            name.to_string(),
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &bytes)
                .unwrap(),
        )
    })
    .collect();
    safetensors::serialize_to_file(views, None, &dir.join("model.safetensors")).unwrap();
    assert!(crate::inference::model::clip::pair::claims_checkpoint_dir(
        &dir
    ));
    let engine = LlmEngine::with_config(InferenceConfig {
        model_id: "pair".to_string(),
        ..InferenceConfig::default()
    });
    let preview = super::placement::preview_file(&dir, &engine.placement_ask()).unwrap();
    let placement = crate::inference::model::clip::pair::placement(&dir);
    let built = placement
        .load(
            "text-image pair",
            || false,
            |plan| Ok::<_, String>(plan.clone()),
        )
        .unwrap();
    let weights = crate::inference::model::clip::pair::weight_bytes(&dir);
    assert_eq!(preview.architecture, "clip");
    assert_eq!(preview.weight_bytes, weights);
    assert_eq!(runs(&preview), plan_runs(&built));
    assert_eq!(runs(&preview), vec![("CPU".to_string(), 0, 0, 1)]);
    assert_eq!(preview.host.weight_bytes, weights);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The routed experts of a wide residual fill the room each card keeps, as the load sizes it:
/// the offer less the weights, the cache and a request's work, plus the cache room the working
/// window leaves, then a card outside the plan.
#[test]
fn the_routed_experts_fill_the_room_the_cards_keep() {
    use super::placement::{card_experts, WideSizing};
    const GB: u64 = 1 << 30;
    let devices = vec![
        DevicePreview {
            device: "CUDA".to_string(),
            index: 0,
            layers: vec![[0, 2]],
            weight_bytes: 4 * GB,
            kv_bytes: GB,
            offered_bytes: Some(10 * GB),
            expert_bytes: 0,
        },
        DevicePreview {
            device: "CPU".to_string(),
            index: 0,
            ..DevicePreview::default()
        },
    ];
    let sizing = WideSizing {
        runtime_reserve: GB,
        working: None,
        experts: 4,
        expert_bytes: vec![GB, GB],
    };
    let outside = [(0, 10 * GB), (1, 3 * GB)];
    assert_eq!(
        card_experts(&devices, &outside, &sizing),
        vec![(0, 10 * GB, 4 * GB), (1, 3 * GB, 2 * GB)]
    );
    // A model that holds its whole stack lends the cache room the working window leaves.
    let lending = WideSizing {
        working: Some(0),
        ..sizing.clone()
    };
    let experts = card_experts(&devices, &outside, &lending);
    assert_eq!(experts, vec![(0, 10 * GB, 5 * GB), (1, 3 * GB, 2 * GB)]);
    // The cards then hold the experts the host would have kept.
    let mut preview = PlacementPreview {
        weight_bytes: 20 * GB,
        devices,
        ..PlacementPreview::default()
    };
    preview.add_card_experts(&experts);
    assert_eq!(preview.devices[0].weight_bytes, 9 * GB);
    assert_eq!(preview.devices[0].expert_bytes, 5 * GB);
    assert_eq!(preview.devices[2].index, 1);
    assert!(preview.devices[2].layers.is_empty());
    assert_eq!(preview.host.weight_bytes, 20 * GB - 11 * GB);
}
