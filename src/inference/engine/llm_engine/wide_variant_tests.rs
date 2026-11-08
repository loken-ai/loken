//! A Sinkhorn wide-residual model whose manifest names a variant file, loaded by the engine from
//! a model store as the server loads it, on the small file of the frozen V4.1 oracle.
//!
//! The answers are compared with the V4.1 backend holding the same 2 files under the same policy.

use super::block_draft_tests::{engine_holding, run, tokenizer};
use super::*;
use crate::inference::engine::model_backend::{DeepseekV41Backend, GenericBackend, TakenBackend};
use crate::inference::generic_transformer::block_draft_tests::prompt;
use crate::inference::generic_transformer::wide_sparse_variant_tests::{
    v41_refusal, v41_with_variant, write_pair, Variant, CHOICES, LAYOUT,
};
use crate::inference::place::expert_variants::VariantChoice;

/// Tokens each request generates.
const NEW: usize = 20;

/// A model store holding the model `toy` whose manifest names the model's file and a variant
/// file of `kind`, its tokenizer beside the blobs.
///
/// Returns the store's root, the model blob and the variant blob.
/// The caller removes the root.
fn store(tag: &str, kind: Variant) -> (PathBuf, PathBuf, PathBuf) {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "loken-wide-variant-store-{tag}-{}-{seq}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    let blobs = root.join("blobs");
    let manifests = root
        .join("manifests")
        .join("registry.ollama.ai")
        .join("library")
        .join("toy");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&manifests).unwrap();
    let digest = |n: u8| format!("{:064x}", n);
    let model = blobs.join(format!("sha256-{}", digest(1)));
    let variant = blobs.join(format!("sha256-{}", digest(2)));
    write_pair(&model, &variant, kind);
    tokenizer()
        .save(blobs.join("tokenizer.json"), false)
        .unwrap();
    let layer = |media: &str, n: u8| serde_json::json!({ "mediaType": media, "digest": format!("sha256:{}", digest(n)) });
    let manifest = serde_json::json!({
        "schemaVersion": 2,
        "layers": [
            layer("application/vnd.ollama.image.model", 1),
            layer("application/vnd.loken.image.model.variant", 2),
        ],
    });
    std::fs::write(manifests.join("latest"), manifest.to_string()).unwrap();
    (root, model, variant)
}

fn config(root: &std::path::Path, choice: VariantChoice) -> InferenceConfig {
    InferenceConfig {
        model_id: "toy".to_string(),
        models_dir: Some(root.to_path_buf()),
        context_length: 128,
        expert_variants: choice,
        ..InferenceConfig::default()
    }
}

/// The variant each routed expert of the engine's generic model reads, per held layer.
async fn engine_choice(engine: &LlmEngine) -> Vec<Vec<usize>> {
    let mut state = engine.model_state.lock().await;
    let held = state.as_mut().expect("a loaded model");
    let backend = std::mem::replace(&mut held.model, Box::new(TakenBackend));
    let model = match backend.take_generic() {
        Ok(model) => model,
        Err(_) => panic!("the generic arm took the file"),
    };
    let choice = model.expert_variant_choice();
    held.model = Box::new(GenericBackend(model));
    choice
}

/// The prompt as the words of the small tokenizer.
fn words() -> String {
    prompt()
        .iter()
        .map(|t| format!("t{t}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Under every policy, the engine loads a file whose manifest names a variant through the generic
/// arm, reads each expert from the file the V4.1 backend reads it from, and answers with its
/// tokens and text.
#[tokio::test(flavor = "multi_thread")]
async fn the_engine_serves_a_variant_file_through_the_generic_arm_as_the_v41_backend() {
    let (root, model, variant) = store("serve", Variant::Q4);
    for choice in CHOICES {
        let engine = LlmEngine::with_config(config(&root, choice));
        engine
            .load_model()
            .await
            .expect("the engine loads the 2 files");
        let (v41, plan) = v41_with_variant(&model, &variant, choice);
        assert_eq!(engine_choice(&engine).await, plan.chosen, "{choice:?}");
        let got = run(&engine, &words(), NEW).await;
        assert!(
            got.tokens.len() > NEW,
            "{choice:?}: the answer has its tokens"
        );
        let (old, prompt) = engine_holding(
            Box::new(DeepseekV41Backend::new(v41, 0, 128)),
            LAYOUT.0.len(),
        )
        .await;
        let want = run(&old, &prompt, NEW).await;
        assert_eq!(
            got.tokens, want.tokens,
            "{choice:?}: tokens of the V4.1 backend"
        );
        assert_eq!(got.text, want.text, "{choice:?}: text of the V4.1 backend");
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// A variant file of another checkpoint fails the load with the words of the V4.1 arm.
#[tokio::test(flavor = "multi_thread")]
async fn the_engine_refuses_a_broken_variant_as_the_v41_arm_did() {
    let (root, model, variant) = store("broken", Variant::OtherNorm);
    let want = v41_refusal(&model, &variant, "toy").expect("the V4.1 arm refused");
    let got = LlmEngine::with_config(config(&root, VariantChoice::Plan))
        .load_model()
        .await
        .expect_err("the engine refused the variant")
        .to_string();
    assert!(got.contains(&want), "{got}");
    let _ = std::fs::remove_dir_all(&root);
}
