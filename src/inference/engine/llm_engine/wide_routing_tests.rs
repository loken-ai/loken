//! The routing heat of a Sinkhorn wide-residual file the engine loads, against the steps the V4.1
//! engine arm took, on the small file of the frozen V4.1 oracle.

use super::block_draft_tests::tokenizer;
use super::*;
use crate::inference::generic_transformer::wide_sparse_routing_tests::{
    layers, profile_file, toy_dir, turning, v41_heat, ROUTED,
};
use crate::inference::generic_transformer::wide_sparse_sibling as sibling;

/// The heat of the generic model the engine holds.
async fn engine_heat(engine: &LlmEngine) -> Vec<Vec<usize>> {
    let mut state = engine.model_state.lock().await;
    let held = state.as_mut().expect("a loaded model");
    let backend = std::mem::replace(
        &mut held.model,
        Box::new(crate::inference::engine::model_backend::TakenBackend),
    );
    let model = match backend.take_generic() {
        Ok(model) => model,
        Err(_) => panic!("the generic arm took the file"),
    };
    let heat = model.routing_heat().to_vec();
    held.model = Box::new(crate::inference::engine::model_backend::GenericBackend(
        model,
    ));
    heat
}

/// The engine ranks the experts of a file that carries no prior by the profile it is given, the
/// prior of a companion file beside it ordering the uncounted, as the V4.1 arm did.
#[tokio::test(flavor = "multi_thread")]
async fn the_engine_ranks_a_wide_files_experts_as_the_v41_arm_did() {
    let model = toy_dir("engine", None);
    tokenizer()
        .save(model.with_file_name("tokenizer.json"), false)
        .unwrap();
    sibling::write_prior(
        &model.with_file_name("side"),
        &turning(layers(), ROUTED as u32),
    )
    .unwrap();
    let mut counts = vec![0u32; layers() * ROUTED];
    counts[ROUTED + 6] = 3;
    counts[2 * ROUTED + 1] = 5;
    let profile = profile_file(&model, layers(), ROUTED, counts);
    let config = |profile: Option<&std::path::Path>| InferenceConfig {
        model_id: "toy".to_string(),
        models_dir: Some(model.clone()),
        context_length: 128,
        expert_profile: profile.map(|p| p.to_string_lossy().into_owned()),
        ..InferenceConfig::default()
    };
    for given in [None, Some(profile.as_path())] {
        let engine = LlmEngine::with_config(config(given));
        engine
            .load_model()
            .await
            .expect("the engine loads the file");
        let want = v41_heat(&model, given);
        assert_eq!(engine_heat(&engine).await, want, "profile {given:?}");
    }
    let _ = std::fs::remove_dir_all(model.parent().unwrap());
}
