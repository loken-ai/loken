//! The placement preview route: where a load of a model would put its layers, from the plan
//! the load runs, without reading a weight.

use super::{validate_model_id, APIServer, ApiError};
use crate::inference::engine::llm_engine::LlmEngine;
use crate::inference::place::preview::PlacementPreview;
use axum::extract::{Json, State};

/// `POST /api/models/plan`: the placement of the model named by `model` (or `name`) on this
/// server's devices as they are now.
pub(crate) async fn plan_model(
    State(state): State<APIServer>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<PlacementPreview>, ApiError> {
    let model = request
        .get("model")
        .or_else(|| request.get("name"))
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::Validation("Missing 'model' field".to_string()))?;
    validate_model_id(model)?;
    let engine = LlmEngine::with_config(state.config_for_model(model));
    if engine.resolve_model_file(model).is_none() {
        return Err(ApiError::NotFound(format!("model {model} not found")));
    }
    engine
        .placement_preview()
        .await
        .map(Json)
        .map_err(ApiError::Validation)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::engine::llm_engine::placement_tests::tiny_model;

    /// The route answers the plan of a model it can find, and a missing model is not found.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_route_reports_the_plan_of_a_model_and_refuses_a_missing_one() {
        let tiny = tiny_model("route");
        let state = APIServer::new(
            "/nonexistent-ollama-models".to_string(),
            tiny.root.to_string_lossy().to_string(),
        );
        let Json(plan) = plan_model(
            State(state.clone()),
            Json(serde_json::json!({ "model": tiny.repo })),
        )
        .await
        .expect("the tiny checkpoint has a plan");
        let missing = plan_model(
            State(state),
            Json(serde_json::json!({ "model": "test/no-such-model" })),
        )
        .await;
        assert_eq!(plan.architecture, "llama");
        assert!(plan.layers > 0);
        assert!(plan.context_length > 0);
        let placed: usize = plan
            .devices
            .iter()
            .flat_map(|d| d.layers.iter().map(|[a, b]| b - a))
            .sum();
        assert_eq!(placed, plan.layers, "every layer has a device");
        assert!(matches!(missing, Err(ApiError::NotFound(_))));
    }
}
