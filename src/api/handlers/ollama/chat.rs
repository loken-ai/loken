//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

// ============================================================================
// Custom Handlers
// ============================================================================

/// List loaded models with topology information
pub(crate) async fn list_loaded_models(
    State(state): State<APIServer>,
) -> Result<Json<ListLoadedModelsResponse>, ApiError> {
    // DEBUG, not INFO - the GUI Hardware tab polls this every 2s.
    debug!("Listing loaded models with topology");

    let engines = state.engines.read().await;
    let mut loaded_models = Vec::new();

    for entry in engines.iter() {
        let device_name = entry.engine.get_device_name().await;
        let model_size = entry.engine.get_model_size().await;

        // Build layer distribution from the actual loaded model state
        let layer_dist_info = entry.engine.get_layer_distribution().await;
        let (num_layers, layer_distribution) = match layer_dist_info {
            Some((total, distributions)) => (Some(total as u32), Some(distributions)),
            None => (None, None),
        };

        loaded_models.push(LoadedModelInfo {
            model: entry.model_id.clone(),
            status: "loaded".to_string(),
            device: Some(device_name),
            size_bytes: Some(model_size),
            num_layers,
            layer_distribution,
            context_length: Some(state.default_inference_config.context_length as u32),
        });
    }

    // Include image model if loaded
    #[cfg(feature = "image")]
    if let Some(img_info) = state.image_engine.get_loaded_model_info().await {
        let primary_device = img_info
            .layer_distribution
            .first()
            .map(|d| d.device_type.clone())
            .unwrap_or_else(|| "CPU".to_string());
        let layer_distribution: Vec<crate::api::types::LayerDistribution> = img_info
            .layer_distribution
            .iter()
            .map(|d| crate::api::types::LayerDistribution {
                device_type: d.device_type.clone(),
                device_id: d.device_id,
                layer_start: d.layer_range.0,
                layer_end: d.layer_range.1,
                memory_bytes: d.memory_bytes,
            })
            .collect();
        loaded_models.push(LoadedModelInfo {
            model: img_info.name,
            status: "loaded".to_string(),
            device: Some(primary_device),
            size_bytes: None,
            num_layers: Some(img_info.total_layers),
            layer_distribution: Some(layer_distribution),
            context_length: None,
        });
    }

    // Include whisper (ASR) if loaded - surfaces in the GUI's
    // Hardware tab so the user can see VRAM is held by the audio
    // engine, not just text/image engines. Reads the actual loaded
    // device kind so a CPU-only fallback also reports correctly.
    #[cfg(feature = "audio")]
    if let Some(asr_info) = state.audio_engine.get_loaded_model_info().await {
        let device = state
            .audio_engine
            .loaded_device()
            .await
            .unwrap_or_else(|| "CUDA".to_string());
        loaded_models.push(LoadedModelInfo {
            model: asr_info.name,
            status: "loaded".to_string(),
            device: Some(device),
            size_bytes: None,
            num_layers: None,
            layer_distribution: None,
            context_length: None,
        });
    }

    // Include parler-tts (or whatever TTS checkpoint is warm) if
    // loaded. The TtsEngine doesn't expose a topology - just the
    // name + device kind - but listing it here is enough for the
    // GUI's Hardware tab "Loaded Models" panel. Parler-tts actually
    // runs on CUDA (load_parler_blocking calls pick_device which
    // prefers CUDA0), not CPU as previously hard-coded.
    #[cfg(feature = "audio")]
    if let Some(tts_name) = state.tts_engine.loaded_name().await {
        let device = state
            .tts_engine
            .loaded_device()
            .await
            .unwrap_or_else(|| "CPU".to_string());
        loaded_models.push(LoadedModelInfo {
            model: tts_name,
            status: "loaded".to_string(),
            device: Some(device),
            size_bytes: None,
            num_layers: None,
            layer_distribution: None,
            context_length: None,
        });
    }

    // What renders now, by the name the request gave; a media job holds no layers.
    for job in state.media_jobs() {
        loaded_models.push(LoadedModelInfo {
            model: job.model,
            status: format!("rendering {}", job.kind),
            device: None,
            size_bytes: None,
            num_layers: None,
            layer_distribution: None,
            context_length: None,
        });
    }

    // Stable alphabetical order for catalog parity with /api/ps,
    // /api/tags, /api/models, and /v1/models. Without this the
    // order is text-engine-insertion -> image -> whisper -> parler,
    // which shifts whenever a model loads/unloads.
    loaded_models.sort_by(|a, b| a.model.cmp(&b.model));

    Ok(Json(ListLoadedModelsResponse::new(loaded_models)))
}

/// Validate a model
pub(crate) async fn validate_model(
    State(state): State<APIServer>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<crate::inference::load::model_manager::ModelValidationResult>, ApiError> {
    let model_name = request
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::Validation("Missing 'name' field".to_string()))?;
    validate_model_id(model_name)?;

    info!(
        "🔍 POST /api/models/validate - Validating model: {}",
        model_name
    );

    match state.model_manager.validate_model(model_name).await {
        Ok(result) => {
            info!("   Exists: {}, Valid: {}", result.exists, result.is_valid);
            info!(
                "   GGUF: {}, SafeTensors: {}",
                result.has_gguf, result.has_safetensors
            );
            if result.total_size > 0 {
                info!("   Size: {} MB", result.total_size / (1024 * 1024));
            }
            if !result.files.is_empty() {
                info!("   Files: {} found", result.files.len());
            }
            // PRIVACY-OK: a checkpoint-validation status, not user content.
            info!("   ✅ Validation complete: {}", result.message);

            Ok(Json(result))
        }
        Err(e) => {
            error!("   ❌ Validation failed: {}", e);
            Err(ApiError::Internal(format!(
                "Failed to validate model: {}",
                e
            )))
        }
    }
}

/// Repair a model
pub(crate) async fn repair_model(
    State(state): State<APIServer>,
    Json(request): Json<serde_json::Value>,
) -> Result<Json<crate::inference::load::model_manager::ModelMetadata>, ApiError> {
    let model_name = request
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::Validation("Missing 'name' field".to_string()))?;
    // Path-traversal screen. Repair flows through model_manager.repair_model
    // (currently a no-op) but defence-in-depth: the next implementation
    // could legitimately touch the filesystem under model_name. Gate
    // here so it can't ever reach a delete/rewrite call shaped like `..`.
    validate_model_id(model_name)?;

    info!(
        "🔧 POST /api/models/repair - Repairing model: {}",
        model_name
    );

    // First unload if loaded
    info!("   Attempting to unload model if loaded...");
    match state.unload_model(model_name).await {
        Ok(UnloadOutcome::Freed) => info!("   Model unloaded"),
        Ok(UnloadOutcome::NotResident) => info!("   Model was not loaded; nothing to free"),
        // Repair reloads from disk anyway, so this is not fatal - but saying it
        // unloaded when it did not is what makes a stuck resident invisible.
        Err(e) => warn!("   Model NOT unloaded before repair: {e}"),
    }

    match state.model_manager.repair_model(model_name).await {
        Ok(metadata) => {
            info!(
                "   ✅ Repair successful: {} ({} MB)",
                metadata.id,
                metadata.size / (1024 * 1024)
            );

            Ok(Json(metadata))
        }
        Err(e) => {
            error!("   ❌ Repair failed: {}", e);
            Err(ApiError::Internal(format!("Failed to repair model: {}", e)))
        }
    }
}
