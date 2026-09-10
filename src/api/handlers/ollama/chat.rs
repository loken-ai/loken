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
        // The window this engine serves, not the server-wide default. A client that
        // sizes its prompts on the listing was being told the default configured for
        // whatever model the file names, whatever model it asked about.
        let context_length = entry.engine.context_window().await;

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
            context_length: context_length.map(|n| n as u32),
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

    // The transcription model, its two towers by device.
    #[cfg(feature = "audio")]
    if let Some(asr_info) = state.audio_engine.get_loaded_model_info().await {
        let parts = state.audio_engine.parts().unwrap_or_default();
        let device = state.audio_engine.loaded_device().await;
        push_parts(
            &mut loaded_models,
            &asr_info.name,
            "loaded",
            device,
            Some(asr_info.resident_bytes),
            &parts,
        );
    }

    // The voice model, by part: a backend has one to three.
    #[cfg(feature = "audio")]
    if let Some(tts_name) = state.tts_engine.loaded_name().await {
        let parts = state.tts_engine.parts().unwrap_or_default();
        let device = state.tts_engine.loaded_device().await;
        let size = state.tts_engine.resident_bytes().await;
        push_parts(
            &mut loaded_models,
            &tts_name,
            "loaded",
            device,
            (size > 0).then_some(size),
            &parts,
        );
    }

    // The sound pipeline kept between renders.
    #[cfg(feature = "audio")]
    if let Some(parts) = crate::inference::model::stable_audio::resident_parts() {
        push_parts(
            &mut loaded_models,
            "stable-audio",
            "loaded",
            None,
            None,
            &parts,
        );
    }

    // The separation model, kept once loaded.
    #[cfg(feature = "audio")]
    if let Some(parts) = crate::api::handlers::separate::resident_parts() {
        push_parts(
            &mut loaded_models,
            crate::api::handlers::separate::SEPARATION_MODEL,
            "loaded",
            None,
            None,
            &parts,
        );
    }

    // What the video pipeline stages on the host between renders.
    #[cfg(feature = "video")]
    {
        let parts = crate::inference::model::wan::pipeline::host_parts();
        if !parts.is_empty() {
            push_parts(&mut loaded_models, "wan", "loaded", None, None, &parts);
        }
    }

    // What renders now, by the name the request gave: one entry per loaded part with its
    // layers by device, or the bare job while nothing is loaded yet.
    for job in state.media_jobs() {
        push_parts(
            &mut loaded_models,
            &job.model,
            &job.status(),
            None,
            None,
            &job.parts,
        );
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

/// One listing entry per placed part of `model`, named `model (part)`, with the part's
/// layers by device; a part with no name is the model itself. A model with no parts yet
/// is one entry with its status alone, so a render that has loaded nothing is still
/// listed.
pub(crate) fn push_parts(
    out: &mut Vec<LoadedModelInfo>,
    model: &str,
    status: &str,
    device: Option<String>,
    size_bytes: Option<u64>,
    parts: &[(
        String,
        Vec<crate::inference::serve::progress::placement::Placed>,
    )],
) {
    if parts.is_empty() {
        out.push(LoadedModelInfo {
            model: model.to_string(),
            status: status.to_string(),
            device,
            size_bytes,
            num_layers: None,
            layer_distribution: None,
            context_length: None,
        });
        return;
    }
    for (part, runs) in parts {
        let layer_distribution: Vec<crate::api::types::LayerDistribution> = runs
            .iter()
            .map(|r| {
                let (device_type, device_id) = match r.device {
                    crate::tensor::DeviceLocation::Cpu => ("CPU".to_string(), 0),
                    crate::tensor::DeviceLocation::Cuda { gpu_id } => ("CUDA".to_string(), gpu_id),
                };
                crate::api::types::LayerDistribution {
                    device_type,
                    device_id,
                    layer_start: r.layer_start as u32,
                    layer_end: r.layer_end.saturating_sub(1) as u32,
                    memory_bytes: r.bytes,
                }
            })
            .collect();
        let bytes: u64 = runs.iter().map(|r| r.bytes).sum();
        out.push(LoadedModelInfo {
            model: if part.is_empty() {
                model.to_string()
            } else {
                format!("{model} ({part})")
            },
            status: status.to_string(),
            device: layer_distribution
                .first()
                .map(|d| d.device_type.clone())
                .or_else(|| device.clone()),
            size_bytes: if bytes > 0 { Some(bytes) } else { size_bytes },
            num_layers: Some(runs.iter().map(|r| r.layer_end).max().unwrap_or(0) as u32),
            layer_distribution: Some(layer_distribution),
            context_length: None,
        });
    }
}
