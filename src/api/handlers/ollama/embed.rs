//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

// ============================================================================
// OpenAI-compatible Handlers (for backward compatibility)
// ============================================================================

/// List models (OpenAI-compatible format)
pub(crate) async fn list_models(
    State(state): State<APIServer>,
) -> Result<Json<ListModelsResponse>, ApiError> {
    info!("📋 GET /api/models - Listing models (OpenAI format)");

    let manager = state.model_manager.clone();

    match manager.list_models().await {
        Ok(mut models) => {
            #[cfg(feature = "media")]
            inject_local_boogu(&state, &mut models);
            let count = models.len();
            info!("   Found {} model(s)", count);
            // Stable alphabetical order matching /api/tags and /v1/models
            // (filesystem walk order would otherwise be platform-dependent
            // and shift between identical-content scans).
            models.sort_by(|a, b| a.id.cmp(&b.id));

            let model_infos: Vec<ModelInfo> = models
                .into_iter()
                .map(|m| {
                    let size_mb = m.size / (1024 * 1024);
                    info!("   • {} ({} MB, source: {})", m.id, size_mb, m.source);
                    ModelInfo::new(m.id, format!("{} MB", size_mb), m.size, m.downloaded_at)
                })
                .collect();

            info!("   ✅ List complete");
            Ok(Json(ListModelsResponse::new(model_infos)))
        }
        Err(e) => {
            error!("   ❌ Failed to list models: {}", e);
            Err(ApiError::Internal(format!("Failed to list models: {}", e)))
        }
    }
}

/// Version (GET /api/version) - Ollama-compatible
/// The Ollama API level this server implements. Clients gate features on this
/// number, so it names the Ollama release whose API is served, not this crate's
/// version, which travels in the `Server` header and `/health`.
pub(crate) const OLLAMA_API_VERSION: &str = "0.15.0";

pub(crate) async fn ollama_version() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "version": OLLAMA_API_VERSION,
        "loken": env!("CARGO_PKG_VERSION"),
    }))
}

// ============================================================================
// Helper Functions for Model Management
// ============================================================================

/// Root endpoint (GET /) - Ollama-compatible
pub(crate) async fn ollama_root() -> &'static str {
    "loken is running"
}

/// Head root endpoint (HEAD /) - Ollama-compatible
pub(crate) async fn ollama_head_root() -> StatusCode {
    StatusCode::OK
}

/// List running models (GET /api/ps) - Ollama-compatible
pub(crate) async fn ollama_ps(
    State(state): State<APIServer>,
) -> Result<Json<OllamaPsResponse>, ApiError> {
    // DEBUG, not INFO - /api/ps is a status endpoint that Ollama-
    // compatible web UIs (Open-WebUI, ollama-webui, etc.) poll on a
    // 5-10 s cadence. INFO would flood the server log with 6-12
    // listing lines / minute per polling client.
    debug!("📝 GET /api/ps - Listing running models");

    // Get all available models to look up digests
    let all_models = state.model_manager.list_models().await.unwrap_or_default();
    let model_by_id: std::collections::HashMap<_, _> = all_models
        .iter()
        .map(|m| (m.id.clone(), m.digest.clone()))
        .collect();

    let engines = state.engines.read().await;
    let mut models = Vec::new();

    for entry in engines.iter() {
        let model_size = entry.engine.get_model_size().await;
        // The window this engine serves, not the server-wide default: a client that
        // sizes its prompts on this listing was being told the default configured for
        // whatever model the config file names.
        let context_length = entry.engine.context_window().await.unwrap_or(0);
        let _device_name = entry.engine.get_device_name().await;

        // Get digest from model manager, or use a placeholder.
        // Strip the "sha256:" prefix for the API response (Ollama returns
        // just the hash) - use the explicit strip_prefix to keep clippy
        // happy.
        let digest = model_by_id
            .get(&entry.model_id)
            .cloned()
            .unwrap_or_else(|| "sha256:0000000000000000000000000000000000000000".to_string());
        let digest = digest
            .strip_prefix("sha256:")
            .map(str::to_string)
            .unwrap_or(digest);

        // Calculate expiration time dynamically (every request, current time + keep_alive)
        let keep_alive_mins = entry.keep_alive_minutes.unwrap_or(state.default_keep_alive);
        let expires_at = if keep_alive_mins == -1 {
            "never".to_string()
        } else {
            // Dynamic: current time + keep_alive duration (recalculated each request)
            let expires = chrono::Utc::now() + chrono::Duration::minutes(keep_alive_mins);
            expires.to_rfc3339()
        };

        // Determine VRAM size based on device (matches Ollama: size_vram for PROCESSOR field calculation)
        // If model is on GPU: size_vram = model_size (shows "100% GPU")
        // If model is on CPU: size_vram = 0 (shows "100% CPU")
        // If model is split: size_vram = GPU portion (shows "X% CPU/Y% GPU")
        let size_vram = entry.engine.get_gpu_portion_size().await;

        let format_str = if entry.model_id.contains(".gguf") || model_size < 5_000_000_000 {
            "gguf"
        } else {
            "safetensors"
        };
        let parameter_size = estimate_parameter_size(model_size, format_str);
        let details = OllamaModelDetails {
            format: format_str.to_string(),
            family: infer_model_family(&entry.model_id).to_string(),
            families: None,
            parameter_size,
            quantization_level: None,
        };

        debug!(
            "   Model: {} (digest: {})",
            entry.model_id,
            &digest[..12.min(digest.len())]
        );

        models.push(OllamaPsModel {
            name: entry.model_id.clone(),
            model: entry.model_id.clone(),
            size: model_size,
            digest,
            details,
            expires_at,
            size_vram,
            context_length: context_length as i32,
        });
    }

    // Include image / audio (ASR) / TTS engines so the catalog
    // matches /api/loaded. /api/ps used to ONLY enumerate text-LLM
    // engines - SDK clients probing the running set saw an empty
    // list when only Z-Image / Whisper / Parler were warm. The
    // structured fields (digest, size_vram, etc.) fall back to
    // zeros for these engines since their size accessors aren't
    // wired through; the entry's presence is the contract.
    let placeholder_digest = "0".repeat(64);

    #[cfg(feature = "image")]
    if let Some(img_info) = state.image_engine.get_loaded_model_info().await {
        // Sum per-layer memory_bytes for an approximate footprint.
        let img_size: u64 = img_info
            .layer_distribution
            .iter()
            .map(|d| d.memory_bytes)
            .sum();
        // VRAM portion = bytes on CUDA / OpenCL devices. CPU layers
        // don't count toward size_vram (matches the LLM path's
        // gpu_portion_size accounting).
        let img_vram: u64 = img_info
            .layer_distribution
            .iter()
            .filter(|d| d.device_type == "CUDA" || d.device_type == "Arc")
            .map(|d| d.memory_bytes)
            .sum();
        let parameter_size = estimate_parameter_size(img_size, "safetensors");
        models.push(OllamaPsModel {
            name: img_info.name.clone(),
            model: img_info.name,
            size: img_size,
            digest: placeholder_digest.clone(),
            details: OllamaModelDetails {
                format: "safetensors".to_string(),
                family: infer_model_family(&img_info.model_type).to_string(),
                families: None,
                parameter_size,
                quantization_level: None,
            },
            expires_at: "never".to_string(),
            size_vram: img_vram,
            context_length: 0,
        });
    }

    #[cfg(feature = "audio")]
    if let Some(asr_info) = state.audio_engine.get_loaded_model_info().await {
        // Measured across the load, like the image engine - see
        // `AudioModelInfo::resident_bytes`. This reported 0, on the theory that the
        // entry's presence was the contract; a dashboard reads 0 as "costs nothing".
        models.push(OllamaPsModel {
            name: asr_info.name.clone(),
            model: asr_info.name.clone(),
            size: asr_info.resident_bytes,
            digest: placeholder_digest.clone(),
            details: OllamaModelDetails {
                format: "safetensors".to_string(),
                family: infer_model_family(&asr_info.name).to_string(),
                families: None,
                parameter_size: "unknown".to_string(),
                quantization_level: None,
            },
            expires_at: "never".to_string(),
            size_vram: asr_info.resident_bytes,
            context_length: 0,
        });
    }

    #[cfg(feature = "audio")]
    if let Some(tts_name) = state.tts_engine.loaded_name().await {
        // Measured across the load, as for the image and ASR engines.
        let tts_vram = state.tts_engine.resident_bytes().await;
        models.push(OllamaPsModel {
            name: tts_name.clone(),
            model: tts_name.clone(),
            size: tts_vram,
            digest: placeholder_digest,
            details: OllamaModelDetails {
                format: "safetensors".to_string(),
                family: infer_model_family(&tts_name).to_string(),
                families: None,
                parameter_size: "unknown".to_string(),
                quantization_level: None,
            },
            expires_at: "never".to_string(),
            size_vram: tts_vram,
            context_length: 0,
        });
    }

    // Mirror 7fea356 / 6615f9a - deterministic alphabetical order
    // for catalog consumers across endpoints. Loaded-engine iteration
    // order here is insertion-dependent.
    models.sort_by(|a, b| a.name.cmp(&b.name));
    debug!("   ✓ {} models running", models.len());
    Ok(Json(OllamaPsResponse { models }))
}

/// Copy model (POST /api/copy) - Ollama-compatible
pub(crate) async fn ollama_copy_model(
    State(state): State<APIServer>,
    OllamaJson(request): OllamaJson<OllamaCopyRequest>,
) -> Result<StatusCode, ApiError> {
    // Empty / whitespace-only / traversal patterns all rejected by
    // validate_model_id (post 52abf6b). The earlier explicit
    // destination.trim().is_empty() check is now redundant.
    validate_model_id(&request.source)?;
    validate_model_id(&request.destination)?;
    let source = normalize_model_id(&request.source);
    let destination = normalize_model_id(&request.destination);
    info!("Copying model: {} -> {}", source, destination);

    if source == destination {
        return Err(ApiError::Validation(
            "`source` and `destination` must differ".into(),
        ));
    }

    // approach - users manage their own ollama dirs). Without an
    // explicit existence check, /api/copy would return 200 OK even
    // when the source doesn't exist locally - misleading clients
    // into thinking the copy succeeded. Probe the manifest path
    // up front so unknown sources return 404 cleanly.
    let path = state
        .model_manager
        .resolve_path(&source, "ollama")
        .await
        .map_err(|e| {
            let msg = e.to_string();
            if msg.contains("not found") || msg.contains("No such file") {
                ApiError::NotFound(format!("Source model '{source}' not found"))
            } else {
                ApiError::Internal(format!("Failed to resolve source '{source}': {msg}"))
            }
        })?;
    if !path.exists() {
        return Err(ApiError::NotFound(format!(
            "Source model '{source}' not found on disk"
        )));
    }

    state
        .model_manager
        .copy_model(&source, &destination)
        .await
        .map_err(|e| {
            let msg = e.to_string();
            // Same 404-vs-500 classification as 88c1e14 + 69106c2.
            if msg.contains("not found") || msg.contains("No such file") {
                ApiError::NotFound(format!("Model '{source}' not found"))
            } else {
                ApiError::Internal(format!("Failed to copy model: {msg}"))
            }
        })?;

    Ok(StatusCode::OK)
}

/// Generate embeddings (POST /api/embed) - Ollama-compatible
pub(crate) async fn ollama_embed(
    State(state): State<APIServer>,
    OllamaJson(request): OllamaJson<OllamaEmbedRequest>,
) -> Result<Json<OllamaEmbedResponse>, ApiError> {
    validate_model_id(&request.model)?;
    let model_name = normalize_model_id(&request.model);
    info!("Embed request for model: {}", model_name);

    // Parse input: can be a single string or array of strings
    let inputs: Vec<String> = match &request.input {
        serde_json::Value::String(s) => vec![s.clone()],
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str().map(std::string::ToString::to_string))
            .collect(),
        _ => {
            return Err(ApiError::Validation(
                "'input' must be a string or array of strings".to_string(),
            ))
        }
    };

    if inputs.is_empty() {
        return Err(ApiError::Validation(
            "'input' must not be empty".to_string(),
        ));
    }
    // Match the OpenAI `/v1/embeddings` validation: empty / whitespace-
    // only entries are 400 rather than producing a degenerate embedding
    // that just confuses the caller.
    if let Some(idx) = inputs.iter().position(|s| s.trim().is_empty()) {
        return Err(ApiError::Validation(format!(
            "'input[{idx}]' is empty; embeddings require non-empty text"
        )));
    }
    // Per-input + array-size caps (parity with /v1/embeddings).
    // 8192 chars ≈ 8k tokens - OpenAI's documented per-input cap.
    // 2048 array entries - OpenAI's documented batch cap.
    const EMBED_MAX_CHARS: usize = 8192;
    const EMBED_MAX_BATCH: usize = 2048;
    if let Some((idx, s)) = inputs
        .iter()
        .enumerate()
        .find(|(_, s)| s.chars().count() > EMBED_MAX_CHARS)
    {
        return Err(ApiError::Validation(format!(
            "'input[{idx}]' is {} chars; limit is {EMBED_MAX_CHARS}",
            s.chars().count()
        )));
    }
    if inputs.len() > EMBED_MAX_BATCH {
        return Err(ApiError::Validation(format!(
            "'input' has {} entries; limit is {EMBED_MAX_BATCH}",
            inputs.len()
        )));
    }

    // Auto-load the model on first use (like /api/chat does) honouring
    // the request's `keep_alive` knob. Without this, /api/embed only
    // worked against an already-loaded model - Ollama's own server
    // auto-loads on demand for embed too.
    let keep_alive_minutes = state.get_effective_keep_alive(request.keep_alive.as_deref());
    if let Err(e) = state.ensure_loaded(&model_name, keep_alive_minutes).await {
        return Err(ApiError::NotFound(format!(
            "Model '{model_name}' could not be auto-loaded: {e}"
        )));
    }
    // Get loaded engine
    let engine = state.get_engine(&model_name).await?;
    if request.truncate == Some(false) {
        // Without truncation an input past the context is a client error, as Ollama
        // reports it; with it (the default) the engine cuts the input to the window.
        for (idx, text) in inputs.iter().enumerate() {
            if let Some(n) = engine.count_tokens(text).await {
                if let Some(window) = engine.context_window().await {
                    if n > window {
                        return Err(ApiError::Validation(format!(
                            "'input[{idx}]' is {n} tokens, past the context of {window}, and truncate is false"
                        )));
                    }
                }
            }
        }
    }

    // Reset expiration timer on use (like Ollama)
    state.reset_expiration(&model_name).await;

    let start = std::time::Instant::now();

    // Generate embeddings for each input
    let mut embeddings = Vec::new();
    for input in &inputs {
        match engine.generate_embeddings(input).await {
            Ok(embedding) => embeddings.push(embedding),
            Err(e) => {
                return Err(ApiError::Internal(format!(
                    "Failed to generate embeddings: {}",
                    e
                )));
            }
        }
    }

    let total_duration = start.elapsed().as_nanos() as u64;
    // Use the same token-count heuristic as the OpenAI endpoint so the
    // two surfaces report consistent prompt-side counts for identical
    // inputs (whitespace count under-estimates for languages without
    // spaces and over-estimates for code).
    let prompt_eval_count: u64 = inputs.iter().map(|s| estimate_token_count(s)).sum();

    // `dimensions` keeps the leading coordinates, as OpenAI's `dimensions` does.
    let embeddings: Vec<Vec<f32>> = match request.dimensions {
        Some(d) if d > 0 => embeddings
            .into_iter()
            .map(|mut v| {
                v.truncate(d);
                v
            })
            .collect(),
        _ => embeddings,
    };
    Ok(Json(OllamaEmbedResponse {
        model: model_name,
        embeddings,
        total_duration: Some(total_duration),
        load_duration: Some(0),
        prompt_eval_count: Some(prompt_eval_count),
    }))
}

/// Create model (POST /api/create) - Ollama-compatible
/// Creates a model configuration (Modelfile-like) by writing metadata.
pub(crate) async fn ollama_create_model(
    State(state): State<APIServer>,
    OllamaJson(request): OllamaJson<OllamaCreateRequest>,
) -> Result<Response, ApiError> {
    // Type-level length caps on system + modelfile (declared on the struct;
    // see b9d05fa / 0ce73c8). validate_request humanizes the error envelope
    // into "<field> length ..." so clients see a useful message.
    validate_request(&request)?;
    validate_model_id(&request.name)?;
    if let Some(from) = request.from.as_deref() {
        validate_model_id(from)?;
    }
    // Cap the structured `parameters` value. Like `system`, it gets
    // serialized into loken_config.json - without a cap, a 50 MB
    // nested-object payload would translate to a 50 MB disk write
    // (and 50 MB held resident in memory through the serialize +
    // tokio::fs::write step). 64 KiB serialized matches the modelfile
    // cap; legitimate parameter bags are kilobyte-scale at most.
    if let Some(params) = request.parameters.as_ref() {
        let params_size = serde_json::to_string(params).map(|s| s.len()).unwrap_or(0);
        const PARAMETERS_MAX_BYTES: usize = 64 * 1024;
        if params_size > PARAMETERS_MAX_BYTES {
            return Err(ApiError::Validation(format!(
                "parameters JSON is {params_size} bytes; cap at {PARAMETERS_MAX_BYTES} (parameter bags are kilobyte-scale knob lists, not payloads)"
            )));
        }
    }
    let model_name = normalize_model_id(&request.name);
    info!("Create model request: {}", model_name);
    if request
        .quantize
        .as_deref()
        .is_some_and(|q| !q.trim().is_empty())
    {
        return Err(ApiError::Validation(
            "`quantize` is not done here: pull the tag that carries the quantisation you want"
                .into(),
        ));
    }
    if request.adapters.as_ref().is_some_and(|a| !a.is_empty()) {
        return Err(ApiError::Validation(
            "`adapters` are not built into a model here; load a LoRA through the adapters API"
                .into(),
        ));
    }
    // A model built from uploaded blobs: a manifest naming the weights, and the
    // template, system, parameters and license as layers of their own.
    if let Some(files) = request.files.as_ref().filter(|f| !f.is_empty()) {
        let store = state.model_manager.ollama();
        let (name, tag) = match model_name.rsplit_once(':') {
            Some((n, t)) => (n.to_string(), t.to_string()),
            None => (model_name.clone(), "latest".to_string()),
        };
        let mut layers: Vec<serde_json::Value> = Vec::new();
        let mut weights: Option<std::path::PathBuf> = None;
        for (file, digest) in files {
            let hex = digest.strip_prefix("sha256:").ok_or_else(|| {
                ApiError::Validation(format!("`files.{file}`: digest must be sha256:<hex>"))
            })?;
            let path = store.blobs_dir().join(format!("sha256-{hex}"));
            let size = std::fs::metadata(&path)
                .map_err(|_| {
                    ApiError::NotFound(format!("`files.{file}`: blob {digest} is not uploaded"))
                })?
                .len();
            let kind = if file.to_ascii_lowercase().contains("mmproj")
                || file.to_ascii_lowercase().contains("projector")
            {
                "projector"
            } else {
                "model"
            };
            if kind == "model" && weights.is_none() {
                weights = Some(path.clone());
            }
            layers.push(serde_json::json!({
                "mediaType": format!("application/vnd.ollama.image.{kind}"),
                "digest": digest,
                "size": size,
            }));
        }
        let weights =
            weights.ok_or_else(|| ApiError::Validation("`files` names no model weights".into()))?;
        let mut text_layer = |kind: &str, content: Option<&str>| -> Result<(), ApiError> {
            if let Some(c) = content.filter(|c| !c.is_empty()) {
                let (digest, size) = store
                    .write_blob(c.as_bytes())
                    .map_err(|e| ApiError::Internal(format!("blob: {e}")))?;
                layers.push(serde_json::json!({
                    "mediaType": format!("application/vnd.ollama.image.{kind}"),
                    "digest": digest,
                    "size": size,
                }));
            }
            Ok(())
        };
        text_layer("template", request.template.as_deref())?;
        text_layer("system", request.system.as_deref())?;
        text_layer("license", request.license.as_deref())?;
        let params = request
            .parameters
            .as_ref()
            .map(|p| serde_json::to_string(p).unwrap_or_default());
        text_layer("params", params.as_deref())?;
        let (_, arch, quant, declared) = super::models::gguf_facts(&weights);
        let config = serde_json::json!({
            "model_format": "gguf",
            "model_family": arch.clone().unwrap_or_default(),
            "model_families": arch.map(|a| vec![a]).unwrap_or_default(),
            "model_type": declared.map(super::models::format_parameter_count).unwrap_or_default(),
            "file_type": quant.unwrap_or_default(),
            "architecture": "amd64",
            "os": "linux",
        });
        let (cfg_digest, cfg_size) = store
            .write_blob(&serde_json::to_vec(&config).unwrap_or_default())
            .map_err(|e| ApiError::Internal(format!("blob: {e}")))?;
        let manifest = serde_json::json!({
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": cfg_digest,
                "size": cfg_size,
            },
            "layers": layers,
        });
        store
            .write_manifest(&name, &tag, &manifest)
            .map_err(|e| ApiError::Internal(format!("manifest: {e}")))?;
        return Ok(create_status_response(
            request.stream,
            &model_name,
            &["writing manifest"],
        ));
    }

    // If 'from' is specified, copy the base model first
    let base_model = request
        .from
        .as_deref()
        .or_else(|| request.modelfile.as_deref().and_then(parse_modelfile_from));

    if let Some(base) = base_model {
        // Modelfile-derived FROM lines bypass the explicit `from`
        // field's validate_model_id call above. Re-validate here so
        // a `FROM ../../../etc/passwd` line in the modelfile can't
        // sneak past path-traversal screening into resolve_path.
        validate_model_id(base)?;
        let base_normalized = normalize_model_id(base);

        // Check if base model exists, pull if needed
        let base_path = state
            .model_manager
            .resolve_path(&base_normalized, "ollama")
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to resolve base model path: {}", e)))?;

        if !base_path.exists() {
            info!("Base model not found locally, pulling: {}", base_normalized);
            state
                .model_manager
                .pull_model_with_source(&base_normalized, "ollama", None)
                .await
                .map_err(|e| ApiError::Internal(format!("Failed to pull base model: {}", e)))?;
        }

        // Copy base model to new name (if different)
        if base_normalized != model_name {
            state
                .model_manager
                .copy_model(&base_normalized, &model_name)
                .await
                .map_err(|e| ApiError::Internal(format!("Failed to copy base model: {}", e)))?;
        }
    }

    // Write custom configuration if system prompt or parameters provided
    if request.system.is_some() || request.parameters.is_some() {
        let model_path = state
            .model_manager
            .resolve_path(&model_name, "ollama")
            .await
            .map_err(|e| ApiError::Internal(format!("Failed to resolve model path: {}", e)))?;

        if model_path.exists() {
            let config = serde_json::json!({
                "system": request.system,
                "parameters": request.parameters,
                "quantize": request.quantize,
            });
            let config_path = model_path.join("loken_config.json");
            let config_str = serde_json::to_string_pretty(&config)
                .map_err(|e| ApiError::Internal(format!("Failed to serialize config: {}", e)))?;
            tokio::fs::write(&config_path, config_str)
                .await
                .map_err(|e| ApiError::Internal(format!("Failed to write config: {}", e)))?;
        }
    }

    let steps = if base_model.is_some() {
        &["using existing layer", "writing manifest"][..]
    } else {
        &["writing manifest"][..]
    };
    Ok(create_status_response(request.stream, &model_name, steps))
}

/// The answer of `/api/create`: NDJSON status frames ending in `success`, as Ollama
/// streams them, or the last frame alone.
fn create_status_response(stream: bool, model_name: &str, steps: &[&str]) -> Response {
    use axum::response::IntoResponse;
    if !stream {
        return Json(serde_json::json!({ "status": "success" })).into_response();
    }
    let mut body = String::new();
    body.push_str(
        &serde_json::json!({ "status": format!("creating model '{model_name}'") }).to_string(),
    );
    body.push('\n');
    for step in steps {
        body.push_str(&serde_json::json!({ "status": step }).to_string());
        body.push('\n');
    }
    body.push_str(&serde_json::json!({ "status": "success" }).to_string());
    body.push('\n');
    Response::builder()
        .header("content-type", "application/x-ndjson")
        .body(axum::body::Body::from(body))
        .unwrap_or_else(|_| Json(serde_json::json!({ "status": "success" })).into_response())
}

/// Push model (POST /api/push) - Ollama-compatible
/// Returns 403 Forbidden: loken does not support pushing models to a registry.
pub(crate) async fn ollama_push_model(
    OllamaJson(request): OllamaJson<OllamaPushRequest>,
) -> Result<Response, ApiError> {
    // Reject malformed names (path traversal, oversize, NULs) up front
    // - even on a 403, echoing an unvalidated name back into logs is
    // sloppy and could be used for log-injection.
    validate_model_id(&request.name)?;
    info!("Push model request (denied by policy): {}", request.name);

    let body = serde_json::json!({
        "error": "loken does not support pushing models to a remote registry"
    });

    Ok((StatusCode::FORBIDDEN, Json(body)).into_response())
}

/// Normalize an already-validated blob digest to its on-disk form.
///
/// Ollama writes blobs to `blobs/sha256-<lowercase-hex>` regardless of
/// which input form the client used. Without normalization a request
/// with `sha256:HEX` would HEAD a non-existent file (returning 404 even
/// when the data is on disk) and POST would write to a parallel
/// `sha256:hex` path that the rest of the manager wouldn't find.
///
/// Caller MUST ensure `validate_blob_digest` accepted the input first  -
/// the function relies on the structure (one of two prefixes, exactly
/// 64 hex chars after).
pub(super) fn normalize_blob_digest(digest: &str) -> String {
    let hex = digest
        .strip_prefix("sha256-")
        .or_else(|| digest.strip_prefix("sha256:"))
        .unwrap_or(digest);
    let mut out = String::with_capacity(7 + hex.len());
    out.push_str("sha256-");
    for c in hex.chars() {
        out.push(c.to_ascii_lowercase());
    }
    out
}

/// Validate that an Ollama blob digest matches the documented form:
/// `sha256-<64 lowercase hex>` (or `sha256:<64 hex>` for older clients).
/// Rejects any path traversal characters (`..`, `/`, `\\`) up front so
/// blob handlers can join the digest into `blobs_dir` without escaping.
pub(super) fn validate_blob_digest(digest: &str) -> Result<(), ApiError> {
    // Cheap path-traversal guards first.
    if digest.is_empty()
        || digest.contains('/')
        || digest.contains('\\')
        || digest.contains("..")
        || digest.starts_with('.')
    {
        return Err(ApiError::Validation(format!(
            "invalid blob digest '{digest}'"
        )));
    }
    // Accept `sha256-<hex>` (new form) and `sha256:<hex>` (legacy form).
    let hex = digest
        .strip_prefix("sha256-")
        .or_else(|| digest.strip_prefix("sha256:"))
        .ok_or_else(|| {
            ApiError::Validation(format!(
                "blob digest must start with 'sha256-' or 'sha256:'; got '{digest}'"
            ))
        })?;
    if hex.len() != 64 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(ApiError::Validation(format!(
            "blob digest hex must be 64 hex chars; got '{hex}' (len={})",
            hex.len()
        )));
    }
    Ok(())
}

/// Check blob existence (HEAD /api/blobs/:digest) - Ollama-compatible
pub(crate) async fn ollama_head_blob(
    State(state): State<APIServer>,
    Path(digest): Path<String>,
) -> StatusCode {
    info!("Checking blob: {}", digest);
    if validate_blob_digest(&digest).is_err() {
        return StatusCode::BAD_REQUEST;
    }
    // Normalize `sha256:hex` (legacy form) and uppercase hex to the
    // on-disk `sha256-lowerhex` filename layout that Ollama writes.
    let on_disk = normalize_blob_digest(&digest);

    let blobs_dir = std::path::PathBuf::from(&state.ollama_models_dir).join("blobs");
    let blob_path = blobs_dir.join(&on_disk);

    if blob_path.exists() {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    }
}

/// Upload blob (POST /api/blobs/:digest) - Ollama-compatible
pub(crate) async fn ollama_create_blob(
    State(state): State<APIServer>,
    Path(digest): Path<String>,
    body: axum::body::Bytes,
) -> Result<StatusCode, ApiError> {
    info!("Creating blob: {} ({} bytes)", digest, body.len());
    validate_blob_digest(&digest)?;
    if body.is_empty() {
        return Err(ApiError::Validation("blob body must not be empty".into()));
    }
    // Verify the body's sha256 matches the URL-supplied digest. Real
    // Ollama enforces this - without it a client could write arbitrary
    // bytes under any sha256-named blob, leaving the model manifest
    // pointing at content that contradicts its own digest. Compute is
    // ~200 MB/s on commodity CPU which is acceptable for the rare
    // create-blob path (model layers are 100MB-10GB, computed once).
    let expected_hex = digest
        .strip_prefix("sha256-")
        .or_else(|| digest.strip_prefix("sha256:"))
        .ok_or_else(|| ApiError::Validation("digest prefix".into()))?
        .to_lowercase();
    let body_clone = body.clone();
    let actual_hex = tokio::task::spawn_blocking(move || {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&body_clone);
        let bytes = hasher.finalize();
        bytes.iter().map(|b| format!("{b:02x}")).collect::<String>()
    })
    .await
    .map_err(|e| ApiError::Internal(format!("hash join: {e}")))?;
    if actual_hex != expected_hex {
        return Err(ApiError::Validation(format!(
            "body sha256 ({actual_hex}) does not match URL digest ({expected_hex})"
        )));
    }

    let blobs_dir = std::path::PathBuf::from(&state.ollama_models_dir).join("blobs");
    tokio::fs::create_dir_all(&blobs_dir)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to create blobs directory: {}", e)))?;

    // Write under the canonical `sha256-lowerhex` filename so a later
    // HEAD/read by either input form (or by the model_manager) finds
    // the same file. Without normalization, a `sha256:HEX` POST would
    // create a parallel `sha256:HEX` file the rest of the manager
    // can't see.
    let blob_path = blobs_dir.join(normalize_blob_digest(&digest));
    tokio::fs::write(&blob_path, &body)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to write blob: {}", e)))?;

    Ok(StatusCode::CREATED)
}

/// The request shape of `POST /api/embeddings`, the endpoint that preceded `/api/embed`
/// and that the LangChain and LlamaIndex integrations still call: one `prompt`, one
/// `embedding` back.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct OllamaLegacyEmbeddingsRequest {
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    #[serde(default)]
    pub keep_alive: Option<serde_json::Value>,
}

pub(crate) async fn ollama_embeddings_legacy(
    State(state): State<APIServer>,
    OllamaJson(request): OllamaJson<OllamaLegacyEmbeddingsRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if request.prompt.trim().is_empty() {
        // Ollama answers an empty prompt with an empty vector, and so does this.
        return Ok(Json(serde_json::json!({ "embedding": [] })));
    }
    let modern: OllamaEmbedRequest = serde_json::from_value(serde_json::json!({
        "model": request.model,
        "input": request.prompt,
        "options": request.options,
        "keep_alive": request.keep_alive,
    }))
    .map_err(|e| ApiError::Validation(format!("embeddings: {e}")))?;
    let Json(mut response) = ollama_embed(State(state), OllamaJson(modern)).await?;
    let embedding = response.embeddings.pop().unwrap_or_default();
    Ok(Json(serde_json::json!({ "embedding": embedding })))
}

/// The embedding routes as served: a request whose model this node's catalogue lacks goes
/// to a peer that holds it, as it is; the rest is answered here.
async fn embed_where_the_model_is<R: serde::Serialize>(
    state: &APIServer,
    headers: &axum::http::HeaderMap,
    model: &str,
    relay: &crate::api::handlers::Relay,
    request: &R,
) -> Option<axum::response::Response> {
    let body = serde_json::to_value(request).ok()?;
    let model = normalize_model_id(model);
    let holds = |n: &crate::distributed::membership::NodeState| {
        crate::distributed::routing::can_serve(n, &model)
    };
    let served_here = holds(&state.local_node_state().await);
    crate::api::handlers::route_media_to_holder(
        state,
        headers,
        &model,
        served_here,
        true,
        0,
        holds,
        relay,
        &body,
    )
    .await
}

pub(crate) async fn ollama_embed_routed(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    OllamaJson(request): OllamaJson<OllamaEmbedRequest>,
) -> Result<axum::response::Response, ApiError> {
    use axum::response::IntoResponse;
    if let Some(relayed) = embed_where_the_model_is(
        &state,
        &headers,
        &request.model,
        &crate::api::handlers::OLLAMA_EMBED,
        &request,
    )
    .await
    {
        return Ok(relayed);
    }
    Ok(ollama_embed(State(state), OllamaJson(request))
        .await?
        .into_response())
}

pub(crate) async fn ollama_embeddings_legacy_routed(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
    OllamaJson(request): OllamaJson<OllamaLegacyEmbeddingsRequest>,
) -> Result<axum::response::Response, ApiError> {
    use axum::response::IntoResponse;
    if let Some(relayed) = embed_where_the_model_is(
        &state,
        &headers,
        &request.model,
        &crate::api::handlers::OLLAMA_EMBEDDINGS,
        &request,
    )
    .await
    {
        return Ok(relayed);
    }
    Ok(ollama_embeddings_legacy(State(state), OllamaJson(request))
        .await?
        .into_response())
}
