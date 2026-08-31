//! System/ops endpoints: health, devices, distributed stats, multi-device
//! config, layer perf, inflight status and speculative-draft attach.

use super::*;

pub(crate) async fn health_check(
    state: axum::extract::State<APIServer>,
) -> Json<serde_json::Value> {
    // Cheap status snapshot: how many models the server has warm,
    // uptime in seconds, and the build version. Useful for k8s
    // readiness probes + simple monitoring dashboards that hit /health
    // periodically. The full Stats line goes to logs every 10s; this
    // endpoint just exposes the most relevant counters.
    //
    // `loaded_models` counts ALL engines that are currently warm:
    // text LLMs (state.engines is a Vec, one entry per loaded LLM)
    // plus image / audio (ASR) / TTS (each engine holds 0 or 1).
    // Pre-fix this only counted LLMs, so a server with Z-Image
    // loaded reported loaded_models=0 - misleading for k8s probes
    // that bound a "ready" threshold to that count.
    let llm_count = state.engines.read().await.len();
    // A family that is not compiled holds nothing, which is the honest answer here and
    // keeps the readiness count adding up rather than needing a second shape.
    #[cfg(feature = "image")]
    let image_loaded = state.image_engine.is_loaded().await;
    #[cfg(not(feature = "image"))]
    let image_loaded = false;
    #[cfg(feature = "audio")]
    let audio_loaded = state.audio_engine.is_loaded().await;
    #[cfg(not(feature = "audio"))]
    let audio_loaded = false;
    #[cfg(feature = "audio")]
    let tts_loaded = state.tts_engine.is_loaded().await;
    #[cfg(not(feature = "audio"))]
    let tts_loaded = false;
    let loaded_models =
        llm_count + usize::from(image_loaded) + usize::from(audio_loaded) + usize::from(tts_loaded);
    let uptime_s = SERVER_START
        .get()
        .map(|s| s.elapsed().as_secs())
        .unwrap_or(0);
    Json(serde_json::json!({
        "status": "healthy",
        "service": "loken",
        "version": env!("CARGO_PKG_VERSION"),
        "time": chrono::Utc::now().to_rfc3339(),
        "uptime_seconds": uptime_s,
        "loaded_models": loaded_models,
        // Per-engine breakdown so clients can tell WHICH modality
        // is warm without separately probing /api/show.
        "loaded": {
            "llms": llm_count,
            "image": image_loaded,
            "audio": audio_loaded,
            "tts":   tts_loaded,
        },
    }))
}

/// Process-wide start time, set once on first server startup so the
/// /health endpoint can report uptime. `OnceLock` keeps it lock-free
/// after the first set.
pub(crate) static SERVER_START: std::sync::OnceLock<std::time::Instant> =
    std::sync::OnceLock::new();

/// Observability: snapshot of in-flight + queued requests by priority.
/// Useful for diagnosing why a request is waiting.
pub(crate) async fn inflight_status(
    State(state): State<APIServer>,
) -> Json<crate::api::gate::GateSnapshot> {
    Json(state.request_gate.snapshot().await)
}

#[derive(Debug, Deserialize)]
pub(crate) struct DraftAttachRequest {
    /// The target model that should receive the draft attachment. Must be
    /// loaded before this call.
    target: String,
    /// Draft model id. Tokenizer must match the target's exactly.
    draft: String,
    /// CUDA device index for the draft. If omitted, uses the highest GPU
    /// index that isn't where the target's primary device lives, falling
    /// back to GPU 0.
    #[serde(default)]
    device_index: Option<usize>,
    /// Number of speculative tokens per cycle. Defaults to 4.
    #[serde(default)]
    k: Option<usize>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DraftAttachResponse {
    target: String,
    draft: String,
    k: usize,
    target_vocab: usize,
    draft_vocab: usize,
    tokenizer_fp_match: bool,
    status: &'static str,
}

/// Attach a speculative draft model to a target. Validates that
/// the tokenizers are byte-identical (same fingerprint, same vocab size)
/// before the attach succeeds; otherwise the call returns an error and
/// the target is left untouched.
pub(crate) async fn draft_attach(
    State(state): State<APIServer>,
    Json(request): Json<DraftAttachRequest>,
) -> Result<Json<DraftAttachResponse>, ApiError> {
    validate_model_id(&request.target)?;
    validate_model_id(&request.draft)?;
    let target_name = normalize_model_id(&request.target);
    let draft_name = normalize_model_id(&request.draft);
    let k = request.k.unwrap_or(4).clamp(1, 16);

    // Confirm target is loaded.
    let target_engine = state
        .get_engine(&target_name)
        .await
        .map_err(|_| ApiError::NotFound(format!("Target model '{}' not loaded", target_name)))?;
    let target_fp = target_engine
        .tokenizer_fingerprint()
        .await
        .ok_or_else(|| ApiError::Internal("Target model has no tokenizer".into()))?;
    let target_vocab = target_engine.vocab_size().await.unwrap_or(0);

    // Choose draft device. Default: GPU index 1 if available, else 0
    // - intentionally generic; operators on any setup can override via
    // `device_index`. Using `unwrap_or(1)` since the default is a
    // cheap constant; closure forms only earn their keep for
    // expressions with side-effects or allocation.
    let draft_device_idx = request.device_index.unwrap_or(1);

    // Build a fresh engine for the draft, loaded on the chosen device.
    let mut draft_config = state.config_for_model(&draft_name);
    draft_config.device_index = Some(draft_device_idx);
    // Force CUDA-only on the draft device (no multi-GPU split for the draft).
    draft_config.disable_arc_layers = true;
    let draft_engine = Arc::new(LlmEngine::with_config(draft_config));
    if let Err(e) = draft_engine.load_model().await {
        return Err(ApiError::Internal(format!("Draft load failed: {}", e)));
    }
    let draft_fp = draft_engine
        .tokenizer_fingerprint()
        .await
        .ok_or_else(|| ApiError::Internal("Draft model has no tokenizer".into()))?;
    let draft_vocab = draft_engine.vocab_size().await.unwrap_or(0);

    let fp_match = target_fp == draft_fp && target_vocab == draft_vocab;
    if !fp_match {
        // Drop the draft engine's resources before erroring out.
        let _ = draft_engine.unload().await;
        return Err(ApiError::Validation(format!(
            "Tokenizer fingerprint mismatch: target {} (vocab {}) vs draft {} (vocab {}). \
             Speculative decoding requires byte-identical tokenizers.",
            target_fp, target_vocab, draft_fp, draft_vocab
        )));
    }

    // Reset the draft KV cache so the next forward starts clean.
    let _ = draft_engine.draft_reset().await;

    // Attach to the target's LoadedModelEntry.
    {
        let mut engines = state.engines.write().await;
        let entry = engines
            .iter_mut()
            .find(|e| e.model_id == target_name)
            .ok_or_else(|| {
                ApiError::NotFound(format!("Target '{}' disappeared mid-attach", target_name))
            })?;
        entry.draft = Some(DraftAttachment {
            model_id: draft_name.clone(),
            engine: draft_engine,
            k,
        });
    }

    info!(
        "✅ Draft '{}' attached to '{}' (k={}, vocab={}, fp_match={})",
        draft_name, target_name, k, target_vocab, fp_match
    );

    Ok(Json(DraftAttachResponse {
        target: target_name,
        draft: draft_name,
        k,
        target_vocab,
        draft_vocab,
        tokenizer_fp_match: fp_match,
        status: "attached",
    }))
}

#[derive(Debug, Deserialize)]
pub(crate) struct DraftDetachRequest {
    target: String,
}

pub(crate) async fn draft_detach(
    State(state): State<APIServer>,
    Json(request): Json<DraftDetachRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_model_id(&request.target)?;
    let target_name = normalize_model_id(&request.target);
    let detached = {
        let mut engines = state.engines.write().await;
        let entry = engines
            .iter_mut()
            .find(|e| e.model_id == target_name)
            .ok_or_else(|| ApiError::NotFound(format!("Target '{}' not loaded", target_name)))?;
        let prev = entry.draft.take();
        prev.map(|a| a.model_id)
    };
    if let Some(draft_id) = detached {
        info!("Draft '{}' detached from '{}'", draft_id, target_name);
        Ok(Json(
            serde_json::json!({"target": target_name, "detached": draft_id}),
        ))
    } else {
        Ok(Json(
            serde_json::json!({"target": target_name, "detached": null}),
        ))
    }
}

pub(crate) async fn draft_status(State(state): State<APIServer>) -> Json<serde_json::Value> {
    let engines = state.engines.read().await;
    let mut pairs = Vec::new();
    for entry in engines.iter() {
        if let Some(d) = &entry.draft {
            pairs.push(serde_json::json!({
                "target": entry.model_id,
                "draft": d.model_id,
                "k": d.k,
            }));
        }
    }
    Json(serde_json::json!({"pairs": pairs}))
}

/// Per-layer inference perf snapshot. The global tracker is updated
/// in-band during forward passes (z-image-turbo, flux-schnell, and
/// LlmEngine's per-layer LayerTimer hooks). Empty if no model has
/// run yet; the GUI's Hardware tab uses this for its
/// per-layer panel.
///
/// Query params (all optional):
///   - `model=<name>` - case-sensitive exact match on model_name.
///   - `sort=tps` - tokens-per-second descending (slowest-first
///     analysis when reading from the bottom); any other value
///     keeps the natural layer_idx order, which is the typical UI
///     default.
pub(crate) async fn layer_performance_endpoint(
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let metrics = crate::inference::place::layer_perf::global_tracker().get_metrics();
    let mut filtered: Vec<_> = match params.get("model") {
        Some(name) => metrics
            .into_iter()
            .filter(|l| &l.model_name == name)
            .collect(),
        None => metrics,
    };
    if matches!(params.get("sort").map(String::as_str), Some("tps")) {
        filtered.sort_by(|a, b| {
            b.tokens_per_second
                .partial_cmp(&a.tokens_per_second)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    Json(serde_json::json!({ "layers": layers_to_json(&filtered) }))
}

/// Where a decode step's time goes, by stage.
///
/// `?enable=1` switches the measurement on and zeroes it, `?enable=0` switches it off.
/// It is off by default because each stage boundary synchronises with the device: that is
/// what makes the numbers real, and what makes them too expensive to leave running.
pub(crate) async fn stage_performance_endpoint(
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    use crate::inference::place::layer_perf::stages;
    match params.get("enable").map(String::as_str) {
        Some("1") => {
            stages::reset();
            stages::set_enabled(true);
        }
        Some("0") => stages::set_enabled(false),
        _ => {}
    }
    let (sums, calls) = stages::snapshot();
    // The share is against the stages that partition a layer-call; the two nested inside
    // the expert block are reported beside them, not counted twice.
    let total: u64 = sums.iter().take(stages::PARTITION).map(|(_, v)| *v).sum();
    Json(serde_json::json!({
        "enabled": stages::enabled(),
        "layer_calls": calls,
        "total_us": total,
        "host_fast_calls": stages::host_paths().0,
        "host_rejects": stages::host_paths().1.iter()
            .map(|(k, v)| serde_json::json!({ "cause": k, "calls": v })).collect::<Vec<_>>(),
        "topk_fast_calls": stages::topk_paths().0,
        "topk_slow_calls": stages::topk_paths().1,
        "stages": sums.iter().map(|(name, us)| serde_json::json!({
            "stage": name,
            "total_us": us,
            "share": if total > 0 { *us as f64 / total as f64 } else { 0.0 },
            "us_per_layer_call": if calls > 0 { *us as f64 / calls as f64 } else { 0.0 },
        })).collect::<Vec<_>>(),
    }))
}

/// Pure transform from in-process LayerPerformance records to the wire
/// JSON shape consumed by the GUI's LayerPerfRecord deserializer.
/// Extracted so a unit test can pin the schema without spinning up axum.
fn layers_to_json(
    metrics: &[crate::inference::place::layer_perf::LayerPerformance],
) -> Vec<serde_json::Value> {
    metrics
        .iter()
        .map(|l| {
            serde_json::json!({
                "layer_idx": l.layer_idx,
                "device_type": l.device_type,
                "model_name": l.model_name,
                "total_duration_ms": l.total_duration.as_secs_f64() * 1000.0,
                "token_count": l.token_count,
                "avg_ms_per_token": l.avg_ms_per_token,
                "tokens_per_second": l.tokens_per_second,
                "early_exit_count": l.early_exit_count,
            })
        })
        .collect()
}

/// List available compute devices with detailed availability status
pub(crate) async fn list_devices(
    State(_state): State<APIServer>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Per-request log at DEBUG, not INFO - the GUI's Hardware-tab
    // auto-refresh polls this every 2s. INFO would flood the server
    // log with ~7.5 listing lines / minute of zero-value noise.
    debug!("Listing compute devices with availability status");

    use crate::distributed::DeviceManager;

    // Detect devices
    let mut device_manager = DeviceManager::new();
    device_manager
        .detect_devices()
        .map_err(|e| ApiError::Internal(format!("Failed to detect devices: {}", e)))?;

    // The 15-line HardwareTopology::log_summary() used to fire here on
    // every request. Combined with the GUI's 2s auto-refresh that was
    // ~112 lines of repeated topology dump per minute. The topology is
    // already logged once at server startup, so dropping the per-request
    // call eliminates the spam without losing observability.

    // Query NVML once up-front for live per-CUDA-device telemetry.
    // Map is by NVML index; non-CUDA devices fall back to None. We
    // capture free VRAM + utilization + temperature + power in a
    // single pass so each device row gets all four without re-init'ing
    // NVML per field.
    #[derive(Clone, Copy, Default)]
    struct CudaLive {
        free: Option<u64>,
        util_gpu_pct: Option<f32>,
        util_mem_pct: Option<f32>,
        temp_c: Option<f32>,
        power_w: Option<f32>,
        power_limit_w: Option<f32>,
    }
    let cuda_live: std::collections::HashMap<usize, CudaLive> = {
        #[cfg(feature = "cuda")]
        {
            let mut m = std::collections::HashMap::new();
            if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                if let Ok(count) = nvml.device_count() {
                    for i in 0..count {
                        let Ok(dev) = nvml.device_by_index(i) else {
                            continue;
                        };
                        let free = dev.memory_info().ok().map(|mi| mi.free);
                        let util = dev.utilization_rates().ok();
                        let util_gpu_pct = util.as_ref().map(|u| u.gpu as f32);
                        let util_mem_pct = util.as_ref().map(|u| u.memory as f32);
                        // Temperature: GPU core sensor.
                        let temp_c = dev
                            .temperature(
                                nvml_wrapper::enum_wrappers::device::TemperatureSensor::Gpu,
                            )
                            .ok()
                            .map(|t| t as f32);
                        // Power usage in mW -> W; limit in mW -> W.
                        let power_w = dev.power_usage().ok().map(|mw| mw as f32 / 1000.0);
                        let power_limit_w =
                            dev.enforced_power_limit().ok().map(|mw| mw as f32 / 1000.0);
                        m.insert(
                            i as usize,
                            CudaLive {
                                free,
                                util_gpu_pct,
                                util_mem_pct,
                                temp_c,
                                power_w,
                                power_limit_w,
                            },
                        );
                    }
                }
            }
            m
        }
        #[cfg(not(feature = "cuda"))]
        std::collections::HashMap::new()
    };

    // Build detailed device info with availability
    let devices: Vec<serde_json::Value> = device_manager.devices().iter().map(|d| {
        let (status, reason, suggestion) = match &d.availability {
            crate::distributed::DeviceAvailability::Available => {
                ("available", None::<String>, None::<String>)
            }
            crate::distributed::DeviceAvailability::Unavailable { reason, suggestion } => {
                ("unavailable", Some(reason.to_string()), Some(suggestion.clone()))
            }
        };

        // Live NVML telemetry lookup for CUDA devices. Map key is the
        // NVML index which equals our device id for CUDA enumerations.
        let live = matches!(d.device_type, crate::distributed::DeviceType::Cuda)
            .then(|| cuda_live.get(&d.id).copied())
            .flatten()
            .unwrap_or_default();

        // NVML answers for a card; the host answers for itself. Reporting null here left the
        // CPU row with no free memory at all, which a reader has to take either as unmeasured
        // or as a full machine.
        let free_bytes = match d.device_type {
            crate::distributed::DeviceType::Cuda => live.free,
            _ => {
                let mut sys = sysinfo::System::new();
                sys.refresh_memory();
                Some(sys.available_memory())
            }
        };

        serde_json::json!({
            "id": d.id,
            "type": d.device_type.to_string(),
            "name": d.name,
            "memory_gb": d.memory_gb(),
            "memory_bytes": d.memory_bytes,
            "free_bytes": free_bytes,
            "utilization_gpu_percent": live.util_gpu_pct,
            "utilization_memory_percent": live.util_mem_pct,
            "temperature_c": live.temp_c,
            "power_watts": live.power_w,
            "power_limit_watts": live.power_limit_w,
            "priority": d.priority,
            "status": status,
            "reason": reason,
            "suggestion": suggestion,
            "usable_memory_gb": d.available_memory_for_model() as f64 / (1024.0 * 1024.0 * 1024.0)
        })
    }).collect();

    let available_count = devices
        .iter()
        .filter(|d| d["status"] == "available")
        .count();
    let total_memory_gb: f32 = device_manager
        .devices()
        .iter()
        .map(crate::distributed::device_manager::ComputeDevice::memory_gb)
        .sum();

    // Cumulative session energy for the GUI's Hardware-tab Energy card.
    // `None` (-> JSON null) when energy reporting is disabled, so the GUI
    // shows a "measurement off" hint instead of stale zeros. This rides
    // the existing 2s Hardware-tab poll - no extra endpoint/request.
    let energy =
        crate::energy_report::is_enabled().then(|| crate::energy_report::tracker().snapshot());

    Ok(Json(serde_json::json!({
        "devices": devices,
        "summary": {
            "total_devices": devices.len(),
            "available_devices": available_count,
            "unavailable_devices": devices.len() - available_count,
            "total_memory_gb": total_memory_gb,
            // What this binary was actually built with, so the Hardware tab can tell "no
            // device" from "no support compiled in". It used to report `sycl`, which is not
            // a feature of this crate and so was always false, and to omit `opencl`, which
            // is the one that gates the Intel Arc backend.
            "compiled_features": {
                "cuda": cfg!(feature = "cuda"),
                "opencl": cfg!(feature = "opencl"),
                "image": cfg!(feature = "image"),
                "video": cfg!(feature = "video"),
                "audio": cfg!(feature = "audio"),
                "midi": cfg!(feature = "midi")
            }
        },
        "energy": energy
    })))
}

/// What this node believes about its peers (GET /api/cluster/peers).
///
/// `/api/cluster/state` answers "what am I"; this answers "who do I see". Only the pair makes
/// an asymmetric partition visible: a node every observer can reach may still be unable to
/// reach its neighbour.
///
/// As cheap and side-effect free as the state endpoint, and for the same reason - the gossip
/// round trip is what prices a hand-over, so anything slow here biases routing.
pub(crate) async fn cluster_peers(
    State(state): State<APIServer>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Some(cluster) = state.cluster_handle() else {
        return Ok(Json(serde_json::json!({ "clustered": false, "peers": [] })));
    };
    let now = crate::distributed::cluster_runtime::now_ms(state.cluster_started());
    Ok(Json(serde_json::json!({
        "clustered": true,
        "peers": cluster.peer_view(now),
    })))
}

/// What this node has to execute with, and what is running on it.
///
/// Read from the state the server already holds. It used to build a `DistributedEngine` and
/// run device detection per request, which probes every card on an HTTP call - and two of the
/// four numbers it returned came from that throwaway engine, so `active_sessions` and
/// `remote_servers` were always zero however busy the node was.
pub(crate) async fn distributed_stats(
    State(state): State<APIServer>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut devices = 0usize;
    let mut total_bytes = 0u64;
    let mut free_bytes = 0u64;
    if let Some(gm) = state.gpu_manager.as_ref() {
        for d in gm.get_devices() {
            devices += 1;
            if let Ok(info) = d.memory_info() {
                total_bytes += info.total;
                free_bytes += info.free;
            }
        }
    }

    let engines = state.engines.read().await;
    let mut resident = Vec::with_capacity(engines.len());
    let mut sessions = 0usize;
    for entry in engines.iter() {
        let live = entry.engine.session_count().await;
        sessions += live;
        resident.push(serde_json::json!({
            "model": entry.model_id,
            "sessions": live,
        }));
    }
    drop(engines);

    // Peers, from the membership the node already maintains. A cluster of one reports zero
    // rather than counting itself: the question is who else could take work.
    let (peers_alive, peers_known) = match state.cluster_handle() {
        Some(cluster) => {
            let now = crate::distributed::cluster_runtime::now_ms(state.cluster_started());
            let view = cluster.peer_view(now);
            let known = view.iter().filter(|p| !p.is_self).count();
            let alive = view.iter().filter(|p| !p.is_self && p.alive).count();
            (alive, known)
        }
        None => (0, 0),
    };

    Ok(Json(serde_json::json!({
        "total_devices": devices,
        "total_memory_gb": total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        "free_memory_gb": free_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        "active_sessions": sessions,
        "loaded_models": resident,
        "peers_alive": peers_alive,
        "peers_known": peers_known,
        "clustered": state.cluster_handle().is_some(),
    })))
}

/// The largest model on this machine that would fit in what is free right now.
///
/// Answered from the catalogue rather than from a ladder of sizes. A table mapping "80 GB or
/// more" to the string "70B" advises about models the machine may not hold and cannot account
/// for quantisation, which is most of what decides whether a checkpoint fits.
pub(crate) async fn recommended_model(
    State(state): State<APIServer>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut free_bytes = 0u64;
    if let Some(gm) = state.gpu_manager.as_ref() {
        for d in gm.get_devices() {
            if let Ok(info) = d.memory_info() {
                free_bytes += info.free;
            }
        }
    }

    // Every load pays a fixed overhead that no formula predicts from the weights - CUDA
    // contexts, the cuBLAS workspace, preloaded module images. Advising up to the last free
    // byte would recommend a model that cannot be loaded.
    let budget = free_bytes.saturating_sub(crate::inference::place::runtime_demand::FIXED_LOAD_OVERHEAD_BYTES);

    let catalogue = state
        .model_manager
        .list_models()
        .await
        .map_err(|e| ApiError::Internal(format!("listing models: {e}")))?;
    let sized: Vec<(String, u64)> = catalogue.iter().map(|m| (m.id.clone(), m.size)).collect();
    let (best, fits) = largest_that_fits(&sized, budget);

    Ok(Json(serde_json::json!({
        "free_memory_gb": free_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        "budget_gb": budget as f64 / (1024.0 * 1024.0 * 1024.0),
        "recommended": best.map(|(id, size)| serde_json::json!({
            "model": id,
            "size_gb": *size as f64 / (1024.0 * 1024.0 * 1024.0),
        })),
        "fits_now": fits,
        "catalogue": catalogue.len(),
        // Named rather than left to be inferred from the gap between the two figures.
        "reserved_per_load_bytes": crate::inference::place::runtime_demand::FIXED_LOAD_OVERHEAD_BYTES,
    })))
}

/// The biggest entry that fits a budget, and how many do.
///
/// Ties go to the name that sorts first, so two builds of one machine answer the same thing;
/// a catalogue walk in directory order would not.
fn largest_that_fits(catalogue: &[(String, u64)], budget: u64) -> (Option<&(String, u64)>, usize) {
    let fits: Vec<&(String, u64)> = catalogue.iter().filter(|(_, size)| *size <= budget).collect();
    let best = fits
        .iter()
        .copied()
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)));
    (best, fits.len())
}

// ============================================================================
// Multi-Device Endpoints
// ============================================================================

// Placement is decided at load time from live VRAM, so there is nothing for a caller to
// plan or configure; what it can ask for is the truth about what is loaded. That is
// `/api/multi-device/status` below, which reports the resident models with their real
// device and layer counts, and `/api/ps`, which reports the per-device segment topology.

/// Get multi-device status and configuration
pub(crate) async fn get_multi_device_status(
    State(state): State<APIServer>,
) -> Result<Json<serde_json::Value>, ApiError> {
    info!("Getting multi-device status");

    // Get loaded engines
    let engines = state.engines.read().await;
    let mut loaded_models = Vec::new();

    for entry in engines.iter() {
        let device_name = entry.engine.get_device_name().await;
        let model_size = entry.engine.get_model_size().await;
        let metadata = entry.engine.get_metadata().await;

        loaded_models.push(serde_json::json!({
            "model": entry.model_id,
            "device": device_name,
            "size_gb": model_size as f64 / (1024.0 * 1024.0 * 1024.0),
            "num_layers": metadata.as_ref().map(|m| m.num_layers as u32),
            "keep_alive_minutes": entry.keep_alive_minutes
        }));
    }

    // GPU memory snapshot: per-device + summed across all GPUs.
    // Previously this only reported the GPU with the most memory,
    // which underreports total VRAM by 50%+ on multi-GPU setups.
    // Also surfaces live utilization / temperature / power so
    // monitoring SDKs don't need a separate /api/gpu/live endpoint
    // (all four call NVML, so amortizing into one request is cheaper
    // than four separate ones).
    let mut per_gpu: Vec<serde_json::Value> = Vec::new();
    let mut total_bytes: u64 = 0;
    let mut total_free_bytes: u64 = 0;
    if let Some(gm) = state.gpu_manager.as_ref() {
        for (idx, d) in gm.get_devices().iter().enumerate() {
            let info = d.memory_info().ok();
            let total = info.as_ref().map(|i| i.total).unwrap_or(0);
            let free = info.as_ref().map(|i| i.free).unwrap_or(0);
            total_bytes += total;
            total_free_bytes += free;

            // Live metrics - each call queries NVML and returns Err
            // on a non-NVIDIA / no-NVML setup. Wrap each in `ok()` so
            // a partial failure (e.g. fan info unavailable on a
            // headless workstation) doesn't blank the whole entry.
            let util = d.utilization().ok();
            let temp = d.temperature().ok();
            let power = d.power_info().ok();

            per_gpu.push(serde_json::json!({
                "id": idx,
                "name": d.name(),
                "total_bytes": total,
                "free_bytes": free,
                "utilization_gpu_percent": util.as_ref().map(|u| u.gpu * 100.0),
                "utilization_memory_percent": util.as_ref().map(|u| u.memory * 100.0),
                "temperature_c": temp.as_ref().map(|t| t.gpu),
                "power_watts": power.as_ref().map(|p| p.power),
                "power_limit_watts": power.as_ref().map(|p| p.limit),
            }));
        }
    }

    Ok(Json(serde_json::json!({
        "loaded_models": loaded_models,
        "gpu_memory_gb": total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        "gpu_free_gb": total_free_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        "gpus": per_gpu,
        "total_loaded_models": loaded_models.len(),
        "multi_device_enabled": true
    })))
}

// ============================================================================
// Model/Layer Swapping Handlers
// ============================================================================

/// Swap model handler (POST /api/swap)
pub(crate) async fn swap_model_handler(
    State(state): State<APIServer>,
    Json(request): Json<SwapModelRequest>,
) -> Result<Json<SwapModelResponse>, ApiError> {
    validate_model_id(&request.current_model)?;
    validate_model_id(&request.new_model)?;
    let current_model = normalize_model_id(&request.current_model);
    let new_model = normalize_model_id(&request.new_model);

    info!("Model swap request: {} -> {}", current_model, new_model);

    let keep_alive = request.keep_alive.as_deref();

    match state
        .swap_model(&current_model, &new_model, keep_alive)
        .await
    {
        Ok(()) => Ok(Json(SwapModelResponse {
            status: "success".to_string(),
            message: format!("Successfully swapped {} with {}", current_model, new_model),
            previous_model: current_model,
            current_model: new_model,
        })),
        Err(e) => Err(e),
    }
}

/// Swap layers handler (POST /api/layers/swap)
///
/// Not yet implemented - returns 501. Real implementation would
/// require extending the LayerExecutor trait for weight replacement,
/// MultiDeviceWrapper per-layer swap methods, and LoRA/adapter
/// loading + merging. Returning 200 OK with `status: "not_implemented"`
/// in the body misleads clients that branch on HTTP status - most
/// SDKs treat 2xx as success and ship code that proceeds as if
/// the swap landed.
pub(crate) async fn swap_layers_handler(
    State(_state): State<APIServer>,
    Json(request): Json<SwapLayerRequest>,
) -> Result<Response, ApiError> {
    info!(
        "Layer swap request (not implemented): model={}, layers={:?}, source={}",
        request.model, request.layer_indices, request.source_model
    );
    // Validate inputs even though we'll 501 - keeps the security
    // screen consistent across surfaces (path traversal, empty model).
    validate_model_id(&request.model)?;
    validate_model_id(&request.source_model)?;

    let body = Json(serde_json::json!({
        "error": {
            "message": "layer swapping is not implemented; would enable hot-swap of LoRA adapters / fine-tuned layers without model reload",
            "type": "not_implemented",
            "code": "feature_unavailable",
        }
    }));
    Ok((StatusCode::NOT_IMPLEMENTED, body).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layers_to_json_emits_expected_wire_keys() {
        use std::time::Duration;
        let metrics = vec![crate::inference::place::layer_perf::LayerPerformance {
            layer_idx: 7,
            device_type: "CUDA".into(),
            model_name: "z-image-turbo".into(),
            total_duration: Duration::from_millis(250),
            token_count: 5,
            avg_ms_per_token: 50.0,
            tokens_per_second: 20.0,
            early_exit_count: 0,
        }];
        let json = layers_to_json(&metrics);
        assert_eq!(json.len(), 1);
        let obj = json[0].as_object().expect("layer entry is an object");
        // Pinning the keys the GUI's LayerPerfRecord depends on.
        for key in [
            "layer_idx",
            "device_type",
            "model_name",
            "total_duration_ms",
            "token_count",
            "avg_ms_per_token",
            "tokens_per_second",
            "early_exit_count",
        ] {
            assert!(obj.contains_key(key), "missing wire key: {key}");
        }
        assert_eq!(obj["layer_idx"], 7);
        assert_eq!(obj["device_type"], "CUDA");
        assert_eq!(obj["model_name"], "z-image-turbo");
        // 250 ms duration -> 250.0 (within float tolerance).
        let dur = obj["total_duration_ms"].as_f64().unwrap();
        assert!((dur - 250.0).abs() < 1e-6, "duration was {dur}");
    }

    #[test]
    fn layers_to_json_empty_input_yields_empty_array() {
        let json = layers_to_json(&[]);
        assert!(json.is_empty());
    }

    #[test]
    fn layers_to_json_model_filter_is_caller_responsibility() {
        // The endpoint applies the ?model= filter before calling
        // layers_to_json; this helper itself is a pure transform so
        // the test just demonstrates that filtering input slices
        // works as expected (covers the same logic the endpoint uses).
        use std::time::Duration;
        let metrics = vec![
            crate::inference::place::layer_perf::LayerPerformance {
                layer_idx: 0,
                device_type: "CUDA".into(),
                model_name: "qwen3:latest".into(),
                total_duration: Duration::from_millis(100),
                token_count: 5,
                avg_ms_per_token: 20.0,
                tokens_per_second: 50.0,
                early_exit_count: 0,
            },
            crate::inference::place::layer_perf::LayerPerformance {
                layer_idx: 0,
                device_type: "CUDA".into(),
                model_name: "gemma4:latest".into(),
                total_duration: Duration::from_millis(200),
                token_count: 4,
                avg_ms_per_token: 50.0,
                tokens_per_second: 20.0,
                early_exit_count: 0,
            },
        ];
        let target = "qwen3:latest";
        let filtered: Vec<_> = metrics
            .into_iter()
            .filter(|l| l.model_name == target)
            .collect();
        let json = layers_to_json(&filtered);
        assert_eq!(json.len(), 1);
        assert_eq!(json[0]["model_name"], "qwen3:latest");
    }
}

/// What this node publishes to its peers.
///
/// Deliberately cheap and side-effect free: a peer polls it on every gossip round, and the
/// round-trip time of THIS request is what the router uses to price a hand-over. Anything slow
/// here would inflate the measured latency of the node it describes and bias routing away from
/// a peer that is in fact fine.
pub(crate) async fn cluster_state(
    State(state): State<APIServer>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let report = state.cluster_report().await;
    Ok(Json(serde_json::to_value(report).map_err(|e| {
        ApiError::Internal(format!("cluster state: {e}"))
    })?))
}

/// How much of a prompt this node already holds, answered for a peer.
///
/// The peer cannot compute this for us: the tokeniser belongs to the model, and only the node
/// that loaded it can turn text into the tokens its own cache is keyed on. Sending the question
/// instead of a hashing convention is what makes prefix-aware routing work whatever the
/// architecture - and it stays exact, because the figure comes from the same function the
/// local prefill path uses.
///
/// A model that is not resident answers zero rather than an error: "I hold none of it" is a
/// perfectly good answer to route on, and an error would make the asking node retry or wait.
pub(crate) async fn cluster_prefix(
    State(state): State<APIServer>,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    let cached = state.cached_prompt_tokens(model, prompt).await.unwrap_or(0);
    Ok(Json(serde_json::json!({ "cached_tokens": cached })))
}

#[cfg(test)]
mod recommendation_tests {
    use super::largest_that_fits;

    fn catalogue() -> Vec<(String, u64)> {
        vec![
            ("small:1b".to_string(), 1 << 30),
            ("mid:8b".to_string(), 5 << 30),
            ("big:70b".to_string(), 40 << 30),
        ]
    }

    /// The largest that fits, not the largest there is.
    #[test]
    fn the_recommendation_is_bounded_by_the_budget() {
        let c = catalogue();
        let (best, fits) = largest_that_fits(&c, 6 << 30);
        assert_eq!(best.map(|(id, _)| id.as_str()), Some("mid:8b"));
        assert_eq!(fits, 2);
    }

    /// A machine with nothing free recommends nothing, rather than the smallest thing on disk.
    #[test]
    fn nothing_fits_is_an_answer() {
        let c = catalogue();
        let (best, fits) = largest_that_fits(&c, 0);
        assert!(best.is_none());
        assert_eq!(fits, 0);
    }

    /// Two entries of one size resolve the same way every time.
    #[test]
    fn a_tie_is_broken_by_name() {
        let c = vec![("b:1".to_string(), 1 << 30), ("a:1".to_string(), 1 << 30)];
        let (best, _) = largest_that_fits(&c, 2 << 30);
        assert_eq!(best.map(|(id, _)| id.as_str()), Some("a:1"));
    }
}
