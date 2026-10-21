//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Parse the base model out of an Ollama modelfile string. Looks for
/// the first `FROM <name>` line (case-insensitive on the directive,
/// preserves the name's casing). Returns `None` if no such line.
///
/// Lifted out of the /api/create handler so it has a dedicated
/// unit test independent of the surrounding handler plumbing - the
/// handler still re-validates the result through `validate_model_id`
/// before letting it reach the model_manager.
///
/// Implementation note: uses `eq_ignore_ascii_case` on the first 5
/// bytes instead of `.to_uppercase().starts_with("FROM ")`. The
/// uppercase version allocates a fresh String for every line scanned;
/// the byte-slice form is allocation-free and gives identical
/// behaviour for the documented ASCII-only `FROM ` directive.
pub(super) fn parse_modelfile_from(modelfile: &str) -> Option<&str> {
    modelfile
        .lines()
        .find(|line| {
            let trimmed = line.trim();
            trimmed
                .get(..5)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("FROM "))
        })
        // Slice safely after the trimmed directive - the find() guard
        // above already proved the first 5 bytes are ASCII "FROM "
        // (case-insensitive), so [5..] is a guaranteed char boundary.
        .map(|line| line.trim()[5..].trim())
}

// ============================================================================
// Ollama-compatible Handlers
// ============================================================================

/// List models (GET /api/tags) - Ollama format
pub(crate) async fn ollama_list_models(
    State(state): State<APIServer>,
    headers: axum::http::HeaderMap,
) -> Result<Json<OllamaListModelsResponse>, ApiError> {
    info!("📋 GET /api/tags - Listing models (Ollama format)");

    let manager = state.model_manager.clone();

    match manager.list_models().await {
        Ok(mut models) => {
            #[cfg(feature = "media")]
            inject_local_media(&state, &mut models);
            // Only models this node can actually load: present weights, and a format the text
            // loader reads. A manifest that names a deleted blob is not one a client can ask for.
            models.retain(|m| state.serves_weights(m));
            let count = models.len();
            info!("   Found {} model(s)", count);
            // Stable alphabetical order (mirror 6615f9a on /v1/models).
            models.sort_by(|a, b| a.id.cmp(&b.id));
            let seen: std::collections::HashSet<std::path::PathBuf> = models
                .iter()
                .filter_map(|m| state.manifest_layer_path(&m.id, "model"))
                .collect();
            retain_facts(&seen);

            let ollama_models: Vec<OllamaModel> = models
                .into_iter()
                .map(|m| {
                    let size_mb = m.size / (1024 * 1024);
                    info!("   • {} ({} MB, source: {})", m.id, size_mb, m.source);
                    let source = match m.source.as_str() {
                        "huggingface" => ModelSource::HuggingFace,
                        _ => ModelSource::Ollama,
                    };
                    // What the checkpoint says about itself: the media pipeline its files
                    // declare, else the text model its header or configuration declares.
                    let hf = state.huggingface_models_dir.as_str();
                    let media = match source {
                        ModelSource::Ollama => None,
                        _ => media_facts(hf, &m.id),
                    };
                    // The format follows the file, not the store it sits in: a GGUF in a
                    // repository's snapshot is a GGUF.
                    let gguf = match source {
                        ModelSource::Ollama => None,
                        ModelSource::HuggingFace => hf_gguf(hf, &m.id),
                    };
                    let format_str = match (&gguf, source) {
                        (Some(_), _) => "gguf",
                        (None, ModelSource::HuggingFace) => "safetensors",
                        (None, ModelSource::Ollama) => "gguf",
                    };
                    let weights = match source {
                        ModelSource::Ollama => state.manifest_layer_path(&m.id, "model"),
                        _ if media.is_some() => None,
                        _ => crate::inference::load::huggingface_manager::HuggingFaceManager::new(
                            std::path::PathBuf::from(hf),
                        )
                        .get_model_path(&m.id),
                    };
                    let (arch, quant, declared, pools) = match (&gguf, &weights) {
                        (Some(g), _) => header_facts(g),
                        (None, Some(w)) if w.is_dir() => checkpoint_facts(w),
                        (None, Some(w)) => header_facts(w),
                        (None, None) => (None, None, None, false),
                    };
                    // An Ollama-store model is a text GGUF whether or not its header reads.
                    let text = match source {
                        ModelSource::Ollama => Some(pools),
                        _ => arch.as_ref().map(|_| pools),
                    };
                    let mut capabilities = capabilities_of(media.as_ref(), text);
                    if capabilities.iter().any(|c| c == "completion")
                        && state.model_has_audio(&m.id)
                    {
                        capabilities.push("audio".to_string());
                    }
                    let family = reported_family(media.as_ref(), arch, || None);
                    let details = OllamaModelDetails {
                        format: format_str.to_string(),
                        families: Some(vec![family.clone()]),
                        family,
                        parameter_size: declared
                            .map(format_parameter_count)
                            .unwrap_or_else(|| estimate_parameter_size(m.size, format_str)),
                        quantization_level: quant,
                    };
                    {
                        let mut om = OllamaModel::new(m.id.clone(), m.size, m.downloaded_at)
                            .with_source(source)
                            .with_digest(m.digest)
                            .with_details(details);
                        om.capabilities = capabilities;
                        om.defaults = media.as_ref().and_then(media_model_defaults);
                        om
                    }
                })
                .collect();

            // What the peers hold and this node does not: a request for one of those is
            // forwarded, so the list names them under the node that holds them.
            let peers =
                crate::api::handlers::cluster_catalogue::peer_models(&state, &headers).await;
            let all = crate::api::handlers::cluster_catalogue::merge(ollama_models, peers);
            info!("   ✅ List complete");
            Ok(Json(OllamaListModelsResponse::new(all)))
        }
        Err(e) => {
            error!("   ❌ Failed to list models: {}", e);
            Err(ApiError::Internal(format!("Failed to list models: {}", e)))
        }
    }
}

/// Add the media checkpoints of the local stores to a listing: the registered pipelines and
/// the checkpoints dropped into weight directories, which the Hugging Face cache listing does
/// not name. When a registered Boogu entry is present, a cache repository whose files
/// declare that family is dropped: its loader reads the registered directory, so the
/// repository is not independently loadable.
#[cfg(feature = "media")]
pub(crate) fn inject_local_media(
    state: &APIServer,
    models: &mut Vec<crate::inference::load::model_manager::ModelMetadata>,
) {
    let hf = state.huggingface_models_dir.as_str();
    let entries = local_media_entries(hf);
    let boogu = Some(crate::inference::media_facts::ImageFamily::Boogu);
    let registered = |e: &MediaEntry| {
        crate::inference::media_facts::read(&e.path).and_then(|f| f.image_family()) == boogu
    };
    if entries.iter().any(registered) {
        models.retain(|m| {
            m.source != "huggingface"
                || media_facts(hf, &m.id).and_then(|f| f.image_family()) != boogu
        });
    }
    for e in entries {
        let size = std::fs::metadata(&e.path).map(|m| m.len()).unwrap_or(0);
        models.push(crate::inference::load::model_manager::ModelMetadata {
            id: e.id.clone(),
            name: e.id,
            size,
            downloaded_at: chrono::Utc::now().to_rfc3339(),
            files: vec![e.path.to_string_lossy().into_owned()],
            source: "huggingface".to_string(),
            digest: "sha256:0000000000000000000000000000000000000000".to_string(),
        });
    }
}

/// What a media checkpoint can do, from what its files declare. Only capabilities with a
/// working wired path are advertised.
pub(crate) fn media_capabilities(facts: &crate::inference::media_facts::MediaFacts) -> Vec<String> {
    use crate::inference::media_facts::{
        ImageFamily, MediaKind, SoundKind, SpeechModel, TranscriptionModel,
    };
    let caps: &[&str] = match facts.kind {
        MediaKind::Image(ImageFamily::Flux) if facts.instruction_edit => {
            &["txt2img", "img2img", "edit"]
        }
        MediaKind::Image(ImageFamily::Flux) => &["txt2img", "img2img"],
        MediaKind::Image(ImageFamily::ZImage) => &["txt2img", "img2img"],
        // An instruction-editing checkpoint edits; one trained for text-to-image only does
        // not, and advertising an edit it cannot do would put it on a tab it silently fails.
        MediaKind::Image(ImageFamily::QwenImage) if facts.instruction_edit => &["txt2img", "edit"],
        MediaKind::Image(ImageFamily::QwenImage) => &["txt2img"],
        // No instruction editor: the Edit tab drives it through img2img with a strength.
        MediaKind::Image(ImageFamily::Sdxl) => &["txt2img", "img2img"],
        // No img2img or instruction-edit path is wired for these.
        MediaKind::Image(ImageFamily::Flux2 | ImageFamily::Boogu) => &["txt2img"],
        MediaKind::Video => &["video"],
        MediaKind::Transcription(TranscriptionModel::Whisper) => &["asr"],
        MediaKind::Speech(
            SpeechModel::Parler | SpeechModel::Kyutai | SpeechModel::PocketTts | SpeechModel::Piper,
        ) => &["tts"],
        // A model this build runs no engine for can do nothing here.
        MediaKind::Transcription(TranscriptionModel::Converted)
        | MediaKind::Speech(SpeechModel::Unserved) => &[],
        MediaKind::Sound(SoundKind::ConditionedDiffusion) => &["sfx", "loops", "audio-variations"],
        MediaKind::Sound(SoundKind::LatentDiffusion) => &["sfx"],
        MediaKind::Music => &["music"],
        MediaKind::Midi => &["midi"],
        MediaKind::Forecast => &["forecast"],
        MediaKind::Component => &[],
    };
    caps.iter().map(|s| s.to_string()).collect()
}

/// What a model can do: what its media files declare, else, for a text model whose header or
/// configuration declares an architecture (`text` is `Some(pools)`), embedding when it pools
/// and completion otherwise. A model whose files declare neither can do nothing here.
pub(crate) fn capabilities_of(
    media: Option<&crate::inference::media_facts::MediaFacts>,
    text: Option<bool>,
) -> Vec<String> {
    match (media, text) {
        (Some(facts), _) => media_capabilities(facts),
        (None, Some(pools)) => vec![if pools { "embedding" } else { "completion" }.to_string()],
        (None, None) => Vec::new(),
    }
}

/// The name a GGUF `general.file_type` stands for.
///
/// The enum is part of the format, so the mapping is fixed by the file and not by this reader.
fn gguf_file_type_name(id: u64) -> Option<&'static str> {
    Some(match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4",
        _ => return None,
    })
}

/// Architecture, quantisation label, parameter count, and whether the file declares a
/// pooling type.
type Facts = (Option<String>, Option<String>, Option<u64>, bool);
type Stamp = (u64, Option<std::time::SystemTime>);
type FactsCache = std::collections::HashMap<std::path::PathBuf, (Stamp, Facts)>;
static CACHE: std::sync::OnceLock<std::sync::Mutex<FactsCache>> = std::sync::OnceLock::new();

fn facts_cache() -> &'static std::sync::Mutex<FactsCache> {
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Forget the facts of every file a listing no longer sees, so a removed model leaves no
/// entry behind.
pub(crate) fn retain_facts(seen: &std::collections::HashSet<std::path::PathBuf>) {
    facts_cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .retain(|path, _| seen.contains(path));
}

/// The architecture, quantisation label, parameter count and pooling a checkpoint declares,
/// read once per file: a header carries the whole vocabulary, and a catalogue of many models
/// read on every listing is what made the list take seconds. A file that changed size or date
/// is read again.
pub(crate) fn header_facts(path: &std::path::Path) -> Facts {
    let stamp: Stamp = match std::fs::metadata(path) {
        Ok(m) => (m.len(), m.modified().ok()),
        Err(_) => return (None, None, None, false),
    };
    let cache = facts_cache();
    if let Some((seen, facts)) = cache.lock().unwrap_or_else(|e| e.into_inner()).get(path) {
        if *seen == stamp {
            return facts.clone();
        }
    }
    let read = crate::tensor::quantized::gguf_file::read_facts(path).unwrap_or_else(|e| {
        warn!("no facts from {}: {e}", path.display());
        Default::default()
    });
    // A part of a split set lists only the tensors it carries: every part is counted.
    let parameter_count = match read.split_parts {
        Some(n) if n > 1 => crate::tensor::quantized::gguf_source::set_parameter_count(path)
            .ok()
            .filter(|n| *n > 0)
            .or(read.parameter_count),
        _ => read.parameter_count,
    };
    let facts = (
        read.architecture,
        read.file_type
            .and_then(gguf_file_type_name)
            .map(str::to_string),
        parameter_count,
        read.pools,
    );
    cache
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(path.to_path_buf(), (stamp, facts.clone()));
    facts
}

/// What a checkpoint says about itself: its metadata as JSON, its architecture, the label its
/// tensors are stored under, and how many parameters it holds.
///
/// Reads the header, not the weights.
pub(crate) fn gguf_facts(
    path: &std::path::Path,
) -> (
    Option<serde_json::Map<String, serde_json::Value>>,
    Option<String>,
    Option<String>,
    Option<u64>,
) {
    use crate::tensor::quantized::gguf_file::{self, Value};
    let content = match gguf_file::open_header(path) {
        Ok(c) => c,
        Err(e) => {
            warn!("no metadata from {}: {e}", path.display());
            return (None, None, None, None);
        }
    };
    fn json(v: &Value) -> serde_json::Value {
        use serde_json::Value as J;
        match v {
            Value::U8(x) => J::from(*x),
            Value::I8(x) => J::from(*x),
            Value::U16(x) => J::from(*x),
            Value::I16(x) => J::from(*x),
            Value::U32(x) => J::from(*x),
            Value::I32(x) => J::from(*x),
            Value::U64(x) => J::from(*x),
            Value::I64(x) => J::from(*x),
            // The shortest decimal that reads back as the same f32. Widening to f64 instead
            // prints 1e-05 as 9.999999747378752e-06.
            Value::F32(x) => x
                .to_string()
                .parse::<serde_json::Number>()
                .map_or(J::from(*x), J::Number),
            Value::F64(x) => J::from(*x),
            Value::Bool(x) => J::from(*x),
            Value::String(x) => J::from(x.clone()),
            Value::Array(xs) => J::Array(xs.iter().map(json).collect()),
        }
    }
    let arch = content
        .metadata
        .get("general.architecture")
        .and_then(|v| match v {
            Value::String(s) => Some(s.clone()),
            _ => None,
        });
    // A file states its own mix under `general.file_type`; the mix is what distinguishes
    // Q4_K_M from Q4_K_S, and no single tensor's dtype carries it.
    let quant = content
        .metadata
        .get("general.file_type")
        .and_then(|v| match v {
            Value::U32(n) => Some(u64::from(*n)),
            Value::I32(n) => u64::try_from(*n).ok(),
            Value::U64(n) => Some(*n),
            _ => None,
        })
        .and_then(gguf_file_type_name)
        .map(str::to_string)
        .or_else(|| {
            // Undeclared: name the width the bulk of the weights are stored at. The token
            // embedding is routinely stored at another, so it does not vote.
            let mut counts: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::new();
            for (name, info) in &content.tensor_infos {
                if !name.contains("token_embd") {
                    *counts.entry(info.ggml_dtype.gguf_name()).or_default() += 1;
                }
            }
            counts
                .into_iter()
                .max_by_key(|(_, n)| *n)
                .map(|(name, _)| name.to_uppercase())
        });
    // Most checkpoints do not declare a count; the shapes are the count, across every part of
    // a split set.
    let split = content
        .metadata
        .get("split.count")
        .and_then(|v| v.to_u32().ok())
        .is_some_and(|n| n > 1);
    let summed = match split {
        true => crate::tensor::quantized::gguf_source::set_parameter_count(path).ok(),
        false => None,
    }
    .unwrap_or_else(|| {
        content
            .tensor_infos
            .values()
            .map(|t| t.elem_count() as u64)
            .sum()
    });
    let params = Some(summed).filter(|n| *n > 0);
    let params = content
        .metadata
        .get("general.parameter_count")
        .and_then(|v| match v {
            Value::U64(n) => Some(*n),
            Value::I64(n) => u64::try_from(*n).ok(),
            Value::U32(n) => Some(u64::from(*n)),
            _ => None,
        })
        .or(params);
    let mut info: serde_json::Map<String, serde_json::Value> = content
        .metadata
        .iter()
        .map(|(k, v)| (k.clone(), json(v)))
        .collect();
    if let Some(n) = params {
        info.entry("general.parameter_count".to_string())
            .or_insert_with(|| serde_json::Value::from(n));
    }
    (Some(info), arch, quant, params)
}

/// The generation defaults a media checkpoint was validated at, from what its files declare.
/// `None` for a kind without presets.
pub(crate) fn media_model_defaults(
    facts: &crate::inference::media_facts::MediaFacts,
) -> Option<serde_json::Value> {
    use crate::inference::media_facts::{MediaKind, SoundKind};
    match facts.kind {
        // A distilled checkpoint runs fewer steps without guidance.
        MediaKind::Music if facts.distilled => Some(serde_json::json!({ "steps": 27, "cfg": 1.0 })),
        MediaKind::Music => Some(serde_json::json!({ "steps": 50, "cfg": 4.5 })),
        MediaKind::Sound(SoundKind::ConditionedDiffusion) => {
            Some(serde_json::json!({ "steps": 25, "cfg": 7.0, "max_seconds": 47 }))
        }
        MediaKind::Sound(SoundKind::LatentDiffusion) => {
            Some(serde_json::json!({ "steps": 60, "cfg": 3.0, "max_seconds": 30 }))
        }
        // Every image family publishes its recommended knobs from the SAME table the server
        // itself samples with, so a client that switches models gets values the model was
        // validated at.
        #[cfg(feature = "image")]
        MediaKind::Image(family) => {
            let d = crate::api::handlers::media::image_defaults_for(family, facts);
            Some(serde_json::json!({
                "steps": d.steps,
                "cfg": d.guidance,
                "size": d.size,
            }))
        }
        _ => None,
    }
}

/// The architecture a text model's files declare: `general.architecture` of a GGUF file,
/// `model_type` of a checkpoint directory's `config.json`.
pub(crate) fn file_architecture(path: &std::path::Path) -> Option<String> {
    if path.is_dir() {
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path.join("config.json")).ok()?).ok()?;
        return config
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(str::to_string);
    }
    header_facts(path).0
}

/// What a checkpoint directory declares: the `model_type` of its `config.json`, and whether
/// it pools, which a sentence-transformers `modules.json` says.
fn checkpoint_facts(dir: &std::path::Path) -> Facts {
    (
        file_architecture(dir),
        None,
        None,
        dir.join("modules.json").exists(),
    )
}

/// The GGUF a Hugging Face entry is, when it is one: the file a load resolves `model_id` to,
/// which for a split set is its first part. A safetensors checkpoint answers `None`.
fn hf_gguf(hf: &str, model_id: &str) -> Option<std::path::PathBuf> {
    use crate::inference::engine::llm_engine::discover::{hub_root, resolve_hub_gguf};
    let root = std::path::PathBuf::from(hf);
    resolve_hub_gguf(model_id, &[hub_root(root.clone()), root])
}

/// The family `/api/tags`, `/api/show` and `/api/ps` report: the media family the model's
/// files declare, else the architecture a GGUF header declares, else the architecture `files`
/// reads from the model's other files, else empty.
pub(crate) fn reported_family(
    media: Option<&crate::inference::media_facts::MediaFacts>,
    header: Option<String>,
    files: impl FnOnce() -> Option<String>,
) -> String {
    media
        .and_then(|f| f.kind.listed())
        .map(str::to_string)
        .or(header)
        .or_else(files)
        .unwrap_or_default()
}

/// Estimate parameter count from on-disk model size and format.
/// GGUF mixes Q4/Q5/Q6 averaging ~0.6 bytes/param; safetensors at F16
/// is 2 bytes/param. Returns "unknown" for size==0. Used by /api/show,
/// /api/tags, and /api/ps so all three surfaces agree.
pub(super) fn format_parameter_count(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else {
        format!("{:.0}M", n as f64 / 1e6)
    }
}

pub(super) fn estimate_parameter_size(size_bytes: u64, format_str: &str) -> String {
    if size_bytes == 0 {
        return "unknown".to_string();
    }
    let bytes_per_param: f64 = if format_str == "gguf" { 0.6 } else { 2.0 };
    format_parameter_count((size_bytes as f64 / bytes_per_param) as u64)
}

/// Pull model (POST /api/pull) - Ollama format
/// Like Ollama: downloads model AND loads it into memory
/// Pick the right HTTP status for a model-load failure string.
/// HF 404s, missing config.json, and other "you asked for a model
/// that doesn't exist" cases -> 404. Network / disk / OOM / etc -> 500.
/// Shared between the whisper + TTS / image load paths.
pub(crate) fn http_status_for_load_error(msg: &str) -> axum::http::StatusCode {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("404")
        || lower.contains("not found")
        || lower.contains("no such file")
        || lower.contains("config.json")
    {
        axum::http::StatusCode::NOT_FOUND
    } else {
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    }
}

/// Map a pull-time error message onto a fitting ApiError. Upstream
/// "manifest 404" / "not found" cases are client-side issues - wrong
/// model name; return 404. Real network / disk failures stay as 500.
pub(super) fn classify_pull_error(model_name: &str, msg: &str) -> ApiError {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("404") || lower.contains("not found") || lower.contains("no such file") {
        ApiError::NotFound(format!("Model '{model_name}' not found in remote registry"))
    } else {
        ApiError::Internal(format!("Failed to pull model: {msg}"))
    }
}

pub(crate) async fn ollama_pull_model(
    State(state): State<APIServer>,
    OllamaJson(request): OllamaJson<OllamaPullRequest>,
) -> Result<Response, ApiError> {
    validate_model_id(&request.name)?;
    let model_name = normalize_model_id(&request.name);
    info!(
        "📥 PULL endpoint called: {} (normalized from: {})",
        model_name, request.name
    );
    info!("   Stream: {}", request.stream);

    // Reject unknown `source` up front with a 400 instead of letting
    // the model_manager surface it as a 500. The two supported values
    // are documented; anything else is a client typo.
    match request.source.as_str() {
        "ollama" | "huggingface" => {}
        other => {
            return Err(ApiError::Validation(format!(
                "source '{other}' not supported; use 'ollama' or 'huggingface'"
            )));
        }
    }

    // Streaming pull: emit NDJSON download progress (Ollama wire format:
    // {status,total,completed}) so GUI/CLI clients get a live progress bar.
    // The sync progress callback feeds an unbounded channel (tx.send is
    // non-blocking) that the async_stream drains, throttled to avoid
    // flooding the socket with per-chunk lines.
    if request.stream {
        let manager = state.model_manager.clone();
        let source = request.source.clone();
        let model = model_name.clone();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<(u64, u64)>();
        let progress: crate::inference::load::model_manager::ProgressCallback =
            Arc::new(move |dl, total| {
                let _ = tx.send((dl, total));
            });

        // Spawn the actual download; when it finishes the progress Arc
        // (and thus tx) is dropped, closing the channel so the drain loop
        // below terminates and we can await the join handle for the result.
        let handle = tokio::spawn(async move {
            manager
                .pull_model_with_source(&model, &source, Some(progress))
                .await
        });

        let model_for_err = model_name.clone();
        let stream = async_stream::stream! {
            // Throttle: only emit when completed advances >=1% or >=8MB.
            const MIN_BYTES_STEP: u64 = 8 * 1024 * 1024;
            let mut last_emitted: u64 = 0;
            let mut last_dl: u64 = 0;
            let mut last_total: u64 = 0;

            while let Some((dl, total)) = rx.recv().await {
                last_dl = dl;
                last_total = total;
                let pct_step = if total > 0 { total / 100 } else { MIN_BYTES_STEP };
                if dl >= last_emitted.saturating_add(pct_step)
                    || dl >= last_emitted.saturating_add(MIN_BYTES_STEP)
                {
                    last_emitted = dl;
                    let line = serde_json::json!({
                        "status": "downloading",
                        "total": total,
                        "completed": dl,
                    }).to_string() + "\n";
                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(line));
                }
            }

            // Channel closed -> download task finished. Report its outcome.
            match handle.await {
                Ok(Ok(_metadata)) => {
                    // Ensure a terminal 100% progress frame before success.
                    if last_total > 0 && last_dl < last_total {
                        last_dl = last_total;
                    }
                    let done = serde_json::json!({
                        "status": "downloading",
                        "total": last_total,
                        "completed": last_dl,
                    }).to_string() + "\n";
                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(done));
                    let ok = serde_json::json!({ "status": "success" }).to_string() + "\n";
                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(ok));
                }
                Ok(Err(e)) => {
                    error!("   ❌ Streaming pull failed: {}", e);
                    let err = serde_json::json!({
                        "status": "error",
                        "error": format!("Failed to pull model '{}': {}", model_for_err, e),
                    }).to_string() + "\n";
                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(err));
                }
                Err(e) => {
                    error!("   ❌ Streaming pull task panicked/cancelled: {}", e);
                    let err = serde_json::json!({
                        "status": "error",
                        "error": format!("Pull task failed: {}", e),
                    }).to_string() + "\n";
                    yield Ok::<_, std::io::Error>(axum::body::Bytes::from(err));
                }
            }
        };

        return Ok(Response::builder()
            .header("content-type", "application/x-ndjson")
            .body(axum::body::Body::from_stream(stream))
            .unwrap());
    }

    let manager = state.model_manager.clone();

    // 1. Download the model
    info!("   Step 1: Downloading model from {}...", request.source);
    let metadata = manager
        .pull_model_with_source(&model_name, &request.source, None)
        .await
        .map_err(|e| {
            error!("   ❌ Download failed: {}", e);
            classify_pull_error(&model_name, &e.to_string())
        })?;

    info!(
        "   ✅ Download successful: {} ({} MB)",
        metadata.id,
        metadata.size / (1024 * 1024)
    );

    // A media checkpoint is loaded by its own route on first request, not as a language model.
    let media = crate::api::handlers::media_catalogue::media_checkpoint(
        &state.huggingface_models_dir,
        &model_name,
    )
    .and_then(|dir| crate::inference::media_facts::read(&dir));
    if media.is_some() {
        info!("   PULL complete: a media checkpoint, loaded on first request");
        return Ok(Json(OllamaPullResponse {
            status: "success".to_string(),
            digest: Some(metadata.id),
            total: Some(metadata.size),
            completed: Some(metadata.size),
        })
        .into_response());
    }

    // 2. Load model into memory (like Ollama does)
    info!("   Step 2: Loading model into memory...");

    // Check if already loaded
    {
        let engines = state.engines.read().await;
        if engines.iter().any(|e| e.model_id == model_name) {
            info!("   Model already loaded in memory");
            return Ok(Json(OllamaPullResponse {
                status: "success".to_string(),
                digest: Some(metadata.id.clone()),
                total: Some(metadata.size),
                completed: Some(metadata.size),
            })
            .into_response());
        }
    }

    // Create engine and load
    let config = state.config_for_model(&model_name);

    let engine = Arc::new(LlmEngine::with_config(config));
    engine.load_model().await.map_err(|e| {
        error!("   ❌ Failed to load model: {}", e);
        ApiError::Internal(format!("Failed to load model into memory: {}", e))
    })?;

    info!("   ✅ Model loaded into memory");

    // Get keep_alive setting (use server default)
    let keep_alive_minutes = state.default_keep_alive;

    // Schedule expiration
    let expire_handle = state.schedule_expiration(model_name.clone(), keep_alive_minutes);

    // Add to loaded engines
    {
        let mut engines = state.engines.write().await;
        let mut entry = LoadedModelEntry::new(model_name.clone(), engine, Some(keep_alive_minutes));
        entry.expire_handle = Some(expire_handle);
        engines.push(entry);
    }

    info!("   ✅ PULL complete: model ready for inference");
    Ok(Json(OllamaPullResponse {
        status: "success".to_string(),
        digest: Some(metadata.id),
        total: Some(metadata.size),
        completed: Some(metadata.size),
    })
    .into_response())
}

/// Delete model (DELETE /api/delete) - Ollama format
pub(crate) async fn ollama_delete_model(
    State(state): State<APIServer>,
    OllamaJson(request): OllamaJson<OllamaDeleteRequest>,
) -> Result<StatusCode, ApiError> {
    validate_model_id(&request.name)?;
    // Normalize model ID
    let model_name = normalize_model_id(&request.name);
    info!(
        "🗑️  DELETE /api/delete - Deleting model: {} (normalized from: {})",
        model_name, request.name
    );

    // Remove from loaded engines if present and unload
    info!("   Checking if model is loaded...");
    {
        let mut engines = state.engines.write().await;
        if let Some(pos) = engines.iter().position(|e| e.model_id == model_name) {
            info!("   Model is loaded, unloading...");
            let entry = engines.remove(pos);
            // Cancel expiration timer
            if let Some(handle) = &entry.expire_handle {
                handle.abort();
            }
            // Unload the engine
            let _ = entry.unload_all().await;
            info!("   Model unloaded from memory");
        } else {
            info!("   Model not currently loaded");
        }
    }

    // through ApiError::Internal so any future real deletion failures
    // are reported as 500 instead of mislabelled as 404 (the error
    // wouldn't be "model not found", it'd be "I/O / permission".
    info!("   Deleting model files from disk...");
    match state.model_manager.delete_model(&model_name).await {
        Ok(()) => {
            info!("   ✅ Model deleted successfully");
            Ok(StatusCode::OK)
        }
        Err(e) => {
            error!("   ❌ Failed to delete model: {}", e);
            Err(ApiError::Internal(format!("Failed to delete model: {e}")))
        }
    }
}

/// What `/api/show` reports for a model held in the HuggingFace cache: the header of its
/// weights, the capabilities its chat template gives, and the template itself.
fn show_hub_model(
    state: &APIServer,
    model_name: &str,
    weights: &std::path::Path,
) -> OllamaShowResponse {
    let (model_info, arch, gguf_quant, declared_params) = gguf_facts(weights);
    let pools = arch.as_ref().map(|a| {
        model_info
            .as_ref()
            .is_some_and(|m| m.contains_key(&format!("{a}.pooling_type")))
    });
    let mut caps = capabilities_of(None, Some(pools.unwrap_or(false)));
    if caps.iter().any(|c| c == "completion") && state.model_has_vision(model_name) {
        caps.push("vision".to_string());
    }
    let template = state.read_chat_format(model_name).effective_template();
    if let Some(t) = template.as_deref() {
        if caps.iter().any(|c| c == "completion") {
            if t.contains(".Tools") || t.contains("tools") {
                caps.push("tools".to_string());
            }
            if crate::api::handlers::prompt_format::template_has_thinking(t) {
                caps.push("thinking".to_string());
            }
        }
    }
    let model_info = model_info.map(|mut m| {
        m.retain(|k, v| !(k.starts_with("tokenizer.ggml.") && v.is_array()));
        m
    });
    OllamaShowResponse {
        license: None,
        modelfile: Some(format!("FROM {}\n", weights.display())),
        parameters: None,
        template,
        details: Some(OllamaModelDetails {
            format: "gguf".to_string(),
            family: reported_family(None, arch, || file_architecture(weights)),
            families: None,
            parameter_size: declared_params
                .map(format_parameter_count)
                .unwrap_or_else(|| "unknown".to_string()),
            quantization_level: gguf_quant,
        }),
        model_info,
        capabilities: Some(caps),
    }
}

/// Show model info (POST /api/show) - Ollama format
pub(crate) async fn ollama_show_model(
    State(state): State<APIServer>,
    OllamaJson(request): OllamaJson<OllamaShowRequest>,
) -> Result<Json<OllamaShowResponse>, ApiError> {
    // Resolve model name: prefer 'model' field, fall back to 'name' field
    // (matches official Ollama API behavior). Use trim()-based emptiness
    // so whitespace-only `model: "  "` correctly falls through to `name`
    // rather than being treated as set and immediately failing validation.
    let model_name = if !request.model.trim().is_empty() {
        request.model.clone()
    } else if !request.name.trim().is_empty() {
        request.name.clone()
    } else {
        error!("📝 POST /api/show - No model name provided");
        debug!(
            "   Request: model='{}', name='{}'",
            request.model, request.name
        );
        return Err(ApiError::Validation("model is required".to_string()));
    };
    validate_model_id(&model_name)?;

    info!("📝 POST /api/show - Model: {}", model_name);
    info!("   Models dir: {}", state.ollama_models_dir);

    let manager = state.model_manager.clone();

    match manager.resolve_path(&model_name, "ollama").await {
        Ok(path) => {
            info!("   ✓ Resolved to: {}", path.display());

            // Check if model exists
            if !path.exists() {
                warn!("   ⚠️  File not found: {}", path.display());
                // The path stays in the log. A 404 body reaches whoever asked, and this endpoint
                // answers before authentication is even configured by default, so echoing the
                // model directory hands out the server's filesystem layout for free.
                warn!("model {} not found at {}", model_name, path.display());
                return Err(ApiError::NotFound(format!(
                    "Model {} not found",
                    model_name
                )));
            }

            info!("   ✅ Model found");

            // Pull on-disk size from the Ollama manifest so we can
            // surface a usable parameter_size estimate. Ollama-shaped
            // clients otherwise show "unknown" on every /api/show.
            let bytes = manager
                .estimate_model_size(&model_name, "ollama")
                .await
                .unwrap_or(0);
            // What the resolver hands back is the manifest, and a manifest describes weights
            // without being them. Metadata comes from the layer the manifest points at.
            let weights = state
                .manifest_layer_path(&model_name, "model")
                .unwrap_or_else(|| path.clone());
            let format_str = if weights
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("safetensors"))
                .unwrap_or(false)
            {
                "safetensors"
            } else {
                "gguf"
            };
            let parameter_size = estimate_parameter_size(bytes, format_str);

            // Capabilities are modality-dependent. Hardcoding
            // ["completion","embedding"] for every model misled SDKs
            // into showing those toggles for image-gen / TTS / ASR
            // checkpoints that can't actually fulfil them.
            // Stage C: surface `vision` when the model ships a projector
            // (accepts image input). Only meaningful for text LLMs - image-
            // gen / TTS / ASR already return no LLM caps. Matches Ollama,
            // which lists `vision` for llava/moondream/etc.
            let (model_info, arch, gguf_quant, declared_params) = gguf_facts(&weights);
            let pools = arch.as_ref().map(|a| {
                model_info
                    .as_ref()
                    .is_some_and(|m| m.contains_key(&format!("{a}.pooling_type")))
            });
            // An Ollama-store model is a text GGUF whether or not its header reads.
            let mut caps = capabilities_of(None, Some(pools.unwrap_or(false)));
            if caps.iter().any(|c| c == "completion") && state.model_has_vision(&model_name) {
                caps.push("vision".to_string());
            }
            if caps.iter().any(|c| c == "completion") && state.model_has_audio(&model_name) {
                caps.push("audio".to_string());
            }
            // The chat template the server wraps user and assistant messages in, so SDK
            // clients can format prompts the way /api/chat does: the declared template,
            // else the fingerprint the vocabulary's markers choose.
            let template = state.read_chat_format(&model_name).effective_template();
            // Tool calling and a separate reasoning channel are template properties: a model
            // supports them when its own template has somewhere to put them. Both spellings
            // occur - Go templates name the fields, Jinja ones name the variables.
            if let Some(t) = template.as_deref() {
                if caps.iter().any(|c| c == "completion") {
                    if t.contains(".Tools") || t.contains("tools") {
                        caps.push("tools".to_string());
                    }
                    if crate::api::handlers::prompt_format::template_has_thinking(t) {
                        caps.push("thinking".to_string());
                    }
                }
            }
            let capabilities = Some(caps);
            // Order of authority: what the distributor packaged, then what the checkpoint
            // declares, then the name and the file size. The two disagree on gpt-oss and
            // olmoe, and clients show the packaged answer.
            let cfg = state.read_manifest_config(&model_name);
            let field = |key: &str| -> Option<String> {
                cfg.as_ref()?.get(key)?.as_str().map(str::to_string)
            };
            let families = cfg
                .as_ref()
                .and_then(|c| c.get("model_families"))
                .and_then(|v| {
                    let names: Vec<String> = v
                        .as_array()?
                        .iter()
                        .filter_map(|f| f.as_str().map(str::to_string))
                        .collect();
                    (!names.is_empty()).then_some(names)
                });
            let license = state.read_manifest_layer(&model_name, "license");
            let parameters = state.read_model_parameters(&model_name);
            let system = state.read_manifest_layer(&model_name, "system");
            // A Modelfile that would rebuild this model from its own blobs: what
            // `ollama show --modelfile` prints.
            let modelfile = {
                let mut mf = format!("FROM {}\n", weights.display());
                if let Some(t) = template.as_deref() {
                    mf.push_str(&format!("TEMPLATE \"\"\"{t}\"\"\"\n"));
                }
                if let Some(sys) = system.as_deref() {
                    mf.push_str(&format!("SYSTEM \"\"\"{sys}\"\"\"\n"));
                }
                if let Some(p) = parameters.as_deref() {
                    mf.push_str(p);
                    if !p.ends_with('\n') {
                        mf.push('\n');
                    }
                }
                if let Some(l) = license.as_deref() {
                    mf.push_str(&format!("LICENSE \"\"\"{l}\"\"\"\n"));
                }
                Some(mf)
            };
            // The vocabulary arrays weigh megabytes; Ollama sends them only when asked.
            let model_info = if request.verbose {
                model_info
            } else {
                model_info.map(|mut m| {
                    m.retain(|k, v| !(k.starts_with("tokenizer.ggml.") && v.is_array()));
                    m
                })
            };
            Ok(Json(OllamaShowResponse {
                license,
                modelfile,
                parameters,
                template,
                details: Some(OllamaModelDetails {
                    format: field("model_format").unwrap_or_else(|| format_str.to_string()),
                    family: field("model_family").unwrap_or_else(|| {
                        reported_family(None, arch, || file_architecture(&weights))
                    }),
                    families,
                    parameter_size: field("model_type")
                        .or_else(|| declared_params.map(format_parameter_count))
                        .unwrap_or(parameter_size),
                    quantization_level: field("file_type").or(gguf_quant),
                }),
                model_info,
                capabilities,
            }))
        }
        Err(_) if hf_gguf(&state.huggingface_models_dir, &model_name).is_some() => {
            let weights =
                hf_gguf(&state.huggingface_models_dir, &model_name).expect("checked by the guard");
            Ok(Json(show_hub_model(&state, &model_name, &weights)))
        }
        Err(e) => {
            error!("   ❌ Resolution failed for '{}': {}", model_name, e);
            error!("   📁 Searching in: {}", state.ollama_models_dir);
            debug!("   💡 Model format: 'name:tag' or 'publisher/name:tag'");
            // "not found" is a client-side issue (404), not server-side
            // (500). Surface as NotFound so monitoring + retries treat it
            // correctly. Other resolver errors stay as Internal.
            let msg = e.to_string();
            if msg.contains("not found") || msg.contains("No such file") {
                Err(ApiError::NotFound(format!(
                    "Model '{model_name}' not found"
                )))
            } else {
                Err(ApiError::Internal(format!(
                    "Failed to resolve model '{}': {}",
                    model_name, e
                )))
            }
        }
    }
}

#[cfg(test)]
mod catalogue_tests {
    use super::*;
    use crate::tensor::gguf_write::{write_gguf_with_metadata, GgufEntry};
    use crate::tensor::quantized::gguf_file::Value;
    use crate::tensor::quantized::GgmlDType;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("loken-cat-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// A split set is named by the parameters of every part, not of the first.
    #[test]
    fn a_split_set_is_counted_across_its_parts() {
        let dir = scratch("split");
        let entry = |name: &str| GgufEntry {
            name: name.to_string(),
            dims: vec![4, 4],
            dtype: GgmlDType::F32,
            data: [0f32.to_le_bytes(); 16].concat(),
        };
        let md = |no: u32| {
            vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".into()),
                ),
                ("split.count".to_string(), Value::U32(2)),
                ("split.no".to_string(), Value::U32(no)),
            ]
        };
        let first = dir.join("set-00001-of-00002.gguf");
        write_gguf_with_metadata(&first, &md(0), &[entry("blk.0.attn_norm.weight")]).unwrap();
        write_gguf_with_metadata(
            &dir.join("set-00002-of-00002.gguf"),
            &md(1),
            &[entry("blk.1.attn_norm.weight")],
        )
        .unwrap();
        let (arch, _, declared, _) = header_facts(&first);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(arch.as_deref(), Some("llama"));
        assert_eq!(declared, Some(32), "two parts of one 4x4 tensor each");
    }

    /// `/api/show` counts a split set across its parts, as the listing does.
    #[test]
    fn show_counts_a_split_set_across_its_parts() {
        let dir = scratch("show");
        let entry = |name: &str| GgufEntry {
            name: name.to_string(),
            dims: vec![4, 4],
            dtype: GgmlDType::F32,
            data: [0f32.to_le_bytes(); 16].concat(),
        };
        let md = |no: u32| {
            vec![
                (
                    "general.architecture".to_string(),
                    Value::String("llama".into()),
                ),
                ("split.count".to_string(), Value::U32(2)),
                ("split.no".to_string(), Value::U32(no)),
            ]
        };
        let first = dir.join("set-00001-of-00002.gguf");
        write_gguf_with_metadata(&first, &md(0), &[entry("blk.0.attn_norm.weight")]).unwrap();
        write_gguf_with_metadata(
            &dir.join("set-00002-of-00002.gguf"),
            &md(1),
            &[entry("blk.1.attn_norm.weight")],
        )
        .unwrap();
        let (_, _, _, params) = gguf_facts(&first);
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(params, Some(32));
    }

    /// A model held only in the Hugging Face store is shown from its weights' header, so a
    /// client that asks for its context window and capabilities gets them.
    #[tokio::test]
    async fn show_answers_for_a_model_held_in_the_hub_store() {
        let dir = scratch("hubshow");
        let snap = dir.join("hub/models--org--quant/snapshots/abc");
        std::fs::create_dir_all(&snap).unwrap();
        let entry = GgufEntry {
            name: "blk.0.attn_norm.weight".to_string(),
            dims: vec![4, 4],
            dtype: GgmlDType::F32,
            data: [0f32.to_le_bytes(); 16].concat(),
        };
        let md = vec![
            (
                "general.architecture".to_string(),
                Value::String("llama".into()),
            ),
            ("llama.context_length".to_string(), Value::U32(4096)),
        ];
        write_gguf_with_metadata(&snap.join("quant-q4.gguf"), &md, &[entry]).unwrap();
        let state = APIServer::new(
            "/nonexistent-ollama-models".to_string(),
            dir.to_string_lossy().to_string(),
        );
        let request: OllamaShowRequest =
            serde_json::from_value(serde_json::json!({"model": "org/quant"})).unwrap();
        let shown = ollama_show_model(axum::extract::State(state), OllamaJson(request)).await;
        let _ = std::fs::remove_dir_all(&dir);
        let shown = shown.expect("a model in the hub store is shown").0;
        let info = shown.model_info.expect("the header's keys");
        assert_eq!(info["llama.context_length"], 4096);
        assert_eq!(shown.details.unwrap().family, "llama");
        assert!(shown
            .capabilities
            .unwrap()
            .contains(&"completion".to_string()));
    }

    /// A GGUF in the Hugging Face store resolves to its file; a checkpoint of safetensors does not.
    #[test]
    fn a_gguf_in_the_hub_store_is_found_by_its_repository() {
        let dir = scratch("hub");
        let snap = dir.join("hub/models--org--quant/snapshots/abc");
        std::fs::create_dir_all(&snap).unwrap();
        std::fs::write(snap.join("quant-q4.gguf"), b"").unwrap();
        let other = dir.join("hub/models--org--plain/snapshots/abc");
        std::fs::create_dir_all(&other).unwrap();
        std::fs::write(other.join("model.safetensors"), b"").unwrap();
        let root = dir.to_string_lossy().to_string();
        let found = hf_gguf(&root, "org/quant");
        let plain = hf_gguf(&root, "org/plain");
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(found, Some(snap.join("quant-q4.gguf")));
        assert_eq!(plain, None);
    }
}
