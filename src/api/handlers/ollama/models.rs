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
) -> Result<Json<OllamaListModelsResponse>, ApiError> {
    info!("📋 GET /api/tags - Listing models (Ollama format)");

    let manager = state.model_manager.clone();

    match manager.list_models().await {
        Ok(mut models) => {
            #[cfg(feature = "media")]
            inject_local_boogu(&state, &mut models);
            let count = models.len();
            info!("   Found {} model(s)", count);
            // Stable alphabetical order (mirror 6615f9a on /v1/models).
            models.sort_by(|a, b| a.id.cmp(&b.id));

            let ollama_models: Vec<OllamaModel> = models
                .into_iter()
                .map(|m| {
                    let size_mb = m.size / (1024 * 1024);
                    info!("   • {} ({} MB, source: {})", m.id, size_mb, m.source);
                    let source = match m.source.as_str() {
                        "huggingface" => ModelSource::HuggingFace,
                        _ => ModelSource::Ollama,
                    };
                    // Fill in details.{format, parameter_size} so SDK
                    // clients (LangChain-Ollama, Open-WebUI, etc.) can
                    // surface a friendly catalog without an extra
                    // /api/show round-trip per row.
                    let format_str = match source {
                        ModelSource::HuggingFace => "safetensors",
                        _ => "gguf",
                    };
                    let details = OllamaModelDetails {
                        format: format_str.to_string(),
                        family: infer_model_family(&m.id).to_string(),
                        families: None,
                        parameter_size: estimate_parameter_size(m.size, format_str),
                        quantization_level: None,
                    };
                    {
                        let mut om = OllamaModel::new(m.id.clone(), m.size, m.downloaded_at)
                            .with_source(source)
                            .with_digest(m.digest)
                            .with_details(details);
                        om.capabilities = model_capabilities(&m.id);
                        om.defaults = media_model_defaults(&m.id);
                        om
                    }
                })
                .collect();

            info!("   ✅ List complete");
            Ok(Json(OllamaListModelsResponse::new(ollama_models)))
        }
        Err(e) => {
            error!("   ❌ Failed to list models: {}", e);
            Err(ApiError::Internal(format!("Failed to list models: {}", e)))
        }
    }
}

#[cfg(feature = "media")]
/// Surface Boogu-Image (text-to-image) in the model list. Its weights live in a plain `boogu/`
/// subdir (not HF-cache `models--*` format), so the standard managers don't discover it; the GUI's
/// dynamic image-model dropdown filters /api/tags by `details.family`, so it needs a listing. Adds
/// a synthetic `boogu` entry only when the DiT checkpoint is actually present on disk.
pub(super) fn inject_local_boogu(
    state: &APIServer,
    models: &mut Vec<crate::inference::load::model_manager::ModelMetadata>,
) {
    // Local single-model weight dirs the standard managers don't list (plain subdirs, not
    // HF-cache repos). Each becomes one synthetic catalog entry when its weights exist.
    // `dedup_family`: collapse same-family manager entries into the canonical local one (used
    // when discovered repo variants are not independently loadable, like the Boogu GGUFs);
    // None = the entry coexists with other models of its family (rayflux next to Flux GGUFs).
    struct LocalModel {
        id: &'static str,
        rel: &'static str,
        dedup_family: Option<&'static str>,
    }
    const LOCAL_MODELS: &[LocalModel] = &[LocalModel {
        id: "boogu",
        rel: "boogu/diffusion_models/boogu_image_turbo_fp8_scaled.safetensors",
        dedup_family: Some("boogu"),
    }];
    // Media engines resolved through snapshot/config dirs (not fixed rels):
    // advertise each one whose weights are actually present, so clients can
    // build audio/music pickers dynamically.
    {
        let mut push = |id: &str, path: std::path::PathBuf| {
            let Ok(meta) = std::fs::metadata(&path) else {
                return;
            };
            models.push(crate::inference::load::model_manager::ModelMetadata {
                id: id.to_string(),
                name: id.to_string(),
                size: meta.len(),
                downloaded_at: chrono::Utc::now().to_rfc3339(),
                files: vec![path.to_string_lossy().into_owned()],
                source: "huggingface".to_string(),
                digest: "sha256:0000000000000000000000000000000000000000".to_string(),
            });
        };
        if let Some(dir) = crate::inference::model::stable_audio::model_dir() {
            push("stable-audio", dir.join("model.safetensors"));
        }
        push(
            "ezaudio",
            crate::inference::model::ezaudio::vae::ezaudio_pt("ckpts/s3/ezaudio_s3_l.pt"),
        );
        for v in ["turbo", "sft", "base", "xl-turbo", "xl-sft", "xl-base"] {
            let gguf = format!("acestep-v15-{}-Q8_0.gguf", v.replace("xl-", "xl-"));
            let gguf = match v {
                "turbo" => "acestep-v15-turbo-Q8_0.gguf".to_string(),
                "sft" => "acestep-v15-sft-Q8_0.gguf".to_string(),
                "base" => "acestep-v15-base-Q8_0.gguf".to_string(),
                "xl-turbo" => "acestep-v15-xl-turbo-Q8_0.gguf".to_string(),
                "xl-sft" => "acestep-v15-xl-sft-Q8_0.gguf".to_string(),
                "xl-base" => "acestep-v15-xl-base-Q8_0.gguf".to_string(),
                _ => gguf,
            };
            push(
                &format!("ace-step-{v}"),
                crate::inference::model::acestep::fsq::acestep_gguf(&gguf),
            );
        }
        push(
            "wan",
            crate::inference::model::wan::vae::wan_file("Wan2.1_VAE.pth"),
        );
        // The 14B, when its checkpoint is there. The loader has always supported it -
        // `WanVariant::from_token` picks it as soon as the requested name contains "14" -
        // and nothing ever advertised it, so the only way to reach the better model was to
        // know that rule and type a name for it. `push` lists it only if the file exists.
        push(
            "wan-14b",
            crate::inference::model::wan::dit::wan_14b_dit_file(),
        );
        // Video fine-tunes dropped in `<models>/wan/`, so a checkpoint that the loader can
        // already resolve is also one the picker can offer. Without this the resolver was
        // reachable only by typing a name nobody could discover - the capability existed
        // and had no way in.
        {
            let dir = std::path::Path::new(&state.huggingface_models_dir)
                .join(crate::inference::model::wan::dit::WAN_CHECKPOINT_DIR);
            if let Ok(rd) = std::fs::read_dir(&dir) {
                for e in rd.flatten() {
                    let p = e.path();
                    // Both formats: the 1.3B ships as safetensors, every community 14B is
                    // a GGUF. Listing only one would have offered the checkpoints nobody has.
                    let known = p.extension().is_some_and(|x| {
                        x.eq_ignore_ascii_case("safetensors") || x.eq_ignore_ascii_case("gguf")
                    });
                    if !known {
                        continue;
                    }
                    let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else {
                        continue;
                    };
                    let tag = crate::inference::model::wan::dit::wan_checkpoint_tag(stem);
                    // The family is read off the NAME, so a file whose tag does not start
                    // with "wan" would be advertised as something other than video. Say so
                    // rather than listing it under the wrong kind.
                    if !tag.starts_with("wan") {
                        tracing::warn!(
                            "{}: a video checkpoint's name must start with 'Wan' to be \
                             offered as one; '{tag}' would be listed as another kind",
                            p.display()
                        );
                        continue;
                    }
                    push(&tag, p);
                }
            }
        }
        // Dedicated instruction editors: named entries so edit-capable pickers
        // can list them (their weights otherwise resolve through name checks).
        let hf = std::path::Path::new(&state.huggingface_models_dir);
        push("flux-kontext", hf.join("flux-kontext-q4km.gguf"));
        push("qwen-image-edit", hf.join("qwen-image-edit/dit-q4km.gguf"));
        // FLUX.2 Klein keeps its parts in diffusers SUBDIRECTORIES, so the plain weight-dir
        // scan below (one level deep) cannot see it. Advertised by its DiT's presence.
        push(
            "flux2-klein-4b",
            crate::inference::engine::flux2_engine::weights_dir(&state.huggingface_models_dir)
                .join("transformer/diffusion_pytorch_model.safetensors"),
        );
    }
    for lm in LOCAL_MODELS {
        let weights = std::path::Path::new(&state.huggingface_models_dir).join(lm.rel);
        if !weights.exists() {
            continue; // no local weights: leave whatever the managers discovered untouched
        }
        if let Some(fam) = lm.dedup_family {
            models.retain(|m| infer_model_family(&m.id) != fam);
        }
        let size = std::fs::metadata(&weights).map(|m| m.len()).unwrap_or(0);
        models.push(crate::inference::load::model_manager::ModelMetadata {
            id: lm.id.to_string(),
            name: lm.id.to_string(),
            size,
            downloaded_at: chrono::Utc::now().to_rfc3339(),
            files: vec![lm.rel.to_string()],
            source: "huggingface".to_string(),
            digest: "sha256:0000000000000000000000000000000000000000".to_string(),
        });
    }

    // Ray / MutantSparrow drop-in checkpoints: SCAN the `ray*` weight dirs and
    // derive one catalog entry per safetensors file - the tag comes from the
    // file stem via `ray_variant_tag` (no per-variant code). A file is only
    // advertised when its tag maps to a family the image engine can actually
    // load (`is_image_gen_model`), so checkpoints of unported bases (e.g. the
    // SDXL Ray variants) stay silent instead of erroring at load time.
    {
        let root = std::path::Path::new(&state.huggingface_models_dir);
        let mut seen: std::collections::HashMap<String, (u64, String)> = Default::default();
        for dir in std::fs::read_dir(root).into_iter().flatten().flatten() {
            let dname = dir.file_name().to_string_lossy().to_ascii_lowercase();
            // EVERY plain weight directory, not just the ones called `ray*`. That name
            // filter is why only the Ray checkpoints ever appeared in the catalogue: a
            // model dropped in under any other name was not scanned, not advertised and
            // not loadable, with nothing said about it. `hub/` is skipped because the
            // standard HF-cache manager already lists it.
            if !dir.path().is_dir() || dname == "hub" || dname.starts_with('.') {
                continue;
            }
            for f in std::fs::read_dir(dir.path())
                .into_iter()
                .flatten()
                .flatten()
            {
                let p = f.path();
                if !p
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))
                {
                    continue;
                }
                let size = f.metadata().map(|m| m.len()).unwrap_or(0);
                if size < crate::api::handlers::media::SMALLEST_IMAGE_MODEL {
                    continue;
                }
                let Some(stem) = p.file_stem().and_then(|x| x.to_str()) else {
                    continue;
                };
                // THE CHECKPOINT DECIDES ITS FAMILY, its name only decides what to call
                // it. Reading the header means a new model is recognised because of what
                // it IS, not because someone added its name to a list in the source.
                let Some(fam) = crate::api::handlers::media::image_family_from_header(&p) else {
                    continue;
                };
                // Families whose engine accepts an ALTERNATE local checkpoint. A family
                // outside this set stays silent rather than erroring at load time.
                if fam != "flux" && fam != "qwen-image" && fam != "zimage" && fam != "sdxl" {
                    continue;
                }
                // Keep the Ray tags exactly as before; anything else is named after its
                // directory, which is what a user dropping a checkpoint would expect.
                // The RESOLVER goes through the same rule, so a model is always fetched
                // under the name it is advertised as.
                let tag = crate::api::handlers::media::local_checkpoint_tag(&dname, stem);
                let rel = format!(
                    "{dname}/{}",
                    p.file_name().unwrap_or_default().to_string_lossy()
                );
                // Same tag from several files (precision variants): keep the largest.
                match seen.get(&tag) {
                    Some((sz, _)) if *sz >= size => {}
                    _ => {
                        seen.insert(tag, (size, rel));
                    }
                }
            }
        }
        for (tag, (size, rel)) in seen {
            models.push(crate::inference::load::model_manager::ModelMetadata {
                id: tag.clone(),
                name: tag,
                size,
                downloaded_at: chrono::Utc::now().to_rfc3339(),
                files: vec![rel],
                source: "huggingface".to_string(),
                digest: "sha256:0000000000000000000000000000000000000000".to_string(),
            });
        }
    }
}

/// Best-effort family inference from the model id. Maps the common
/// Ollama families so SDK clients see a non-misleading badge in their
/// catalog UI (`details.family`). Defaults to "llama" - Ollama's own
/// fallback for unrecognized GGUF arch strings.
/// What a model can DO, derived from its family + name - the SERVER-side
/// single source of truth the /api/tags listing exposes so clients never
/// hardcode model lists. Only capabilities with a WORKING wired path are
/// advertised (e.g. boogu has no img2img yet, so it is txt2img-only).
pub(crate) fn model_capabilities(model_id: &str) -> Vec<String> {
    let lower = model_id.to_lowercase();
    let fam = infer_model_family(model_id);
    let caps: &[&str] = match fam {
        "flux" => {
            if lower.contains("kontext") {
                &["txt2img", "img2img", "edit"]
            } else {
                &["txt2img", "img2img"]
            }
        }
        "z-image" => &["txt2img", "img2img"],
        // The family's own DiT here IS the edit checkpoint, so it instruction-edits.
        // A drop-in of the family (RayQwest, the step-distilled Flash) is trained for
        // text-to-image only, and advertising an edit it cannot do would put it in
        // front of users on a tab it silently fails.
        "qwen-image" => {
            #[cfg(feature = "image")]
            let step_distilled =
                crate::inference::engine::qwen_image_engine::is_step_distilled(model_id);
            #[cfg(not(feature = "image"))]
            let step_distilled = false;
            if lower.contains("rayqwest") || step_distilled {
                &["txt2img"]
            } else {
                &["txt2img", "edit"]
            }
        }
        // SDXL generates only: the port has no img2img or instruction-edit path, so
        // advertising either would put it in front of users on tabs it cannot serve.
        // SDXL generates and re-draws. It has no INSTRUCTION editor - there is no
        // Kontext-style checkpoint for the family here - so "edit" stays off and the
        // Edit tab drives it through img2img with a strength.
        "sdxl" => &["txt2img", "img2img"],
        // No img2img or instruction-edit path is wired for FLUX.2 yet, and advertising one
        // would put it in front of users on a tab it silently fails.
        "flux2" | "boogu" | "stable-diffusion" => &["txt2img"],
        "whisper" => &["asr"],
        "tts" => &["tts"],
        "bert" => &["embedding"],
        _ => {
            if lower.contains("stable-audio") {
                &["sfx", "loops", "audio-variations"]
            } else if lower.contains("ezaudio") {
                &["sfx"]
            } else if lower.contains("ace-step") || lower.contains("acestep") {
                &["music"]
            } else if lower.starts_with("wan") {
                &["video"]
            } else if lower.contains("midi") {
                &["midi"]
            } else {
                // The name every client gates on for text generation.
                &["completion"]
            }
        }
    };
    caps.iter().map(|s| s.to_string()).collect()
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

/// What a checkpoint says about itself: its metadata as JSON, its architecture, the label its
/// tensors are stored under, and how many parameters it holds.
///
/// Reads the header, not the weights.
fn gguf_facts(
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
    // Most checkpoints do not declare a count; the shapes are the count.
    let params = Some(
        content
            .tensor_infos
            .values()
            .map(|t| t.elem_count() as u64)
            .sum::<u64>(),
    )
    .filter(|n| *n > 0);
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

/// Recommended generation defaults per media model (the knowledge the GUI
/// used to hardcode in its combo tables). None for models without presets.
pub(crate) fn media_model_defaults(model_id: &str) -> Option<serde_json::Value> {
    let lower = model_id.to_lowercase();
    if lower.contains("ace-step") || lower.contains("acestep") {
        let (steps, cfg) = if lower.contains("turbo") {
            (27, 1.0)
        } else {
            (50, 4.5)
        };
        return Some(serde_json::json!({ "steps": steps, "cfg": cfg }));
    }
    if lower.contains("stable-audio") {
        return Some(serde_json::json!({ "steps": 25, "cfg": 7.0, "max_seconds": 47 }));
    }
    if lower.contains("ezaudio") {
        return Some(serde_json::json!({ "steps": 60, "cfg": 3.0, "max_seconds": 30 }));
    }
    if lower.contains("kontext") {
        return Some(serde_json::json!({ "steps": 28, "cfg": 2.5 }));
    }
    // Every image family publishes its recommended knobs from the SAME table the
    // server itself samples with, so a client that switches models gets values the
    // model was validated at. Carrying a previous model's slider over is why an
    // instruction edit sometimes came back barely applied (a Kontext-class editor
    // needs ~28 steps where a distilled one is done in 4).
    #[cfg(feature = "image")]
    if crate::api::handlers::family::is_image(model_id) {
        // A family with no recipe of its own publishes NOTHING rather than another
        // family's numbers: a client that reads them sets its sliders from them.
        let d = crate::api::handlers::media::image_model_defaults(model_id).ok()?;
        return Some(serde_json::json!({
            "steps": d.steps,
            "cfg": d.guidance,
            "size": d.size,
        }));
    }
    None
}

pub(crate) fn infer_model_family(model_id: &str) -> &'static str {
    let lower = model_id.to_lowercase();
    // Order matters - match the more specific names first
    // (gemma4-style names contain "gemma" but should resolve to "gemma").
    // Video first: the family decides what a picker offers, and "wan" was falling all the
    // way through to the text default - so the server described its video model as an LLM
    // and no client could tell it apart from one. A fine-tune dropped in `<models>/wan/`
    // is advertised under its own tag, which starts with "wan" by the same convention the
    // Ray families use, so it lands here too.
    if lower.starts_with("wan") {
        "video"
    } else if lower.contains("gemma") {
        "gemma"
    }
    // Qwen-Image is a diffusion T2I model, NOT the Qwen text LLM - must match
    // before the generic "qwen" arm so it resolves to the image family (and so
    // /api/show + /api/tags don't advertise chat/embedding on it).
    // BEFORE any `flux` test: "flux2-klein" contains "flux".
    else if {
        #[cfg(feature = "image")]
        {
            crate::inference::engine::flux2_engine::is_flux2_model(&lower)
        }
        #[cfg(not(feature = "image"))]
        {
            false
        }
    } {
        "flux2"
    } else if lower.contains("qwen-image")
        || lower.contains("qwen_image")
        || lower.contains("rayqwest")
    {
        "qwen-image"
    } else if lower.contains("qwen") {
        "qwen"
    } else if lower.contains("mistral") || lower.contains("mixtral") {
        "mistral"
    } else if lower.contains("deepseek") {
        "deepseek"
    } else if lower.contains("phi") {
        "phi"
    } else if lower.contains("starcoder") || lower.contains("starchat") {
        "starcoder"
    } else if lower.contains("falcon") {
        "falcon"
    } else if lower.contains("codellama") {
        "codellama"
    } else if lower.contains("codestral") {
        "codestral"
    } else if lower.contains("devstral") {
        "devstral"
    } else if lower.contains("granite") {
        "granite"
    } else if lower.contains("moondream") {
        "moondream"
    } else if lower.contains("nomic") || lower.contains("bge") || lower.contains("embed") {
        "bert"
    } else if lower.contains("glm") {
        "chatglm"
    }
    // Non-LLM image-gen families. MUST come before the generic
    // "stable" -> "stablelm" arm so that "stable-diffusion" /
    // "stable-cascade" don't get mislabelled as StableLM. The
    // /api/show family is what SDK clients read to decide which
    // schema-aware UI to show.
    else if lower.contains("stable-diffusion") || lower.contains("stable-cascade") {
        "stable-diffusion"
    } else if lower.contains("sdxl")
        || lower.contains("raymnants")
        || lower.contains("rayctifier")
        || lower.contains("rayburn")
    {
        "sdxl"
    } else if lower.contains("z-image") || lower.contains("z_image") || lower.contains("rayzist") {
        "z-image"
    } else if lower.contains("boogu") {
        "boogu"
    } else if lower.contains("flux") {
        "flux"
    } else if lower.contains("whisper") {
        "whisper"
    } else if lower.contains("parler") || lower.contains("/tts-") || lower.starts_with("tts") {
        "tts"
    }
    // StableLM text LLM family - must come after stable-diffusion above
    // so the broader "stable" substring doesn't swallow the SD variants.
    else if lower.contains("stable") {
        "stablelm"
    } else {
        "llama"
    }
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
    Json(request): Json<OllamaPullRequest>,
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
    Json(request): Json<OllamaDeleteRequest>,
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
            if let Some(handle) = entry.expire_handle {
                handle.abort();
            }
            // Unload the engine
            let _ = entry.engine.unload().await;
            info!("   Model unloaded from memory");
        } else {
            info!("   Model not currently loaded");
        }
    }

    // Delete from disk. NOTE: model_manager.delete_model is currently
    // a no-op (directory-based approach) - but route the Err arm
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

/// Show model info (POST /api/show) - Ollama format
pub(crate) async fn ollama_show_model(
    State(state): State<APIServer>,
    Json(request): Json<OllamaShowRequest>,
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
            let mut caps = model_capabilities(&model_name);
            if caps.iter().any(|c| c == "completion") && state.model_has_vision(&model_name) {
                caps.push("vision".to_string());
            }
            // Surface the chat template the server uses to wrap user/
            // assistant messages so SDK clients can format prompts the
            // same way they'd appear in /api/chat. Prefer the manifest-
            // declared template; fall back to the name-inferred minimal
            // template (`<start_of_turn>` / `<|im_start|>` / etc.) so
            // the response is non-null for every loaded model.
            let template = state
                .read_chat_template(&model_name)
                .or_else(|| infer_template_from_model_name(&model_name));
            // Tool calling and a separate reasoning channel are template properties: a model
            // supports them when its own template has somewhere to put them. Both spellings
            // occur - Go templates name the fields, Jinja ones name the variables.
            if let Some(t) = template.as_deref() {
                if caps.iter().any(|c| c == "completion") {
                    if t.contains(".Tools") || t.contains("tools") {
                        caps.push("tools".to_string());
                    }
                    if t.contains(".Thinking")
                        || t.contains("enable_thinking")
                        || t.contains("reasoning_content")
                    {
                        caps.push("thinking".to_string());
                    }
                }
            }
            let capabilities = Some(caps);
            let (model_info, arch, gguf_quant, declared_params) = gguf_facts(&weights);
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
            Ok(Json(OllamaShowResponse {
                license: state.read_manifest_layer(&model_name, "license"),
                modelfile: None,
                parameters: state.read_model_parameters(&model_name),
                template,
                details: Some(OllamaModelDetails {
                    format: field("model_format").unwrap_or_else(|| format_str.to_string()),
                    family: field("model_family")
                        .or(arch)
                        .unwrap_or_else(|| infer_model_family(&model_name).to_string()),
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
