//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// Find a local Flux GGUF file from model name or HF models directory
/// Walk `dir` recursively (one level deep - typical HF snapshot
/// layouts don't nest deeper for GGUF repos) and return the largest
/// `*.gguf` file found. Multi-quant repos publish several Q-grade
/// variants (Q4_K_S, Q4_K_M, Q5_K_S, -); picking the largest is a
/// proxy for "highest quality that the user clearly downloaded
/// intentionally," with the actual VRAM-fit check done downstream
/// by find_local_flux_gguf's NVML pass.
pub(super) fn pick_largest_gguf_under(dir: &std::path::Path) -> Option<PathBuf> {
    let mut best: Option<(PathBuf, u64)> = None;
    fn walk(dir: &std::path::Path, depth: usize, best: &mut Option<(PathBuf, u64)>) {
        if depth > 3 {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(md) = entry.metadata() else {
                continue;
            };
            if md.is_dir() {
                walk(&path, depth + 1, best);
            } else if md.is_file() {
                let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                if name.to_ascii_lowercase().ends_with(".gguf") {
                    let size = md.len();
                    if best.as_ref().is_none_or(|(_, s)| size > *s) {
                        *best = Some((path, size));
                    }
                }
            }
        }
    }
    walk(dir, 0, &mut best);
    best.map(|(p, _)| p)
}

pub fn find_sdxl_checkpoint(hf_models_dir: &str, model_name: &str) -> Option<PathBuf> {
    let want = ray_variant_tag(model_name);
    let root = PathBuf::from(hf_models_dir);
    let mut best: Option<(u64, PathBuf)> = None;
    for dir in std::fs::read_dir(&root).ok()?.flatten() {
        if !dir.path().is_dir() {
            continue;
        }
        let dname = dir.file_name().to_string_lossy().to_ascii_lowercase();
        for f in std::fs::read_dir(dir.path()).ok()?.flatten() {
            let p = f.path();
            if !p
                .extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))
            {
                continue;
            }
            // THE CHECKPOINT DECIDES, NOT ITS NAME. This used to skip any directory
            // whose name failed `is_sdxl_model` - a hard-coded list of three fine-tune
            // names plus the substring "sdxl" - so an ordinary SDXL checkpoint dropped
            // in under any other name was never scanned, never listed, and failed with
            // no message. The header carries the architecture; read it.
            if image_family_from_header(&p) != Some("sdxl") {
                continue;
            }
            let stem = p.file_stem().and_then(|x| x.to_str()).unwrap_or_default();
            // An empty request tag means "any SDXL"; otherwise the tags must match.
            if !want.is_empty() && ray_variant_tag(stem) != want && dname != want {
                continue;
            }
            let size = f.metadata().map(|m| m.len()).unwrap_or(0);
            if best.as_ref().is_none_or(|(b, _)| size > *b) {
                best = Some((size, p));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// The CLIP BPE vocabulary both SDXL towers tokenize with, from the HF cache.
pub fn find_clip_tokenizer() -> Option<PathBuf> {
    crate::inference::cache::hf::find("models--openai--clip-vit-large-patch14", "tokenizer.json")
}

pub(super) fn find_local_flux_gguf(hf_models_dir: &str, model_name: &str) -> Option<PathBuf> {
    // Ray fp8-scaled Flux fine-tunes live as plain safetensors under `rayflux/` (not GGUF, not an
    // HF-cache repo): resolve them directly to the largest safetensors in that dir.
    if is_ray_fp8_model(model_name) {
        let dir = PathBuf::from(hf_models_dir).join("rayflux");
        // Several Ray Flux variants coexist in the dir (Rayflux, Rayflux_Krea,
        // Rayflux.Horndog, ...). Select by NORMALIZED VARIANT TAG equality - the
        // requested tag and each file stem reduce through `ray_variant_tag`, so a
        // "rayflux-horndog" request only matches a Horndog file and a plain
        // "rayflux" request excludes every named variant. Size only breaks ties
        // within the same variant.
        let want = ray_variant_tag(model_name);
        let best = std::fs::read_dir(&dir)
            .ok()?
            .flatten()
            .filter(|e| {
                let p = e.path();
                let is_st = p
                    .extension()
                    .is_some_and(|x| x.eq_ignore_ascii_case("safetensors"));
                let tag = p
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(ray_variant_tag)
                    .unwrap_or_default();
                is_st && tag == want
            })
            .max_by_key(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
            .map(|e| e.path());
        if best.is_some() {
            return best;
        }
        // fall through: maybe the name also matches a cached repo below
    }
    let hf_manager = crate::inference::load::huggingface_manager::HuggingFaceManager::new(
        PathBuf::from(hf_models_dir),
    );
    // First try exact model name match. get_model_path may return the
    // snapshot directory (typical for HF cache repos like
    // lmz/candle-flux/snapshots/<sha>/) rather than the GGUF file
    // itself - recurse inside to pick the largest *.gguf, which for
    // single-file repos like lmz/candle-flux is the unique
    // flux1-schnell.gguf, and for multi-quant repos like
    // city96/FLUX.1-schnell-gguf picks the highest-quality variant.
    if let Some(path) = hf_manager.get_model_path(model_name) {
        if path.is_file() {
            return Some(path);
        }
        if path.is_dir() {
            return pick_largest_gguf_under(&path);
        }
    }

    // Scan directory for flux schnell GGUFs - pick the best one for available VRAM
    let hf_dir = PathBuf::from(hf_models_dir);
    if !hf_dir.exists() {
        return None;
    }

    // A Kontext model resolves to a kontext GGUF; anything else to a schnell GGUF.
    let want = if model_name.to_lowercase().contains("kontext") {
        "kontext"
    } else {
        "schnell"
    };
    let entries = std::fs::read_dir(&hf_dir).ok()?;
    let mut candidates: Vec<(PathBuf, u64)> = entries
        .flatten()
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().to_lowercase();
            if name.contains("flux") && name.contains(want) && name.ends_with(".gguf") {
                let size = entry.metadata().ok()?.len();
                Some((entry.path(), size))
            } else {
                None
            }
        })
        .collect();

    if candidates.is_empty() {
        return None;
    }

    // Sort by size ascending
    candidates.sort_by_key(|(_path, size)| *size);

    // Detect GPU free VRAM via NVML - reserve 2GB for CLIP (~400MB) + activations (~1.5GB)
    let vram_headroom_bytes: u64 = 2 * 1024 * 1024 * 1024;

    let (_total_vram, free_vram) = (|| -> Option<(u64, u64)> {
        let nvml = nvml_wrapper::Nvml::init().ok()?;
        let gpu = nvml.device_by_index(0).ok()?;
        let mem = gpu.memory_info().ok()?;
        info!(
            "GPU VRAM: total={:.1}GB, free={:.1}GB, used={:.1}GB",
            mem.total as f64 / 1e9,
            mem.free as f64 / 1e9,
            mem.used as f64 / 1e9
        );
        Some((mem.total, mem.free))
    })()
    .unwrap_or((8 * 1024 * 1024 * 1024, 6 * 1024 * 1024 * 1024));

    // Use free VRAM (not total) to account for other processes
    let max_model_size = free_vram.saturating_sub(vram_headroom_bytes);
    info!(
        "Flux GGUF budget: {:.1}GB (free {:.1}GB - {:.1}GB headroom)",
        max_model_size as f64 / 1e9,
        free_vram as f64 / 1e9,
        vram_headroom_bytes as f64 / 1e9
    );

    // Pick largest GGUF that fits, or smallest if none fit
    let best = candidates
        .iter()
        .rev() // Largest first
        .find(|(_path, size)| *size <= max_model_size)
        .or_else(|| candidates.first()) // Fallback to smallest
        .map(|(path, _size)| path.clone());

    if let Some(ref path) = best {
        let size_gb = std::fs::metadata(path)
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0);
        info!(
            "Selected Flux GGUF: {} ({:.1}GB, VRAM budget: {:.1}GB)",
            path.file_name().unwrap_or_default().to_string_lossy(),
            size_gb,
            max_model_size as f64 / 1e9
        );
    }

    best
}

/// How long an image load waits for a GPU to free up before telling the caller to
/// retry. Long enough to cover a chat generation or two (the usual thing holding the
/// card), short enough that a user is not left hanging.
pub(super) const IMAGE_HEADROOM_WAIT_SECS: u64 = 90;

/// How many times a load may lose its card to another engine mid-upload before the
/// caller is told to retry. Two retries cover the usual case (a chat generation and
/// an audio render finishing) without letting a request grind on indefinitely.
pub(super) const IMAGE_LOAD_ATTEMPTS: u32 = 3;

/// Detect whether model name refers to Z-Image. Includes the Ray drop-in
/// variant Rayzist (an fp8-scaled Z-Image fine-tune).
pub(crate) fn is_z_image_model(model_name: &str) -> bool {
    let lower = model_name.to_lowercase();
    lower.contains("z-image") || lower.contains("z_image") || lower.contains("rayzist")
}

/// Detect whether model name refers to Qwen-Image (text-to-image). Includes the
/// Ray drop-in variant RayQwest (an fp8-scaled Qwen-Image fine-tune).
pub(crate) fn is_qwen_image_model(model_name: &str) -> bool {
    let lower = model_name.to_lowercase();
    lower.contains("qwen-image") || lower.contains("qwen_image") || lower.contains("rayqwest")
}

/// Normalized variant tag of a Ray checkpoint file stem or model name:
/// lowercase, split on `.`/`_`/`-`/`:`, keep only name words (version,
/// precision and container tokens dropped), joined with `-`.
/// "Rayflux.Horndog" -> "rayflux-horndog"; "Rayflux.v1.0_fp8" -> "rayflux";
/// "Rayflux_Krea.v1.0" -> "rayflux-krea"; "Rayzist.v1.0.fp8_e4m3fn.full" -> "rayzist".
/// Both sides of a lookup (file stem, requested model name) normalize through
/// this, so new variants need no code change.
pub(crate) fn ray_variant_tag(stem: &str) -> String {
    fn dropped(t: &str) -> bool {
        t.is_empty()
            || t == "latest"
            || t.chars().all(|c| c.is_ascii_digit())
            || (t.starts_with('v') && t[1..].chars().next().is_some_and(|c| c.is_ascii_digit()))
            || matches!(
                t,
                "fp8"
                    | "fp16"
                    | "bf16"
                    | "full"
                    | "unet"
                    | "only"
                    | "scaled"
                    | "safetensors"
                    | "gguf"
            )
            || ((t.starts_with('q') || t.starts_with('e'))
                && t.len() >= 2
                && t[1..].chars().next().is_some_and(|c| c.is_ascii_digit()))
    }
    stem.to_ascii_lowercase()
        .split(['.', '_', '-', ':'])
        .filter(|t| !dropped(t))
        .collect::<Vec<_>>()
        .join("-")
}

/// The catalogue tag of a checkpoint sitting in a plain weight directory: its file
/// stem's variant tag when that names a family the image engine serves, else the
/// DIRECTORY name - which is what someone dropping a checkpoint into
/// `<models>/<name>/` would expect to ask for.
///
/// Both the listing (`/api/tags`) and the resolver below go through this, so a model
/// is always fetched under the name it was advertised as. They used to derive it
/// separately, which is only harmless while the two copies agree.
pub(crate) fn local_checkpoint_tag(dir_name: &str, file_stem: &str) -> String {
    match ray_variant_tag(file_stem) {
        t if !t.is_empty() && is_image_gen_model(&t) => t,
        _ => dir_name.to_ascii_lowercase(),
    }
}

/// A whole image model is billions of parameters; anything smaller in a weight
/// directory is a VAE, a text encoder or an adapter, and offering it as a model
/// would hand the user something unloadable.
pub(crate) const SMALLEST_IMAGE_MODEL: u64 = 1_500_000_000;

/// Resolve a checkpoint dropped into a plain weight directory by the tag it is
/// advertised under, whatever family it belongs to.
///
/// This is the generic form of the per-family finders below. Those key off a FIXED
/// directory (`rayzist/`, `rayqwest/`), so the second drop-in of a family needed a
/// second hardcoded name, and the third would have needed a third; a checkpoint in
/// any other directory resolved to nothing and was served by whatever the engine
/// happened to have. Returns the file and the family its HEADER declares - the
/// checkpoint decides what it is, its name only decides what to call it.
pub(crate) fn find_local_image_checkpoint(
    hf_models_dir: &str,
    model_name: &str,
) -> Option<(PathBuf, &'static str)> {
    let want = model_name.to_ascii_lowercase();
    let mut best: Option<(u64, PathBuf, &'static str)> = None;
    for dir in std::fs::read_dir(hf_models_dir)
        .into_iter()
        .flatten()
        .flatten()
    {
        let dname = dir.file_name().to_string_lossy().to_ascii_lowercase();
        // `hub/` is the HF-cache layout the standard managers already resolve.
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
            if size < SMALLEST_IMAGE_MODEL {
                continue;
            }
            let Some(stem) = p.file_stem().and_then(|x| x.to_str()) else {
                continue;
            };
            if local_checkpoint_tag(&dname, stem) != want {
                continue;
            }
            let Some(fam) = image_family_from_header(&p) else {
                continue;
            };
            // Same tag over several files (precision variants): keep the largest, as
            // the listing does, so both name the same file.
            if best.as_ref().is_none_or(|(sz, _, _)| *sz < size) {
                best = Some((size, p, fam));
            }
        }
    }
    best.map(|(_, p, fam)| (p, fam))
}

/// Resolve a Ray checkpoint under `<hf_dir>/<family_dir>/` by normalized
/// variant-tag equality ("rayzist" matches Rayzist.v2.0.safetensors). Largest
/// file breaks ties within the same tag (precision variants).
pub(crate) fn find_ray_checkpoint(
    hf_models_dir: &str,
    family_dir: &str,
    model_name: &str,
) -> Option<PathBuf> {
    let want = ray_variant_tag(model_name);
    std::fs::read_dir(PathBuf::from(hf_models_dir).join(family_dir))
        .ok()?
        .flatten()
        .filter(|e| {
            let p = e.path();
            p.extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))
                && p.file_stem().and_then(|s| s.to_str()).map(ray_variant_tag) == Some(want.clone())
        })
        .max_by_key(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .map(|e| e.path())
}

/// Detect whether a model name refers to a Ray / MutantSparrow fp8-scaled
/// checkpoint (Rayflux / Rayflux-Krea over Flux, RayQwest over Qwen-Image,
/// Rayzist over Z-Image). These load through the fp8-scaled safetensors bridge
/// ([`crate::inference::load::fp8_scaled`]) rather than the GGUF path. `rayflux`
/// already matches the Flux family via its `flux` substring; `rayqwest` /
/// `rayzist` are routed by the family predicates above.
pub(crate) fn is_ray_fp8_model(model_name: &str) -> bool {
    let lower = model_name.to_lowercase();
    lower.contains("rayflux") || lower.contains("rayqwest") || lower.contains("rayzist")
}

/// Detect whether a model name refers to Boogu-Image (3-stream MMDiT + Qwen3-VL encoder).
pub(crate) fn is_boogu_model(model_name: &str) -> bool {
    model_name.to_lowercase().contains("boogu")
}

/// The image-gen family a model name maps to: "boogu", "qwen-image", "zimage", or "flux"
/// (the default for FLUX-family names). Only meaningful for names that pass
/// [`is_image_gen_model`].
///
/// NOTE (design): this classifies by NAME substring, which is brittle (renames, new families,
/// fp8 variants all need edits here). The robust source of truth is the checkpoint's METADATA
/// (safetensors tensor signature / GGUF `general.architecture`) - see
/// [`image_family_from_header`], which detects the family from the DiT weights. The routing still
/// keys off the name because it must pick a loader BEFORE resolving the file; migrating the whole
/// path to metadata (resolve file -> read arch -> route) is the clean follow-up.
/// The SDXL family: the three Ray SDXL fine-tunes, plus anything named sdxl.
pub(crate) fn is_sdxl_model(model_name: &str) -> bool {
    let l = model_name.to_lowercase();
    l.contains("sdxl")
        || l.contains("raymnants")
        || l.contains("rayctifier")
        || l.contains("rayburn")
}
