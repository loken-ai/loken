//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// The largest `*.gguf` file under `dir`, walked a few levels deep. Multi-quant repos
/// publish several Q-grade variants; the largest stands for the highest quality the user
/// downloaded.
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

/// The CLIP BPE vocabulary both SDXL towers tokenize with, from the HF cache.
pub fn find_clip_tokenizer() -> Option<PathBuf> {
    crate::inference::cache::hf::find("models--openai--clip-vit-large-patch14", "tokenizer.json")
}

/// The single weights file `model_name` names, when its family's loader takes one in place
/// of its own: a checkpoint dropped into a weight directory, a registered file, or the
/// largest GGUF of a cached FLUX repository. `None` for a family that resolves its weights
/// from a directory, and for a name whose files declare no image model.
pub fn requested_checkpoint(hf_models_dir: &str, model_name: &str) -> Option<PathBuf> {
    let path = media_checkpoint(hf_models_dir, model_name)?;
    let family = crate::inference::media_facts::read(&path)?.image_family()?;
    match family {
        ImageFamily::Boogu | ImageFamily::Flux2 => None,
        _ if path.is_file() => Some(path),
        ImageFamily::Flux => pick_largest_gguf_under(&path),
        ImageFamily::QwenImage | ImageFamily::ZImage | ImageFamily::Sdxl => None,
    }
}
