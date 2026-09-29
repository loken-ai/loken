//! Which checkpoint a media model id names, and what its files say it is.
//!
//! A request names a model. The catalogue turns that name into files - a checkpoint in a
//! plain weight directory, a pipeline directory, a Hugging Face cache snapshot - and the
//! files decide everything else through [`crate::inference::media_facts`]. The name is only
//! ever looked up, never read for what it contains.

use std::path::{Path, PathBuf};

use crate::inference::media_facts::{self, ImageFamily, MediaFacts, MediaKind};

/// A media checkpoint the Hugging Face cache listing does not name on its own: the id it
/// is offered under and the file or directory holding it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MediaEntry {
    pub id: String,
    pub path: PathBuf,
}

/// Where Piper voices are kept, one directory per voice.
pub(crate) const PIPER_DIR: &str = "piper";

/// A whole image model is billions of parameters; anything smaller in a weight directory is
/// a VAE, a text encoder or an adapter, and offering it as a model would hand the user
/// something unloadable.
pub(crate) const SMALLEST_IMAGE_MODEL: u64 = 1_500_000_000;

/// The variant tag of a checkpoint's file stem: lowercase, split on `.`/`_`/`-`/`:`, name
/// words kept (version, precision and container tokens dropped), joined with `-`.
/// `Tune.Other` -> `tune-other`; `Tune.v1.0_fp8` -> `tune`; `Tune_Third.v1.0` ->
/// `tune-third`.
pub(crate) fn variant_tag(stem: &str) -> String {
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

/// The image families whose loader accepts a checkpoint dropped into a weight directory in
/// place of its own. A drop-in of another family stays unlisted rather than failing at load.
fn takes_a_drop_in(family: ImageFamily) -> bool {
    match family {
        ImageFamily::Flux | ImageFamily::QwenImage | ImageFamily::ZImage | ImageFamily::Sdxl => {
            true
        }
        ImageFamily::Flux2 | ImageFamily::Boogu => false,
    }
}

/// The image checkpoints dropped into plain weight directories, one entry per model.
///
/// A directory holding one model is offered under the directory's name, which is what
/// someone dropping a checkpoint into `<models>/<name>/` asks for. A directory holding
/// several is offered per file, under each file's variant tag. Precision variants of one
/// model share a tag, and the largest file answers for them.
pub(crate) fn drop_in_entries(hf_dir: &str) -> Vec<MediaEntry> {
    let mut out = Vec::new();
    for dir in std::fs::read_dir(hf_dir).into_iter().flatten().flatten() {
        let dname = dir.file_name().to_string_lossy().to_ascii_lowercase();
        // `hub/` is the HF-cache layout the cache listing already names.
        if !dir.path().is_dir() || dname == "hub" || dname.starts_with('.') {
            continue;
        }
        let mut by_tag: std::collections::BTreeMap<String, (u64, PathBuf)> = Default::default();
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
            let family = media_facts::read(&p).and_then(|m| m.image_family());
            if !family.is_some_and(takes_a_drop_in) {
                continue;
            }
            let Some(stem) = p.file_stem().and_then(|x| x.to_str()) else {
                continue;
            };
            let tag = variant_tag(stem);
            match by_tag.get(&tag) {
                Some((sz, _)) if *sz >= size => {}
                _ => {
                    by_tag.insert(tag, (size, p));
                }
            }
        }
        let single = by_tag.len() == 1;
        for (tag, (_, path)) in by_tag {
            let id = if single || tag.is_empty() {
                dname.clone()
            } else {
                tag
            };
            out.push(MediaEntry { id, path });
        }
    }
    out
}

/// The media checkpoints the server registers by where their engine keeps them, and the
/// video checkpoints dropped into the video engine's directory.
#[cfg(feature = "media")]
fn registered_entries(hf_dir: &str) -> Vec<MediaEntry> {
    let mut out = Vec::new();
    let mut push = |id: &str, path: PathBuf| {
        if path.exists() {
            out.push(MediaEntry {
                id: id.to_string(),
                path,
            });
        }
    };
    let hf = Path::new(hf_dir);
    push(
        "boogu",
        hf.join("boogu/diffusion_models/boogu_image_turbo_fp8_scaled.safetensors"),
    );
    if let Some(dir) = crate::inference::model::stable_audio::model_dir() {
        push("stable-audio", dir.join("model.safetensors"));
    }
    push(
        "ezaudio",
        crate::inference::model::ezaudio::vae::ezaudio_pt("ckpts/s3/ezaudio_s3_l.pt"),
    );
    for v in ["turbo", "sft", "base", "xl-turbo", "xl-sft", "xl-base"] {
        push(
            &format!("{}{v}", super::catalogue::MUSIC_ENTRY_PREFIX),
            crate::inference::model::acestep::fsq::acestep_gguf(&format!(
                "acestep-v15-{v}-Q8_0.gguf"
            )),
        );
    }
    push(
        "wan",
        crate::inference::model::wan::vae::wan_file("Wan2.1_VAE.pth"),
    );
    push(
        "wan-14b",
        crate::inference::model::wan::dit::wan_14b_dit_file(),
    );
    // Video checkpoints dropped in `<models>/wan/`, both formats: the 1.3B ships as
    // safetensors and every community 14B as a GGUF.
    let wan_dir = hf.join(crate::inference::model::wan::dit::WAN_CHECKPOINT_DIR);
    for e in std::fs::read_dir(&wan_dir).into_iter().flatten().flatten() {
        let p = e.path();
        let known = p.extension().is_some_and(|x| {
            x.eq_ignore_ascii_case("safetensors") || x.eq_ignore_ascii_case("gguf")
        });
        if !known || media_facts::read(&p).map(|f| f.kind) != Some(MediaKind::Video) {
            continue;
        }
        let Some(stem) = p.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        push(
            &crate::inference::model::wan::dit::wan_checkpoint_tag(stem),
            p,
        );
    }
    // Piper voices, each its own checkpoint: `<models>/piper/<voice>/<voice>.onnx`, with the
    // phoneme table beside it. The directory steps out of a `hub/` cache as the voices do.
    let root = if hf.file_name().is_some_and(|n| n == "hub") {
        hf.parent().unwrap_or(hf)
    } else {
        hf
    };
    for dir in std::fs::read_dir(root.join(PIPER_DIR))
        .into_iter()
        .flatten()
        .flatten()
    {
        let voice = dir.file_name().to_string_lossy().to_string();
        push(
            &format!("{PIPER_DIR}/{voice}"),
            dir.path().join(format!("{voice}.onnx")),
        );
    }
    #[cfg(feature = "midi")]
    push(
        "midi",
        crate::config::ollama_models_dir().join("MIDI-LLM_Llama-3.2-1B.Q8_0.gguf"),
    );
    push("flux-kontext", hf.join("flux-kontext-q4km.gguf"));
    push("qwen-image-edit", hf.join("qwen-image-edit/dit-q4km.gguf"));
    push(
        "flux2-klein-4b",
        crate::inference::engine::flux2_engine::weights_dir(hf_dir)
            .join("transformer/diffusion_pytorch_model.safetensors"),
    );
    out
}

#[cfg(not(feature = "media"))]
fn registered_entries(_hf_dir: &str) -> Vec<MediaEntry> {
    Vec::new()
}

/// Every media checkpoint of the local stores that the Hugging Face cache listing does not
/// name, registered entries first.
pub(crate) fn local_media_entries(hf_dir: &str) -> Vec<MediaEntry> {
    let mut out = registered_entries(hf_dir);
    for e in drop_in_entries(hf_dir) {
        if !out.iter().any(|o| o.id == e.id) {
            out.push(e);
        }
    }
    out
}

/// A request's model id as the catalogue compares it: without the `:latest` tag Ollama-style
/// clients append.
fn bare(id: &str) -> &str {
    let id = id.trim();
    id.strip_suffix(":latest").unwrap_or(id)
}

/// Names the API takes for a checkpoint the catalogue holds under another id: OpenAI's
/// documented model ids, and the engine words the speech and music routes have always
/// accepted. A name only looks a checkpoint up; what it is, its files say.
const ALIASES: &[(&str, &str)] = &[
    ("whisper-1", "openai/whisper-small"),
    ("tts-1", "parler-tts/parler-tts-mini-v1"),
    ("tts-1-1106", "parler-tts/parler-tts-mini-v1"),
    ("tts-1-hd", "parler-tts/parler-tts-large-v1"),
    ("tts-1-hd-1106", "parler-tts/parler-tts-large-v1"),
    ("kyutai", "kyutai/tts-1.6b-en_fr"),
    ("piper", "piper/fr_FR-tom-medium"),
    ("ace-step", "ace-step-turbo"),
    ("acestep", "ace-step-turbo"),
    ("music", "ace-step-turbo"),
];

/// The id an alias or an entry of the OpenAI model list stands for.
pub(crate) fn aliased(id: &str) -> Option<&'static str> {
    ALIASES
        .iter()
        .find(|(alias, _)| alias.eq_ignore_ascii_case(id))
        .map(|(_, target)| *target)
        .or_else(|| crate::api::handlers::offered_repository(id))
}

/// A local entry or cached repository `id` names, with the catalogue id it is known by:
/// the entry's id, a cached repository by its id or its name without the owner, or what an
/// alias stands for.
pub(crate) fn resolve(hf_dir: &str, id: &str) -> Option<(String, PathBuf)> {
    let want = bare(id);
    if want.is_empty() {
        return None;
    }
    if let Some(e) = local_media_entries(hf_dir)
        .into_iter()
        .find(|e| e.id.eq_ignore_ascii_case(want))
    {
        return Some((e.id, e.path));
    }
    let hf =
        crate::inference::load::huggingface_manager::HuggingFaceManager::new(PathBuf::from(hf_dir));
    if let Some(repo) = hf.repository_named(want) {
        return hf.get_model_path(&repo).map(|p| (repo, p));
    }
    let target = aliased(want).filter(|t| !t.eq_ignore_ascii_case(want))?;
    // An alias stands for an entry or a repository, never for another alias.
    if let Some(e) = local_media_entries(hf_dir)
        .into_iter()
        .find(|e| e.id.eq_ignore_ascii_case(target))
    {
        return Some((e.id, e.path));
    }
    let repo = hf.repository_named(target)?;
    hf.get_model_path(&repo).map(|p| (repo, p))
}

/// The files `id` names.
pub(crate) fn media_checkpoint(hf_dir: &str, id: &str) -> Option<PathBuf> {
    resolve(hf_dir, id).map(|(_, p)| p)
}

/// The catalogue id `id` stands for; the id an alias names, or the id itself, when nothing
/// on disk answers to it, so a repository not downloaded yet is fetched under that id.
pub(crate) fn canonical_id(hf_dir: &str, id: &str) -> String {
    resolve(hf_dir, id)
        .map(|(c, _)| c)
        .or_else(|| aliased(bare(id)).map(str::to_string))
        .unwrap_or_else(|| bare(id).to_string())
}

/// A speech request: the synthesiser its files declare, the catalogue id of its checkpoint,
/// and the voice that follows the checkpoint's name in the request.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SpeechRequest {
    pub model: crate::inference::media_facts::SpeechModel,
    pub id: String,
    pub path: PathBuf,
    pub voice: Option<String>,
}

impl SpeechRequest {
    /// The name the engine holds this request's checkpoint and voice under.
    pub(crate) fn canonical(&self) -> String {
        match &self.voice {
            Some(v) => format!("{}/{v}", self.id),
            None => self.id.clone(),
        }
    }
}

/// Every way to cut `requested` at a separator into a checkpoint name and what follows it,
/// the whole name first and the shortest head last.
pub(crate) fn heads(requested: &str) -> Vec<(&str, &str)> {
    let requested = bare(requested);
    let mut cuts: Vec<usize> = requested
        .char_indices()
        .filter(|(_, c)| matches!(c, '-' | '/' | ':' | '_'))
        .map(|(i, _)| i)
        .collect();
    cuts.push(requested.len());
    cuts.iter()
        .rev()
        .map(|&cut| requested.split_at(cut))
        .collect()
}

/// Whether a catalogue entry, which lists names only, is the checkpoint `name` names: its
/// id, its name without the owner, or what an alias of `name` stands for.
pub(crate) fn entry_named(entry: &str, name: &str) -> bool {
    let (entry, name) = (bare(entry), bare(name));
    if name.is_empty() {
        return false;
    }
    let short = entry.rsplit('/').next().unwrap_or(entry);
    [Some(name), aliased(name)]
        .into_iter()
        .flatten()
        .any(|n| entry.eq_ignore_ascii_case(n) || short.eq_ignore_ascii_case(n))
}

/// What a speech request names: the longest part of `requested`, cut at a separator, that
/// names a checkpoint whose files declare a speech synthesiser, and the voice after it
/// (`<checkpoint>-<voice>`, `<checkpoint>/<voice>`).
pub(crate) fn speech_request(hf_dir: &str, requested: &str) -> Option<SpeechRequest> {
    for (head, tail) in heads(requested) {
        let Some((id, path)) = resolve(hf_dir, head) else {
            continue;
        };
        let Some(MediaKind::Speech(model)) = media_facts::read(&path).map(|f| f.kind) else {
            continue;
        };
        let voice = tail.trim_start_matches(['-', '/', ':', '_']);
        return Some(SpeechRequest {
            model,
            id,
            path,
            voice: (!voice.is_empty()).then(|| voice.to_string()),
        });
    }
    None
}

/// Whether `id` names a speech synthesiser this build runs an engine for.
pub(crate) fn is_served_speech(hf_dir: &str, id: &str) -> bool {
    use crate::inference::media_facts::SpeechModel;
    speech_request(hf_dir, id).is_some_and(|r| r.model != SpeechModel::Unserved)
}

/// Whether `id` names a speech recogniser this build runs an engine for.
pub(crate) fn is_served_transcription(hf_dir: &str, id: &str) -> bool {
    media_facts(hf_dir, id).is_some_and(|f| {
        f.kind
            == MediaKind::Transcription(crate::inference::media_facts::TranscriptionModel::Whisper)
    })
}

/// What the checkpoint `id` names declares itself to be.
pub(crate) fn media_facts(hf_dir: &str, id: &str) -> Option<MediaFacts> {
    media_facts::read(&media_checkpoint(hf_dir, id)?)
}

/// The image family `id`'s checkpoint declares.
pub(crate) fn image_family_of(hf_dir: &str, id: &str) -> Option<ImageFamily> {
    media_facts(hf_dir, id)?.image_family()
}

/// Whether `id` names a cached repository that holds no weight file anywhere in it.
pub(crate) fn holds_no_weights(hf_dir: &str, id: &str) -> bool {
    fn any_weights(dir: &Path, depth: usize) -> bool {
        std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .any(|e| {
                let p = e.path();
                if p.is_dir() {
                    return depth > 0 && any_weights(&p, depth - 1);
                }
                let name = p.to_string_lossy().to_ascii_lowercase();
                [
                    ".gguf",
                    ".safetensors",
                    ".bin",
                    ".pt",
                    ".pth",
                    ".ckpt",
                    ".onnx",
                ]
                .iter()
                .any(|x| name.ends_with(x))
            })
    }
    let hf =
        crate::inference::load::huggingface_manager::HuggingFaceManager::new(PathBuf::from(hf_dir));
    let Some(repo) = hf.repository_named(bare(id)) else {
        return false;
    };
    hf.get_model_path(&repo)
        .is_some_and(|dir| dir.is_dir() && !any_weights(&dir, 3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::media_facts::fixture::*;

    #[test]
    fn a_directory_holding_one_model_is_offered_under_its_name() {
        let s = Scratch::new("dropin-one");
        let dir = s.0.join("mymodel");
        safetensors(
            &dir.join("Some_Checkpoint.v2.safetensors"),
            &flux(true),
            None,
        );
        // Past the size floor without writing gigabytes: the floor reads the file length.
        let f = std::fs::OpenOptions::new()
            .append(true)
            .open(dir.join("Some_Checkpoint.v2.safetensors"))
            .unwrap();
        f.set_len(SMALLEST_IMAGE_MODEL).unwrap();
        let entries = drop_in_entries(s.0.to_str().unwrap());
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "mymodel");
        assert_eq!(
            image_family_of(s.0.to_str().unwrap(), "mymodel"),
            Some(ImageFamily::Flux)
        );
    }

    #[test]
    fn a_directory_holding_several_models_is_offered_per_file() {
        let s = Scratch::new("dropin-many");
        let dir = s.0.join("family");
        for stem in ["Tune.v1.0", "Tune.Other", "Tune_Third.v1.0"] {
            let p = dir.join(format!("{stem}.safetensors"));
            safetensors(&p, &["transformer_blocks.0.a", "img_in.w"], None);
            std::fs::OpenOptions::new()
                .append(true)
                .open(&p)
                .unwrap()
                .set_len(SMALLEST_IMAGE_MODEL)
                .unwrap();
        }
        let mut ids: Vec<String> = drop_in_entries(s.0.to_str().unwrap())
            .into_iter()
            .map(|e| e.id)
            .collect();
        ids.sort();
        assert_eq!(ids, ["tune", "tune-other", "tune-third"]);
    }

    #[test]
    fn a_cached_repository_answers_to_its_name() {
        let s = Scratch::new("hub");
        let snap = s.0.join("hub/models--Owner--Some-Model/snapshots/abc");
        json(
            &snap.join("config.json"),
            serde_json::json!({"model_type": "whisper"}),
        );
        std::fs::create_dir_all(s.0.join("hub/models--Owner--Some-Model/refs")).unwrap();
        std::fs::write(s.0.join("hub/models--Owner--Some-Model/refs/main"), "abc").unwrap();
        let hf = s.0.to_str().unwrap();
        for id in ["Owner/Some-Model", "some-model", "SOME-MODEL:latest"] {
            assert!(
                media_facts(hf, id).map(|f| f.kind)
                    == Some(MediaKind::Transcription(
                        crate::inference::media_facts::TranscriptionModel::Whisper
                    )),
                "{id} did not resolve to the repository"
            );
        }
        assert_eq!(media_facts(hf, "unknown-model"), None);
    }
}
