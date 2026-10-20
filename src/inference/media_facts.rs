//! What a media checkpoint is, read from its own files.
//!
//! A language model declares its architecture in `general.architecture`. A media checkpoint
//! declares itself in the files it ships with: a diffusers repository names its pipeline and
//! its denoiser (`model_index.json`, `<component>/config.json`, key `_class_name`), a
//! transformers repository its `model_type`, a GGUF its `general.architecture`, and a bare
//! safetensors file the names of its tensors. This module reads those files and nothing else.
//!
//! The identifiers the files carry are matched here and only here: the tables below map a
//! declared class, model type or architecture to what the server does with it. Everywhere
//! else a checkpoint is a [`MediaFacts`], never a name.

use std::path::{Path, PathBuf};

/// The image denoisers the server runs, one per loader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImageFamily {
    Flux,
    Flux2,
    QwenImage,
    ZImage,
    Sdxl,
    Boogu,
}

impl ImageFamily {
    pub const ALL: [ImageFamily; 6] = [
        ImageFamily::Sdxl,
        ImageFamily::Boogu,
        ImageFamily::QwenImage,
        ImageFamily::ZImage,
        ImageFamily::Flux2,
        ImageFamily::Flux,
    ];

    /// The family tag the render routes and the engine report.
    pub fn tag(self) -> &'static str {
        match self {
            ImageFamily::Flux => "flux",
            ImageFamily::Flux2 => "flux2",
            ImageFamily::QwenImage => "qwen-image",
            ImageFamily::ZImage => "zimage",
            ImageFamily::Sdxl => "sdxl",
            ImageFamily::Boogu => "boogu",
        }
    }

    /// The family `/api/tags` publishes in `details.family`.
    pub fn listed(self) -> &'static str {
        match self {
            ImageFamily::ZImage => "z-image",
            other => other.tag(),
        }
    }

    /// The family a tag written by [`ImageFamily::tag`] stands for.
    pub fn from_tag(tag: &str) -> Option<ImageFamily> {
        ImageFamily::ALL.into_iter().find(|f| f.tag() == tag)
    }
}

/// How a sound generator is described by its files.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SoundKind {
    /// A conditioned diffusion model whose configuration declares the generator.
    ConditionedDiffusion,
    /// A latent diffusion generator whose files declare only its audio autoencoder.
    LatentDiffusion,
}

/// A speech synthesiser, by the graph its files describe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeechModel {
    /// A text-described voice decoder over audio codebooks (`parler_tts`).
    Parler,
    /// A delayed-streams text-to-speech transformer (`tts`).
    Kyutai,
    /// A flow language model over a neural audio codec.
    PocketTts,
    /// A VITS voice driven by a phoneme table.
    Piper,
    /// A speech model this build runs no engine for: a cloning synthesiser, a converter.
    Unserved,
}

/// A speech recogniser, by the format its weights ship in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TranscriptionModel {
    /// Encoder-decoder weights in safetensors, the format the engine loads.
    Whisper,
    /// A CTranslate2 conversion, which this build does not load.
    Converted,
}

/// What a media checkpoint produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image(ImageFamily),
    Video,
    Transcription(TranscriptionModel),
    Speech(SpeechModel),
    Sound(SoundKind),
    Music,
    /// Symbolic music: MIDI events.
    Midi,
    /// A time series forecaster: a series in, quantiles of its continuation out.
    Forecast,
    /// A part another pipeline loads: a text or image encoder, an autoencoder, a tokenizer.
    Component,
}

impl MediaKind {
    /// The family `/api/tags` publishes for this kind, `None` for a component, whose family is
    /// the `model_type` its configuration declares.
    pub fn listed(self) -> Option<&'static str> {
        Some(match self {
            MediaKind::Image(f) => f.listed(),
            MediaKind::Video => "video",
            MediaKind::Transcription(_) => "whisper",
            MediaKind::Speech(_) => "tts",
            MediaKind::Sound(_) => "sound",
            MediaKind::Music => "music",
            MediaKind::Midi => "midi",
            MediaKind::Forecast => "forecast",
            MediaKind::Component => return None,
        })
    }
}

/// What a media checkpoint's files say about it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MediaFacts {
    pub kind: MediaKind,
    /// The denoiser embeds its guidance scale (a `guidance_in` block): a guidance-distilled
    /// FLUX.1, sampled over many steps with the scale fed to the network.
    pub guidance_embedded: bool,
    /// The declared pipeline edits an image by instruction.
    pub instruction_edit: bool,
    /// The checkpoint declares itself distilled to a few unguided steps.
    pub distilled: bool,
    /// The sigma shift the checkpoint's scheduler fixes for every resolution.
    pub fixed_shift: Option<f32>,
}

impl MediaFacts {
    fn of(kind: MediaKind) -> Self {
        Self {
            kind,
            guidance_embedded: false,
            instruction_edit: false,
            distilled: false,
            fixed_shift: None,
        }
    }

    /// The image family, when the checkpoint is an image denoiser.
    pub fn image_family(&self) -> Option<ImageFamily> {
        match self.kind {
            MediaKind::Image(f) => Some(f),
            _ => None,
        }
    }
}

/// A diffusers `_class_name`: a pipeline or the denoiser class it loads.
fn diffusers_class(class: &str) -> Option<MediaFacts> {
    use ImageFamily::*;
    let image = |f| MediaFacts::of(MediaKind::Image(f));
    let editor = |f| MediaFacts {
        instruction_edit: true,
        ..image(f)
    };
    Some(match class {
        "FluxPipeline"
        | "FluxImg2ImgPipeline"
        | "FluxInpaintPipeline"
        | "FluxTransformer2DModel" => image(Flux),
        "FluxKontextPipeline" | "FluxKontextInpaintPipeline" => editor(Flux),
        "Flux2Pipeline" | "Flux2KleinPipeline" | "Flux2Transformer2DModel" => image(Flux2),
        "QwenImagePipeline" | "QwenImageImg2ImgPipeline" | "QwenImageTransformer2DModel" => {
            image(QwenImage)
        }
        "QwenImageEditPipeline" | "QwenImageEditPlusPipeline" => editor(QwenImage),
        "ZImagePipeline" | "ZImageTransformer2DModel" => image(ZImage),
        "StableDiffusionXLPipeline" | "StableDiffusionXLImg2ImgPipeline" => image(Sdxl),
        "WanModel" | "WanPipeline" | "WanImageToVideoPipeline" | "WanTransformer3DModel" => {
            MediaFacts::of(MediaKind::Video)
        }
        "StableAudioPipeline" | "StableAudioDiTModel" => {
            MediaFacts::of(MediaKind::Sound(SoundKind::ConditionedDiffusion))
        }
        _ => return None,
    })
}

/// A transformers `model_type`.
fn transformers_model_type(model_type: &str) -> Option<MediaKind> {
    Some(match model_type {
        "whisper" => MediaKind::Transcription(TranscriptionModel::Whisper),
        "parler_tts" => MediaKind::Speech(SpeechModel::Parler),
        "tts" => MediaKind::Speech(SpeechModel::Kyutai),
        "clip"
        | "clip_text_model"
        | "clip_vision_model"
        | "siglip"
        | "siglip_vision_model"
        | "t5"
        | "mt5"
        | "umt5" => MediaKind::Component,
        _ => return None,
    })
}

/// A transformers `architectures` class that names a time series forecaster whose encoder
/// this server runs.
fn forecaster_class(class: &str) -> bool {
    matches!(class, "Chronos2Model")
}

/// A GGUF `general.architecture`.
fn gguf_architecture(arch: &str) -> Option<MediaKind> {
    Some(match arch {
        "flux" => MediaKind::Image(ImageFamily::Flux),
        "qwen_image" => MediaKind::Image(ImageFamily::QwenImage),
        "wan" => MediaKind::Video,
        "acestep-dit" => MediaKind::Music,
        "acestep-lm" | "acestep-vae" => MediaKind::Component,
        _ => return None,
    })
}

/// A stable-audio-tools `model_type`: the conditioned generator of an audio pipeline, or
/// the autoencoder of one whose generator its files do not describe.
fn audio_tools_model_type(model_type: &str) -> Option<SoundKind> {
    Some(match model_type {
        "diffusion_cond" | "diffusion_uncond" => SoundKind::ConditionedDiffusion,
        "autoencoder" => SoundKind::LatentDiffusion,
        _ => return None,
    })
}

/// A coqui-TTS `model`.
fn coqui_model(model: &str) -> bool {
    matches!(model, "xtts" | "tortoise" | "bark" | "vits")
}

/// The kind a set of tensor names declares, `joined` being every name separated by a newline.
///
/// Ordered most specific first: the Boogu DiT is a Flux-shaped three-stream MMDiT, FLUX.2 is
/// named like Qwen-Image, and the SDXL and the Z-Image single-file packs share a prefix.
fn tensor_signature(joined: &str) -> Option<MediaFacts> {
    let has = |s: &str| joined.contains(s);
    let image = |f| MediaFacts::of(MediaKind::Image(f));
    Some(if has("double_stream_layers") && has("instruct") {
        image(ImageFamily::Boogu)
    } else if has("double_blocks") && has("single_blocks") {
        MediaFacts {
            guidance_embedded: has("guidance_in."),
            ..image(ImageFamily::Flux)
        }
    } else if has("single_transformer_blocks") && has("x_embedder") {
        image(ImageFamily::Flux2)
    } else if has("transformer_blocks") && has("img_in") {
        image(ImageFamily::QwenImage)
    } else if has("model.diffusion_model.input_blocks") || has("conditioner.embedders") {
        image(ImageFamily::Sdxl)
    } else if (has("time_text_embed") && has("norm_out.linear"))
        || (has("cap_embedder") && has("context_refiner"))
    {
        image(ImageFamily::ZImage)
    } else if has("patch_embedding") && has("text_embedding") && has("time_embedding") {
        MediaFacts::of(MediaKind::Video)
    } else if has("self_attn.q") && has("cross_attn.k") && has("lora_down") {
        // An adapter of the video denoiser: its blocks' attention, without the embeddings.
        MediaFacts::of(MediaKind::Video)
    } else if has("flow_lm.") && has("mimi.") {
        MediaFacts::of(MediaKind::Speech(SpeechModel::PocketTts))
    } else if has("encoder.conv_in") && has("decoder.conv_in") {
        MediaFacts::of(MediaKind::Component)
    } else {
        return None;
    })
}

/// The tensor names of a safetensors file, newline-joined, and its dtypes. `None` for a file
/// that is not safetensors.
fn safetensors_names(path: &Path) -> Option<String> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).ok()?;
    let mut len8 = [0u8; 8];
    f.read_exact(&mut len8).ok()?;
    let hlen = u64::from_le_bytes(len8);
    // A header length past the file is not a safetensors header (a GGUF magic reads as one).
    let flen = f.metadata().ok()?.len();
    if hlen == 0 || hlen + 8 > flen {
        return None;
    }
    let mut hbuf = vec![0u8; hlen as usize];
    f.read_exact(&mut hbuf).ok()?;
    let header: serde_json::Map<String, serde_json::Value> = serde_json::from_slice(&hbuf).ok()?;
    Some(
        header
            .keys()
            .filter(|k| *k != "__metadata__")
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn from_safetensors(path: &Path) -> Option<MediaFacts> {
    tensor_signature(&safetensors_names(path)?)
}

fn from_gguf(path: &Path) -> Option<MediaFacts> {
    let content = crate::tensor::quantized::gguf_file::open_header(path).ok()?;
    let meta = |k: &str| content.metadata.get(k);
    let names = content
        .tensor_infos
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join("\n");
    let by_names = tensor_signature(&names);
    let declared = meta("general.architecture")
        .and_then(|v| v.to_string().ok())
        .map(String::as_str);
    let mut facts = match declared {
        Some(arch) => match gguf_architecture(arch) {
            Some(kind) => MediaFacts::of(kind),
            // A language model whose vocabulary carries MIDI event tokens writes music as
            // symbolic events.
            None if vocabulary_has_prefix(&content, MIDI_EVENT_TOKEN) => {
                MediaFacts::of(MediaKind::Midi)
            }
            None => return None,
        },
        None => by_names?,
    };
    if let Some(named) = by_names {
        facts.guidance_embedded = named.guidance_embedded;
    }
    // Written under the architecture's own key prefix by the converters that record it.
    facts.distilled = content.metadata.iter().any(|(k, v)| {
        (k.ends_with(".is_turbo") || k.ends_with(".is_distilled")) && v.to_bool().unwrap_or(false)
    });
    Some(facts)
}

/// The vocabulary piece that opens the MIDI event tokens of a symbolic-music language model.
const MIDI_EVENT_TOKEN: &str = "<|midi_token_";

/// Whether a GGUF's vocabulary holds a piece starting with `prefix`.
fn vocabulary_has_prefix(
    content: &crate::tensor::quantized::gguf_file::Content,
    prefix: &str,
) -> bool {
    content
        .metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.to_vec().ok())
        .is_some_and(|tokens| {
            tokens
                .iter()
                .any(|t| t.to_string().is_ok_and(|s| s.starts_with(prefix)))
        })
}

/// A JSON file, including one written by Python's encoder, which spells unbounded values
/// `Infinity` and `NaN`; those read as null.
fn read_json(path: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok().or_else(|| {
        let tolerant = text
            .replace("-Infinity", "null")
            .replace("Infinity", "null")
            .replace("NaN", "null");
        serde_json::from_str(&tolerant).ok()
    })
}

/// What a configuration file declares, whichever library wrote it.
fn from_config(config: &serde_json::Value) -> Option<MediaFacts> {
    let text = |k: &str| config.get(k).and_then(|v| v.as_str());
    if let Some(mut facts) = text("_class_name").and_then(diffusers_class) {
        facts.distilled = config
            .get("is_distilled")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        return Some(facts);
    }
    // A forecaster declares the transformers `model_type` of the encoder it was built from,
    // so its class is read first.
    if config
        .pointer("/architectures/0")
        .and_then(|v| v.as_str())
        .is_some_and(forecaster_class)
    {
        return Some(MediaFacts::of(MediaKind::Forecast));
    }
    // A patched forecaster that names no class: its layer schedule, its quantile levels and
    // its patch length.
    if ["block_schedule", "quantile_knots", "patch_size"]
        .iter()
        .all(|k| config.get(*k).is_some())
    {
        return Some(MediaFacts::of(MediaKind::Forecast));
    }
    if let Some(kind) = text("model_type").and_then(transformers_model_type) {
        return Some(MediaFacts::of(kind));
    }
    if let Some(sound) = text("model_type").and_then(audio_tools_model_type) {
        if config.get("sample_rate").is_some() {
            return Some(MediaFacts::of(MediaKind::Sound(sound)));
        }
    }
    if text("model").is_some_and(coqui_model) {
        return Some(MediaFacts::of(MediaKind::Speech(SpeechModel::Unserved)));
    }
    // A Piper voice: the phoneme table its ONNX graph is driven by.
    if config.get("phoneme_id_map").is_some() && config.get("espeak").is_some() {
        return Some(MediaFacts::of(MediaKind::Speech(SpeechModel::Piper)));
    }
    // A CTranslate2 conversion of a speech recogniser: the alignment heads and language
    // tokens its decoder is driven by.
    if config.get("alignment_heads").is_some() && config.get("lang_ids").is_some() {
        return Some(MediaFacts::of(MediaKind::Transcription(
            TranscriptionModel::Converted,
        )));
    }
    // A VITS voice converter: a waveform decoder configured by its sampling rate.
    let model = config.get("model");
    if config.pointer("/data/sampling_rate").is_some()
        && model.and_then(|m| m.get("resblock_kernel_sizes")).is_some()
    {
        return Some(MediaFacts::of(MediaKind::Speech(SpeechModel::Unserved)));
    }
    None
}

/// The sigma shift a diffusers scheduler configuration fixes, when it fixes one.
fn fixed_shift(config: &serde_json::Value) -> Option<f32> {
    let dynamic = config
        .get("use_dynamic_shifting")
        .and_then(|v| v.as_bool())?;
    if dynamic {
        return None;
    }
    config
        .get("shift")
        .and_then(|v| v.as_f64())
        .map(|s| s as f32)
}

/// The configuration files of `dir` a pipeline is described by, most authoritative first.
const CONFIGS: [&str; 3] = ["model_index.json", "config.json", "model_config.json"];

/// What the configuration files directly in `dir` declare.
fn configs_in(dir: &Path) -> Option<MediaFacts> {
    CONFIGS
        .iter()
        .filter_map(|c| read_json(&dir.join(c)))
        .find_map(|c| from_config(&c))
        .or_else(|| {
            std::fs::read_dir(dir)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.to_string_lossy().ends_with(".onnx.json"))
                .find_map(|p| from_config(&read_json(&p)?))
        })
}

/// The sub-directories of `dir`, sorted so a scan reads them in a stable order.
fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_dir()
                && !p
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        })
        .collect();
    out.sort();
    out
}

/// The weight files directly in `dir`, largest first.
fn weights_in(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<(u64, PathBuf)> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension().is_some_and(|x| {
                    x.eq_ignore_ascii_case("safetensors") || x.eq_ignore_ascii_case("gguf")
                })
        })
        .map(|p| (std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0), p))
        .collect();
    out.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    out.into_iter().map(|(_, p)| p).collect()
}

/// What the weights directly in `dir` declare: the first GGUF naming a media architecture,
/// else the tensor names of every safetensors shard together.
fn weights_declare(dir: &Path) -> Option<MediaFacts> {
    let files = weights_in(dir);
    let is = |p: &PathBuf, ext: &str| p.extension().is_some_and(|x| x.eq_ignore_ascii_case(ext));
    if let Some(f) = files
        .iter()
        .filter(|p| is(p, "gguf"))
        .find_map(|p| from_gguf(p))
    {
        return Some(f);
    }
    let names: Vec<String> = files
        .iter()
        .filter(|p| is(p, "safetensors"))
        .filter_map(|p| safetensors_names(p))
        .collect();
    tensor_signature(&names.join("\n"))
}

/// What a directory holding a pipeline declares: its configuration first, then its weights,
/// then the configuration and weights of its components.
fn from_dir(dir: &Path) -> Option<MediaFacts> {
    let mut facts = configs_in(dir);
    // A repository whose configuration declares another kind of model is that model.
    let declared = read_json(&dir.join("config.json"))
        .is_some_and(|c| c.get("architectures").is_some() || c.get("model_type").is_some());
    if facts.is_none() && declared {
        return None;
    }
    facts = facts.or_else(|| weights_declare(dir));
    if facts.is_none() {
        // A pipeline laid out by component: the denoiser's directory says what it is.
        let parts = subdirs(dir);
        facts = parts
            .iter()
            .find_map(|d| configs_in(d).filter(|f| f.kind != MediaKind::Component))
            .or_else(|| parts.iter().find_map(|d| weights_declare(d)))
            .or_else(|| {
                parts
                    .iter()
                    .flat_map(|d| subdirs(d))
                    .find_map(|d| configs_in(&d).or_else(|| weights_declare(&d)))
            })
            .or_else(|| parts.iter().find_map(|d| configs_in(d)));
    }
    let mut facts = facts?;
    if let Some(shift) = scheduler_shift(dir) {
        facts.fixed_shift = Some(shift);
    }
    Some(facts)
}

/// The fixed shift a scheduler configuration in `dir` or its `scheduler/` declares.
fn scheduler_shift(dir: &Path) -> Option<f32> {
    [
        dir.join("scheduler_config.json"),
        dir.join("scheduler").join("scheduler_config.json"),
    ]
    .iter()
    .find_map(|p| fixed_shift(&read_json(p)?))
}

/// What a single weights file declares, completed by the pipeline files around it.
fn from_file(path: &Path) -> Option<MediaFacts> {
    let ext = path
        .extension()
        .and_then(|x| x.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let own = match ext.as_str() {
        "gguf" => from_gguf(path),
        "safetensors" => from_safetensors(path),
        "onnx" => read_json(&PathBuf::from(format!("{}.json", path.display())))
            .and_then(|c| from_config(&c)),
        _ => None,
    };
    let dir = path.parent()?;
    let up = dir.parent();
    let Some(mut facts) = own else {
        // A container this reader cannot open (a pickle): the pipeline it sits in says what
        // it is, from the configuration beside it or one level up.
        return [Some(dir), up]
            .into_iter()
            .flatten()
            .find_map(|d| configs_in(d).or_else(|| subdirs(d).iter().find_map(|s| configs_in(s))))
            .filter(|f| f.kind != MediaKind::Component);
    };
    // The pipeline index beside a component, or one level up from it, declares what the
    // component's own header cannot: an instruction editor, a distillation.
    let index = [Some(dir), up]
        .into_iter()
        .flatten()
        .find_map(|d| read_json(&d.join("model_index.json")));
    if let Some(declared) = index.as_ref().and_then(from_config) {
        if declared.kind == facts.kind {
            facts.instruction_edit |= declared.instruction_edit;
            facts.distilled |= declared.distilled;
        }
    }
    if let Some(shift) = [Some(dir), up]
        .into_iter()
        .flatten()
        .find_map(scheduler_shift)
    {
        facts.fixed_shift = Some(shift);
    }
    Some(facts)
}

/// The facts of files already read, with the size and time they were read at.
type Seen =
    std::collections::HashMap<PathBuf, (u64, Option<std::time::SystemTime>, Option<MediaFacts>)>;

fn seen() -> &'static std::sync::Mutex<Seen> {
    static SEEN: std::sync::OnceLock<std::sync::Mutex<Seen>> = std::sync::OnceLock::new();
    SEEN.get_or_init(Default::default)
}

/// The denoiser a music render loads from a checkpoint directory that holds several: among
/// the GGUF files whose own header declares a music denoiser, the one that declares itself
/// distilled, then the smallest.
pub fn music_denoiser(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<(bool, u64, PathBuf)> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "gguf"))
        .filter_map(|p| {
            let facts = read(&p).filter(|f| f.kind == MediaKind::Music)?;
            let size = std::fs::metadata(&p).ok()?.len();
            Some((!facts.distilled, size, p))
        })
        .collect();
    found.sort();
    found.into_iter().next().map(|(_, _, p)| p)
}

/// What the checkpoint at `path` is: a weights file, or a directory holding a pipeline.
/// `None` when its files declare no media model this server knows.
///
/// A path is read once per size and modification time: a weights file does not change in
/// place, and a directory changes its time when a file is added to it.
pub fn read(path: &Path) -> Option<MediaFacts> {
    let meta = std::fs::metadata(path).ok()?;
    let stamp = (meta.len(), meta.modified().ok());
    let mut cache = seen().lock().unwrap_or_else(|e| e.into_inner());
    if let Some((len, time, facts)) = cache.get(path) {
        if (*len, *time) == stamp {
            return *facts;
        }
    }
    drop(cache);
    let facts = if meta.is_dir() {
        from_dir(path)
    } else {
        from_file(path)
    };
    cache = seen().lock().unwrap_or_else(|e| e.into_inner());
    cache.insert(path.to_path_buf(), (stamp.0, stamp.1, facts));
    facts
}

#[cfg(test)]
pub(crate) mod fixture {
    //! Checkpoints that carry only what the readers above look at: headers, no weights.

    use std::path::Path;

    /// A safetensors file whose header lists `names`, each a one-element F32 tensor.
    pub fn safetensors(path: &Path, names: &[&str], metadata: Option<serde_json::Value>) {
        let mut header = serde_json::Map::new();
        for (i, n) in names.iter().enumerate() {
            header.insert(
                n.to_string(),
                serde_json::json!({"dtype": "F32", "shape": [1], "data_offsets": [i * 4, i * 4 + 4]}),
            );
        }
        if let Some(m) = metadata {
            header.insert("__metadata__".into(), m);
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut bytes = (h.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&h);
        bytes.extend(std::iter::repeat_n(0u8, names.len() * 4));
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, bytes).unwrap();
    }

    /// A GGUF file carrying `metadata` and one-element F32 tensors named `names`.
    pub fn gguf(
        path: &Path,
        metadata: &[(&str, crate::tensor::quantized::gguf_file::Value)],
        names: &[&str],
    ) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let entries: Vec<crate::tensor::gguf_write::GgufEntry> = names
            .iter()
            .map(|n| crate::tensor::gguf_write::GgufEntry {
                name: n.to_string(),
                dims: vec![1],
                dtype: crate::tensor::quantized::GgmlDType::F32,
                data: vec![0; 4],
            })
            .collect();
        let meta: Vec<(String, crate::tensor::quantized::gguf_file::Value)> = metadata
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect();
        crate::tensor::gguf_write::write_gguf_with_metadata(path, &meta, &entries).unwrap();
    }

    /// A JSON file.
    pub fn json(path: &Path, value: serde_json::Value) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, value.to_string()).unwrap();
    }

    /// A scratch directory removed when dropped.
    pub struct Scratch(pub std::path::PathBuf);

    impl Scratch {
        pub fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "loken-media-{tag}-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Scratch(p)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// FLUX.1 tensor names, with or without the guidance embedding.
    pub fn flux(guidance: bool) -> Vec<&'static str> {
        let mut v = vec![
            "double_blocks.0.img_attn.qkv.weight",
            "single_blocks.0.linear1.weight",
        ];
        if guidance {
            v.push("guidance_in.in_layer.weight");
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::*;
    use super::*;
    use crate::tensor::quantized::gguf_file::Value;

    #[test]
    fn a_denoiser_is_known_by_its_tensors() {
        let s = Scratch::new("tensors");
        let cases: &[(&[&str], ImageFamily)] = &[
            (
                &["double_stream_layers.0.x", "instruct_proj.w"],
                ImageFamily::Boogu,
            ),
            (
                &["double_blocks.0.a", "single_blocks.0.b"],
                ImageFamily::Flux,
            ),
            (
                &["single_transformer_blocks.0.a", "x_embedder.w"],
                ImageFamily::Flux2,
            ),
            (
                &["transformer_blocks.0.a", "img_in.w"],
                ImageFamily::QwenImage,
            ),
            (
                &["model.diffusion_model.input_blocks.0.w"],
                ImageFamily::Sdxl,
            ),
            (&["conditioner.embedders.0.w"], ImageFamily::Sdxl),
            (
                &["cap_embedder.0.w", "context_refiner.0.w"],
                ImageFamily::ZImage,
            ),
        ];
        for (i, (names, fam)) in cases.iter().enumerate() {
            let p = s.0.join(format!("m{i}.safetensors"));
            safetensors(&p, names, None);
            assert_eq!(
                read(&p).and_then(|f| f.image_family()),
                Some(*fam),
                "{names:?}"
            );
        }
    }

    #[test]
    fn a_flux_denoiser_says_whether_it_embeds_its_guidance() {
        let s = Scratch::new("guidance");
        let dev = s.0.join("dev.safetensors");
        let schnell = s.0.join("schnell.gguf");
        safetensors(&dev, &flux(true), None);
        gguf(&schnell, &[], &flux(false));
        assert!(read(&dev).unwrap().guidance_embedded);
        let f = read(&schnell).unwrap();
        assert_eq!(f.image_family(), Some(ImageFamily::Flux));
        assert!(!f.guidance_embedded);
    }

    #[test]
    fn a_gguf_is_known_by_its_declared_architecture() {
        let s = Scratch::new("gguf");
        let music = s.0.join("m.gguf");
        gguf(
            &music,
            &[
                ("general.architecture", Value::String("acestep-dit".into())),
                ("acestep.is_turbo", Value::Bool(true)),
            ],
            &["decoder.w"],
        );
        let f = read(&music).unwrap();
        assert_eq!(f.kind, MediaKind::Music);
        assert!(f.distilled);
        let video = s.0.join("v.gguf");
        gguf(
            &video,
            &[("general.architecture", Value::String("wan".into()))],
            &["blocks.0.w"],
        );
        assert_eq!(read(&video).unwrap().kind, MediaKind::Video);
        let text = s.0.join("t.gguf");
        gguf(
            &text,
            &[("general.architecture", Value::String("llama".into()))],
            &["blk.0.w"],
        );
        assert_eq!(read(&text), None);
    }

    #[test]
    fn a_repository_is_known_by_its_configuration() {
        let s = Scratch::new("repos");
        let cases: &[(serde_json::Value, MediaKind)] = &[
            (
                serde_json::json!({"model_type": "whisper"}),
                MediaKind::Transcription(TranscriptionModel::Whisper),
            ),
            (
                serde_json::json!({"model_type": "parler_tts"}),
                MediaKind::Speech(SpeechModel::Parler),
            ),
            (
                serde_json::json!({"model_type": "tts"}),
                MediaKind::Speech(SpeechModel::Kyutai),
            ),
            (
                serde_json::json!({"phoneme_id_map": {}, "espeak": {}}),
                MediaKind::Speech(SpeechModel::Piper),
            ),
            (
                serde_json::json!({"model_type": "clip"}),
                MediaKind::Component,
            ),
            (
                serde_json::json!({"model_type": "t5"}),
                MediaKind::Component,
            ),
            (
                serde_json::json!({"model_type": "t5", "architectures": ["Chronos2Model"]}),
                MediaKind::Forecast,
            ),
            (
                serde_json::json!({"patch_size": 32, "block_schedule": ["time"], "quantile_knots": [0.5]}),
                MediaKind::Forecast,
            ),
            (
                serde_json::json!({"model": "xtts"}),
                MediaKind::Speech(SpeechModel::Unserved),
            ),
            (
                serde_json::json!({"alignment_heads": [], "lang_ids": []}),
                MediaKind::Transcription(TranscriptionModel::Converted),
            ),
            (
                serde_json::json!({"_class_name": "WanModel"}),
                MediaKind::Video,
            ),
            (
                serde_json::json!({"model_type": "diffusion_cond", "sample_rate": 44100}),
                MediaKind::Sound(SoundKind::ConditionedDiffusion),
            ),
        ];
        for (i, (config, kind)) in cases.iter().enumerate() {
            let d = s.0.join(format!("r{i}"));
            json(&d.join("config.json"), config.clone());
            assert_eq!(read(&d).map(|f| f.kind), Some(*kind), "{config}");
        }
        let text = s.0.join("text");
        json(
            &text.join("config.json"),
            serde_json::json!({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        );
        assert_eq!(read(&text), None);
    }

    #[test]
    fn a_configuration_written_by_python_is_read() {
        let s = Scratch::new("python-json");
        let d = s.0.join("repo");
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(
            d.join("config.json"),
            r#"{"model": "xtts", "max_audio_len": Infinity, "floor": -Infinity}"#,
        )
        .unwrap();
        assert_eq!(
            read(&d).map(|f| f.kind),
            Some(MediaKind::Speech(SpeechModel::Unserved))
        );
    }

    #[test]
    fn a_pipeline_index_declares_what_its_denoiser_cannot() {
        let s = Scratch::new("index");
        let d = s.0.join("klein");
        json(
            &d.join("model_index.json"),
            serde_json::json!({"_class_name": "Flux2KleinPipeline", "is_distilled": true}),
        );
        let dit = d
            .join("transformer")
            .join("diffusion_pytorch_model.safetensors");
        safetensors(
            &dit,
            &["single_transformer_blocks.0.a", "x_embedder.w"],
            None,
        );
        let f = read(&dit).unwrap();
        assert_eq!(f.image_family(), Some(ImageFamily::Flux2));
        assert!(f.distilled);
        assert!(read(&d).unwrap().distilled);

        let e = s.0.join("editor");
        json(
            &e.join("model_index.json"),
            serde_json::json!({"_class_name": "FluxKontextPipeline"}),
        );
        let file = e.join("dit.gguf");
        gguf(
            &file,
            &[("general.architecture", Value::String("flux".into()))],
            &flux(true),
        );
        assert!(read(&file).unwrap().instruction_edit);
    }

    #[test]
    fn a_scheduler_that_fixes_its_shift_is_read() {
        let s = Scratch::new("shift");
        let file = s.0.join("dit.safetensors");
        safetensors(&file, &["transformer_blocks.0.a", "img_in.w"], None);
        assert_eq!(read(&file).unwrap().fixed_shift, None);
        json(
            &s.0.join("scheduler_config.json"),
            serde_json::json!({"use_dynamic_shifting": false, "shift": 3.0}),
        );
        let other = s.0.join("dit2.safetensors");
        safetensors(&other, &["transformer_blocks.0.a", "img_in.w"], None);
        assert_eq!(read(&other).unwrap().fixed_shift, Some(3.0));
    }

    #[test]
    fn a_pipeline_laid_out_by_component_is_known_by_its_denoiser() {
        let s = Scratch::new("parts");
        let d = s.0.join("repo");
        safetensors(
            &d.join("transformer")
                .join("diffusion_pytorch_model-00001-of-00002.safetensors"),
            &["cap_embedder.0.w", "x_pad_token"],
            None,
        );
        safetensors(
            &d.join("transformer")
                .join("diffusion_pytorch_model-00002-of-00002.safetensors"),
            &["context_refiner.0.w", "layers.0.w"],
            None,
        );
        safetensors(
            &d.join("vae").join("diffusion_pytorch_model.safetensors"),
            &["decoder.w"],
            None,
        );
        assert_eq!(
            read(&d).and_then(|f| f.image_family()),
            Some(ImageFamily::ZImage)
        );
    }

    #[test]
    fn a_pickled_checkpoint_is_known_by_the_pipeline_around_it() {
        let s = Scratch::new("pickle");
        let d = s.0.join("snapshot");
        json(
            &d.join("config.json"),
            serde_json::json!({"_class_name": "WanModel"}),
        );
        let pth = d.join("vae.pth");
        std::fs::write(&pth, b"PK").unwrap();
        assert_eq!(read(&pth).map(|f| f.kind), Some(MediaKind::Video));
        let a = s.0.join("audio").join("ckpts");
        json(
            &a.join("vae").join("config.json"),
            serde_json::json!({"model_type": "autoencoder", "sample_rate": 24000}),
        );
        let pt = a.join("s3").join("model.pt");
        std::fs::create_dir_all(pt.parent().unwrap()).unwrap();
        std::fs::write(&pt, b"PK").unwrap();
        assert_eq!(
            read(&pt).map(|f| f.kind),
            Some(MediaKind::Sound(SoundKind::LatentDiffusion))
        );
    }
}
