//! The ids and the sampling values of a checkpoint, read from its `config.json` and its
//! `generation_config.json`.

use crate::tensor::{Error, Result};
use serde_json::Value as Json;

/// The ids the prompt and the frame loop use, and the sizes they derive from.
#[derive(Debug, Clone)]
pub struct SpeechIds {
    /// Text ids embedded through the text table and its projection.
    pub tts_bos: u32,
    pub tts_eos: u32,
    pub tts_pad: u32,
    /// Codec ids of the talker's table.
    pub codec_bos: u32,
    pub codec_eos: u32,
    pub codec_pad: u32,
    pub codec_think: u32,
    pub codec_nothink: u32,
    pub codec_think_bos: u32,
    pub codec_think_eos: u32,
    /// The codec id of each language the checkpoint names.
    pub languages: Vec<(String, u32)>,
    /// Rows of the talker's codec table and head.
    pub talker_vocab: usize,
    /// Rows of each code predictor table and head: the codebook of every group.
    pub codebook: usize,
    /// Codes of a frame: group 0 from the talker, the others from the code predictor.
    pub groups: usize,
    /// Positions the talker's rotary table covers.
    pub talker_positions: usize,
}

fn field<'a>(v: &'a Json, path: &[&str]) -> Result<&'a Json> {
    path.iter()
        .try_fold(v, |v, k| v.get(k))
        .filter(|v| !v.is_null())
        .ok_or_else(|| Error::msg(format!("config.json has no {}", path.join("."))))
}

fn id(v: &Json, path: &[&str]) -> Result<u32> {
    field(v, path)?
        .as_u64()
        .map(|x| x as u32)
        .ok_or_else(|| Error::msg(format!("config.json {} is not an id", path.join("."))))
}

fn size(v: &Json, path: &[&str]) -> Result<usize> {
    id(v, path).map(|x| x as usize)
}

impl SpeechIds {
    /// The ids of the configuration `root`.
    pub fn read(root: &Json) -> Result<Self> {
        let t = "talker_config";
        let p = "code_predictor_config";
        let languages = field(root, &[t, "codec_language_id"])?
            .as_object()
            .ok_or_else(|| Error::msg("config.json codec_language_id is not an object"))?
            .iter()
            .filter_map(|(name, v)| Some((name.clone(), v.as_u64()? as u32)))
            .collect();
        Ok(Self {
            tts_bos: id(root, &["tts_bos_token_id"])?,
            tts_eos: id(root, &["tts_eos_token_id"])?,
            tts_pad: id(root, &["tts_pad_token_id"])?,
            codec_bos: id(root, &[t, "codec_bos_id"])?,
            codec_eos: id(root, &[t, "codec_eos_token_id"])?,
            codec_pad: id(root, &[t, "codec_pad_id"])?,
            codec_think: id(root, &[t, "codec_think_id"])?,
            codec_nothink: id(root, &[t, "codec_nothink_id"])?,
            codec_think_bos: id(root, &[t, "codec_think_bos_id"])?,
            codec_think_eos: id(root, &[t, "codec_think_eos_id"])?,
            languages,
            talker_vocab: size(root, &[t, "vocab_size"])?,
            codebook: size(root, &[t, p, "vocab_size"])?,
            groups: size(root, &[t, "num_code_groups"])?,
            talker_positions: size(root, &[t, "max_position_embeddings"])?,
        })
    }

    /// The codec id of the language `name`, `None` for a language the prompt leaves to the
    /// model.
    pub fn language(&self, name: Option<&str>) -> Result<Option<u32>> {
        let Some(name) = name else { return Ok(None) };
        let wanted = name.to_lowercase();
        self.languages
            .iter()
            .find(|(n, _)| *n == wanted)
            .map(|(_, id)| Some(*id))
            .ok_or_else(|| {
                let names: Vec<&str> = self.languages.iter().map(|(n, _)| n.as_str()).collect();
                Error::msg(format!(
                    "the checkpoint speaks no language {name:?}: it names {}",
                    names.join(", ")
                ))
            })
    }

    /// The talker ids that can end no frame: every id past the codebook but the end of speech.
    pub fn suppressed(&self) -> Vec<u32> {
        (self.codebook as u32..self.talker_vocab as u32)
            .filter(|id| *id != self.codec_eos)
            .collect()
    }
}

/// A variant of the checkpoint this build serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// A voice described in words.
    VoiceDesign,
    /// A voice cloned from a clip: a speaker embedding, and a reference text and its codes.
    Base,
}

/// The names `tts_model_type` gives the variants this build serves.
pub const SERVED_VARIANTS: [(&str, Variant); 2] = [
    ("voice_design", Variant::VoiceDesign),
    ("base", Variant::Base),
];

/// The variant the configuration `root` declares in `tts_model_type`, refused unless this
/// build serves it.
pub fn served_variant(root: &Json) -> Result<Variant> {
    let variant = field(root, &["tts_model_type"])?
        .as_str()
        .ok_or_else(|| Error::msg("config.json tts_model_type is not a name"))?;
    SERVED_VARIANTS
        .iter()
        .find(|(name, _)| *name == variant)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let names: Vec<String> = SERVED_VARIANTS
                .iter()
                .map(|(n, _)| format!("{n:?}"))
                .collect();
            Error::msg(format!(
                "the checkpoint is of the variant {variant:?}: only {} are served",
                names.join(" and ")
            ))
        })
}

/// The sample rate the speaker encoder of the configuration `root` takes, `None` when the
/// configuration has no speaker encoder.
pub fn speaker_rate(root: &Json) -> Result<Option<usize>> {
    match root.get("speaker_encoder_config").filter(|v| !v.is_null()) {
        None => Ok(None),
        Some(_) => size(root, &["speaker_encoder_config", "sample_rate"]).map(Some),
    }
}

/// The width of a talker row of the configuration `root`, which a speaker embedding matches.
pub fn talker_width(root: &Json) -> Result<usize> {
    size(root, &["talker_config", "hidden_size"])
}

/// How a code is drawn from a row of logits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// Draw from the distribution, else take the most probable code.
    pub sample: bool,
    pub temperature: f32,
    /// The most probable codes kept, 0 for every code.
    pub top_k: usize,
    /// The smallest set of leading codes whose probabilities reach this share.
    pub top_p: f32,
}

/// The sampling values of `generation_config.json`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Generation {
    /// How the talker draws the code of group 0.
    pub talker: Sampling,
    /// Divides a positive logit of a group 0 code already drawn, multiplies a negative one.
    pub repetition_penalty: f32,
    /// How the code predictor draws the other groups.
    pub predictor: Sampling,
    /// Frames an utterance may hold at most.
    pub max_frames: usize,
}

impl Generation {
    /// The values of the generation configuration `v`.
    pub fn read(v: &Json) -> Result<Self> {
        let get = |k: &str| {
            v.get(k)
                .filter(|x| !x.is_null())
                .ok_or_else(|| Error::msg(format!("generation_config.json has no {k}")))
        };
        let float = |k: &str| -> Result<f32> {
            get(k)?
                .as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| Error::msg(format!("generation_config.json {k} is not a number")))
        };
        let count = |k: &str| -> Result<usize> {
            get(k)?
                .as_u64()
                .map(|x| x as usize)
                .ok_or_else(|| Error::msg(format!("generation_config.json {k} is not a count")))
        };
        let flag = |k: &str| -> Result<bool> {
            get(k)?
                .as_bool()
                .ok_or_else(|| Error::msg(format!("generation_config.json {k} is not a flag")))
        };
        Ok(Self {
            talker: Sampling {
                sample: flag("do_sample")?,
                temperature: float("temperature")?,
                top_k: count("top_k")?,
                top_p: float("top_p")?,
            },
            repetition_penalty: float("repetition_penalty")?,
            predictor: Sampling {
                sample: flag("subtalker_dosample")?,
                temperature: float("subtalker_temperature")?,
                top_k: count("subtalker_top_k")?,
                top_p: float("subtalker_top_p")?,
            },
            max_frames: count("max_new_tokens")?,
        })
    }
}
