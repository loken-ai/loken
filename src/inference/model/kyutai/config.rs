//! The `config.json` published beside a Kyutai checkpoint, which states the shape of every
//! stage the checkpoint holds.

use crate::tensor::{Error, Result};

pub struct KyutaiConfig(serde_json::Value);

impl KyutaiConfig {
    /// Read the `config.json` in the directory of `checkpoint`.
    pub fn read(checkpoint: &str) -> Result<Self> {
        let dir = std::path::Path::new(checkpoint)
            .parent()
            .ok_or_else(|| Error(format!("kyutai: no directory for {checkpoint}")))?;
        let path = dir.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| Error(format!("{}: {e}", path.display())))?;
        let cfg =
            serde_json::from_str(&raw).map_err(|e| Error(format!("{}: {e}", path.display())))?;
        Ok(Self(cfg))
    }

    /// The value at `path`, one key per level.
    fn at(&self, path: &[&str]) -> &serde_json::Value {
        path.iter().fold(&self.0, |v, k| &v[*k])
    }

    fn missing(path: &[&str]) -> Error {
        Error(format!("kyutai config.json: missing {}", path.join(".")))
    }

    pub fn usize(&self, path: &[&str]) -> Result<usize> {
        self.at(path)
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| Self::missing(path))
    }

    /// An integer that may be absent or null.
    pub fn opt_usize(&self, path: &[&str]) -> Option<usize> {
        self.at(path).as_u64().map(|v| v as usize)
    }

    pub fn f64(&self, path: &[&str]) -> Result<f64> {
        self.at(path).as_f64().ok_or_else(|| Self::missing(path))
    }

    pub fn usizes(&self, path: &[&str]) -> Result<Vec<usize>> {
        self.at(path)
            .as_array()
            .and_then(|a| a.iter().map(|v| v.as_u64().map(|v| v as usize)).collect())
            .ok_or_else(|| Self::missing(path))
    }

    pub fn strings(&self, path: &[&str]) -> Result<Vec<String>> {
        self.at(path)
            .as_array()
            .and_then(|a| a.iter().map(|v| v.as_str().map(str::to_string)).collect())
            .ok_or_else(|| Self::missing(path))
    }

    /// The width of one half of a SiLU-GLU gate: two thirds of the feed-forward width it
    /// stands for.
    pub fn gate_width(feed_forward: usize) -> usize {
        2 * feed_forward / 3
    }
}

/// How the text stream and the audio codebooks are laid out frame by frame.
#[derive(Debug, Clone)]
pub struct Streams {
    /// The text vocabulary. Its size is the token a text stream starts from, and one more is
    /// the modulus two multiplexed text streams share.
    pub text_card: usize,
    /// One codebook's size, which is also the token an audio stream starts from.
    pub audio_card: usize,
    /// Audio codebooks per frame.
    pub audio_codebooks: usize,
    /// Frames each stream lags the first by: the text stream, then every codebook.
    pub delays: Vec<usize>,
    /// The text token that fills a frame with no word.
    pub text_padding: i64,
    /// Words the second text stream runs ahead of the first.
    pub second_stream_ahead: usize,
    /// Seconds of speech the text leads the audio by.
    pub audio_delay: f64,
}

impl Streams {
    pub fn from_config(cfg: &KyutaiConfig) -> Result<Self> {
        let audio_codebooks = cfg.usize(&["dep_q"])?;
        let delays = cfg.usizes(&["delays"])?;
        if delays.len() != 1 + audio_codebooks {
            return Err(Error(format!(
                "kyutai config.json: {} delays for 1 text and {audio_codebooks} audio streams",
                delays.len()
            )));
        }
        Ok(Self {
            text_card: cfg.usize(&["text_card"])?,
            audio_card: cfg.usize(&["card"])?,
            audio_codebooks,
            delays,
            text_padding: cfg.usize(&["existing_text_padding_id"])? as i64,
            second_stream_ahead: cfg.usize(&["tts_config", "second_stream_ahead"])?,
            audio_delay: cfg.f64(&["tts_config", "audio_delay"])?,
        })
    }

    /// Streams per frame: the text, then every codebook.
    pub fn per_frame(&self) -> usize {
        1 + self.audio_codebooks
    }
}
