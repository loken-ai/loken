//! The `clip.vision.*` metadata a llama.cpp `mmproj` GGUF carries about its vision tower.

use crate::tensor::quantized::gguf_file::{Content, Value};
use crate::tensor::{Error, Result};

/// A vision tower's shape and its image normalisation, as the `mmproj` states them.
#[derive(Debug, Clone)]
pub struct MmprojVision {
    pub width: usize,
    pub heads: usize,
    pub blocks: usize,
    pub feed_forward: usize,
    /// Pixels on a side of one patch, and of the square image the tower was trained on.
    pub patch: usize,
    pub image_size: usize,
    /// Width of the projector's output: the language model's embedding width.
    pub projection: usize,
    pub eps: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl MmprojVision {
    pub fn read(content: &Content) -> Result<Self> {
        let get = |key: &str| -> Result<&Value> {
            content
                .metadata
                .get(key)
                .ok_or_else(|| Error(format!("mmproj: missing {key}")))
        };
        let n = |key: &str| -> Result<usize> { Ok(get(key)?.to_u32()? as usize) };
        let rgb = |key: &str| -> Result<[f32; 3]> {
            let v = get(key)?
                .to_vec()?
                .iter()
                .map(|x| x.to_f32())
                .collect::<Result<Vec<f32>>>()?;
            v.try_into()
                .map_err(|v: Vec<f32>| Error(format!("mmproj: {key} has {} channels", v.len())))
        };
        Ok(Self {
            width: n("clip.vision.embedding_length")?,
            heads: n("clip.vision.attention.head_count")?,
            blocks: n("clip.vision.block_count")?,
            feed_forward: n("clip.vision.feed_forward_length")?,
            patch: n("clip.vision.patch_size")?,
            image_size: n("clip.vision.image_size")?,
            projection: n("clip.vision.projection_dim")?,
            eps: get("clip.vision.attention.layer_norm_epsilon")?.to_f32()?,
            mean: rgb("clip.vision.image_mean")?,
            std: rgb("clip.vision.image_std")?,
        })
    }

    /// Patches on a side of the training image.
    pub fn grid(&self) -> usize {
        self.image_size / self.patch.max(1)
    }

    pub fn head_dim(&self) -> usize {
        self.width / self.heads.max(1)
    }
}

/// An audio tower's shape, as the `mmproj` states it: a speech encoder and the projector
/// that maps its frames into the language model's embedding space.
#[derive(Debug, Clone)]
pub struct MmprojAudio {
    /// The `clip.projector_type`: which projector follows the encoder.
    pub projector_type: String,
    pub width: usize,
    pub heads: usize,
    pub blocks: usize,
    pub feed_forward: usize,
    /// Mel bands the encoder's first convolution reads.
    pub n_mels: usize,
    /// Consecutive encoder frames the projector stacks into one input row.
    pub stack: usize,
    /// Width of the projector's output: the language model's embedding width.
    pub projection: usize,
    pub eps: f32,
}

impl MmprojAudio {
    pub fn read(content: &Content) -> Result<Self> {
        let get = |key: &str| -> Result<&Value> {
            content
                .metadata
                .get(key)
                .ok_or_else(|| Error(format!("mmproj: missing {key}")))
        };
        let n = |key: &str| -> Result<usize> { Ok(get(key)?.to_u32()? as usize) };
        let has_audio = content
            .metadata
            .get("clip.has_audio_encoder")
            .and_then(|v| v.to_bool().ok())
            .unwrap_or(false);
        if !has_audio {
            return Err(Error(
                "mmproj: the file carries no audio encoder".to_string(),
            ));
        }
        Ok(Self {
            projector_type: get("clip.projector_type")?.to_string()?.clone(),
            width: n("clip.audio.embedding_length")?,
            heads: n("clip.audio.attention.head_count")?,
            blocks: n("clip.audio.block_count")?,
            feed_forward: n("clip.audio.feed_forward_length")?,
            n_mels: n("clip.audio.num_mel_bins")?,
            stack: n("clip.audio.projector.stack_factor")?,
            projection: n("clip.audio.projection_dim")?,
            eps: get("clip.audio.attention.layer_norm_epsilon")?.to_f32()?,
        })
    }

    pub fn head_dim(&self) -> usize {
        self.width / self.heads.max(1)
    }
}
