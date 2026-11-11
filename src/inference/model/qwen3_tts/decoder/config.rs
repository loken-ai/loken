//! The geometry of the codec decoder, read from the codec's `config.json`, and what derives
//! from it: the left context of a chunk and the frames a memory reserve holds.

use crate::tensor::{Error, Result};
use serde_json::Value as Json;

/// The sizes, rates and epsilons of the codec decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct CodecConfig {
    /// Samples per second of the waveform.
    pub sample_rate: usize,
    /// Samples a frame of codes decodes to.
    pub frame_samples: usize,
    /// Codes of a frame.
    pub quantizers: usize,
    /// Entries of each codebook.
    pub codebook_size: usize,
    /// Codes of a frame read by the first quantizer.
    pub semantic_quantizers: usize,
    /// Width of the transformer's input and output, and of the upsample stages.
    pub latent: usize,
    /// Width of the transformer.
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub layers: usize,
    /// Positions a query of the transformer sees, itself included.
    pub window: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    /// Positions the rotary table of the transformer covers.
    pub max_positions: usize,
    /// Stride of each upsample stage after the transformer.
    pub upsampling_ratios: Vec<usize>,
    /// Width of the first conv of the waveform stack.
    pub decoder_dim: usize,
    /// Stride of each block of the waveform stack.
    pub upsample_rates: Vec<usize>,
}

fn field<'a>(v: &'a Json, key: &str) -> Result<&'a Json> {
    v.get(key)
        .filter(|v| !v.is_null())
        .ok_or_else(|| Error::msg(format!("the codec config.json has no {key}")))
}

fn size(v: &Json, key: &str) -> Result<usize> {
    field(v, key)?
        .as_u64()
        .map(|x| x as usize)
        .ok_or_else(|| Error::msg(format!("the codec config.json {key} is not a size")))
}

fn float(v: &Json, key: &str) -> Result<f32> {
    field(v, key)?
        .as_f64()
        .map(|x| x as f32)
        .ok_or_else(|| Error::msg(format!("the codec config.json {key} is not a number")))
}

fn sizes(v: &Json, key: &str) -> Result<Vec<usize>> {
    field(v, key)?
        .as_array()
        .and_then(|a| a.iter().map(|x| x.as_u64().map(|x| x as usize)).collect())
        .ok_or_else(|| {
            Error::msg(format!(
                "the codec config.json {key} is not a list of sizes"
            ))
        })
}

impl CodecConfig {
    /// The decoder of the codec configuration `root`.
    ///
    /// The strides of every stage multiply to the samples of a frame.
    pub fn read(root: &Json) -> Result<Self> {
        let d = field(root, "decoder_config")?;
        let config = Self {
            sample_rate: size(root, "output_sample_rate")?,
            frame_samples: size(root, "decode_upsample_rate")?,
            quantizers: size(d, "num_quantizers")?,
            codebook_size: size(d, "codebook_size")?,
            semantic_quantizers: size(d, "num_semantic_quantizers")?,
            latent: size(d, "latent_dim")?,
            hidden: size(d, "hidden_size")?,
            heads: size(d, "num_attention_heads")?,
            head_dim: size(d, "head_dim")?,
            intermediate: size(d, "intermediate_size")?,
            layers: size(d, "num_hidden_layers")?,
            window: size(d, "sliding_window")?,
            rope_theta: float(d, "rope_theta")?,
            rms_eps: float(d, "rms_norm_eps")?,
            max_positions: size(d, "max_position_embeddings")?,
            upsampling_ratios: sizes(d, "upsampling_ratios")?,
            decoder_dim: size(d, "decoder_dim")?,
            upsample_rates: sizes(d, "upsample_rates")?,
        };
        let strides: usize = config.strides().product();
        if strides != config.frame_samples {
            return Err(Error::msg(format!(
                "the codec strides give {strides} samples a frame, its config names {}",
                config.frame_samples
            )));
        }
        Ok(config)
    }

    /// The stride of every stage after the transformer, in order.
    pub fn strides(&self) -> impl Iterator<Item = usize> + '_ {
        self.upsampling_ratios
            .iter()
            .chain(&self.upsample_rates)
            .copied()
    }
}

/// A causal stage of the decoder, as far back as one of its outputs reads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Reach {
    /// A conv of `kernel` taps spaced by `dilation`.
    Conv { kernel: usize, dilation: usize },
    /// A transposed conv of `kernel` taps that multiplies the rate by `stride`.
    ConvTr { kernel: usize, stride: usize },
    /// `layers` attention layers whose query sees `window` positions, itself included.
    Window { window: usize, layers: usize },
}

/// The frames of left context that make the output of a chain of causal stages after it
/// exact.
///
/// `stages` run in order from the frame rate.
/// A position at a rate `r` times the frame rate is `frame_samples / r` samples long.
/// The reach of each stage is summed in samples, then rounded up to whole frames.
/// An output of a transposed conv reads the input it lies in and the inputs its taps reach
/// before it.
pub fn left_context(stages: &[Reach], frame_samples: usize) -> Result<usize> {
    let mut rate = 1usize;
    let mut samples = 0usize;
    for stage in stages {
        let span = frame_samples / rate;
        match *stage {
            Reach::Conv { kernel, dilation } => samples += (kernel - 1) * dilation * span,
            Reach::ConvTr { kernel, stride } => {
                samples += (kernel - 1) / stride * span + span - span / stride;
                rate *= stride;
            }
            Reach::Window { window, layers } => samples += (window - 1) * layers * span,
        }
        if !frame_samples.is_multiple_of(rate) {
            return Err(Error::msg(format!(
                "a stage runs at {rate} positions a frame, which does not divide {frame_samples}"
            )));
        }
    }
    Ok(samples.div_ceil(frame_samples))
}

/// The most frames, up to `limit`, whose decode keeps `bytes(frames)` within `reserve`.
///
/// `bytes` grows with the frames.
/// `None` when 1 frame already passes the reserve.
pub fn frames_within(reserve: u64, limit: usize, bytes: impl Fn(usize) -> u64) -> Option<usize> {
    if limit == 0 || bytes(1) > reserve {
        return None;
    }
    let (mut lo, mut hi) = (1usize, limit);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if bytes(mid) <= reserve {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo)
}
