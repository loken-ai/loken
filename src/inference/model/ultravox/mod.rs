//! Ultravox (audio->text) - full-Rust audio front-end: the Whisper-large-v3-turbo
//! encoder (loaded from the `audio_tower.*` tensors via [`crate::inference::model::whisper::model`])
//! + the trained `multi_modal_projector` that maps encoder frames into the
//! Llama-3.2-1B embedding space.
//!
//! Reference: fixie-ai/ultravox-v0_5-llama-3_2-1b (`ultravox_model.py`). The projector
//! (from `UltravoxProjector`): stack `stack_factor` (8) consecutive encoder frames ->
//! `[T/8, C*8]` (C=1280 -> 10240) -> `ln_pre` (RMSNorm) -> `linear_1` (10240->4096, no bias)
//! -> SwiGLU (->2048) -> `ln_mid` (RMSNorm) -> `linear_2` (2048->2048, no bias) -> `[T/8, 2048]`.
//! SwiGLU here is `x, gate = chunk(2, -1); silu(gate) * x` (gate = the SECOND half).
//! `ln_post` is `Identity` for v0.5 (`projector_ln_mid=true`).
//!
//! Everything runs on the native tensor substrate so the projector composes directly
//! with the native Whisper `AudioEncoder` - no compat/native boundary conversions.

use crate::inference::model::whisper::model::AudioEncoder;
use crate::inference::model::whisper::Config as WhisperConfig;
use crate::tensor::layer::{linear_no_bias, rms_norm, Linear, RmsNorm};
use crate::tensor::VarBuilder;
use crate::tensor::{Result, Tensor};

/// Static dims for the v0.5 llama-3.2-1b checkpoint (Whisper-large-v3-turbo audio tower).
#[derive(Clone, Debug)]
pub struct UltravoxProjectorConfig {
    /// Whisper encoder hidden size (`audio_config.d_model`).
    pub audio_hidden: usize,
    /// Consecutive frames stacked into one projector input (`stack_factor`).
    pub stack_factor: usize,
    /// `linear_1` output width (`hidden_size`); SwiGLU halves it.
    pub inter_dim: usize,
    /// Llama embedding width (`text_config.hidden_size`).
    pub text_hidden: usize,
    /// RMSNorm epsilon.
    pub norm_eps: f32,
}

impl Default for UltravoxProjectorConfig {
    fn default() -> Self {
        Self {
            audio_hidden: 1280,
            stack_factor: 8,
            inter_dim: 4096,
            text_hidden: 2048,
            norm_eps: 1e-6,
        }
    }
}

/// The Ultravox audio->text projector (the only net-new module vs the existing
/// Whisper encoder + Llama decoder).
pub struct UltravoxProjector {
    ln_pre: RmsNorm,
    linear_1: Linear,
    ln_mid: RmsNorm,
    linear_2: Linear,
    stack_factor: usize,
}

impl UltravoxProjector {
    /// Load from a VarBuilder rooted at the safetensors top level (expects the
    /// `multi_modal_projector.*` tensors). All projector linears are bias-free.
    pub fn load(vb: &VarBuilder, cfg: &UltravoxProjectorConfig) -> Result<Self> {
        let vb = vb.pp("multi_modal_projector");
        let dim_in = cfg.audio_hidden * cfg.stack_factor; // 10240
        let dim_mid = cfg.inter_dim / 2; // 2048 (SwiGLU halves linear_1's output)

        let ln_pre = rms_norm(dim_in, cfg.norm_eps, &vb.pp("ln_pre"))?;
        let linear_1 = linear_no_bias(dim_in, cfg.inter_dim, &vb.pp("linear_1"))?;
        let ln_mid = rms_norm(dim_mid, cfg.norm_eps, &vb.pp("ln_mid"))?;
        let linear_2 = linear_no_bias(dim_mid, cfg.text_hidden, &vb.pp("linear_2"))?;

        Ok(Self {
            ln_pre,
            linear_1,
            ln_mid,
            linear_2,
            stack_factor: cfg.stack_factor,
        })
    }

    /// `audio_features`: `[T, C]` Whisper-encoder frames for one clip (batch = 1).
    /// Returns `[ceil(T / stack_factor), text_hidden]` Llama-space embeddings.
    pub fn forward(&self, audio_features: &Tensor) -> Result<Tensor> {
        let (t, c) = audio_features.shape().dims2()?;
        let s = self.stack_factor;
        // Pad the frame axis up to a multiple of stack_factor (mirrors StackAudioFrames).
        let t_pad = t.div_ceil(s) * s;
        let x = if t_pad != t {
            audio_features.pad_with_zeros(0, 0, t_pad - t)?
        } else {
            audio_features.clone()
        };
        // Stack s consecutive frames along the feature axis: [t_pad, c] -> [t_pad/s, c*s].
        let x = x.reshape((t_pad / s, c * s))?;

        let x = self.ln_pre.forward(&x)?;
        let x = self.linear_1.forward(&x)?;

        // SwiGLU: x, gate = x.chunk(2, -1); silu(gate) * x. After the reshape x is 2-D,
        // so the feature axis is dim 1.
        let half = x.dim(1)? / 2;
        let x1 = x.narrow(1, 0, half)?;
        let gate = x.narrow(1, half, half)?;
        let x = gate.silu()?.mul(&x1)?;

        let x = self.ln_mid.forward(&x)?;
        self.linear_2.forward(&x)
    }
}

/// Full Ultravox audio front-end: Whisper-large-v3-turbo encoder (loaded from the
/// `audio_tower.*` tensors) + the projector. Maps a mel spectrogram to Llama-space
/// audio embeddings ready to splice into the text sequence.
pub struct UltravoxAudio {
    encoder: AudioEncoder,
    projector: UltravoxProjector,
}

impl UltravoxAudio {
    /// The Whisper-large-v3-turbo encoder config baked into ultravox-v0.5 (128 mel
    /// bins, d_model 1280, 32 encoder layers, 20 heads). Decoder fields are unused
    /// (encoder-only) and set to plausible large-v3 values.
    pub fn whisper_config() -> WhisperConfig {
        WhisperConfig {
            num_mel_bins: 128,
            max_source_positions: 1500,
            d_model: 1280,
            encoder_attention_heads: 20,
            encoder_layers: 32,
            vocab_size: 51866,
            max_target_positions: 448,
            decoder_attention_heads: 20,
            decoder_layers: 4,
            suppress_tokens: Vec::new(),
        }
    }

    /// Load both stages from a VarBuilder rooted at the ultravox safetensors top level.
    pub fn load(vb: &VarBuilder) -> Result<Self> {
        let mut encoder = AudioEncoder::load(vb.pp("audio_tower"), &Self::whisper_config())?;
        // Ultravox froze the exact Whisper; use its STORED positional embedding rather
        // than recomputed sinusoids (they differ ~1e-5, which compounds over 32 layers).
        let pos = vb
            .pp("audio_tower")
            .get((1500, 1280), "embed_positions.weight")?;
        encoder.set_positional_embedding(pos);
        let projector = UltravoxProjector::load(vb, &UltravoxProjectorConfig::default())?;
        Ok(Self { encoder, projector })
    }

    /// `mel`: `[1, 128, frames]` -> audio embeds `[frames/2/8, 2048]` in Llama space.
    pub fn forward(&mut self, mel: &Tensor) -> Result<Tensor> {
        let feats = self.encoder.forward(mel, true)?; // [1, frames/2, 1280]
        let feats = feats.squeeze(0)?; // [frames/2, 1280]
        self.projector.forward(&feats)
    }
}
