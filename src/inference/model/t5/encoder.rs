//! T5 **encoder** - the text encoder shared by the image-gen models
//! (flux / z-image) and parler-TTS.
//!
//! PHASE-5 MIGRATED: loads and computes on the
//! NATIVE substrate. The stack runs at the builder's dtype/device (BF16 on GPU
//! for flux's T5-xxl - bf16 is required, its FF activations overflow f16;
//! F32 for parler). The embedding table is kept on HOST F32 (CPU lookup  - 
//! avoids bouncing a 250MB table off the GPU every forward) and the looked-up
//! rows move to the stack's device. T5 numerics preserved: the norm is an
//! rms_norm, NO 1/sqrt(d) attention scaling (T5 convention), relative-position
//! bias computed once at block 0 (F32, on-device) and threaded through.
//! Public boundary speaks the facade `Tensor` until the flip.

use crate::tensor::layer as nl;
use crate::tensor::layer::{Embedding, Linear, Mlp, RmsNorm};
use crate::tensor::ops::Activation;
use crate::tensor::Module;
use crate::inference::model::attention::{Kv, Mask, MultiHeadAttention};
use crate::tensor::VarBuilder;
use crate::tensor::{self, DType, Device, Tensor};
use serde::Deserialize;
use std::sync::Arc;

// What a T5 checkpoint states when its config file does not.
crate::serde_defaults! {
    default_relative_attention_max_distance: usize = 128;
    default_is_decoder: bool = false;
    default_use_cache: bool = true;
    default_tie_word_embeddings: bool = true;
}

/// T5 feed-forward activations.
#[derive(Debug, Deserialize, Default, Clone, Copy, PartialEq)]
pub enum Act {
    #[default]
    Relu,
    Gelu,
    NewGelu,
    Silu,
    Sigmoid,
}

impl Act {
    fn apply(&self, x: &Tensor) -> tensor::Result<Tensor> {
        match self {
            Act::Relu => x.relu(),
            Act::Gelu => x.gelu_erf(),
            Act::NewGelu => x.gelu(),
            Act::Silu => x.silu(),
            Act::Sigmoid => x.sigmoid(),
        }
    }
}

#[derive(Debug, Deserialize, Default, Clone, PartialEq)]
pub struct ActivationWithOptionalGating {
    pub gated: bool,
    pub activation: Act,
}

/// What the config's `feed_forward_proj` names, as a gate and an activation.
///
/// A `gated-` prefix means the feed-forward has a second projection whose output multiplies the
/// first's; without it there is one. The rest of the name is the activation, and several names
/// mean the same one - a checkpoint may say `swish` or `silu`, `gelu_new` or
/// `gelu_pytorch_tanh`, and mean what the row says.
const FEED_FORWARD_NAMES: &[(&str, bool, Act)] = &[
    ("gated-gelu", true, Act::NewGelu),
    ("gated-silu", true, Act::Silu),
    ("relu", false, Act::Relu),
    ("gelu", false, Act::Gelu),
    ("gelu_new", false, Act::NewGelu),
    ("gelu_pytorch_tanh", false, Act::NewGelu),
    ("silu", false, Act::Silu),
    ("swish", false, Act::Silu),
    ("sigmoid", false, Act::Sigmoid),
];

pub fn deserialize_feed_forward_proj_activation<'de, D>(
    deserializer: D,
) -> std::result::Result<ActivationWithOptionalGating, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    let name = String::deserialize(deserializer)?;
    FEED_FORWARD_NAMES
        .iter()
        .find(|(spelling, _, _)| *spelling == name)
        .map(|&(_, gated, activation)| ActivationWithOptionalGating { gated, activation })
        .ok_or_else(|| serde::de::Error::custom(format!("unknown T5 activation: {name}")))
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,
    pub dropout_rate: f64,
    pub layer_norm_epsilon: f64,
    pub initializer_factor: f64,
    #[serde(default, deserialize_with = "deserialize_feed_forward_proj_activation")]
    pub feed_forward_proj: ActivationWithOptionalGating,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_is_decoder")]
    pub is_decoder: bool,
    pub is_encoder_decoder: bool,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub decoder_start_token_id: Option<usize>,
}

/// The normalisation this family puts before every sub-layer: RMS - no mean subtraction  - 
/// over `d_model`, with the epsilon the config states.
fn norm(cfg: &Config, vb: &VarBuilder) -> tensor::Result<RmsNorm> {
    nl::rms_norm(cfg.d_model, cfg.layer_norm_epsilon as f32, vb)
}

#[derive(Debug, Clone)]
struct T5DenseGatedActDense {
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
    act: Act,
}

impl T5DenseGatedActDense {
    fn load(vb: &VarBuilder, cfg: &Config) -> tensor::Result<Self> {
        Ok(Self {
            wi_0: nl::linear_no_bias(cfg.d_model, cfg.d_ff, &vb.pp("wi_0"))?,
            wi_1: nl::linear_no_bias(cfg.d_model, cfg.d_ff, &vb.pp("wi_1"))?,
            wo: nl::linear_no_bias(cfg.d_ff, cfg.d_model, &vb.pp("wo"))?,
            act: cfg.feed_forward_proj.activation,
        })
    }

    fn forward(&self, xs: &Tensor) -> tensor::Result<Tensor> {
        let gate = self.act.apply(&self.wi_0.forward(xs)?)?;
        let up = self.wi_1.forward(xs)?;
        self.wo.forward(&gate.mul(&up)?)
    }
}

/// The feed-forward, in whichever of the two shapes the config's `feed_forward_proj` named.
///
/// Ungated it is the widen-activate-narrow every transformer has, and the activation is always
/// the relu the ungated checkpoints were trained with. Gated, a second widening projection
/// multiplies the first's output and the activation is the config's.
#[derive(Debug, Clone)]
enum Dense {
    Plain(Mlp),
    Gated(T5DenseGatedActDense),
}

impl Dense {
    fn load(vb: &VarBuilder, cfg: &Config) -> tensor::Result<Self> {
        let vb = vb.pp("DenseReluDense");
        if cfg.feed_forward_proj.gated {
            Ok(Self::Gated(T5DenseGatedActDense::load(&vb, cfg)?))
        } else {
            Ok(Self::Plain(Mlp::new(
                nl::linear_no_bias(cfg.d_model, cfg.d_ff, &vb.pp("wi"))?,
                Activation::Relu,
                nl::linear_no_bias(cfg.d_ff, cfg.d_model, &vb.pp("wo"))?,
            )))
        }
    }

    fn forward(&self, xs: &Tensor) -> tensor::Result<Tensor> {
        match self {
            Self::Plain(mlp) => mlp.forward(xs),
            Self::Gated(gated) => gated.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
struct T5LayerFF {
    dense: Dense,
    layer_norm: RmsNorm,
}

impl T5LayerFF {
    fn load(vb: &VarBuilder, cfg: &Config) -> tensor::Result<Self> {
        let layer_norm = norm(cfg, &vb.pp("layer_norm"))?;
        Ok(Self {
            dense: Dense::load(vb, cfg)?,
            layer_norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> tensor::Result<Tensor> {
        xs.add(&self.dense.forward(&self.layer_norm.forward(xs)?)?)
    }
}

/// Which bucket a pair of positions falls in, for every pair of a sequence of `kv_len`.
///
/// Half the buckets are for keys ahead of the query and half for keys behind it - the encoder
/// reads in both directions. Within each half the near distances get one bucket each and the far
/// ones are spread logarithmically up to `max_distance`, past which they all share the last
/// bucket: the model is asked to tell apart neighbours precisely and distant positions only
/// roughly.
///
/// It depends on nothing but its three arguments - no weights, no device - which is why it is
/// stated here rather than on the attention that happens to hold the table.
fn bucket_ids(buckets: usize, max_distance: usize, kv_len: usize) -> Vec<u32> {
    let half = buckets as u32 / 2;
    let exact = half / 2;
    let spread = |distance: u32| -> u32 {
        if distance < exact {
            distance
        } else {
            let far = f32::log(
                distance as f32 / exact as f32,
                max_distance as f32 / exact as f32,
            ) * (half - exact) as f32;
            u32::min(exact + far as u32, half - 1)
        }
    };
    let mut out = Vec::with_capacity(kv_len * kv_len);
    for query in 0..kv_len as u32 {
        for key in 0..kv_len as u32 {
            out.push(if key > query {
                half + spread(key - query)
            } else {
                spread(query - key)
            });
        }
    }
    out
}

/// Encoder self-attention: the shared attention, plus the relative-position bias this family
/// adds to the scores.
///
/// Two things here are T5's own. The scores are taken unscaled - the projections are trained to
/// absorb the `1/sqrt(head_dim)` every other family applies - and each pair of positions
/// contributes a learnt term that depends only on how far apart they are, per head. That term
/// is not a mask: it prefers, it does not forbid, and it is the same for every layer of the
/// stack, so the first layer that owns the table builds it and the rest are handed it.
#[derive(Debug, Clone)]
struct T5Attention {
    attn: MultiHeadAttention,
    /// Kept on the host in F32 - it is `[buckets, heads]`, and the bias it produces moves to
    /// the stack's device once per sequence length rather than per layer.
    relative_attention_bias: Option<Embedding>,
    buckets: usize,
    max_distance: usize,
    heads: usize,
}

impl T5Attention {
    fn load(has_bias: bool, vb: &VarBuilder, cfg: &Config) -> tensor::Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        let relative_attention_bias = if has_bias {
            let host_vb = vb.to(DType::F32, &Device::Cpu);
            Some(nl::embedding(
                cfg.relative_attention_num_buckets,
                cfg.num_heads,
                &host_vb.pp("relative_attention_bias"),
            )?)
        } else {
            None
        };
        let one = |name: &str, from: usize, to: usize| nl::linear_no_bias(from, to, &vb.pp(name));
        Ok(Self {
            attn: MultiHeadAttention::new(
                one("q", cfg.d_model, inner_dim)?,
                one("k", cfg.d_model, inner_dim)?,
                one("v", cfg.d_model, inner_dim)?,
                one("o", inner_dim, cfg.d_model)?,
                inner_dim,
                cfg.num_heads,
                cfg.num_heads,
                Kv::None,
            )?
            .with_scale(1.0),
            relative_attention_bias,
            buckets: cfg.relative_attention_num_buckets,
            max_distance: cfg.relative_attention_max_distance,
            heads: cfg.num_heads,
        })
    }

    /// The bias for a sequence of this length, shaped like the scores: `[1, heads, q, k]`.
    fn position_bias(&self, kv_len: usize, device: &Device) -> tensor::Result<Option<Tensor>> {
        let Some(table) = &self.relative_attention_bias else {
            return Ok(None);
        };
        let ids = Tensor::from_vec_u32(
            bucket_ids(self.buckets, self.max_distance, kv_len),
            vec![kv_len * kv_len],
        )?;
        Ok(Some(
            table
                .forward(&ids)?
                .reshape(vec![kv_len, kv_len, self.heads])?
                .transpose(0, 2)?
                .transpose(1, 2)?
                .reshape(vec![1, self.heads, kv_len, kv_len])?
                .to_device(device)?,
        ))
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> tensor::Result<(Tensor, Option<Tensor>)> {
        let kv_len = xs.dim(1)?;
        let bias = match position_bias {
            Some(pb) => Some(pb.clone()),
            None => self.position_bias(kv_len, &xs.device())?,
        };
        let mask = match &bias {
            Some(pb) => Mask::Added(pb),
            None => Mask::All,
        };
        Ok((self.attn.forward_stateless(xs, None, mask)?, bias))
    }
}

#[derive(Debug, Clone)]
struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: RmsNorm,
}

impl T5LayerSelfAttention {
    fn load(h: bool, vb: &VarBuilder, cfg: &Config) -> tensor::Result<Self> {
        Ok(Self {
            self_attention: T5Attention::load(h, &vb.pp("SelfAttention"), cfg)?,
            layer_norm: norm(cfg, &vb.pp("layer_norm"))?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> tensor::Result<(Tensor, Option<Tensor>)> {
        let normed = self.layer_norm.forward(xs)?;
        let (ys, position_bias) = self.self_attention.forward(&normed, position_bias)?;
        Ok((xs.add(&ys)?, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    ff: T5LayerFF,
}

impl T5Block {
    fn load(has_bias: bool, vb: &VarBuilder, cfg: &Config) -> tensor::Result<Self> {
        let vb = vb.pp("layer");
        Ok(Self {
            self_attn: T5LayerSelfAttention::load(has_bias, &vb.pp("0"), cfg)?,
            ff: T5LayerFF::load(&vb.pp("1"), cfg)?,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
    ) -> tensor::Result<(Tensor, Option<Tensor>)> {
        let (xs, position_bias) = self.self_attn.forward(xs, position_bias)?;
        Ok((self.ff.forward(&xs)?, position_bias))
    }
}

#[derive(Debug, Clone)]
struct T5Stack {
    block: Vec<T5Block>,
    /// Embedding table on HOST F32 (CPU lookup; rows move to the stack device).
    shared: Arc<Embedding>,
    final_layer_norm: RmsNorm,
}

impl T5Stack {
    fn load(vb: &VarBuilder, shared: &Arc<Embedding>, cfg: &Config) -> tensor::Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| T5Block::load(i == 0, &vb.pp(format!("block.{i}")), cfg))
            .collect::<tensor::Result<Vec<_>>>()?;
        let final_layer_norm = norm(cfg, &vb.pp("final_layer_norm"))?;
        Ok(Self {
            block,
            shared: shared.clone(),
            final_layer_norm,
        })
    }

    fn forward(&self, input_ids: &Tensor, device: &Device, dtype: DType) -> tensor::Result<Tensor> {
        let embeds = self.shared.forward(input_ids)?; // CPU F32 lookup
        let mut hidden = embeds.to_device(device)?;
        if dtype != DType::F32 {
            hidden = hidden.to_dtype(dtype)?;
        }
        let mut position_bias: Option<Tensor> = None;
        for block in self.block.iter() {
            let (hs, pb) = block.forward(&hidden, position_bias.as_ref())?;
            hidden = hs;
            position_bias = pb;
        }
        self.final_layer_norm.forward(&hidden)
    }
}

/// T5 encoder on the native substrate.
#[derive(Debug, Clone)]
pub struct T5EncoderModel {
    encoder: T5Stack,
    device: Device,
    dtype: DType,
}

impl T5EncoderModel {
    /// `vb` carries the stack's dtype/device (BF16+GPU for flux, F32 for
    /// parler); the embedding table is retargeted to host F32 internally.
    pub fn load(vb: VarBuilder, cfg: &Config) -> tensor::Result<Self> {
        let host_vb = vb.to(DType::F32, &Device::Cpu);
        let shared_vb = if host_vb.contains("shared.weight") {
            host_vb.pp("shared")
        } else if host_vb.contains("decoder.embed_tokens.weight") {
            host_vb.pp("decoder").pp("embed_tokens")
        } else {
            host_vb.pp("encoder").pp("embed_tokens")
        };
        let shared = Arc::new(nl::embedding(cfg.vocab_size, cfg.d_model, &shared_vb)?);
        let encoder = T5Stack::load(&vb.pp("encoder"), &shared, cfg)?;
        Ok(Self {
            encoder,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Boundary: facade ids in, F32 facade embeddings out on the caller's
    /// device (callers cast to their compute dtype as before).
    pub fn forward(
        &mut self,
        input_ids: &crate::tensor::Tensor,
    ) -> crate::tensor::Result<crate::tensor::Tensor> {
        let dims = input_ids.dims().to_vec();
        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let nids = Tensor::from_vec_u32(ids, dims)?;
        let out = self.encoder.forward(&nids, &self.device, self.dtype)?;
        let out = if out.dtype() != DType::F32 {
            out.to_dtype(DType::F32)?
        } else {
            out
        };
        out.to_device(&input_ids.device())
    }

    /// No-op: the encoder is bidirectional, no decode KV cache.
    pub fn clear_kv_cache(&mut self) {}
}

#[cfg(test)]
mod bucket_tests {
    use super::*;

    /// The buckets are the ones the checkpoint was trained against.
    ///
    /// The bidirectional split, the exact-then-logarithmic spread and the clamp all fold into
    /// one expression per direction here, and folding is where an off-by-one lives. So the
    /// result is held against the rule written out the long way, over lengths that reach past
    /// the exact region and past the maximum distance - the two places the branches change.
    #[test]
    fn the_position_buckets_are_the_ones_the_checkpoint_learnt() {
        for (buckets, max_distance) in [(32usize, 128usize), (32, 16), (64, 128), (8, 4)] {
            for len in [1usize, 5, 40, 200] {
                let got = bucket_ids(buckets, max_distance, len);
                // The rule, one branch per case, with nothing folded.
                let half = buckets as u32 / 2;
                let exact = half / 2;
                let mut want = Vec::with_capacity(len * len);
                for i in 0..len as u32 {
                    for j in 0..len as u32 {
                        want.push(if i < j {
                            if j - i < exact {
                                j - i + half
                            } else {
                                let b = f32::log(
                                    (j - i) as f32 / exact as f32,
                                    max_distance as f32 / exact as f32,
                                ) * (half - exact) as f32;
                                u32::min(exact + half + b as u32, buckets as u32 - 1)
                            }
                        } else if i - j < exact {
                            i - j
                        } else {
                            let b = f32::log(
                                (i - j) as f32 / exact as f32,
                                max_distance as f32 / exact as f32,
                            ) * (half - exact) as f32;
                            u32::min(exact + b as u32, half - 1)
                        });
                    }
                }
                assert_eq!(
                    got, want,
                    "buckets={buckets} max_distance={max_distance} len={len}"
                );
                assert!(
                    got.iter().all(|&b| (b as usize) < buckets),
                    "a bucket id fell outside the table"
                );
            }
        }
    }
}
