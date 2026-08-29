//! The text encoder Z-Image conditions on.
//!
//! A decoder tower: rotary positions, grouped-query attention with a per-head RMSNorm on the
//! queries and the keys, and a gated-SiLU feed-forward. What it hands back is the hidden state
//! of the SECOND-TO-LAST layer, taken before the final norm - that is the conditioning
//! Z-Image was trained against, so the layers past it are never run.
//!
//! The embedding table is a gigabyte and a half of F32 and is read one row at a time, so it is
//! kept on the host and looked up there; everything after it computes on the device.

use crate::tensor::layer as nl;
use crate::tensor::layer::{Embedding, Linear, QkNorm, RmsNorm, SwiGlu};
use crate::tensor::VarBuilder;
use crate::tensor::{self, DType, Device, Tensor};
use std::sync::Arc;

/// What a checkpoint says about the tower, read outside in: the tower itself, then the two
/// blocks a layer is made of, then what surrounds a token.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextEncoderConfig {
    // How tall the tower is, and how wide the residual stream running through it.
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    // One attention block: how the projected width is cut into heads, how many queries share a
    // key/value head, and whether the projections carry a bias.
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,

    // One feed-forward block: the width it opens out to before closing back.
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    // What every norm in the tower is settled with.
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,

    // Where a token sits: the base the rotation is built on, and how far the positions run.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    // What a token is drawn from.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
}

// What the published Z-Image text encoder states where its config file is silent, in the order
// the fields above are declared, so an omission can be read against its neighbours.
crate::serde_defaults! {
    default_num_hidden_layers: usize = 36;
    default_hidden_size: usize = 2560;
    default_num_attention_heads: usize = 32;
    default_num_key_value_heads: usize = 8;
    default_head_dim: usize = 128;
    default_attention_bias: bool = false;
    default_intermediate_size: usize = 9728;
    default_rms_norm_eps: f64 = 1e-6;
    default_rope_theta: f64 = 1_000_000.0;
    default_max_position_embeddings: usize = 40960;
    default_vocab_size: usize = 151936;
}

/// The published Z-Image text encoder, which is what every fallback above states.
///
/// Read back by decoding a document that omits everything, rather than by restating the eleven
/// fields here: the preset and a config file that says nothing are then the same object by
/// construction, and there is no second list to keep in step. The decode cannot fail - every
/// field names a fallback - and it runs once, when a model is loaded.
impl Default for TextEncoderConfig {
    fn default() -> Self {
        serde_json::from_str("{}").expect("every field of the config names a fallback")
    }
}

impl TextEncoderConfig {
    pub fn z_image() -> Self {
        Self::default()
    }
}

/// Host-precomputed RoPE tables; per-forward [seq, half] slices move to the
/// stack's device (cheap: kilobytes).
#[derive(Debug, Clone)]
struct RotaryTables {
    cos: Vec<f32>,
    sin: Vec<f32>,
    half: usize,
}

impl RotaryTables {
    fn new(cfg: &TextEncoderConfig) -> Self {
        let dim = cfg.head_dim;
        let half = dim / 2;
        let max = cfg.max_position_embeddings;
        let mut cos = vec![0f32; max * half];
        let mut sin = vec![0f32; max * half];
        for t in 0..max {
            for i in 0..half {
                let inv = 1f64 / cfg.rope_theta.powf(2.0 * i as f64 / dim as f64);
                let ang = t as f64 * inv;
                cos[t * half + i] = ang.cos() as f32;
                sin[t * half + i] = ang.sin() as f32;
            }
        }
        Self { cos, sin, half }
    }

    fn slices(&self, seq: usize, device: &Device) -> tensor::Result<(Tensor, Tensor)> {
        let c = Tensor::from_vec_f32(self.cos[..seq * self.half].to_vec(), vec![seq, self.half])?
            .to_device(device)?;
        let s = Tensor::from_vec_f32(self.sin[..seq * self.half].to_vec(), vec![seq, self.half])?
            .to_device(device)?;
        Ok((c, s))
    }
}

/// How the projected width is cut up.
///
/// The two head counts and the head width are what every shape in an attention block follows
/// from; anything that follows from them - how many queries share a key/value head, how wide
/// the heads recombine into - is asked of this rather than kept beside it, because a stored
/// number that repeats two others is a number that can come to disagree with them.
#[derive(Debug, Clone, Copy)]
struct HeadShape {
    query: usize,
    key_value: usize,
    dim: usize,
}

impl HeadShape {
    fn kv_groups(self) -> usize {
        self.query / self.key_value
    }

    fn width(self) -> usize {
        self.dim * self.query
    }
}

/// The four projections, the pair of per-head norms the queries and keys pass through, and the
/// shape everything is cut to. The values go unnormed, which is why the norms are a pair and
/// not one per stream.
#[derive(Debug, Clone)]
struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    qk_norm: QkNorm,
    shape: HeadShape,
}

impl Attention {
    fn new(cfg: &TextEncoderConfig, vb: &VarBuilder) -> tensor::Result<Self> {
        // Everything the shape of this attention depends on, taken out of the config in one
        // place; the rest of the config describes the tower around it.
        let &TextEncoderConfig {
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            attention_bias,
            ..
        } = cfg;
        let shape = HeadShape {
            query: num_attention_heads,
            key_value: num_key_value_heads,
            dim: head_dim,
        };
        // A checkpoint carries a bias on all four projections or on none of them; the per-head
        // norms never have one.
        let projection = |inp: usize, out: usize, name: &str| -> tensor::Result<Linear> {
            if attention_bias {
                nl::linear(inp, out, &vb.pp(name))
            } else {
                nl::linear_no_bias(inp, out, &vb.pp(name))
            }
        };
        // A per-head norm is over one head's width, not the stream's.
        let head_norm = |name: &str| -> tensor::Result<RmsNorm> {
            nl::rms_norm(head_dim, rms_norm_eps as f32, &vb.pp(name))
        };
        Ok(Self {
            query: projection(hidden_size, shape.width(), "q_proj")?,
            key: projection(hidden_size, shape.key_value * head_dim, "k_proj")?,
            value: projection(hidden_size, shape.key_value * head_dim, "v_proj")?,
            out: projection(shape.width(), hidden_size, "o_proj")?,
            qk_norm: QkNorm::new(head_norm("q_norm")?, head_norm("k_norm")?),
            shape,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &(Tensor, Tensor),
    ) -> tensor::Result<Tensor> {
        let dims = x.dims();
        let (b, l) = (dims[0], dims[1]);
        // Project, then cut the projected width into heads and bring the head axis in front of
        // the sequence. The three streams differ only in how many heads they are cut into.
        let heads = |proj: &Linear, count: usize| -> tensor::Result<Tensor> {
            proj.forward(x)?
                .reshape(vec![b, l, count, self.shape.dim])?
                .transpose(1, 2)
        };
        let q = heads(&self.query, self.shape.query)?;
        let k = heads(&self.key, self.shape.key_value)?;
        let v = heads(&self.value, self.shape.key_value)?;
        // per-head RMSNorm (norm runs over the last dim regardless of shape)
        let (q, k) = self.qk_norm.forward(&q, &k)?;
        let (cos, sin) = rope;
        let q = q.rope(cos, sin)?;
        let k = k.rope(cos, sin)?;
        let k = crate::tensor::ops::repeat_kv(k, self.shape.kv_groups())?;
        let v = crate::tensor::ops::repeat_kv(v, self.shape.kv_groups())?;
        let scale = 1.0 / (self.shape.dim as f32).sqrt();
        // q.kᵀ.scale, an additive mask, softmax, then .v - the shared kernel, no softcapping.
        let ctx = crate::inference::model::acestep::ops::sdpa(&q, &k, &v, mask, false, scale, 1.0)?;
        let out = ctx
            .transpose(1, 2)?
            .reshape(vec![b, l, self.shape.width()])?;
        self.out.forward(&out)
    }
}

/// A sublayer with the norm its input passes through first - the pair one residual step of the
/// tower is made of. Keeping the two together is what stops a layer from growing a norm that
/// belongs to nothing in particular.
#[derive(Debug, Clone)]
struct Normed<S> {
    norm: RmsNorm,
    inner: S,
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    attention: Normed<Attention>,
    feed_forward: Normed<SwiGlu>,
}

impl DecoderLayer {
    fn new(cfg: &TextEncoderConfig, vb: &VarBuilder) -> tensor::Result<Self> {
        let norm = |name: &str| -> tensor::Result<RmsNorm> {
            nl::rms_norm(cfg.hidden_size, cfg.rms_norm_eps as f32, &vb.pp(name))
        };
        let ff = vb.pp("mlp");
        Ok(Self {
            attention: Normed {
                norm: norm("input_layernorm")?,
                inner: Attention::new(cfg, &vb.pp("self_attn"))?,
            },
            feed_forward: Normed {
                norm: norm("post_attention_layernorm")?,
                // `gate_proj` / `up_proj` / `down_proj` - the names this checkpoint gives the
                // three weights of the gated feed-forward.
                inner: SwiGlu::new(
                    nl::linear_no_bias(
                        cfg.hidden_size,
                        cfg.intermediate_size,
                        &ff.pp("gate_proj"),
                    )?,
                    nl::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, &ff.pp("up_proj"))?,
                    nl::linear_no_bias(
                        cfg.intermediate_size,
                        cfg.hidden_size,
                        &ff.pp("down_proj"),
                    )?,
                ),
            },
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        rope: &(Tensor, Tensor),
    ) -> tensor::Result<Tensor> {
        let h = self.attention.norm.forward(x)?;
        let h = self.attention.inner.forward(&h, mask, rope)?;
        let x = x.add(&h)?;
        let h2 = self.feed_forward.norm.forward(&x)?;
        let h2 = self.feed_forward.inner.forward(&h2)?;
        x.add(&h2)
    }
}

/// Z-Image text encoder on the native substrate.
#[derive(Debug, Clone)]
pub struct ZImageTextEncoder {
    embed_tokens: Arc<Embedding>,
    layers: Vec<DecoderLayer>,
    rotary: RotaryTables,
    /// Which layer's output is the conditioning: the second-to-last. Settled once here rather
    /// than counted back from the layer total on every forward.
    conditioning_layer: usize,
    hidden_size: usize,
    device: Device,
    dtype: DType,
}

impl ZImageTextEncoder {
    pub fn new(cfg: &TextEncoderConfig, vb: VarBuilder) -> tensor::Result<Self> {
        let vb_model = vb.pp("model");
        // big table on host F32, CPU lookup (T5's recipe)
        let host_vb = vb_model.to(DType::F32, &Device::Cpu);
        let embed_tokens = Arc::new(nl::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            &host_vb.pp("embed_tokens"),
        )?);
        let vb_layers = vb_model.pp("layers");
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| DecoderLayer::new(cfg, &vb_layers.pp(i)))
            .collect::<tensor::Result<Vec<_>>>()?;
        let (device, dtype) = (vb.device().clone(), vb.dtype());
        Ok(Self {
            embed_tokens,
            layers,
            rotary: RotaryTables::new(cfg),
            conditioning_layer: cfg.num_hidden_layers - 2,
            hidden_size: cfg.hidden_size,
            device,
            dtype,
        })
    }

    /// Returns the second-to-last layer hidden states WITHOUT the final norm.
    /// Boundary: facade ids in / F32 hidden states out on the caller's device.
    pub fn forward(
        &self,
        input_ids: &crate::tensor::Tensor,
    ) -> crate::tensor::Result<crate::tensor::Tensor> {
        let (b, l) = input_ids.dims2()?;
        let ids: Vec<u32> = input_ids.flatten_all()?.to_vec1::<u32>()?;
        let out = self.forward_ids(&ids, b, l)?;
        let out = if out.dtype() != DType::F32 {
            out.to_dtype(DType::F32)?
        } else {
            out
        };
        out.to_device(&input_ids.device())
    }

    fn forward_ids(&self, ids: &[u32], b: usize, l: usize) -> tensor::Result<Tensor> {
        let nids = Tensor::from_vec_u32(ids.to_vec(), vec![b, l])?;
        let mut hidden = self.embed_tokens.forward(&nids)?.to_device(&self.device)?;
        if self.dtype != DType::F32 {
            hidden = hidden.to_dtype(self.dtype)?;
        }
        let mask = if l == 1 {
            None
        } else {
            Some(
                crate::tensor::ops::causal_mask(l, f32::NEG_INFINITY, &self.device)?
                    .reshape(vec![1, 1, l, l])?,
            )
        };
        let rope = self.rotary.slices(l, &self.device)?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, mask.as_ref(), &rope)?;
            if i == self.conditioning_layer {
                return Ok(hidden);
            }
        }
        Err(tensor::Error("layer index out of bounds".into()))
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
