//! The transformer of the codec decoder, at the frame rate.
//!
//! Each layer is pre-norm: RMSNorm, attention with separate q, k and v, LayerScale, then
//! RMSNorm, a SwiGLU feed-forward and LayerScale.
//! The attention is causal and sees a sliding window of positions.
//! The rotary embedding rotates the 2 halves of each head.

use super::config::CodecConfig;
use crate::inference::model::sdpa::sdpa;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

/// A linear layer `[out, in]` with its optional bias `[out]`.
pub(super) struct Linear {
    w: Tensor,
    b: Option<Tensor>,
}

impl Linear {
    /// The stored `{name}.weight`, with `{name}.bias` when the checkpoint has it.
    pub(super) fn load(vb: &VarBuilder, name: &str) -> Result<Self> {
        let shape = vb.shape_of(&format!("{name}.weight"))?;
        let bias = format!("{name}.bias");
        let b = match vb.contains(&bias) {
            true => Some(vb.get(shape[0], &bias)?),
            false => None,
        };
        Ok(Self {
            w: vb.get(shape, &format!("{name}.weight"))?,
            b,
        })
    }

    /// `x [S, in]` to `[S, out]`.
    pub(super) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.matmul_t(&self.w)?;
        match &self.b {
            None => Ok(y),
            Some(b) => y.broadcast_add(&b.reshape((1, b.dims()[0]))?),
        }
    }
}

/// One layer of the transformer.
struct Layer {
    input_norm: Tensor,
    post_norm: Tensor,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
    attn_scale: Tensor,
    mlp_scale: Tensor,
}

impl Layer {
    fn load(vb: &VarBuilder, hidden: usize) -> Result<Self> {
        Ok(Self {
            input_norm: vb.get(hidden, "input_layernorm.weight")?,
            post_norm: vb.get(hidden, "post_attention_layernorm.weight")?,
            q: Linear::load(vb, "self_attn.q_proj")?,
            k: Linear::load(vb, "self_attn.k_proj")?,
            v: Linear::load(vb, "self_attn.v_proj")?,
            o: Linear::load(vb, "self_attn.o_proj")?,
            gate: Linear::load(vb, "mlp.gate_proj")?,
            up: Linear::load(vb, "mlp.up_proj")?,
            down: Linear::load(vb, "mlp.down_proj")?,
            attn_scale: vb.get(hidden, "self_attn_layer_scale.scale")?,
            mlp_scale: vb.get(hidden, "mlp_layer_scale.scale")?,
        })
    }

    /// `x [S, hidden]` through the layer.
    ///
    /// `cos` and `sin` are `[S, head_dim / 2]`.
    /// `mask` is `[S, S]`, 0 where a query sees a key and minus infinity elsewhere.
    fn forward(
        &self,
        x: &Tensor,
        c: &CodecConfig,
        cos: &Tensor,
        sin: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let s = x.dims()[0];
        let hidden = x.dims()[1];
        let heads = |t: Tensor| -> Result<Tensor> {
            t.reshape((s, c.heads, c.head_dim))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()
        };
        let h = x.rms_norm(&self.input_norm, c.rms_eps)?;
        let q = heads(self.q.forward(&h)?)?.rope(cos, sin)?;
        let k = heads(self.k.forward(&h)?)?.rope(cos, sin)?;
        let v = heads(self.v.forward(&h)?)?;
        let scale = 1.0 / (c.head_dim as f32).sqrt();
        let att = sdpa(&q, &k, &v, Some(mask), false, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, c.heads * c.head_dim))?;
        let att = self.o.forward(&att)?;
        let x = x.add(&att.broadcast_mul(&self.attn_scale.reshape((1, hidden))?)?)?;
        let h = x.rms_norm(&self.post_norm, c.rms_eps)?;
        let h = self
            .down
            .forward(&self.gate.forward(&h)?.silu()?.mul(&self.up.forward(&h)?)?)?;
        x.add(&h.broadcast_mul(&self.mlp_scale.reshape((1, hidden))?)?)
    }
}

/// The input projection, the layers, the final norm and the output projection.
pub struct Transformer {
    input_proj: Linear,
    layers: Vec<Layer>,
    norm: Tensor,
    output_proj: Linear,
}

impl Transformer {
    /// The transformer under `vb`, `pre_transformer` of the decoder.
    pub fn load(vb: &VarBuilder, c: &CodecConfig) -> Result<Self> {
        let layers = (0..vb.count_children("layers"))
            .map(|i| Layer::load(&vb.pp(format!("layers.{i}")), c.hidden))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            input_proj: Linear::load(vb, "input_proj")?,
            layers,
            norm: vb.get(c.hidden, "norm.weight")?,
            output_proj: Linear::load(vb, "output_proj")?,
        })
    }

    /// `x [S, latent]` at positions `first..first + S` to `[S, latent]`.
    pub fn forward(&self, x: &Tensor, first: usize, c: &CodecConfig) -> Result<Tensor> {
        let s = x.dims()[0];
        let dev = &x.device();
        let (cos, sin) = rope_tables(first, s, c.head_dim, c.rope_theta);
        let half = c.head_dim / 2;
        let cos = Tensor::from_vec_f32(cos, (s, half))?.to_device(dev)?;
        let sin = Tensor::from_vec_f32(sin, (s, half))?.to_device(dev)?;
        let mask = window_mask(s, c.window, dev)?;
        let mut h = self.input_proj.forward(x)?;
        for layer in &self.layers {
            h = layer.forward(&h, c, &cos, &sin, &mask)?;
        }
        self.output_proj
            .forward(&h.rms_norm(&self.norm, c.rms_eps)?)
    }

    /// Layers of the transformer.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Whether the transformer has no layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

/// The cosine and the sine of positions `first..first + s` for heads of `d` values, `[s, d/2]`.
///
/// Pair `j` turns by `p * theta^(-2j/d)` at position `p`.
pub(super) fn rope_tables(first: usize, s: usize, d: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = d / 2;
    let inv: Vec<f32> = (0..half)
        .map(|j| 1.0 / theta.powf((2 * j) as f32 / d as f32))
        .collect();
    let mut cos = Vec::with_capacity(s * half);
    let mut sin = Vec::with_capacity(s * half);
    for p in first..first + s {
        for f in &inv {
            let a = p as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    (cos, sin)
}

/// `[s, s]`: 0 where key `j` is at most `window - 1` positions before query `i`, minus
/// infinity elsewhere.
pub(super) fn window_mask(s: usize, window: usize, dev: &Device) -> Result<Tensor> {
    let mut data = vec![f32::NEG_INFINITY; s * s];
    for i in 0..s {
        for j in i.saturating_sub(window - 1)..=i {
            data[i * s + j] = 0.0;
        }
    }
    Tensor::from_vec_f32(data, (s, s))?.to_device(dev)
}
