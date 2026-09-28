//! Kyutai `tts-1.6b-en_fr` - the Helium main transformer (component 4 of the port).
//!
//! Streaming decoder read from the checkpoint's `config.json`: RoPE, causal over a sliding
//! window, RMSNorm (`alpha` scale), SiLU-GLU gating, and CROSS-ATTENTION to the
//! conditioning source (voice/text fuser output). Per-layer forward:
//!   x += self_attn(rms_norm1(x))                 # causal RoPE self-attn
//!   x += cross_attn(layer_norm_cross(x), ca)     # attends the condition tokens
//!   x += gating(rms_norm2(x))                    # SiLU(h0).h1 GLU FFN
//! No LayerScale (config layer_scale=None). Output is pre-`out_norm` (out_norm lives
//! in the LM, applied after this stack). Reuses the tensor-native ops + sdpa + rope_i.

use crate::inference::model::acestep::fsq::rope_tables;
use crate::inference::model::sdpa::sdpa;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

const RMS_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-5;

/// The transformer's shape, read from the `config.json` published beside the checkpoint.
#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub dim: usize,
    pub heads: usize,
    pub layers: usize,
    /// Width of one half of the SiLU-GLU gate.
    pub hidden: usize,
    /// Positions the self-attention window reaches back.
    pub context: usize,
    pub rope_theta: f32,
}

impl Geometry {
    /// Read the geometry from the parsed `config.json`.
    pub fn from_config(cfg: &serde_json::Value) -> Result<Self> {
        let n = |k: &str| -> Result<usize> {
            cfg[k]
                .as_u64()
                .map(|v| v as usize)
                .ok_or_else(|| crate::tensor::Error(format!("kyutai config.json: missing {k}")))
        };
        let f = |k: &str| -> Result<f64> {
            cfg[k]
                .as_f64()
                .ok_or_else(|| crate::tensor::Error(format!("kyutai config.json: missing {k}")))
        };
        let dim = n("dim")?;
        // The gate's two halves together are two thirds of the feed-forward width.
        let feed_forward = (f("hidden_scale")? * dim as f64) as usize;
        Ok(Self {
            dim,
            heads: n("num_heads")?,
            layers: n("num_layers")?,
            hidden: 2 * feed_forward / 3,
            context: n("context")?,
            rope_theta: f("max_period")? as f32,
        })
    }

    fn head_dim(&self) -> usize {
        self.dim / self.heads.max(1)
    }

    /// F32 bytes one layer's self-attention cache holds over the full window.
    pub fn kv_bytes_per_layer(&self) -> u64 {
        const F32_BYTES: u64 = 4;
        2 * self.dim as u64 * self.context as u64 * F32_BYTES
    }
}

struct Layer {
    norm1: Tensor,    // rms alpha [dim]
    norm2: Tensor,    // rms alpha [dim]
    ncross_w: Tensor, // layernorm w [dim]
    ncross_b: Tensor, // layernorm b [dim]
    sa_in: Tensor,    // [3*dim, dim] fused QKV
    sa_out: Tensor,   // [dim, dim]
    ca_in: Tensor,    // [3*dim, dim] (Q from x, K/V from ca)
    ca_out: Tensor,   // [dim, dim]
    g_in: Tensor,     // [2*hidden, dim]
    g_out: Tensor,    // [dim, hidden]
    /// HeteroPlan device this layer's weights + KV live on. The layer relocates the
    /// incoming activation (and the conditioning) onto it - no-op within a segment, a
    /// GPU↔CPU copy only at a plan boundary.
    device: Device,
    g: Geometry,
}
impl Layer {
    fn load(vb: &VarBuilder, i: usize, device: Device, g: Geometry) -> Result<Self> {
        let p = vb.pp(format!("layers.{i}"));
        let (dim, hidden) = (g.dim, g.hidden);
        Ok(Self {
            norm1: p.get((1, 1, dim), "norm1.alpha")?.reshape(dim)?,
            norm2: p.get((1, 1, dim), "norm2.alpha")?.reshape(dim)?,
            ncross_w: p.get(dim, "norm_cross.weight")?,
            ncross_b: p.get(dim, "norm_cross.bias")?,
            sa_in: p.get((3 * dim, dim), "self_attn.in_proj_weight")?,
            sa_out: p.get((dim, dim), "self_attn.out_proj.weight")?,
            ca_in: p.get((3 * dim, dim), "cross_attention.in_proj_weight")?,
            ca_out: p.get((dim, dim), "cross_attention.out_proj.weight")?,
            g_in: p.get((2 * hidden, dim), "gating.linear_in.weight")?,
            g_out: p.get((dim, hidden), "gating.linear_out.weight")?,
            device,
            g,
        })
    }

    // heads: [S, dim] -> [1, H, S, hd]
    fn heads(&self, t: &Tensor, s: usize) -> Result<Tensor> {
        t.reshape((s, self.g.heads, self.g.head_dim()))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()
    }

    fn forward(&self, x: &Tensor, ca: &Tensor) -> Result<Tensor> {
        let dev = &self.device;
        let (dim, hd, hidden) = (self.g.dim, self.g.head_dim(), self.g.hidden);
        let x = x.to_device(dev)?;
        let ca = ca.to_device(dev)?;
        let (x, ca) = (&x, &ca);
        let s = x.shape().dims2()?.0;
        let sc = ca.shape().dims2()?.0;
        let scale = 1.0 / (hd as f32).sqrt();
        // -- self-attention (causal RoPE, sliding ctx) --
        let h = x.rms_norm(&self.norm1, RMS_EPS)?;
        let qkv = h.matmul_t(&self.sa_in)?;
        let (cosv, sinv) = rope_tables(s, hd, self.g.rope_theta);
        let cos = Tensor::from_vec_f32(cosv, (s, hd / 2))?.to_device(dev)?;
        let sin = Tensor::from_vec_f32(sinv, (s, hd / 2))?.to_device(dev)?;
        let q = self.heads(&qkv.narrow(1, 0, dim)?, s)?.rope_i(&cos, &sin)?;
        let k = self
            .heads(&qkv.narrow(1, dim, dim)?, s)?
            .rope_i(&cos, &sin)?;
        let v = self.heads(&qkv.narrow(1, 2 * dim, dim)?, s)?;
        let mut mask = vec![0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                if j > i || i - j >= self.g.context {
                    mask[i * s + j] = f32::NEG_INFINITY;
                }
            }
        }
        let m = Tensor::from_vec_f32(mask, (s, s))?.to_device(dev)?;
        let att = sdpa(&q, &k, &v, Some(&m), false, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, dim))?
            .matmul_t(&self.sa_out)?;
        let x = x.add(&att)?;
        // -- cross-attention to the conditioning source --
        let h = x.layer_norm(&self.ncross_w, Some(&self.ncross_b), LN_EPS)?;
        let q = self.heads(&h.matmul_t(&self.ca_in.narrow(0, 0, dim)?)?, s)?;
        let k = self.heads(&ca.matmul_t(&self.ca_in.narrow(0, dim, dim)?)?, sc)?;
        let v = self.heads(&ca.matmul_t(&self.ca_in.narrow(0, 2 * dim, dim)?)?, sc)?;
        let att = sdpa(&q, &k, &v, None, false, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, dim))?
            .matmul_t(&self.ca_out)?;
        let x = x.add(&att)?;
        // -- SiLU-GLU gating FFN --
        let h = x.rms_norm(&self.norm2, RMS_EPS)?.matmul_t(&self.g_in)?; // [s, 2*hidden]
        let gate = h
            .narrow(1, 0, hidden)?
            .silu()?
            .mul(&h.narrow(1, hidden, hidden)?)?;
        x.add(&gate.matmul_t(&self.g_out)?)
    }

    /// Streaming single-token step: `x: [1, dim]` at absolute position `pos`, with a
    /// per-layer KV cache (self-attn K/V accumulated; cross-attn K/V precomputed once from
    /// the constant conditioning). Equivalent to `forward` over the full history but O(1)
    /// in prior steps instead of re-running them.
    fn forward_step(&self, x: &Tensor, lc: &mut LayerCache, pos: usize) -> Result<Tensor> {
        let dev = &self.device;
        let (dim, hd, hidden) = (self.g.dim, self.g.head_dim(), self.g.hidden);
        let x = x.to_device(dev)?;
        let x = &x;
        let scale = 1.0 / (hd as f32).sqrt();
        // -- self-attention (append new K/V, attend over the cache) --
        let h = x.rms_norm(&self.norm1, RMS_EPS)?;
        let qkv = h.matmul_t(&self.sa_in)?; // [1, 3*dim]
        let (cosv, sinv) = rope_row(pos, hd, self.g.rope_theta);
        let cos = Tensor::from_vec_f32(cosv, (1, hd / 2))?.to_device(dev)?;
        let sin = Tensor::from_vec_f32(sinv, (1, hd / 2))?.to_device(dev)?;
        let q = self.heads(&qkv.narrow(1, 0, dim)?, 1)?.rope_i(&cos, &sin)?; // [1,H,1,hd]
        let k = self
            .heads(&qkv.narrow(1, dim, dim)?, 1)?
            .rope_i(&cos, &sin)?;
        let v = self.heads(&qkv.narrow(1, 2 * dim, dim)?, 1)?;
        let mut kc = match lc.k.take() {
            None => k,
            Some(prev) => Tensor::cat(&[&prev, &k], 2)?,
        };
        let mut vc = match lc.v.take() {
            None => v,
            Some(prev) => Tensor::cat(&[&prev, &v], 2)?,
        };
        // sliding window: keep only the last `context` keys/values
        let clen = kc.shape().dims()[2];
        if clen > self.g.context {
            kc = kc
                .narrow(2, clen - self.g.context, self.g.context)?
                .contiguous()?;
            vc = vc
                .narrow(2, clen - self.g.context, self.g.context)?
                .contiguous()?;
        }
        let att = sdpa(&q, &kc, &vc, None, false, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, dim))?
            .matmul_t(&self.sa_out)?;
        lc.k = Some(kc);
        lc.v = Some(vc);
        let x = x.add(&att)?;
        // -- cross-attention (K/V precomputed from the constant conditioning) --
        let h = x.layer_norm(&self.ncross_w, Some(&self.ncross_b), LN_EPS)?;
        let q = self.heads(&h.matmul_t(&self.ca_in.narrow(0, 0, dim)?)?, 1)?;
        let att = sdpa(&q, &lc.ck, &lc.cv, None, false, scale, 1.0)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((1, dim))?
            .matmul_t(&self.ca_out)?;
        let x = x.add(&att)?;
        // -- SiLU-GLU gating FFN --
        let h = x.rms_norm(&self.norm2, RMS_EPS)?.matmul_t(&self.g_in)?;
        let gate = h
            .narrow(1, 0, hidden)?
            .silu()?
            .mul(&h.narrow(1, hidden, hidden)?)?;
        x.add(&gate.matmul_t(&self.g_out)?)
    }
}

/// Precompute one position's RoPE cos/sin row (`a = pos.θ^(-2j/d)`) - matches
/// `rope_tables` but for a single absolute position, so a streaming step is O(1).
fn rope_row(pos: usize, d: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = d / 2;
    let (mut cos, mut sin) = (vec![0f32; half], vec![0f32; half]);
    for j in 0..half {
        let a = (pos as f32) * theta.powf(-2.0 * (j as f32) / (d as f32));
        cos[j] = a.cos();
        sin[j] = a.sin();
    }
    (cos, sin)
}

/// Per-layer streaming state: accumulated self-attn K/V + precomputed cross-attn K/V.
struct LayerCache {
    k: Option<Tensor>, // [1, H, cached, hd]
    v: Option<Tensor>,
    ck: Tensor, // [1, H, Sc, hd] cross keys
    cv: Tensor, // [1, H, Sc, hd] cross values
}

/// Streaming KV cache for the whole Helium stack (one `LayerCache` per layer + position).
pub struct HeliumCache {
    layers: Vec<LayerCache>,
    pos: usize,
}

/// The Helium transformer stack (pre-out_norm output).
pub struct HeliumTransformer {
    layers: Vec<Layer>,
    head: Device, // first layer's device (input side)
    tail: Device, // last layer's device (output side)
}
impl HeliumTransformer {
    /// Load the layers across `layer_devs` (one device per layer), each from the
    /// VarBuilder of its device.
    pub fn load_hetero(
        vbs: &crate::inference::place::plan::VbSet,
        layer_devs: &[Device],
        g: Geometry,
    ) -> Result<Self> {
        use crate::inference::place::plan::vb_on;
        let mut layers = Vec::with_capacity(g.layers);
        for (i, d) in layer_devs.iter().enumerate().take(g.layers) {
            let t = vb_on(vbs, d).pp("transformer");
            layers.push(Layer::load(&t, i, d.clone(), g)?);
        }
        let head = layer_devs.first().cloned().unwrap_or(Device::Cpu);
        let tail = layer_devs.last().cloned().unwrap_or(Device::Cpu);
        Ok(Self { layers, head, tail })
    }

    pub fn head(&self) -> &Device {
        &self.head
    }
    pub fn tail(&self) -> &Device {
        &self.tail
    }

    /// `x: [S, dim]`, `ca: [Sc, dim]` -> `[S, dim]` on `tail` (before the LM's out_norm).
    pub fn forward(&self, x: &Tensor, ca: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for l in &self.layers {
            h = l.forward(&h, ca)?;
        } // each layer relocates h + ca
        Ok(h)
    }

    /// Build a streaming cache for `ca: [Sc, dim]` - precomputes each layer's cross-attn
    /// K/V once (the conditioning is constant across the whole generation), on that layer's
    /// device so the streaming step needs no per-step transfer.
    pub fn new_cache(&self, ca: &Tensor) -> Result<HeliumCache> {
        let sc = ca.shape().dims2()?.0;
        let mut layers = Vec::with_capacity(self.layers.len());
        for l in &self.layers {
            let ca = ca.to_device(&l.device)?;
            let dim = l.g.dim;
            let ck = l.heads(&ca.matmul_t(&l.ca_in.narrow(0, dim, dim)?)?, sc)?;
            let cv = l.heads(&ca.matmul_t(&l.ca_in.narrow(0, 2 * dim, dim)?)?, sc)?;
            layers.push(LayerCache {
                k: None,
                v: None,
                ck,
                cv,
            });
        }
        Ok(HeliumCache { layers, pos: 0 })
    }

    /// Streaming single-token step: `x: [1, dim]` -> `[1, dim]` on `tail` (pre-out_norm),
    /// advancing the cache one position. Equivalent to `forward` over the full history.
    pub fn forward_step(&self, x: &Tensor, cache: &mut HeliumCache) -> Result<Tensor> {
        let pos = cache.pos;
        let mut h = x.clone();
        for (l, lc) in self.layers.iter().zip(cache.layers.iter_mut()) {
            h = l.forward_step(&h, lc, pos)?;
        }
        cache.pos += 1;
        Ok(h)
    }
}

impl HeliumTransformer {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::runs(
            self.layers.iter().map(|l| l.device.location()),
            0,
        )
    }
}
