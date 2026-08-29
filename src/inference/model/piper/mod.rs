//! Native Piper TTS - VITS inference on `tensor` (no onnxruntime, no new crates).
//!
//! Pipeline (synthesis): phoneme ids -> TextEncoder (`enc_p`) -> m_p/logs_p + stochastic duration
//! predictor (`dp`) -> length-regulate (monotonic expand) -> z_p -> residual-coupling flow (`flow`,
//! reverse) -> HiFi-GAN decoder (`dec`) -> 22.05/44.1 kHz waveform.
//!
//! Weights come from the Piper `.onnx` via `native_onnx::OnnxModel`. This module builds the forward
//! pass by hand against the named initializers (the ONNX *graph* is ignored). Architecture is the
//! standard VITS used by Piper (see `piper_inspect` for the exact tensor map).
//!
//! Implemented incrementally; this file currently provides the model scaffold, weight helpers, and
//! the HiFi-GAN decoder. Encoder / flow / duration predictor follow.

use crate::inference::load::onnx::OnnxModel;
use crate::tensor::layer::{Conv1d, Conv1dConfig};
use crate::tensor::{self, Device, Tensor};

type Result<T> = tensor::Result<T>;
fn err(m: impl Into<String>) -> tensor::Error {
    tensor::Error(m.into())
}

/// VITS hyper-params for a Piper "medium" voice (read off the weight shapes).
#[derive(Clone)]
pub struct PiperConfig {
    pub n_symbols: usize,    // phoneme vocab (256)
    pub hidden: usize,       // 192
    pub inter: usize,        // 192 (flow/decoder channels)
    pub filter: usize,       // 768 (FFN)
    pub n_heads: usize,      // 2
    pub n_enc_layers: usize, // 6
    pub window: usize,       // 4 (relative-attention window)
    pub sample_rate: u32,    // 44100
    // HiFi-GAN
    pub upsample_rates: Vec<usize>,          // [8,8,4]
    pub upsample_kernels: Vec<usize>,        // [16,16,8]
    pub upsample_init: usize,                // 256
    pub resblock_kernels: Vec<usize>,        // [3,5,7]
    pub resblock_dilations: Vec<Vec<usize>>, // [[1,3]]x3
    // inference scales
    pub noise_scale: f32,  // 0.667
    pub noise_w: f32,      // 0.8 (duration noise)
    pub length_scale: f32, // 1.0
}

impl Default for PiperConfig {
    fn default() -> Self {
        Self {
            n_symbols: 256,
            hidden: 192,
            inter: 192,
            filter: 768,
            n_heads: 2,
            n_enc_layers: 6,
            window: 4,
            sample_rate: 44100,
            upsample_rates: vec![8, 8, 4],
            upsample_kernels: vec![16, 16, 8],
            upsample_init: 256,
            // Per-kernel MRF dilations read off the ONNX Conv node attributes (NOT the HiFi-GAN
            // ResBlock2 default [1,3]): k3->[1,2], k5->[2,6], k7->[3,12]. Padding follows (k-1).d/2.
            resblock_kernels: vec![3, 5, 7],
            resblock_dilations: vec![vec![1, 2], vec![2, 6], vec![3, 12]],
            noise_scale: 0.667,
            noise_w: 0.8,
            length_scale: 1.0,
        }
    }
}

const LRELU: f32 = 0.1;
/// LeakyReLU(x, slope) = slope.x + (1-slope).relu(x) - no dedicated kernel needed.
fn leaky_s(x: &Tensor, slope: f32) -> Result<Tensor> {
    x.affine(slope, 0.0)?
        .add(&x.relu()?.affine(1.0 - slope, 0.0)?)
}
/// HiFi-GAN body LeakyReLU (slope 0.1) - used in the upsample loop + resblocks.
fn leaky(x: &Tensor) -> Result<Tensor> {
    leaky_s(x, LRELU)
}

/// Load a Conv1d from onnx by `{prefix}.weight`/`.bias` with the given conv config.
fn conv1d(m: &OnnxModel, dev: &Device, prefix: &str, cfg: Conv1dConfig) -> Result<Conv1d> {
    let w = m.conv_weight(prefix, dev)?; // resolves anonymised weight-normed weights via node map
    let bname = format!("{prefix}.bias");
    let b = if m.contains(&bname) {
        Some(m.get_raw(&bname, dev)?)
    } else {
        None
    };
    Ok(Conv1d::new(w, b, cfg))
}
fn cfg1(pad: usize, dil: usize) -> Conv1dConfig {
    Conv1dConfig {
        padding: pad,
        stride: 1,
        dilation: dil,
        groups: 1,
    }
}

/// Load a Conv1d taking its padding/stride/dilation/groups from the ONNX node *attributes*
/// (not assumed). Used where the hyper-params can't be inferred from weight shapes - e.g. the
/// MRF resblock dilations. Falls back to `fallback` when the node has no attributes.
fn conv1d_auto(
    m: &OnnxModel,
    dev: &Device,
    prefix: &str,
    fallback: Conv1dConfig,
) -> Result<Conv1d> {
    let cfg = match m.conv_attrs(prefix) {
        Some(a) if !a.dilations.is_empty() || !a.pads.is_empty() => Conv1dConfig {
            padding: a.pads.first().copied().unwrap_or(fallback.padding),
            stride: a.strides.first().copied().unwrap_or(fallback.stride),
            dilation: a.dilations.first().copied().unwrap_or(fallback.dilation),
            groups: if a.group > 0 {
                a.group
            } else {
                fallback.groups
            },
        },
        _ => fallback,
    };
    conv1d(m, dev, prefix, cfg)
}

// -- HiFi-GAN decoder (`dec`) ------------------------------------------------------------

/// One MRF ResBlock (type 2): for each of its convs, `x = x + conv(leaky(x))` (dilated, same-pad).
struct ResBlock {
    convs: Vec<Conv1d>,
}
impl ResBlock {
    fn load(m: &OnnxModel, dev: &Device, prefix: &str, k: usize, dils: &[usize]) -> Result<Self> {
        let mut convs = Vec::new();
        // Per-conv dilation/padding come from the ONNX node attributes (authoritative); the
        // config `dils` is only a fallback for the count + when attributes are absent.
        for (i, &d) in dils.iter().enumerate() {
            convs.push(conv1d_auto(
                m,
                dev,
                &format!("{prefix}.convs.{i}"),
                cfg1((k - 1) * d / 2, d),
            )?);
        }
        Ok(Self { convs })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for c in &self.convs {
            let xt = c.forward(&leaky(&x)?)?;
            x = x.add(&xt)?;
        }
        Ok(x)
    }
}

pub struct Decoder {
    conv_pre: Conv1d,
    ups: Vec<(Tensor, Tensor, usize, usize)>, // (weight[in,out,k], bias[out], stride, pad)
    resblocks: Vec<ResBlock>,
    conv_post: Conv1d,
    n_kernels: usize,
}

impl Decoder {
    pub fn load(m: &OnnxModel, dev: &Device, c: &PiperConfig) -> Result<Self> {
        let conv_pre = conv1d(m, dev, "dec.conv_pre", cfg1(3, 1))?; // k7 pad3
        let mut ups = Vec::new();
        for i in 0..c.upsample_rates.len() {
            let w = m.get_raw(&format!("dec.ups.{i}.weight"), dev)?; // [in, out, k]
            let b = m.get_raw(&format!("dec.ups.{i}.bias"), dev)?;
            let (k, s) = (c.upsample_kernels[i], c.upsample_rates[i]);
            ups.push((w, b, s, (k - s) / 2));
        }
        let n_kernels = c.resblock_kernels.len();
        let mut resblocks = Vec::new();
        for i in 0..c.upsample_rates.len() {
            for (j, &k) in c.resblock_kernels.iter().enumerate() {
                resblocks.push(ResBlock::load(
                    m,
                    dev,
                    &format!("dec.resblocks.{}", i * n_kernels + j),
                    k,
                    &c.resblock_dilations[j.min(c.resblock_dilations.len() - 1)],
                )?);
            }
        }
        let conv_post = conv1d(m, dev, "dec.conv_post", cfg1(3, 1))?; // k7 pad3 -> 1 ch
        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            conv_post,
            n_kernels,
        })
    }

    /// z: [1, inter(192), T] -> waveform [1, 1, T.∏upsample]. Mirrors HiFiGAN.generator.
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = self.conv_pre.forward(z)?;
        for (i, (w, b, stride, pad)) in self.ups.iter().enumerate() {
            x = leaky(&x)?;
            x = x
                .conv_transpose1d(w, *pad, 0, *stride, 1, 1)?
                .add_channel_bias(b)?;
            // MRF: average the n_kernels resblocks for this stage.
            let mut acc: Option<Tensor> = None;
            for j in 0..self.n_kernels {
                let r = self.resblocks[i * self.n_kernels + j].forward(&x)?;
                acc = Some(match acc {
                    Some(a) => a.add(&r)?,
                    None => r,
                });
            }
            x = acc.unwrap().affine(1.0 / self.n_kernels as f32, 0.0)?;
        }
        // Final activation before conv_post uses PyTorch's DEFAULT leaky_relu slope
        // (0.01), NOT the 0.1 used in the body - see VITS Generator.forward. Using
        // 0.1 here passes 10x the negative signal -> asymmetric HF distortion.
        x = leaky_s(&x, 0.01)?;
        x = self.conv_post.forward(&x)?;
        x.tanh()
    }
}

// -- helpers ------------------------------------------------------------

/// LayerNorm over the channel dim of a [1,C,T] tensor (VITS `modules.LayerNorm`).
fn layer_norm(x: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let mean = x.mean_keepdim(1)?; // [1,1,T]
    let xc = x.broadcast_add(&mean.affine(-1.0, 0.0)?)?; // x - mean
    let var = xc.mul(&xc)?.mean_keepdim(1)?; // [1,1,T]
    let xn = xc.broadcast_div(&var.affine(1.0, 1e-5)?.sqrt()?)?;
    let c = gamma.dims()[0];
    xn.broadcast_mul(&gamma.reshape((1, c, 1))?)?
        .broadcast_add(&beta.reshape((1, c, 1))?)
}

// -- relative-position multi-head attention (VITS `attentions.MultiHeadAttention`) --
struct RelAttention {
    q: Conv1d,
    k: Conv1d,
    v: Conv1d,
    o: Conv1d,
    rel_k: Tensor,
    rel_v: Tensor, // [1, 2w+1, head_dim]
    n_heads: usize,
    head_dim: usize,
    window: usize,
}
impl RelAttention {
    fn load(m: &OnnxModel, dev: &Device, p: &str, c: &PiperConfig) -> Result<Self> {
        let cf = cfg1(0, 1);
        Ok(Self {
            q: conv1d(m, dev, &format!("{p}.conv_q"), cf)?,
            k: conv1d(m, dev, &format!("{p}.conv_k"), cf)?,
            v: conv1d(m, dev, &format!("{p}.conv_v"), cf)?,
            o: conv1d(m, dev, &format!("{p}.conv_o"), cf)?,
            rel_k: m.get_raw(&format!("{p}.emb_rel_k"), dev)?,
            rel_v: m.get_raw(&format!("{p}.emb_rel_v"), dev)?,
            n_heads: c.n_heads,
            head_dim: c.hidden / c.n_heads,
            window: c.window,
        })
    }
    /// Slice/pad the [1,2w+1,d] relative embeddings to [1,2t-1,d] for sequence length t.
    fn get_rel(&self, emb: &Tensor, t: usize) -> Result<Tensor> {
        let w = self.window;
        let pad = (t as isize - (w as isize + 1)).max(0) as usize;
        let start = ((w as isize + 1) - t as isize).max(0) as usize;
        let padded = if pad > 0 {
            emb.pad_with_zeros(1, pad, pad)?
        } else {
            emb.clone()
        };
        padded.narrow(1, start, 2 * t - 1)
    }
    /// x[h,t,2t-1] -> [h,t,t] (relative->absolute position).
    fn rel_to_abs(x: &Tensor, h: usize, t: usize) -> Result<Tensor> {
        let x = x.pad_with_zeros(2, 0, 1)?; // [h,t,2t]
        let x = x.reshape((h, t * 2 * t))?.pad_with_zeros(1, 0, t - 1)?; // [h, t*2t + t-1]
        x.reshape((h, t + 1, 2 * t - 1))?
            .narrow(1, 0, t)?
            .narrow(2, t - 1, t)
    }
    /// x[h,t,t] -> [h,t,2t-1] (absolute->relative position).
    fn abs_to_rel(x: &Tensor, h: usize, t: usize) -> Result<Tensor> {
        let x = x.pad_with_zeros(2, 0, t - 1)?; // [h,t,2t-1]
        let x = x.reshape((h, t * (2 * t - 1)))?.pad_with_zeros(1, t, 0)?;
        x.reshape((h, t, 2 * t))?.narrow(2, 1, 2 * t - 1)
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dims()[2];
        let (h, d) = (self.n_heads, self.head_dim);
        let scale = (d as f32).powf(-0.5);
        // [1,C,T] -> [h, t, d]
        let to_heads = |c: &Conv1d| -> Result<Tensor> {
            c.forward(x)?.reshape((h, d, t))?.transpose(1, 2) // [h,t,d]
        };
        let q = to_heads(&self.q)?;
        let k = to_heads(&self.k)?;
        let v = to_heads(&self.v)?;
        // content scores [h,t,t]
        let qs = q.affine(scale, 0.0)?;
        let mut scores = qs.matmul(&k.transpose(1, 2)?)?; // [h,t,t]
                                                          // relative-position scores
        let rk = self.get_rel(&self.rel_k, t)?.reshape((2 * t - 1, d))?; // [2t-1,d]
        let rel_logits = qs.matmul(
            &rk.transpose(0, 1)?
                .reshape((1, d, 2 * t - 1))?
                .broadcast_as((h, d, 2 * t - 1))?,
        )?; // [h,t,2t-1]
        scores = scores.add(&Self::rel_to_abs(&rel_logits, h, t)?)?;
        let attn = scores.softmax_last_dim()?; // [h,t,t]
        let mut out = attn.matmul(&v)?; // [h,t,d]
                                        // relative values
        let rel_w = Self::abs_to_rel(&attn, h, t)?; // [h,t,2t-1]
        let rv = self.get_rel(&self.rel_v, t)?.reshape((2 * t - 1, d))?;
        out = out.add(&rel_w.matmul(&rv.reshape((1, 2 * t - 1, d))?.broadcast_as((
            h,
            2 * t - 1,
            d,
        ))?)?)?;
        let out = out.transpose(1, 2)?.reshape((1, h * d, t))?; // [1,C,T]
        self.o.forward(&out)
    }
}

struct Ffn {
    c1: Conv1d,
    c2: Conv1d,
}
impl Ffn {
    fn load(m: &OnnxModel, dev: &Device, p: &str) -> Result<Self> {
        Ok(Self {
            c1: conv1d(m, dev, &format!("{p}.conv_1"), cfg1(1, 1))?, // k3 pad1
            c2: conv1d(m, dev, &format!("{p}.conv_2"), cfg1(1, 1))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.c2.forward(&self.c1.forward(x)?.relu()?)
    }
}

struct EncLayer {
    attn: RelAttention,
    n1g: Tensor,
    n1b: Tensor,
    ffn: Ffn,
    n2g: Tensor,
    n2b: Tensor,
}

struct TextEncoder {
    emb: Tensor,
    scale: f32,
    layers: Vec<EncLayer>,
    proj: Conv1d,
}
impl TextEncoder {
    fn load(m: &OnnxModel, dev: &Device, c: &PiperConfig) -> Result<Self> {
        let (en, _) = m
            .find_by_shape(&[c.n_symbols, c.hidden])
            .ok_or_else(|| err("piper: no phoneme embedding"))?;
        let emb = m.get_raw(en, dev)?;
        let mut layers = Vec::new();
        for i in 0..c.n_enc_layers {
            layers.push(EncLayer {
                attn: RelAttention::load(m, dev, &format!("enc_p.encoder.attn_layers.{i}"), c)?,
                n1g: m.get_raw(&format!("enc_p.encoder.norm_layers_1.{i}.gamma"), dev)?,
                n1b: m.get_raw(&format!("enc_p.encoder.norm_layers_1.{i}.beta"), dev)?,
                ffn: Ffn::load(m, dev, &format!("enc_p.encoder.ffn_layers.{i}"))?,
                n2g: m.get_raw(&format!("enc_p.encoder.norm_layers_2.{i}.gamma"), dev)?,
                n2b: m.get_raw(&format!("enc_p.encoder.norm_layers_2.{i}.beta"), dev)?,
            });
        }
        Ok(Self {
            emb,
            scale: (c.hidden as f32).sqrt(),
            layers,
            proj: conv1d(m, dev, "enc_p.proj", cfg1(0, 1))?,
        })
    }
    /// ids[T] -> (x_hidden[1,192,T], m_p[1,192,T], logs_p[1,192,T]). x_hidden feeds the duration predictor.
    fn forward(&self, ids: &Tensor, hidden: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let t = ids.dims()[0];
        let e = self.emb.index_select(ids, 0)?; // [T,192]
        let mut x = e
            .affine(self.scale, 0.0)?
            .reshape((1, t, hidden))?
            .transpose(1, 2)?; // [1,192,T]
        for l in &self.layers {
            let y = l.attn.forward(&x)?;
            x = layer_norm(&x.add(&y)?, &l.n1g, &l.n1b)?;
            let y = l.ffn.forward(&x)?;
            x = layer_norm(&x.add(&y)?, &l.n2g, &l.n2b)?;
        }
        let stats = self.proj.forward(&x)?; // [1,384,T]
        let m_p = stats.narrow(1, 0, hidden)?;
        let logs_p = stats.narrow(1, hidden, hidden)?;
        Ok((x, m_p, logs_p))
    }
}

// -- residual-coupling flow (`flow`) - reverse direction -----------------------------------------
struct Wn {
    in_layers: Vec<Conv1d>,
    res_skip: Vec<Conv1d>,
    hidden: usize,
}
impl Wn {
    fn load(
        m: &OnnxModel,
        dev: &Device,
        p: &str,
        hidden: usize,
        n_layers: usize,
        k: usize,
    ) -> Result<Self> {
        let pad = (k - 1) / 2;
        let mut in_layers = Vec::new();
        let mut res_skip = Vec::new();
        for i in 0..n_layers {
            in_layers.push(conv1d(m, dev, &format!("{p}.in_layers.{i}"), cfg1(pad, 1))?);
            res_skip.push(conv1d(
                m,
                dev,
                &format!("{p}.res_skip_layers.{i}"),
                cfg1(0, 1),
            )?);
        }
        Ok(Self {
            in_layers,
            res_skip,
            hidden,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        let mut out: Option<Tensor> = None;
        let n = self.in_layers.len();
        let hid = self.hidden;
        for i in 0..n {
            let xi = self.in_layers[i].forward(&x)?; // [1,2h,T]
            let acts = xi
                .narrow(1, 0, hid)?
                .tanh()?
                .mul(&xi.narrow(1, hid, hid)?.sigmoid()?)?;
            let rs = self.res_skip[i].forward(&acts)?;
            if i < n - 1 {
                x = x.add(&rs.narrow(1, 0, hid)?)?;
                let skip = rs.narrow(1, hid, hid)?;
                out = Some(match out {
                    Some(o) => o.add(&skip)?,
                    None => skip,
                });
            } else {
                out = Some(match out {
                    Some(o) => o.add(&rs)?,
                    None => rs,
                });
            }
        }
        Ok(out.unwrap())
    }
}

struct Coupling {
    pre: Conv1d,
    wn: Wn,
    post: Conv1d,
    half: usize,
}
impl Coupling {
    fn load(m: &OnnxModel, dev: &Device, p: &str, c: &PiperConfig) -> Result<Self> {
        let half = c.inter / 2;
        Ok(Self {
            pre: conv1d(m, dev, &format!("{p}.pre"), cfg1(0, 1))?,
            wn: Wn::load(m, dev, &format!("{p}.enc"), c.inter, 4, 5)?,
            post: conv1d(m, dev, &format!("{p}.post"), cfg1(0, 1))?,
            half,
        })
    }
    /// reverse, mean-only: x1 ← x1 - m(x0).
    fn forward_rev(&self, x: &Tensor) -> Result<Tensor> {
        let x0 = x.narrow(1, 0, self.half)?;
        let x1 = x.narrow(1, self.half, self.half)?;
        let h = self.wn.forward(&self.pre.forward(&x0)?)?;
        let m = self.post.forward(&h)?; // [1,half,T] (mean)
        let x1 = x1.add(&m.affine(-1.0, 0.0)?)?;
        Tensor::cat(&[&x0, &x1], 1)
    }
}

struct Flow {
    couplings: Vec<Coupling>,
    flip_idx: Tensor,
}
impl Flow {
    fn load(m: &OnnxModel, dev: &Device, c: &PiperConfig) -> Result<Self> {
        let mut couplings = Vec::new();
        for i in (0..8).step_by(2) {
            couplings.push(Coupling::load(m, dev, &format!("flow.flows.{i}"), c)?);
        }
        let idx: Vec<u32> = (0..c.inter as u32).rev().collect();
        Ok(Self {
            couplings,
            flip_idx: Tensor::from_vec_u32(idx, (c.inter,))?.to_device(dev)?,
        })
    }
    fn flip(&self, x: &Tensor) -> Result<Tensor> {
        x.index_select(&self.flip_idx, 1)
    }
    /// reverse: iterate [.., Flip, Coupling, ..] in reverse order.
    fn forward_rev(&self, z: &Tensor) -> Result<Tensor> {
        let mut x = z.clone();
        for c in self.couplings.iter().rev() {
            x = self.flip(&x)?; // the Flip that follows each coupling (its own inverse)
            x = c.forward_rev(&x)?;
        }
        Ok(x)
    }
}

// -- stochastic duration predictor (`dp`) - reverse ---------------------------------------------
fn cfgg(pad: usize, dil: usize, groups: usize) -> Conv1dConfig {
    Conv1dConfig {
        padding: pad,
        stride: 1,
        dilation: dil,
        groups,
    }
}

/// Dilated depth-separable conv stack (3 layers), GELU, optional additive conditioning `g`.
struct DdsConv {
    sep: Vec<Conv1d>,
    c1x1: Vec<Conv1d>,
    n1: Vec<(Tensor, Tensor)>,
    n2: Vec<(Tensor, Tensor)>,
}
impl DdsConv {
    fn load(m: &OnnxModel, dev: &Device, p: &str, ch: usize) -> Result<Self> {
        let (mut sep, mut c1x1, mut n1, mut n2) = (vec![], vec![], vec![], vec![]);
        for i in 0..3 {
            let dil = 3usize.pow(i as u32);
            sep.push(conv1d(
                m,
                dev,
                &format!("{p}.convs_sep.{i}"),
                cfgg((3 - 1) * dil / 2, dil, ch),
            )?);
            c1x1.push(conv1d(m, dev, &format!("{p}.convs_1x1.{i}"), cfg1(0, 1))?);
            n1.push((
                m.get_raw(&format!("{p}.norms_1.{i}.gamma"), dev)?,
                m.get_raw(&format!("{p}.norms_1.{i}.beta"), dev)?,
            ));
            n2.push((
                m.get_raw(&format!("{p}.norms_2.{i}.gamma"), dev)?,
                m.get_raw(&format!("{p}.norms_2.{i}.beta"), dev)?,
            ));
        }
        Ok(Self { sep, c1x1, n1, n2 })
    }
    fn forward(&self, x: &Tensor, g: Option<&Tensor>) -> Result<Tensor> {
        let mut x = match g {
            Some(g) => x.add(g)?,
            None => x.clone(),
        };
        for i in 0..3 {
            let y = self.sep[i].forward(&x)?;
            let y = layer_norm(&y, &self.n1[i].0, &self.n1[i].1)?.gelu_erf()?;
            let y = self.c1x1[i].forward(&y)?;
            let y = layer_norm(&y, &self.n2[i].0, &self.n2[i].1)?.gelu_erf()?;
            x = x.add(&y)?;
        }
        Ok(x)
    }
}

const NUM_BINS: usize = 10;
const TAIL: f32 = 5.0;
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}
/// Inverse of one piecewise rational-quadratic spline element (linear tails) - VITS `transforms`.
fn spline_inverse(y: f32, uw: &[f32], uh: &[f32], ud: &[f32]) -> f32 {
    if !(-TAIL..=TAIL).contains(&y) {
        return y;
    } // linear tail = identity
    let (mbw, mbh, md) = (1e-3f32, 1e-3f32, 1e-3f32);
    let nb = NUM_BINS;
    let softmax = |u: &[f32]| {
        let mx = u.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f32> = u.iter().map(|v| (v - mx).exp()).collect();
        let s: f32 = e.iter().sum();
        e.iter().map(|v| v / s).collect::<Vec<_>>()
    };
    let w: Vec<f32> = softmax(uw)
        .iter()
        .map(|v| mbw + (1.0 - mbw * nb as f32) * v)
        .collect();
    let h: Vec<f32> = softmax(uh)
        .iter()
        .map(|v| mbh + (1.0 - mbh * nb as f32) * v)
        .collect();
    // cum* in domain [-TAIL, TAIL]
    let cum = |vals: &[f32]| {
        let mut c = vec![-TAIL];
        let mut a = 0.0;
        for &v in vals {
            a += v;
            c.push(-TAIL + 2.0 * TAIL * a);
        }
        *c.last_mut().unwrap() = TAIL;
        c
    };
    let cumw = cum(&w); // nb+1
    let cumh = cum(&h);
    let bw: Vec<f32> = (0..nb).map(|i| cumw[i + 1] - cumw[i]).collect();
    let bh: Vec<f32> = (0..nb).map(|i| cumh[i + 1] - cumh[i]).collect();
    // derivatives: nb+1, ends = 1.0 (constant), middle = md + softplus(ud)
    let mut der = vec![1.0f32; nb + 1];
    for i in 0..nb - 1 {
        der[i + 1] = md + softplus(ud[i]);
    }
    // find bin by output value y in cumh
    let mut bin = 0;
    while bin + 1 < cumh.len() && cumh[bin + 1] <= y {
        bin += 1;
    }
    if bin >= nb {
        bin = nb - 1;
    }
    let (icw, ibw, ich, ih) = (cumw[bin], bw[bin], cumh[bin], bh[bin]);
    let delta = ih / ibw;
    let (d0, d1) = (der[bin], der[bin + 1]);
    let dy = y - ich;
    let a = dy * (d0 + d1 - 2.0 * delta) + ih * (delta - d0);
    let b = ih * d0 - dy * (d0 + d1 - 2.0 * delta);
    let c = -delta * dy;
    let disc = (b * b - 4.0 * a * c).max(0.0);
    let root = (2.0 * c) / (-b - disc.sqrt());
    root * ibw + icw
}

struct ConvFlow {
    pre: Conv1d,
    convs: DdsConv,
    proj: Conv1d,
    filter: usize,
}
impl ConvFlow {
    fn load(m: &OnnxModel, dev: &Device, p: &str, c: &PiperConfig) -> Result<Self> {
        // The SDP's ConvFlow operates at hidden_channels (192), NOT the encoder FFN filter (768).
        Ok(Self {
            pre: conv1d(m, dev, &format!("{p}.pre"), cfg1(0, 1))?,
            convs: DdsConv::load(m, dev, &format!("{p}.convs"), c.hidden)?,
            proj: conv1d(m, dev, &format!("{p}.proj"), cfg1(0, 1))?,
            filter: c.hidden,
        })
    }
    fn forward_rev(&self, z: &Tensor, g: &Tensor, dev: &Device) -> Result<Tensor> {
        let t = z.dims()[2];
        let z0 = z.narrow(1, 0, 1)?;
        let z1 = z.narrow(1, 1, 1)?.reshape((t,))?.to_vec_f32();
        let h = self
            .proj
            .forward(&self.convs.forward(&self.pre.forward(&z0)?, Some(g))?)?; // [1,29,T]
        let hv = h.reshape((3 * NUM_BINS - 1, t))?.to_vec_f32(); // [29*t], channel-major
        let sf = (self.filter as f32).sqrt();
        let mut out = vec![0f32; t];
        for pos in 0..t {
            let uw: Vec<f32> = (0..NUM_BINS).map(|b| hv[b * t + pos] / sf).collect();
            let uh: Vec<f32> = (0..NUM_BINS)
                .map(|b| hv[(NUM_BINS + b) * t + pos] / sf)
                .collect();
            let ud: Vec<f32> = (0..NUM_BINS - 1)
                .map(|b| hv[(2 * NUM_BINS + b) * t + pos])
                .collect();
            out[pos] = spline_inverse(z1[pos], &uw, &uh, &ud);
        }
        let z1n = Tensor::from_vec_f32(out, (1, 1, t))?.to_device(dev)?;
        Tensor::cat(&[&z0, &z1n], 1)
    }
}

struct Sdp {
    pre: Conv1d,
    convs: DdsConv,
    proj: Conv1d,
    ea_m: Tensor,
    ea_logs: Option<Tensor>,
    flows: Vec<ConvFlow>,
    flip_idx: Tensor,
}
impl Sdp {
    fn load(m: &OnnxModel, dev: &Device, c: &PiperConfig) -> Result<Self> {
        let ea_logs = if m.contains("dp.flows.0.logs") {
            Some(m.get_raw("dp.flows.0.logs", dev)?)
        } else {
            None
        };
        // reverse-inference applies ConvFlows 7,5,3 (flow.1 is the dropped "useless vflow").
        let mut flows = vec![];
        for i in [7, 5, 3] {
            flows.push(ConvFlow::load(m, dev, &format!("dp.flows.{i}"), c)?);
        }
        Ok(Self {
            pre: conv1d(m, dev, "dp.pre", cfg1(0, 1))?,
            convs: DdsConv::load(m, dev, "dp.convs", c.hidden)?,
            proj: conv1d(m, dev, "dp.proj", cfg1(0, 1))?,
            ea_m: m.get_raw("dp.flows.0.m", dev)?,
            ea_logs,
            flows,
            flip_idx: Tensor::from_vec_u32(vec![1, 0], (2,))?.to_device(dev)?,
        })
    }
    /// x_hidden[1,192,T] -> durations (frames per phoneme).
    fn durations(
        &self,
        x: &Tensor,
        noise_w: f32,
        length_scale: f32,
        seed: u64,
        dev: &Device,
    ) -> Result<Vec<usize>> {
        let t = x.dims()[2];
        let g = self
            .proj
            .forward(&self.convs.forward(&self.pre.forward(x)?, None)?)?; // [1,192,T] condition
        let mut z = randn_seeded((1, 2, t), seed, dev)?.affine(noise_w, 0.0)?;
        for f in &self.flows {
            z = z.index_select(&self.flip_idx, 1)?; // Flip (2 channels)
            z = f.forward_rev(&z, &g, dev)?;
        }
        z = z.index_select(&self.flip_idx, 1)?; // final Flip
                                                // ElementwiseAffine reverse: z = (z - m) * exp(-logs)
        let z = z.broadcast_add(&self.ea_m.reshape((1, 2, 1))?.affine(-1.0, 0.0)?)?;
        let z = match &self.ea_logs {
            Some(l) => z.broadcast_mul(&l.reshape((1, 2, 1))?.affine(-1.0, 0.0)?.exp()?)?,
            None => z,
        };
        let logw = z.narrow(1, 0, 1)?.reshape((t,))?.to_vec_f32();
        Ok(logw
            .iter()
            .map(|&lw| ((lw.exp() * length_scale).ceil().max(1.0)) as usize)
            .collect())
    }
}

// -- Full model ------------------------------------------------------------

pub struct PiperModel {
    pub cfg: PiperConfig,
    enc: TextEncoder,
    dp: Sdp,
    flow: Flow,
    dec: Decoder,
    pub device: Device,
}

impl PiperModel {
    pub fn load(onnx_path: &std::path::Path, dev: &Device) -> Result<Self> {
        let onnx = OnnxModel::read(onnx_path)?;
        let cfg = PiperConfig::default();
        let enc = TextEncoder::load(&onnx, dev, &cfg)?;
        let dp = Sdp::load(&onnx, dev, &cfg)?;
        let flow = Flow::load(&onnx, dev, &cfg)?;
        let dec = Decoder::load(&onnx, dev, &cfg)?;
        Ok(Self {
            cfg,
            enc,
            dp,
            flow,
            dec,
            device: dev.clone(),
        })
    }

    /// Synthesize a waveform from phoneme ids. Scales mirror Piper's `infer` (noise_scale 0.667,
    /// noise_w 0.8, length_scale 1.0). `durations` overrides the stochastic predictor when `Some`.
    pub fn synthesize(
        &self,
        ids: &[u32],
        durations: Option<&[usize]>,
        seed: u64,
        noise_scale: f32,
        noise_w: f32,
        length_scale: f32,
    ) -> Result<Vec<f32>> {
        let dev = &self.device;
        let hidden = self.cfg.hidden;
        let ids_t = Tensor::from_vec_u32(ids.to_vec(), (ids.len(),))?.to_device(dev)?;
        let (x_hidden, m_p, logs_p) = self.enc.forward(&ids_t, hidden)?; // [1,192,T]
        let t = ids.len();
        let dur: Vec<usize> = match durations {
            Some(d) => d.to_vec(),
            None => self
                .dp
                .durations(&x_hidden, noise_w, length_scale, seed ^ 0x9e37, dev)?,
        };
        // length-regulate: expand each phoneme column `dur[i]` times -> [1,192,Ty] (CPU-side).
        let mp = m_p.reshape((hidden, t))?.to_vec_f32();
        let lp = logs_p.reshape((hidden, t))?.to_vec_f32();
        let ty: usize = dur.iter().sum();
        let (mut mpe, mut lpe) = (vec![0f32; hidden * ty], vec![0f32; hidden * ty]);
        for ch in 0..hidden {
            let mut col = 0;
            for i in 0..t {
                for _ in 0..dur[i] {
                    mpe[ch * ty + col] = mp[ch * t + i];
                    lpe[ch * ty + col] = lp[ch * t + i];
                    col += 1;
                }
            }
        }
        let m_e = Tensor::from_vec_f32(mpe, (1, hidden, ty))?.to_device(dev)?;
        let l_e = Tensor::from_vec_f32(lpe, (1, hidden, ty))?.to_device(dev)?;
        // z_p = m + eps.exp(logs).noise_scale  (deterministic eps via seed)
        let eps = randn_seeded((1, hidden, ty), seed, dev)?;
        let z_p = m_e.add(&eps.mul(&l_e.exp()?)?.affine(noise_scale, 0.0)?)?;
        let z = self.flow.forward_rev(&z_p)?;
        let wav = self.dec.forward(&z)?;
        Ok(wav.reshape((wav.elem_count(),))?.to_vec_f32())
    }
}

// -- High-level voice: text -> phonemes (espeak) -> ids -> waveform ---------------------------------
use std::collections::HashMap;

pub struct PiperVoice {
    model: PiperModel,
    /// phoneme symbol -> id(s) (from the `.onnx.json` `phoneme_id_map`).
    pmap: HashMap<String, Vec<u32>>,
    espeak_voice: String,
    pub sample_rate: u32,
}

impl PiperVoice {
    pub fn load(onnx_path: &std::path::Path, dev: &Device) -> Result<Self> {
        let model = PiperModel::load(onnx_path, dev)?;
        let json_path = format!("{}.json", onnx_path.display());
        let cfg: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&json_path)
                .map_err(|e| err(format!("piper json {json_path}: {e}")))?,
        )
        .map_err(|e| err(format!("piper json parse: {e}")))?;
        let mut pmap = HashMap::new();
        if let Some(m) = cfg["phoneme_id_map"].as_object() {
            for (k, v) in m {
                let ids: Vec<u32> = v
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_u64().map(|n| n as u32))
                            .collect()
                    })
                    .unwrap_or_default();
                pmap.insert(k.clone(), ids);
            }
        }
        let espeak_voice = cfg["espeak"]["voice"].as_str().unwrap_or("fr").to_string();
        let sample_rate = cfg["audio"]["sample_rate"].as_u64().unwrap_or(22050) as u32;
        Ok(Self {
            model,
            pmap,
            espeak_voice,
            sample_rate,
        })
    }

    /// Text -> IPA phoneme symbols. Prefers espeak-ng when it is installed; otherwise the
    /// built-in FRENCH grapheme-to-phoneme, which is the only one written here.
    ///
    /// That fallback is correct for a French voice and WRONG for every other one, and it
    /// used to run for all of them in silence: install an English Piper voice on a
    /// machine without espeak-ng and English text was read with French letter rules -
    /// no error, no log line, just audio that sounds like a French speaker guessing.
    ///
    /// It now refuses instead, naming what would fix it. A voice that cannot be
    /// phonemised is not a degraded voice, it is the wrong voice, and a caller who is
    /// told can install espeak-ng or pick a French voice in a minute. Guessing costs
    /// them the time to work out why their English sounds like that.
    fn phonemize(&self, text: &str) -> Result<Vec<String>> {
        if let Ok(out) = std::process::Command::new("espeak-ng")
            .args(["-q", "-v", &self.espeak_voice, "--ipa=3", text])
            .output()
        {
            if out.status.success() {
                let ipa = String::from_utf8_lossy(&out.stdout);
                return Ok(ipa
                    .trim()
                    .chars()
                    .map(|c| c.to_string())
                    .filter(|c| self.pmap.contains_key(c) || c == " ")
                    .collect());
            }
        }
        if !voice_is_french(&self.espeak_voice) {
            return Err(crate::tensor::Error(format!(
                "the voice '{}' needs espeak-ng to be phonemised, and it is not installed.                  Install espeak-ng, or use a French Piper voice - the built-in fallback                  only knows French letter rules and would read this with them",
                self.espeak_voice
            )));
        }
        // Native fallback (full-Rust): keep only symbols the model knows.
        Ok(french_g2p(text)
            .into_iter()
            .filter(|c| self.pmap.contains_key(c) || c == " ")
            .collect())
    }

    /// Piper tokenization: [BOS, PAD, p, PAD, ..., EOS], BOS=`^`, EOS=`$`, PAD=`_`.
    fn tokenize(&self, phonemes: &[String]) -> Vec<u32> {
        let id = |s: &str| self.pmap.get(s).and_then(|v| v.first().copied());
        let (bos, eos, pad) = (
            id("^").unwrap_or(1),
            id("$").unwrap_or(2),
            id("_").unwrap_or(0),
        );
        let mut ids = vec![bos, pad];
        for p in phonemes {
            if let Some(v) = self.pmap.get(p) {
                ids.extend(v);
                ids.push(pad);
            }
        }
        ids.push(eos);
        ids
    }

    /// Full text->speech: returns f32 PCM at `sample_rate`, peak-normalized to a usable level.
    pub fn tts(&self, text: &str, o: SynthOpts) -> Result<Vec<f32>> {
        let phonemes = self.phonemize(text)?;
        if phonemes.is_empty() {
            return Err(err("piper: no phonemes for input"));
        }
        let ids = self.tokenize(&phonemes);
        Ok(normalize_peak(
            self.model
                .synthesize(&ids, None, o.seed, o.noise_scale, o.noise_w, o.length_scale)?,
            0.7,
        ))
    }
}

/// Synthesis controls (mirror Piper's `infer` defaults). Higher noise_scale = more variation/noise.
#[derive(Clone, Copy)]
pub struct SynthOpts {
    pub seed: u64,
    pub noise_scale: f32,
    pub noise_w: f32,
    pub length_scale: f32,
}
impl Default for SynthOpts {
    fn default() -> Self {
        Self {
            seed: 0,
            noise_scale: 0.667,
            noise_w: 0.8,
            length_scale: 1.0,
        }
    }
}

/// Native rule-based French grapheme->IPA converter (no external G2P). Approximate (espeak is
/// better) but dependency-free; nasal vowels are emitted as vowel + combining tilde `̃`.
/// Is this espeak voice a French one? The built-in grapheme rules only fit those.
///
/// espeak names a voice by language first - `fr`, `fr-fr`, `fr_FR` - so the prefix is
/// what decides, and an unknown or empty name is treated as NOT French: the fallback is
/// the thing being guarded, so the doubtful case must not reach it.
fn voice_is_french(espeak_voice: &str) -> bool {
    let v = espeak_voice.trim().to_lowercase();
    v == "fr" || v.starts_with("fr-") || v.starts_with("fr_")
}

fn french_g2p(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split(|c: char| c.is_whitespace() || (c.is_ascii_punctuation() && c != '\'')) {
        if word.is_empty() {
            continue;
        }
        out.extend(french_word(&word.to_lowercase()));
        out.push(" ".into());
    }
    out
}

fn french_word(w: &str) -> Vec<String> {
    let c: Vec<char> = w.chars().collect();
    let n = c.len();
    let is_vowel = |ch: char| "aeiouyàâéèêëîïôûüùœæ".contains(ch);
    let mut o: Vec<String> = Vec::new();
    let mut i = 0;
    let push = |o: &mut Vec<String>, s: &str| {
        for p in s.split(' ') {
            if !p.is_empty() {
                o.push(p.into());
            }
        }
    };
    while i < n {
        let s3: String = c[i..(i + 3).min(n)].iter().collect();
        let s2: String = c[i..(i + 2).min(n)].iter().collect();
        let ch = c[i];
        let next = c.get(i + 1).copied();
        let nasal_follows = |j: usize| {
            matches!(c.get(j), Some(&x) if x == 'n' || x == 'm')
                && !matches!(c.get(j + 1), Some(&y) if is_vowel(y) || y == 'n' || y == 'm')
        };
        // trigraphs
        if ["eau"].contains(&s3.as_str()) {
            push(&mut o, "o");
            i += 3;
            continue;
        }
        if ["ain", "aim", "ein", "eim"].contains(&s3.as_str()) {
            push(&mut o, "ɛ ̃");
            i += 3;
            continue;
        }
        if s3 == "oin" {
            push(&mut o, "w ɛ ̃");
            i += 3;
            continue;
        }
        if s3 == "ien" && !matches!(c.get(i + 3), Some(&x) if is_vowel(x)) {
            push(&mut o, "j ɛ ̃");
            i += 3;
            continue;
        }
        if s3 == "ion" {
            push(&mut o, "j ɔ ̃");
            i += 3;
            continue;
        }
        // yod (y-glide) clusters: -ouille/-aille/-eille/-ille -> vowel + j (travaille->tʁavaj, fille->fij)
        let s5: String = c[i..(i + 5).min(n)].iter().collect();
        let s4: String = c[i..(i + 4).min(n)].iter().collect();
        if s5 == "ouill" {
            push(&mut o, "u j");
            i += 5;
            continue;
        }
        if s4 == "aill" {
            push(&mut o, "a j");
            i += 4;
            continue;
        }
        if s4 == "eill" {
            push(&mut o, "ɛ j");
            i += 4;
            continue;
        }
        if s3 == "ill" {
            push(&mut o, "i j");
            i += 3;
            continue;
        }
        // digraph vowels
        if s2 == "ou" {
            push(&mut o, "u");
            i += 2;
            continue;
        }
        if s2 == "au" {
            push(&mut o, "o");
            i += 2;
            continue;
        }
        if s2 == "ai" || s2 == "ei" || s2 == "ès" {
            push(&mut o, "ɛ");
            i += 2;
            continue;
        }
        if s2 == "eu" || s2 == "œu" || s2 == "oe" {
            push(&mut o, "ø");
            i += 2;
            continue;
        }
        if s2 == "oi" {
            push(&mut o, "w a");
            i += 2;
            continue;
        }
        // nasal vowels (vowel + n/m not before vowel)
        if "ao".contains(ch) && nasal_follows(i + 1) && (ch == 'a') {
            push(&mut o, "ɑ ̃");
            i += 2;
            continue;
        }
        if ch == 'o' && nasal_follows(i + 1) {
            push(&mut o, "ɔ ̃");
            i += 2;
            continue;
        }
        if ch == 'e' && nasal_follows(i + 1) {
            push(&mut o, "ɑ ̃");
            i += 2;
            continue;
        }
        if (ch == 'i' || ch == 'y') && nasal_follows(i + 1) {
            push(&mut o, "ɛ ̃");
            i += 2;
            continue;
        }
        if ch == 'u' && nasal_follows(i + 1) {
            push(&mut o, "œ ̃");
            i += 2;
            continue;
        }
        // consonant digraphs
        if s2 == "ch" {
            push(&mut o, "ʃ");
            i += 2;
            continue;
        }
        if s2 == "gn" {
            push(&mut o, "ɲ");
            i += 2;
            continue;
        }
        if s2 == "ph" {
            push(&mut o, "f");
            i += 2;
            continue;
        }
        if s2 == "qu" {
            push(&mut o, "k");
            i += 2;
            continue;
        }
        if s2 == "th" {
            push(&mut o, "t");
            i += 2;
            continue;
        }
        if s2 == "ss" {
            push(&mut o, "s");
            i += 2;
            continue;
        }
        // collapse a doubled consonant letter - pronounced once (pomme->pɔm, cette->sɛt, sonne->sɔn)
        if next == Some(ch) && ch.is_alphabetic() && !is_vowel(ch) {
            i += 1;
            continue;
        }
        // single graphemes
        let sym: &str = match ch {
            'a' | 'à' => "a",
            'â' => "ɑ",
            'é' => "e",
            'è' | 'ê' | 'ë' => "ɛ",
            'e' => {
                // 'e' is silent when a prior vowel exists AND only silent finals follow:
                // porte->pɔʁt (final e), pommes->pɔm (e before plural -s). else schwa (le->lə).
                let tail_silent = i + 1 < n && c[i + 1..].iter().all(|&x| "stdxz".contains(x));
                if n > 1 && c[..i].iter().any(|&x| is_vowel(x)) && (i == n - 1 || tail_silent) {
                    i += 1;
                    continue;
                }
                "ə"
            }
            'i' | 'î' | 'ï' | 'y' => "i",
            'o' | 'ô' => "o",
            'u' | 'û' | 'ü' => "y",
            'ù' => "u",
            'œ' => "œ",
            'c' => {
                if matches!(
                    next,
                    Some('e') | Some('i') | Some('y') | Some('é') | Some('è') | Some('\'')
                ) {
                    "s"
                } else {
                    "k"
                }
            }
            'ç' => "s",
            'g' => {
                if matches!(next, Some('e') | Some('i') | Some('y')) {
                    "ʒ"
                } else {
                    "g"
                }
            }
            'j' => "ʒ",
            'h' => {
                i += 1;
                continue;
            }
            's' => {
                if i > 0 && i + 1 < n && is_vowel(c[i - 1]) && is_vowel(c[i + 1]) {
                    "z"
                } else {
                    "s"
                }
            }
            'x' => "k s",
            'w' => "w",
            'r' => "ʁ",
            'b' => "b",
            'd' => "d",
            'f' => "f",
            'k' => "k",
            'l' => "l",
            'm' => "m",
            'n' => "n",
            'p' => "p",
            't' => "t",
            'v' => "v",
            'z' => "z",
            'q' => "k",
            '\'' => {
                i += 1;
                continue;
            }
            _ => {
                i += 1;
                continue;
            }
        };
        // drop a typically-silent final consonant (s, t, d, x, z) at word end (dans->dɑ̃, petit->pəti)
        if i == n - 1 && i > 0 && "stdxz".contains(ch) {
            i += 1;
            continue;
        }
        push(&mut o, sym);
        i += 1;
    }
    o
}

/// Peak-normalize PCM to `target` (no-op on silence) - the VITS output level varies per utterance.
fn normalize_peak(mut wav: Vec<f32>, target: f32) -> Vec<f32> {
    let peak = wav.iter().fold(0f32, |a, &x| a.max(x.abs()));
    if peak > 1e-4 {
        let g = target / peak;
        for s in &mut wav {
            *s *= g;
        }
    }
    wav
}

/// Deterministic N(0,1) tensor (LCG + Box-Muller) - reproducible synthesis without a global RNG.
fn randn_seeded(shape: (usize, usize, usize), seed: u64, dev: &Device) -> Result<Tensor> {
    let n = shape.0 * shape.1 * shape.2;
    let mut s = seed.max(1);
    // Uniform in [0,1) using only the TOP 24 bits - exact in f32 (mantissa = 24 bits). Taking 31
    // bits and casting to f32 quantizes the high range to steps of 128, which turns the injected
    // Gaussian noise gritty/quantized (audible as broadband "bruit" over a correct voice).
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as u32) as f32 / 16_777_216.0
    };
    let mut v = vec![0f32; n];
    let mut i = 0;
    while i < n {
        let (u1, u2) = (next().max(1e-7), next());
        let r = (-2.0 * u1.ln()).sqrt();
        v[i] = r * (std::f32::consts::TAU * u2).cos();
        if i + 1 < n {
            v[i + 1] = r * (std::f32::consts::TAU * u2).sin();
        }
        i += 2;
    }
    Tensor::from_vec_f32(v, shape)?.to_device(dev)
}

#[cfg(test)]
mod french_fallback_tests {
    use super::voice_is_french;

    /// The built-in grapheme rules are French, so they may only run for a French voice.
    #[test]
    fn french_voices_are_recognised() {
        for v in ["fr", "fr-fr", "fr_FR", "FR-FR", " fr "] {
            assert!(voice_is_french(v), "{v}");
        }
    }

    /// Every other voice must be refused rather than read with French letter rules -
    /// that fallback ran for all of them, silently, and English came out sounding like
    /// a French speaker guessing at it.
    #[test]
    fn other_voices_are_not() {
        for v in [
            "en-us", "en_GB", "de", "es", "it", "nl", "fra", "french", "",
        ] {
            assert!(!voice_is_french(v), "{v}");
        }
    }

    /// A name that merely STARTS with the letters is not a French voice: `fra` is a
    /// three-letter code espeak does not use for the same thing, and matching loosely
    /// would let exactly the case this guards slip through.
    #[test]
    fn a_lookalike_prefix_does_not_count() {
        assert!(!voice_is_french("fra"));
        assert!(!voice_is_french("frisian"));
    }
}
