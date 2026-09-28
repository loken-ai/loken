//! Kyutai `tts-1.6b-en_fr` - Mimi RVQ codec DECODER (component 3 of the TTS port).
//!
//! codes `[K, T]` (1 semantic + the acoustic RVQ codebooks, text codebook dropped first)
//!   -> RVQ dequant -> latent [C, T]
//!   -> upsample (depthwise convtr) -> decoder-transformer (RoPE + LayerScale)
//!   -> SEANet decoder (strided transposed convs) -> 24 kHz mono waveform.
//!
//! Every width, codebook count, layer count and stride is read from the tensors the
//! checkpoint stores. Codebook = `embedding_sum / max(cluster_usage, eps)` (Mimi EMA).

use crate::inference::model::acestep::fsq::rope_tables;
use crate::inference::model::sdpa::sdpa;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Error, Result, Tensor};

/// Mimi's transformer settings, which its reference code fixes and its checkpoint does not
/// store: the head count its attention splits the width into, and the positions the
/// sliding causal window reaches back.
pub(crate) const ATTENTION_HEADS: usize = 8;
pub(crate) const ATTENTION_WINDOW: usize = 250;
const ROPE_THETA: f32 = 10_000.0;
const LN_EPS: f32 = 1e-5;
const VQ_EPS: f32 = 1e-5;
/// Mimi's output sample rate, fixed by the codec.
pub const SAMPLE_RATE: u32 = 24_000;

// -- shared conv/act helpers (same math as native_pocket_tts) -----------------
fn add_bias(y: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    match bias {
        None => Ok(y),
        Some(b) => {
            let out = b.dims()[0];
            y.broadcast_add(&b.reshape((1, out, 1))?)
        }
    }
}
/// Causal Conv1d: left-pad `(k-1)*dilation+1-stride`, then conv with no pad.
fn causal_conv1d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor> {
    let k = w.shape().dims3()?.2;
    let pad = (k - 1) * dilation + 1 - stride;
    let xp = x.pad_with_zeros(2, pad, 0)?;
    add_bias(xp.conv1d(w, 0, stride, dilation, groups)?, bias)
}
/// Causal ConvTranspose1d: transpose then trim `(k-stride)` from the right.
fn causal_convtr1d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    groups: usize,
) -> Result<Tensor> {
    let k = w.shape().dims3()?.2;
    let y = x.conv_transpose1d(w, 0, 0, stride, 1, groups)?;
    let l = y.shape().dims3()?.2;
    add_bias(y.narrow(2, 0, l.saturating_sub(k - stride))?, bias)
}
/// ELU(1.0) = relu(x) + (exp(min(x,0)) - 1).
fn elu(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    let min_x0 = x.affine(-1.0, 0.0)?.relu()?.affine(-1.0, 0.0)?;
    pos.add(&min_x0.exp()?.affine(1.0, -1.0)?)
}
/// max(x, c) = relu(x - c) + c - no native clamp, built from relu/affine.
fn max_scalar(x: &Tensor, c: f32) -> Result<Tensor> {
    x.affine(1.0, -c)?.relu()?.affine(1.0, c)
}

struct Conv {
    w: Tensor,
    b: Option<Tensor>,
}
impl Conv {
    /// A conv whose shape is the stored `{name}.weight`, with its bias when `bias`.
    fn load(vb: &VarBuilder, name: &str, bias: bool) -> Result<Self> {
        let shape = vb.shape_of(&format!("{name}.weight"))?;
        let w = vb.get(shape.clone(), &format!("{name}.weight"))?;
        let b = if bias {
            Some(vb.get(shape[0], &format!("{name}.bias"))?)
        } else {
            None
        };
        Ok(Self { w, b })
    }
    /// A transposed conv `[in, out, k]` with its bias over `out`.
    fn load_tr(vb: &VarBuilder, name: &str) -> Result<Self> {
        let shape = vb.shape_of(&format!("{name}.weight"))?;
        Ok(Self {
            b: Some(vb.get(shape[1], &format!("{name}.bias"))?),
            w: vb.get(shape, &format!("{name}.weight"))?,
        })
    }
    /// The stride of a SEANet resampling conv, whose kernel is twice its stride.
    fn resampling_stride(&self) -> Result<usize> {
        Ok(self.w.shape().dims3()?.2 / 2)
    }
}
/// SEANet residual block: ELU -> conv(k3) -> ELU -> conv(k1) -> + input.
struct ResBlock {
    conv1: Conv,
    conv2: Conv,
}
impl ResBlock {
    fn load(vb: &VarBuilder, idx: usize) -> Result<Self> {
        let p = format!("{idx}.block");
        Ok(Self {
            conv1: Conv::load(vb, &format!("{p}.1.conv.conv"), true)?,
            conv2: Conv::load(vb, &format!("{p}.3.conv.conv"), true)?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = elu(x)?;
        let h = causal_conv1d(&h, &self.conv1.w, self.conv1.b.as_ref(), 1, 1, 1)?;
        let h = elu(&h)?;
        let h = causal_conv1d(&h, &self.conv2.w, self.conv2.b.as_ref(), 1, 1, 1)?;
        x.add(&h)
    }
}

// -- RVQ dequant (NEW for tts-1.6b) -------------------------------------------
/// One ResidualVectorQuantizer's decode: Σ_k codebook_k[codes_k] -> output_proj.
struct Rvq {
    codebooks: Vec<Tensor>, // each [entries, vq_dim], EMA-normalized
    output_proj: Tensor,    // [dim, vq_dim, 1] conv1x1
}
impl Rvq {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let n_cb = vb.count_children("vq.layers");
        let mut codebooks = Vec::with_capacity(n_cb);
        for k in 0..n_cb {
            let sum = format!("vq.layers.{k}._codebook.embedding_sum");
            let es = vb.get(vb.shape_of(&sum)?, &sum)?;
            let (entries, _) = es.shape().dims2()?;
            let cu = vb.get(entries, &format!("vq.layers.{k}._codebook.cluster_usage"))?;
            // codebook = embedding_sum / max(cluster_usage, eps)  (broadcast over vq_dim)
            let denom = max_scalar(&cu, VQ_EPS)?.reshape((entries, 1))?;
            codebooks.push(es.broadcast_div(&denom)?);
        }
        Ok(Self {
            codebooks,
            output_proj: vb.get(vb.shape_of("output_proj.weight")?, "output_proj.weight")?,
        })
    }
    fn len(&self) -> usize {
        self.codebooks.len()
    }
    /// `codes`: `[n_cb, T]` (u32 indices) -> `[dim, T]`.
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let t = codes.shape().dims2()?.1;
        // Σ over codebooks of embedding[code]  -> [T, vq_dim]
        let mut acc: Option<Tensor> = None;
        for (k, cb) in self.codebooks.iter().enumerate() {
            let idx = codes.narrow(0, k, 1)?.reshape(t)?; // [T]
            let e = cb.index_select(&idx, 0)?; // [T, vq_dim]
            acc = Some(match acc {
                None => e,
                Some(a) => a.add(&e)?,
            });
        }
        let acc = acc.ok_or_else(|| Error("mimi: a quantizer with no codebooks".into()))?;
        let vq_dim = acc.shape().dims2()?.1;
        let q = acc.transpose(0, 1)?.reshape((1, vq_dim, t))?;
        // output_proj is a conv1x1 -> [1,dim,T] -> [dim,T]
        let dim = self.output_proj.shape().dims3()?.0;
        causal_conv1d(&q, &self.output_proj, None, 1, 1, 1)?.reshape((dim, t))
    }
}

// -- decoder transformer layer (pre-norm, fused-QKV MHA + RoPE-i + LayerScale) -
struct XfLayer {
    norm1_w: Tensor,
    norm1_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    in_proj: Tensor,
    out_proj: Tensor,
    linear1: Tensor,
    linear2: Tensor,
    ls1: Tensor,
    ls2: Tensor,
    dim: usize,
}
impl XfLayer {
    fn load(vb: &VarBuilder, idx: usize) -> Result<Self> {
        let p = vb.pp(format!("layers.{idx}"));
        let dim = p.shape_of("norm1.weight")?[0];
        let ffn = p.shape_of("linear1.weight")?[0];
        Ok(Self {
            norm1_w: p.get(dim, "norm1.weight")?,
            norm1_b: p.get(dim, "norm1.bias")?,
            norm2_w: p.get(dim, "norm2.weight")?,
            norm2_b: p.get(dim, "norm2.bias")?,
            in_proj: p.get((3 * dim, dim), "self_attn.in_proj_weight")?,
            out_proj: p.get((dim, dim), "self_attn.out_proj.weight")?,
            linear1: p.get((ffn, dim), "linear1.weight")?,
            linear2: p.get((dim, ffn), "linear2.weight")?,
            ls1: p.get(dim, "layer_scale_1.scale")?,
            ls2: p.get(dim, "layer_scale_2.scale")?,
            dim,
        })
    }
    /// `x: [S, dim]` - sliding-causal RoPE attention.
    fn forward(&self, x: &Tensor, dev: &Device) -> Result<Tensor> {
        let dim = self.dim;
        let s = x.shape().dims2()?.0;
        let hd = dim / ATTENTION_HEADS;
        let h = x.layer_norm(&self.norm1_w, Some(&self.norm1_b), LN_EPS)?;
        let qkv = h.matmul_t(&self.in_proj)?;
        let q = qkv.narrow(1, 0, dim)?;
        let k = qkv.narrow(1, dim, dim)?;
        let v = qkv.narrow(1, 2 * dim, dim)?;
        let shp = |t: &Tensor| -> Result<Tensor> {
            t.reshape((s, ATTENTION_HEADS, hd))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()
        };
        let (cosv, sinv) = rope_tables(s, hd, ROPE_THETA);
        let cos = Tensor::from_vec_f32(cosv, (s, hd / 2))?.to_device(dev)?;
        let sin = Tensor::from_vec_f32(sinv, (s, hd / 2))?.to_device(dev)?;
        let q = shp(&q)?.rope_i(&cos, &sin)?;
        let k = shp(&k)?.rope_i(&cos, &sin)?;
        let v = shp(&v)?;
        let scale = 1.0 / (hd as f32).sqrt();
        // sliding causal window (a sequence shorter than the window is full causal)
        let mut data = vec![0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                if j > i || i - j >= ATTENTION_WINDOW {
                    data[i * s + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec_f32(data, (s, s))?.to_device(dev)?;
        let att = sdpa(&q, &k, &v, Some(&mask), false, scale, 1.0)?;
        let att = att
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, dim))?
            .matmul_t(&self.out_proj)?;
        let x = x.add(&att.broadcast_mul(&self.ls1.reshape((1, dim))?)?)?;
        let h = x.layer_norm(&self.norm2_w, Some(&self.norm2_b), LN_EPS)?;
        let h = h
            .matmul_t(&self.linear1)?
            .gelu_erf()?
            .matmul_t(&self.linear2)?;
        x.add(&h.broadcast_mul(&self.ls2.reshape((1, dim))?)?)
    }
}

/// Full Mimi decoder for tts-1.6b: codes -> 24 kHz waveform.
pub struct KyutaiMimiDecoder {
    rvq_first: Rvq,   // the semantic codebooks
    rvq_rest: Rvq,    // the acoustic codebooks
    upsample: Tensor, // [dim,1,k] depthwise convtr
    xf: Vec<XfLayer>,
    conv_in: Conv,
    up: Vec<(Conv, ResBlock)>, // (convtr, resblock) per SEANet stage
    conv_out: Conv,
    device: Device,
}
impl KyutaiMimiDecoder {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let q = vb.pp("quantizer");
        let rvq_first = Rvq::load(&q.pp("rvq_first"))?;
        let rvq_rest = Rvq::load(&q.pp("rvq_rest"))?;
        let up_name = "upsample.convtr.convtr.convtr.weight";
        let upsample = vb.get(vb.shape_of(up_name)?, up_name)?;
        let xt = vb.pp("decoder_transformer.transformer");
        let xf = (0..xt.count_children("layers"))
            .map(|i| XfLayer::load(&xt, i))
            .collect::<Result<Vec<_>>>()?;
        // SEANet decoder: `model.0` is the input conv, then each stage is an ELU, a
        // transposed conv and a residual block (three entries), and the output conv
        // follows the last stage's ELU.
        let dec = vb.pp("decoder.model");
        let conv_in = Conv::load(&dec, "0.conv.conv", true)?;
        let mut up = Vec::new();
        let mut idx = 2;
        while dec.contains(&format!("{idx}.convtr.convtr.weight")) {
            up.push((
                Conv::load_tr(&dec, &format!("{idx}.convtr.convtr"))?,
                ResBlock::load(&dec, idx + 1)?,
            ));
            idx += 3;
        }
        let conv_out = Conv::load(&dec, &format!("{idx}.conv.conv"), true)?;
        Ok(Self {
            rvq_first,
            rvq_rest,
            upsample,
            xf,
            conv_in,
            up,
            conv_out,
            device,
        })
    }

    /// Audio samples one code frame decodes to: the product of every upsampling stride.
    pub fn frame_samples(&self) -> Result<usize> {
        let mut n = self.upsample.shape().dims3()?.2 / 2;
        for (convtr, _) in &self.up {
            n *= convtr.resampling_stride()?;
        }
        Ok(n)
    }

    /// RVQ dequant only (for parity vs mimi_latent): `codes [K,T]` -> `[dim,T]`.
    pub fn decode_latent(&self, codes: &Tensor) -> Result<Tensor> {
        let t = codes.shape().dims2()?.1;
        let n_first = self.rvq_first.len();
        let sem = self.rvq_first.decode(&codes.narrow(0, 0, n_first)?)?;
        let rest = self
            .rvq_rest
            .decode(&codes.narrow(0, n_first, self.rvq_rest.len())?)?;
        debug_assert_eq!(t, sem.shape().dims2()?.1);
        sem.add(&rest)
    }

    /// Full decode: `codes [K,T]` (u32) -> 24 kHz mono waveform `Vec<f32>`.
    pub fn decode(&self, codes: &Tensor) -> Result<Vec<f32>> {
        let latent = self.decode_latent(codes)?; // [dim,T]
        let (dim, t) = latent.shape().dims2()?;
        let emb = latent.reshape((1, dim, t))?;
        // upsample (depthwise convtr)
        let up_stride = self.upsample.shape().dims3()?.2 / 2;
        let x = causal_convtr1d(&emb, &self.upsample, None, up_stride, dim)?;
        let s = x.shape().dims3()?.2;
        // decoder transformer at the upsampled frame rate
        let mut h = x.reshape((dim, s))?.transpose(0, 1)?.contiguous()?; // [S,dim]
        for l in &self.xf {
            h = l.forward(&h, &self.device)?;
        }
        let mut y = h.transpose(0, 1)?.reshape((1, dim, s))?.contiguous()?; // [1,dim,S]
                                                                            // SEANet decoder: conv_in -> per stage (ELU->convtr->resblock) -> ELU -> conv_out
        y = causal_conv1d(&y, &self.conv_in.w, self.conv_in.b.as_ref(), 1, 1, 1)?;
        for (convtr, res) in &self.up {
            y = elu(&y)?;
            let stride = convtr.resampling_stride()?;
            y = causal_convtr1d(&y, &convtr.w, convtr.b.as_ref(), stride, 1)?;
            y = res.forward(&y)?;
        }
        y = elu(&y)?;
        y = causal_conv1d(&y, &self.conv_out.w, self.conv_out.b.as_ref(), 1, 1, 1)?;
        y.flatten_all()?.to_vec1_f32()
    }
}

impl KyutaiMimiDecoder {
    /// Where this model's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, self.xf.len())
    }
}
