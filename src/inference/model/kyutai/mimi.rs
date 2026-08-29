//! Kyutai `tts-1.6b-en_fr` - Mimi RVQ codec DECODER (component 3 of the TTS port).
//!
//! codes `[K=32, T]` (1 semantic + 31 acoustic RVQ, drop the text codebook first)
//!   -> RVQ dequant -> 512-d latent [512,T]
//!   -> upsample x2 (depthwise convtr) -> decoder-transformer (8 layers, RoPE + LayerScale)
//!   -> SEANet decoder (ratios 8.6.5.4 = x960) -> 24 kHz mono waveform.
//!
//! Structure mirrors `native_pocket_tts`'s Mimi decoder (reuses the tensor-native
//! conv/attn ops) but for the tts-1.6b checkpoint: DISCRETE RVQ input (vs pocket's
//! continuous latent), 8 transformer layers, 4 SEANet stages, `self_attn.in_projs.0`
//! weight names. Codebook = `embedding_sum / max(cluster_usage, eps)` (Mimi EMA).

use crate::inference::model::acestep::fsq::rope_tables;
use crate::inference::model::acestep::ops::sdpa;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

const CODEC_DIM: usize = 512;
const VQ_DIM: usize = 256;
const CODEBOOK: usize = 2048;
const N_SEMANTIC: usize = 1;
const N_ACOUSTIC: usize = 31;
const FFN: usize = 2048;
const N_HEAD: usize = 8;
const N_XF: usize = 8;
const XF_CONTEXT: usize = 250;
const ROPE_THETA: f32 = 10_000.0;
const LN_EPS: f32 = 1e-5;
const VQ_EPS: f32 = 1e-5;

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
    fn load(vb: &VarBuilder, name: &str, shape: (usize, usize, usize), bias: bool) -> Result<Self> {
        let w = vb.get(shape, &format!("{name}.weight"))?;
        let b = if bias {
            Some(vb.get(shape.0, &format!("{name}.bias"))?)
        } else {
            None
        };
        Ok(Self { w, b })
    }
    fn load_tr(vb: &VarBuilder, name: &str, shape: (usize, usize, usize)) -> Result<Self> {
        Ok(Self {
            w: vb.get(shape, &format!("{name}.weight"))?,
            b: Some(vb.get(shape.1, &format!("{name}.bias"))?),
        })
    }
}
/// SEANet residual block: ELU -> conv(k3) -> ELU -> conv(k1) -> + input.
struct ResBlock {
    conv1: Conv,
    conv2: Conv,
}
impl ResBlock {
    fn load(vb: &VarBuilder, idx: usize, ch: usize, mid: usize) -> Result<Self> {
        let p = format!("{idx}.block");
        Ok(Self {
            conv1: Conv::load(vb, &format!("{p}.1.conv.conv"), (mid, ch, 3), true)?,
            conv2: Conv::load(vb, &format!("{p}.3.conv.conv"), (ch, mid, 1), true)?,
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
/// One ResidualVectorQuantizer's decode: Σ_k codebook_k[codes_k] -> output_proj (256->512).
struct Rvq {
    codebooks: Vec<Tensor>, // each [CODEBOOK, VQ_DIM], EMA-normalized
    output_proj: Tensor,    // [512, 256, 1] conv1x1
}
impl Rvq {
    fn load(vb: &VarBuilder, n_cb: usize) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(n_cb);
        for k in 0..n_cb {
            let es = vb.get(
                (CODEBOOK, VQ_DIM),
                &format!("vq.layers.{k}._codebook.embedding_sum"),
            )?;
            let cu = vb.get(CODEBOOK, &format!("vq.layers.{k}._codebook.cluster_usage"))?;
            // codebook = embedding_sum / max(cluster_usage, eps)  (broadcast over VQ_DIM)
            let denom = max_scalar(&cu, VQ_EPS)?.reshape((CODEBOOK, 1))?;
            codebooks.push(es.broadcast_div(&denom)?);
        }
        Ok(Self {
            codebooks,
            output_proj: vb.get((CODEC_DIM, VQ_DIM, 1), "output_proj.weight")?,
        })
    }
    /// `codes`: `[n_cb, T]` (u32 indices) -> `[512, T]`.
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let t = codes.shape().dims2()?.1;
        // Σ over codebooks of embedding[code]  -> [T, VQ_DIM]
        let mut acc: Option<Tensor> = None;
        for (k, cb) in self.codebooks.iter().enumerate() {
            let idx = codes.narrow(0, k, 1)?.reshape(t)?; // [T]
            let e = cb.index_select(&idx, 0)?; // [T, VQ_DIM]
            acc = Some(match acc {
                None => e,
                Some(a) => a.add(&e)?,
            });
        }
        let q = acc.unwrap().transpose(0, 1)?.reshape((1, VQ_DIM, t))?; // [1,256,T]
                                                                        // output_proj is a conv1x1 -> [1,512,T] -> [512,T]
        causal_conv1d(&q, &self.output_proj, None, 1, 1, 1)?.reshape((CODEC_DIM, t))
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
}
impl XfLayer {
    fn load(vb: &VarBuilder, idx: usize) -> Result<Self> {
        let p = vb.pp(format!("layers.{idx}"));
        Ok(Self {
            norm1_w: p.get(CODEC_DIM, "norm1.weight")?,
            norm1_b: p.get(CODEC_DIM, "norm1.bias")?,
            norm2_w: p.get(CODEC_DIM, "norm2.weight")?,
            norm2_b: p.get(CODEC_DIM, "norm2.bias")?,
            in_proj: p.get((3 * CODEC_DIM, CODEC_DIM), "self_attn.in_proj_weight")?,
            out_proj: p.get((CODEC_DIM, CODEC_DIM), "self_attn.out_proj.weight")?,
            linear1: p.get((FFN, CODEC_DIM), "linear1.weight")?,
            linear2: p.get((CODEC_DIM, FFN), "linear2.weight")?,
            ls1: p.get(CODEC_DIM, "layer_scale_1.scale")?,
            ls2: p.get(CODEC_DIM, "layer_scale_2.scale")?,
        })
    }
    /// `x: [S, dim]` - sliding-causal RoPE attention.
    fn forward(&self, x: &Tensor, dev: &Device) -> Result<Tensor> {
        let s = x.shape().dims2()?.0;
        let hd = CODEC_DIM / N_HEAD;
        let h = x.layer_norm(&self.norm1_w, Some(&self.norm1_b), LN_EPS)?;
        let qkv = h.matmul_t(&self.in_proj)?;
        let q = qkv.narrow(1, 0, CODEC_DIM)?;
        let k = qkv.narrow(1, CODEC_DIM, CODEC_DIM)?;
        let v = qkv.narrow(1, 2 * CODEC_DIM, CODEC_DIM)?;
        let shp = |t: &Tensor| -> Result<Tensor> {
            t.reshape((s, N_HEAD, hd))?
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
        // sliding causal window (ctx 250; for T<250 == full causal)
        let mut data = vec![0f32; s * s];
        for i in 0..s {
            for j in 0..s {
                if j > i || i - j >= XF_CONTEXT {
                    data[i * s + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec_f32(data, (s, s))?.to_device(dev)?;
        let att = sdpa(&q, &k, &v, Some(&mask), false, scale, 1.0)?;
        let att = att
            .transpose(1, 2)?
            .contiguous()?
            .reshape((s, CODEC_DIM))?
            .matmul_t(&self.out_proj)?;
        let x = x.add(&att.broadcast_mul(&self.ls1.reshape((1, CODEC_DIM))?)?)?;
        let h = x.layer_norm(&self.norm2_w, Some(&self.norm2_b), LN_EPS)?;
        let h = h
            .matmul_t(&self.linear1)?
            .gelu_erf()?
            .matmul_t(&self.linear2)?;
        x.add(&h.broadcast_mul(&self.ls2.reshape((1, CODEC_DIM))?)?)
    }
}

/// Full Mimi decoder for tts-1.6b: codes -> 24 kHz waveform.
pub struct KyutaiMimiDecoder {
    rvq_first: Rvq,   // 1 semantic codebook
    rvq_rest: Rvq,    // 31 acoustic codebooks
    upsample: Tensor, // [512,1,4] depthwise convtr, stride 2
    xf: Vec<XfLayer>,
    conv_in: Conv,             // model.0  [1024,512,7]
    up: Vec<(Conv, ResBlock)>, // (convtr, resblock) x 4
    conv_out: Conv,            // model.14 [1,64,3]
    device: Device,
}
impl KyutaiMimiDecoder {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let q = vb.pp("quantizer");
        let rvq_first = Rvq::load(&q.pp("rvq_first"), N_SEMANTIC)?;
        let rvq_rest = Rvq::load(&q.pp("rvq_rest"), N_ACOUSTIC)?;
        let upsample = vb.get((CODEC_DIM, 1, 4), "upsample.convtr.convtr.convtr.weight")?;
        let xt = vb.pp("decoder_transformer.transformer");
        let mut xf = Vec::with_capacity(N_XF);
        for i in 0..N_XF {
            xf.push(XfLayer::load(&xt, i)?);
        }
        let dec = vb.pp("decoder.model");
        let conv_in = Conv::load(&dec, "0.conv.conv", (1024, 512, 7), true)?;
        // (convtr (in,out,k), stride, convtr_idx, resblock_idx, (ch,mid))
        let stages: [((usize, usize, usize), usize, usize, usize, (usize, usize)); 4] = [
            ((1024, 512, 16), 8, 2, 3, (512, 256)),
            ((512, 256, 12), 6, 5, 6, (256, 128)),
            ((256, 128, 10), 5, 8, 9, (128, 64)),
            ((128, 64, 8), 4, 11, 12, (64, 32)),
        ];
        let mut up = Vec::with_capacity(4);
        for (shape, _stride, ctr_idx, rb_idx, (ch, mid)) in stages {
            up.push((
                Conv::load_tr(&dec, &format!("{ctr_idx}.convtr.convtr"), shape)?,
                ResBlock::load(&dec, rb_idx, ch, mid)?,
            ));
        }
        let conv_out = Conv::load(&dec, "14.conv.conv", (1, 64, 3), true)?;
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

    /// RVQ dequant only (for parity vs mimi_latent): `codes [32,T]` -> `[512,T]`.
    pub fn decode_latent(&self, codes: &Tensor) -> Result<Tensor> {
        let t = codes.shape().dims2()?.1;
        let sem = self.rvq_first.decode(&codes.narrow(0, 0, N_SEMANTIC)?)?;
        let rest = self
            .rvq_rest
            .decode(&codes.narrow(0, N_SEMANTIC, N_ACOUSTIC)?)?;
        debug_assert_eq!(t, sem.shape().dims2()?.1);
        sem.add(&rest)
    }

    /// Full decode: `codes [32,T]` (u32) -> 24 kHz mono waveform `Vec<f32>`.
    pub fn decode(&self, codes: &Tensor) -> Result<Vec<f32>> {
        let latent = self.decode_latent(codes)?; // [512,T]
        let t = latent.shape().dims2()?.1;
        let emb = latent.reshape((1, CODEC_DIM, t))?;
        // upsample x2 (depthwise convtr)
        let x = causal_convtr1d(&emb, &self.upsample, None, 2, CODEC_DIM)?; // [1,512,2T]
        let s = x.shape().dims3()?.2;
        // decoder transformer at 2T framerate
        let mut h = x.reshape((CODEC_DIM, s))?.transpose(0, 1)?.contiguous()?; // [S,512]
        for l in &self.xf {
            h = l.forward(&h, &self.device)?;
        }
        let mut y = h
            .transpose(0, 1)?
            .reshape((1, CODEC_DIM, s))?
            .contiguous()?; // [1,512,S]
                            // SEANet decoder: conv_in -> 4x(ELU->convtr->resblock) -> ELU -> conv_out
        y = causal_conv1d(&y, &self.conv_in.w, self.conv_in.b.as_ref(), 1, 1, 1)?;
        let strides = [8usize, 6, 5, 4];
        for ((convtr, res), &stride) in self.up.iter().zip(strides.iter()) {
            y = elu(&y)?;
            y = causal_convtr1d(&y, &convtr.w, convtr.b.as_ref(), stride, 1)?;
            y = res.forward(&y)?;
        }
        y = elu(&y)?;
        y = causal_conv1d(&y, &self.conv_out.w, self.conv_out.b.as_ref(), 1, 1, 1)?;
        y.flatten_all()?.to_vec1_f32()
    }
}

/// Full Mimi ENCODER for tts-1.6b: 24 kHz waveform -> unquantized 512-d latent `[512,T]`
/// at 12.5 Hz - the `speaker_wavs` voice conditioning. Mirror of the decoder: SEANet
/// encoder (strides 4.5.6.8 = x960 -> 25 Hz) -> encoder-transformer (8 layers) -> causal
/// downsample conv (÷2 -> 12.5 Hz). No RVQ (the conditioning is the continuous latent).
pub struct KyutaiMimiEncoder {
    conv_in: Conv,               // model.0  [64,1,7]
    down: Vec<(ResBlock, Conv)>, // (resblock, strided conv) x 4
    conv_out: Conv,              // model.14 [512,1024,3]
    xf: Vec<XfLayer>,
    downsample: Tensor, // [512,512,4] causal conv, stride 2
    device: Device,
}
impl KyutaiMimiEncoder {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let enc = vb.pp("encoder.model");
        let conv_in = Conv::load(&enc, "0.conv.conv", (64, 1, 7), true)?;
        // (resblock_idx, (ch, mid), conv_idx, conv_shape (out,in,k), stride)
        let stages: [(usize, (usize, usize), usize, (usize, usize, usize), usize); 4] = [
            (1, (64, 32), 3, (128, 64, 8), 4),
            (4, (128, 64), 6, (256, 128, 10), 5),
            (7, (256, 128), 9, (512, 256, 12), 6),
            (10, (512, 256), 12, (1024, 512, 16), 8),
        ];
        let mut down = Vec::with_capacity(4);
        for (rb_idx, (ch, mid), cv_idx, shape, _stride) in stages {
            down.push((
                ResBlock::load(&enc, rb_idx, ch, mid)?,
                Conv::load(&enc, &format!("{cv_idx}.conv.conv"), shape, true)?,
            ));
        }
        let conv_out = Conv::load(&enc, "14.conv.conv", (512, 1024, 3), true)?;
        let xt = vb.pp("encoder_transformer.transformer");
        let mut xf = Vec::with_capacity(N_XF);
        for i in 0..N_XF {
            xf.push(XfLayer::load(&xt, i)?);
        }
        let downsample = vb.get(
            (CODEC_DIM, CODEC_DIM, 4),
            "downsample.conv.conv.conv.weight",
        )?;
        Ok(Self {
            conv_in,
            down,
            conv_out,
            xf,
            downsample,
            device,
        })
    }

    /// `wav`: mono 24 kHz samples (host `Vec<f32>`), length ideally a multiple of 1920
    /// (frame_size); it is right-padded to a multiple otherwise. Returns `[512, T]`.
    pub fn encode(&self, wav: &[f32]) -> Result<Tensor> {
        const FRAME: usize = 1920;
        let mut samples = wav.to_vec();
        if samples.len() % FRAME != 0 {
            samples.resize(samples.len().div_ceil(FRAME) * FRAME, 0.0);
        }
        let l = samples.len();
        let mut y = Tensor::from_vec_f32(samples, (1, 1, l))?.to_device(&self.device)?;
        // SEANet encoder
        y = causal_conv1d(&y, &self.conv_in.w, self.conv_in.b.as_ref(), 1, 1, 1)?;
        let strides = [4usize, 5, 6, 8];
        for ((res, conv), &stride) in self.down.iter().zip(strides.iter()) {
            y = res.forward(&y)?;
            y = elu(&y)?;
            y = causal_conv1d(&y, &conv.w, conv.b.as_ref(), stride, 1, 1)?;
        }
        y = elu(&y)?;
        y = causal_conv1d(&y, &self.conv_out.w, self.conv_out.b.as_ref(), 1, 1, 1)?; // [1,512,T25]
                                                                                     // encoder transformer at 25 Hz
        let s = y.shape().dims3()?.2;
        let mut h = y.reshape((CODEC_DIM, s))?.transpose(0, 1)?.contiguous()?; // [S,512]
        for lyr in &self.xf {
            h = lyr.forward(&h, &self.device)?;
        }
        y = h
            .transpose(0, 1)?
            .reshape((1, CODEC_DIM, s))?
            .contiguous()?; // [1,512,S]
                            // downsample ÷2 (causal conv, stride 2) -> 12.5 Hz
        y = causal_conv1d(&y, &self.downsample, None, 2, 1, 1)?;
        let t = y.shape().dims3()?.2;
        y.reshape((CODEC_DIM, t))
    }
}
