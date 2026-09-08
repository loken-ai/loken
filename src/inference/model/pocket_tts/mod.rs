//! Kyutai pocket-tts - Stage 1: the Mimi decoder (synthesis path).
//!
//! pocket-tts is a flow-matching TTS over a CONTINUOUS 32-dim latent (no discrete
//! RVQ codebooks - the checkpoint has a single `quantizer.output_proj` 32->512 and
//! `emb_mean/emb_std` instead of codebooks). Stage 1 ports the audio synthesis half:
//! a 32-dim latent at 12.5 Hz -> 24 kHz waveform, via
//!   output_proj(32->512) -> upsample.convtr(x16, depthwise, causal)
//!   -> decoder_transformer(2 layers, RoPE+LayerScale) -> SEANet decoder(x120, ELU)
//! Total upsampling 16x120 = 1920 (12.5 Hz -> 24 kHz). All convs are causal
//! (left-pad) and conv-transposes trim the right tail, matching the streaming model.
//!
//! Net-new vs existing VAEs: ELU activation, LayerScale, fused-qkv codec transformer
//! with RoPE. Reuses `tensor` conv1d/conv_transpose1d, `sdpa`, `rope_tables`.

use crate::inference::model::acestep::fsq::rope_tables;
use crate::inference::model::acestep::ops::sdpa;
use crate::tensor::VarBuilder;
use crate::tensor::{Device, Result, Tensor};

pub const SAMPLE_RATE: u32 = 24_000;
pub const LATENT_DIM: usize = 32;
pub const CODEC_DIM: usize = 512;
const ROPE_THETA: f32 = 10_000.0;
const LN_EPS: f32 = 1e-5;

/// `bias [out]` added to a conv output `[B, out, L]`.
fn add_bias(y: Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    match bias {
        None => Ok(y),
        Some(b) => {
            let out = b.dims()[0];
            y.broadcast_add(&b.reshape((1, out, 1))?)
        }
    }
}

/// Causal Conv1d (left-pad only): pad time by the streaming-conv amount
/// `(k-1)*dilation + 1 - stride` (= `(k-1)*dilation` for stride 1), conv with no pad.
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
    let y = xp.conv1d(w, 0, stride, dilation, groups)?;
    add_bias(y, bias)
}

/// Causal ConvTranspose1d: transpose then trim `(k - stride)` from the right so
/// length becomes exactly `L_in * stride`.
fn causal_convtr1d(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    groups: usize,
) -> Result<Tensor> {
    let (_in, _outg, k) = w.shape().dims3()?;
    let y = x.conv_transpose1d(w, 0, 0, stride, 1, groups)?;
    let l = y.shape().dims3()?.2;
    let keep = l.saturating_sub(k - stride);
    let y = y.narrow(2, 0, keep)?;
    add_bias(y, bias)
}

/// ELU(1.0) = relu(x) + (exp(min(x,0)) - 1), built from relu/affine/exp/add.
fn elu(x: &Tensor) -> Result<Tensor> {
    let pos = x.relu()?;
    // min(x,0) = -relu(-x)
    let min_x0 = x.affine(-1.0, 0.0)?.relu()?.affine(-1.0, 0.0)?;
    let neg = min_x0.exp()?.affine(1.0, -1.0)?;
    pos.add(&neg)
}

/// Raw conv weights + bias.
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
    /// ConvTranspose bias has `out` channels = shape.1 (weight is [in,out,k]).
    fn load_tr(vb: &VarBuilder, name: &str, shape: (usize, usize, usize)) -> Result<Self> {
        let w = vb.get(shape, &format!("{name}.weight"))?;
        let b = Some(vb.get(shape.1, &format!("{name}.bias"))?);
        Ok(Self { w, b })
    }
}

/// SEANet residual block: ELU -> conv(k3) -> ELU -> conv(k1) -> + input.
struct ResBlock {
    conv1: Conv, // [mid, in, 3]
    conv2: Conv, // [in, mid, 1]
}
impl ResBlock {
    fn load(vb: &VarBuilder, idx: usize, ch: usize, mid: usize) -> Result<Self> {
        let p = format!("{idx}.block");
        Ok(Self {
            conv1: Conv::load(vb, &format!("{p}.1.conv"), (mid, ch, 3), true)?,
            conv2: Conv::load(vb, &format!("{p}.3.conv"), (ch, mid, 1), true)?,
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

/// One codec transformer layer (pre-norm, fused-qkv MHA + RoPE + LayerScale, GELU MLP).
struct XfLayer {
    norm1_w: Tensor,
    norm1_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    in_proj: Tensor,     // [3*dim, dim]
    out_proj: Tensor,    // [dim, dim]
    linear1: Tensor,     // [ffn, dim]
    linear2: Tensor,     // [dim, ffn]
    ls1: Option<Tensor>, // [dim] (codec only; flow_lm has no LayerScale)
    ls2: Option<Tensor>,
    /// Sliding causal-attention window (codec transformers use 250; flow_lm = None = full causal).
    context: Option<usize>,
}
impl XfLayer {
    fn load(
        vb: &VarBuilder,
        idx: usize,
        dim: usize,
        ffn: usize,
        layer_scale: bool,
        context: Option<usize>,
    ) -> Result<Self> {
        let p = vb.pp(format!("layers.{idx}"));
        let (ls1, ls2) = if layer_scale {
            (
                Some(p.get(dim, "layer_scale_1.scale")?),
                Some(p.get(dim, "layer_scale_2.scale")?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            context,
            norm1_w: p.get(dim, "norm1.weight")?,
            norm1_b: p.get(dim, "norm1.bias")?,
            norm2_w: p.get(dim, "norm2.weight")?,
            norm2_b: p.get(dim, "norm2.bias")?,
            in_proj: p.get((3 * dim, dim), "self_attn.in_proj.weight")?,
            out_proj: p.get((dim, dim), "self_attn.out_proj.weight")?,
            linear1: p.get((ffn, dim), "linear1.weight")?,
            linear2: p.get((dim, ffn), "linear2.weight")?,
            ls1,
            ls2,
        })
    }

    /// `x: [S, dim]` (single batch). Causal RoPE attention.
    fn forward(&self, x: &Tensor, dim: usize, n_head: usize, dev: &Device) -> Result<Tensor> {
        let s = x.shape().dims2()?.0;
        let hd = dim / n_head;
        // --- attention ---
        let h = x.layer_norm(&self.norm1_w, Some(&self.norm1_b), LN_EPS)?;
        let qkv = h.matmul_t(&self.in_proj)?; // [S, 3*dim]
        let q = qkv.narrow(1, 0, dim)?;
        let k = qkv.narrow(1, dim, dim)?;
        let v = qkv.narrow(1, 2 * dim, dim)?;
        let shp = |t: &Tensor| -> Result<Tensor> {
            t.reshape((s, n_head, hd))?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()
        }; // [1, H, S, hd]
        let (cosv, sinv) = rope_tables(s, hd, ROPE_THETA);
        let cos = Tensor::from_vec_f32(cosv, (s, hd / 2))?.to_device(dev)?;
        let sin = Tensor::from_vec_f32(sinv, (s, hd / 2))?.to_device(dev)?;
        // Interleaved RoPE (pocket-tts packs rotation pairs as adjacent dims).
        let q = shp(&q)?.rope_i(&cos, &sin)?;
        let k = shp(&k)?.rope_i(&cos, &sin)?;
        let v = shp(&v)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let att = match self.context {
            // Sliding causal window: key j visible iff j<=i and i-j<context.
            Some(ctx) => {
                let mut data = vec![0f32; s * s];
                for i in 0..s {
                    for j in 0..s {
                        if j > i || i - j >= ctx {
                            data[i * s + j] = f32::NEG_INFINITY;
                        }
                    }
                }
                let mask = Tensor::from_vec_f32(data, (s, s))?.to_device(dev)?;
                sdpa(&q, &k, &v, Some(&mask), false, scale, 1.0)?
            }
            None => sdpa(&q, &k, &v, None, true, scale, 1.0)?,
        }; // [1,H,S,hd]
        let att = att.transpose(1, 2)?.contiguous()?.reshape((s, dim))?;
        let att = att.matmul_t(&self.out_proj)?;
        let att = match &self.ls1 {
            Some(s) => att.broadcast_mul(&s.reshape((1, dim))?)?,
            None => att,
        };
        let x = x.add(&att)?;
        // --- mlp ---
        let h = x.layer_norm(&self.norm2_w, Some(&self.norm2_b), LN_EPS)?;
        let h = h
            .matmul_t(&self.linear1)?
            .gelu_erf()?
            .matmul_t(&self.linear2)?;
        let h = match &self.ls2 {
            Some(s) => h.broadcast_mul(&s.reshape((1, dim))?)?,
            None => h,
        };
        x.add(&h)
    }
}

/// Mimi decoder: 32-dim latent @12.5 Hz -> 24 kHz mono waveform.
pub struct MimiDecoder {
    output_proj: Tensor,       // [512, 32, 1] conv1d 32->512
    upsample: Tensor,          // [512, 1, 32] depthwise convtr, stride 16
    xf: Vec<XfLayer>,          // 2 decoder-transformer layers
    conv_in: Conv,             // model.0  [512,512,7]
    up: Vec<(Conv, ResBlock)>, // (convtr, resblock) x 3
    conv_out: Conv,            // model.11 [1,64,3]
    device: Device,
    n_head: usize,
}

impl MimiDecoder {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let mimi = vb.pp("mimi");
        let q = mimi.pp("quantizer");
        let output_proj = q.get((CODEC_DIM, LATENT_DIM, 1), "output_proj.weight")?;
        let upsample = mimi.get((CODEC_DIM, 1, 32), "upsample.convtr.convtr.weight")?;

        let xt = mimi.pp("decoder_transformer.transformer");
        let mut xf = Vec::with_capacity(2);
        for i in 0..2 {
            xf.push(XfLayer::load(&xt, i, CODEC_DIM, 2048, true, Some(250))?);
        }

        let dec = mimi.pp("decoder.model");
        let conv_in = Conv::load(&dec, "0.conv", (512, 512, 7), true)?;
        // (convtr in/out/k/stride, resblock idx, resblock ch/mid)
        let stages: [((usize, usize, usize, usize), usize, (usize, usize)); 3] = [
            ((512, 256, 12, 6), 3, (256, 128)),
            ((256, 128, 10, 5), 6, (128, 64)),
            ((128, 64, 8, 4), 9, (64, 32)),
        ];
        let mut up = Vec::with_capacity(3);
        for ((ci, co, k, _s), rb, (ch, mid)) in stages {
            let convtr_idx = rb - 1; // 2,5,8
            let convtr = Conv::load_tr(&dec, &format!("{convtr_idx}.convtr"), (ci, co, k))?;
            let res = ResBlock::load(&dec, rb, ch, mid)?;
            up.push((convtr, res));
        }
        let conv_out = Conv::load(&dec, "11.conv", (1, 64, 3), true)?;

        Ok(Self {
            output_proj,
            upsample,
            xf,
            conv_in,
            up,
            conv_out,
            device,
            n_head: 8,
        })
    }

    /// `latent: [32, T]` (single batch, conv layout) -> waveform `[T*1920]`.
    pub fn decode(&self, latent: &Tensor) -> Result<Vec<f32>> {
        let (c, t) = latent.shape().dims2()?;
        debug_assert_eq!(c, LATENT_DIM);
        let x = latent.reshape((1, c, t))?;
        // 32 -> 512 (k1 conv = DummyQuantizer.output_proj), then the codec decode path.
        let x = causal_conv1d(&x, &self.output_proj, None, 1, 1, 1)?;
        self.decode_from_codec(&x)
    }

    /// Decode from the 512-dim codec embedding `[1,512,T@12.5Hz]` (post-quantizer):
    /// depthwise upsample x16 -> decoder transformer -> SEANet -> 24 kHz waveform.
    pub fn decode_from_codec(&self, emb: &Tensor) -> Result<Vec<f32>> {
        let mut x = causal_convtr1d(emb, &self.upsample, None, 16, CODEC_DIM)?; // [1,512,16T]
        let s = x.shape().dims3()?.2;
        let mut h = x.reshape((CODEC_DIM, s))?.transpose(0, 1)?.contiguous()?; // [S,512]
        for l in &self.xf {
            h = l.forward(&h, CODEC_DIM, self.n_head, &self.device)?;
        }
        x = h
            .transpose(0, 1)?
            .contiguous()?
            .reshape((1, CODEC_DIM, s))?;
        x = causal_conv1d(&x, &self.conv_in.w, self.conv_in.b.as_ref(), 1, 1, 1)?;
        for (convtr, res) in &self.up {
            x = elu(&x)?;
            let stride = convtr.w.shape().dims3()?.2.div_ceil(2); // k = 2*stride
            x = causal_convtr1d(&x, &convtr.w, convtr.b.as_ref(), stride, 1)?;
            x = res.forward(&x)?;
        }
        x = elu(&x)?;
        x = causal_conv1d(&x, &self.conv_out.w, self.conv_out.b.as_ref(), 1, 1, 1)?;
        let l = x.shape().dims3()?.2;
        x.reshape((l,))?.to_vec1_f32()
    }
}

/// Mimi encoder (waveform -> 512-dim codec embedding @12.5 Hz) - for the decoder
/// round-trip diagnostic and the speaker conditioning.
pub struct MimiEncoder {
    conv0: Conv,
    res: Vec<ResBlock>, // 3 residual blocks (after conv0, and after each downsample)
    down: Vec<Conv>,    // 3 strided downsample convs (ratios 4,5,6)
    conv_final: Conv,   // model.11 [512,512,3]
    xf: Vec<XfLayer>,   // 2 encoder-transformer layers
    downsample: Tensor, // [512,512,32] stride16
    device: Device,
}
impl MimiEncoder {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let mimi = vb.pp("mimi");
        let enc = mimi.pp("encoder.model");
        let conv0 = Conv::load(&enc, "0.conv", (64, 1, 7), true)?;
        // (resblock idx, ch, mid), (downsample idx, in, out, k)
        let res = vec![
            ResBlock::load(&enc, 1, 64, 32)?,
            ResBlock::load(&enc, 4, 128, 64)?,
            ResBlock::load(&enc, 7, 256, 128)?,
        ];
        let down = vec![
            Conv::load(&enc, "3.conv", (128, 64, 8), true)?,
            Conv::load(&enc, "6.conv", (256, 128, 10), true)?,
            Conv::load(&enc, "9.conv", (512, 256, 12), true)?,
        ];
        let conv_final = Conv::load(&enc, "11.conv", (512, 512, 3), true)?;
        let xt = mimi.pp("encoder_transformer.transformer");
        let mut xf = Vec::with_capacity(2);
        for i in 0..2 {
            xf.push(XfLayer::load(&xt, i, CODEC_DIM, 2048, true, Some(250))?);
        }
        let downsample = mimi.get((CODEC_DIM, CODEC_DIM, 32), "downsample.conv.conv.weight")?;
        Ok(Self {
            conv0,
            res,
            down,
            conv_final,
            xf,
            downsample,
            device,
        })
    }

    /// `samples`: mono f32 @24 kHz -> codec embedding `[1,512,T@12.5Hz]`.
    pub fn encode(&self, samples: &[f32]) -> Result<Tensor> {
        // Pad to a whole number of frames (frame_size = sample_rate/frame_rate = 1920)
        // so the last conv window is full - mirrors `pad_for_conv1d` in encode_to_latent.
        const FRAME_SIZE: usize = 1920;
        let mut buf = samples.to_vec();
        let t = buf.len().div_ceil(FRAME_SIZE) * FRAME_SIZE;
        buf.resize(t, 0.0);
        let mut x = Tensor::from_vec_f32(buf, (1, 1, t))?.to_device(&self.device)?;
        x = causal_conv1d(&x, &self.conv0.w, self.conv0.b.as_ref(), 1, 1, 1)?;
        let strides = [4usize, 5, 6];
        for i in 0..3 {
            x = self.res[i].forward(&x)?;
            x = elu(&x)?;
            x = causal_conv1d(
                &x,
                &self.down[i].w,
                self.down[i].b.as_ref(),
                strides[i],
                1,
                1,
            )?;
        }
        x = elu(&x)?;
        x = causal_conv1d(&x, &self.conv_final.w, self.conv_final.b.as_ref(), 1, 1, 1)?;
        let downsample_replicate = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
            // ConvDownsample1d uses pad_mode="replicate": left-pad with the first
            // frame repeated (not zeros), then strided conv.
            let k = w.shape().dims3()?.2;
            let pad = k - 16; // (k-1)+1-stride, stride=16
            let first = x.narrow(2, 0, 1)?; // [1,C,1]
            let reps: Vec<&Tensor> = std::iter::repeat_n(&first, pad).collect();
            let mut parts = reps;
            parts.push(x);
            let xp = Tensor::cat(&parts, 2)?;
            xp.conv1d(w, 0, 16, 1, 1)
        };
        // encoder transformer
        let s = x.shape().dims3()?.2;
        let mut h = x.reshape((CODEC_DIM, s))?.transpose(0, 1)?.contiguous()?;
        for l in &self.xf {
            h = l.forward(&h, CODEC_DIM, 8, &self.device)?;
        }
        x = h
            .transpose(0, 1)?
            .contiguous()?
            .reshape((1, CODEC_DIM, s))?;
        // downsample x16 -> 12.5 Hz (replicate padding)
        downsample_replicate(&x, &self.downsample)
    }
}

/// The pocket-tts checkpoint and the SentencePiece tokenizer that goes with it.
///
/// The tokenizer is published in the sibling repository rather than alongside the
/// checkpoint, so the two are resolved separately and returned together - no caller
/// has a use for one without the other.
pub fn resolve_paths() -> (std::path::PathBuf, std::path::PathBuf) {
    let find = crate::inference::cache::hf::file;
    (
        find("models--kyutai--pocket-tts", "tts_b6369a24.safetensors"),
        find(
            "models--kyutai--pocket-tts-without-voice-cloning",
            "tokenizer.model",
        ),
    )
}

/// Read a 16-bit PCM WAV (mono or multi-channel; downmixed) and linearly resample
/// to 24 kHz mono f32 - for reference-voice loading in the TTS CLI.
pub fn load_wav_mono_24k(path: &str) -> std::io::Result<Vec<f32>> {
    let b = std::fs::read(path)?;
    if b.len() < 44 || &b[0..4] != b"RIFF" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a RIFF/WAV",
        ));
    }
    let u16le = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let u32le = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let channels = u16le(22).max(1) as usize;
    let sr = u32le(24).max(1);
    // Find the "data" chunk (skip past fmt and any others).
    let mut i = 12usize;
    let (mut data_off, mut data_len) = (44usize, b.len() - 44);
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let sz = u32le(i + 4) as usize;
        if id == b"data" {
            data_off = i + 8;
            data_len = sz.min(b.len() - (i + 8));
            break;
        }
        i += 8 + sz + (sz & 1);
    }
    // Decode i16 -> f32, downmix to mono.
    let frames = data_len / (2 * channels);
    let mut mono = Vec::with_capacity(frames);
    for f in 0..frames {
        let mut acc = 0f32;
        for c in 0..channels {
            let o = data_off + (f * channels + c) * 2;
            acc += i16::from_le_bytes([b[o], b[o + 1]]) as f32 / 32768.0;
        }
        mono.push(acc / channels as f32);
    }
    if sr == 24_000 {
        return Ok(mono);
    }
    // Linear resample to 24 kHz.
    let ratio = 24_000.0 / sr as f32;
    let out_len = (mono.len() as f32 * ratio) as usize;
    let mut out = Vec::with_capacity(out_len);
    for j in 0..out_len {
        let src = j as f32 / ratio;
        let i0 = src.floor() as usize;
        let frac = src - i0 as f32;
        let a = mono.get(i0).copied().unwrap_or(0.0);
        let b2 = mono.get(i0 + 1).copied().unwrap_or(a);
        out.push(a + (b2 - a) * frac);
    }
    Ok(out)
}

// ======================= Stage 2: flow_lm (text -> latents) =======================

const DIM: usize = 1024; // flow_lm transformer dim
const FLOW_DIM: usize = 512; // flow_net hidden
const FREQ_EMB: usize = 256; // time embedding freq size (freqs len = 128)
const EOS_THRESHOLD: f32 = -4.0;
const TEMP: f32 = 0.7;

/// `y = x @ wᵀ + b` for a 2-D `x [N, in]`, `w [out, in]`, `b [out]`.
fn linear_b(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let out = w.shape().dims2()?.0;
    x.matmul_t(w)?.broadcast_add(&b.reshape((1, out))?)
}

/// AdaLN modulate: `x * (1 + scale) + shift` (all `[1, D]`).
fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
    x.mul(&scale.affine(1.0, 1.0)?)?.add(shift)
}

/// pocket-tts RMSNorm: `x * alpha / sqrt(var(x) + eps)`, var over last dim
/// (UNBIASED, /(D-1); mean is subtracted for the variance but NOT for x).
fn rms_norm_alpha(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let d = *x.shape().dims().last().unwrap();
    let mean = x.mean_keepdim(crate::tensor::D::Minus1)?; // [1,1]
    let xc = x.broadcast_add(&mean.affine(-1.0, 0.0)?)?;
    let var = xc
        .mul(&xc)?
        .sum_keepdim(crate::tensor::D::Minus1)?
        .affine(1.0 / (d - 1) as f32, 0.0)?;
    let denom = var.affine(1.0, eps)?.sqrt()?; // [1,1]
    x.broadcast_mul(&denom.powf(-1.0)?)?
        .mul(&alpha.reshape((1, d))?)
}

/// Sinusoidal timestep embedder: freqs -> cos/sin -> Linear->SiLU->Linear->RMSNorm.
struct TimeEmbed {
    freqs: Vec<f32>, // [128]
    w0: Tensor,
    b0: Tensor, // [512,256],[512]
    w2: Tensor,
    b2: Tensor,    // [512,512],[512]
    alpha: Tensor, // RMSNorm [512]
}
impl TimeEmbed {
    fn load(vb: &VarBuilder) -> Result<Self> {
        let half = FREQ_EMB / 2; // 128
        Ok(Self {
            freqs: vb.get(half, "freqs")?.to_vec1_f32()?,
            w0: vb.get((FLOW_DIM, FREQ_EMB), "mlp.0.weight")?,
            b0: vb.get(FLOW_DIM, "mlp.0.bias")?,
            w2: vb.get((FLOW_DIM, FLOW_DIM), "mlp.2.weight")?,
            b2: vb.get(FLOW_DIM, "mlp.2.bias")?,
            alpha: vb.get(FLOW_DIM, "mlp.3.alpha")?,
        })
    }
    fn forward(&self, t: f32, dev: &Device) -> Result<Tensor> {
        // emb = cat(cos(t*freqs), sin(t*freqs)) -> [1,256]
        let mut emb = vec![0f32; FREQ_EMB];
        let half = self.freqs.len();
        for (i, &f) in self.freqs.iter().enumerate() {
            emb[i] = (t * f).cos();
            emb[half + i] = (t * f).sin();
        }
        let e = Tensor::from_vec_f32(emb, (1, FREQ_EMB))?.to_device(dev)?;
        let h = linear_b(&e, &self.w0, &self.b0)?.silu()?;
        let h = linear_b(&h, &self.w2, &self.b2)?;
        rms_norm_alpha(&h, &self.alpha, 1e-5)
    }
}

/// flow_net residual block (AdaLN modulate -> MLP -> gated residual).
struct FlowResBlock {
    in_ln_w: Tensor,
    in_ln_b: Tensor,
    mlp0_w: Tensor,
    mlp0_b: Tensor,
    mlp2_w: Tensor,
    mlp2_b: Tensor,
    ada_w: Tensor,
    ada_b: Tensor, // [1536,512],[1536]
}
impl FlowResBlock {
    fn load(vb: &VarBuilder) -> Result<Self> {
        Ok(Self {
            in_ln_w: vb.get(FLOW_DIM, "in_ln.weight")?,
            in_ln_b: vb.get(FLOW_DIM, "in_ln.bias")?,
            mlp0_w: vb.get((FLOW_DIM, FLOW_DIM), "mlp.0.weight")?,
            mlp0_b: vb.get(FLOW_DIM, "mlp.0.bias")?,
            mlp2_w: vb.get((FLOW_DIM, FLOW_DIM), "mlp.2.weight")?,
            mlp2_b: vb.get(FLOW_DIM, "mlp.2.bias")?,
            ada_w: vb.get((3 * FLOW_DIM, FLOW_DIM), "adaLN_modulation.1.weight")?,
            ada_b: vb.get(3 * FLOW_DIM, "adaLN_modulation.1.bias")?,
        })
    }
    fn forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let m = linear_b(&y.silu()?, &self.ada_w, &self.ada_b)?; // [1,1536]
        let shift = m.narrow(1, 0, FLOW_DIM)?;
        let scale = m.narrow(1, FLOW_DIM, FLOW_DIM)?;
        let gate = m.narrow(1, 2 * FLOW_DIM, FLOW_DIM)?;
        let h = x.layer_norm(&self.in_ln_w, Some(&self.in_ln_b), 1e-6)?;
        let h = modulate(&h, &shift, &scale)?;
        let h = linear_b(&h, &self.mlp0_w, &self.mlp0_b)?.silu()?;
        let h = linear_b(&h, &self.mlp2_w, &self.mlp2_b)?;
        x.add(&h.mul(&gate)?)
    }
}

/// flow_net (SimpleMLPAdaLN): velocity field for the flow-matching latent.
struct FlowNet {
    input_proj_w: Tensor,
    input_proj_b: Tensor, // [512,32]
    cond_w: Tensor,
    cond_b: Tensor,            // [512,1024]
    time: Vec<TimeEmbed>,      // 2
    blocks: Vec<FlowResBlock>, // 6
    fnorm_ones: Tensor,        // ones[512] for no-affine norm_final
    flin_ada_w: Tensor,
    flin_ada_b: Tensor, // [1024,512]
    flin_w: Tensor,
    flin_b: Tensor, // [32,512]
    device: Device,
}
impl FlowNet {
    fn load(vb: &VarBuilder, device: Device) -> Result<Self> {
        let mut time = Vec::with_capacity(2);
        for i in 0..2 {
            time.push(TimeEmbed::load(&vb.pp(format!("time_embed.{i}")))?);
        }
        let mut blocks = Vec::with_capacity(6);
        for i in 0..6 {
            blocks.push(FlowResBlock::load(&vb.pp(format!("res_blocks.{i}")))?);
        }
        let fl = vb.pp("final_layer");
        Ok(Self {
            input_proj_w: vb.get((FLOW_DIM, LATENT_DIM), "input_proj.weight")?,
            input_proj_b: vb.get(FLOW_DIM, "input_proj.bias")?,
            cond_w: vb.get((FLOW_DIM, DIM), "cond_embed.weight")?,
            cond_b: vb.get(FLOW_DIM, "cond_embed.bias")?,
            time,
            blocks,
            fnorm_ones: Tensor::from_vec_f32(vec![1f32; FLOW_DIM], FLOW_DIM)?.to_device(&device)?,
            flin_ada_w: fl.get((2 * FLOW_DIM, FLOW_DIM), "adaLN_modulation.1.weight")?,
            flin_ada_b: fl.get(2 * FLOW_DIM, "adaLN_modulation.1.bias")?,
            flin_w: fl.get((LATENT_DIM, FLOW_DIM), "linear.weight")?,
            flin_b: fl.get(LATENT_DIM, "linear.bias")?,
            device,
        })
    }
    /// velocity `[1,32]` given conditioning `c [1,1024]`, times s,t, latent `x [1,32]`.
    fn velocity(&self, c: &Tensor, s: f32, t: f32, x: &Tensor) -> Result<Tensor> {
        let mut h = linear_b(x, &self.input_proj_w, &self.input_proj_b)?; // [1,512]
        let te0 = self.time[0].forward(s, &self.device)?;
        let te1 = self.time[1].forward(t, &self.device)?;
        let tc = te0.add(&te1)?.affine(0.5, 0.0)?;
        let ce = linear_b(c, &self.cond_w, &self.cond_b)?;
        let y = tc.add(&ce)?; // [1,512]
        for b in &self.blocks {
            h = b.forward(&h, &y)?;
        }
        // final layer (norm_final has no affine)
        let m = linear_b(&y.silu()?, &self.flin_ada_w, &self.flin_ada_b)?; // [1,1024]
        let shift = m.narrow(1, 0, FLOW_DIM)?;
        let scale = m.narrow(1, FLOW_DIM, FLOW_DIM)?;
        let hn = h.layer_norm(&self.fnorm_ones, None, 1e-6)?;
        let hn = modulate(&hn, &shift, &scale)?;
        linear_b(&hn, &self.flin_w, &self.flin_b)
    }
}

/// Reference text preprocessing: trim, normalize whitespace, capitalize the first
/// letter, ensure a trailing punctuation, and pad very short inputs with leading
/// spaces (the model is unreliable on too-few tokens). Returns `(text, frames_after_eos)`.
fn preprocess_text(text: &str) -> (String, usize) {
    let mut t = text.trim().replace(['\n', '\r'], " ");
    while t.contains("  ") {
        t = t.replace("  ", " ");
    }
    let n_words = t.split_whitespace().count();
    let frames_after_eos = if n_words <= 4 { 3 } else { 1 };
    // Capitalize first letter.
    if let Some(c) = t.chars().next() {
        if c.is_lowercase() {
            t = c.to_uppercase().collect::<String>() + &t[c.len_utf8()..];
        }
    }
    // Ensure ending punctuation.
    if t.chars().last().is_some_and(|c| c.is_alphanumeric()) {
        t.push('.');
    }
    // Pad very short inputs.
    if n_words < 5 {
        t = " ".repeat(8) + &t;
    }
    (t, frames_after_eos)
}

/// Box-Muller Gaussian noise with the given std, from a seeded RNG.
fn gauss_noise(n: usize, std: f32, rng: &mut rand::rngs::StdRng) -> Vec<f32> {
    use rand::distr::{Distribution, Uniform};
    let uni = Uniform::new(0.0f32, 1.0f32).expect("valid uniform range");
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = uni.sample(rng).max(1e-9);
        let u2 = uni.sample(rng);
        let r = (-2.0f32 * u1.ln()).sqrt();
        out.push(std * r * (std::f32::consts::TAU * u2).cos());
        if out.len() < n {
            out.push(std * r * (std::f32::consts::TAU * u2).sin());
        }
    }
    out
}

/// The full pocket-tts model: flow_lm (text -> normalized latents) + Mimi decoder.
pub struct PocketTts {
    // flow_lm
    embed: Tensor,        // [4001,1024] text token embedding
    input_linear: Tensor, // [1024,32]
    xf: Vec<XfLayer>,     // 6 transformer layers (no LayerScale)
    out_norm_w: Tensor,
    out_norm_b: Tensor,
    out_eos_w: Tensor,
    out_eos_b: Tensor,    // [1,1024],[1]
    bos_emb: Tensor,      // [32]
    emb_mean: Vec<f32>,   // [32]
    emb_std: Vec<f32>,    // [32]
    speaker_proj: Tensor, // [1024,512] voice conditioning projection
    flow: FlowNet,
    mimi: MimiDecoder,
    device: Device,
}

impl PocketTts {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let f = vb.pp("flow_lm");
        let mut xf = Vec::with_capacity(6);
        let xt = f.pp("transformer");
        for i in 0..6 {
            xf.push(XfLayer::load(&xt, i, DIM, 4096, false, None)?);
        }
        let flow = FlowNet::load(&f.pp("flow_net"), device.clone())?;
        let mimi = MimiDecoder::from_safetensors(path, device.clone())?;
        Ok(Self {
            embed: f.get((4001, DIM), "conditioner.embed.weight")?,
            input_linear: f.get((DIM, LATENT_DIM), "input_linear.weight")?,
            xf,
            out_norm_w: f.get(DIM, "out_norm.weight")?,
            out_norm_b: f.get(DIM, "out_norm.bias")?,
            out_eos_w: f.get((1, DIM), "out_eos.weight")?,
            out_eos_b: f.get(1, "out_eos.bias")?,
            bos_emb: f.get(LATENT_DIM, "bos_emb")?,
            emb_mean: f.get(LATENT_DIM, "emb_mean")?.to_vec1_f32()?,
            emb_std: f.get(LATENT_DIM, "emb_std")?.to_vec1_f32()?,
            speaker_proj: f.get((DIM, CODEC_DIM), "speaker_proj_weight")?,
            flow,
            mimi,
            device,
        })
    }

    /// Build a voice conditioning `[T_spk, 1024]` from a 24 kHz mono reference clip:
    /// `speaker_proj( mimi_encode(ref) )`.
    pub fn voice_from_audio(&self, enc: &MimiEncoder, samples: &[f32]) -> Result<Tensor> {
        let emb = enc.encode(samples)?; // [1,512,T]
        let (_, c, t) = emb.shape().dims3()?;
        let lat = emb.reshape((c, t))?.transpose(0, 1)?.contiguous()?; // [T,512]
        lat.matmul_t(&self.speaker_proj) // [T,1024]
    }

    /// Embed text token ids -> `[T,1024]`.
    fn embed_text(&self, ids: &[u32]) -> Result<Tensor> {
        let rows: Vec<Tensor> = ids
            .iter()
            .map(|&id| self.embed.narrow(0, id as usize, 1))
            .collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&refs, 0)
    }

    /// Run the 6-layer transformer over `seq [S,1024]`, return out_norm of the last row `[1,1024]`.
    fn backbone_last(&self, seq: &Tensor) -> Result<Tensor> {
        let mut h = seq.clone();
        for l in &self.xf {
            h = l.forward(&h, DIM, 16, &self.device)?;
        }
        let s = h.shape().dims2()?.0;
        let last = h.narrow(0, s - 1, 1)?; // [1,1024]
        last.layer_norm(&self.out_norm_w, Some(&self.out_norm_b), LN_EPS)
    }

    /// Generate normalized latent frames for `text_ids`, up to `max_frames`,
    /// stopping `frames_after_eos` frames past the EOS logit crossing.
    /// `seed` makes the flow-matching noise reproducible.
    pub fn generate_latents(
        &self,
        text_ids: &[u32],
        voice: Option<&Tensor>,
        max_frames: usize,
        frames_after_eos: usize,
        seed: u64,
    ) -> Result<Vec<Vec<f32>>> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let text_emb = self.embed_text(text_ids)?; // [T,1024]
                                                   // audio token rows projected to transformer space; first = input_linear(bos).
        let mut audio_rows: Vec<Tensor> = vec![self
            .bos_emb
            .reshape((1, LATENT_DIM))?
            .matmul_t(&self.input_linear)?];
        let mut latents: Vec<Vec<f32>> = Vec::new();
        let mut eos_step: Option<usize> = None;
        for step in 0..max_frames {
            // Autoregressive decode runs in spawn_blocking, so a dropped request can
            // only be observed here. See `cancel::scoped` for why this loop reads a
            // thread-published token instead of taking one.
            crate::inference::serve::cancel::scoped::bail()?;
            // seq = [voice? ; text_emb ; audio_rows...] - voice is conditioned FIRST
            // (the reference caches it before the text prefix).
            let mut parts: Vec<&Tensor> = Vec::new();
            if let Some(v) = voice {
                parts.push(v);
            }
            parts.push(&text_emb);
            for r in &audio_rows {
                parts.push(r);
            }
            let seq = Tensor::cat(&parts, 0)?;
            let c = self.backbone_last(&seq)?; // [1,1024]
                                               // EOS
            let eos = linear_b(&c, &self.out_eos_w, &self.out_eos_b)?
                .flatten_all()?
                .to_vec1_f32()?[0];
            if eos > EOS_THRESHOLD && eos_step.is_none() {
                eos_step = Some(step);
            }
            // flow-matching: 1 Euler step from noise N(0, temp)
            let noise = gauss_noise(LATENT_DIM, TEMP.sqrt(), &mut rng);
            let x0 = Tensor::from_vec_f32(noise, (1, LATENT_DIM))?.to_device(&self.device)?;
            let v = self.flow.velocity(&c, 0.0, 1.0, &x0)?;
            let latent = x0.add(&v)?; // normalized latent [1,32]
            latents.push(latent.reshape((LATENT_DIM,))?.to_vec1_f32()?);
            // next input
            audio_rows.push(latent.matmul_t(&self.input_linear)?);
            if let Some(es) = eos_step {
                if step >= es + frames_after_eos {
                    break;
                }
            }
        }
        Ok(latents)
    }

    /// Full text->speech (auto length): preprocess text the way the reference does
    /// (capitalize, ensure ending punctuation, pad very short inputs), tokenize,
    /// bound generation by the token-count heuristic, generate and decode.
    /// `voice` is an optional `[T_spk,1024]` conditioning from [`Self::voice_from_audio`].
    /// `frames_after_eos` overrides how many frames to keep generating past the EOS
    /// crossing (None = the reference word-count heuristic: 3 for <=4 words, else 1).
    /// Larger values keep more of the tail (e.g. the final word) at the cost of length.
    pub fn synthesize_text(
        &self,
        sp: &crate::inference::token::sentencepiece::SentencePiece,
        text: &str,
        voice: Option<&Tensor>,
        frames_after_eos: Option<usize>,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let (clean, fae_auto) = preprocess_text(text);
        let ids = sp.encode(&clean);
        // Reference cap: ceil((n_tokens/3 + 2) * 12.5 frames/s).
        let max_frames = (((ids.len() as f32 / 3.0) + 2.0) * 12.5).ceil() as usize;
        self.synthesize(
            &ids,
            voice,
            max_frames,
            frames_after_eos.unwrap_or(fae_auto),
            seed,
        )
    }

    /// Full synthesis: text token ids -> 24 kHz mono waveform.
    pub fn synthesize(
        &self,
        text_ids: &[u32],
        voice: Option<&Tensor>,
        max_frames: usize,
        frames_after_eos: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let latents = self.generate_latents(text_ids, voice, max_frames, frames_after_eos, seed)?;
        if latents.is_empty() {
            return Ok(Vec::new());
        }
        // Denormalize (z*std+mean) and stack into [32, n] (conv layout).
        let n = latents.len();
        let mut data = vec![0f32; LATENT_DIM * n];
        for (j, lat) in latents.iter().enumerate() {
            for c in 0..LATENT_DIM {
                data[c * n + j] = lat[c] * self.emb_std[c] + self.emb_mean[c];
            }
        }
        let lat = Tensor::from_vec_f32(data, (LATENT_DIM, n))?.to_device(&self.device)?;
        self.mimi.decode(&lat)
    }
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    fn ckpt() -> std::path::PathBuf {
        resolve_paths().0
    }

    fn tokenizer_model() -> std::path::PathBuf {
        resolve_paths().1
    }

    #[test]
    fn preprocess_text_frames_and_punct() {
        // <=4 words -> 3 frames_after_eos + leading pad; capitalized; period added.
        let (t, f) = preprocess_text("hello world");
        assert_eq!(f, 3);
        assert!(t.starts_with("        H"), "padded + capitalized: {t:?}");
        assert!(t.ends_with("world."), "punctuation added: {t:?}");
        // >4 words -> 1 frame_after_eos, no pad.
        let (t2, f2) = preprocess_text("this is a longer test sentence");
        assert_eq!(f2, 1);
        assert!(!t2.starts_with(' ') && t2.ends_with('.'));
        // Existing punctuation preserved.
        assert!(preprocess_text("Done!").0.ends_with('!'));
    }

    fn read_wav16_mono(path: &str) -> Vec<f32> {
        let b = std::fs::read(path).unwrap();
        // skip 44-byte canonical header; interpret the rest as i16 LE.
        b[44..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect()
    }

    /// Read the `[i32 ndim, i32xndim shape, f32 data]` dump format.
    fn load_bin(path: &str) -> (Vec<f32>, Vec<usize>) {
        let b = std::fs::read(path).unwrap();
        let nd = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let mut shape = Vec::with_capacity(nd);
        for i in 0..nd {
            shape.push(i32::from_le_bytes(b[4 + i * 4..8 + i * 4].try_into().unwrap()) as usize);
        }
        let off = 4 + nd * 4;
        let data = b[off..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (data, shape)
    }
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let (mut d, mut na, mut nb) = (0f32, 0f32, 0f32);
        for i in 0..n {
            d += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        d / (na.sqrt() * nb.sqrt() + 1e-9)
    }

    #[test]
    #[ignore = "parity: compare flow_lm c/velocity against /tmp/ref_*.bin from ref_flowlm.py - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn flowlm_parity_vs_ref() {
        let m = PocketTts::from_safetensors(ckpt().to_str().unwrap(), Device::Cpu).unwrap();
        let enc = MimiEncoder::from_safetensors(ckpt().to_str().unwrap(), Device::Cpu).unwrap();
        let dog = format!(
            "{}/results/ezaudio/a_dog_barking.wav",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut vs = read_wav16_mono(&dog);
        vs.truncate(72000);
        let voice = m.voice_from_audio(&enc, &vs).unwrap();
        let vv = voice.flatten_all().unwrap().to_vec1_f32().unwrap();
        let (ref_voice, rvs) = load_bin("/tmp/ref_voice.bin");
        println!(
            "VOICE   cosine={:.5}  my_frames={:?} ref_frames={:?}",
            cosine(&vv, &ref_voice),
            voice.shape().dims(),
            rvs
        );
        // frame-0 conditioning c (voice ; text ; bos)
        let ids: Vec<u32> = vec![10, 42, 7, 100, 3, 55, 8];
        let text_emb = m.embed_text(&ids).unwrap();
        let te = text_emb.flatten_all().unwrap().to_vec1_f32().unwrap();
        let (ref_te, _) = load_bin("/tmp/ref_textemb.bin");
        println!("TEXTEMB cosine={:.5}", cosine(&te, &ref_te));
        let bos_row = m
            .bos_emb
            .reshape((1, LATENT_DIM))
            .unwrap()
            .matmul_t(&m.input_linear)
            .unwrap();
        let br = bos_row.flatten_all().unwrap().to_vec1_f32().unwrap();
        let (ref_inp, _) = load_bin("/tmp/ref_inp.bin");
        println!(
            "BOS_ROW cosine={:.5} (my_rms={:.4} ref_rms={:.4})",
            cosine(&br, &ref_inp),
            (br.iter().map(|x| x * x).sum::<f32>() / br.len() as f32).sqrt(),
            (ref_inp.iter().map(|x| x * x).sum::<f32>() / ref_inp.len() as f32).sqrt()
        );
        let seq = Tensor::cat(&[&voice, &text_emb, &bos_row], 0).unwrap();
        let sv = seq.flatten_all().unwrap().to_vec1_f32().unwrap();
        let (ref_seq, _) = load_bin("/tmp/ref_seq.bin");
        println!("SEQ     cosine={:.5}", cosine(&sv, &ref_seq));
        // raw transformer output (pre out_norm), last row
        let mut h = seq.clone();
        for l in &m.xf {
            h = l.forward(&h, DIM, 16, &m.device).unwrap();
        }
        let s = h.shape().dims2().unwrap().0;
        let tr_last = h
            .narrow(0, s - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1_f32()
            .unwrap();
        let (ref_tr, _) = load_bin("/tmp/ref_tr_raw_last.bin");
        println!("TR_RAW  cosine={:.5}", cosine(&tr_last, &ref_tr));
        // CONTROLLED: feed the EXACT reference seq into my transformer -> isolate transformer bug.
        let (ref_seq_f, ref_seq_s) = load_bin("/tmp/ref_seq.bin"); // [S,1024]
        let rseq = Tensor::from_vec_f32(ref_seq_f, (ref_seq_s[0], ref_seq_s[1]))
            .unwrap()
            .to_device(&m.device)
            .unwrap();
        let mut h2 = rseq;
        for l in &m.xf {
            h2 = l.forward(&h2, DIM, 16, &m.device).unwrap();
        }
        let s2 = h2.shape().dims2().unwrap().0;
        let tr2 = h2
            .narrow(0, s2 - 1, 1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1_f32()
            .unwrap();
        println!("TR_RAW(ref_seq in) cosine={:.5}", cosine(&tr2, &ref_tr));
        let c = m.backbone_last(&seq).unwrap();
        let cv = c.flatten_all().unwrap().to_vec1_f32().unwrap();
        let (ref_c, _) = load_bin("/tmp/ref_c.bin");
        println!("C       cosine={:.5}", cosine(&cv, &ref_c));
        // flow_net velocity at x0=0, s=0, t=1
        let x0 = Tensor::from_vec_f32(vec![0f32; LATENT_DIM], (1, LATENT_DIM)).unwrap();
        let v = m.flow.velocity(&c, 0.0, 1.0, &x0).unwrap();
        let vvel = v.flatten_all().unwrap().to_vec1_f32().unwrap();
        let (ref_vel, _) = load_bin("/tmp/ref_velocity.bin");
        println!("VELOCITY cosine={:.5}", cosine(&vvel, &ref_vel));
    }

    #[test]
    #[ignore = "parity: compare encode/decode against /tmp/ref_*.bin from ref_dump.py - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn mimi_parity_vs_ref() {
        let dev = Device::Cpu;
        let dec = MimiDecoder::from_safetensors(ckpt().to_str().unwrap(), dev.clone()).unwrap();
        let enc = MimiEncoder::from_safetensors(ckpt().to_str().unwrap(), dev).unwrap();
        // 1) DECODER: feed the reference latent, compare audio to the reference round-trip.
        let (lat_f, lat_s) = load_bin("/tmp/ref_latent.bin"); // [512,25]
        let lat = Tensor::from_vec_f32(lat_f, (lat_s[0], lat_s[1])).unwrap();
        let my_out = dec
            .decode_from_codec(&lat.reshape((1, lat_s[0], lat_s[1])).unwrap())
            .unwrap();
        let (ref_out, _) = load_bin("/tmp/ref_roundtrip.bin");
        println!(
            "DECODER vs ref: cosine={:.5} (my_len={} ref_len={})",
            cosine(&my_out, &ref_out),
            my_out.len(),
            ref_out.len()
        );
        // 2) ENCODER: encode the dog, compare latent to the reference latent.
        let wav_in = format!(
            "{}/results/ezaudio/a_dog_barking.wav",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut src = read_wav16_mono(&wav_in);
        src.truncate(48000);
        let my_lat = enc.encode(&src).unwrap(); // [1,512,25]
        let my_lat_v = my_lat.flatten_all().unwrap().to_vec1_f32().unwrap();
        let (ref_lat, _) = load_bin("/tmp/ref_latent.bin");
        println!(
            "ENCODER vs ref: cosine={:.5} (my_rms={:.4} ref_rms={:.4})",
            cosine(&my_lat_v, &ref_lat),
            (my_lat_v.iter().map(|x| x * x).sum::<f32>() / my_lat_v.len() as f32).sqrt(),
            (ref_lat.iter().map(|x| x * x).sum::<f32>() / ref_lat.len() as f32).sqrt()
        );
    }

    #[test]
    #[ignore = "needs kyutai/pocket-tts checkpoint + tokenizer (config huggingface_models_dir)"]
    fn pocket_tts_text_to_speech() {
        use crate::inference::token::sentencepiece::SentencePiece;
        let m = PocketTts::from_safetensors(ckpt().to_str().unwrap(), Device::Cpu).unwrap();
        let sp = SentencePiece::from_file(tokenizer_model().to_str().unwrap()).unwrap();
        let enc = MimiEncoder::from_safetensors(ckpt().to_str().unwrap(), Device::Cpu).unwrap();
        let voice_path = format!(
            "{}/results/ezaudio/a_dog_barking.wav",
            env!("CARGO_MANIFEST_DIR")
        );
        let mut vsamp = read_wav16_mono(&voice_path);
        vsamp.truncate(72000); // 3s reference
        let voice = m.voice_from_audio(&enc, &vsamp).unwrap();
        let wav = m
            .synthesize_text(&sp, "Hello, this is a test.", Some(&voice), None, 7)
            .unwrap();
        assert!(!wav.is_empty() && wav.len().is_multiple_of(1920));
        assert!(wav.iter().all(|v| v.is_finite()));
        let peak = wav.iter().fold(0f32, |a, &v| a.max(v.abs()));
        // Write a WAV next to the target dir for manual listening / inspection.
        let path = std::env::temp_dir().join("pocket_tts_out.wav");
        let _ = std::fs::write(
            &path,
            crate::inference::media::audio_io::write_wav(&wav, SAMPLE_RATE),
        );
        println!(
            "text->speech: {} frames -> {} samples ({:.2}s), peak {peak:.3} -> {}",
            wav.len() / 1920,
            wav.len(),
            wav.len() as f32 / SAMPLE_RATE as f32,
            path.display()
        );
        assert!(peak > 0.0);
    }

    #[test]
    #[ignore = "needs kyutai/pocket-tts checkpoint (config huggingface_models_dir)"]
    fn pocket_tts_generates_audio() {
        // Full text->audio with dummy token ids (the sentencepiece tokenizer is
        // Stage 4; here we exercise flow_lm generation + Mimi decode end-to-end).
        let m = PocketTts::from_safetensors(ckpt().to_str().unwrap(), Device::Cpu).unwrap();
        let ids: Vec<u32> = vec![10, 42, 7, 100, 3, 55, 8];
        let wav = m.synthesize(&ids, None, 20, 2, 1234).unwrap();
        assert!(!wav.is_empty(), "produced some audio");
        assert_eq!(wav.len() % 1920, 0, "whole number of 12.5 Hz frames");
        assert!(wav.iter().all(|v| v.is_finite()), "no NaN/Inf");
        let peak = wav.iter().fold(0f32, |a, &v| a.max(v.abs()));
        println!(
            "pocket-tts: {} frames -> {} samples, peak {peak:.3}",
            wav.len() / 1920,
            wav.len()
        );
        assert!(peak > 0.0, "non-silent output");
    }

    #[test]
    #[ignore = "needs kyutai/pocket-tts checkpoint (config huggingface_models_dir)"]
    fn mimi_decoder_loads_and_runs() {
        let dev = Device::Cpu;
        let dec = MimiDecoder::from_safetensors(ckpt().to_str().unwrap(), dev).unwrap();
        // A short random latent -> audio of the expected length, finite samples.
        let t = 25usize; // 2 s @ 12.5 Hz
        let latent = Tensor::from_vec_f32(vec![0.0f32; LATENT_DIM * t], (LATENT_DIM, t)).unwrap();
        let wav = dec.decode(&latent).unwrap();
        assert_eq!(wav.len(), t * 1920, "1920x upsampling (12.5 Hz -> 24 kHz)");
        assert!(wav.iter().all(|v| v.is_finite()), "no NaN/Inf in output");
    }
}
