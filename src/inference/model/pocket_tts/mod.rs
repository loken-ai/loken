//! Kyutai pocket-tts - Stage 1: the Mimi decoder (synthesis path).
//!
//! pocket-tts is a flow-matching TTS over a CONTINUOUS latent (no discrete RVQ codebooks -
//! the checkpoint has a single `quantizer.output_proj` from the latent to the codec width
//! and `emb_mean/emb_std` instead of codebooks). Stage 1 ports the audio synthesis half:
//! a latent frame -> 24 kHz waveform, via
//!   output_proj -> upsample.convtr(depthwise, causal)
//!   -> decoder_transformer(RoPE+LayerScale) -> SEANet decoder(ELU)
//! Every width, layer count and stride is read from the stored tensors. All convs are
//! causal (left-pad) and conv-transposes trim the right tail, matching the streaming model.
//!
//! Net-new vs existing VAEs: ELU activation, LayerScale, fused-qkv codec transformer
//! with RoPE. Reuses `tensor` conv1d/conv_transpose1d.
//!
//! The codec's transformers, in its decoder and its encoder, are transformers of the generic
//! route, placed by `HeteroPlan` and the ladder with their own demand.
//! The rest of the codec sits on the device it is given.
//!
//! The flow model's transformer is a transformer of the generic route too, with its own demand.
//! A generation prefills the voice and the text once, then runs a step per frame on its cache.

use crate::inference::generic_transformer::class_load::{self, Holding, Rows};
use crate::inference::generic_transformer::GenericHeteroTransformer;
use crate::inference::model::adaln::modulate;
use crate::inference::model::kyutai::mimi::{
    ATTENTION_HEADS as MIMI_HEADS, ATTENTION_WINDOW, TRANSFORMER_CLASS as CODEC_CLASS,
};
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Error, Result, Tensor};

/// Where the codec is in the checkpoint.
const CODEC_PREFIX: &str = "mimi";

/// Where the codec's decoder transformer is in the checkpoint.
const DECODER_PREFIX: &str = "mimi.decoder_transformer.transformer";

/// Where the codec's encoder transformer is in the checkpoint.
const ENCODER_PREFIX: &str = "mimi.encoder_transformer.transformer";

pub const SAMPLE_RATE: u32 = 24_000;
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
    /// A conv whose shape is the stored `{name}.weight`, with its bias when `bias`.
    fn load(vb: &VarBuilder, name: &str, bias: bool) -> Result<Self> {
        let shape = vb.shape_of(&format!("{name}.weight"))?;
        let b = if bias {
            Some(vb.get(shape[0], &format!("{name}.bias"))?)
        } else {
            None
        };
        let w = vb.get(shape, &format!("{name}.weight"))?;
        Ok(Self { w, b })
    }
    /// ConvTranspose bias has `out` channels = shape[1] (weight is [in,out,k]).
    fn load_tr(vb: &VarBuilder, name: &str) -> Result<Self> {
        let shape = vb.shape_of(&format!("{name}.weight"))?;
        let b = Some(vb.get(shape[1], &format!("{name}.bias"))?);
        let w = vb.get(shape, &format!("{name}.weight"))?;
        Ok(Self { w, b })
    }
    /// The stride of a SEANet resampling conv, whose kernel is twice its stride.
    fn resampling_stride(&self) -> Result<usize> {
        Ok(self.w.shape().dims3()?.2 / 2)
    }
}

/// SEANet residual block: ELU -> conv(k3) -> ELU -> conv(k1) -> + input.
struct ResBlock {
    conv1: Conv, // [mid, in, 3]
    conv2: Conv, // [in, mid, 1]
}
impl ResBlock {
    fn load(vb: &VarBuilder, idx: usize) -> Result<Self> {
        let p = format!("{idx}.block");
        Ok(Self {
            conv1: Conv::load(vb, &format!("{p}.1.conv"), true)?,
            conv2: Conv::load(vb, &format!("{p}.3.conv"), true)?,
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

/// Mimi decoder: latent frames -> 24 kHz mono waveform.
///
/// The decoder transformer's layers sit where their placement put them.
/// The rest sits on the decoder's device.
pub struct MimiDecoder {
    output_proj: Tensor, // [codec, latent, 1] conv1d latent->codec
    upsample: Tensor,    // [codec, 1, k] depthwise convtr, stride k/2
    layers: CodecTransformer,
    conv_in: Conv,             // model.0
    up: Vec<(Conv, ResBlock)>, // (convtr, resblock) per SEANet stage
    conv_out: Conv,
    device: Device,
}

impl MimiDecoder {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let mimi = vb.pp("mimi");
        let q = mimi.pp("quantizer");
        let output_proj = q.get(q.shape_of("output_proj.weight")?, "output_proj.weight")?;
        let up_name = "upsample.convtr.convtr.weight";
        let upsample = mimi.get(mimi.shape_of(up_name)?, up_name)?;
        let layers = CodecTransformer::load(path, &vb, DECODER_PREFIX, CODEC_CLASS, &device)?;
        // SEANet decoder: `model.0` is the input conv, then each stage is an ELU, a
        // transposed conv and a residual block (three entries), and the output conv
        // follows the last stage's ELU.
        let dec = mimi.pp("decoder.model");
        let conv_in = Conv::load(&dec, "0.conv", true)?;
        let mut up = Vec::new();
        let mut idx = 2;
        while dec.contains(&format!("{idx}.convtr.weight")) {
            up.push((
                Conv::load_tr(&dec, &format!("{idx}.convtr"))?,
                ResBlock::load(&dec, idx + 1)?,
            ));
            idx += 3;
        }
        let conv_out = Conv::load(&dec, &format!("{idx}.conv"), true)?;
        Ok(Self {
            output_proj,
            upsample,
            layers,
            conv_in,
            up,
            conv_out,
            device,
        })
    }

    /// Latent channels one frame carries.
    pub fn latent_dim(&self) -> Result<usize> {
        Ok(self.output_proj.shape().dims3()?.1)
    }

    /// Audio samples one latent frame decodes to: the product of every upsampling stride.
    pub fn frame_samples(&self) -> Result<usize> {
        let mut n = self.upsample.shape().dims3()?.2 / 2;
        for (convtr, _) in &self.up {
            n *= convtr.resampling_stride()?;
        }
        Ok(n)
    }

    /// `latent: [latent, T]` (single batch, conv layout) -> waveform `[T*frame_samples]`.
    pub fn decode(&self, latent: &Tensor) -> Result<Vec<f32>> {
        let (c, t) = latent.shape().dims2()?;
        debug_assert_eq!(c, self.latent_dim()?);
        let x = latent.reshape((1, c, t))?;
        // latent -> codec width (k1 conv = DummyQuantizer.output_proj), then the codec path.
        let x = causal_conv1d(&x, &self.output_proj, None, 1, 1, 1)?;
        self.decode_from_codec(&x)
    }

    /// Decode from the codec embedding `[1, codec, T]` (post-quantizer):
    /// depthwise upsample -> decoder transformer -> SEANet -> 24 kHz waveform.
    pub fn decode_from_codec(&self, emb: &Tensor) -> Result<Vec<f32>> {
        let rows = self.upsampled_rows(emb)?;
        let rows = self.layers.transform(&rows, &self.device)?;
        self.seanet(&rows)
    }

    /// The codec embedding `[1, codec, T]` upsampled to the transformer's rate, `[S, codec]`.
    fn upsampled_rows(&self, emb: &Tensor) -> Result<Tensor> {
        let (codec, up_k) = {
            let (c, _, k) = self.upsample.shape().dims3()?;
            (c, k)
        };
        let x = causal_convtr1d(emb, &self.upsample, None, up_k / 2, codec)?;
        let s = x.shape().dims3()?.2;
        x.reshape((codec, s))?.transpose(0, 1)?.contiguous() // [S,codec]
    }

    /// `rows [S, codec]` through the SEANet decoder: the input conv, then per stage an ELU, a
    /// transposed conv and a residual block, then an ELU and the output conv.
    fn seanet(&self, rows: &Tensor) -> Result<Vec<f32>> {
        let (s, codec) = rows.shape().dims2()?;
        let mut x = rows.transpose(0, 1)?.contiguous()?.reshape((1, codec, s))?;
        x = causal_conv1d(&x, &self.conv_in.w, self.conv_in.b.as_ref(), 1, 1, 1)?;
        for (convtr, res) in &self.up {
            x = elu(&x)?;
            let stride = convtr.resampling_stride()?;
            x = causal_convtr1d(&x, &convtr.w, convtr.b.as_ref(), stride, 1)?;
            x = res.forward(&x)?;
        }
        x = elu(&x)?;
        x = causal_conv1d(&x, &self.conv_out.w, self.conv_out.b.as_ref(), 1, 1, 1)?;
        let l = x.shape().dims3()?.2;
        x.reshape((l,))?.to_vec1_f32()
    }
}

/// Mimi encoder (waveform -> codec embedding at the frame rate) - for the decoder
/// round-trip diagnostic and the speaker conditioning.
pub struct MimiEncoder {
    conv0: Conv,
    /// Per SEANet stage: a residual block, then a strided downsample conv.
    stages: Vec<(ResBlock, Conv)>,
    conv_final: Conv,
    layers: CodecTransformer,
    downsample: Tensor, // [codec, codec, k] stride k/2
    device: Device,
}
impl MimiEncoder {
    pub fn from_safetensors(path: &str, device: Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let mimi = vb.pp("mimi");
        let enc = mimi.pp("encoder.model");
        let conv0 = Conv::load(&enc, "0.conv", true)?;
        // Each stage is a residual block, an ELU and a strided conv (three entries); the
        // final conv follows the last stage's ELU.
        let mut stages = Vec::new();
        let mut idx = 1;
        while enc.contains(&format!("{idx}.block.1.conv.weight")) {
            stages.push((
                ResBlock::load(&enc, idx)?,
                Conv::load(&enc, &format!("{}.conv", idx + 2), true)?,
            ));
            idx += 3;
        }
        let conv_final = Conv::load(&enc, &format!("{}.conv", idx + 1), true)?;
        let layers = CodecTransformer::load(path, &vb, ENCODER_PREFIX, CODEC_CLASS, &device)?;
        let down_name = "downsample.conv.conv.weight";
        let downsample = mimi.get(mimi.shape_of(down_name)?, down_name)?;
        Ok(Self {
            conv0,
            stages,
            conv_final,
            layers,
            downsample,
            device,
        })
    }

    /// `samples`: mono f32 @24 kHz -> codec embedding `[1, codec, T]` at the frame rate.
    pub fn encode(&self, samples: &[f32]) -> Result<Tensor> {
        let rows = self.seanet_rows(samples)?;
        let rows = self.layers.transform(&rows, &self.device)?;
        self.downsampled(&rows)
    }

    /// `samples` through the SEANet encoder, `[S, codec]` at the transformer's rate.
    fn seanet_rows(&self, samples: &[f32]) -> Result<Tensor> {
        // Pad to a whole number of frames (the product of every stride) so the last conv
        // window is full - mirrors `pad_for_conv1d` in encode_to_latent.
        let down_stride = self.downsample.shape().dims3()?.2 / 2;
        let mut frame_size = down_stride;
        for (_, conv) in &self.stages {
            frame_size *= conv.resampling_stride()?;
        }
        let mut buf = samples.to_vec();
        let t = buf.len().div_ceil(frame_size) * frame_size;
        buf.resize(t, 0.0);
        let mut x = Tensor::from_vec_f32(buf, (1, 1, t))?.to_device(&self.device)?;
        x = causal_conv1d(&x, &self.conv0.w, self.conv0.b.as_ref(), 1, 1, 1)?;
        for (res, down) in &self.stages {
            x = res.forward(&x)?;
            x = elu(&x)?;
            let stride = down.resampling_stride()?;
            x = causal_conv1d(&x, &down.w, down.b.as_ref(), stride, 1, 1)?;
        }
        x = elu(&x)?;
        x = causal_conv1d(&x, &self.conv_final.w, self.conv_final.b.as_ref(), 1, 1, 1)?;
        let (_, codec, s) = x.shape().dims3()?;
        x.reshape((codec, s))?.transpose(0, 1)?.contiguous()
    }

    /// `rows [S, codec]` downsampled to the frame rate, `[1, codec, T]`.
    fn downsampled(&self, rows: &Tensor) -> Result<Tensor> {
        let down_stride = self.downsample.shape().dims3()?.2 / 2;
        let (s, codec) = rows.shape().dims2()?;
        let x = rows.transpose(0, 1)?.contiguous()?.reshape((1, codec, s))?;
        let downsample_replicate = |x: &Tensor, w: &Tensor| -> Result<Tensor> {
            // ConvDownsample1d uses pad_mode="replicate": left-pad with the first
            // frame repeated (not zeros), then strided conv.
            let k = w.shape().dims3()?.2;
            let pad = k - down_stride; // (k-1)+1-stride
            let first = x.narrow(2, 0, 1)?; // [1,C,1]
            let reps: Vec<&Tensor> = std::iter::repeat_n(&first, pad).collect();
            let mut parts = reps;
            parts.push(x);
            let xp = Tensor::cat(&parts, 2)?;
            xp.conv1d(w, 0, down_stride, 1, 1)
        };
        // downsample to the frame rate (replicate padding)
        downsample_replicate(&x, &self.downsample)
    }
}

/// A transformer of the codec on the generic layers.
struct CodecTransformer {
    layers: std::sync::Mutex<GenericHeteroTransformer>,
}

impl CodecTransformer {
    /// The transformer under `prefix` in the checkpoint `path`, of class `class`.
    ///
    /// Its layers are placed on the cards the ladder sees when `device` is a card, on the host
    /// otherwise.
    fn load(
        path: &str,
        vb: &VarBuilder,
        prefix: &str,
        class: &str,
        device: &Device,
    ) -> Result<Self> {
        let cards = match device.is_cuda() {
            true => crate::inference::place::ladder::cards(0),
            false => Vec::new(),
        };
        let layers = codec_layers(path, vb, prefix, class, &cards)?;
        Ok(Self {
            layers: std::sync::Mutex::new(layers),
        })
    }

    /// `rows [S, codec]` at positions `0..S` through the layers, `[S, codec]` F32 on `device`.
    fn transform(&self, rows: &Tensor, device: &Device) -> Result<Tensor> {
        let out = {
            let mut layers = self
                .layers
                .lock()
                .map_err(|_| Error::msg("the codec's transformer is poisoned"))?;
            layers.forward_embeddings_rows(rows)?
        };
        let out = match out.dtype() {
            DType::F32 => out,
            _ => out.to_dtype(DType::F32)?,
        };
        out.to_device(device)
    }

    /// Where the layers sit, by device.
    fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        match self.layers.lock() {
            Ok(layers) => layers.placed_runs(),
            Err(_) => Vec::new(),
        }
    }
}

/// The configuration of the codec's transformer under `prefix`, of class `class`: the shapes
/// its tensors have and the settings Mimi's reference fixes.
fn codec_config(vb: &VarBuilder, prefix: &str, class: &str) -> Result<serde_json::Value> {
    let xt = vb.pp(prefix);
    let first = xt.pp("layers.0");
    Ok(serde_json::json!({
        "architectures": [class],
        "hidden_size": first.shape_of("norm1.weight")?[0],
        "num_attention_heads": MIMI_HEADS,
        "num_key_value_heads": MIMI_HEADS,
        "intermediate_size": first.shape_of("linear1.weight")?[0],
        "num_hidden_layers": xt.count_children("layers"),
        "sliding_window": ATTENTION_WINDOW,
        "rope_theta": ROPE_THETA,
        "layer_norm_eps": LN_EPS,
    }))
}

/// The layers of the codec's transformer under `prefix` in the checkpoint `path`, placed on
/// `cards`.
///
/// The blocks are read under `prefix` in the codec.
/// A card holds the stored 16-bit float weights at F32, as the host does.
/// The cache holds the rows a layer windowed in every layer keeps.
/// A run takes every row from position 0, whatever its length.
fn codec_layers(
    path: &str,
    vb: &VarBuilder,
    prefix: &str,
    class: &str,
    cards: &[(usize, u64)],
) -> Result<GenericHeteroTransformer> {
    let root = codec_config(vb, prefix, class)?;
    let shards = [path.to_string()];
    let in_codec = prefix
        .strip_prefix(CODEC_PREFIX)
        .and_then(|p| p.strip_prefix('.'))
        .unwrap_or(prefix);
    let blocks = format!("{in_codec}.layers");
    let holding = Holding {
        blocks: Some(&blocks),
        wide_floats: true,
    };
    let held = class_load::class_demand_held(&root, &shards, class, 0, holding)?
        .held_rows
        .ok_or_else(|| Error::msg("the codec's transformer is windowed in every layer"))?;
    let loaded = class_load::load_class_held(
        &root,
        &shards,
        class,
        Rows::Exactly(held as usize),
        0,
        cards,
        holding,
    )?;
    Ok(loaded.model)
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

/// The flow_lm transformer's head count, which pocket-tts's published model settings fix
/// and its checkpoint does not store.
const FLOW_LM_HEADS: usize = 16;
const EOS_THRESHOLD: f32 = -4.0;
const TEMP: f32 = 0.7;

/// Where the flow model is in the checkpoint.
const FLOW_PREFIX: &str = "flow_lm";

/// The class of the flow model, whose transformer and output norm run on the generic layers.
const FLOW_CLASS: &str = "FlowLMModel";

/// The flow model's transformer and the norm after it, on the generic layers.
struct FlowTransformer {
    layers: std::sync::Mutex<GenericHeteroTransformer>,
    /// Rows the cache holds.
    context: usize,
}

impl FlowTransformer {
    /// The flow model's transformer in the checkpoint `path`.
    ///
    /// Its layers are placed on the cards the ladder sees when `device` is a card, on the host
    /// otherwise.
    /// Its cache holds up to `rows` rows.
    fn load(path: &str, vb: &VarBuilder, device: &Device, rows: usize) -> Result<Self> {
        let cards = match device.is_cuda() {
            true => crate::inference::place::ladder::cards(0),
            false => Vec::new(),
        };
        let (layers, context) = flow_layers(path, vb, &cards, rows)?;
        Ok(Self {
            layers: std::sync::Mutex::new(layers),
            context,
        })
    }

    /// The layers, held for the whole of a generation: their cache is its state.
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, GenericHeteroTransformer>> {
        self.layers
            .lock()
            .map_err(|_| Error::msg("the flow model's transformer is poisoned"))
    }

    /// Where the layers sit, by device.
    fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        match self.layers.lock() {
            Ok(layers) => layers.placed_runs(),
            Err(_) => Vec::new(),
        }
    }
}

/// `rows [S, dim]` prefilled from position 0, the cache emptied first.
/// Gives the last row after the output norm, `[1, dim]` F32 on `device`.
fn flow_prefill(
    layers: &mut GenericHeteroTransformer,
    rows: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let out = layers.forward_embeddings_prefill(rows, false)?;
    out.normed.to_device(device)
}

/// The row `row [1, dim]` at position `pos` after the cached rows.
/// Gives it after the output norm, `[1, dim]` F32 on `device`.
fn flow_step(
    layers: &mut GenericHeteroTransformer,
    row: &Tensor,
    pos: usize,
    device: &Device,
) -> Result<Tensor> {
    let out = layers.forward_embedding_step(row, pos, false)?;
    out.normed.to_device(device)
}

/// The configuration of the flow model's transformer: the shapes its tensors have and the
/// settings pocket-tts's published model fixes.
fn flow_config(vb: &VarBuilder) -> Result<serde_json::Value> {
    let xt = vb.pp(FLOW_PREFIX).pp("transformer");
    let first = xt.pp("layers.0");
    Ok(serde_json::json!({
        "architectures": [FLOW_CLASS],
        "hidden_size": first.shape_of("norm1.weight")?[0],
        "num_attention_heads": FLOW_LM_HEADS,
        "num_key_value_heads": FLOW_LM_HEADS,
        "intermediate_size": first.shape_of("linear1.weight")?[0],
        "num_hidden_layers": xt.count_children("layers"),
        "rope_theta": ROPE_THETA,
        "layer_norm_eps": LN_EPS,
    }))
}

/// The layers of the flow model's transformer in the checkpoint `path`, placed on `cards`, and
/// the rows their cache holds.
///
/// A card holds the stored 16-bit float weights at F32, as the host does.
/// The cache holds up to `rows` rows: as many as the cards hold with every layer on them, and
/// `rows` on the host.
/// It grows as a generation fills it.
fn flow_layers(
    path: &str,
    vb: &VarBuilder,
    cards: &[(usize, u64)],
    rows: usize,
) -> Result<(GenericHeteroTransformer, usize)> {
    let root = flow_config(vb)?;
    let loaded = class_load::load_class_held(
        &root,
        &[path.to_string()],
        FLOW_CLASS,
        Rows::UpTo(rows),
        0,
        cards,
        Holding {
            blocks: None,
            wide_floats: true,
        },
    )?;
    Ok((loaded.model, loaded.context))
}

/// `y = x @ wᵀ + b` for a 2-D `x [N, in]`, `w [out, in]`, `b [out]`.
fn linear_b(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let out = w.shape().dims2()?.0;
    x.matmul_t(w)?.broadcast_add(&b.reshape((1, out))?)
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
    freqs: Vec<f32>, // [half]
    w0: Tensor,
    b0: Tensor, // [flow, 2*half],[flow]
    w2: Tensor,
    b2: Tensor,    // [flow,flow],[flow]
    alpha: Tensor, // RMSNorm [flow]
}
impl TimeEmbed {
    fn load(vb: &VarBuilder, flow: usize) -> Result<Self> {
        let half = vb.shape_of("freqs")?[0];
        Ok(Self {
            freqs: vb.get(half, "freqs")?.to_vec1_f32()?,
            w0: vb.get((flow, 2 * half), "mlp.0.weight")?,
            b0: vb.get(flow, "mlp.0.bias")?,
            w2: vb.get((flow, flow), "mlp.2.weight")?,
            b2: vb.get(flow, "mlp.2.bias")?,
            alpha: vb.get(flow, "mlp.3.alpha")?,
        })
    }
    fn forward(&self, t: f32, dev: &Device) -> Result<Tensor> {
        // emb = cat(cos(t*freqs), sin(t*freqs)) -> [1, 2*half]
        let half = self.freqs.len();
        let width = 2 * half;
        let mut emb = vec![0f32; width];
        for (i, &f) in self.freqs.iter().enumerate() {
            emb[i] = (t * f).cos();
            emb[half + i] = (t * f).sin();
        }
        let e = Tensor::from_vec_f32(emb, (1, width))?.to_device(dev)?;
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
    ada_b: Tensor, // [3*flow,flow],[3*flow]
    flow: usize,
}
impl FlowResBlock {
    fn load(vb: &VarBuilder, flow: usize) -> Result<Self> {
        Ok(Self {
            in_ln_w: vb.get(flow, "in_ln.weight")?,
            in_ln_b: vb.get(flow, "in_ln.bias")?,
            mlp0_w: vb.get((flow, flow), "mlp.0.weight")?,
            mlp0_b: vb.get(flow, "mlp.0.bias")?,
            mlp2_w: vb.get((flow, flow), "mlp.2.weight")?,
            mlp2_b: vb.get(flow, "mlp.2.bias")?,
            ada_w: vb.get((3 * flow, flow), "adaLN_modulation.1.weight")?,
            ada_b: vb.get(3 * flow, "adaLN_modulation.1.bias")?,
            flow,
        })
    }
    fn forward(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        let flow = self.flow;
        let m = linear_b(&y.silu()?, &self.ada_w, &self.ada_b)?; // [1,3*flow]
        let shift = m.narrow(1, 0, flow)?;
        let scale = m.narrow(1, flow, flow)?;
        let gate = m.narrow(1, 2 * flow, flow)?;
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
    input_proj_b: Tensor, // [flow,latent]
    cond_w: Tensor,
    cond_b: Tensor, // [flow,dim]
    time: Vec<TimeEmbed>,
    blocks: Vec<FlowResBlock>,
    fnorm_ones: Tensor, // ones[flow] for no-affine norm_final
    flin_ada_w: Tensor,
    flin_ada_b: Tensor, // [2*flow,flow]
    flin_w: Tensor,
    flin_b: Tensor, // [latent,flow]
    flow: usize,
    device: Device,
}
impl FlowNet {
    fn load(vb: &VarBuilder, device: Device) -> Result<Self> {
        let [flow, latent] = vb.shape_of("input_proj.weight")?[..] else {
            return Err(crate::tensor::Error(
                "pocket-tts: input_proj is not a matrix".into(),
            ));
        };
        let dim = vb.shape_of("cond_embed.weight")?[1];
        let time = (0..vb.count_children("time_embed"))
            .map(|i| TimeEmbed::load(&vb.pp(format!("time_embed.{i}")), flow))
            .collect::<Result<Vec<_>>>()?;
        let blocks = (0..vb.count_children("res_blocks"))
            .map(|i| FlowResBlock::load(&vb.pp(format!("res_blocks.{i}")), flow))
            .collect::<Result<Vec<_>>>()?;
        let fl = vb.pp("final_layer");
        Ok(Self {
            input_proj_w: vb.get((flow, latent), "input_proj.weight")?,
            input_proj_b: vb.get(flow, "input_proj.bias")?,
            cond_w: vb.get((flow, dim), "cond_embed.weight")?,
            cond_b: vb.get(flow, "cond_embed.bias")?,
            time,
            blocks,
            fnorm_ones: Tensor::from_vec_f32(vec![1f32; flow], flow)?.to_device(&device)?,
            flin_ada_w: fl.get((2 * flow, flow), "adaLN_modulation.1.weight")?,
            flin_ada_b: fl.get(2 * flow, "adaLN_modulation.1.bias")?,
            flin_w: fl.get((latent, flow), "linear.weight")?,
            flin_b: fl.get(latent, "linear.bias")?,
            flow,
            device,
        })
    }
    /// velocity `[1,latent]` given conditioning `c [1,dim]`, times s,t, latent `x [1,latent]`.
    fn velocity(&self, c: &Tensor, s: f32, t: f32, x: &Tensor) -> Result<Tensor> {
        let mut h = linear_b(x, &self.input_proj_w, &self.input_proj_b)?; // [1,flow]
                                                                          // The time embedders take the two ends of the step in turn, and their mean
                                                                          // conditions the net.
        let mut tc: Option<Tensor> = None;
        for (embed, at) in self.time.iter().zip([s, t]) {
            let e = embed.forward(at, &self.device)?;
            tc = Some(match tc {
                None => e,
                Some(acc) => acc.add(&e)?,
            });
        }
        let tc = tc
            .ok_or_else(|| crate::tensor::Error("pocket-tts: no time embedder".into()))?
            .affine(1.0 / self.time.len() as f32, 0.0)?;
        let ce = linear_b(c, &self.cond_w, &self.cond_b)?;
        let y = tc.add(&ce)?; // [1,flow]
        for b in &self.blocks {
            h = b.forward(&h, &y)?;
        }
        // final layer (norm_final has no affine)
        let m = linear_b(&y.silu()?, &self.flin_ada_w, &self.flin_ada_b)?; // [1,2*flow]
        let shift = m.narrow(1, 0, self.flow)?;
        let scale = m.narrow(1, self.flow, self.flow)?;
        let hn = h.layer_norm(&self.fnorm_ones, None, 1e-6)?;
        let hn = modulate(&hn, &shift, &scale)?;
        linear_b(&hn, &self.flin_w, &self.flin_b)
    }
}

/// Leading spaces the reference puts before a text of fewer than 5 words.
const SHORT_TEXT_PAD: usize = 8;

/// The frames the reference allows a text of `tokens` tokens at `frame_rate` frames a second.
fn frame_budget(tokens: usize, frame_rate: f32) -> usize {
    (((tokens as f32 / 3.0) + 2.0) * frame_rate).ceil() as usize
}

/// Whether the tensor `name` belongs to a transformer the generic route places: the flow
/// model's, with its output norm, or one of the codec's.
pub fn placed_by_the_route(name: &str) -> bool {
    let flow_layers = format!("{FLOW_PREFIX}.transformer.layers.");
    let flow_norm = format!("{FLOW_PREFIX}.out_norm.");
    [DECODER_PREFIX, ENCODER_PREFIX]
        .iter()
        .any(|p| name.starts_with(&format!("{p}.layers.")))
        || name.starts_with(&flow_layers)
        || name.starts_with(&flow_norm)
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
        t = " ".repeat(SHORT_TEXT_PAD) + &t;
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
    embed: Tensor,        // [vocab,dim] text token embedding
    input_linear: Tensor, // [dim,latent]
    /// The transformer and its output norm, placed with their own demand.
    xf: FlowTransformer,
    out_eos_w: Tensor,
    out_eos_b: Tensor,    // [1,dim],[1]
    bos_emb: Tensor,      // [latent]
    emb_mean: Vec<f32>,   // [latent]
    emb_std: Vec<f32>,    // [latent]
    speaker_proj: Tensor, // [dim,codec] voice conditioning projection
    flow: FlowNet,
    mimi: MimiDecoder,
    /// Channels of one latent frame.
    latent: usize,
    device: Device,
}

impl PocketTts {
    /// The model in the checkpoint `path` on `device`.
    ///
    /// The flow model's cache is sized for a call on a text of `text_chars` characters with a
    /// voice clip of `voice_samples` samples: the voice's frames, a token per character and the
    /// short text's padding, the start frame, and the frames the reference allows the text.
    pub fn from_safetensors(
        path: &str,
        device: Device,
        text_chars: usize,
        voice_samples: usize,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[path], crate::tensor::DType::F32, &device)? };
        let f = vb.pp(FLOW_PREFIX);
        let mimi = MimiDecoder::from_safetensors(path, device.clone())?;
        let frame = mimi.frame_samples()?;
        let tokens = text_chars + SHORT_TEXT_PAD + 1;
        let rows = voice_samples.div_ceil(frame)
            + tokens
            + frame_budget(tokens, SAMPLE_RATE as f32 / frame as f32);
        let xf = FlowTransformer::load(path, &vb, &device, rows)?;
        let flow = FlowNet::load(&f.pp("flow_net"), device.clone())?;
        let [dim, latent] = f.shape_of("input_linear.weight")?[..] else {
            return Err(crate::tensor::Error(
                "pocket-tts: input_linear is not a matrix".into(),
            ));
        };
        let vocab = f.shape_of("conditioner.embed.weight")?[0];
        let codec = f.shape_of("speaker_proj_weight")?[1];
        Ok(Self {
            embed: f.get((vocab, dim), "conditioner.embed.weight")?,
            input_linear: f.get((dim, latent), "input_linear.weight")?,
            xf,
            out_eos_w: f.get((1, dim), "out_eos.weight")?,
            out_eos_b: f.get(1, "out_eos.bias")?,
            bos_emb: f.get(latent, "bos_emb")?,
            emb_mean: f.get(latent, "emb_mean")?.to_vec1_f32()?,
            emb_std: f.get(latent, "emb_std")?.to_vec1_f32()?,
            speaker_proj: f.get((dim, codec), "speaker_proj_weight")?,
            flow,
            mimi,
            latent,
            device,
        })
    }

    /// Build a voice conditioning `[T_spk, dim]` from a 24 kHz mono reference clip:
    /// `speaker_proj( mimi_encode(ref) )`.
    pub fn voice_from_audio(&self, enc: &MimiEncoder, samples: &[f32]) -> Result<Tensor> {
        let emb = enc.encode(samples)?; // [1,codec,T]
        let (_, c, t) = emb.shape().dims3()?;
        let lat = emb.reshape((c, t))?.transpose(0, 1)?.contiguous()?; // [T,codec]
        lat.matmul_t(&self.speaker_proj) // [T,dim]
    }

    /// Embed text token ids -> `[T,dim]`.
    fn embed_text(&self, ids: &[u32]) -> Result<Tensor> {
        let rows: Vec<Tensor> = ids
            .iter()
            .map(|&id| self.embed.narrow(0, id as usize, 1))
            .collect::<Result<_>>()?;
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&refs, 0)
    }

    /// The rows a generation prefills, `[S, dim]`: the voice when given, then the text, then
    /// the start frame's row.
    /// The voice comes first, as the reference caches it before the text.
    fn prompt_rows(&self, text_ids: &[u32], voice: Option<&Tensor>) -> Result<Tensor> {
        let text_emb = self.embed_text(text_ids)?; // [T,dim]
        let bos = self.frame_row(&self.bos_emb.reshape((1, self.latent))?)?;
        let mut parts: Vec<&Tensor> = Vec::new();
        if let Some(v) = voice {
            parts.push(v);
        }
        parts.push(&text_emb);
        parts.push(&bos);
        Tensor::cat(&parts, 0)
    }

    /// The transformer's row of a latent frame `[1, latent]`, `[1, dim]`.
    fn frame_row(&self, latent: &Tensor) -> Result<Tensor> {
        latent.matmul_t(&self.input_linear)
    }

    /// Generate normalized latent frames for `text_ids`, up to `max_frames`,
    /// stopping `frames_after_eos` frames past the EOS logit crossing.
    /// `seed` makes the flow-matching noise reproducible.
    ///
    /// The prompt is prefilled once.
    /// Each frame after the first runs 1 step fed the previous frame's row, on the cache.
    /// The frames stop where the cache is full, its size bounded by the cards' memory.
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
        let prompt = self.prompt_rows(text_ids, voice)?;
        let rows = prompt.dims2()?.0;
        let room = (self.xf.context + 1).saturating_sub(rows);
        if room == 0 {
            return Err(Error::msg(format!(
                "a prompt of {rows} rows leaves no frame in the flow model's cache of {} rows, the most the cards' memory holds",
                self.xf.context
            )));
        }
        if max_frames > room {
            tracing::warn!(
                "pocket-tts: {max_frames} frames asked, the flow model's cache of {} rows, the most the cards' memory holds, takes {room}",
                self.xf.context
            );
        }
        let frames = max_frames.min(room);
        let mut layers = self.xf.lock()?;
        let mut c = flow_prefill(&mut layers, &prompt, &self.device)?; // [1,dim]
        let mut latents: Vec<Vec<f32>> = Vec::new();
        let mut eos_step: Option<usize> = None;
        for step in 0..frames {
            // Autoregressive decode runs in spawn_blocking, so a dropped request can
            // only be observed here. See `cancel::scoped` for why this loop reads a
            // thread-published token instead of taking one.
            crate::inference::serve::cancel::scoped::bail()?;
            let eos = linear_b(&c, &self.out_eos_w, &self.out_eos_b)?
                .flatten_all()?
                .to_vec1_f32()?[0];
            if eos > EOS_THRESHOLD && eos_step.is_none() {
                eos_step = Some(step);
            }
            // flow-matching: 1 Euler step from noise N(0, temp)
            let noise = gauss_noise(self.latent, TEMP.sqrt(), &mut rng);
            let x0 = Tensor::from_vec_f32(noise, (1, self.latent))?.to_device(&self.device)?;
            let v = self.flow.velocity(&c, 0.0, 1.0, &x0)?;
            let latent = x0.add(&v)?; // normalized latent [1,latent]
            latents.push(latent.reshape((self.latent,))?.to_vec1_f32()?);
            if let Some(es) = eos_step {
                if step >= es + frames_after_eos {
                    break;
                }
            }
            if step + 1 < frames {
                // The next frame's condition: this frame's row after the cached ones.
                let row = self.frame_row(&latent)?;
                c = flow_step(&mut layers, &row, rows + step, &self.device)?;
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
        // Reference cap: ceil((n_tokens/3 + 2) * frames/s).
        let frame_rate = SAMPLE_RATE as f32 / self.mimi.frame_samples()? as f32;
        let max_frames = frame_budget(ids.len(), frame_rate);
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
        // Denormalize (z*std+mean) and stack into [latent, n] (conv layout).
        let n = latents.len();
        let mut data = vec![0f32; self.latent * n];
        for (j, lat) in latents.iter().enumerate() {
            for c in 0..self.latent {
                data[c * n + j] = lat[c] * self.emb_std[c] + self.emb_mean[c];
            }
        }
        let lat = Tensor::from_vec_f32(data, (self.latent, n))?.to_device(&self.device)?;
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
        let m =
            PocketTts::from_safetensors(ckpt().to_str().unwrap(), Device::Cpu, 64, 72_000).unwrap();
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
            .reshape((1, m.latent))
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
        let c = flow_prefill(&mut m.xf.lock().unwrap(), &seq, &m.device).unwrap();
        let cv = c.flatten_all().unwrap().to_vec1_f32().unwrap();
        let (ref_c, _) = load_bin("/tmp/ref_c.bin");
        println!("C       cosine={:.5}", cosine(&cv, &ref_c));
        // flow_net velocity at x0=0, s=0, t=1
        let x0 = Tensor::from_vec_f32(vec![0f32; m.latent], (1, m.latent)).unwrap();
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
        let m =
            PocketTts::from_safetensors(ckpt().to_str().unwrap(), Device::Cpu, 64, 72_000).unwrap();
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
        let m =
            PocketTts::from_safetensors(ckpt().to_str().unwrap(), Device::Cpu, 64, 72_000).unwrap();
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
        let latent_dim = dec.latent_dim().unwrap();
        let latent = Tensor::from_vec_f32(vec![0.0f32; latent_dim * t], (latent_dim, t)).unwrap();
        let wav = dec.decode(&latent).unwrap();
        assert_eq!(wav.len(), t * 1920, "1920x upsampling (12.5 Hz -> 24 kHz)");
        assert!(wav.iter().all(|v| v.is_finite()), "no NaN/Inf in output");
    }
}

impl PocketTts {
    /// Where the flow model's transformer's layers sit, by device.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        self.xf.placement()
    }
}

impl PocketTts {
    /// Where the codec's decoder transformer's layers sit, by device.
    pub fn codec_transformer_placement(
        &self,
    ) -> Vec<crate::inference::serve::progress::placement::Placed> {
        self.mimi.layers.placement()
    }
}

impl MimiDecoder {
    /// Where the decoder sits: 1 block on its device, the transformer's layers aside.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, 1)
    }
}

impl MimiEncoder {
    /// Where the encoder sits: 1 block on its device, the transformer's layers aside.
    pub fn placement(&self) -> Vec<crate::inference::serve::progress::placement::Placed> {
        crate::inference::serve::progress::placement::whole(&self.device, 1)
    }

    /// Where the encoder transformer's layers sit, by device.
    pub fn transformer_placement(
        &self,
    ) -> Vec<crate::inference::serve::progress::placement::Placed> {
        self.layers.placement()
    }
}

#[cfg(test)]
mod codec_tests {
    //! The codec of the published checkpoint, on the host and on a card.
    //!
    //! `generic_transformer::mimi_class_tests` holds the transformers' layers to a plain host
    //! run in F64.

    use super::*;
    use crate::inference::generic_transformer::class_load::tests::values;

    /// The rows of a long run: beyond the window and a prefill chunk.
    fn long_rows() -> usize {
        crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS + 2 * ATTENTION_WINDOW + 3
    }

    /// The published checkpoint, `None` when the test store lacks it.
    fn published() -> Option<String> {
        let file = Some(resolve_paths().0).filter(|f| f.exists());
        if file.is_none() {
            eprintln!("the published checkpoint is not in the test store");
        }
        file.map(|f| f.to_string_lossy().to_string())
    }

    /// The published codec on `device` decodes every frame of a fixed latent and encodes a
    /// fixed clip to finite values, the same on a second run.
    fn published_runs(device: Device) {
        let Some(path) = published() else { return };
        let dec = MimiDecoder::from_safetensors(&path, device.clone()).unwrap();
        // A fixed latent whose rows pass the window and a prefill chunk.
        let frames = long_rows().div_ceil(2);
        let latent_dim = dec.latent_dim().unwrap();
        let mut seed = 11u32;
        let latent = Tensor::from_vec(
            values(&mut seed, latent_dim * frames, 0.0, 1.0),
            (latent_dim, frames),
            &Device::Cpu,
        )
        .unwrap()
        .to_device(&device)
        .unwrap();
        let wave = dec.decode(&latent).unwrap();
        assert_eq!(wave.len(), frames * dec.frame_samples().unwrap());
        assert!(wave.iter().all(|v| v.is_finite()));
        assert_eq!(dec.decode(&latent).unwrap(), wave);
        drop(dec);
        let enc = MimiEncoder::from_safetensors(&path, device).unwrap();
        // A fixed clip of the reference length the engine keeps.
        let clip: Vec<f32> = (0..144_000)
            .map(|i| 0.3 * (i as f32 * 0.0137).sin() + 0.1 * (i as f32 * 0.271).sin())
            .collect();
        let emb = enc
            .encode(&clip)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1_f32()
            .unwrap();
        assert!(!emb.is_empty() && emb.iter().all(|v| v.is_finite()));
        let again = enc
            .encode(&clip)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1_f32()
            .unwrap();
        assert_eq!(again, emb);
    }

    /// On the host.
    #[test]
    fn the_published_codec_decodes_and_encodes() {
        published_runs(Device::Cpu);
    }

    /// With the codec and the transformers' layers on the cards.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_card_published_codec_decodes_and_encodes() {
        if crate::inference::place::ladder::cards(0).is_empty() {
            eprintln!("no card");
            return;
        }
        published_runs(Device::new_cuda(0).unwrap());
    }
}

#[cfg(test)]
mod flow_tests {
    //! The flow model's transformer of the published checkpoint, on the host and on a card.
    //!
    //! It prefills a prompt and runs a step per frame on its cache.
    //! `generic_transformer::mimi_class_tests` holds its layers to a plain host run in F64, as a
    //! prefill and as single steps.

    use super::*;
    use crate::inference::generic_transformer::class_load::tests::values;

    /// Rows of a prompt longer than a prefill chunk.
    fn long_prompt() -> usize {
        crate::inference::engine::llm_engine::PREFILL_CHUNK_TOKENS + 3
    }

    /// The frames each run gives after the prompt's.
    const FRAMES: usize = 24;

    /// `t` as F32 values on the host.
    fn host(t: &Tensor) -> Vec<f32> {
        t.to_device(&Device::Cpu)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }

    /// The host view of the file `path` at F32.
    fn view(path: &str) -> VarBuilder {
        unsafe { VarBuilder::from_files(&[path], DType::F32, &Device::Cpu).unwrap() }
    }

    /// The frames of the layers of `path` over `rows [n, dim]`, placed on `cards`: the prompt
    /// prefilled, then a step for each row after it.
    fn generic_frames(
        path: &str,
        rows: &Tensor,
        prompt: usize,
        cards: &[(usize, u64)],
    ) -> Vec<f32> {
        let n = rows.dims2().unwrap().0;
        let (mut layers, context) = flow_layers(path, &view(path), cards, n).unwrap();
        assert!(context >= n);
        assert_eq!(layers.layer_cuda_ordinals().is_empty(), cards.is_empty());
        let first = rows.narrow(0, 0, prompt).unwrap();
        let mut out = host(&flow_prefill(&mut layers, &first, &Device::Cpu).unwrap());
        for pos in prompt..n {
            let row = rows.narrow(0, pos, 1).unwrap();
            out.extend(host(
                &flow_step(&mut layers, &row, pos, &Device::Cpu).unwrap(),
            ));
        }
        out
    }

    /// The rows the cache of the published flow model holds on `cards` when asked for `rows`,
    /// and the layers that plan sends to the host.
    fn planned(path: &str, cards: &[(usize, u64)], rows: usize) -> (usize, usize) {
        let root = flow_config(&view(path)).unwrap();
        let shards = [path.to_string()];
        let holding = Holding {
            blocks: None,
            wide_floats: true,
        };
        let demand = class_load::class_demand_held(&root, &shards, FLOW_CLASS, 0, holding).unwrap();
        let held = demand.rows_on(Rows::UpTo(rows), cards, |_| None);
        let plan = demand.at(held as u64).plan(cards);
        let host = plan
            .segments
            .iter()
            .filter(|s| {
                !matches!(
                    s.kind,
                    crate::inference::place::layer_executor::DeviceKind::Cuda(_)
                )
            })
            .map(|s| s.num_layers())
            .sum();
        (held, host)
    }

    /// A dry plan of the cache a call on the longest text asks: a large card holds every row, a
    /// small card holds fewer rows with every layer on it, the host holds every row.
    #[test]
    fn the_flow_cache_is_sized_by_the_cards() {
        let Some(path) = published() else { return };
        const GB: u64 = 1 << 30;
        let rows = 1 << 18;
        assert_eq!(planned(&path, &[(0, 80 * GB)], rows), (rows, 0));
        let (small, host) = planned(&path, &[(0, 4 * GB)], rows);
        assert!(small < rows && small >= crate::inference::cache::KV_WORKING_WINDOW_TOKENS);
        assert_eq!(host, 0);
        assert_eq!(planned(&path, &[], rows).0, rows);
    }

    /// A call past the route's working window runs on the host: its last frames are finite.
    #[test]
    fn a_long_call_runs_past_the_working_window_on_the_host() {
        let Some(path) = published() else { return };
        let (rows, _) = published_rows(&path);
        let parts: Vec<&Tensor> = std::iter::repeat_n(&rows, 8).collect();
        let rows = Tensor::cat(&parts, 0).unwrap();
        let n = rows.dims2().unwrap().0;
        let prompt = n - FRAMES;
        assert!(prompt > crate::inference::cache::KV_WORKING_WINDOW_TOKENS);
        let frames = generic_frames(&path, &rows, prompt, &[]);
        assert!(frames.iter().all(|v| v.is_finite()));
    }

    /// The rows a generation feeds the published flow model: text rows for a prompt longer
    /// than a prefill chunk, the start frame's row, then the rows of fixed latent frames.
    fn published_rows(path: &str) -> (Tensor, usize) {
        let vb = view(path);
        let f = vb.pp(FLOW_PREFIX);
        let [vocab, dim] = f.shape_of("conditioner.embed.weight").unwrap()[..] else {
            panic!("the text table is not a matrix")
        };
        let latent = f.shape_of("input_linear.weight").unwrap()[1];
        let table = f.get((vocab, dim), "conditioner.embed.weight").unwrap();
        let input = f.get((dim, latent), "input_linear.weight").unwrap();
        let ids: Vec<u32> = (0..long_prompt() - 1)
            .map(|i| (i * 37 % vocab) as u32)
            .collect();
        let count = ids.len();
        let ids = Tensor::from_vec(ids, count, &Device::Cpu).unwrap();
        let text = table.index_select(&ids, 0).unwrap();
        let bos = f
            .get(latent, "bos_emb")
            .unwrap()
            .reshape((1, latent))
            .unwrap();
        let mut seed = 13u32;
        let frames = Tensor::from_vec(
            values(&mut seed, FRAMES * latent, 0.0, 1.0),
            (FRAMES, latent),
            &Device::Cpu,
        )
        .unwrap();
        let audio = Tensor::cat(&[&bos, &frames], 0)
            .unwrap()
            .matmul_t(&input)
            .unwrap();
        (Tensor::cat(&[&text, &audio], 0).unwrap(), long_prompt())
    }

    /// The published checkpoint, `None` when the test store lacks it.
    fn published() -> Option<String> {
        let file = Some(resolve_paths().0).filter(|f| f.exists());
        if file.is_none() {
            eprintln!("the published checkpoint is not in the test store");
        }
        file.map(|f| f.to_string_lossy().to_string())
    }

    /// The published flow model's frames on `cards` are finite, the same on a second run.
    fn published_runs(cards: &[(usize, u64)]) {
        let Some(path) = published() else { return };
        let (rows, prompt) = published_rows(&path);
        let frames = generic_frames(&path, &rows, prompt, cards);
        assert_eq!(frames.len(), (FRAMES + 1) * rows.dims2().unwrap().1);
        assert!(frames.iter().all(|v| v.is_finite()));
        assert_eq!(generic_frames(&path, &rows, prompt, cards), frames);
    }

    /// On the host.
    #[test]
    fn the_published_flow_transformer_prefills_and_steps() {
        published_runs(&[]);
    }

    /// With the layers on the cards.
    #[cfg(feature = "gpu")]
    #[test]
    fn the_card_published_flow_transformer_prefills_and_steps() {
        let cards = crate::inference::place::ladder::cards(0);
        if cards.is_empty() {
            eprintln!("no card");
            return;
        }
        published_runs(&cards);
    }
}
