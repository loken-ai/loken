//! The codec decoder of Qwen3-TTS: frames of codes to the waveform.
//!
//! The decode runs in 2 parts.
//! The front dequantises the codes, runs a causal conv and the transformer at the frame rate.
//! The transformer's layers are a transformer of the generic route, placed apart from the rest.
//! The back runs the upsample stages and the snake stack up to the sample rate.
//! Each part decodes in chunks whose left context covers its receptive field, so a chunked
//! decode gives the samples of a whole one.
//! The frames of a chunk are the most whose working set fits the reserve the block is placed
//! with.
//!
//! A [`DecodeStream`] takes frames as they are drawn and gives the samples of the new ones.

pub mod blocks;
pub mod config;
#[cfg(test)]
mod tests;
pub mod transformer;

pub use config::CodecConfig;

use crate::inference::model::codec_blocks::{max_scalar, Rvq};
use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Error, Result, Tensor};
use blocks::{Conv, Snake, SnakeBlock, StageBytes, Upsample};
use config::{frames_within, left_context, Reach};
use transformer::Transformer;

/// The epsilon the usage of a codebook entry is floored to before it divides the entry.
pub const RVQ_EPS: f32 = 1e-5;
/// The bound of the waveform's magnitude.
pub const WAVEFORM_BOUND: f32 = 1.0;

/// The directory of the codec under a checkpoint.
const CODEC_DIR: &str = "speech_tokenizer";
/// The prefix of the decoder's tensors in the codec's file.
const DECODER_PREFIX: &str = "decoder.";

/// Frames a pass of each part holds, left context included, and the left context of each.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chunking {
    pub front_frames: usize,
    pub front_context: usize,
    pub back_frames: usize,
    pub back_context: usize,
}

/// Sizes of the stages of a decoder, which set the rounding of a product chain through them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StageWidths {
    /// Taps of the conv after the quantizer.
    pub pre_conv_kernel: usize,
    /// Upsample stages, and the widest feed-forward among them.
    pub upsample_layers: usize,
    pub upsample_width: usize,
    /// Taps of the first conv of the back.
    pub conv_in_kernel: usize,
    /// For each block of the snake stack, the products of its terms in its transposed conv and
    /// its residual units.
    pub blocks: Vec<BlockWidth>,
    /// Terms of a product of the last conv.
    pub conv_out_width: usize,
}

/// A block of the snake stack: the terms of a product of its transposed conv and the units it
/// holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockWidth {
    pub convtr_width: usize,
    pub units: usize,
}

/// The codec decoder: its transformer's layers placed by their own demand, the rest placed
/// whole on one device.
pub struct CodecDecoder {
    config: CodecConfig,
    rvq_first: Rvq,
    rvq_rest: Rvq,
    pre_conv: Conv,
    transformer: Transformer,
    upsample: Vec<Upsample>,
    conv_in: Conv,
    blocks: Vec<SnakeBlock>,
    act_out: Snake,
    conv_out: Conv,
    device: Device,
    chunking: Chunking,
}

/// The `[from, end)` frames of each pass that keeps the frames from `start` to `total`, and
/// the frame its kept output starts at.
///
/// A pass holds at most `span` frames.
/// A pass starts `context` frames before the frames it keeps, or at frame 0.
fn passes(start: usize, total: usize, span: usize, context: usize) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    let mut start = start;
    while start < total {
        let from = start.saturating_sub(context);
        let end = from.saturating_add(span).min(total);
        out.push((from, start, end));
        start = end;
    }
    out
}

impl CodecDecoder {
    /// The decoder of the checkpoint at `dir`.
    ///
    /// The transformer's layers are placed through `HeteroPlan` and the ladder with their own
    /// demand.
    /// The rest is placed whole through the ladder with the reserve its load needs.
    /// The chunks of the decode derive from that reserve.
    pub fn load(dir: &std::path::Path) -> Result<Self> {
        let codec = dir.join(CODEC_DIR);
        let (root, config) = read_config(&codec)?;
        let weights = weights_of(dir);
        let cards = crate::inference::place::ladder::cards(0);
        let layers = Transformer::layers(&root, &weights, config.max_positions, &cards)?;
        let bytes = whole_bytes(&weights)?;
        let reserve = crate::inference::place::runtime_demand::load_runtime_floor(bytes);
        let mut layers = Some(layers);
        crate::inference::place::ladder::load_whole(
            "Qwen3-TTS codec decoder",
            bytes,
            reserve,
            |device| Self::build(&weights, config.clone(), device, reserve, &mut layers),
        )
    }

    /// The decoder of the checkpoint at `dir` built on the host, its transformer's layers
    /// included, with the reserve its load needs.
    ///
    /// It skips the cards, so it runs on the host whatever the cards hold.
    pub fn on_host(dir: &std::path::Path) -> Result<Self> {
        let (root, config) = read_config(&dir.join(CODEC_DIR))?;
        let weights = weights_of(dir);
        let layers = Transformer::layers(&root, &weights, config.max_positions, &[])?;
        let reserve =
            crate::inference::place::runtime_demand::load_runtime_floor(whole_bytes(&weights)?);
        Self::build(&weights, config, &Device::Cpu, reserve, &mut Some(layers))
    }

    /// The sizes of the stages that set the rounding of a product chain through them.
    pub fn stage_widths(&self) -> StageWidths {
        StageWidths {
            pre_conv_kernel: self.pre_conv.kernel(),
            upsample_layers: self.upsample.len(),
            upsample_width: self
                .upsample
                .iter()
                .map(|u| u.widths().1)
                .max()
                .unwrap_or(0),
            conv_in_kernel: self.conv_in.kernel(),
            blocks: self
                .blocks
                .iter()
                .map(|b| BlockWidth {
                    convtr_width: b.convtr.w.dims()[0] * b.convtr.kernel(),
                    units: b.units.len(),
                })
                .collect(),
            conv_out_width: self.conv_out.conv_in(1) * self.conv_out.kernel(),
        }
    }

    /// The decoder of the file `weights` on `device` around the transformer's `layers`, its
    /// chunks fitted to `reserve` bytes.
    ///
    /// The layers are taken once every part on `device` is built.
    fn build(
        weights: &std::path::Path,
        config: CodecConfig,
        device: &Device,
        reserve: u64,
        layers: &mut Option<crate::inference::generic_transformer::GenericHeteroTransformer>,
    ) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(&[weights], DType::F32, device)? };
        let vb = vb.pp(DECODER_PREFIX.trim_end_matches('.'));
        let q = vb.pp("quantizer");
        let rvq_first = Rvq::load(&q.pp("rvq_first"), RVQ_EPS)?;
        let rvq_rest = Rvq::load(&q.pp("rvq_rest"), RVQ_EPS)?;
        if rvq_first.len() != config.semantic_quantizers
            || rvq_first.len() + rvq_rest.len() != config.quantizers
        {
            return Err(Error::msg(format!(
                "the codec stores {} and {} codebooks, its config names {} and {}",
                rvq_first.len(),
                rvq_rest.len(),
                config.semantic_quantizers,
                config.quantizers
            )));
        }
        let up = vb.pp("upsample");
        let upsample = config
            .upsampling_ratios
            .iter()
            .enumerate()
            .map(|(i, &s)| Upsample::load(&up.pp(i.to_string()), s))
            .collect::<Result<Vec<_>>>()?;
        let dec = vb.pp("decoder");
        let blocks = config
            .upsample_rates
            .iter()
            .enumerate()
            .map(|(i, &s)| SnakeBlock::load(&dec.pp((i + 1).to_string()), s))
            .collect::<Result<Vec<_>>>()?;
        let last = blocks.len() + 1;
        let pre_conv = Conv::load_conv(&vb, "pre_conv.conv")?;
        let conv_in = Conv::load_conv(&dec, "0.conv")?;
        let act_out = Snake::load(&dec, &last.to_string())?;
        let conv_out = Conv::load_conv(&dec, &format!("{}.conv", last + 1))?;
        let projections = vb.pp(transformer::PREFIX
            .trim_start_matches(DECODER_PREFIX)
            .trim_end_matches('.'));
        let layers = layers
            .take()
            .ok_or_else(|| Error::msg("the codec's transformer was taken by an earlier build"))?;
        if layers.layers().len() != config.layers {
            return Err(Error::msg(format!(
                "the codec stores {} transformer layers, its config names {}",
                layers.layers().len(),
                config.layers
            )));
        }
        let mut decoder = Self {
            rvq_first,
            rvq_rest,
            pre_conv,
            transformer: Transformer::new(&projections, layers)?,
            upsample,
            conv_in,
            blocks,
            act_out,
            conv_out,
            device: device.clone(),
            config,
            chunking: Chunking {
                front_frames: 0,
                front_context: 0,
                back_frames: 0,
                back_context: 0,
            },
        };
        decoder.chunking = decoder.chunking_within(reserve)?;
        Ok(decoder)
    }

    /// The codec's configuration.
    pub fn config(&self) -> &CodecConfig {
        &self.config
    }

    /// The device the decoder sits on, its transformer's layers aside.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Where the transformer's layers sit, by device.
    pub fn transformer_placement(
        &self,
    ) -> Vec<crate::inference::serve::progress::placement::Placed> {
        self.transformer.placement()
    }

    /// The chunks the decode runs in.
    pub fn chunking(&self) -> Chunking {
        self.chunking
    }

    /// The stages of the front past the dequantisation, in order.
    fn front_reach(&self) -> Vec<Reach> {
        vec![
            Reach::Conv {
                kernel: self.pre_conv.kernel(),
                dilation: 1,
            },
            Reach::Window {
                window: self.config.window,
                layers: self.transformer.len(),
            },
        ]
    }

    /// The stages of the back, in order.
    fn back_reach(&self) -> Vec<Reach> {
        let mut r: Vec<Reach> = self.upsample.iter().flat_map(|u| u.reach()).collect();
        r.push(Reach::Conv {
            kernel: self.conv_in.kernel(),
            dilation: 1,
        });
        r.extend(self.blocks.iter().flat_map(|b| b.reach()));
        r.push(Reach::Conv {
            kernel: self.conv_out.kernel(),
            dilation: 1,
        });
        r
    }

    /// The bytes of F32 values a front pass over `frames` holds at once on the decoder's device.
    ///
    /// The rows of the widest step of every stage grow with the frames.
    /// The attention of the transformer's layers works within their own reserve, a prefill
    /// chunk at a time.
    fn front_bytes(&self, frames: usize) -> u64 {
        let c = &self.config;
        let k = self.pre_conv.kernel();
        let rows = frames * (self.pre_conv.conv_in(1) * (k + 2) + 2 * c.latent + 3 * c.hidden);
        (rows * std::mem::size_of::<f32>()) as u64
    }

    /// The bytes a back pass holds at once, per frame: its widest stage.
    fn back_bytes_per_frame(&self) -> u64 {
        let mut rate = 1;
        let mut most = 0;
        for u in &self.upsample {
            most = most.max(u.bytes_per_frame(rate));
            rate *= u.stride();
        }
        most = most.max(self.conv_in.bytes_per_frame(rate));
        for b in &self.blocks {
            most = most.max(b.bytes_per_frame(rate));
            rate *= b.stride();
        }
        most.max(self.conv_out.bytes_per_frame(rate))
    }

    /// The chunks whose passes fit `reserve` bytes, with the left context of each part.
    fn chunking_within(&self, reserve: u64) -> Result<Chunking> {
        let fs = self.config.frame_samples;
        let front_context = left_context(&self.front_reach(), fs)?;
        let back_context = left_context(&self.back_reach(), fs)?;
        let front_frames =
            frames_within(reserve, self.config.max_positions, |n| self.front_bytes(n)).unwrap_or(0);
        let back_frames = (reserve / self.back_bytes_per_frame().max(1)) as usize;
        for (part, frames, context) in [
            ("front", front_frames, front_context),
            ("back", back_frames, back_context),
        ] {
            if frames <= context {
                return Err(Error::msg(format!(
                    "a reserve of {reserve} bytes holds {frames} frames of the decoder's {part}, \
                     its left context is {context}"
                )));
            }
        }
        Ok(Chunking {
            front_frames,
            front_context,
            back_frames,
            back_context,
        })
    }

    /// `frames` of codes as `[quantizers, T]` on the decoder's device.
    pub fn codes(&self, frames: &[Vec<u32>]) -> Result<Tensor> {
        let q = self.config.quantizers;
        let entries = self.config.codebook_size as u32;
        let mut data = vec![0u32; q * frames.len()];
        for (t, frame) in frames.iter().enumerate() {
            if frame.len() != q {
                return Err(Error::msg(format!(
                    "frame {t} holds {} codes, the codec reads {q}",
                    frame.len()
                )));
            }
            for (g, &code) in frame.iter().enumerate() {
                if code >= entries {
                    return Err(Error::msg(format!(
                        "frame {t} code {g} is {code}, the codebook holds {entries}"
                    )));
                }
                data[g * frames.len() + t] = code;
            }
        }
        Tensor::from_vec_u32(data, (q, frames.len()))?.to_device(&self.device)
    }

    /// `codes [quantizers, T]` to the latent of the quantizer, `[1, codebook_dim, T]`.
    pub fn dequantize(&self, codes: &Tensor) -> Result<Tensor> {
        let t = codes.dims()[1];
        let n = self.rvq_first.len();
        let first = self.rvq_first.decode(&codes.narrow(0, 0, n)?)?;
        let rest = self
            .rvq_rest
            .decode(&codes.narrow(0, n, self.rvq_rest.len())?)?;
        let q = first.add(&rest)?;
        let dim = q.dims()[0];
        q.reshape((1, dim, t))
    }

    /// The causal conv after the quantizer, `[1, codebook_dim, T]` to `[1, latent, T]`.
    pub fn pre_conv(&self, q: &Tensor) -> Result<Tensor> {
        self.pre_conv.forward(q, 1, 1)
    }

    /// `codes [quantizers, T]` of a pass that starts afresh through the front, `[T, latent]`.
    ///
    /// The transformer sees the pass's frames at positions `0..T`.
    pub fn front(&self, codes: &Tensor) -> Result<Tensor> {
        let x = self.pre_conv(&self.dequantize(codes)?)?;
        let (_, c, t) = x.shape().dims3()?;
        let rows = x.reshape((c, t))?.transpose(0, 1)?.contiguous()?;
        self.transformer.forward(&rows)
    }

    /// `x [1, latent, T]` through the upsample stages, `[1, latent, T * ratios]`.
    pub(crate) fn upsample(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = x.clone();
        for u in &self.upsample {
            h = u.forward(&h)?;
        }
        Ok(h)
    }

    /// `x [1, latent, T]` through the back, every output of each stage kept in order.
    ///
    /// The first is the upsampled latent, the last the clamped waveform `[1, 1, samples]`.
    pub fn back_stages(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut out = vec![self.upsample(x)?];
        let mut h = self.conv_in.forward(&out[0], 1, 1)?;
        out.push(h.clone());
        for b in &self.blocks {
            h = b.forward(&h)?;
            out.push(h.clone());
        }
        h = self.act_out.forward(&h)?;
        out.push(h.clone());
        h = self.conv_out.forward(&h, 1, 1)?;
        out.push(clamp(&h, WAVEFORM_BOUND)?);
        Ok(out)
    }

    /// `x [1, latent, T]` through the back, the waveform `[1, 1, samples]`.
    fn back(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.conv_in.forward(&self.upsample(x)?, 1, 1)?;
        for b in &self.blocks {
            h = b.forward(&h)?;
        }
        h = self.conv_out.forward(&self.act_out.forward(&h)?, 1, 1)?;
        clamp(&h, WAVEFORM_BOUND)
    }

    /// The waveform of `frames`, `frame_samples` samples a frame, in chunks that fit the
    /// reserve.
    pub fn decode(&self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        self.decode_in(frames, self.chunking)
    }

    /// The waveform of `frames` in the chunks of `chunking`.
    pub fn decode_in(&self, frames: &[Vec<u32>], chunking: Chunking) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let rows = self.front_rows(&self.codes(frames)?, chunking)?;
        self.back_wave(&rows, chunking)
    }

    /// `codes [quantizers, T]` through the front in passes of `chunking`, `[T, latent]`.
    pub(crate) fn front_rows(&self, codes: &Tensor, chunking: Chunking) -> Result<Tensor> {
        let total = codes.dims()[1];
        let mut rows = Vec::new();
        for (from, start, end) in passes(0, total, chunking.front_frames, chunking.front_context) {
            let out = self.front(&codes.narrow(1, from, end - from)?)?;
            rows.push(out.narrow(0, start - from, end - start)?);
        }
        let rows: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&rows, 0)
    }

    /// `rows [T, latent]` through the back in passes of `chunking`, the waveform.
    fn back_wave(&self, rows: &Tensor, chunking: Chunking) -> Result<Vec<f32>> {
        self.back_wave_from(rows, 0, chunking)
    }

    /// The waveform of the frames of `rows [T, latent]` from frame `first` on, the frames
    /// before it read as left context.
    fn back_wave_from(&self, rows: &Tensor, first: usize, chunking: Chunking) -> Result<Vec<f32>> {
        let (total, latent) = rows.shape().dims2()?;
        let fs = self.config.frame_samples;
        let mut wave = Vec::with_capacity(total.saturating_sub(first) * fs);
        for (from, start, end) in passes(first, total, chunking.back_frames, chunking.back_context)
        {
            let x = rows
                .narrow(0, from, end - from)?
                .transpose(0, 1)?
                .contiguous()?
                .reshape((1, latent, end - from))?;
            let w = self.back(&x)?.flatten_all()?;
            let kept = w.narrow(0, (start - from) * fs, (end - start) * fs)?;
            wave.extend(kept.to_device(&Device::Cpu)?.to_vec1_f32()?);
        }
        Ok(wave)
    }
}

/// A decode fed frames as they are drawn.
///
/// Each call gives the samples of the frames it is given.
/// The front of a call runs over the new frames and the left context of the front before them.
/// The front rows of a frame do not change once computed: every stage is causal.
/// The back reads the front rows of its left context, kept from the calls before.
/// The samples equal those of a whole decode within the rounding of F32 through the decoder.
pub struct DecodeStream<'a> {
    decoder: &'a CodecDecoder,
    chunking: Chunking,
    /// The frames given so far.
    frames: Vec<Vec<u32>>,
    /// The front rows of the last frames given, as many as the back's left context.
    tail: Option<Tensor>,
}

impl<'a> DecodeStream<'a> {
    /// The samples of `frames`, the frames that follow those given before.
    pub fn push(&mut self, frames: &[Vec<u32>]) -> Result<Vec<f32>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }
        let c = self.chunking;
        let start = self.frames.len();
        self.frames.extend_from_slice(frames);
        let total = self.frames.len();
        let first = start.saturating_sub(c.front_context);
        let codes = self.decoder.codes(&self.frames[first..])?;
        let mut rows: Vec<Tensor> = self.tail.take().into_iter().collect();
        let kept = rows.first().map_or(0, |t| t.dims()[0]);
        for (from, s, end) in passes(start, total, c.front_frames, c.front_context) {
            let out = self
                .decoder
                .front(&codes.narrow(1, from - first, end - from)?)?;
            rows.push(out.narrow(0, s - from, end - s)?);
        }
        let rows = Tensor::cat(&rows.iter().collect::<Vec<_>>(), 0)?;
        let wave = self.decoder.back_wave_from(&rows, kept, c)?;
        let n = rows.dims()[0];
        let keep = n.min(c.back_context);
        self.tail = match keep {
            0 => None,
            _ => Some(rows.narrow(0, n - keep, keep)?.contiguous()?),
        };
        Ok(wave)
    }
}

impl CodecDecoder {
    /// A decode fed frames as they come, in the chunks the decoder derived.
    pub fn stream(&self) -> DecodeStream<'_> {
        self.stream_in(self.chunking)
    }

    /// A decode fed frames as they come, in the chunks of `chunking`.
    pub fn stream_in(&self, chunking: Chunking) -> DecodeStream<'_> {
        DecodeStream {
            decoder: self,
            chunking,
            frames: Vec::new(),
            tail: None,
        }
    }
}

/// `x` clamped to `[-bound, bound]`, built from relu and affine.
fn clamp(x: &Tensor, bound: f32) -> Result<Tensor> {
    let low = max_scalar(x, -bound)?;
    max_scalar(&low.affine(-1.0, 0.0)?, -bound)?.affine(-1.0, 0.0)
}

/// The codec's configuration in the directory `codec`, as read and as the decoder's sizes.
fn read_config(codec: &std::path::Path) -> Result<(serde_json::Value, CodecConfig)> {
    let path = codec.join("config.json");
    let text = std::fs::read_to_string(&path)
        .map_err(|e| Error::msg(format!("read {}: {e}", path.display())))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::msg(format!("parse {}: {e}", path.display())))?;
    let config = CodecConfig::read(&json)?;
    Ok((json, config))
}

/// The codec's file under the checkpoint at `dir`.
pub(crate) fn weights_of(dir: &std::path::Path) -> std::path::PathBuf {
    dir.join(CODEC_DIR).join("model.safetensors")
}

/// The bytes of the decoder's tensors in the codec's file.
pub(crate) fn decoder_bytes(weights: &std::path::Path) -> Result<u64> {
    let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::new(weights)? };
    Ok(st
        .byte_sizes()
        .iter()
        .filter(|(n, _)| n.starts_with(DECODER_PREFIX))
        .map(|(_, b)| *b as u64)
        .sum())
}

/// The bytes of the decoder's tensors placed whole: its transformer's layers aside.
fn whole_bytes(weights: &std::path::Path) -> Result<u64> {
    let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::new(weights)? };
    Ok(decoder_bytes(weights)? - transformer::layer_bytes(&st))
}
