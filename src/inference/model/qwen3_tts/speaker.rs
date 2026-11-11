//! The speaker encoder of the Qwen3-TTS Base variant: a clip to the speaker embedding the clone
//! prompt carries.
//!
//! The encoder is an ECAPA network: time-delay convolutions, squeeze-excitation Res2Net blocks,
//! the aggregation of their outputs and an attentive statistics pooling.
//! Every width, kernel and count is read from the stored tensors.
//! The dilations and the mel front end exist only in the reference code: they are the named
//! constants below.
//! The encoder has no layers: it is placed whole on one device through the ladder.

use crate::tensor::VarBuilder;
use crate::tensor::{DType, Device, Error, Result, Tensor};

/// Samples of an analysis frame of the mel front end.
pub const MEL_N_FFT: usize = 1024;
/// Samples between the starts of 2 frames.
pub const MEL_HOP: usize = 256;
/// Samples of the Hann window.
pub const MEL_WINDOW: usize = 1024;
/// The lowest frequency of the filter bank, in hertz.
pub const MEL_FMIN: f64 = 0.0;
/// The floor added to the squared magnitude before its root.
pub const MEL_MAGNITUDE_EPS: f64 = 1e-9;
/// The floor of a mel band before its logarithm.
pub const MEL_LOG_FLOOR: f64 = 1e-5;
/// The dilation of the convolution of each block, then of the aggregation.
pub const ENCODER_DILATIONS: [usize; 5] = [1, 2, 3, 4, 1];
/// The floor of a variance of the attentive statistics pooling.
pub const POOLING_EPS: f32 = 1e-12;

/// The prefix of the speaker encoder's tensors.
pub const SPEAKER_PREFIX: &str = "speaker_encoder.";

/// Slaney mel scale: linear below 1 kHz, logarithmic above.
fn hz_to_mel(hz: f64) -> f64 {
    let (break_hz, slope) = (1000.0, 3.0 / 200.0);
    let log_step = 6.4f64.ln() / 27.0;
    match hz >= break_hz {
        true => break_hz * slope + (hz / break_hz).ln() / log_step,
        false => hz * slope,
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    let (break_hz, slope) = (1000.0, 3.0 / 200.0);
    let log_step = 6.4f64.ln() / 27.0;
    let break_mel = break_hz * slope;
    match mel >= break_mel {
        true => break_hz * ((mel - break_mel) * log_step).exp(),
        false => mel / slope,
    }
}

/// The Slaney filter bank `[bands, MEL_N_FFT / 2 + 1]` row-major, from `MEL_FMIN` to the
/// Nyquist frequency of `rate`.
///
/// Each band is a triangle between its neighbours' centres.
/// Each band is divided by half its width in hertz.
pub fn filter_bank(bands: usize, rate: usize) -> Vec<f64> {
    let bins = MEL_N_FFT / 2 + 1;
    let nyquist = rate as f64 / 2.0;
    let (lo, hi) = (hz_to_mel(MEL_FMIN), hz_to_mel(nyquist));
    let edges: Vec<f64> = (0..bands + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (bands + 1) as f64))
        .collect();
    let mut bank = vec![0f64; bands * bins];
    for band in 0..bands {
        let (left, centre, right) = (edges[band], edges[band + 1], edges[band + 2]);
        let norm = 2.0 / (right - left);
        for bin in 0..bins {
            let freq = bin as f64 * nyquist / (bins - 1) as f64;
            let rise = (freq - left) / (centre - left);
            let fall = (right - freq) / (right - centre);
            bank[band * bins + bin] = rise.min(fall).max(0.0) * norm;
        }
    }
    bank
}

/// `samples` with `pad` mirrored samples at each end, the edge sample not repeated.
fn reflect(samples: &[f32], pad: usize) -> Result<Vec<f32>> {
    let n = samples.len();
    if n <= pad {
        return Err(Error::msg(format!(
            "a clip of {n} samples is shorter than the {pad} its front end mirrors"
        )));
    }
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| samples[i]));
    out.extend_from_slice(samples);
    out.extend((1..=pad).map(|i| samples[n - 1 - i]));
    Ok(out)
}

/// In-place radix-2 transform of `re` and `im`, of a length that is a power of 2.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        for start in (0..n).step_by(len) {
            for k in 0..len / 2 {
                let (s, c) = (angle * k as f64).sin_cos();
                let (a, b) = (start + k, start + k + len / 2);
                let (vr, vi) = (re[b] * c - im[b] * s, re[b] * s + im[b] * c);
                re[b] = re[a] - vr;
                im[b] = im[a] - vi;
                re[a] += vr;
                im[a] += vi;
            }
        }
        len <<= 1;
    }
}

/// The log-mel spectrogram of `samples` at `rate`, `[bands, frames]` row-major, and its frames.
///
/// The clip is mirrored by `(MEL_N_FFT - MEL_HOP) / 2` samples at each end.
/// Each frame starts `MEL_HOP` samples after the last and is weighted by a periodic Hann window.
/// A bin's magnitude is the root of its squared magnitude plus `MEL_MAGNITUDE_EPS`.
/// A band is the filter bank's sum of magnitudes, floored at `MEL_LOG_FLOOR`, then its log.
pub fn log_mel(samples: &[f32], rate: usize, bands: usize) -> Result<(Vec<f32>, usize)> {
    let padded = reflect(samples, (MEL_N_FFT - MEL_HOP) / 2)?;
    if padded.len() < MEL_N_FFT {
        return Err(Error::msg(format!(
            "a clip of {} samples is shorter than a frame",
            samples.len()
        )));
    }
    let frames = 1 + (padded.len() - MEL_N_FFT) / MEL_HOP;
    let bins = MEL_N_FFT / 2 + 1;
    let bank = filter_bank(bands, rate);
    let offset = (MEL_N_FFT - MEL_WINDOW) / 2;
    let window: Vec<f64> = (0..MEL_WINDOW)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / MEL_WINDOW as f64).cos())
        .collect();
    let mut out = vec![0f32; bands * frames];
    let (mut re, mut im) = (vec![0f64; MEL_N_FFT], vec![0f64; MEL_N_FFT]);
    let mut magnitude = vec![0f64; bins];
    for f in 0..frames {
        let start = f * MEL_HOP;
        re.iter_mut().for_each(|x| *x = 0.0);
        im.iter_mut().for_each(|x| *x = 0.0);
        for (i, w) in window.iter().enumerate() {
            re[offset + i] = padded[start + offset + i] as f64 * w;
        }
        fft(&mut re, &mut im);
        for (b, m) in magnitude.iter_mut().enumerate() {
            *m = (re[b] * re[b] + im[b] * im[b] + MEL_MAGNITUDE_EPS).sqrt();
        }
        for band in 0..bands {
            let row = &bank[band * bins..(band + 1) * bins];
            let sum: f64 = row.iter().zip(&magnitude).map(|(w, m)| w * m).sum();
            out[band * frames + f] = sum.max(MEL_LOG_FLOOR).ln() as f32;
        }
    }
    Ok((out, frames))
}

/// A convolution `[out, in, k]` with its bias, padded to keep the length by mirrored samples.
struct Conv {
    w: Tensor,
    b: Tensor,
    dilation: usize,
}

impl Conv {
    /// `{name}.conv` when the file stores it, else `{name}`, with its weight and bias.
    fn load(vb: &VarBuilder, name: &str, dilation: usize) -> Result<Self> {
        let stem = match vb.shape_of(&format!("{name}.conv.weight")) {
            Ok(_) => format!("{name}.conv"),
            Err(_) => name.to_string(),
        };
        let shape = vb.shape_of(&format!("{stem}.weight"))?;
        Ok(Self {
            b: vb.get(shape[0], &format!("{stem}.bias"))?,
            w: vb.get(shape, &format!("{stem}.weight"))?,
            dilation,
        })
    }

    /// Output channels.
    fn width(&self) -> usize {
        self.w.dims()[0]
    }

    /// The convolution of `x [1, in, L]` to `[1, out, L]`.
    ///
    /// The `dilation * (k - 1)` padded samples go half on the left, the rest on the right.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let k = self.w.dims()[2];
        let total = self.dilation * (k - 1);
        let (left, right) = (total / 2, total - total / 2);
        let x = reflect_tensor(x, left, right)?;
        let y = x.conv1d(&self.w, 0, 1, self.dilation, 1)?;
        let out = self.b.dims()[0];
        y.broadcast_add(&self.b.reshape((1, out, 1))?)
    }

    /// The convolution then the rectifier.
    fn tdnn(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)?.relu()
    }
}

/// `x [1, C, L]` with `left` and `right` mirrored samples, the edge sample not repeated.
fn reflect_tensor(x: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(x.clone());
    }
    let l = x.dims()[2];
    if l <= left.max(right) {
        return Err(Error::msg(format!(
            "{l} frames are too few to mirror {}",
            left.max(right)
        )));
    }
    let mut parts: Vec<Tensor> = (1..=left)
        .rev()
        .map(|i| x.narrow(2, i, 1))
        .collect::<Result<_>>()?;
    parts.push(x.clone());
    for i in 1..=right {
        parts.push(x.narrow(2, l - 1 - i, 1)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 2)?.contiguous()
}

/// A squeeze-excitation Res2Net block.
struct Block {
    tdnn1: Conv,
    res2net: Vec<Conv>,
    tdnn2: Conv,
    se1: Conv,
    se2: Conv,
}

impl Block {
    fn load(vb: &VarBuilder, dilation: usize) -> Result<Self> {
        let mut res2net = Vec::new();
        while vb
            .shape_of(&format!(
                "res2net_block.blocks.{}.conv.weight",
                res2net.len()
            ))
            .is_ok()
        {
            let name = format!("res2net_block.blocks.{}", res2net.len());
            res2net.push(Conv::load(vb, &name, dilation)?);
        }
        Ok(Self {
            tdnn1: Conv::load(vb, "tdnn1", 1)?,
            res2net,
            tdnn2: Conv::load(vb, "tdnn2", 1)?,
            se1: Conv::load(vb, "se_block.conv1", 1)?,
            se2: Conv::load(vb, "se_block.conv2", 1)?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.tdnn1.tdnn(x)?;
        let parts = h.chunk(self.res2net.len() + 1, 1)?;
        let mut outs: Vec<Tensor> = vec![parts[0].clone()];
        for (i, conv) in self.res2net.iter().enumerate() {
            let input = match i {
                0 => parts[1].clone(),
                _ => (&parts[i + 1] + &outs[i])?,
            };
            outs.push(conv.tdnn(&input.contiguous()?)?);
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        let h = self.tdnn2.tdnn(&Tensor::cat(&refs, 1)?)?;
        let s = self.se1.forward(&h.mean_keepdim(2)?)?.relu()?;
        let s = self.se2.forward(&s)?.sigmoid()?;
        h.broadcast_mul(&s)? + x
    }
}

/// The weighted mean and standard deviation over time of `x [1, C, L]`, `w` summing to 1 over
/// time, each `[1, C, 1]`.
fn statistics(x: &Tensor, w: &Tensor) -> Result<(Tensor, Tensor)> {
    let mean = x.broadcast_mul(w)?.sum_keepdim(2)?;
    let var = x
        .broadcast_sub(&mean)?
        .sqr()?
        .broadcast_mul(w)?
        .sum_keepdim(2)?;
    let floor = var.ones_like()?.affine(POOLING_EPS, 0.0)?;
    Ok((mean, var.maximum(&floor)?.sqrt()?))
}

/// The speaker encoder, placed whole on one device.
pub struct SpeakerEncoder {
    first: Conv,
    blocks: Vec<Block>,
    mfa: Conv,
    asp_tdnn: Conv,
    asp_conv: Conv,
    fc: Conv,
    /// The sample rate the encoder takes, read from the configuration.
    rate: usize,
    device: Device,
}

/// Bytes of the speaker encoder's tensors as stored in `st`.
pub fn speaker_bytes(st: &crate::tensor::safetensors::MmapedSafetensors) -> u64 {
    st.byte_sizes()
        .iter()
        .filter(|(n, _)| n.starts_with(SPEAKER_PREFIX))
        .map(|(_, b)| *b as u64)
        .sum()
}

impl SpeakerEncoder {
    /// The speaker encoder of the checkpoint at `dir`, its sample rate read from its
    /// configuration's `speaker_encoder_config`.
    ///
    /// It is placed whole through the ladder, its tensors held in F32.
    pub fn load(dir: &std::path::Path, rate: usize) -> Result<Self> {
        let (_, shards) = crate::inference::load::awq_loader::checkpoint_shards(dir)
            .map_err(|e| Error::msg(e.to_string()))?;
        let st = unsafe { crate::tensor::safetensors::MmapedSafetensors::multi(&shards)? };
        let bytes = 2 * speaker_bytes(&st);
        let reserve = crate::inference::place::runtime_demand::load_runtime_floor(bytes);
        crate::inference::place::ladder::load_whole(
            "Qwen3-TTS speaker encoder",
            bytes,
            reserve,
            |device| Self::build(&shards, rate, device),
        )
    }

    /// The speaker encoder of the tensors of `shards` on `device`.
    pub fn build(shards: &[String], rate: usize, device: &Device) -> Result<Self> {
        let vb = unsafe { VarBuilder::from_files(shards, DType::F32, device)? };
        let vb = vb.pp(SPEAKER_PREFIX.trim_end_matches('.'));
        let stored = |i: usize| {
            ["conv", "tdnn1.conv"]
                .iter()
                .any(|c| vb.shape_of(&format!("blocks.{i}.{c}.weight")).is_ok())
        };
        let count = (0..).take_while(|i| stored(*i)).count();
        if count + 1 != ENCODER_DILATIONS.len() {
            return Err(Error::msg(format!(
                "the speaker encoder stores {count} blocks, its dilations name {}",
                ENCODER_DILATIONS.len() - 1
            )));
        }
        let first = Conv::load(&vb, "blocks.0", ENCODER_DILATIONS[0])?;
        let blocks = (1..count)
            .map(|i| Block::load(&vb.pp(format!("blocks.{i}")), ENCODER_DILATIONS[i]))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            first,
            blocks,
            mfa: Conv::load(&vb, "mfa", ENCODER_DILATIONS[count])?,
            asp_tdnn: Conv::load(&vb, "asp.tdnn", 1)?,
            asp_conv: Conv::load(&vb, "asp.conv", 1)?,
            fc: Conv::load(&vb, "fc", 1)?,
            rate,
            device: device.clone(),
        })
    }

    /// Mel bands the encoder takes: the input channels of its first convolution.
    pub fn bands(&self) -> usize {
        self.first.w.dims()[1]
    }

    /// Values of the embedding: the output channels of its last convolution.
    pub fn width(&self) -> usize {
        self.fc.width()
    }

    /// The sample rate the encoder takes.
    pub fn rate(&self) -> usize {
        self.rate
    }

    /// The device the encoder runs on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The embedding of the log-mel spectrogram `mel [bands, frames]` row-major.
    pub fn embed_mel(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        let x = Tensor::from_vec(mel.to_vec(), (1, self.bands(), frames), &Device::Cpu)?
            .to_device(&self.device)?;
        let mut h = self.first.tdnn(&x)?;
        let mut outs = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            h = block.forward(&h)?;
            outs.push(h.clone());
        }
        let refs: Vec<&Tensor> = outs.iter().collect();
        let h = self.mfa.tdnn(&Tensor::cat(&refs, 1)?)?;
        let uniform = Tensor::from_vec(
            vec![1.0 / frames as f32; frames],
            (1, 1, frames),
            &Device::Cpu,
        )?
        .to_device(&self.device)?;
        let (mean, std) = statistics(&h, &uniform)?;
        let shape = h.dims().to_vec();
        let context = Tensor::cat(
            &[
                &h,
                &mean.broadcast_as(shape.clone())?,
                &std.broadcast_as(shape)?,
            ],
            1,
        )?
        .contiguous()?;
        let a = self.asp_tdnn.tdnn(&context)?.tanh()?;
        let a = crate::tensor::ops::softmax_last_dim(&self.asp_conv.forward(&a)?)?;
        let (mean, std) = statistics(&h, &a)?;
        let pooled = Tensor::cat(&[&mean, &std], 1)?;
        let out = self.fc.forward(&pooled)?;
        out.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()
    }

    /// The speaker embedding of the mono clip `samples` at `rate` hertz.
    ///
    /// A clip at another rate than the encoder's is resampled first.
    pub fn embed(&self, samples: &[f32], rate: u32) -> Result<Vec<f32>> {
        let resampled;
        let samples = match rate as usize == self.rate {
            true => samples,
            false => {
                resampled =
                    crate::inference::media::audio_io::resample(samples, rate, self.rate as u32)
                        .map_err(|e| Error::msg(e.to_string()))?;
                &resampled
            }
        };
        let (mel, frames) = log_mel(samples, self.rate, self.bands())?;
        self.embed_mel(&mel, frames)
    }
}
