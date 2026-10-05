//! The log-mel features a speech-conditioned language model is trained on.
//!
//! The conventions are those of the feature extractor the family was trained with: the
//! waveform is reflect-padded by half a frame at both ends so frames are centred, a periodic
//! Hann window is applied, the power spectrum is pooled by a Slaney-scale filter bank, the
//! last frame is dropped, bands are floored eight decades below the loudest bin of the whole
//! clip and the result is scaled into roughly `[-1, 1]`.
//!
//! The ASR front end in `whisper` keeps its own conventions (zero padding, a window that
//! starts at the first sample); a model fed these features must not be fed those.

use rayon::prelude::*;

/// How a waveform becomes a spectrogram.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MelSpec {
    /// Mel bands per frame: the encoder's input channels.
    pub n_mels: usize,
    /// Samples per second the filter bank is laid out for.
    pub sample_rate: usize,
    /// Samples per analysis frame.
    pub n_fft: usize,
    /// Samples between frame starts.
    pub hop: usize,
}

impl MelSpec {
    /// The conventions of the speech-encoder family, for `n_mels` bands.
    pub fn speech(n_mels: usize) -> Self {
        Self {
            n_mels,
            sample_rate: crate::inference::model::whisper::SAMPLE_RATE,
            n_fft: crate::inference::model::whisper::N_FFT,
            hop: crate::inference::model::whisper::HOP_LENGTH,
        }
    }

    /// Frequency bins of a real transform of one frame.
    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Frames in one encoder window: the encoder's context in mel frames.
    pub fn window_frames(&self) -> usize {
        crate::inference::model::whisper::CHUNK_LENGTH * self.sample_rate / self.hop
    }

    /// Samples in one encoder window.
    pub fn window_samples(&self) -> usize {
        crate::inference::model::whisper::CHUNK_LENGTH * self.sample_rate
    }
}

/// Slaney mel scale: linear below 1 kHz, logarithmic above.
fn hz_to_mel(hz: f64) -> f64 {
    const BREAK_HZ: f64 = 1000.0;
    const LINEAR_SLOPE: f64 = 3.0 / 200.0;
    let break_mel = BREAK_HZ * LINEAR_SLOPE;
    let log_step = (6.4f64).ln() / 27.0;
    if hz >= BREAK_HZ {
        break_mel + (hz / BREAK_HZ).ln() / log_step
    } else {
        hz * LINEAR_SLOPE
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const BREAK_HZ: f64 = 1000.0;
    const LINEAR_SLOPE: f64 = 3.0 / 200.0;
    let break_mel = BREAK_HZ * LINEAR_SLOPE;
    let log_step = (6.4f64).ln() / 27.0;
    if mel >= break_mel {
        BREAK_HZ * ((mel - break_mel) * log_step).exp()
    } else {
        mel / LINEAR_SLOPE
    }
}

/// The triangular filter bank `[n_mels, bins]` row-major, each band normalised by its width
/// in hertz so a flat spectrum gives every band the same weight.
pub fn filter_bank(spec: &MelSpec) -> Vec<f32> {
    let bins = spec.bins();
    let nyquist = spec.sample_rate as f64 / 2.0;
    let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(nyquist));
    let edges: Vec<f64> = (0..spec.n_mels + 2)
        .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (spec.n_mels + 1) as f64))
        .collect();
    let mut bank = vec![0f32; spec.n_mels * bins];
    for band in 0..spec.n_mels {
        let (left, centre, right) = (edges[band], edges[band + 1], edges[band + 2]);
        let norm = 2.0 / (right - left);
        for bin in 0..bins {
            let freq = bin as f64 * nyquist / (bins - 1) as f64;
            let rise = (freq - left) / (centre - left);
            let fall = (right - freq) / (right - centre);
            let w = rise.min(fall).max(0.0);
            bank[band * bins + bin] = (w * norm) as f32;
        }
    }
    bank
}

/// `samples` with `pad` reflected samples at each end. A clip shorter than `pad + 1` is
/// first extended with silence, so the reflection always has samples to mirror.
fn reflect_pad(samples: &[f32], pad: usize) -> Vec<f32> {
    let mut x = samples.to_vec();
    if x.len() <= pad {
        x.resize(pad + 1, 0.0);
    }
    let n = x.len();
    let mut out = Vec::with_capacity(n + 2 * pad);
    out.extend((1..=pad).rev().map(|i| x[i]));
    out.extend_from_slice(&x);
    out.extend((1..=pad).map(|i| x[n - 1 - i]));
    out
}

/// The log-mel spectrogram of `samples`, `[n_mels, frames]` row-major, with
/// `frames = samples.len() / hop` (the frame the centring adds past the end is dropped).
///
/// The whole clip is normalised by its own loudest bin, so the caller pads the waveform
/// BEFORE calling when the model was trained on padded clips.
pub fn log_mel(samples: &[f32], spec: &MelSpec) -> (Vec<f32>, usize) {
    let frames = samples.len() / spec.hop;
    if frames == 0 {
        return (Vec::new(), 0);
    }
    let bank = filter_bank(spec);
    let bins = spec.bins();
    let padded = reflect_pad(samples, spec.n_fft / 2);
    // Periodic Hann: it reaches zero at the first sample and not at the last.
    let window: Vec<f32> = (0..spec.n_fft)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / spec.n_fft as f32).cos())
        .collect();
    let columns: Vec<Vec<f32>> = (0..frames)
        .into_par_iter()
        .map(|f| {
            let start = f * spec.hop;
            let frame: Vec<f32> = padded[start..start + spec.n_fft]
                .iter()
                .zip(&window)
                .map(|(s, w)| s * w)
                .collect();
            let fourier = crate::inference::model::whisper::audio::transform(&frame);
            let power: Vec<f32> = (0..bins)
                .map(|k| fourier[2 * k] * fourier[2 * k] + fourier[2 * k + 1] * fourier[2 * k + 1])
                .collect();
            (0..spec.n_mels)
                .map(|band| {
                    let w = &bank[band * bins..(band + 1) * bins];
                    let pooled: f32 = power.iter().zip(w).map(|(p, w)| p * w).sum();
                    pooled.max(1e-10).log10()
                })
                .collect()
        })
        .collect();
    let loudest = columns
        .iter()
        .flat_map(|c| c.iter().copied())
        .fold(f32::NEG_INFINITY, f32::max);
    let floor = loudest - 8.0;
    let mut mel = vec![0f32; spec.n_mels * frames];
    for (f, column) in columns.iter().enumerate() {
        for (band, v) in column.iter().enumerate() {
            mel[band * frames + f] = (v.max(floor) + 4.0) / 4.0;
        }
    }
    (mel, frames)
}

/// One window of `frames` columns of a `[n_mels, total]` spectrogram, starting at column
/// `from`, as `[n_mels, frames]`.
pub fn window_of(mel: &[f32], n_mels: usize, total: usize, from: usize, frames: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n_mels * frames);
    for band in 0..n_mels {
        out.extend_from_slice(&mel[band * total + from..band * total + from + frames]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The computed bank is the one the speech checkpoints ship.
    #[test]
    fn the_computed_bank_is_the_shipped_one() {
        let shipped: &[u8] = include_bytes!("../../../../assets/whisper/melfilters128.bytes");
        let shipped: Vec<f32> = shipped
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let computed = filter_bank(&MelSpec::speech(128));
        assert_eq!(shipped.len(), computed.len());
        let worst = shipped
            .iter()
            .zip(&computed)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-6, "worst difference {worst}");
    }

    /// The spectrogram of a clip against the reference feature extractor: the clip alone
    /// (`longest`), and the clip zero-padded to a whole 30 s window (`padded`), where the
    /// silence takes part in the normalisation.
    #[test]
    fn the_features_match_the_reference_extractor() {
        use crate::inference::model::audio_in::gate::{compare, tensor, values, vectors};
        let dir = vectors("features");
        let spec = MelSpec::speech(128);
        let wave = values(&tensor(&dir, "wave"));

        let (bank, want) = (filter_bank(&spec), tensor(&dir, "filters"));
        let got = crate::tensor::Tensor::from_vec(
            bank,
            want.dims().to_vec(),
            &crate::tensor::Device::Cpu,
        )
        .unwrap();
        let (diff, _) = compare(&got, &want);
        assert!(diff < 1e-6, "filter bank {diff}");

        let (mel, frames) = log_mel(&wave, &spec);
        let want = tensor(&dir, "longest");
        assert_eq!(want.dims(), [128, frames]);
        let got = crate::tensor::Tensor::from_vec(mel, (128, frames), &crate::tensor::Device::Cpu)
            .unwrap();
        let (diff, scale) = compare(&got, &want);
        assert!(
            diff <= 1e-3 * scale,
            "clip alone: {diff} off, scale {scale}"
        );

        let mut padded = wave.clone();
        padded.resize(spec.window_samples(), 0.0);
        let (mel, frames) = log_mel(&padded, &spec);
        assert_eq!(frames, spec.window_frames());
        let head = window_of(&mel, 128, frames, 0, 200);
        let got =
            crate::tensor::Tensor::from_vec(head, (128, 200), &crate::tensor::Device::Cpu).unwrap();
        let (diff, scale) = compare(&got, &tensor(&dir, "padded_head"));
        assert!(
            diff <= 1e-3 * scale,
            "padded window: {diff} off, scale {scale}"
        );
    }

    #[test]
    fn reflection_mirrors_without_repeating_the_edge() {
        let p = reflect_pad(&[1., 2., 3., 4., 5.], 2);
        assert_eq!(p, vec![3., 2., 1., 2., 3., 4., 5., 4., 3.]);
    }

    #[test]
    fn a_clip_gives_one_frame_per_hop() {
        let spec = MelSpec::speech(16);
        let clip: Vec<f32> = (0..spec.hop * 7)
            .map(|i| ((i * 31 % 17) as f32) / 17.0 - 0.5)
            .collect();
        let (mel, frames) = log_mel(&clip, &spec);
        assert_eq!(frames, 7);
        assert_eq!(mel.len(), 16 * 7);
        assert!(mel.iter().all(|v| v.is_finite()));
    }
}
