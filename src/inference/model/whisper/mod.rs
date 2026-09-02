//! Native Whisper config + audio front-end - replaces
//! `candle_transformers::models::whisper::{Config, audio, constants}`.
//! The Whisper model itself already lives in
//! `inference/model/whisper/model.rs` (native); this module supplies the remaining shared
//! pieces: the config struct, the tokenizer/audio constants, and the
//! log-mel-spectrogram front-end (f32-specialized port of whisper.cpp's audio
//! code, threaded via `num_cpus`).
/// The encoder/decoder itself, already on the substrate: this family was on the flat debt
/// list by mistake, because NOTICE.md attributes the MEL FRONT-END to whisper.cpp and the
/// note was read as a live dependency of the model.
pub mod model;

use serde::Deserialize;

/// The shape of a checkpoint, read straight off the file that ships with it.
///
/// The two halves are sized separately. `num_mel_bins` is how many bands the spectrogram carries
/// and `max_source_positions` how many the encoder holds once its stride-two front end has
/// shortened them; `vocab_size` and `max_target_positions` bound what the decoder may emit and
/// how much of it. `d_model` is the width both halves work at, and each names its own head count
/// and depth.
///
/// `suppress_tokens` is the checkpoint's own list of things never to emit, and one that names
/// none still loads.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub d_model: usize,
    pub encoder_attention_heads: usize,
    pub encoder_layers: usize,
    pub vocab_size: usize,
    pub max_target_positions: usize,
    pub decoder_attention_heads: usize,
    pub decoder_layers: usize,
    #[serde(default)]
    pub suppress_tokens: Vec<u32>,
}

pub const DTYPE: crate::tensor::DType = crate::tensor::DType::F32;

// Audio parameters.
pub const SAMPLE_RATE: usize = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const CHUNK_LENGTH: usize = 30;
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE; // 480000 samples / 30 s
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000 mel frames

// The reference's temperature-fallback criteria: a segment whose average logprob falls below
// LOGPROB_THRESHOLD, or whose gzip compression ratio exceeds COMPRESSION_RATIO_THRESHOLD, is
// decoded again at the next temperature up.
//
// NOT WIRED. `audio_engine` computes and REPORTS both metrics on every segment - the client
// sees `avg_logprob` and `compression_ratio` - but nothing retries. These four are kept as the
// specification of the retry, not as a claim that it happens; a reader comparing us against
// whisper.cpp should know the difference.
pub const NO_SPEECH_THRESHOLD: f64 = 0.6;
pub const LOGPROB_THRESHOLD: f64 = -1.0;
pub const TEMPERATURES: [f64; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
pub const COMPRESSION_RATIO_THRESHOLD: f64 = 2.4;

// Tokenizer-dependent bits.
pub const SOT_TOKEN: &str = "<|startoftranscript|>";
pub const TRANSCRIBE_TOKEN: &str = "<|transcribe|>";
pub const TRANSLATE_TOKEN: &str = "<|translate|>";
pub const NO_TIMESTAMPS_TOKEN: &str = "<|notimestamps|>";
pub const EOT_TOKEN: &str = "<|endoftext|>";
pub const NO_SPEECH_TOKENS: [&str; 2] = ["<|nocaptions|>", "<|nospeech|>"];

/// What the encoder is actually fed: a log-mel spectrogram of the waveform.
///
/// The waveform is cut into frames of [`N_FFT`] samples every [`HOP_LENGTH`], each windowed and
/// transformed; the power in each frequency bin is then pooled into mel bands by a filterbank
/// that ships with the checkpoint, and the result is taken to a log and flattened into a range
/// the encoder was trained on.
///
/// Two conventions here are the model's rather than the transform's, and both matter because
/// the filterbank was fitted against them. The spectrum is folded - each bin is summed with its
/// mirror image from the negative frequencies rather than being doubled or dropped - and the
/// final normalisation is a floor eight decades below the loudest bin in the WHOLE utterance,
/// so a quiet clip and a loud one arrive at the encoder at the same scale.
pub mod audio {
    use std::sync::Arc;
    use std::thread;

    /// The discrete Fourier transform of a real frame, as interleaved real and imaginary parts.
    ///
    /// A transform of length `n` is two of length `n/2` - one over the even-indexed samples and
    /// one over the odd - recombined by rotating the second by `e^(-2πik/n)` and adding it to
    /// the first for the low half and subtracting for the high one. That halving is the whole
    /// saving, and it only applies while the length stays even; an odd length falls back to
    /// summing every term against every frequency.
    fn transform(frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        if n == 1 {
            return vec![frame[0], 0.0];
        }
        if n % 2 == 1 {
            return transform_directly(frame);
        }

        let even: Vec<f32> = frame.iter().step_by(2).copied().collect();
        let odd: Vec<f32> = frame.iter().skip(1).step_by(2).copied().collect();
        let (even, odd) = (transform(&even), transform(&odd));

        let mut out = vec![0.0f32; n * 2];
        let half = n / 2;
        for k in 0..half {
            let theta = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
            let (re, im) = (theta.cos(), -theta.sin());
            // The odd half's term for bin k, rotated by that angle, added to the even half's
            // for the low bin and subtracted for the high one. Written as one expression per
            // part rather than through a named rotation: the sum is a float, so where the
            // parentheses fall is part of the answer, and this is the association the model was
            // fed before.
            let (odd_re, odd_im) = (odd[2 * k], odd[2 * k + 1]);
            out[2 * k] = even[2 * k] + re * odd_re - im * odd_im;
            out[2 * k + 1] = even[2 * k + 1] + re * odd_im + im * odd_re;
            out[2 * (k + half)] = even[2 * k] - re * odd_re + im * odd_im;
            out[2 * (k + half) + 1] = even[2 * k + 1] - re * odd_im - im * odd_re;
        }
        out
    }

    /// Every frequency against every sample, for the lengths the halving cannot reach.
    fn transform_directly(frame: &[f32]) -> Vec<f32> {
        let n = frame.len();
        let mut out = Vec::with_capacity(2 * n);
        for k in 0..n {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (j, &v) in frame.iter().enumerate() {
                let angle = 2.0 * std::f32::consts::PI * k as f32 * j as f32 / n as f32;
                re += v * angle.cos();
                im -= v * angle.sin();
            }
            out.push(re);
            out.push(im);
        }
        out
    }

    /// How every frame of one spectrogram is cut and measured, which is the same for all of
    /// them and for all the workers.
    ///
    /// `n_threads` belongs here with the rest because it is what tells a worker which frames are
    /// its own: it takes every `n_threads`-th one.
    #[derive(Clone, Copy)]
    struct Framing {
        fft_size: usize,
        fft_step: usize,
        n_len: usize,
        n_mel: usize,
        n_threads: usize,
    }

    /// The mel bands of every `n_threads`-th frame, starting at `ith`.
    ///
    /// Each worker fills only the frames it owns and leaves the rest at zero, so the callers'
    /// results add together into one spectrogram without any of them having to agree on a
    /// boundary.
    fn mel_of_every_nth_frame(
        ith: usize,
        hann: &[f32],
        samples: &[f32],
        filters: &[f32],
        framing: Framing,
    ) -> Vec<f32> {
        let Framing {
            fft_size,
            fft_step,
            n_len,
            n_mel,
            n_threads,
        } = framing;
        let bins = 1 + fft_size / 2;
        let mut frame = vec![0.0f32; fft_size];
        let mut mel = vec![0.0f32; n_len * n_mel];
        let last = std::cmp::min(samples.len() / fft_step + 1, n_len);

        for i in (ith..last).step_by(n_threads) {
            let offset = i * fft_step;
            let taken = std::cmp::min(fft_size, samples.len() - offset);
            for j in 0..taken {
                frame[j] = hann[j] * samples[offset + j];
            }
            frame[taken..].fill(0.0);

            let spectrum = transform(&frame);
            // Power, then folded onto the positive frequencies: bin j and bin n-j are the same
            // frequency seen from either side, and the filterbank expects their sum.
            let mut power: Vec<f32> = (0..fft_size)
                .map(|j| {
                    spectrum[2 * j] * spectrum[2 * j] + spectrum[2 * j + 1] * spectrum[2 * j + 1]
                })
                .collect();
            for j in 1..fft_size / 2 {
                power[j] += power[fft_size - j];
            }

            for band in 0..n_mel {
                let weights = &filters[band * bins..band * bins + bins];
                // Four at a time, as the sum was measured: the order the partial sums are
                // added in is the order the result is defined by.
                let mut sum = 0.0f32;
                let mut k = 0;
                while k < bins.saturating_sub(3) {
                    sum += power[k] * weights[k]
                        + power[k + 1] * weights[k + 1]
                        + power[k + 2] * weights[k + 2]
                        + power[k + 3] * weights[k + 3];
                    k += 4;
                }
                while k < bins {
                    sum += power[k] * weights[k];
                    k += 1;
                }
                mel[band * n_len + i] = f32::max(sum, 1e-10).log10();
            }
        }
        mel
    }

    /// The spectrogram of a whole utterance, `[n_mel, frames]` in row-major order.
    pub fn log_mel_spectrogram_(
        samples: &[f32],
        filters: &[f32],
        fft_size: usize,
        fft_step: usize,
        n_mel: usize,
    ) -> Vec<f32> {
        // The periodic Hann window: it reaches zero at both ends, so consecutive frames overlap
        // without the seam between them showing up as a frequency of its own.
        let hann: Vec<f32> = (0..fft_size)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos()))
            .collect();

        // The encoder reads a fixed number of frames, so the waveform is padded out to a whole
        // number of chunks and then one further chunk of silence.
        let pad = 100 * super::CHUNK_LENGTH / 2;
        let frames = samples.len() / fft_step;
        let n_len = frames.div_ceil(pad) * pad + pad;
        let mut samples = samples.to_vec();
        samples.resize(n_len * fft_step, 0.0);

        // Physical cores only, and an even number of them: a second thread on an SMT sibling
        // adds no throughput to a transform this bandwidth-bound and only burns package power.
        let cores = num_cpus::get_physical();
        let n_threads = std::cmp::max(std::cmp::min(cores - cores % 2, 12), 2);

        // The window and both inputs are read by every worker and written by none, so one owner
        // each and a handle per worker.
        let (hann, samples, filters) = (Arc::new(hann), Arc::new(samples), Arc::new(filters));
        let framing = Framing {
            fft_size,
            fft_step,
            n_len,
            n_mel,
            n_threads,
        };
        let per_thread = thread::scope(|s| {
            // Every worker is started before any is waited on; joining as they are made would
            // run them one after another.
            let mut workers = Vec::with_capacity(n_threads);
            for ith in 0..n_threads {
                let (hann, samples, filters) = (
                    Arc::clone(&hann),
                    Arc::clone(&samples),
                    Arc::clone(&filters),
                );
                workers.push(s.spawn(move || {
                    mel_of_every_nth_frame(ith, &hann, &samples, &filters, framing)
                }));
            }
            let mut parts = Vec::with_capacity(workers.len());
            for worker in workers {
                parts.push(worker.join().expect("a spectrogram worker panicked"));
            }
            parts
        });

        // Each worker left the frames it did not own at zero, so the pieces simply add.
        let mut mel = vec![0.0f32; per_thread[0].len()];
        for part in &per_thread {
            for (m, p) in mel.iter_mut().zip(part) {
                *m += p;
            }
        }

        // Anything eight decades below the loudest bin of the utterance is floored, and the
        // whole thing is brought into roughly [-1, 1]. The reference is the utterance and not
        // the frame, which is what keeps a quiet passage quiet relative to a loud one.
        let loudest = mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // A quiet utterance's loudest bin is still well below one, so the reference is that
        // maximum whatever its sign; zero stands in only for a spectrogram with no bins at all.
        let floor = if loudest.is_finite() { loudest } else { 0.0 } - 8.0;
        for m in mel.iter_mut() {
            *m = f32::max(*m, floor) / 4.0 + 1.0;
        }
        mel
    }

    /// The transform, reachable from the test that holds it against its definition.
    #[cfg(test)]
    pub(super) fn transform_for_test(frame: &[f32]) -> Vec<f32> {
        transform(frame)
    }

    /// The spectrogram a checkpoint asks for: its own number of bands, over the frame length
    /// and hop the whole family was trained with.
    pub fn pcm_to_mel(cfg: &super::Config, samples: &[f32], filters: &[f32]) -> Vec<f32> {
        let (fft_size, fft_step) = (super::N_FFT, super::HOP_LENGTH);
        log_mel_spectrogram_(samples, filters, fft_size, fft_step, cfg.num_mel_bins)
    }
}

#[cfg(test)]
mod audio_tests {
    use super::{audio, HOP_LENGTH, N_FFT, SAMPLE_RATE};

    /// A deterministic waveform: two tones and a chirp, so the spectrum is not flat and a
    /// misplaced bin shows.
    fn waveform(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                0.6 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 1970.0 * t).sin()
                    + 0.1 * (2.0 * std::f32::consts::PI * (200.0 + 900.0 * t) * t).sin()
            })
            .collect()
    }

    /// A filterbank with the shape of a real one - overlapping triangles across the bins.
    fn filters(n_mel: usize, bins: usize) -> Vec<f32> {
        let mut f = vec![0.0f32; n_mel * bins];
        for band in 0..n_mel {
            let centre = (band + 1) as f32 * bins as f32 / (n_mel + 1) as f32;
            let width = bins as f32 / (n_mel + 1) as f32;
            for bin in 0..bins {
                let d = (bin as f32 - centre).abs();
                if d < width {
                    f[band * bins + bin] = 1.0 - d / width;
                }
            }
        }
        f
    }

    /// The transform is the transform.
    ///
    /// The halving that makes it fast is where it can go wrong - a rotation applied to the
    /// wrong half, or the wrong sign on the imaginary part, still returns a spectrum of the
    /// right length. So it is held against every frequency summed against every sample, which
    /// is the definition and admits no such mistake.
    #[test]
    fn the_fast_transform_answers_what_the_definition_does() {
        for n in [1usize, 2, 4, 8, 16, 400] {
            let frame: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 23) as f32) * 0.11 - 1.2)
                .collect();

            let mut want = Vec::with_capacity(2 * n);
            for k in 0..n {
                let (mut re, mut im) = (0.0f64, 0.0f64);
                for (j, &v) in frame.iter().enumerate() {
                    let angle = 2.0 * std::f64::consts::PI * k as f64 * j as f64 / n as f64;
                    re += v as f64 * angle.cos();
                    im -= v as f64 * angle.sin();
                }
                want.push(re as f32);
                want.push(im as f32);
            }

            let got = audio::transform_for_test(&frame);
            let scale = want.iter().fold(1e-6f32, |m, x| m.max(x.abs()));
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() / scale < 2e-5, "n={n}, term {i}: {g} vs {w}");
            }
        }
    }

    /// The spectrogram has the shape the encoder expects, and says where the tones are.
    ///
    /// A front-end that produced plausible numbers in the wrong layout would pass a shape
    /// check, so this also asks that the bands the two tones fall in are louder than the ones
    /// between them - which is the only thing about the output a reader can verify by hand.
    #[test]
    fn the_spectrogram_puts_the_tones_where_they_are() {
        let n_mel = 20usize;
        let bins = 1 + N_FFT / 2;
        let samples = waveform(SAMPLE_RATE / 4);
        let mel =
            audio::log_mel_spectrogram_(&samples, &filters(n_mel, bins), N_FFT, HOP_LENGTH, n_mel);

        assert_eq!(
            mel.len() % n_mel,
            0,
            "the spectrogram is not a whole number of bands"
        );
        let frames = mel.len() / n_mel;
        assert!(frames > 0);
        assert!(
            mel.iter().all(|v| v.is_finite()),
            "the spectrogram has a bin that is not a number"
        );
        // Eight decades, divided by four: whatever the filterbank's absolute scale, the
        // spectrogram the encoder sees spans exactly two.
        let (lo, hi) = mel
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        assert!(
            hi - lo <= 2.0 + 1e-5,
            "the spectrogram spans {} rather than the two decades the floor allows",
            hi - lo
        );

        // Average each band over the frames that carry signal, and find the loudest.
        let voiced = frames.min(samples.len() / HOP_LENGTH);
        let band_level: Vec<f32> = (0..n_mel)
            .map(|b| (0..voiced).map(|f| mel[b * frames + f]).sum::<f32>() / voiced.max(1) as f32)
            .collect();
        let loudest = band_level
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        // 440 Hz sits in the lowest fifth of a linear filterbank over 8 kHz.
        assert!(
            loudest < n_mel / 5,
            "the loudest band is {loudest} of {n_mel}, but the strongest tone is at 440 Hz"
        );
    }
}
