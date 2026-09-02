//! Shared WAV I/O + resampling helpers (canonical implementations).
//!
//! One home for the RIFF/WAVE plumbing that used to be copy-pasted across the
//! audio models (ACE-Step VAE, pocket-tts): a 16-bit PCM
//! writer/reader (mono convenience wrappers over the planar multi-channel
//! forms) and a linear-interpolation resampler with `align_corners=False`
//! (half-sample-offset) semantics.
//!
//! Dev bins under `bin/*.rs` intentionally keep their own tiny local copies  -
//! they are standalone and some are feature-gated.

use crate::tensor::{Error, Result};

/// Encode planar f32 audio `[c.t]` (channel-major) as a 16-bit little-endian
/// PCM WAV byte stream. Samples are clamped to `[-1, 1]`, scaled symmetrically
/// (`x32767`, round-to-nearest) and interleaved into WAV frame order.
pub fn write_wav_planar(planar: &[f32], c: usize, t: usize, sample_rate: u32) -> Vec<u8> {
    debug_assert_eq!(planar.len(), c * t);
    let bits = 16u16;
    let block_align = c as u16 * (bits / 8);
    let byte_rate = sample_rate * block_align as u32;
    let data_len = (c * t * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio_format = PCM
    out.extend_from_slice(&(c as u16).to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for ti in 0..t {
        for ci in 0..c {
            let s = planar[ci * t + ti].clamp(-1.0, 1.0);
            // symmetric scale; round-to-nearest
            let v = (s * 32767.0).round() as i32;
            out.extend_from_slice(&(v as i16).to_le_bytes());
        }
    }
    out
}

/// Encode mono f32 PCM as a 16-bit little-endian PCM WAV byte stream.
pub fn write_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    write_wav_planar(pcm, 1, pcm.len(), sample_rate)
}

/// Decode a 16-bit little-endian PCM WAV byte stream -> planar f32 `[c.t]`
/// (channel-major), plus `(channels, sample_rate)`. Minimal RIFF parser: scans
/// chunks for `fmt ` (PCM s16 only) + `data`. The inverse of
/// [`write_wav_planar`].
pub fn read_wav_planar(bytes: &[u8]) -> Result<(Vec<f32>, usize, u32)> {
    let err = |m: &str| Error(format!("wav decode: {m}"));
    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(err("not a RIFF/WAVE file"));
    }
    let rd_u16 = |b: &[u8], o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
    let rd_u32 = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
    let (mut channels, mut sample_rate, mut bits) = (0usize, 0u32, 0u16);
    let (mut data_off, mut data_len) = (0usize, 0usize);
    let mut p = 12usize;
    while p + 8 <= bytes.len() {
        let id = &bytes[p..p + 4];
        let sz = rd_u32(bytes, p + 4) as usize;
        let body = p + 8;
        if id == b"fmt " && body + 16 <= bytes.len() {
            if rd_u16(bytes, body) != 1 {
                return Err(err("only PCM (format 1) supported"));
            }
            channels = rd_u16(bytes, body + 2) as usize;
            sample_rate = rd_u32(bytes, body + 4);
            bits = rd_u16(bytes, body + 14);
        } else if id == b"data" {
            data_off = body;
            data_len = sz.min(bytes.len() - body);
        }
        p = body + sz + (sz & 1); // chunks are word-aligned
    }
    if channels == 0 || data_off == 0 {
        return Err(err("missing fmt/data chunk"));
    }
    if bits != 16 {
        return Err(err("only 16-bit PCM supported"));
    }
    let frames = data_len / (2 * channels);
    let mut planar = vec![0f32; channels * frames];
    for f in 0..frames {
        for c in 0..channels {
            let o = data_off + (f * channels + c) * 2;
            let s = i16::from_le_bytes([bytes[o], bytes[o + 1]]) as f32 / 32768.0;
            planar[c * frames + f] = s;
        }
    }
    Ok((planar, channels, sample_rate))
}

/// Decode a 16-bit PCM WAV to mono f32 + sample rate. Multi-channel input is
/// mono-ized by taking channel 0 (the convention of the existing readers).
pub fn read_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let (mut planar, channels, sample_rate) = read_wav_planar(bytes)?;
    if channels > 1 {
        planar.truncate(planar.len() / channels); // channel-major -> ch0 is the first `frames` samples
    }
    Ok((planar, sample_rate))
}

/// Linear-interpolation resampler (`align_corners=False` semantics via
/// input/output ratio - sample positions offset by half a sample, edges
/// clamped).
pub fn resample_linear(sig: &[f32], from_sr: u32, to_sr: u32) -> Vec<f32> {
    if from_sr == to_sr || sig.is_empty() {
        return sig.to_vec();
    }
    let out = (sig.len() as u64 * to_sr as u64 / from_sr as u64) as usize;
    let scale = sig.len() as f32 / out as f32;
    (0..out)
        .map(|j| {
            let pos = ((j as f32) + 0.5) * scale - 0.5;
            let i0 = pos.floor();
            let w = pos - i0;
            let a = (i0 as isize).clamp(0, sig.len() as isize - 1) as usize;
            let b = (i0 as isize + 1).clamp(0, sig.len() as isize - 1) as usize;
            sig[a] * (1.0 - w) + sig[b] * w
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_mono() {
        let pcm: Vec<f32> = (0..480)
            .map(|i| (i as f32 / 480.0 * std::f32::consts::TAU).sin() * 0.5)
            .collect();
        let bytes = write_wav(&pcm, 24_000);
        assert_eq!(&bytes[0..4], b"RIFF");
        let (back, sr) = read_wav(&bytes).unwrap();
        assert_eq!(sr, 24_000);
        assert_eq!(back.len(), pcm.len());
        let max_err = pcm
            .iter()
            .zip(&back)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_err < 1.0 / 32000.0,
            "16-bit quantization error only, got {max_err}"
        );
    }

    #[test]
    fn wav_roundtrip_stereo_ch0() {
        // ch0 = ramp, ch1 = zeros; mono read must return ch0.
        let t = 100usize;
        let mut planar = vec![0f32; 2 * t];
        for i in 0..t {
            planar[i] = i as f32 / t as f32 * 0.9;
        }
        let bytes = write_wav_planar(&planar, 2, t, 16_000);
        let (p2, c, sr) = read_wav_planar(&bytes).unwrap();
        assert_eq!((c, sr), (2, 16_000));
        assert_eq!(p2.len(), 2 * t);
        let (mono, _) = read_wav(&bytes).unwrap();
        assert_eq!(mono.len(), t);
        assert!(mono[t - 1] > 0.8, "ch0 taken, not ch1/average");
    }

    #[test]
    fn resample_identity_and_length() {
        let sig: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin()).collect();
        assert_eq!(resample_linear(&sig, 24_000, 24_000), sig);
        assert_eq!(resample_linear(&sig, 24_000, 12_000).len(), 500);
        assert_eq!(resample_linear(&sig, 16_000, 24_000).len(), 1500);
        assert!(resample_linear(&[], 16_000, 24_000).is_empty());
    }
}
