//! Full-Rust STFT / iSTFT (audio->audio #4 foundation) - no new crates.
//!
//! Net-new primitive the modality audit flagged as missing: a Short-Time Fourier
//! Transform and its inverse, needed by spectral-domain audio models (source
//! separation / speech enhancement) and reusable by the spectral-analysis tooling.
//! Uses an iterative radix-2 complex FFT (power-of-two frame size), a Hann window,
//! and Constant-OverLap-Add (COLA) synthesis so analysis->synthesis round-trips to
//! the identity on the interior.

use std::f32::consts::PI;

/// Periodic Hann window of length `n` (matches torch `hann_window(n, periodic=True)`).
pub fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// In-place iterative radix-2 FFT over interleaved (re, im) pairs. `n` must be a
/// power of two. `inverse=true` computes the unnormalized inverse transform (the
/// 1/n scaling is applied by the caller / `istft`).
pub fn fft_inplace(data: &mut [f32], inverse: bool) {
    let n = data.len() / 2;
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of two");
    if n <= 1 {
        return;
    }
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(2 * i, 2 * j);
            data.swap(2 * i + 1, 2 * j + 1);
        }
    }
    // Danielson-Lanczos butterflies. Twiddles are ACCUMULATED in f64 (the
    // iterative f32 product drifts by ~1e-4 over a 2048-point transform, which
    // dominated STFT parity error vs torch); butterflies stay f32.
    let sign = if inverse { 1.0f64 } else { -1.0f64 };
    let mut len = 2usize;
    while len <= n {
        let ang = sign * 2.0 * std::f64::consts::PI / len as f64;
        let (wlr, wli) = (ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let (mut wr, mut wi) = (1.0f64, 0.0f64);
            for k in 0..half {
                let a = 2 * (i + k);
                let b = 2 * (i + k + half);
                let (ur, ui) = (data[a], data[a + 1]);
                let (vr0, vi0) = (data[b], data[b + 1]);
                let (wrf, wif) = (wr as f32, wi as f32);
                let vr = vr0 * wrf - vi0 * wif;
                let vi = vr0 * wif + vi0 * wrf;
                data[a] = ur + vr;
                data[a + 1] = ui + vi;
                data[b] = ur - vr;
                data[b + 1] = ui - vi;
                let nwr = wr * wlr - wi * wli;
                wi = wr * wli + wi * wlr;
                wr = nwr;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// One spectrum frame: `n_fft/2 + 1` complex bins as (re, im) pairs.
pub type Frame = Vec<(f32, f32)>;

/// STFT with centered framing (reflect-free zero padding), Hann window and `hop`
/// stride. Returns one `Frame` per hop. `n_fft` must be a power of two.
pub fn stft(signal: &[f32], n_fft: usize, hop: usize) -> Vec<Frame> {
    stft_padded(signal, n_fft, hop, false)
}

/// STFT with torch-style REFLECT center padding - matches
/// `torch.stft(center=True, pad_mode='reflect')` (the Mel-Band RoFormer / torch
/// default). The existing [`stft`] keeps its zero-padded centering so previous
/// callers' behavior is unchanged. Requires `signal.len() > n_fft / 2`.
pub fn stft_reflect(signal: &[f32], n_fft: usize, hop: usize) -> Vec<Frame> {
    stft_padded(signal, n_fft, hop, true)
}

fn stft_padded(signal: &[f32], n_fft: usize, hop: usize, reflect: bool) -> Vec<Frame> {
    assert!(n_fft.is_power_of_two() && hop > 0);
    let win = hann_window(n_fft);
    let n_bins = n_fft / 2 + 1;
    let pad = n_fft / 2;
    // Center-pad so the first window is centered on sample 0.
    let mut padded = vec![0.0f32; signal.len() + 2 * pad];
    padded[pad..pad + signal.len()].copy_from_slice(signal);
    if reflect {
        // torch reflect padding excludes the edge sample: left[i] = signal[pad - i],
        // right[j] = signal[len - 2 - j].
        assert!(
            signal.len() > pad,
            "stft_reflect: signal ({}) must be longer than n_fft/2 ({pad})",
            signal.len()
        );
        for i in 0..pad {
            padded[i] = signal[pad - i];
        }
        let l = signal.len();
        for j in 0..pad {
            padded[pad + l + j] = signal[l - 2 - j];
        }
    }
    let n_frames = if padded.len() >= n_fft {
        1 + (padded.len() - n_fft) / hop
    } else {
        0
    };
    let mut out = Vec::with_capacity(n_frames);
    let mut buf = vec![0.0f32; 2 * n_fft];
    for f in 0..n_frames {
        let off = f * hop;
        for i in 0..n_fft {
            buf[2 * i] = padded[off + i] * win[i];
            buf[2 * i + 1] = 0.0;
        }
        fft_inplace(&mut buf, false);
        let mut frame = Vec::with_capacity(n_bins);
        for b in 0..n_bins {
            frame.push((buf[2 * b], buf[2 * b + 1]));
        }
        out.push(frame);
    }
    out
}

/// Inverse STFT (COLA overlap-add) reconstructing a signal of length `out_len`.
/// Mirrors `stft`'s centered framing and Hann window; divides by the windowed
/// overlap-add envelope so a windowed COLA pair round-trips to the identity.
pub fn istft(frames: &[Frame], n_fft: usize, hop: usize, out_len: usize) -> Vec<f32> {
    assert!(n_fft.is_power_of_two() && hop > 0);
    let win = hann_window(n_fft);
    let pad = n_fft / 2;
    let total = out_len + 2 * pad;
    let mut acc = vec![0.0f32; total];
    let mut env = vec![0.0f32; total];
    let mut buf = vec![0.0f32; 2 * n_fft];
    for (f, frame) in frames.iter().enumerate() {
        // Rebuild the full Hermitian-symmetric spectrum from the half spectrum.
        for b in 0..n_fft {
            let (re, im) = if b < frame.len() {
                frame[b]
            } else {
                let mirror = n_fft - b;
                let (r, i) = frame[mirror];
                (r, -i) // conjugate symmetry for a real signal
            };
            buf[2 * b] = re;
            buf[2 * b + 1] = im;
        }
        fft_inplace(&mut buf, true);
        let inv_n = 1.0 / n_fft as f32;
        let off = f * hop;
        for i in 0..n_fft {
            if off + i >= total {
                break;
            }
            let s = buf[2 * i] * inv_n * win[i];
            acc[off + i] += s;
            env[off + i] += win[i] * win[i];
        }
    }
    // Normalize by the overlap-add window envelope and strip the center padding.
    let mut out = vec![0.0f32; out_len];
    for i in 0..out_len {
        let e = env[i + pad];
        out[i] = if e > 1e-8 { acc[i + pad] / e } else { 0.0 };
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_roundtrip_identity() {
        let n = 16;
        let mut data: Vec<f32> = (0..n).flat_map(|i| [(i as f32).sin(), 0.0]).collect();
        let orig = data.clone();
        fft_inplace(&mut data, false);
        fft_inplace(&mut data, true);
        for v in data.iter_mut() {
            *v /= n as f32;
        }
        for (a, b) in data.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-4, "fft roundtrip {a} vs {b}");
        }
    }

    #[test]
    fn stft_istft_roundtrip() {
        // A non-trivial signal: sum of two sinusoids.
        let len = 4096;
        let sig: Vec<f32> = (0..len)
            .map(|i| (0.03 * i as f32).sin() + 0.5 * (0.11 * i as f32).cos())
            .collect();
        let (n_fft, hop) = (1024, 256); // 75% overlap -> COLA-satisfying for Hann
        let spec = stft(&sig, n_fft, hop);
        let rec = istft(&spec, n_fft, hop, len);
        assert_eq!(rec.len(), len);
        // Compare on the interior (edges see fewer overlapping windows).
        let (lo, hi) = (n_fft, len - n_fft);
        let mut max_err = 0.0f32;
        for i in lo..hi {
            max_err = max_err.max((rec[i] - sig[i]).abs());
        }
        assert!(
            max_err < 1e-2,
            "STFT round-trip max interior error {max_err}"
        );
    }
}
