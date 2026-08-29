//! Deterministic sampling shared by the tests and by the measurement harness.
//!
//! Every number a test in this module depends on comes from here, so a failure is a
//! failure of the code under test rather than of a generator that changed its stream
//! between dependency versions.

/// SplitMix64: a fixed-constant bit mixer, written out so the stream never moves.
pub fn split_mix_64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A uniform double in `[0, 1)` from the mixer.
pub fn next_uniform(state: &mut u64) -> f64 {
    (split_mix_64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// `n` standard normal samples, by the Box-Muller transform.
pub fn gaussian_sample(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        // u1 is pulled away from zero so the logarithm stays finite.
        let u1 = next_uniform(&mut state).max(1e-12);
        let u2 = next_uniform(&mut state);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        out.push((r * theta.cos()) as f32);
        if out.len() < n {
            out.push((r * theta.sin()) as f32);
        }
    }
    out
}
