//! Drawing a value: the token sampler for text, and the flow-matching solver the
//! image and audio stacks step along.
//!
//! `crate::inference::sample::<part>`

pub mod flow_unipc;
pub mod token_sampling;

/// Receives `(completed, total)` as a denoise advances: `(0, total)` before the first
/// step, then once after every step. `total` is the number of steps this run takes,
/// after any truncation of the schedule.
pub type StepSink<'a> = &'a dyn Fn(usize, usize);

/// A [`StepSink`] for callers with nobody to report to.
pub fn no_progress(_completed: usize, _total: usize) {}

/// `n` samples of standard gaussian noise, one sequential stream from `seed`.
///
/// Box-Muller over a seeded xorshift, for the samplers that draw their initial latent on
/// the host. The same seed always yields the same stream.
pub fn seeded_gaussian(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    (0..n)
        .map(|_| {
            let (a, b) = (u().max(1e-9), u());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_same_seed_draws_the_same_standard_noise() {
        let a = seeded_gaussian(4096, 42);
        assert_eq!(a, seeded_gaussian(4096, 42));
        assert_ne!(a, seeded_gaussian(4096, 43));
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "variance {var}");
    }
}
