//! FLUX.2 Klein sampling: the flow-match Euler schedule, and the geometry that feeds it.
//!
//! The schedule is where this family departs from every other flow-match model here, in a way the
//! config file actively misleads about. `scheduler/scheduler_config.json` advertises
//! `use_dynamic_shifting: true` with `base_shift 0.5` / `max_shift 1.15` / `max_image_seq_len
//! 4096` - the ordinary diffusers interpolation - and ALSO a static `shift: 3.0`. The pipeline
//! uses neither: it computes `mu` from its own empirical two-line fit over BOTH the token count
//! and the step count, and passes that in. Reading the config and implementing what it says would
//! give a plausible, wrong schedule.

/// The `mu` FLUX.2 samples with, ported verbatim from the reference `compute_empirical_mu`.
///
/// Two linear fits - one calibrated at 10 steps, one at 200 - interpolated (and extrapolated) in
/// the step count, with the 10-step line dropped entirely above 4300 tokens. The constants are
/// the reference's; they are a fit, not a formula, so there is nothing to derive them from.
pub fn empirical_mu(image_seq_len: usize, num_steps: usize) -> f32 {
    let (a1, b1) = (8.738_095_24e-05f64, 1.898_333_33f64);
    let (a2, b2) = (0.000_169_27f64, 0.456_666_66f64);
    let seq = image_seq_len as f64;
    let m_200 = a2 * seq + b2;
    if seq > 4300.0 {
        return m_200 as f32;
    }
    let m_10 = a1 * seq + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    (a * num_steps as f64 + b) as f32
}

/// The sigma schedule: diffusers' `linspace(1, 1/steps, steps)` time-shifted by `exp(mu)`, with
/// the terminal 0 appended. Returns `steps + 1` values, so `sig[i+1] - sig[i]` is the Euler step.
pub fn sigmas(image_seq_len: usize, steps: usize) -> Vec<f32> {
    let steps = steps.max(1);
    let e = empirical_mu(image_seq_len, steps).exp() as f64;
    let mut out: Vec<f32> = (0..steps)
        .map(|i| {
            // linspace(1, 1/steps, steps)
            let s = if steps == 1 {
                1.0
            } else {
                1.0 - (i as f64) * (1.0 - 1.0 / steps as f64) / (steps as f64 - 1.0)
            };
            crate::inference::model::flow_match::time_shift(e.ln(), 1.0, s) as f32
        })
        .collect();
    out.push(0.0);
    out
}

/// The DiT grid a requested pixel size maps to, and the token count that follows.
///
/// The VAE compresses by 8 and the pipeline patchifies the latent by a further 2, so a DiT token
/// is 16 pixels a side - not 8, which is the figure every other family here uses. The reference
/// rounds through that pair explicitly (`2 * (px // (vae_scale_factor * 2))` gives the latent
/// side, halved again for the grid), so a size that is not a multiple of 16 is floored rather
/// than rejected.
pub fn grid_for(height: usize, width: usize) -> (usize, usize) {
    (height / 16, width / 16)
}

/// Tokens the DiT sees for a pixel size - the value `empirical_mu` is a function of.
pub fn image_seq_len(height: usize, width: usize) -> usize {
    let (gh, gw) = grid_for(height, width);
    gh * gw
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against a reference run on the published pipeline source. These are
    /// the numbers the schedule is BUILT from, so an error here is invisible downstream - every
    /// shape stays right and the image is merely wrong.
    #[test]
    fn mu_matches_the_reference_fit() {
        for (seq, steps, want) in [
            (4096usize, 4usize, 2.291_179_89f32),
            (4096, 28, 2.151_443_16),
            (1024, 4, 2.030_689_71),
            // Above 4300 tokens the step count stops mattering: the 200-step line alone.
            (6400, 20, 1.539_994_66),
            (6400, 4, 1.539_994_66),
        ] {
            let got = empirical_mu(seq, steps);
            assert!(
                (got - want).abs() < 1e-6,
                "mu({seq},{steps}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn sigma_schedule_matches_the_reference() {
        let s = sigmas(4096, 4);
        let want = [1.0f32, 0.967_384, 0.908_144, 0.767_2, 0.0];
        assert_eq!(s.len(), want.len());
        for (got, want) in s.iter().zip(want) {
            assert!((got - want).abs() < 1e-5, "sigmas {s:?} != {want:?}");
        }
    }

    /// Monotone decreasing to exactly zero: an Euler step reads `sig[i+1] - sig[i]`, so a
    /// schedule that is not ordered integrates backwards over part of the trajectory.
    #[test]
    fn schedule_descends_to_zero() {
        for steps in [1usize, 2, 4, 28, 50] {
            let s = sigmas(4096, steps);
            assert_eq!(s.len(), steps + 1);
            assert!((s[0] - 1.0).abs() < 1e-6, "starts at {}", s[0]);
            assert_eq!(*s.last().unwrap(), 0.0);
            assert!(s.windows(2).all(|w| w[0] > w[1]), "not decreasing: {s:?}");
        }
    }

    /// A DiT token is 16 pixels a side here, not the 8 the rest of the fleet uses.
    #[test]
    fn grid_accounts_for_both_the_vae_and_the_patchify() {
        assert_eq!(grid_for(1024, 1024), (64, 64));
        assert_eq!(image_seq_len(1024, 1024), 4096);
        assert_eq!(image_seq_len(512, 512), 1024);
        // Non-multiples floor, matching the reference's integer division.
        assert_eq!(grid_for(1000, 1000), (62, 62));
    }
}
