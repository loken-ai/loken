//! What every flow-matching sampler in this tree says about its time axis.
//!
//! A rectified-flow model is trained to walk a straight line from noise to image, so a sampler
//! only has to choose WHERE along that line to stop and look, and then walk from one stop to the
//! next. Four things are therefore the same in every family here: the stops are a straight ramp
//! over `[0, 1]`; that ramp is bent toward the noisy end, by an amount that may depend on the
//! image's size; the model is conditioned on those positions read against the horizon its
//! training counted in; and one step moves the sample by the velocity times the distance
//! crossed. Each of the four is stated once, here.
//!
//! Spacing the stops uniformly wastes them: the early, noisy end is where the picture is
//! decided. That is what the two bends below are for.

use crate::tensor::{Device, Result, Tensor};

/// How far to push, for an image of this many tokens.
///
/// A bigger image needs more of its steps spent at the noisy end, so the push is interpolated
/// between two measured points - a small image's and a large one's - and read off at this
/// image's token count. Each family carries its own two points; the interpolation is the same.
pub fn shift_for(image_seq_len: usize, base: (usize, f64), max: (usize, f64)) -> f64 {
    let (base_len, base_shift) = (base.0 as f64, base.1);
    let (max_len, max_shift) = (max.0 as f64, max.1);
    let slope = (max_shift - base_shift) / (max_len - base_len);
    slope * (image_seq_len as f64 - base_len) + base_shift
}

/// Bend a time in `[0, 1]` toward the noisy end.
///
/// `mu` is the push from [`shift_for`], as a log: at `mu = 0` the time is unchanged, and each
/// unit of it multiplies the odds `t / (1 - t)` by `e`. `sigma` is how sharply the bend takes
/// hold - every caller here uses one, which is the plain logistic form.
pub fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    if t <= 0.0 {
        return 0.0;
    }
    let e = mu.exp();
    e / (e + (1.0 / t - 1.0).powf(sigma))
}

/// A latent of standard normal noise, which is where every one of these samplers starts.
pub fn gaussian_latent(shape: (usize, usize, usize, usize), device: &Device) -> Result<Tensor> {
    Tensor::randn(0f32, 1.0, shape, device)
}

/// The other way to bend the time axis: a fixed push, the same at every image size.
///
/// `shift` of one leaves the axis alone; above one it moves every stop toward the noisy end.
/// This is [`time_shift`]'s formula with the fraction cleared - `e^mu` replaced by a constant
/// the checkpoint states outright - which is why a schedule uses one or the other and never
/// both.
pub fn static_shift(shift: f64, t: f64) -> f64 {
    shift * t / (1.0 + (shift - 1.0) * t)
}

/// Where a bent time came from: [`static_shift`] read backwards.
///
/// A caller that is handed a position on the bent axis - a strength dial names one - and wants
/// to lay a straight ramp through it has to undo the bend first, because the ramp is straight
/// only in the axis the bend has not been applied to yet.
///
/// A push of ONE is the identity, here as in [`static_shift`]. A push of zero is not: it sends
/// every position to zero, so it has no inverse, and a caller must not ask for one - the
/// schedule guards on a positive push before reaching here. The denominator vanishes at
/// `t = shift / (shift - 1)`, which sits above one for every push worth bending with, so it is
/// unreachable from a position on the schedule; the guard below is for the degenerate push
/// alone, where the numerator vanishes with it.
pub fn static_unshift(shift: f64, t: f64) -> f64 {
    let d = shift - (shift - 1.0) * t;
    if d.abs() > 1e-12 {
        t / d
    } else {
        t
    }
}

// ------------------------- where the stops sit -------------------------

/// Stop `i` of `span` along the straight line from `from` to `to`.
///
/// `span` is the number of steps the run was asked for, and stop `span` - the far end - is NOT
/// one of them: a schedule is closed off separately, below its last stop, at zero noise.
pub fn ramp_stop(from: f64, to: f64, span: usize, i: usize) -> f64 {
    let fraction = i as f64 / span as f64;
    from * (1.0 - fraction) + to * fraction
}

/// The stops of a straight descent from `top` to zero: `steps + 1` of them, ending at 0.
///
/// This is [`ramp_stop`]'s line with the far end pinned to zero, and it is spelled out here
/// rather than deferred to it. The two agree to about a part in 10^16 and NOT to the last bit  - 
/// `top` scaled by a whole number of steps, against `top` scaled by one minus a fraction, round
/// apart - and a stop that moves in its last bit is a different denoise step, so a family keeps
/// the spelling its checkpoint was rendered against. The test below holds them apart on purpose.
pub fn descent_to_zero(top: f64, steps: usize) -> Vec<f64> {
    let n = steps.max(1);
    (0..=n).map(|v| top * v as f64 / n as f64).rev().collect()
}

/// The scale a checkpoint counts its time on.
///
/// A sigma is a position in `[0, 1]`; the timestep the model is conditioned on is that same
/// position read against the number of steps its training counted. One value, two units - but
/// the trip out and back is not free, so a caller that takes it does so deliberately.
#[derive(Debug, Clone, Copy)]
pub struct Horizon(pub f64);

impl Horizon {
    /// The position `sigma` names, in the units the model is conditioned in.
    pub fn timestep(self, sigma: f64) -> f64 {
        sigma * self.0
    }

    /// The same position back in `[0, 1]`.
    pub fn sigma(self, timestep: f64) -> f64 {
        timestep / self.0
    }
}

// ------------------------- walking them -------------------------

/// The steps of a schedule: every stop paired with the one its step lands on.
///
/// A schedule of `n` steps is `n + 1` stops, so the pairs are its overlapping windows of two.
pub fn steps(schedule: &[f64]) -> impl Iterator<Item = (f64, f64)> + '_ {
    schedule.windows(2).map(|w| (w[0], w[1]))
}

/// One Euler step: the sample moved by the velocity times the distance crossed.
///
/// `dt` is signed and negative in every schedule here - a step walks DOWN toward zero noise  - 
/// and it is applied at the sample's own precision, which is what the model predicted in.
pub fn euler_step(sample: &Tensor, velocity: &Tensor, dt: f64) -> Result<Tensor> {
    sample + (velocity * dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three families wrote this bend three ways. They are the same function, so the ways
    /// they wrote it have to agree - including the one that cleared the fraction to avoid a
    /// division, which is where a sign or a factor would hide.
    #[test]
    fn the_three_spellings_of_the_bend_agree() {
        for mu in [-1.5f64, -0.3, 0.0, 0.7, 2.0] {
            let e = mu.exp();
            for step in 0..=20 {
                let t = step as f64 / 20.0;
                let ours = time_shift(mu, 1.0, t);
                // The form that clears the fraction: e.t / (e.t + 1 - t).
                let cleared = if t <= 0.0 {
                    0.0
                } else {
                    e * t / (e * t + 1.0 - t)
                };
                assert!(
                    (ours - cleared).abs() < 1e-12,
                    "mu={mu} t={t}: {ours} vs {cleared}"
                );
            }
        }
    }

    /// A fixed push of one changes nothing, and any push is the logistic bend at the matching
    /// `mu` - the two are the same function reached from two directions, which is what lets a
    /// schedule pick either.
    #[test]
    fn the_fixed_push_is_the_logistic_one_at_the_matching_mu() {
        for shift in [1.0f64, 1.5, 3.0, 7.0] {
            for step in 0..=20 {
                let t = step as f64 / 20.0;
                let fixed = static_shift(shift, t);
                let logistic = time_shift(shift.ln(), 1.0, t);
                assert!(
                    (fixed - logistic).abs() < 1e-12,
                    "shift={shift} t={t}: {fixed} vs {logistic}"
                );
            }
        }
        for step in 0..=10 {
            let t = step as f64 / 10.0;
            assert!((static_shift(1.0, t) - t).abs() < 1e-12, "a push of one moved {t}");
        }
    }

    /// Undoing the fixed push gives the position back, so a ramp can be laid in the straight
    /// axis and bent afterwards without the start drifting off the value that was asked for.
    #[test]
    fn undoing_the_fixed_push_returns_the_position_it_started_at() {
        for shift in [1.0f64, 1.5, 3.0, 7.0] {
            for step in 0..=20 {
                let t = step as f64 / 20.0;
                let there_and_back = static_shift(shift, static_unshift(shift, t));
                assert!(
                    (there_and_back - t).abs() < 1e-12,
                    "shift={shift}: {t} came back as {there_and_back}"
                );
            }
        }
        // A push of one is the identity in both directions.
        for step in 0..=10 {
            let t = step as f64 / 10.0;
            assert!((static_unshift(1.0, t) - t).abs() < 1e-12, "a push of one moved {t}");
        }
        // A push of zero collapses every position onto zero, so nothing can undo it - and at
        // the top of the schedule it is worse than that: the quotient is zero over zero. Since
        // a schedule starts at a sigma of one, that is a position a caller really can hold,
        // which is why the schedule guards on a positive push before asking for an inverse.
        for step in 1..10 {
            let t = step as f64 / 10.0;
            assert_eq!(static_shift(0.0, t), 0.0, "a push of zero should collapse {t}");
        }
        assert!(static_shift(0.0, 1.0).is_nan(), "zero over zero at the top of the schedule");
        // The singularity of the inverse sits above one for every push that bends, so no
        // position on a schedule can reach it.
        for shift in [1.5f64, 3.0, 7.0] {
            assert!(shift / (shift - 1.0) > 1.0);
        }
    }

    /// The two spellings of a stop are NOT interchangeable, and this measures where.
    ///
    /// They describe the same straight line and agree to about a part in 10^16, so merging them
    /// looks free - and on a 4-step schedule it IS free, because quarters are exact in binary.
    /// At the step counts a full model runs they part company in the last bit on roughly half
    /// the stops, and a stop that moves is a denoise step at a different noise level. So the
    /// disagreement is asserted rather than tolerated: whoever collapses these two has to see
    /// this fail.
    #[test]
    fn the_two_spellings_of_a_stop_disagree() {
        for steps in [9usize, 20, 28, 50] {
            for top in [1.0f64, 0.9, 0.75, 0.55] {
                let descent = descent_to_zero(top, steps);
                let differing = (0..=steps)
                    .filter(|&i| descent[i] != ramp_stop(top, 0.0, steps, i))
                    .count();
                assert!(
                    differing > 0,
                    "{steps} steps from {top}: the two spellings agreed on every stop, so one \
                     of them has been folded into the other"
                );
            }
        }
        // Both still describe the same line, to the precision anything but a render cares about.
        let descent = descent_to_zero(1.0, 28);
        for (i, a) in descent.iter().enumerate() {
            let b = ramp_stop(1.0, 0.0, 28, i);
            assert!((a - b).abs() < 1e-15, "stop {i}: {a} vs {b}");
        }
    }

    /// A descent starts where it was asked to, ends at zero, and never turns back.
    #[test]
    fn a_descent_spans_its_ends_and_only_falls() {
        for steps in [1usize, 4, 9, 28] {
            for top in [1.0f64, 0.6] {
                let d = descent_to_zero(top, steps);
                assert_eq!(d.len(), steps + 1, "{steps} steps gave {} stops", d.len());
                assert!((d[0] - top).abs() < 1e-15, "started at {}", d[0]);
                assert_eq!(*d.last().unwrap(), 0.0, "did not finish denoised");
                for w in d.windows(2) {
                    assert!(w[1] < w[0], "{steps} steps from {top} stopped descending");
                }
            }
        }
        // A run of no steps is still a run: one step from the top, not a division by zero.
        assert_eq!(descent_to_zero(1.0, 0), vec![1.0, 0.0]);
    }

    /// A horizon is a change of units and nothing else.
    #[test]
    fn a_horizon_is_the_same_position_in_other_units() {
        let h = Horizon(1000.0);
        for step in 0..=20 {
            let sigma = step as f64 / 20.0;
            assert!((h.sigma(h.timestep(sigma)) - sigma).abs() < 1e-15);
        }
        assert_eq!(h.timestep(1.0), 1000.0);
    }

    /// The pairs a schedule is walked in: every stop with the one its step lands on.
    #[test]
    fn a_schedule_is_walked_in_overlapping_pairs() {
        let s = [1.0, 0.6, 0.3, 0.0];
        assert_eq!(
            steps(&s).collect::<Vec<_>>(),
            vec![(1.0, 0.6), (0.6, 0.3), (0.3, 0.0)]
        );
        // A schedule too short to step is walked zero times rather than panicking.
        assert_eq!(steps(&s[..1]).count(), 0);
        assert_eq!(steps(&[]).count(), 0);
    }

    /// The push passes through both of the points it interpolates between, whatever they are.
    #[test]
    fn the_push_meets_the_points_it_is_drawn_through() {
        let base = (256usize, 0.5);
        let max = (4096usize, 1.15);
        assert!((shift_for(base.0, base, max) - base.1).abs() < 1e-12);
        assert!((shift_for(max.0, base, max) - max.1).abs() < 1e-12);
        // And it keeps going past them rather than clamping, which is what the callers expect
        // for an image larger than the two it is drawn through.
        assert!(shift_for(8192, base, max) > max.1);
    }

    /// A step moves the sample by exactly the velocity times the distance it crossed.
    ///
    /// Every family's denoise loop reaches this one expression now, so it is checked against
    /// arithmetic done by hand, element by element, over a real descent rather than a made-up
    /// pair of times: a step that takes the wrong two stops scales every prediction by the wrong
    /// amount and changes nothing about the shape of what comes out.
    #[test]
    fn a_step_moves_the_sample_by_the_distance_it_crossed() {
        let dev = Device::Cpu;
        let n = 64usize;
        let start: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let velocity: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.11).cos() * 2.0 - 0.5)
            .collect();
        let v = Tensor::from_vec(velocity.clone(), (4, 16), &dev).unwrap();

        let mut sample = start;
        for (now, next) in steps(&descent_to_zero(1.0, 4)) {
            let dt = next - now;
            assert!(dt < 0.0, "a denoise step must walk toward less noise");
            let want: Vec<f32> = sample
                .iter()
                .zip(&velocity)
                .map(|(s, p)| s + p * dt as f32)
                .collect();

            let x = Tensor::from_vec(sample, (4, 16), &dev).unwrap();
            sample = euler_step(&x, &v, dt)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            for (i, (got, want)) in sample.iter().zip(&want).enumerate() {
                assert!((got - want).abs() < 1e-6, "element {i}: {got} vs {want}");
            }
        }
    }
}
