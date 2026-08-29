//! Z-Image sampling and its FlowMatch-Euler scheduler: where the denoise loop stops along the
//! straight line from noise to image, and what one step of it does.
//!
//! Pure tensor and scalar math, no weights. The two corrections to the time axis are the shared
//! ones in [`crate::inference::model::flow_match`]; what is written here is the ladder they bend
//! and the Euler step that walks it.

use crate::inference::model::flow_match;
use crate::tensor::{DType, Device, Result, Tensor};

// ------------------------- sampling -------------------------

pub fn get_noise(
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    flow_match::gaussian_latent((batch_size, channels, height, width), device)
}

pub fn postprocess_image(image: &Tensor) -> Result<Tensor> {
    let lo = Tensor::full(-1.0f32, image.dims(), &image.device())?;
    let hi = Tensor::full(1.0f32, image.dims(), &image.device())?;
    let image = image.maximum(&lo)?.minimum(&hi)?;
    let image = ((image + 1.0)? * 127.5)?;
    image.to_dtype(DType::U8)
}

// ------------------------- scheduler -------------------------

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SchedulerConfig {
    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,
    #[serde(default = "default_shift")]
    pub shift: f64,
    #[serde(default = "default_use_dynamic_shifting")]
    pub use_dynamic_shifting: bool,
}

// What the published scheduler states when its config file does not. Stated once, so a config
// file that omits a field and a config built in code cannot drift apart.
crate::serde_defaults! {
    default_num_train_timesteps: usize = 1000;
    default_shift: f64 = 3.0;
    default_use_dynamic_shifting: bool = false;
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: default_num_train_timesteps(),
            shift: default_shift(),
            use_dynamic_shifting: default_use_dynamic_shifting(),
        }
    }
}

impl SchedulerConfig {
    /// What the published Turbo checkpoint states - which is what the defaults above already
    /// say, because they were taken from it.
    pub fn z_image_turbo() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub config: SchedulerConfig,
    pub timesteps: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub sigma_min: f64,
    pub sigma_max: f64,
    step_index: usize,
}

/// The stops of a straight ramp from `top` down to `bottom`, stated on the training horizon.
///
/// `stops` is how many there are and NOT one of them: the ramp is open at the bottom end, and
/// what closes a run at zero noise is `install`. Every run this scheduler lays out is this same
/// ramp - a fresh one spans the two training ends, an img2img one starts part-way down - so
/// both read it from here rather than each spelling it out.
fn ramp_timesteps(horizon: flow_match::Horizon, top: f64, bottom: f64, stops: usize) -> Vec<f64> {
    (0..stops)
        .map(|i| horizon.timestep(flow_match::ramp_stop(top, bottom, stops, i)))
        .collect()
}

impl FlowMatchEulerDiscreteScheduler {
    /// The training schedule itself: one rung per training timestep.
    ///
    /// A sigma is a position on the straight line from noise to image, so rung `k` of
    /// `num_train_timesteps` sits at the fraction `k / num_train_timesteps` of the way along
    /// it, counted down from full noise. The push bends those fractions toward the noisy end.
    /// `timesteps` is the same ladder read on the training horizon rather than on `[0, 1]`.
    ///
    /// What survives this is the pair of ENDS: a run of a few steps interpolates between them,
    /// and replaces the ladder itself on its first call to [`Self::set_timesteps`].
    pub fn new(config: SchedulerConfig) -> Self {
        let horizon = flow_match::Horizon(config.num_train_timesteps as f64);
        // Dynamic shifting takes its push per image, from the `mu` handed to `set_timesteps`,
        // so at this point it leaves the axis alone - and a push of one is exactly that.
        let push = if config.use_dynamic_shifting {
            1.0
        } else {
            config.shift
        };
        let sigmas: Vec<f64> = (1..=config.num_train_timesteps)
            .rev()
            .map(|rung| flow_match::static_shift(push, horizon.sigma(rung as f64)))
            .collect();
        let timesteps: Vec<f64> = sigmas.iter().map(|&s| horizon.timestep(s)).collect();
        Self {
            // A schedule with no rungs spans the whole interval and nothing else can be said
            // about it; every published config states a horizon, so nothing reaches this.
            sigma_max: sigmas.first().copied().unwrap_or(1.0),
            sigma_min: sigmas.last().copied().unwrap_or(0.0),
            config,
            timesteps,
            sigmas,
            step_index: 0,
        }
    }

    /// Lay out the ladder for a run of this many steps.
    ///
    /// The ends come from the fields, which hold what the full training schedule spans, and NOT
    /// from `self.sigmas` - that is the ladder, and after one call it is this method's own
    /// output, ending in the zero it appends. Reading it back made a second call build between
    /// the wrong ends, and there is a second call on every img2img request: the caller lays out
    /// a schedule and then `set_timesteps_denoised` lays out a longer one to keep its tail.
    pub fn set_timesteps(&mut self, num_inference_steps: usize, mu: Option<f64>) {
        let sigma_max = self.sigma_max;
        let sigma_min = self.sigma_min;
        let horizon = flow_match::Horizon(self.config.num_train_timesteps as f64);
        let timesteps = ramp_timesteps(horizon, sigma_max, sigma_min, num_inference_steps);
        // The sigmas are read back off the horizon instead of taken from the ramp a second
        // time: the trip out and back can differ in the last bit, and these are the stops a
        // render is judged at.
        let mut sigmas: Vec<f64> = timesteps.iter().map(|&t| horizon.sigma(t)).collect();
        // One push or the other, never both: a per-image `mu` is what a dynamically shifted
        // schedule is waiting for, and the config's fixed push is for a schedule that has no
        // `mu` to take. Either half missing its other half leaves the ramp straight.
        match (mu, self.config.use_dynamic_shifting) {
            (Some(mu), true) => {
                for sigma in &mut sigmas {
                    *sigma = flow_match::time_shift(mu, 1.0, *sigma);
                }
            }
            (None, false) => {
                for sigma in &mut sigmas {
                    *sigma = flow_match::static_shift(self.config.shift, *sigma);
                }
            }
            _ => {}
        }
        self.install(timesteps, sigmas);
    }

    /// Take a freshly laid-out ladder, close it at zero noise, and start at its top.
    ///
    /// The trailing zero is the rung a step lands on after the last one, so a schedule of `n`
    /// steps carries `n + 1` sigmas - which is why every reader indexes `sigmas[i + 1]` and
    /// none of them checks.
    fn install(&mut self, timesteps: Vec<f64>, mut sigmas: Vec<f64>) {
        sigmas.push(0.0);
        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.reset();
    }

    pub fn current_sigma(&self) -> f64 {
        self.sigmas[self.step_index]
    }

    /// How far along the run the current stop is, counted from the noisy end.
    ///
    /// The stop is stated on the training horizon, so the horizon is what turns it back into a
    /// fraction; a checkpoint that counted its training in some other number of steps would be
    /// read on its own scale rather than on this one's.
    pub fn current_timestep_normalized(&self) -> f64 {
        let horizon = self.config.num_train_timesteps as f64;
        let t = self.timesteps.get(self.step_index).copied().unwrap_or(0.0);
        (horizon - t) / horizon
    }

    pub fn step(&mut self, model_output: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let prev_sample = flow_match::euler_step(sample, model_output, sigma_next - sigma)?;
        self.step_index += 1;
        Ok(prev_sample)
    }

    pub fn reset(&mut self) {
        self.step_index = 0;
    }

    pub fn num_inference_steps(&self) -> usize {
        self.timesteps.len()
    }

    pub fn step_index(&self) -> usize {
        self.step_index
    }

    /// Rebuild the ladder to run from EXACTLY `sigma` down to 0 in `num_steps` steps.
    ///
    /// `sigma` is the fraction of the source replaced by noise, so it is the strength
    /// slider's own meaning. Two things were wrong with starting part-way along the
    /// full-range ladder instead.
    ///
    /// It quantised the control. A 9-step Turbo ladder is
    /// `[1.000, 0.960, 0.913, 0.857, ...]`, and `round((1-strength)*steps)` gave 0.95
    /// and 1.00 the SAME node - sigma 1.0, which discards the source outright and turns
    /// an edit into a fresh generation. 0.90 and 0.85 collapsed together too. Nine
    /// reachable values across the whole slider, with a cliff at the end.
    ///
    /// And it starved the schedule at the other end: only the nodes BELOW the start
    /// remain, so strength 0.3 left one step and 0.2 left none - the edit that asked
    /// for the least change also got the least denoising to finish it.
    ///
    /// Rebuilding fixes both: the start is exact at any strength, and every strength
    /// gets the steps the caller asked for. The ladder is laid out in the schedule's
    /// PRE-shift domain, where it is a straight line, then shifted - the same
    /// construction `set_timesteps` uses, so `timesteps` and `sigmas` keep the exact
    /// relationship the model is conditioned on.
    pub fn restart_from_sigma(&mut self, sigma: f64, num_steps: usize) {
        let sigma = sigma.clamp(0.0, self.sigma_max);
        let shift = self.config.shift;
        let statically_shifted = !self.config.use_dynamic_shifting && shift > 0.0;
        // Undo the bend to get the position the straight ramp is drawn in.
        let u_start = if statically_shifted {
            flow_match::static_unshift(shift, sigma)
        } else {
            sigma
        };
        let n = num_steps.max(1);
        let horizon = flow_match::Horizon(self.config.num_train_timesteps as f64);
        let timesteps = ramp_timesteps(horizon, u_start, self.sigma_min, n);
        let sigmas: Vec<f64> = timesteps
            .iter()
            .map(|&t| {
                let u = horizon.sigma(t);
                if statically_shifted {
                    flow_match::static_shift(shift, u)
                } else {
                    u
                }
            })
            .collect();
        self.install(timesteps, sigmas);
    }

    /// Schedule for an img2img run at `denoise` strength.
    ///
    /// The schedule is truncated, not rescaled: lay out a LONGER run - as many steps as
    /// `denoise` is a fraction of - and keep its last `steps + 1` sigmas. That gives the caller
    /// exactly the steps they asked for, starting at the noise level `denoise` names, on the
    /// schedule shape the checkpoint states, which is the part a ramp rebuilt from the start
    /// point gets wrong. On an unshifted schedule the two agree exactly (the tail of a straight
    /// ramp from 1 is a straight ramp from `denoise`), which is why rebuilding measured clean on
    /// this checkpoint; on a shifted one they do not, and the truncation is what the model was
    /// conditioned against.
    ///
    /// The alternative - truncating a fixed-length schedule - is what gave a 9-step
    /// ladder nine reachable settings, with 0.95 and 1.00 sharing the node that discards
    /// the source, and left low strengths with almost no steps to finish in.
    pub fn set_timesteps_denoised(&mut self, steps: usize, mu: Option<f64>, denoise: f64) {
        if denoise >= 0.9999 || steps == 0 {
            self.set_timesteps(steps, mu);
            return;
        }
        if denoise <= 0.0 {
            self.set_timesteps(steps, mu);
            self.step_index = self.timesteps.len();
            return;
        }
        // Truncation, not rounding: a cast is what states the longer run's length.
        let new_steps = ((steps as f64) / denoise) as usize;
        let new_steps = new_steps.max(steps);
        self.set_timesteps(new_steps, mu);
        // sigmas has new_steps+1 entries (the trailing 0), timesteps has new_steps.
        let keep_sigmas = steps + 1;
        let cut = self.sigmas.len().saturating_sub(keep_sigmas);
        self.sigmas.drain(0..cut);
        let tcut = self.timesteps.len().saturating_sub(steps);
        self.timesteps.drain(0..tcut);
        self.step_index = 0;
    }
}

/// Z-Image's spelling of [`flow_match::shift_for`]: it carries the two points it interpolates
/// through as four separate numbers, which is how its caller has them.
pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    flow_match::shift_for(
        image_seq_len,
        (base_seq_len, base_shift),
        (max_seq_len, max_shift),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor;

    fn turbo_ladder(steps: usize) -> FlowMatchEulerDiscreteScheduler {
        let mut s = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        s.set_timesteps(steps, None);
        s
    }

    /// The img2img strength control has to be usable, not merely present.
    ///
    /// Rounding strength to a ladder node gave a 9-step Turbo schedule only nine
    /// reachable settings: 0.95 and 1.00 landed on the SAME node, and that node is
    /// sigma 1.0 - the source discarded outright. 0.90 and 0.85 collapsed too. That is
    /// the reported "changes everything around 0.95, does nothing at 0.9".
    #[test]
    fn strength_moves_the_start_smoothly_instead_of_snapping_to_a_node() {
        let nodes = turbo_ladder(9).sigmas.clone();
        assert!(
            nodes[0] > nodes[1] && nodes[1] > nodes[2],
            "sigmas must descend"
        );

        let mut prev_sigma = f64::NEG_INFINITY;
        for pct in (5..=100).step_by(5) {
            let strength = pct as f64 / 100.0;
            let mut sch = turbo_ladder(9);
            sch.restart_from_sigma(strength, 9);
            let sigma = sch.current_sigma();

            // Exactly what was asked for, not the nearest node.
            assert!(
                (sigma - strength).abs() < 1e-9,
                "strength {strength} started at {sigma}, not at itself"
            );
            // Strictly monotone: no two distinct strengths may behave identically.
            assert!(
                sigma > prev_sigma,
                "strength {strength} did not move the start"
            );
            prev_sigma = sigma;
            // Descends to zero, so the render always finishes.
            for w in sch.sigmas.windows(2) {
                assert!(
                    w[1] < w[0],
                    "sigmas stopped descending at strength {strength}"
                );
            }
            assert_eq!(*sch.sigmas.last().unwrap(), 0.0, "must end fully denoised");
        }

        // Full strength still means "discard the source", unchanged from before.
        let mut sch = turbo_ladder(9);
        sch.restart_from_sigma(1.0, 9);
        assert!((sch.current_sigma() - 1.0).abs() < 1e-12);
    }

    /// The img2img schedule must be the tail of a longer run, not merely look reasonable:
    /// `steps/denoise` steps truncated to a whole number, keeping the last `steps + 1` sigmas.
    #[test]
    fn the_denoised_schedule_truncates_rather_than_rescales() {
        for steps in [4usize, 9, 20] {
            for pct in [10usize, 25, 50, 75, 90] {
                let denoise = pct as f64 / 100.0;
                let mut sch = turbo_ladder(steps);
                sch.set_timesteps_denoised(steps, None, denoise);

                // Exactly the requested run length, every time.
                assert_eq!(
                    sch.sigmas.len(),
                    steps + 1,
                    "denoise {denoise} at {steps} steps kept {} sigmas",
                    sch.sigmas.len()
                );
                assert_eq!(sch.timesteps.len(), steps);

                // It is literally the tail of the longer schedule.
                let new_steps = (((steps as f64) / denoise) as usize).max(steps);
                let mut full = turbo_ladder(steps);
                full.set_timesteps(new_steps, None);
                let tail = &full.sigmas[full.sigmas.len() - (steps + 1)..];
                assert_eq!(&sch.sigmas[..], tail, "denoise {denoise}: not the tail");

                // Descends to zero, so the render always finishes.
                assert_eq!(*sch.sigmas.last().unwrap(), 0.0);
                for w in sch.sigmas.windows(2) {
                    assert!(
                        w[1] < w[0],
                        "denoise {denoise}: schedule stopped descending"
                    );
                }
            }
            // Full strength is the plain schedule, untouched.
            let mut a = turbo_ladder(steps);
            a.set_timesteps_denoised(steps, None, 1.0);
            let mut b = turbo_ladder(steps);
            b.set_timesteps(steps, None);
            assert_eq!(a.sigmas, b.sigmas, "denoise=1 must be the plain schedule");
        }
    }

    /// Lower strength must start at lower noise - the property the dial exists for.
    #[test]
    fn a_lower_strength_starts_at_less_noise() {
        let mut prev = f64::INFINITY;
        for pct in [90usize, 75, 50, 25, 10] {
            let mut sch = turbo_ladder(9);
            sch.set_timesteps_denoised(9, None, pct as f64 / 100.0);
            let s0 = sch.current_sigma();
            assert!(
                s0 < prev,
                "strength {pct}% started at {s0}, not below {prev}"
            );
            prev = s0;
        }
    }

    /// Every strength must get the steps the caller asked for.
    ///
    /// Reusing only the nodes BELOW the start looks reasonable and quietly starves the
    /// low end: on a 9-step ladder strength 0.3 left ONE step and 0.2 left none, so the
    /// edit asking for the least change also got the least denoising to finish it.
    #[test]
    fn every_strength_gets_the_full_step_budget() {
        for steps in [4usize, 9, 20] {
            for pct in [5usize, 20, 30, 50, 90, 100] {
                let mut sch = turbo_ladder(steps);
                sch.restart_from_sigma(pct as f64 / 100.0, steps);
                let executed = sch.sigmas.len() - 1;
                assert_eq!(
                    executed, steps,
                    "strength {pct}% at {steps} steps executed {executed}"
                );
            }
        }
    }

    /// The conditioning must follow the sigma. This schedule keeps `timesteps`
    /// unshifted while shifting `sigmas`, so moving a node's sigma without moving its
    /// `t` would silently tell the model a different noise level than the latent holds.
    #[test]
    fn the_timestep_still_matches_the_sigma_after_moving_the_start() {
        let cfg = SchedulerConfig::z_image_turbo();
        let shift = cfg.shift;
        for pct in [55usize, 70, 85, 95] {
            let strength = pct as f64 / 100.0;
            let mut sch = turbo_ladder(9);
            sch.restart_from_sigma(strength, 9);
            let j = 0usize;
            // Re-applying the schedule's own shift to the recovered position must give
            // back the sigma we started at.
            let u = sch.timesteps[j] / cfg.num_train_timesteps as f64;
            let reshifted = flow_match::static_shift(shift, u);
            assert!(
                (reshifted - strength).abs() < 1e-9,
                "t at strength {strength} re-shifts to {reshifted}"
            );
        }
    }

    /// The Euler step moves the sample by exactly the velocity times the sigma it crossed.
    ///
    /// A flow-matching step has one degree of freedom - which two rungs of the ladder it sits
    /// between - and getting that wrong scales every prediction by the wrong amount without
    /// changing anything about the shape of the result. So each step is checked against the
    /// sigmas it claims to have walked, over a production-shaped schedule.
    #[test]
    fn each_step_moves_the_sample_by_the_sigma_it_crossed() {
        let num_steps = 9;
        let mut sched = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig {
            num_train_timesteps: 1000,
            shift: 3.0,
            use_dynamic_shifting: true,
        });
        sched.set_timesteps(num_steps, Some(0.875));

        let dev = Device::Cpu;
        let n = 48usize;
        let mut sample: Vec<f32> = (0..n).map(|i| (i as f32 * 0.21).sin() * 1.5).collect();
        for step in 0..num_steps {
            let pred: Vec<f32> = (0..n)
                .map(|i| ((i + step * 7) as f32 * 0.13).cos() - 0.3)
                .collect();
            let before = sample.clone();
            let dt = (sched.sigmas[sched.step_index() + 1] - sched.sigmas[sched.step_index()])
                as f32;

            let out = sched
                .step(
                    &Tensor::from_vec(pred.clone(), (3, 16), &dev).unwrap(),
                    &Tensor::from_vec(sample.clone(), (3, 16), &dev).unwrap(),
                )
                .unwrap();
            sample = out.flatten_all().unwrap().to_vec1().unwrap();

            assert_eq!(sched.step_index(), step + 1, "step {step} did not advance");
            for (i, ((s, b), p)) in sample.iter().zip(&before).zip(&pred).enumerate() {
                let want = b + p * dt;
                assert!(
                    (s - want).abs() < 1e-6,
                    "step {step}, element {i}: {s} rather than {want}"
                );
            }
        }
        assert!(
            sched.current_sigma().abs() < 1e-12,
            "the ladder did not end at zero noise"
        );
    }
}

#[cfg(test)]
mod ladder_tests {
    use super::*;

    fn turbo() -> FlowMatchEulerDiscreteScheduler {
        FlowMatchEulerDiscreteScheduler::new(SchedulerConfig {
            num_train_timesteps: 1000,
            shift: 3.0,
            use_dynamic_shifting: true,
        })
    }

    /// Laying out the same schedule twice lays out the same schedule.
    ///
    /// It did not: the ends were read back out of `self.sigmas`, which the first call had
    /// already replaced with its own output - and every img2img request lays one out twice,
    /// once for the run and once for the longer run whose tail it keeps.
    #[test]
    fn laying_out_the_ladder_twice_gives_the_same_ladder() {
        for mu in [None, Some(0.875)] {
            let mut once = turbo();
            once.set_timesteps(9, mu);
            let mut twice = turbo();
            twice.set_timesteps(9, mu);
            twice.set_timesteps(9, mu);
            assert_eq!(once.sigmas, twice.sigmas, "mu={mu:?}: sigmas");
            assert_eq!(once.timesteps, twice.timesteps, "mu={mu:?}: timesteps");
        }
    }

    /// And the img2img schedule is the tail of the longer one, whichever way it was reached.
    #[test]
    fn the_img2img_ladder_does_not_depend_on_what_came_before() {
        let mut fresh = turbo();
        fresh.set_timesteps_denoised(9, Some(0.875), 0.6);
        let mut used = turbo();
        used.set_timesteps(9, Some(0.875));
        used.set_timesteps_denoised(9, Some(0.875), 0.6);
        assert_eq!(fresh.sigmas, used.sigmas, "sigmas");
        assert_eq!(fresh.timesteps, used.timesteps, "timesteps");
    }
}
