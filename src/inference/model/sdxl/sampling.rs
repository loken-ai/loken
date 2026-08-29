//! SDXL sampling: the discrete epsilon-prediction schedule and a Euler sampler.
//!
//! Every other image engine in this repo is FLOW-MATCHING (the model predicts a
//! velocity and the sampler integrates it directly). SDXL is the older formulation:
//! the model predicts the NOISE at a discrete timestep drawn from a beta schedule,
//! so this module carries its own schedule, its own sigma<->timestep mapping and its
//! own input scaling. Mixing the two conventions produces a plausible-looking image
//! that ignores the prompt, which is why the formulas below are transcribed from their
//! published statements (Karras et al. 2022; DPM-Solver++) rather than
//! recalled:
//!
//! - `scaled_linear` betas: `linspace(sqrt(1e-4 * 8.5), sqrt(0.012), 1000)^2`;
//! - `sigma(t) = sqrt((1 - alphas_cumprod(t)) / alphas_cumprod(t))`;
//! - the model sees `x / sqrt(sigma^2 + 1)`, never `x`;
//! - `denoised = x - eps * sigma`, and Euler steps on `d = (x - denoised) / sigma`;
//! - the timestep fed to the UNet is the index whose `log sigma` is closest.

/// Training timesteps in the schedule.
pub const TRAIN_STEPS: usize = 1000;
/// SDXL's `scaled_linear` beta endpoints.
const LINEAR_START: f64 = 0.00085;
const LINEAR_END: f64 = 0.012;

/// The discrete noise schedule: `sigmas[i]` for `i` in `0..TRAIN_STEPS`, increasing.
pub struct Schedule {
    sigmas: Vec<f32>,
    log_sigmas: Vec<f32>,
}

impl Default for Schedule {
    fn default() -> Self {
        Self::new()
    }
}

impl Schedule {
    pub fn new() -> Self {
        let n = TRAIN_STEPS;
        let (a, b) = (LINEAR_START.sqrt(), LINEAR_END.sqrt());
        let mut acp = 1.0f64;
        let mut sigmas = Vec::with_capacity(n);
        for i in 0..n {
            // scaled_linear: the betas are linear in SQRT space, then squared.
            let beta = (a + (b - a) * i as f64 / (n - 1) as f64).powi(2);
            acp *= 1.0 - beta;
            sigmas.push((((1.0 - acp) / acp).sqrt()) as f32);
        }
        let log_sigmas = sigmas.iter().map(|s| s.ln()).collect();
        Self { sigmas, log_sigmas }
    }

    pub fn sigma_min(&self) -> f32 {
        self.sigmas[0]
    }

    pub fn sigma_max(&self) -> f32 {
        self.sigmas[self.sigmas.len() - 1]
    }

    /// Interpolated sigma at a (possibly fractional) timestep - the reference
    /// interpolates in LOG space, which matters at the low-sigma end.
    pub fn sigma_at(&self, t: f32) -> f32 {
        let t = t.clamp(0.0, (self.sigmas.len() - 1) as f32);
        let lo = t.floor() as usize;
        let hi = t.ceil() as usize;
        let w = t - t.floor();
        ((1.0 - w) * self.log_sigmas[lo] + w * self.log_sigmas[hi]).exp()
    }

    /// The timestep the UNet is conditioned on for a given sigma: the index whose
    /// log-sigma is closest. Feeding the raw sigma instead silently mis-conditions
    /// every step.
    pub fn timestep_for(&self, sigma: f32) -> f32 {
        let ls = sigma.ln();
        let mut best = 0usize;
        let mut best_d = f32::MAX;
        for (i, l) in self.log_sigmas.iter().enumerate() {
            let d = (ls - l).abs();
            if d < best_d {
                best_d = d;
                best = i;
            }
        }
        best as f32
    }

    /// The "normal" sigma ladder: `steps` sigmas spread evenly over the TIMESTEP
    /// axis from the top of the schedule down, with a trailing zero.
    pub fn sigma_ladder(&self, steps: usize) -> Vec<f32> {
        let steps = steps.max(1);
        let start = self.timestep_for(self.sigma_max());
        let end = self.timestep_for(self.sigma_min());
        let mut out = Vec::with_capacity(steps + 1);
        for i in 0..steps {
            let t = start + (end - start) * i as f32 / (steps - 1).max(1) as f32;
            out.push(self.sigma_at(t));
        }
        out.push(0.0);
        out
    }
}

/// Which sigma curve the steps are placed on.
///
/// The schedule decides WHERE the budget is spent, and the samplers a checkpoint was
/// tuned against assume a particular one - which is why "25 steps" from a community
/// workflow does not mean the same thing on an evenly-spaced ladder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulerKind {
    /// Even spacing in timestep space - what this pipeline shipped with.
    #[default]
    Normal,
    /// Karras et al. (2022): bottom-weighted, the SDXL ecosystem default.
    Karras,
    /// Geometric spacing between sigma_max and sigma_min.
    Exponential,
}

/// Which solver advances the latent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SamplerKind {
    /// First order. Robust, and what this pipeline shipped with.
    #[default]
    Euler,
    /// DPM-Solver++(2M): second order at the SAME cost per step, because the extra
    /// order comes from reusing the previous step's denoised estimate rather than
    /// from another model call. Converges in fewer steps, which is why SDXL
    /// checkpoints quote step counts against it.
    DpmPP2M,
}

impl SchedulerKind {
    /// Parse the sampler name as the ecosystem spells it; unknown names fall back to the
    /// default rather than failing a render.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "karras" => Self::Karras,
            "exponential" => Self::Exponential,
            _ => Self::Normal,
        }
    }
}

impl SamplerKind {
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "dpmpp_2m" | "dpmpp2m" | "dpm++2m" => Self::DpmPP2M,
            _ => Self::Euler,
        }
    }
}

/// Karras et al. (2022) noise schedule - crowsonkb's `k_diffusion/sampling.py` (MIT):
///
/// ```text
///     ramp = linspace(0, 1, n)
///     sigmas = (max^(1/rho) + ramp * (min^(1/rho) - max^(1/rho)))^rho, then append 0
/// ```
///
/// Spends more of the budget at low sigma, where the image is decided, instead of the
/// even spacing the plain ladder uses. It is the default the SDXL ecosystem tunes
/// against, so a checkpoint's recommended step counts assume it.
pub fn karras_sigmas(steps: usize, sigma_min: f32, sigma_max: f32, rho: f32) -> Vec<f32> {
    let n = steps.max(1);
    let min_inv = sigma_min.powf(1.0 / rho);
    let max_inv = sigma_max.powf(1.0 / rho);
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        // linspace(0,1,n) is n points INCLUDING both ends; with n == 1 it is just 0.
        let ramp = if n == 1 {
            0.0
        } else {
            i as f32 / (n - 1) as f32
        };
        out.push((max_inv + ramp * (min_inv - max_inv)).powf(rho));
    }
    out.push(0.0);
    out
}

/// Exponential schedule: `linspace(ln(max), ln(min), n).exp()`, then 0.
pub fn exponential_sigmas(steps: usize, sigma_min: f32, sigma_max: f32) -> Vec<f32> {
    let n = steps.max(1);
    let (a, b) = (sigma_max.ln(), sigma_min.ln());
    let mut out = Vec::with_capacity(n + 1);
    for i in 0..n {
        let t = if n == 1 {
            0.0
        } else {
            i as f32 / (n - 1) as f32
        };
        out.push((a + (b - a) * t).exp());
    }
    out.push(0.0);
    out
}

/// State carried between steps of DPM-Solver++(2M).
///
/// The method is second order by REUSING the previous step's denoised estimate rather
/// than taking an extra model call, so it costs the same as Euler per step and converges
/// in noticeably fewer of them. That is why it, not Euler, is what SDXL checkpoints are
/// tuned against.
#[derive(Default)]
pub struct DpmPP2M {
    old_denoised: Option<Vec<f32>>,
}

impl DpmPP2M {
    pub fn new() -> Self {
        Self::default()
    }

    /// One DPM-Solver++(2M) step - `k_diffusion/sampling.py::sample_dpmpp_2m`:
    ///
    /// ```text
    ///     t, t_next = -ln(sigma), -ln(sigma_next);  h = t_next - t
    ///     first step or sigma_next == 0:
    ///         x = (sigma_next/sigma) * x - expm1(-h) * denoised
    ///     otherwise, with r = h_last/h:
    ///         d = (1 + 1/(2r)) * denoised - (1/(2r)) * old_denoised
    ///         x = (sigma_next/sigma) * x - expm1(-h) * d
    /// ```
    ///
    /// `sigma_prev` is the sigma of the step BEFORE this one, used only for `h_last`.
    pub fn step(
        &mut self,
        x: &[f32],
        denoised: &[f32],
        sigma_prev: Option<f32>,
        sigma: f32,
        sigma_next: f32,
    ) -> Vec<f32> {
        let t = -sigma.ln();
        let t_next = -sigma_next.max(f32::MIN_POSITIVE).ln();
        let h = t_next - t;
        let ratio = sigma_next / sigma;
        // expm1(-h) = e^-h - 1, kept as expm1 for accuracy at small h.
        let em1 = (-h).exp_m1();
        let out: Vec<f32> = match (&self.old_denoised, sigma_prev) {
            // The last step lands on sigma 0: the second-order correction is skipped
            // there, exactly as the reference does, because t_next is unbounded.
            (Some(old), Some(sp)) if sigma_next > 0.0 => {
                let h_last = t - (-sp.ln());
                let r = h_last / h;
                let a = 1.0 + 1.0 / (2.0 * r);
                let b = 1.0 / (2.0 * r);
                x.iter()
                    .zip(denoised)
                    .zip(old)
                    .map(|((xi, d), o)| ratio * xi - em1 * (a * d - b * o))
                    .collect()
            }
            _ => x
                .iter()
                .zip(denoised)
                .map(|(xi, d)| ratio * xi - em1 * d)
                .collect(),
        };
        self.old_denoised = Some(denoised.to_vec());
        out
    }
}

/// What the model is fed and what its output means, for the epsilon parameterisation.
///
/// `sigma_data` is 1 for SD-family models, so the input scaling is `1/sqrt(s^2+1)`.
pub fn input_scale(sigma: f32) -> f32 {
    1.0 / (sigma * sigma + 1.0).sqrt()
}

/// `denoised = x - eps * sigma`.
pub fn denoised(x: &[f32], eps: &[f32], sigma: f32) -> Vec<f32> {
    x.iter().zip(eps).map(|(a, e)| a - e * sigma).collect()
}

/// One Euler step: `x + (x - denoised)/sigma * (sigma_next - sigma)`.
///
/// Written over slices so the scheduler is testable without a device.
pub fn euler_step(x: &[f32], denoised: &[f32], sigma: f32, sigma_next: f32) -> Vec<f32> {
    let dt = sigma_next - sigma;
    x.iter()
        .zip(denoised)
        .map(|(a, d)| {
            let deriv = (a - d) / sigma;
            a + deriv * dt
        })
        .collect()
}

/// Classifier-free guidance over two epsilon predictions.
pub fn cfg(eps_cond: &[f32], eps_uncond: &[f32], scale: f32) -> Vec<f32> {
    eps_uncond
        .iter()
        .zip(eps_cond)
        .map(|(u, c)| u + scale * (c - u))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pinned against the published formula, evaluated independently at the SDXL sigma
    /// range. A schedule that merely "looks like" Karras is worth nothing: the whole
    /// point is that checkpoints are tuned against THIS curve, so a step count borrowed
    /// from a community workflow means what it says.
    #[test]
    fn karras_matches_the_reference_curve() {
        let got = karras_sigmas(10, 0.0292, 14.6146, 7.0);
        let want = [
            14.614600, 9.103405, 5.479027, 3.169204, 1.749901, 0.914420, 0.447145, 0.201532,
            0.081981, 0.029200, 0.0,
        ];
        assert_eq!(got.len(), want.len(), "must append the terminal zero");
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.max(1e-3),
                "karras[{i}] = {g}, reference {w}"
            );
        }
        // It has to descend, and it must spend more of its budget at low sigma than an
        // even split would - that IS the schedule's reason to exist.
        for w in got.windows(2) {
            assert!(w[1] < w[0], "karras must descend");
        }
        let mid = got[got.len() / 2];
        assert!(
            mid < 14.6146 / 2.0,
            "karras must be bottom-weighted, mid was {mid}"
        );
    }

    #[test]
    fn exponential_matches_the_reference_curve() {
        let got = exponential_sigmas(6, 0.0292, 14.6146);
        let want = [
            14.614600, 4.216054, 1.216257, 0.350869, 0.101219, 0.029200, 0.0,
        ];
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-4 * w.max(1e-3),
                "exponential[{i}] = {g}, reference {w}"
            );
        }
    }

    /// DPM-Solver++(2M) against the reference update, including the detail that decides
    /// whether it is second order at all: the last step, which lands on sigma 0, drops
    /// the correction term.
    #[test]
    fn dpmpp_2m_matches_the_reference_update() {
        let sigmas = [10.0f32, 5.0, 2.0, 0.0];
        let denoised = [0.5f32, 0.4, 0.3];
        let mut solver = DpmPP2M::new();
        let mut x = vec![1.0f32];
        for i in 0..sigmas.len() - 1 {
            let prev = if i == 0 { None } else { Some(sigmas[i - 1]) };
            x = solver.step(&x, &[denoised[i]], prev, sigmas[i], sigmas[i + 1]);
        }
        // Landing on sigma 0 means the result IS the last denoised estimate.
        assert!((x[0] - 0.300000).abs() < 1e-5, "dpmpp_2m gave {}", x[0]);
    }

    /// The second-order term must actually engage on a middle step - otherwise this is
    /// Euler wearing a different name, which is exactly the bug that would go unnoticed.
    #[test]
    fn dpmpp_2m_is_second_order_after_the_first_step() {
        let (s0, s1, s2) = (10.0f32, 5.0, 2.0);
        let x0 = vec![1.0f32];
        // Same inputs, differing only in whether history exists.
        let mut with_hist = DpmPP2M::new();
        let a = with_hist.step(&x0, &[0.5], None, s0, s1);
        let b = with_hist.step(&a, &[0.4], Some(s0), s1, s2);
        let mut no_hist = DpmPP2M::new();
        let c = no_hist.step(&a, &[0.4], None, s1, s2);
        assert!(
            (b[0] - c[0]).abs() > 1e-6,
            "the history term changed nothing ({} vs {}) - this is first order",
            b[0],
            c[0]
        );
    }

    /// The schedule's endpoints are the well-known SD values; a wrong beta
    /// convention (linear instead of scaled_linear) moves them a long way.
    #[test]
    fn the_schedule_spans_the_expected_sigma_range() {
        let s = Schedule::new();
        assert!(
            (s.sigma_min() - 0.0292).abs() < 0.002,
            "sigma_min {} is off",
            s.sigma_min()
        );
        assert!(
            (s.sigma_max() - 14.6).abs() < 0.5,
            "sigma_max {} is off",
            s.sigma_max()
        );
    }

    /// Sigmas increase with the timestep, so the ladder must DESCEND and end at 0.
    #[test]
    fn the_ladder_descends_to_zero() {
        let s = Schedule::new();
        let l = s.sigma_ladder(20);
        assert_eq!(l.len(), 21, "steps + the trailing zero");
        assert!((l[0] - s.sigma_max()).abs() < 1e-3, "starts at sigma_max");
        assert_eq!(*l.last().unwrap(), 0.0, "ends at zero");
        for w in l.windows(2) {
            assert!(w[0] >= w[1], "the ladder must not go back up: {w:?}");
        }
    }

    /// sigma -> timestep -> sigma must round-trip on the schedule's own points.
    #[test]
    fn sigma_and_timestep_round_trip() {
        let s = Schedule::new();
        for t in [0usize, 1, 250, 500, 999] {
            let sigma = s.sigma_at(t as f32);
            let back = s.timestep_for(sigma);
            assert!(
                (back - t as f32).abs() < 1.0,
                "t {t} -> sigma {sigma} -> {back}"
            );
        }
    }

    /// The input scaling is what makes the epsilon parameterisation work; at
    /// sigma = 0 it is 1 (the image passes through) and it decays as noise grows.
    #[test]
    fn the_input_scaling_decays_with_sigma() {
        assert!((input_scale(0.0) - 1.0).abs() < 1e-6);
        assert!(input_scale(14.6) < 0.07);
        assert!(input_scale(1.0) > input_scale(2.0));
    }

    /// A perfect prediction (eps == the noise actually present) must denoise to the
    /// clean signal, and the Euler step must then leave it alone.
    #[test]
    fn a_perfect_prediction_recovers_the_signal() {
        let clean = [0.5f32, -0.25, 1.0];
        let noise = [1.0f32, -2.0, 0.5];
        let sigma = 3.0f32;
        let x: Vec<f32> = clean
            .iter()
            .zip(noise)
            .map(|(c, n)| c + sigma * n)
            .collect();
        let d = denoised(&x, &noise, sigma);
        for (a, b) in d.iter().zip(clean.iter()) {
            assert!((a - b).abs() < 1e-5, "denoised {a} vs clean {b}");
        }
        // Stepping to sigma = 0 lands exactly on the denoised signal.
        let stepped = euler_step(&x, &d, sigma, 0.0);
        for (a, b) in stepped.iter().zip(clean.iter()) {
            assert!((a - b).abs() < 1e-4, "stepped {a} vs clean {b}");
        }
    }

    /// Guidance of 1 is a no-op; higher values push away from the unconditional.
    #[test]
    fn guidance_interpolates_from_the_unconditional() {
        let c = [1.0f32, 2.0];
        let u = [0.0f32, 1.0];
        assert_eq!(cfg(&c, &u, 1.0), vec![1.0, 2.0]);
        assert_eq!(cfg(&c, &u, 0.0), vec![0.0, 1.0]);
        assert_eq!(cfg(&c, &u, 2.0), vec![2.0, 3.0]);
    }
}
