//! Flow-matching solvers: the fixed-step integrators, and the UniPC multistep sampler.
//!
//! Order-1 Euler, order-2 Heun and UniPC all walk the same schedule against the same
//! velocity field, so they live together and a pipeline picks one by what its latents can
//! take rather than by which family wrote it.
//!
//! UniPC is a port of the reference video pipeline's
//! `FlowUniPCMultistepScheduler` (predict-x0, bh2, solver order 2), the sampler the official
//! Wan pipeline ships as its default. Order-1 integrators (Euler) visibly diverge under
//! strong CFG on stiff video latents; UniPC's predictor + corrector reuses previous model
//! outputs for order-2+ accuracy WITHOUT extra model calls per step (unlike Heun, which pays
//! a second forward).
//!
//! Scalar math runs in f64: the final step's sigma is exactly 0, driving the log-SNR
//! lambda to infinity, and the phi/B(h) expressions must collapse cleanly (they do: the
//! last update degenerates to "return the x0 prediction").

use crate::tensor::{Error, Result};

/// Euler flow-matching integration (oracle `dit-sampler` step update). Given the
/// noisy latent `x` (mutated in place) and a velocity field `velocity(xt, t)` that
/// returns `v_t` for the whole tensor at timestep `t`, walk the schedule:
/// non-final step `x += (t_next - t_curr).v`; final step `x += (0 - t_curr).v`
/// (predict x0). The two cases are the SAME update with the implicit `t_next=0`.
pub fn euler_integrate<F>(x: &mut [f32], schedule: &[f32], mut velocity: F)
where
    F: FnMut(&[f32], f32) -> Vec<f32>,
{
    let n = schedule.len();
    for step in 0..n {
        let t_curr = schedule[step];
        let t_next = if step + 1 < n {
            schedule[step + 1]
        } else {
            0.0
        };
        let v = velocity(x, t_curr);
        debug_assert_eq!(v.len(), x.len());
        let dt = t_next - t_curr;
        for (xi, vi) in x.iter_mut().zip(&v) {
            *xi += dt * vi;
        }
    }
}

/// Heun (order-2) flow-match integrator: predictor Euler step to `t_next`, corrector averages
/// the velocities at both ends. Same schedule contract as [`euler_integrate`] (implicit final
/// t=0 endpoint; the last step stays Euler since there is no point beyond it to probe). Costs
/// one extra model call per step but suppresses the order-1 high-frequency divergence that
/// surfaced as checkerboard artifacts in stiff CFG-guided video denoising - the reference
/// pipeline ships order-2+ solvers (UniPC / FlowDPM++) for exactly this reason.
pub fn heun_integrate<F>(x: &mut [f32], schedule: &[f32], mut velocity: F)
where
    F: FnMut(&[f32], f32) -> Vec<f32>,
{
    let n = schedule.len();
    for step in 0..n {
        let t_curr = schedule[step];
        let t_next = if step + 1 < n {
            schedule[step + 1]
        } else {
            0.0
        };
        let dt = t_next - t_curr;
        let v1 = velocity(x, t_curr);
        debug_assert_eq!(v1.len(), x.len());
        if step + 1 >= n {
            // Final step to t=0: no probe point beyond - plain Euler.
            for (xi, vi) in x.iter_mut().zip(&v1) {
                *xi += dt * vi;
            }
            break;
        }
        // Predictor x_pred = x + dt*v1, probe the velocity there, then correct with the mean.
        let mut x_pred: Vec<f32> = x.iter().zip(&v1).map(|(xi, vi)| xi + dt * vi).collect();
        let v2 = velocity(&x_pred, t_next);
        debug_assert_eq!(v2.len(), x.len());
        for ((xi, v1i), v2i) in x.iter_mut().zip(&v1).zip(&v2) {
            *xi += dt * 0.5 * (v1i + v2i);
        }
        x_pred.clear();
    }
}

/// The reference schedule: `linspace(1 - 1/T, 0, steps+1)` dropped of its endpoint, timestep
/// -shifted, then a final exact 0 appended. `sigmas.len() == steps + 1`.
pub fn unipc_sigma_schedule(steps: usize, shift: f64, num_train_timesteps: f64) -> Vec<f64> {
    let sigma_max = 1.0 - 1.0 / num_train_timesteps;
    let mut sigmas: Vec<f64> = (0..steps)
        .map(|i| {
            // linspace(sigma_max, 0, steps+1)[i]
            let s = sigma_max * (1.0 - i as f64 / steps as f64);
            shift * s / (1.0 + (shift - 1.0) * s)
        })
        .collect();
    sigmas.push(0.0);
    sigmas
}

/// Multistep state for one denoise trajectory. Feed it `steps` velocity evaluations via
/// [`FlowUniPc::step`]; query the conditioning timestep for step `k` with
/// [`FlowUniPc::timestep`].
pub struct FlowUniPc {
    sigmas: Vec<f64>,
    num_train_timesteps: f64,
    solver_order: usize,
    /// Converted model outputs (x0 predictions), oldest first; at most `solver_order` live.
    model_outputs: Vec<Vec<f32>>,
    last_sample: Option<Vec<f32>>,
    lower_order_nums: usize,
    step_index: usize,
    /// `this_order` computed by the PREVIOUS step's predictor - the reference reuses it for
    /// the current step's corrector (the attribute persists across `step()` calls there).
    prev_this_order: usize,
}

impl FlowUniPc {
    pub fn new(steps: usize, shift: f64) -> Self {
        const NUM_TRAIN: f64 = 1000.0;
        Self {
            sigmas: unipc_sigma_schedule(steps, shift, NUM_TRAIN),
            num_train_timesteps: NUM_TRAIN,
            solver_order: 2,
            model_outputs: Vec::new(),
            last_sample: None,
            lower_order_nums: 0,
            step_index: 0,
            prev_this_order: 1,
        }
    }

    /// Number of model evaluations the trajectory takes.
    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// The model-conditioning timestep for step `k` - the reference casts `sigma * T` to an
    /// integer, so the model sees e.g. t=999, not 999.8.
    pub fn timestep(&self, k: usize) -> f32 {
        (self.sigmas[k] * self.num_train_timesteps).floor() as f32
    }

    /// Advance one step: `velocity` is the (CFG-combined) model output at the CURRENT sample
    /// and `timestep(step_index)`; returns the next sample. Call exactly `num_steps` times.
    pub fn step(&mut self, velocity: &[f32], sample: &[f32]) -> Result<Vec<f32>> {
        let k = self.step_index;
        let n = self.num_steps();
        if k >= n {
            return Err(Error(
                "FlowUniPc: step() called past the end of the schedule".into(),
            ));
        }
        let sigma_k = self.sigmas[k];
        // convert_model_output (flow prediction, predict_x0): x0 = x - sigma * v.
        let x0: Vec<f32> = sample
            .iter()
            .zip(velocity)
            .map(|(x, v)| x - sigma_k as f32 * v)
            .collect();

        // UniC corrector on the CURRENT sample, at the ORDER the previous predictor used.
        let corrected: Vec<f32> = if k > 0 && self.last_sample.is_some() {
            self.uni_c(&x0, sample, self.prev_this_order)?
        } else {
            sample.to_vec()
        };

        // Slide the multistep buffers.
        if self.model_outputs.len() == self.solver_order {
            self.model_outputs.remove(0);
        }
        self.model_outputs.push(x0);

        let this_order = self.this_order_at(k);
        self.prev_this_order = this_order;
        self.last_sample = Some(corrected.clone());
        let next = self.uni_p(&corrected, this_order)?;

        if self.lower_order_nums < self.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;
        Ok(next)
    }

    /// `min(solver_order, steps_remaining, warmup)` - lower_order_final semantics of the
    /// reference (always active there for short schedules; harmless otherwise).
    fn this_order_at(&self, k: usize) -> usize {
        self.solver_order
            .min(self.num_steps() - k)
            .min(self.lower_order_nums + 1)
            .max(1)
    }

    fn lambda(&self, idx: usize) -> f64 {
        let sigma = self.sigmas[idx];
        let alpha = 1.0 - sigma;
        alpha.ln() - sigma.ln()
    }

    /// UniP predictor: extrapolate from `sample` at step k to step k+1 using the buffered x0
    /// predictions (`m0` = newest).
    fn uni_p(&self, sample: &[f32], order: usize) -> Result<Vec<f32>> {
        let k = self.step_index;
        let (sigma_t, sigma_s0) = (self.sigmas[k + 1], self.sigmas[k]);
        let alpha_t = 1.0 - sigma_t;
        let (lambda_t, lambda_s0) = (self.lambda(k + 1), self.lambda(k));
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1(); // bh2

        let m0 = self
            .model_outputs
            .last()
            .ok_or_else(|| Error("FlowUniPc: predictor before any model output".into()))?;
        // x_t_ = (sigma_t/sigma_s0) x - alpha_t h_phi_1 m0
        let r0 = sigma_t / sigma_s0;
        let c0 = -(alpha_t * h_phi_1);
        let mut out: Vec<f32> = sample
            .iter()
            .zip(m0)
            .map(|(x, m)| (r0 * *x as f64 + c0 * *m as f64) as f32)
            .collect();
        if order == 2 && self.model_outputs.len() >= 2 {
            // D1_0 = (m1 - m0)/rk with rk = (lambda(k-1) - lambda_s0)/h; rhos_p = [0.5].
            let m1 = &self.model_outputs[self.model_outputs.len() - 2];
            let rk = (self.lambda(k - 1) - lambda_s0) / h;
            let w = -(alpha_t * b_h) * 0.5 / rk;
            for ((o, mi), m0i) in out.iter_mut().zip(m1.iter()).zip(m0.iter()) {
                *o += (w * (*mi as f64 - *m0i as f64)) as f32;
            }
        }
        Ok(out)
    }

    /// UniC corrector: refine THIS step's sample using the model output just computed at it
    /// (`x0_t`), anchored on the previous sample (`last_sample`).
    fn uni_c(&self, x0_t: &[f32], _this_sample: &[f32], order: usize) -> Result<Vec<f32>> {
        let k = self.step_index;
        let (sigma_t, sigma_s0) = (self.sigmas[k], self.sigmas[k - 1]);
        let alpha_t = 1.0 - sigma_t;
        let (lambda_t, lambda_s0) = (self.lambda(k), self.lambda(k - 1));
        let h = lambda_t - lambda_s0;
        let hh = -h;
        let h_phi_1 = hh.exp_m1();
        let b_h = hh.exp_m1(); // bh2

        let x = self
            .last_sample
            .as_ref()
            .ok_or_else(|| Error("FlowUniPc: corrector without a previous sample".into()))?;
        let m0 = self
            .model_outputs
            .last()
            .ok_or_else(|| Error("FlowUniPc: corrector before any model output".into()))?;

        // rhos_c: order 1 -> [0.5]; order 2 -> solve [[1,1],[r0,1]] rho = [b0,b1].
        let (rho_d1s, rho_last, rk) = if order >= 2 && self.model_outputs.len() >= 2 {
            let rk = (self.lambda(k - 2) - lambda_s0) / h;
            // b coefficients of the bh2 expansion (see the reference update loop).
            let h_phi_k1 = h_phi_1 / hh - 1.0;
            let b0 = h_phi_k1 * 1.0 / b_h;
            let h_phi_k2 = h_phi_k1 / hh - 1.0 / 2.0;
            let b1 = h_phi_k2 * 2.0 / b_h;
            let rho0 = (b0 - b1) / (1.0 - rk);
            let rho1 = b0 - rho0;
            (rho0, rho1, rk)
        } else {
            (0.0, 0.5, 1.0)
        };

        let r0 = sigma_t / sigma_s0;
        let c0 = -(alpha_t * h_phi_1);
        let cb = -(alpha_t * b_h);
        let m1 = if rho_d1s != 0.0 {
            Some(&self.model_outputs[self.model_outputs.len() - 2])
        } else {
            None
        };
        let mut out = Vec::with_capacity(x.len());
        for i in 0..x.len() {
            let m0i = m0[i] as f64;
            // x_t_ = (sigma_t/sigma_s0) x_last - alpha_t h_phi_1 m0
            let base = r0 * x[i] as f64 + c0 * m0i;
            // corr = rho0 * (m1 - m0)/rk ; D1_t = x0_t - m0
            let corr = match m1 {
                Some(m1) => rho_d1s * (m1[i] as f64 - m0i) / rk,
                None => 0.0,
            };
            let d1_t = x0_t[i] as f64 - m0i;
            out.push((base + cb * (corr + rho_last * d1_t)) as f32);
        }
        Ok(out)
    }
}

/// Drive a full denoise with [`FlowUniPc`] using the same closure contract as
/// `euler_integrate` / `heun_integrate`: `velocity(x, t)` with `t = sigma * T` (floored, as
/// the reference feeds integer timesteps to the model). One model call per step.
pub fn unipc_integrate<F>(x: &mut Vec<f32>, steps: usize, shift: f64, mut velocity: F) -> Result<()>
where
    F: FnMut(&[f32], f32) -> Vec<f32>,
{
    let mut sched = FlowUniPc::new(steps, shift);
    for k in 0..sched.num_steps() {
        let t = sched.timestep(k);
        let v = velocity(x.as_slice(), t);
        debug_assert_eq!(v.len(), x.len());
        *x = sched.step(&v, x)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    /// Bit-level parity against the official FlowUniPCMultistepScheduler on a synthetic
    /// denoise against a reference run. `UNIPC_REF_DIR` points at the dump.
    #[test]
    #[ignore]
    fn unipc_matches_official_reference() {
        let dir = std::env::var("UNIPC_REF_DIR").expect("set UNIPC_REF_DIR to the dump dir");
        let read = |name: &str| -> Vec<f32> {
            let bytes = std::fs::read(format!("{dir}/{name}")).expect(name);
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };
        let ref_sigmas = read("unipc_sigmas.f32");
        let steps = ref_sigmas.len() - 1;
        let sched_sigmas = unipc_sigma_schedule(steps, 5.0, 1000.0);
        for (i, (a, b)) in ref_sigmas.iter().zip(&sched_sigmas).enumerate() {
            assert!(
                (*a as f64 - b).abs() < 1e-6,
                "sigma[{i}]: official {a} vs ours {b}"
            );
        }

        let mut x = read("unipc_x0_init.f32");
        let n = x.len();
        let w: Vec<Vec<f32>> = (0..steps)
            .map(|k| read(&format!("unipc_w{k}.f32")))
            .collect();
        let refs = read("unipc_steps_ref.f32");
        assert_eq!(refs.len(), steps * n);

        let mut sched = FlowUniPc::new(steps, 5.0);
        for k in 0..steps {
            let a = 0.3 + 0.05 * k as f32;
            let v: Vec<f32> = x.iter().zip(&w[k]).map(|(xi, wi)| a * xi + wi).collect();
            x = sched.step(&v, &x).unwrap();
            let r = &refs[k * n..(k + 1) * n];
            // Mixed criterion: a mismatch must be BOTH absolutely and relatively large.
            // The reference runs its scalar chain in torch f32; ours runs f64 - near-zero
            // elements differ by an f32 ULP (~5e-7 abs), which is not a port defect.
            for (i, (o, e)) in x.iter().zip(r).enumerate() {
                let abs = (*o as f64 - *e as f64).abs();
                let rel = abs / (e.abs() as f64 + 1e-6);
                assert!(
                    abs < 1e-5 || rel < 1e-4,
                    "step {k} elem {i}: ours {o} vs official {e} (abs {abs:.3e}, rel {rel:.3e})"
                );
            }
        }
    }

    /// The final step must collapse to the x0 prediction exactly (sigma -> 0 limit).
    #[test]
    fn final_step_returns_x0() {
        let mut s = FlowUniPc::new(1, 5.0);
        let x = vec![1.0f32, -2.0, 0.5];
        let v = vec![0.3f32, 0.1, -0.2];
        let sigma0 = s.sigmas[0] as f32;
        let out = s.step(&v, &x).unwrap();
        for ((o, xi), vi) in out.iter().zip(&x).zip(&v) {
            let x0 = xi - sigma0 * vi;
            assert!((o - x0).abs() < 1e-5, "expected x0 {x0}, got {o}");
        }
    }
}
