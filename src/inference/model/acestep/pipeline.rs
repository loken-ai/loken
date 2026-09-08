//! ACE-Step 1.5 - M1 pipeline assembly. Wires the oracle-validated
//! components into the DiT-passthrough text2music path (audio_codes provided):
//!   codes -> FSQ detok -> context[T,128] = [detok(64) | mask=1.0(64)]
//!   text_ids -> text encoder -> text_hidden ; lyric_ids -> embed lookup -> lyric_embed
//!   (text_hidden, lyric_embed, timbre) -> cond encoder -> enc_hidden
//!   enc = DiT.condition_embedder(enc_hidden)   (= enc_after_cond_emb)
//!   noise[T,64] -> 8x Euler(velocity_forward(concat(context,xt), enc)) -> latent
//!   latent -> Oobleck VAE -> WAV
//! Every link is validated individually; this module is the glue + layout joins.

use crate::inference::model::acestep::dit::{turbo_schedule, DitModel};
use crate::tensor::Tensor;

/// context[T,128] (token-major, `flat[t.128+c]`) from detok latents `[T.64]`
/// (`flat[t.64+c]`): channels 0..64 = detok (or `silence` beyond `decoded_t`),
/// 64..128 = mask = 1.0 (training distribution). `silence` is the oracle's
/// `silence_full` (`[ (T-decoded_t).64 ]`); pass empty when `decoded_t==t`.
pub fn build_context(detok: &[f32], t: usize, decoded_t: usize, silence: &[f32]) -> Vec<f32> {
    let oc = 64usize;
    let mut ctx = vec![0f32; t * 128];
    for ti in 0..t {
        let src = if ti < decoded_t {
            &detok[ti * oc..]
        } else {
            &silence[(ti - decoded_t) * oc..]
        };
        for c in 0..oc {
            ctx[ti * 128 + c] = src[c];
        }
        for c in 0..oc {
            ctx[ti * 128 + oc + c] = 1.0;
        }
    }
    ctx
}

/// DiT input `[in_ch.T]` channel-major (`flat[c.T+t]`) = concat(context[T,128], xt[T,64]),
/// transposing both from token-major - the exact layout `velocity_forward` expects.
pub fn build_dit_input(context: &[f32], xt: &[f32], t: usize) -> Vec<f32> {
    let (c_ctx, c_xt) = (128usize, 64usize);
    let in_ch = c_ctx + c_xt;
    let mut input = vec![0f32; in_ch * t];
    for ti in 0..t {
        for c in 0..c_ctx {
            input[c * t + ti] = context[ti * c_ctx + c];
        }
        for c in 0..c_xt {
            input[(c_ctx + c) * t + ti] = xt[ti * c_xt + c];
        }
    }
    input
}

/// Apply the DiT condition embedder to `enc_hidden [S,2048]` -> `enc [S,2048]`
/// (= `enc_after_cond_emb`), the cross-attn source. `enc_hidden @ Wᵀ + b`.
pub fn cond_emb_apply(
    dit: &DitModel,
    enc_hidden: &[f32],
    s: usize,
) -> crate::tensor::Result<Vec<f32>> {
    // The condition_embedder maps the ENCODER hidden size (its weight in-dim) -> the DiT
    // hidden size (out-dim). On 2B models both are 2048; on XL the encoder stays 2048 but
    // the DiT hidden is 2560, so the input width must come from the weight, not dit.hidden.
    let in_dim = dit.cond_emb.w.dims()[1];
    // cond_emb weight is on the DiT's device (GPU); move the activation there too.
    let x = Tensor::from_vec_f32(enc_hidden.to_vec(), (s, in_dim))?.to_device(&dit.device)?;
    let y = x.matmul_t(&dit.cond_emb.w)?;
    let y = y.broadcast_add(dit.cond_emb.b.as_ref().unwrap())?;
    y.flatten_all()?.to_vec1_f32()
}

/// DiT classifier-free-guidance algorithm (ACE-Step `cfg_type`). `Apg` (the reference
/// default) decouples magnitude from direction so a strong scale steers timbre without
/// inflating level - the key to running the non-turbo checkpoints without saturation.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CfgType {
    Cfg,
    Apg,
    CfgStar,
}

/// Flow-matching ODE/SDE solver. `Euler` (1st order, 1 eval/step) is the validated default
/// and the only one that runs the hyper-optimized on-device trajectory. `Heun` (2nd order
/// predictor-corrector, 2 evals/step) trades compute for accuracy; `Pingpong` is a
/// stochastic SDE step (re-injects noise each step).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Solver {
    Euler,
    Heun,
    Pingpong,
}

/// DiT sampling + guidance parameters, threaded as one struct through both trajectory
/// paths. [`DitSample::turbo`] reproduces the validated 8-step no-guidance turbo run
/// byte-for-byte (shift 3.0, vanilla CFG off, neutral omega).
#[derive(Clone)]
pub struct DitSample {
    pub steps: usize,
    pub cfg: f32,
    /// Flow-matching timestep shift `t' = shift.t/(1+(shift-1).t)`. Turbo 3.0, base/sft 1.0.
    pub shift: f32,
    pub cfg_type: CfgType,
    /// Fraction of steps (centred window) where guidance is active; 1.0 = every step.
    pub guidance_interval: f32,
    /// >0 lerps the scale from `cfg`->`min_guidance_scale` across the window; 0 = constant.
    pub guidance_interval_decay: f32,
    pub min_guidance_scale: f32,
    /// cfg_star: zero the prediction for steps `i <= zero_steps`.
    pub zero_steps: usize,
    /// Scheduler mean-shift strength; 0 = neutral (rescale 1.0, no change).
    pub omega_scale: f32,
    /// ODE/SDE solver. Only `Euler` runs the on-device fast path.
    pub solver: Solver,
    /// RNG seed for the stochastic `Pingpong` solver (unused by Euler/Heun).
    pub seed: u64,
    /// Explicit sigma schedule (descending, excluding the implicit final 0). `None` =
    /// `turbo_schedule(steps, shift)`. A non-euler solver routes off the on-device path.
    pub custom_sched: Option<Vec<f32>>,
    /// repaint/inpaint: when set, the `keep` frames are forced back to the reference's noised
    /// trajectory each step (only the unmasked region is freely generated). Routes off the
    /// on-device path. `ref_latent` is channel-major `[64.t]`, `keep` is per-frame (len t).
    pub repaint: Option<RepaintMask>,
    /// dual-condition guidance: separate scales for the caption (text) vs the lyrics. When set,
    /// each step blends three velocities `(1-gt).uncond + (gt-gl).text_only + gl.cond`. Needs
    /// `uncond`. Routes off the on-device path.
    pub dual: Option<DualCond>,
    /// DCW wavelet-domain correction applied after each solver step. Routes off the on-device path.
    pub dcw: Option<Dcw>,
}

/// Text-only encoding + the two scales for dual-condition guidance (caption vs lyrics).
#[derive(Clone)]
pub struct DualCond {
    pub text_enc: Vec<f32>,
    pub text_s: usize,
    pub gs_text: f32,
    pub gs_lyric: f32,
}

/// Reference latent + per-frame keep mask for repaint/extend (inpaint/outpaint).
#[derive(Clone)]
pub struct RepaintMask {
    pub ref_latent: Vec<f32>,
    pub keep: Vec<bool>,
}

/// DCW (Differential Correction in Wavelet domain, CVPR 2026) band selector.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DcwMode {
    Low,
    High,
    Double,
    Pix,
}

/// DCW sampler-side correction params (per-step, after the solver step). `scaler`/`high_scaler`
/// 0 = off. Skipped for the stochastic Pingpong solver (it injects noise).
#[derive(Clone)]
pub struct Dcw {
    pub mode: DcwMode,
    pub scaler: f32,
    pub high_scaler: f32,
}

/// Forward Haar DWT along the time axis of `[T,C]` frame-major data -> low/high bands, each
/// `[Tl.C]` with `Tl=(T+1)/2` (odd T zero-pads the last high index). `1/√2` normalized.
fn haar_fwd(src: &[f32], t: usize, c: usize) -> (Vec<f32>, Vec<f32>) {
    let inv = std::f32::consts::FRAC_1_SQRT_2;
    let tl = t.div_ceil(2);
    let (mut lo, mut hi) = (vec![0f32; tl * c], vec![0f32; tl * c]);
    for til in 0..tl {
        let (i0, i1) = (2 * til, 2 * til + 1);
        for ci in 0..c {
            let a = src[i0 * c + ci];
            if i1 < t {
                let b = src[i1 * c + ci];
                lo[til * c + ci] = (a + b) * inv;
                hi[til * c + ci] = (a - b) * inv;
            } else {
                lo[til * c + ci] = a * inv;
                hi[til * c + ci] = a * inv;
            }
        }
    }
    (lo, hi)
}

/// Inverse Haar IDWT -> `out [T,C]` frame-major (the exact inverse of [`haar_fwd`]).
fn haar_inv(lo: &[f32], hi: &[f32], t: usize, c: usize, out: &mut [f32]) {
    let inv = std::f32::consts::FRAC_1_SQRT_2;
    let tl = t.div_ceil(2);
    for til in 0..tl {
        let (i0, i1) = (2 * til, 2 * til + 1);
        for ci in 0..c {
            let (lc, hc) = (lo[til * c + ci], hi[til * c + ci]);
            out[i0 * c + ci] = (lc + hc) * inv;
            if i1 < t {
                out[i1 * c + ci] = (lc - hc) * inv;
            }
        }
    }
}

/// Apply DCW to `x [T,C]` (token/frame-major) in place using `denoised` and the per-band
/// scalers (`s_low`, `s_high`). Ported from acestep.cpp dwt-haar.h.
fn dcw_apply(
    x: &mut [f32],
    denoised: &[f32],
    t: usize,
    c: usize,
    mode: DcwMode,
    s_low: f32,
    s_high: f32,
) {
    if mode == DcwMode::Pix {
        if s_low != 0.0 {
            for i in 0..t * c {
                x[i] += s_low * (x[i] - denoised[i]);
            }
        }
        return;
    }
    if s_low == 0.0 && s_high == 0.0 {
        return;
    }
    let (mut xl, mut xh) = haar_fwd(x, t, c);
    let (yl, yh) = haar_fwd(denoised, t, c);
    if matches!(mode, DcwMode::Low | DcwMode::Double) && s_low != 0.0 {
        for i in 0..xl.len() {
            xl[i] += s_low * (xl[i] - yl[i]);
        }
    }
    if matches!(mode, DcwMode::High | DcwMode::Double) && s_high != 0.0 {
        for i in 0..xh.len() {
            xh[i] += s_high * (xh[i] - yh[i]);
        }
    }
    haar_inv(&xl, &xh, t, c, x);
}

impl DitSample {
    /// Validated turbo defaults: `steps` Euler steps, shift 3.0, guidance off -> identical
    /// to the original sampler when `cfg <= 1.0`.
    pub fn turbo(steps: usize, cfg: f32) -> Self {
        DitSample {
            steps,
            cfg,
            shift: 3.0,
            cfg_type: CfgType::Cfg,
            guidance_interval: 1.0,
            guidance_interval_decay: 0.0,
            min_guidance_scale: cfg,
            zero_steps: 0,
            omega_scale: 0.0,
            solver: Solver::Euler,
            seed: 0,
            custom_sched: None,
            repaint: None,
            dual: None,
            dcw: None,
        }
    }
    /// The sigma schedule: an explicit `custom_sched` if set, else `turbo_schedule`.
    pub fn schedule(&self) -> Vec<f32> {
        self.custom_sched
            .clone()
            .unwrap_or_else(|| turbo_schedule(self.steps, self.shift))
    }
    /// Guidance window `[start, end)` in step index (reference `pipeline_ace_step`).
    fn guidance_window(&self) -> (usize, usize) {
        let n = self.steps as f32;
        (
            (n * ((1.0 - self.guidance_interval) / 2.0)) as usize,
            (n * (self.guidance_interval / 2.0 + 0.5)) as usize,
        )
    }
    /// Effective guidance scale at step `i`, or `None` when guidance is inactive this step
    /// (`cfg <= 1` or `i` outside the window) -> the cond-only prediction is used.
    pub fn scale_at(&self, i: usize) -> Option<f32> {
        if self.cfg <= 1.0 {
            return None;
        }
        let (start, end) = self.guidance_window();
        if i < start || i >= end {
            return None;
        }
        if self.guidance_interval_decay > 0.0 && end > start + 1 {
            let progress = (i - start) as f32 / (end - start - 1) as f32;
            Some(
                self.cfg
                    - (self.cfg - self.min_guidance_scale)
                        * progress
                        * self.guidance_interval_decay,
            )
        } else {
            Some(self.cfg)
        }
    }
}

/// Scheduler mean-shift rescale from `omega` (logistic L=0.9, U=1.1, k=0.1): `omega=0`
/// maps to exactly 1.0 (neutral, the turbo path), the reference default 10 to ≈1.046.
pub fn omega_rescale(omega: f32) -> f32 {
    0.9 + 0.2 * (1.0 / (1.0 + (-0.1 * omega).exp()))
}

/// Blend the conditional and unconditional velocities (channel-major `[oc.t]`, `v[c.t+ti]`)
/// per the selected `cfg_type`. `apg_avg` carries APG's momentum running-average across the
/// guided steps. Ported from ACE-Step `apg_guidance.py` / acestep.cpp `dit-sampler.h`
/// (per-channel L2 over time, f64 internal math); cfg_star uses a global optimal scale.
pub fn apply_guidance(
    vc: &[f32],
    vu: &[f32],
    oc: usize,
    t: usize,
    scale: f32,
    cfg_type: CfgType,
    step: usize,
    zero_steps: usize,
    apg_avg: &mut Option<Vec<f64>>,
) -> Vec<f32> {
    let n = oc * t;
    match cfg_type {
        CfgType::Cfg => vu
            .iter()
            .zip(vc)
            .map(|(u, c)| u + scale * (c - u))
            .collect(),
        CfgType::CfgStar => {
            if step <= zero_steps {
                return vec![0f32; n];
            }
            let (mut dot, mut sq) = (0f64, 0f64);
            for i in 0..n {
                dot += vc[i] as f64 * vu[i] as f64;
                sq += (vu[i] as f64) * (vu[i] as f64);
            }
            let alpha = (dot / (sq + 1e-8)) as f32;
            (0..n)
                .map(|i| vu[i] * alpha + scale * (vc[i] - vu[i] * alpha))
                .collect()
        }
        CfgType::Apg => {
            // diff = cond - uncond, smoothed by the momentum buffer (β = -0.75).
            let mut diff: Vec<f64> = (0..n).map(|i| vc[i] as f64 - vu[i] as f64).collect();
            match apg_avg {
                Some(avg) => {
                    for i in 0..n {
                        avg[i] = diff[i] + (-0.75) * avg[i];
                        diff[i] = avg[i];
                    }
                }
                None => *apg_avg = Some(diff.clone()),
            }
            // Per-channel L2 norm clip over time (channel-major -> contiguous blocks).
            const THR: f64 = 2.5;
            for c in 0..oc {
                let base = c * t;
                let mut norm2 = 0f64;
                for ti in 0..t {
                    norm2 += diff[base + ti] * diff[base + ti];
                }
                let norm = norm2.sqrt();
                if norm > 1e-60 {
                    let s = (THR / norm).min(1.0);
                    if s < 1.0 {
                        for ti in 0..t {
                            diff[base + ti] *= s;
                        }
                    }
                }
            }
            // Orthogonal projection of diff onto vc, per channel; out = vc + (scale-1).orth.
            let w = (scale - 1.0) as f64;
            let mut out = vec![0f32; n];
            for c in 0..oc {
                let base = c * t;
                let mut norm2 = 0f64;
                for ti in 0..t {
                    let v = vc[base + ti] as f64;
                    norm2 += v * v;
                }
                let inv = if norm2 > 1e-60 {
                    1.0 / norm2.sqrt()
                } else {
                    0.0
                };
                let mut dot = 0f64;
                for ti in 0..t {
                    dot += diff[base + ti] * (vc[base + ti] as f64 * inv);
                }
                for ti in 0..t {
                    let v1n = vc[base + ti] as f64 * inv;
                    let orth = diff[base + ti] - dot * v1n;
                    out[base + ti] = (vc[base + ti] as f64 + w * orth) as f32;
                }
            }
            out
        }
    }
}

/// Run the turbo Euler trajectory: `noise [T,64]` (token-major) + the fixed `context` +
/// `enc` -> latent `[T,64]` = the DiT output. Thin wrapper over [`dit_latent_cfg`] with the
/// validated turbo sampling params (8 steps, shift 3.0, no guidance).
pub fn dit_latent(
    dit: &DitModel,
    context: &[f32],
    enc: &[f32],
    enc_s: usize,
    noise: &[f32],
    t: usize,
) -> crate::tensor::Result<Vec<f32>> {
    dit_latent_cfg(
        dit,
        context,
        enc,
        enc_s,
        None,
        noise,
        t,
        &DitSample::turbo(8, 1.0),
        None,
    )
}

/// Generalized Euler trajectory with the full ACE-Step guidance stack (`sp`): per-checkpoint
/// shift, APG/cfg_star/vanilla CFG, a centred guidance window with optional decay, and
/// omega mean-shift. With `DitSample::turbo` and `cfg <= 1.0` it is byte-for-byte identical
/// to the original 8-step no-CFG sampler (shift 3.0, neutral omega, guidance never fires).
pub fn dit_latent_cfg(
    dit: &DitModel,
    context: &[f32],
    enc: &[f32],
    enc_s: usize,
    uncond: Option<(&[f32], usize)>,
    noise: &[f32],
    t: usize,
    sp: &DitSample,
    progress: Option<&crate::inference::serve::progress::ProgressTryFn<'_>>,
) -> crate::tensor::Result<Vec<f32>> {
    // Fast on-device trajectory: keeps the activation on the GPU across all blocks and
    // hoists every per-step-fixed tensor (enc upload, RoPE tables, SWA/morph masks, proj
    // weights) + the on-device AdaLN out of the per-block loop. Only when the whole DiT is
    // on one CUDA device; a HeteroPlan that spilled blocks keeps the eager path below.
    // Only Euler (no repaint) runs the hyper-optimized on-device trajectory (the validated
    // turbo path); Heun/Pingpong/repaint use this generalized loop (as does any spilled plan).
    if dit.all_on_primary()
        && sp.solver == Solver::Euler
        && sp.repaint.is_none()
        && sp.dual.is_none()
        && sp.dcw.is_none()
    {
        return dit.dit_trajectory_ondevice(context, enc, enc_s, uncond, noise, t, sp, progress);
    }
    let oc = 64usize;
    let use_cfg = sp.cfg > 1.0 && uncond.is_some();
    let sched = sp.schedule();
    let rescale = omega_rescale(sp.omega_scale);
    let noise0 = noise.to_vec(); // init noise, reused to re-noise the repaint `keep` frames
    let mut x = noise.to_vec(); // [T,64] token-major
    let mut apg_avg: Option<Vec<f64>> = None;
    // Guided velocity at (x, sigma) -> channel-major `[oc.T]` (cond, plus CFG/APG when active).
    let guided_vel = |x: &[f32],
                      sigma: f32,
                      step: usize,
                      apg: &mut Option<Vec<f64>>|
     -> crate::tensor::Result<Vec<f32>> {
        let (temb, tproj) = dit.time_embed_forward(sigma)?;
        let input = build_dit_input(context, x, t);
        let vel = dit.velocity_forward(&input, &tproj, &temb, enc, t, enc_s)?;
        // dual-condition: 3-way blend (1-gt).uncond + (gt-gl).text + gl.cond, every step.
        if let (Some(d), Some((uenc, uenc_s))) = (&sp.dual, uncond) {
            let vt = dit.velocity_forward(&input, &tproj, &temb, &d.text_enc, t, d.text_s)?;
            let vu = dit.velocity_forward(&input, &tproj, &temb, uenc, t, uenc_s)?;
            let (gt, gl) = (d.gs_text, d.gs_lyric);
            return Ok((0..vel.len())
                .map(|i| (1.0 - gt) * vu[i] + (gt - gl) * vt[i] + gl * vel[i])
                .collect());
        }
        Ok(if use_cfg {
            if let Some(scale) = sp.scale_at(step) {
                let (uenc, uenc_s) = uncond.unwrap();
                let vel_un = dit.velocity_forward(&input, &tproj, &temb, uenc, t, uenc_s)?;
                apply_guidance(
                    &vel,
                    &vel_un,
                    oc,
                    t,
                    scale,
                    sp.cfg_type,
                    step,
                    sp.zero_steps,
                    apg,
                )
            } else {
                vel
            }
        } else {
            vel
        })
    };
    // x += mean-shift(dt . vel): channel-major vel applied into token-major x (omega neutral ⟹ plain).
    let euler_apply = |x: &mut [f32], vel: &[f32], dt: f32| {
        if rescale == 1.0 {
            for c in 0..oc {
                for ti in 0..t {
                    x[ti * oc + c] += dt * vel[c * t + ti];
                }
            }
        } else {
            let sum: f64 = vel.iter().map(|&v| v as f64).sum();
            let m = dt * (sum / vel.len() as f64) as f32;
            for c in 0..oc {
                for ti in 0..t {
                    let dx = dt * vel[c * t + ti];
                    x[ti * oc + c] += (dx - m) * rescale + m;
                }
            }
        }
    };
    for step in 0..sched.len() {
        // The hook may ABORT: a music render takes minutes in spawn_blocking, so
        // this is the only place it can observe that its caller is gone.
        crate::inference::serve::progress::try_note(
            progress,
            crate::inference::serve::progress::phase::DENOISE,
            step + 1,
            sched.len(),
        )?;
        let sigma = sched[step];
        let sigma_next = if step + 1 < sched.len() {
            sched[step + 1]
        } else {
            0.0
        };
        let dt = sigma_next - sigma;
        // Each solver advances x and returns the velocity used (for the DCW denoised estimate).
        let dcw_vel: Option<Vec<f32>> = match sp.solver {
            Solver::Euler => {
                let v = guided_vel(&x, sigma, step, &mut apg_avg)?;
                euler_apply(&mut x, &v, dt);
                Some(v)
            }
            Solver::Heun => {
                let v1 = guided_vel(&x, sigma, step, &mut apg_avg)?;
                if sigma_next <= 0.0 {
                    euler_apply(&mut x, &v1, dt); // last step: Euler (no 2nd-order correction at σ=0)
                    Some(v1)
                } else {
                    let mut x_pred = x.clone();
                    for c in 0..oc {
                        for ti in 0..t {
                            x_pred[ti * oc + c] += dt * v1[c * t + ti];
                        }
                    }
                    let v2 = guided_vel(&x_pred, sigma_next, step, &mut apg_avg)?;
                    let deriv: Vec<f32> = v1.iter().zip(&v2).map(|(a, b)| 0.5 * (a + b)).collect();
                    euler_apply(&mut x, &deriv, dt);
                    Some(deriv)
                }
            }
            Solver::Pingpong => {
                let v = guided_vel(&x, sigma, step, &mut apg_avg)?;
                // Estimate x0 = x - σ.v, then re-noise to σ_next: x = (1-σ_next).x0 + σ_next.ε.
                let mut rng = sp
                    .seed
                    .wrapping_add(step as u64)
                    .wrapping_mul(0x9E3779B97F4A7C15)
                    .max(1);
                let mut u01 = || {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    (((rng >> 33) as f64 / (1u64 << 31) as f64) as f32).clamp(1e-7, 1.0 - 1e-7)
                };
                for c in 0..oc {
                    for ti in 0..t {
                        let idx = ti * oc + c;
                        let den = x[idx] - sigma * v[c * t + ti];
                        let (u1, u2) = (u01(), u01());
                        let eps = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
                        x[idx] = (1.0 - sigma_next) * den + sigma_next * eps;
                    }
                }
                None // stochastic -> DCW is skipped (it injects noise)
            }
        };
        // DCW: wavelet-domain correction on the post-step latent (denoised = x - v.σ_next), per
        // the band/mode with t-modulated scalers. Before repaint so masking keeps the final say.
        if let (Some(d), Some(v)) = (&sp.dcw, &dcw_vel) {
            let mut denoised = vec![0f32; t * oc];
            for c in 0..oc {
                for ti in 0..t {
                    denoised[ti * oc + c] = x[ti * oc + c] - v[c * t + ti] * sigma_next;
                }
            }
            let (s_low, s_high) = match d.mode {
                DcwMode::Low => (sigma * d.scaler, 0.0),
                DcwMode::High => (0.0, (1.0 - sigma) * d.scaler),
                DcwMode::Double => (sigma * d.scaler, (1.0 - sigma) * d.high_scaler),
                DcwMode::Pix => (d.scaler, 0.0),
            };
            dcw_apply(&mut x, &denoised, t, oc, d.mode, s_low, s_high);
        }
        // repaint/inpaint: force the kept frames back onto the reference's noised trajectory at
        // σ_next, so only the masked region is freely generated (the final step lands them exactly).
        if let Some(rp) = &sp.repaint {
            for ti in 0..t {
                if rp.keep[ti] {
                    for c in 0..oc {
                        x[ti * oc + c] = (1.0 - sigma_next) * rp.ref_latent[c * t + ti]
                            + sigma_next * noise0[ti * oc + c];
                    }
                }
            }
        }
    }
    Ok(x)
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;
    use crate::inference::model::acestep::fsq::{acestep_gguf, DetokModel};

    fn load_dump(path: &str) -> (Vec<f32>, Vec<usize>) {
        let b = std::fs::read(path).unwrap();
        let nd = i32::from_le_bytes(b[0..4].try_into().unwrap()) as usize;
        let mut shape = Vec::with_capacity(nd);
        for i in 0..nd {
            shape.push(i32::from_le_bytes(b[4 + i * 4..8 + i * 4].try_into().unwrap()) as usize);
        }
        let off = 4 + nd * 4;
        (
            b[off..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect(),
            shape,
        )
    }
    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    // CHEAP: codes -> FSQ detok -> build_context vs the oracle context dump.
    #[test]
    #[ignore = "needs DiT GGUF (config.test HF hub) + /tmp/acedump_detok + /tmp/detok_codes.csv - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn validate_context_build_vs_oracle() {
        let m = DetokModel::from_gguf(
            acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let codes: Vec<u32> = std::fs::read_to_string("/tmp/detok_codes.csv")
            .unwrap()
            .trim()
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let detok = m.decode(&codes).unwrap(); // [T.64]
        let t = codes.len() * 5;
        let ctx = build_context(&detok, t, t, &[]); // all-decoded (decoded_t == t)
        let (oref, osh) = load_dump("/tmp/acedump_detok/context.bin"); // [T,128]
        assert_eq!(osh, vec![t, 128]);
        let c = cosine(&ctx, &oref);
        println!("context build cosine={c:.6}");
        assert!(c > 0.999, "context cosine {c} too low");
    }

    // CHEAP: cond enc enc_hidden -> DiT condition_embedder vs enc_after_cond_emb.
    #[test]
    #[ignore = "needs DiT GGUF (config.test HF hub) + /tmp/acedump_detok dumps - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn validate_cond_emb_vs_oracle() {
        let dit = DitModel::from_gguf(
            acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let m = crate::inference::model::acestep::cond::CondModel::from_gguf(
            acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let (lyric_in, ls) = load_dump("/tmp/acedump_detok/lyric_embed.bin");
        let (text_in, ts) = load_dump("/tmp/acedump_detok/text_hidden.bin");
        let (timbre_in, tfs) = load_dump("/tmp/acedump_detok/timbre_feats.bin");
        let s_ref = tfs.iter().product::<usize>() / 64;
        let (enc_hidden, s_total) = m
            .forward(&text_in, ts[0], &lyric_in, ls[0], Some((&timbre_in, s_ref)))
            .unwrap();
        let enc = cond_emb_apply(&dit, &enc_hidden, s_total).unwrap();
        // enc_after_cond_emb dumped ggml [H, S] = flat[s.H+h] = [S,H] row-major.
        let (oref, _osh) = load_dump("/tmp/acedump_detok/enc_after_cond_emb.bin");
        let c = cosine(&enc, &oref);
        println!("cond_emb cosine={c:.6} (S_total={s_total})");
        assert!(c > 0.999, "cond_emb cosine {c} too low");
    }

    // Full DiT trajectory through the pipeline: codes -> FSQ detok -> context, +
    // dumped enc + noise -> 8-step Euler(dit_latent) -> latent, vs the dit_output
    // dump. Validates the Euler loop driving velocity_forward end-to-end (the last
    // DiT-chain link). Slow (8x velocity_forward at T=320).
    #[test]
    #[ignore = "needs DiT GGUF (config.test HF hub) + /tmp/acedump_detok + /tmp/detok_codes.csv; slow - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn validate_dit_latent_vs_oracle() {
        let dit = DitModel::from_gguf(
            acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let detok_m = DetokModel::from_gguf(
            acestep_gguf("acestep-v15-turbo-Q8_0.gguf")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let codes: Vec<u32> = std::fs::read_to_string("/tmp/detok_codes.csv")
            .unwrap()
            .trim()
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let detok = detok_m.decode(&codes).unwrap();
        let t = codes.len() * 5;
        let context = build_context(&detok, t, t, &[]);
        let (enc, esh) = load_dump("/tmp/acedump_detok/enc_after_cond_emb.bin"); // ggml [H,S]->[S,H]
        let enc_s = esh[1];
        let (noise, _) = load_dump("/tmp/acedump_detok/noise.bin"); // [T,64]
        let latent = dit_latent(&dit, &context, &enc, enc_s, &noise, t).unwrap();
        let (oref, _) = load_dump("/tmp/acedump_detok/dit_output.bin"); // [T,64]
        let c = cosine(&latent, &oref);
        println!("dit_latent (full 8-step Euler) cosine={c:.6} (T={t})");
        assert!(c > 0.95, "dit_latent cosine {c} too low");
    }

    // FIRST REAL AUDIBLE ARTIFACT: decode the oracle's validated DiT latent through
    // the GPU VAE and write an actual WAV to results/acestep/sample.wav. Proves the
    // audio pipeline RUNS and produces a real file (request0 content, VAE cosine 1.0).
    #[test]
    #[ignore = "needs VAE GGUF (config.test HF hub) + /tmp/acedump/dit_output.bin; writes results/acestep/sample.wav - NOTE: the script that produces these dumps is NOT in this repository, so this cannot be run as written; it is kept because the Rust half of the harness is reusable once the oracle is rebuilt"]
    fn render_dumped_latent_to_wav() {
        use crate::inference::model::acestep::vae::{encode_wav_s16le, OobleckDecoder};
        let gguf = acestep_gguf("vae-BF16.gguf");
        let dec = OobleckDecoder::from_gguf(gguf.to_str().unwrap(), 1e-12).unwrap();
        let (lat_tmajor, ls) = load_dump("/tmp/acedump/dit_output.bin"); // [T,64] token-major
        let (t_lat, oc) = (ls[0], ls[1]);
        // -> channel-major [oc.T] for the VAE.
        let mut latent = vec![0f32; oc * t_lat];
        for ti in 0..t_lat {
            for c in 0..oc {
                latent[c * t_lat + ti] = lat_tmajor[ti * oc + c];
            }
        }
        let t0 = std::time::Instant::now();
        let (audio, c_audio, t_audio) = dec.decode(&latent, oc, t_lat).unwrap();
        let secs = t_audio as f32 / 48000.0;
        let wav = encode_wav_s16le(&audio, c_audio, t_audio, 48000);
        let dir = format!("{}/results/acestep", env!("CARGO_MANIFEST_DIR"));
        std::fs::create_dir_all(&dir).ok();
        let path = format!("{dir}/sample.wav");
        std::fs::write(&path, &wav).unwrap();
        println!("wrote {path}: {c_audio}ch x {t_audio} samples ({secs:.1}s @48kHz), VAE decode {:.2}s, {} bytes",
                 t0.elapsed().as_secs_f32(), wav.len());
        assert!(wav.len() > 44 && t_audio == t_lat * 1920);
    }
}
