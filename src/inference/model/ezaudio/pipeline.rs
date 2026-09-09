//! EzAudio text->SFX pipeline (Stage 4 capstone) - full Rust on the native substrate.
//!
//! Wires the three validated EzAudio components into one end-to-end text-to-audio path:
//!   prompt -> FLAN-T5 encode -> DiT cross-attn context
//!   N(0,1) latent [128,T] -> N-step v-prediction CFG denoise (DiT velocity field)
//!   latent [128,T] -> Oobleck VAE decode (480x, 24 kHz) -> mono WAV
//!
//! The diffusion sampler is a DDIM v-prediction loop built to the EzAudio `diff:` config
//! (scaled_linear betas, zero-terminal-SNR rescale, 'trailing' timestep spacing). The DiT
//! input is the faithful 257-channel UDiT tensor `cat([noisy 128, gt 128, mask 1])` ("ncm",
//! exactly `MaskDiT.forward`): `gt` is the learned `mask_embed` broadcast over T and the mask
//! channel is all-ones (generate-all). CFG uses `guidance_rescale` (over-exposure fix).

use crate::inference::model::ezaudio::dit::EzAudioDiT;
use crate::inference::model::ezaudio::vae::{ezaudio_pt, load_ezaudio_decoder};
use crate::inference::model::t5::flan::{flan_t5_dir, t5_tokenize, T5Encoder};
use crate::tensor::{Result, Tensor};

/// EzAudio latent frame rate (Hz): `T = round(seconds . LATENT_SR)`.
pub const LATENT_SR: f32 = 50.0;
/// VAE channel count = DiT `out_chans` (the codec/latent dim).
const LATENT_CH: usize = 128;
/// CFG guidance-rescale factor (EzAudio `rescale_noise_cfg`, the over-exposure fix). 0.75 is
/// the reference text->audio default (`api/ezaudio.py generate_audio`). 0.0 disables.
const GUIDANCE_RESCALE: f32 = 0.75;

/// A v-prediction DDIM scheduler built from the EzAudio `diff:` config.
///
/// `alphas_cumprod` carries the scaled-linear schedule AFTER the zero-terminal-SNR rescale
/// (so `alphas_cumprod[T-1] == 0`, i.e. the highest train step is pure noise). `timesteps`
/// is the descending inference schedule selected with 'trailing' spacing. `final_alpha`
/// is the cumulative product used past the first step (`set_alpha_to_one` ⟹ 1.0).
pub struct VPredScheduler {
    alphas_cumprod: Vec<f32>,
    pub timesteps: Vec<usize>,
    final_alpha: f32,
    num_train: usize,
}

impl VPredScheduler {
    /// Build the scheduler for `steps` inference steps. Mirrors diffusers
    /// `DDIMScheduler(beta_schedule="scaled_linear", rescale_betas_zero_snr=True,
    /// timestep_spacing="trailing", prediction_type="v_prediction")`.
    pub fn new(steps: usize) -> Self {
        let num_train = 1000usize;
        let (beta_start, beta_end) = (0.00085f64, 0.012f64);
        // scaled_linear: betas = linspace(√β0, √β1, N)².
        let mut alphas_cumprod = vec![0f64; num_train];
        let mut prod = 1.0f64;
        for i in 0..num_train {
            let bs = beta_start.sqrt()
                + (beta_end.sqrt() - beta_start.sqrt()) * (i as f64) / (num_train as f64 - 1.0);
            let beta = bs * bs;
            prod *= 1.0 - beta;
            alphas_cumprod[i] = prod;
        }
        // rescale_zero_terminal_snr: shift √ᾱ so the terminal value is 0, rescale so the
        // first stays put -> ᾱ[T-1] = 0 (zero SNR at the highest timestep).
        let mut sqrt_bar: Vec<f64> = alphas_cumprod.iter().map(|x| x.sqrt()).collect();
        let (s0, st) = (sqrt_bar[0], sqrt_bar[num_train - 1]);
        for v in sqrt_bar.iter_mut() {
            *v = (*v - st) * (s0 / (s0 - st));
        }
        let alphas_cumprod: Vec<f32> = sqrt_bar.iter().map(|x| (x * x) as f32).collect();

        // 'trailing' spacing: timesteps = round(arange(T, 0, -T/N)) - 1, descending.
        let step_ratio = num_train as f64 / steps as f64;
        let mut timesteps = Vec::with_capacity(steps);
        let mut t = num_train as f64;
        while t > 0.5 {
            let ts = (t.round() as i64 - 1).max(0) as usize;
            timesteps.push(ts);
            t -= step_ratio;
        }

        // set_alpha_to_one default ⟹ the past-first-step cumulative product is 1.0.
        Self {
            alphas_cumprod,
            timesteps,
            final_alpha: 1.0,
            num_train,
        }
    }

    /// One DDIM v-prediction update (eta=0, deterministic). `sample` is x_t `[N]`, `v` the
    /// model velocity at step index `i` (into `self.timesteps`). Returns x_{t_prev} `[N]`.
    /// v->x0/eps: `x0 = √ᾱ_t.x_t - √(1-ᾱ_t).v`; `eps = √ᾱ_t.v + √(1-ᾱ_t).x_t`. clip_sample
    /// is false (no x0 clamp).
    pub fn step(&self, i: usize, sample: &[f32], v: &[f32]) -> Vec<f32> {
        let t = self.timesteps[i];
        let prev_t = t as i64 - (self.num_train / self.timesteps.len()) as i64;
        let a_t = self.alphas_cumprod[t];
        let a_prev = if prev_t >= 0 {
            self.alphas_cumprod[prev_t as usize]
        } else {
            self.final_alpha
        };
        let (sqrt_a_t, sqrt_b_t) = (a_t.sqrt(), (1.0 - a_t).sqrt());
        let (sqrt_a_prev, sqrt_b_prev) = (a_prev.sqrt(), (1.0 - a_prev).sqrt());
        sample
            .iter()
            .zip(v)
            .map(|(&x, &vv)| {
                let x0 = sqrt_a_t * x - sqrt_b_t * vv;
                let eps = sqrt_a_t * vv + sqrt_b_t * x;
                sqrt_a_prev * x0 + sqrt_b_prev * eps // eta=0 -> direction is √(1-ᾱ_prev).eps
            })
            .collect()
    }
}

/// Assemble the faithful 257-channel UDiT input `[in_chans, T]` (channel-major `flat[c.T+t]`),
/// exactly as `MaskDiT.forward` builds it for pure text->audio inference (no reference latent):
/// `cat([noisy(128), gt(128), mae_mask(1)], dim=channels)` - order **"ncm"**. Here `gt` is the
/// learned `mask_embed` broadcast over T (the MAE "everything is masked" conditioning tensor,
/// absmax ~0.017), and the mask channel is all-ones (`mae_mask = ones_like(x)` ⟹ generate all).
/// `latent` is `[128.T]` channel-major; `mask_embed` is `[128]`.
fn assemble_input(
    latent: &[f32],
    mask_embed: &[f32],
    t: usize,
    in_chans: usize,
    mask_val: f32,
) -> Vec<f32> {
    let mut x = vec![0f32; in_chans * t];
    // block n: noisy latent (channels 0..128).
    for c in 0..LATENT_CH {
        for ti in 0..t {
            x[c * t + ti] = latent[c * t + ti];
        }
    }
    // block c(gt): mask_embed broadcast over T (channels 128..256).
    for c in 0..LATENT_CH {
        for ti in 0..t {
            x[(LATENT_CH + c) * t + ti] = mask_embed[c];
        }
    }
    // block m: mae_mask (channel 256), all `mask_val` (1.0 = generate all).
    for ti in 0..t {
        x[2 * LATENT_CH * t + ti] = mask_val;
    }
    debug_assert_eq!(in_chans, 2 * LATENT_CH + 1);
    x
}

/// Render `prompt` to mono 24 kHz audio. Returns `(samples, t_audio)`.
///
/// `seconds` sets the latent length `T = round(seconds.50)`; `steps` the diffusion steps;
/// `cfg` the classifier-free guidance scale (cond = T5(prompt), uncond = T5("")); `seed`
/// the noise seed; `mask_val` the UDiT mask-channel value (the flagged "generate all"
/// selector - 1.0 = fully masked/generate, the MAE convention). Components are loaded here
/// (free-VRAM-gated placement is internal to each loader).
pub fn render(
    prompt: &str,
    seconds: f32,
    steps: usize,
    cfg: f32,
    seed: u64,
    mask_val: f32,
    negative: &str,
) -> Result<(Vec<f32>, usize)> {
    render_with_progress(
        prompt,
        seconds,
        steps,
        cfg,
        seed,
        mask_val,
        negative,
        |_, _, _| {},
    )
}

/// [`render`] with a per-denoise-step progress callback `(done, total)`.
pub fn render_with_progress(
    prompt: &str,
    seconds: f32,
    steps: usize,
    cfg: f32,
    seed: u64,
    mask_val: f32,
    // What the guidance steers AWAY from. Empty is the plain unconditional branch the
    // model was trained against, and stays the default.
    negative: &str,
    mut on_step: impl FnMut(&str, usize, usize),
) -> Result<(Vec<f32>, usize)> {
    let t = (seconds * LATENT_SR).round().max(1.0) as usize;

    // 1) Text encoder: cond = T5(prompt); uncond (CFG) = T5(negative). The second branch
    //    used to be pinned to "", so a caller's negative prompt reached here and did nothing.
    let t5_dir = flan_t5_dir();
    let t5 = T5Encoder::from_dir(&t5_dir)?;
    let t5_part =
        crate::inference::serve::progress::placement::part("text-encoder", &t5.placement());
    let ctx_cond = t5.encode(&t5_tokenize(&t5_dir, prompt)?)?;
    let use_cfg = cfg > 1.0;
    let ctx_uncond = if use_cfg {
        Some(t5.encode(&t5_tokenize(&t5_dir, negative)?)?)
    } else {
        None
    };
    drop(t5);
    drop(t5_part);

    // 2) DiT.
    let dit = EzAudioDiT::load_s3_large()?;
    let _dit_part = crate::inference::serve::progress::placement::part("dit", &dit.placement());

    // 3) init latent ~ N(0,1) [128, T] channel-major (LCG + Box-Muller, seed-driven).
    let mut rng = seed.max(1);
    let mut u01 = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((rng >> 33) as f64 / (1u64 << 31) as f64) as f32).clamp(1e-7, 1.0 - 1e-7)
    };
    let mut latent: Vec<f32> = (0..LATENT_CH * t)
        .map(|_| {
            let (u1, u2) = (u01(), u01());
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        })
        .collect();

    // 4) N-step CFG v-prediction denoise.
    let sched = VPredScheduler::new(steps);
    let in_chans = dit.in_chans();
    let t0 = std::time::Instant::now();
    let bypass = std::env::var("EZ_BYPASS").is_ok(); // decode raw noise latent (VAE isolation)
                                                     // Faithful UDiT input = cat([noisy, mask_embed, ones]) ("ncm"), the exact MaskDiT.forward
                                                     // assembly. This became correct once the block residual gating was fixed to (1 - gate)
                                                     // (inference/model/ezaudio/dit.rs) - per-tensor oracle vs torch now matches cos≈1.0 at every stage.
    let mask_embed = dit.mask_embed().to_vec();
    for i in 0..sched.timesteps.len() {
        if bypass {
            break;
        }
        let ts = sched.timesteps[i] as f32;
        let xin = assemble_input(&latent, &mask_embed, t, in_chans, mask_val);
        let xin_t = Tensor::from_vec_f32(xin, (in_chans, t))?;
        let v_cond = dit.forward(&xin_t, ts, &ctx_cond)?.to_vec_f32();
        let v = if let Some(cu) = &ctx_uncond {
            let v_un = dit.forward(&xin_t, ts, cu)?.to_vec_f32();
            let mut vp: Vec<f32> = v_un
                .iter()
                .zip(&v_cond)
                .map(|(u, c)| u + cfg * (c - u))
                .collect();
            // guidance_rescale (rescale_noise_cfg, EzAudio inference.py): scale the CFG output
            // back to the cond prediction's std to fix over-exposure/clipping, then blend.
            // 0.75 = the reference text->audio default (api/ezaudio.py generate_audio).
            if GUIDANCE_RESCALE > 0.0 {
                let std = |a: &[f32]| {
                    let m = a.iter().sum::<f32>() / a.len() as f32;
                    (a.iter().map(|x| (x - m).powi(2)).sum::<f32>() / a.len() as f32).sqrt()
                };
                let (s_text, s_cfg) = (std(&v_cond), std(&vp).max(1e-8));
                let r = s_text / s_cfg;
                for x in vp.iter_mut() {
                    *x = GUIDANCE_RESCALE * (*x * r) + (1.0 - GUIDANCE_RESCALE) * *x;
                }
            }
            vp
        } else {
            v_cond
        };
        if std::env::var("EZ_DEBUG").is_ok() && (i == 0 || i + 1 == sched.timesteps.len()) {
            let st = |a: &[f32]| {
                let m = a.iter().sum::<f32>() / a.len() as f32;
                let sd = (a.iter().map(|x| (x - m).powi(2)).sum::<f32>() / a.len() as f32).sqrt();
                (m, sd, a.iter().cloned().fold(f32::MIN, f32::max))
            };
            let (vm, vs, vx) = st(&v);
            let (lm, ls, lx) = st(&latent);
            eprintln!("[ez-dbg] step{i} t={ts} v(mean={vm:.4} std={vs:.4} max={vx:.4}) latent_in(mean={lm:.4} std={ls:.4} max={lx:.4})");
        }
        latent = sched.step(i, &latent, &v);
        on_step(
            crate::inference::serve::progress::phase::DENOISE,
            i + 1,
            sched.timesteps.len(),
        );
    }
    if std::env::var("EZ_DEBUG").is_ok() {
        let m = latent.iter().sum::<f32>() / latent.len() as f32;
        let sd = (latent.iter().map(|x| (x - m).powi(2)).sum::<f32>() / latent.len() as f32).sqrt();
        eprintln!("[ez-dbg] FINAL latent mean={m:.4} std={sd:.4}");
    }
    eprintln!(
        "[ezaudio] {} diffusion steps in {:.1}s",
        steps,
        t0.elapsed().as_secs_f32()
    );

    // 5) VAE decode (autoencoder scale=1.0 shift=0.0 ⟹ latent fed directly). Channel-major
    //    [128.T] is exactly the decoder's layout. decode_chunked derives the 480x upsample
    //    from the loaded block strides.
    let vae_pt = ezaudio_pt("ckpts/vae/1m.pt");
    let vae = load_ezaudio_decoder(
        vae_pt
            .to_str()
            .ok_or_else(|| crate::tensor::Error("ezaudio: bad VAE path".into()))?,
        1e-12,
    )?;
    let _vae_part = crate::inference::serve::progress::placement::part("vae", &vae.placement());
    let t1 = std::time::Instant::now();
    let (audio, c_out, t_audio) = vae.decode_chunked(&latent, LATENT_CH, t, 192, 64)?;
    eprintln!(
        "[ezaudio] VAE decode {:.1}s -> {} samples ({} ch)",
        t1.elapsed().as_secs_f32(),
        t_audio,
        c_out
    );
    debug_assert_eq!(c_out, 1);
    Ok((audio, t_audio))
}
