//! FLUX sampling: the noise it starts from, the schedule it walks, and the packing that turns a
//! latent into tokens and back.
//!
//! The schedule is a rectified flow's, so where the stops sit and what one step does are the
//! shared statements in [`crate::inference::model::flow_match`]; what is written here is FLUX's
//! own two-by-two packing and the loops that drive a transformer along the schedule.
//!
//! There are two of those loops per task, and both drive whatever implements [`WithForward`].
//! [`denoise`] takes the state as the text encoders published it and casts per step;
//! [`denoise_native`] takes a state cast ONCE ([`NativeState`]) and keeps its tensors f32 on the
//! model's device for the whole loop, so the dtype cast is paid once at each boundary instead of
//! once per step - and it is the only one that carries step reuse, an inpaint mask and regions.

use crate::inference::model::flow_match;
use crate::inference::model::patches;
use crate::tensor::{Device, Result, Tensor};

/// The transformer interface [`denoise`] drives: one velocity prediction per step.
pub trait WithForward {
    fn forward(
        &self,
        img: &Tensor,
        img_ids: &Tensor,
        txt: &Tensor,
        txt_ids: &Tensor,
        timesteps: &Tensor,
        y: &Tensor,
        guidance: Option<&Tensor>,
    ) -> Result<Tensor>;

    /// Called between two denoise steps. The default does nothing; a model whose blocks
    /// are spread over several devices uses it to revisit that placement while the render
    /// is still running, instead of living with the one it was given when it loaded.
    fn between_steps(&mut self) {}
}

pub fn get_noise(
    num_samples: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    // The VAE divides each side by eight and the packing by another two, so a latent side is
    // the image's rounded up to a whole patch and cut by eight.
    flow_match::gaussian_latent(
        (
            num_samples,
            16,
            height.div_ceil(16) * 2,
            width.div_ceil(16) * 2,
        ),
        device,
    )
}

#[derive(Debug, Clone)]
pub struct State {
    pub img: Tensor,
    pub img_ids: Tensor,
    pub txt: Tensor,
    pub txt_ids: Tensor,
    pub vec: Tensor,
}

/// Where each packed token sits: the image it belongs to, then its row and its column.
///
/// FLUX addresses a token by that triple, and the FIRST entry is the image index - 0 for the
/// noise grid being denoised, 1 for a clean reference concatenated beside it. It is the only
/// thing telling the attention which of the two streams a token came from, so an edit that
/// numbered its reference 0 would hand the model two grids it cannot tell apart.
fn position_ids(
    image_index: f32,
    bs: usize,
    h: usize,
    w: usize,
    dtype: DType,
    dev: &Device,
) -> Result<Tensor> {
    Tensor::stack(
        &[
            // Every row is f32: `stack` concatenates them and the tensor library takes one
            // dtype across the whole stack. The triple is cast to the model's dtype below.
            Tensor::full(image_index, (h / 2, w / 2), dev)?,
            Tensor::arange(0.0, (h as u32 / 2) as f32)?
                .to_device(dev)?
                .reshape((h / 2, 1))?
                .broadcast_as((h / 2, w / 2))?,
            Tensor::arange(0.0, (w as u32 / 2) as f32)?
                .to_device(dev)?
                .reshape((1, w / 2))?
                .broadcast_as((h / 2, w / 2))?,
        ],
        2,
    )?
    .to_dtype(dtype)?
    .reshape((1, h / 2 * w / 2, 3))?
    .repeat((bs, 1, 1))
}

impl State {
    pub fn new(t5_emb: &Tensor, clip_emb: &Tensor, img: &Tensor) -> Result<Self> {
        let dtype = img.dtype();
        let (bs, _c, h, w) = img.dims4()?;
        let dev = img.device();
        let img = pack(img)?;
        let img_ids = position_ids(0f32, bs, h, w, dtype, &dev)?;
        let txt = t5_emb.repeat(bs)?;
        let txt_ids = Tensor::zeros_on((bs, txt.dim(1)?, 3), dtype, &dev)?;
        let vec = clip_emb.repeat(bs)?;
        Ok(Self {
            img,
            img_ids,
            txt,
            txt_ids,
            vec,
        })
    }
}

/// `shift` is a triple `(image_seq_len, base_shift, max_shift)`.
/// A schedule that starts at `t_start` instead of at 1.0, with the FULL step count.
///
/// img2img used `get_schedule(n)[start..]` with `start = round((1-strength)*n)`, which
/// has two faults at the small step counts these distilled models run at. It quantised
/// the control - on 4-step schnell, strength 0.9 and 1.0 both truncate nothing, so both
/// discard the source entirely - and it starved the schedule at the other end, because
/// what remains after truncation IS the step budget. Below about 0.4 strength fewer
/// than two timesteps were left and the request failed outright with "Strength too
/// low": asking for a gentle edit returned an error.
///
/// Rebuilding from `t_start` gives an exact, monotone control at every strength and
/// always leaves `num_steps` steps. `t_start = 1.0` reproduces [`get_schedule`] exactly.
pub fn get_schedule_from(
    t_start: f64,
    num_steps: usize,
    shift: Option<(usize, f64, f64)>,
) -> Vec<f64> {
    let timesteps = flow_match::descent_to_zero(t_start.clamp(0.0, 1.0), num_steps);
    match shift {
        None => timesteps,
        Some((image_seq_len, base_shift, max_shift)) => {
            // FLUX draws its push through a 256-token image and a 4096-token one.
            let mu = flow_match::shift_for(image_seq_len, (256, base_shift), (4096, max_shift));
            timesteps
                .into_iter()
                .map(|v| flow_match::time_shift(mu, 1., v))
                .collect()
        }
    }
}

/// The whole ladder, from pure noise: [`get_schedule_from`] starting at one.
pub fn get_schedule(num_steps: usize, shift: Option<(usize, f64, f64)>) -> Vec<f64> {
    get_schedule_from(1.0, num_steps, shift)
}

/// The inverse of [`unpack`]: `[b, c, h, w]` latents -> `[b, (h/2)*(w/2), c*4]` tokens.
///
/// `State::new` does this inline for the sampling latent; an inpaint needs the SAME
/// packing for the source, or the mask would address different pixels than the model.
pub fn pack(latent: &Tensor) -> Result<Tensor> {
    patches::cut(latent, 2, patches::Order::ByChannel)
}

/// `height` and `width` are the IMAGE's, in pixels: the VAE has already divided them by eight
/// and the patching by another two, which is where the sixteen comes from.
pub fn unpack(xs: &Tensor, height: usize, width: usize) -> Result<Tensor> {
    let channels = xs.dim(2)? / 4;
    patches::weave(
        xs,
        height.div_ceil(16),
        width.div_ceil(16),
        2,
        channels,
        patches::Order::ByChannel,
    )
}

/// Cooperative cancellation for the sampling loops: a dropped client must not
/// leave a render burning the GPU to completion. Every loop checks ONCE PER STEP -
/// a step is the natural granularity (a forward cannot be interrupted) and the
/// check costs an atomic load.
fn cancelled(cancel: Option<&crate::inference::serve::cancel::CancelToken>) -> bool {
    cancel.is_some_and(|c| c.is_cancelled())
}

fn cancel_err() -> crate::tensor::Error {
    // Log it: a render that stops early is otherwise invisible server-side (the
    // error travels back over a connection the client has already dropped).
    tracing::info!("flux: generation cancelled by the client; stopping at the current step");
    crate::tensor::Error::msg("flux: generation cancelled")
}

pub fn denoise<M: WithForward>(
    model: &mut M,
    img: &Tensor,
    img_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
    guidance: f64,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> Result<Tensor> {
    let b_sz = img.dim(0)?;
    let dev = img.device();
    let guidance = Tensor::full(guidance as f32, b_sz, &dev)?;
    let mut img = img.clone();
    for (t_curr, t_prev) in flow_match::steps(timesteps) {
        if cancelled(cancel) {
            return Err(cancel_err());
        }
        let t_vec = Tensor::full(t_curr as f32, b_sz, &dev)?;
        let pred = model.forward(&img, img_ids, txt, txt_ids, &t_vec, vec_, Some(&guidance))?;
        img = flow_match::euler_step(&img, &pred, t_prev - t_curr)?;
        model.between_steps();
    }
    Ok(img)
}

/// Pack a VAE latent `[bs, c, h, w]` into Kontext CONTEXT tokens: the same packing `State::new`
/// gives the noise grid, numbered as the reference image instead of as the grid.
/// Returns `(tokens [bs, h/2.w/2, c.4], ids [bs, h/2.w/2, 3])`.
pub fn pack_context(latent: &Tensor) -> Result<(Tensor, Tensor)> {
    let dtype = latent.dtype();
    let (bs, _c, h, w) = latent.dims4()?;
    let dev = latent.device();
    let img = pack(latent)?;
    let ids = position_ids(1f32, bs, h, w, dtype, &dev)?;
    Ok((img, ids))
}

/// Kontext denoise: the noise tokens are denoised while the clean reference-image CONTEXT
/// tokens are concatenated as a fixed condition each step - the transformer attends over both,
/// but only the noise portion integrates the predicted velocity. Returns the denoised noise
/// latent, unpacked and VAE-decoded downstream exactly like text->image.
pub fn denoise_kontext<M: WithForward>(
    model: &mut M,
    noise: &Tensor,
    noise_ids: &Tensor,
    ctx: &Tensor,
    ctx_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
    guidance: f64,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> Result<Tensor> {
    let b_sz = noise.dim(0)?;
    let n_noise = noise.dim(1)?;
    let dev = noise.device();
    let guidance = Tensor::full(guidance as f32, b_sz, &dev)?;
    let ids = Tensor::cat(&[noise_ids, ctx_ids], 1)?;
    let mut img = noise.clone();
    for (t_curr, t_prev) in flow_match::steps(timesteps) {
        if cancelled(cancel) {
            return Err(cancel_err());
        }
        let t_vec = Tensor::full(t_curr as f32, b_sz, &dev)?;
        let full = Tensor::cat(&[&img, ctx], 1)?; // [noise ‖ fixed context]
        let pred = model.forward(&full, &ids, txt, txt_ids, &t_vec, vec_, Some(&guidance))?;
        let pred_noise = pred.narrow(1, 0, n_noise)?; // only the noise tokens' velocity
        img = flow_match::euler_step(&img, &pred_noise, t_prev - t_curr)?;
        model.between_steps();
    }
    Ok(img)
}

// ---------------------------------------------------------------------------
// The loops that cast the state once instead of once per step.
//
// The latent, timestep and guidance tensors stay f32 on the model's own device for the WHOLE
// loop, so the dtype cast is paid ONCE at each boundary - the text-encoder products going in
// ([`NativeState::to_f32`]), the VAE-decode input coming out - instead of a round trip per step.
// A welcome side effect: the model's identity-keyed txt/y/guidance embedder caches HIT across
// steps, where a per-step cast allocated fresh storage every step and forced a miss.
//
// They ask for [`WithForward`] like the loops above, so a placement that spans several devices
// reaches them too: what these add - a state cast once, step reuse, an inpaint mask, regions  -
// is a property of the LOOP and not of where the blocks sit.
// ---------------------------------------------------------------------------

use crate::tensor;
use crate::tensor::DType;

/// The text a forward is conditioned on: the token stream, its position ids, and the pooled
/// vector the modulation reads.
///
/// A whole request carries one of these and so does a region, and that is the only difference
/// between a plain step and a regional one.
type Conditioning<'a> = (&'a tensor::Tensor, &'a tensor::Tensor, &'a tensor::Tensor);

/// The denoise-loop state cast once, before the loop, from the [`State`] the text encoders and
/// the initial noise produced.
pub struct NativeState {
    pub img: tensor::Tensor,
    pub img_ids: tensor::Tensor,
    pub txt: tensor::Tensor,
    pub txt_ids: tensor::Tensor,
    pub vec: tensor::Tensor,
}

impl NativeState {
    pub fn to_f32(state: &State) -> tensor::Result<Self> {
        Ok(Self {
            img: state.img.to_dtype(DType::F32)?,
            img_ids: state.img_ids.to_dtype(DType::F32)?,
            txt: state.txt.to_dtype(DType::F32)?,
            txt_ids: state.txt_ids.to_dtype(DType::F32)?,
            vec: state.vec.to_dtype(DType::F32)?,
        })
    }

    fn conditioning(&self) -> Conditioning<'_> {
        (&self.txt, &self.txt_ids, &self.vec)
    }
}

/// One velocity prediction for `tokens` at `t_vec`, conditioned on `txt`.
///
/// Written once because the argument list is long and two of its entries are 1-D tensors that
/// would swap without a type error: the timestep vector and the pooled text vector.
fn predict<M: WithForward>(
    model: &M,
    tokens: &tensor::Tensor,
    ids: &tensor::Tensor,
    txt: Conditioning<'_>,
    t_vec: &tensor::Tensor,
    guidance: &tensor::Tensor,
) -> tensor::Result<tensor::Tensor> {
    model.forward(tokens, ids, txt.0, txt.1, t_vec, txt.2, Some(guidance))
}

/// `Tensor::full` for the native substrate: a length-`len` 1-D fill on `dev`
/// (the timestep/guidance vectors the DiT forward expects).
fn full_native(value: f32, len: usize, dev: &tensor::Device) -> tensor::Result<tensor::Tensor> {
    tensor::Tensor::from_vec_f32(vec![value; len], len)?.to_device(dev)
}

/// Adaptive step reuse for flow-matching samplers ("EasyCache"/TeaCache class).
///
/// Consecutive denoise steps often predict nearly the same velocity, and re-running
/// the DiT for a prediction that barely moved is the single largest avoidable cost of
/// a multi-step render. The estimator is the published one: track how much the LATENT
/// changed between steps and how much the PREDICTION changed last time it was
/// recomputed; their ratio predicts this step's output change without running the
/// model. While the predicted change accumulates below `threshold`, reuse the last
/// prediction; the first recompute after a skip resets the accumulator.
///
/// This is an APPROXIMATION - it is opt-in per request and off by default, and the
/// first and last steps are never skipped (they set the composition and the final
/// detail). `threshold = 0` disables it entirely.
pub struct StepReuse {
    threshold: f32,
    /// Mean |latent| difference basis from the previous step.
    x_prev: Option<tensor::Tensor>,
    pred_prev: Option<tensor::Tensor>,
    /// Mean |prediction| of the last real forward - the scale the change is relative to.
    pred_norm: f32,
    /// Last measured d(prediction)/d(latent).
    rate: Option<f32>,
    cumulative: f32,
    pub skipped: usize,
}

/// Mean absolute value of a tensor, host-side (one small reduction per step).
fn mean_abs(t: &tensor::Tensor) -> tensor::Result<f32> {
    let v = t.to_dtype(tensor::DType::F32)?.to_vec_f32();
    if v.is_empty() {
        return Ok(0.0);
    }
    Ok(v.iter().map(|x| x.abs()).sum::<f32>() / v.len() as f32)
}

/// Mean absolute difference between two tensors of the same shape.
fn mean_abs_diff(a: &tensor::Tensor, b: &tensor::Tensor) -> tensor::Result<f32> {
    mean_abs(&a.sub(b)?)
}

impl StepReuse {
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            x_prev: None,
            pred_prev: None,
            pred_norm: 0.0,
            rate: None,
            cumulative: 0.0,
            skipped: 0,
        }
    }

    pub fn enabled(&self) -> bool {
        self.threshold > 0.0
    }

    /// The prediction to reuse for this step, or None when the model must run.
    /// `last` is true on the final step, which is never skipped.
    pub fn reuse(
        &mut self,
        x: &tensor::Tensor,
        last: bool,
    ) -> tensor::Result<Option<tensor::Tensor>> {
        if !self.enabled() || last {
            return Ok(None);
        }
        let (Some(x_prev), Some(pred_prev), Some(rate)) =
            (self.x_prev.as_ref(), self.pred_prev.as_ref(), self.rate)
        else {
            return Ok(None);
        };
        if self.pred_norm <= 0.0 {
            return Ok(None);
        }
        let input_change = mean_abs_diff(x, x_prev)?;
        self.cumulative += rate * input_change / self.pred_norm;
        if self.cumulative < self.threshold {
            self.skipped += 1;
            return Ok(Some(pred_prev.clone()));
        }
        self.cumulative = 0.0;
        Ok(None)
    }

    /// Record a REAL forward: refresh the change rate and the reuse candidate.
    pub fn observe(&mut self, x: &tensor::Tensor, pred: &tensor::Tensor) -> tensor::Result<()> {
        if !self.enabled() {
            return Ok(());
        }
        if let (Some(x_prev), Some(pred_prev)) = (self.x_prev.as_ref(), self.pred_prev.as_ref()) {
            let input_change = mean_abs_diff(x, x_prev)?;
            if input_change > 0.0 {
                self.rate = Some(mean_abs_diff(pred, pred_prev)? / input_change);
            }
        }
        self.pred_norm = mean_abs(pred)?;
        self.x_prev = Some(x.clone());
        self.pred_prev = Some(pred.clone());
        Ok(())
    }
}

/// What an inpaint pins down: the source image's own latent, and where to keep it.
///
/// Masked editing is not a different sampler - it is the ordinary loop with the
/// untouched region overwritten after every step by the SOURCE at that step's noise
/// level. The model still attends to the whole canvas, so what it generates inside the
/// hole stays consistent with the surroundings; it simply never gets to keep its
/// opinion about the outside.
///
/// Doing this only once at the end would produce a hole that ignores its
/// surroundings; doing it without re-noising to the step's level would feed the model
/// a clean region next to a noisy one, which it reads as an edge and paints over.
pub struct Inpaint {
    /// The source image's clean latent, packed like `img`.
    pub clean: tensor::Tensor,
    /// The noise the source is re-noised with. Drawn ONCE: a fresh draw per step is a
    /// different image each time and the untouched region visibly crawls.
    pub noise: tensor::Tensor,
    /// Per-token weight, `[b, seq, 1]`: 1 keeps the source, 0 lets the model generate.
    pub keep: tensor::Tensor,
}

impl Inpaint {
    /// Overwrite the kept region with the source at noise level `t`.
    fn apply(&self, img: &tensor::Tensor, t: f64) -> tensor::Result<tensor::Tensor> {
        let t = t as f32;
        // Flow matching: the sample at time t on the path from image to noise.
        let src = self
            .clean
            .affine(1.0 - t, 0.0)?
            .add(&self.noise.affine(t, 0.0)?)?;
        let keep = &self.keep;
        let generated = img.broadcast_mul(&keep.affine(-1.0, 1.0)?)?;
        generated.add(&src.broadcast_mul(keep)?)
    }
}

/// One area of the canvas that carries its OWN text conditioning.
///
/// Two subjects asked for in a single prompt get blended: the attention has no reason to
/// keep them apart, so two bodies merge into one. Regional conditioning removes the
/// ambiguity instead of fighting it - each region is denoised against its own prompt and
/// the predictions are recombined per token, so nothing couples the two descriptions.
///
/// `weight` is per IMAGE TOKEN, shaped `[1, tokens, 1]` to broadcast over the channel
/// axis, and the set of regions handed to `denoise_native` must already sum to 1 at
/// every token - see [`region_weights`].
pub struct Region {
    pub txt: tensor::Tensor,
    pub txt_ids: tensor::Tensor,
    pub vec: tensor::Tensor,
    pub weight: tensor::Tensor,
}

impl Region {
    fn conditioning(&self) -> Conditioning<'_> {
        (&self.txt, &self.txt_ids, &self.vec)
    }
}

/// Kontext denoise from a state cast once: what [`denoise_kontext`] does, in the loop that
/// keeps its tensors on the model's device.
///
/// A Kontext checkpoint is a dev checkpoint - same config, same blocks - so the transformer
/// loads it unchanged, and this loop was the only piece missing. Until it existed an edit
/// request fell back to the per-step-cast loop, which is the slower of the two.
///
/// Step reuse, inpainting and regions are deliberately absent rather than forwarded: each
/// decides what to do with the noise grid alone, and here the forward sees noise and a fixed
/// reference concatenated. Passing them through would apply a mask, or a reuse threshold, to
/// a tensor that is half reference image.
pub fn denoise_kontext_native<M: WithForward>(
    model: &M,
    state: &NativeState,
    ctx: &tensor::Tensor,
    ctx_ids: &tensor::Tensor,
    timesteps: &[f64],
    guidance: f64,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> tensor::Result<tensor::Tensor> {
    let b_sz = state.img.dim(0)?;
    let n_noise = state.img.dim(1)?;
    let dev = state.img.device();
    let guidance = full_native(guidance as f32, b_sz, &dev)?;
    let ids = tensor::Tensor::cat(&[&state.img_ids, ctx_ids], 1)?;
    let mut img = state.img.clone();
    for (t_curr, t_prev) in flow_match::steps(timesteps) {
        if cancelled(cancel) {
            return Err(cancel_err());
        }
        let t_vec = full_native(t_curr as f32, b_sz, &dev)?;
        let full = tensor::Tensor::cat(&[&img, ctx], 1)?;
        let pred = predict(model, &full, &ids, state.conditioning(), &t_vec, &guidance)?;
        // Only the noise tokens integrate the velocity; the reference stays clean.
        let pred_noise = pred.narrow(1, 0, n_noise)?;
        img = flow_match::euler_step(&img, &pred_noise, t_prev - t_curr)?;
    }
    Ok(img)
}

/// Per-token weights for a base prompt plus `rects`, normalised so every token sums to 1.
///
/// `rects` are `(x, y, w, h)` in 0..1 of the image, with a strength. The latent is packed into
/// 2x2 patches scanned row-major, so token `t` is at patch row `t / gw` and column `t % gw` -
/// the mapping this depends on, and the one thing here that would put a region silently in the
/// wrong place if it were wrong.
///
/// The base prompt keeps weight 1 everywhere, so an area no rectangle covers is still
/// described, and a region with strength `s` ends up at `s / (1 + s)` against it.
pub fn region_weights(
    height: usize,
    width: usize,
    rects: &[(f32, f32, f32, f32, f32)],
    dev: &tensor::Device,
) -> tensor::Result<Vec<tensor::Tensor>> {
    let (gh, gw) = (height.div_ceil(16), width.div_ceil(16));
    let n = gh * gw;
    let mut raw = vec![vec![1.0f32; n]]; // the base prompt, everywhere
    for &(x, y, w, h, strength) in rects {
        let mut m = vec![0.0f32; n];
        let x0 = ((x * gw as f32).floor().max(0.0) as usize).min(gw);
        let y0 = ((y * gh as f32).floor().max(0.0) as usize).min(gh);
        let x1 = (((x + w) * gw as f32).ceil().max(0.0) as usize).clamp(x0, gw);
        let y1 = (((y + h) * gh as f32).ceil().max(0.0) as usize).clamp(y0, gh);
        for r in y0..y1 {
            for c in x0..x1 {
                m[r * gw + c] = strength.max(0.0);
            }
        }
        raw.push(m);
    }
    // Normalise per token. A token covered by nothing keeps the base at 1; a token in
    // several regions splits between them in proportion.
    let mut out = Vec::with_capacity(raw.len());
    for k in 0..raw.len() {
        let mut v = Vec::with_capacity(n);
        for t in 0..n {
            let total: f32 = raw.iter().map(|m| m[t]).sum();
            v.push(if total > 0.0 {
                raw[k][t] / total
            } else {
                f32::from(k == 0)
            });
        }
        out.push(tensor::Tensor::from_vec_f32(v, vec![1, n, 1])?.to_device(dev)?);
    }
    Ok(out)
}

/// [`denoise`] from a state cast once, with no per-step dtype cast in the loop.
///
/// `on_step(i)` fires after step `i`, zero-based, for progress reporting.
pub fn denoise_native<M: WithForward>(
    model: &M,
    state: &NativeState,
    timesteps: &[f64],
    guidance: f64,
    mut on_step: impl FnMut(usize),
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    reuse_threshold: f32,
    inpaint: Option<&Inpaint>,
    regions: &[Region],
) -> tensor::Result<tensor::Tensor> {
    let b_sz = state.img.dim(0)?;
    let dev = state.img.device();
    let guidance = full_native(guidance as f32, b_sz, &dev)?;
    let mut img = state.img.clone();
    // Pin the kept region before the first forward too, so the model's very first look
    // at the canvas already shows the source outside the hole.
    if let (Some(ip), Some(t0)) = (inpaint, timesteps.first()) {
        img = ip.apply(&img, *t0)?;
    }
    // Step reuse decides whether to skip a forward by how little the latent moved.
    // An inpaint moves it every step from the outside, which reads as motion the model
    // did not cause, so the two cannot be combined; the mask wins.
    // Regions denoise against DIFFERENT prompts, so the latent moves for a reason step
    // reuse cannot see either - same argument as the inpaint mask above.
    let mut reuse = StepReuse::new(if inpaint.is_some() || !regions.is_empty() {
        0.0
    } else {
        reuse_threshold
    });
    let total = timesteps.len().saturating_sub(1);
    for (step, (t_curr, t_prev)) in flow_match::steps(timesteps).enumerate() {
        if cancelled(cancel) {
            return Err(cancel_err());
        }
        let pred = match reuse.reuse(&img, step + 1 == total)? {
            Some(cached) => cached,
            None => {
                let t_vec = full_native(t_curr as f32, b_sz, &dev)?;
                // REGIONAL: one forward per region, recombined per token by the weights.
                // Each region sees the whole canvas but is described by its own prompt,
                // and the weights decide who owns which token - so two subjects cannot
                // borrow each other's description, which is what merges two bodies into
                // one. The cost is honest and linear: N regions is N forwards per step.
                let pred = if regions.is_empty() {
                    predict(
                        model,
                        &img,
                        &state.img_ids,
                        state.conditioning(),
                        &t_vec,
                        &guidance,
                    )?
                } else {
                    let mut acc: Option<tensor::Tensor> = None;
                    for r in regions {
                        let p = predict(
                            model,
                            &img,
                            &state.img_ids,
                            r.conditioning(),
                            &t_vec,
                            &guidance,
                        )?;
                        let w = r.weight.to_dtype(p.dtype())?;
                        let wp = p.broadcast_mul(&w)?;
                        acc = Some(match acc {
                            None => wp,
                            Some(a) => a.add(&wp)?,
                        });
                    }
                    acc.ok_or_else(|| tensor::Error("regional denoise produced nothing".into()))?
                };
                reuse.observe(&img, &pred)?;
                pred
            }
        };
        img = flow_match::euler_step(&img, &pred, t_prev - t_curr)?;
        if let Some(ip) = inpaint {
            img = ip.apply(&img, t_prev)?;
        }
        on_step(step);
    }
    if reuse.skipped > 0 {
        tracing::info!(
            "flux: reused {} of {total} denoise steps (threshold {reuse_threshold})",
            reuse.skipped
        );
    }
    Ok(img)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The img2img control must be exact, monotone, and never cost the caller steps.
    ///
    /// Truncating the full schedule failed all three at the step counts distilled
    /// models use: on 4-step schnell, strength 0.9 and 1.0 both truncate nothing (so
    /// both discard the source), and below ~0.4 fewer than two timesteps survived and
    /// the request ERRORED with "Strength too low" - a gentle edit returned a failure.
    #[test]
    fn a_started_schedule_is_exact_monotone_and_keeps_the_step_budget() {
        for steps in [4usize, 9, 28] {
            let mut prev = f64::NEG_INFINITY;
            for pct in (5..=100).step_by(5) {
                let strength = pct as f64 / 100.0;
                let ts = get_schedule_from(strength, steps, None);
                assert_eq!(ts.len(), steps + 1, "strength {strength} lost steps");
                assert!(
                    (ts[0] - strength).abs() < 1e-12,
                    "start {} != {strength}",
                    ts[0]
                );
                assert!(ts[0] > prev, "strength {strength} did not move the start");
                prev = ts[0];
                assert_eq!(*ts.last().unwrap(), 0.0, "must finish fully denoised");
                for w in ts.windows(2) {
                    assert!(w[1] < w[0], "schedule stopped descending at {strength}");
                }
            }
        }
        // Full strength reproduces the untruncated schedule exactly.
        for steps in [4usize, 28] {
            assert_eq!(
                get_schedule_from(1.0, steps, None),
                get_schedule(steps, None)
            );
        }
    }

    /// Both loops in this file take their step from one place, so what that place computes is
    /// held against the expression the loops used to write out - on a real schnell schedule,
    /// chaining the outputs the way a render does.
    ///
    /// A step is where a schedule turns into an image: take the wrong two stops and every
    /// prediction is scaled by the wrong amount, with nothing about the result looking wrong.
    #[test]
    fn a_step_matches_the_expression_the_loops_are_written_from() {
        let dev = Device::Cpu;
        let n = 64usize;
        let img_v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let pred_v: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.11).cos() * 2.0 - 0.5)
            .collect();

        let mut want = Tensor::from_vec(img_v.clone(), (4, 16), &dev).unwrap();
        let mut got = Tensor::from_vec(img_v, (4, 16), &dev).unwrap();
        let pred = Tensor::from_vec(pred_v, (4, 16), &dev).unwrap();

        for (t_curr, t_prev) in flow_match::steps(&get_schedule(4, None)) {
            want = (want + pred.clone() * (t_prev - t_curr)).unwrap();
            got = flow_match::euler_step(&got, &pred, t_prev - t_curr).unwrap();

            let w: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();
            let g: Vec<f32> = got.flatten_all().unwrap().to_vec1().unwrap();
            for (a, b) in w.iter().zip(g.iter()) {
                assert!((a - b).abs() < 1e-6, "{a} vs {b}");
            }
        }
    }
}

#[cfg(test)]
mod region_tests {
    use super::*;

    /// A rectangle must land where it was asked for, and the weights must be a partition.
    ///
    /// This is the one place regional conditioning can be silently wrong: Flux packs the
    /// latent into 2x2 patches scanned row-major, so a token index is `row * gw + col`.
    /// Get that backwards and the left-hand region conditions the TOP of the image - the
    /// render still looks plausible, just not what was asked, and no error is raised.
    #[test]
    fn a_left_half_region_covers_the_left_half_of_the_token_grid() {
        let (h, w) = (640usize, 1024usize);
        let (gh, gw) = (h.div_ceil(16), w.div_ceil(16));
        let dev = tensor::Device::Cpu;
        // Left half, at full strength against the base.
        let ws = region_weights(h, w, &[(0.0, 0.0, 0.5, 1.0, 1.0)], &dev).expect("weights");
        assert_eq!(ws.len(), 2, "a base plus one region");
        let base = ws[0].to_vec_f32();
        let left = ws[1].to_vec_f32();
        assert_eq!(base.len(), gh * gw);

        for r in 0..gh {
            for c in 0..gw {
                let t = r * gw + c;
                if c < gw / 2 {
                    // Inside: the region and the base split evenly (both weight 1).
                    assert!(
                        (left[t] - 0.5).abs() < 1e-5,
                        "token ({r},{c}) is inside the left half but the region weighs {}",
                        left[t]
                    );
                } else {
                    assert!(
                        left[t] < 1e-6,
                        "token ({r},{c}) is in the RIGHT half yet the left region weighs {} \
                         - the row/column mapping is transposed",
                        left[t]
                    );
                    assert!(
                        (base[t] - 1.0).abs() < 1e-5,
                        "uncovered token lost its base prompt"
                    );
                }
            }
        }
        // A partition: every token's weights sum to exactly 1, so the recombined
        // prediction is an average and never brightens or dims the latent.
        for t in 0..gh * gw {
            let sum = base[t] + left[t];
            assert!((sum - 1.0).abs() < 1e-5, "token {t} sums to {sum}, not 1");
        }
    }

    /// Strength is a ratio against the base, not an absolute.
    #[test]
    fn strength_shifts_the_split_against_the_base_prompt() {
        let dev = tensor::Device::Cpu;
        let ws = region_weights(64, 64, &[(0.0, 0.0, 1.0, 1.0, 3.0)], &dev).expect("weights");
        let (base, region) = (ws[0].to_vec_f32(), ws[1].to_vec_f32());
        // strength 3 against a base of 1 => 3/4 for the region.
        assert!((region[0] - 0.75).abs() < 1e-5, "region got {}", region[0]);
        assert!((base[0] - 0.25).abs() < 1e-5, "base got {}", base[0]);
    }

    /// Overlapping regions share their tokens instead of one silently winning.
    #[test]
    fn overlapping_regions_split_the_shared_tokens() {
        let dev = tensor::Device::Cpu;
        let ws = region_weights(
            64,
            64,
            &[(0.0, 0.0, 1.0, 1.0, 1.0), (0.0, 0.0, 1.0, 1.0, 1.0)],
            &dev,
        )
        .expect("weights");
        let sums: Vec<f32> = (0..ws[0].to_vec_f32().len())
            .map(|t| ws.iter().map(|w| w.to_vec_f32()[t]).sum())
            .collect();
        for (t, s) in sums.iter().enumerate().take(4) {
            assert!((s - 1.0).abs() < 1e-5, "token {t} sums to {s}");
        }
        // base + two equal regions => a third each.
        assert!((ws[1].to_vec_f32()[0] - 1.0 / 3.0).abs() < 1e-5);
    }
}
