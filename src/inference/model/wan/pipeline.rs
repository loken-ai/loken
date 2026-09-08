//! Wan 2.1 video - STAGE 4 capstone: the full text->video pipeline (full-Rust,
//! `crate::tensor`). Wires the validated components end-to-end:
//!   prompt -> umT5 encode -> flow-match CFG Euler denoise (Wan DiT) -> Wan-VAE decode -> frames
//!
//! ## Flow-match scheduler (Wan `wan/configs` + `wan/utils/fm_solvers*`)
//! Wan is a rectified-flow model: the DiT predicts the velocity `v` of the probability-flow
//! ODE. We use a plain flow-match Euler integrator (an accepted drop-in for the reference
//! Flow-UniPC/DPM++ solvers - same ODE, lower-order step). The σ schedule mirrors the
//! diffusers `FlowMatchEulerDiscreteScheduler`: linspace `σ` from 1 -> 1/`num_train_timesteps`
//! over `steps`, then the timestep-SHIFT `σ ← shift.σ/(1+(shift-1).σ)` (Wan `sample_shift=5`).
//! The model timestep is `t = σ.num_train_timesteps` (the DiT's sinusoid takes the unscaled
//! 0..1000 value). The Euler update is `x ← x + (σ_next - σ_curr).v` with an implicit final
//! `σ_next = 0` (predict x0) - `euler_integrate` from the ACE-Step flow loop does exactly this.
//!
//! ## CFG
//! Classifier-free guidance blends a conditional (umT5(prompt)) and an unconditional
//! (umT5("")) velocity: `v = v_uncond + guide.(v_cond - v_uncond)` (Wan `guide_scale≈6`).
//!
//! ## Latent / clip sizing
//! Output `[3, 4.(F-1)+1, 8.Hl, 8.Wl]`; latent `[16, F, Hl, Wl]` with `Hl=height/8`,
//! `Wl=width/8` and `F=(out_frames-1)/4+1`. After the DiT patch (1,2,2) the dense-attention
//! token count is `F.(Hl/2).(Wl/2)` - kept small by using a short low-res clip.
//!
//! ## VRAM orchestration
//! Nothing here is placed by role, by device index or by which checkpoint asked: every
//! component states what it needs - its checkpoint's size on disk plus what its own forward
//! holds - and the fleet gate answers with a card that can carry both, or with the host.
//! umT5 runs once per render, before the denoise, and is dropped as soon as its (small,
//! host-side) contexts exist; the DiT and the VAE are resident and are planned after the
//! encoder's memory has been returned to the driver. Every GPU placement has a CPU fallback
//! (on no-fit or a CUDA OOM) so the render never crashes. [`render_many`] loads each model
//! ONCE and renders a list of prompts in a single process (no per-clip reload/re-encode);
//! [`render`] is the backward-compatible single-prompt wrapper.

use crate::inference::model::umt5::encoder::{umt5_pth, umt5_tokenize, Umt5Encoder};
use crate::inference::model::wan::dit::{WanDit, WanVariant};
use crate::inference::model::wan::vae::{load_wan_vae_decoder, wan_file};
use crate::inference::sample::flow_unipc::heun_integrate;
use crate::inference::serve::progress as ph;
use crate::tensor::{Device, Result, Tensor};

const NUM_TRAIN_TIMESTEPS: f32 = 1000.0;
const LATENT_CH: usize = 16;

/// Decoded RGB frames (denormalized to bytes), ready to write to disk.
pub struct WanFrames {
    /// One `Vec<u8>` per frame, RGB row-major (`r,g,b` per pixel, top-left first).
    pub frames: Vec<Vec<u8>>,
    pub width: usize,
    pub height: usize,
}

/// Flow-match σ schedule (diffusers `FlowMatchEulerDiscreteScheduler`): σ linspace
/// `1 -> 1/num_train_timesteps` over `steps`, then the `shift` timestep-warp. The implicit
/// final σ=0 endpoint is supplied by `euler_integrate`, so it is NOT in the returned vector.
pub fn wan_sigma_schedule(steps: usize, shift: f32) -> Vec<f32> {
    let last = 1.0 / NUM_TRAIN_TIMESTEPS;
    let denom = (steps.max(2) - 1) as f32;
    (0..steps)
        .map(|i| {
            let s = 1.0 + (i as f32 / denom) * (last - 1.0); // 1.0 -> 1/1000
            shift * s / (1.0 + (shift - 1.0) * s)
        })
        .collect()
}

/// The timestep SHIFT this render should use, from the model and the frame size.
///
/// Wan does not use one value: the shift warps the sigma schedule, and how much warping a
/// checkpoint wants depends on what it was trained to produce. Their published
/// configurations are 5.0 for the 14B text-to-video, 8.0 for the 1.3B at 480p-class sizes,
/// and 3.0 for image-to-video at 480p rising to 5.0 at 720p.
///
/// This was 5.0 everywhere, hardcoded. A wrong shift does not fail - it under- or
/// over-resolves the high frequencies, which is what a texture that goes spiky over a clip
/// looks like.
///
/// The size classes are decided on pixel COUNT rather than on a width, because 832x480 and
/// 512x512 are the same class of picture and neither matches a 720p frame.
fn wan_sample_shift(width: usize, height: usize, is_1_3b: bool, is_i2v: bool) -> f32 {
    // Halfway between 832x480 (399k) and 1280x720 (921k).
    const HD_PIXELS: usize = 660_000;
    let hd = width.saturating_mul(height) >= HD_PIXELS;
    match (is_i2v, is_1_3b, hd) {
        (true, _, false) => 3.0,
        (true, _, true) => 5.0,
        (false, true, false) => 8.0,
        _ => 5.0,
    }
}

/// Deterministic N(0,1) latent `[16.F.Hl.Wl]` (channel-major) via a seeded LCG + Box-Muller.
fn gaussian_latent(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = seed.max(1);
    let mut u01 = || {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (((rng >> 33) as f64 / (1u64 << 31) as f64) as f32).clamp(1e-7, 1.0 - 1e-7)
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (u01(), u01());
            (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
        })
        .collect()
}

/// umT5-encode a tokenized prompt -> flat context `[S*4096]` + token count `S` (on the
/// encoder's device). The ids are tokenized by the caller, which needs the longest of them
/// BEFORE it can say what the encode will hold on a card.
fn encode_text(enc: &Umt5Encoder, ids: &[u32]) -> Result<(Vec<f32>, usize)> {
    let ctx = enc.encode(ids)?; // [S, 4096]
    let s = ctx.dims()[0];
    Ok((ctx.flatten_all()?.to_vec_f32(), s))
}

/// One prompt's umT5 context: conditional `(flat [S.4096], S)` plus an optional
/// unconditional `(flat, S)` (present iff CFG > 1; identical for every prompt).
type PromptCtx = (Vec<f32>, usize, Option<(Vec<f32>, usize)>);

/// Substring test for a CUDA out-of-memory error (the fallback trigger).
fn is_oom(e: &crate::tensor::Error) -> bool {
    let s = format!("{e:?}").to_lowercase();
    s.contains("out of memory") || s.contains("out_of_memory")
}

/// Process-global cache of the CPU-staged (half-precision) umT5-XXL weights. Parsing and
/// converting the checkpoint takes far longer than uploading it, so caching it in host RAM
/// lets each `/v1/video` request just upload a fresh GPU copy via `to_device`, leaving the
/// cache intact. Held behind a Mutex (the encoder's RefCell rope-scratch is not Sync) - the
/// lock also serializes the per-request GPU upload, which is desirable: two concurrent Wan
/// renders would each want a full copy of the encoder on the same card. The staged weights
/// are device-independent, so one entry serves every request regardless of where the
/// encoder is placed.
static UMT5_CPU_CACHE: std::sync::OnceLock<std::sync::Mutex<Option<Umt5Encoder>>> =
    std::sync::OnceLock::new();

/// Host-RAM cache of the CPU-staged Wan-VAE decoder (~0.5 GB, ~2 s to load). Shared via
/// `Arc` because the decoder is Send+Sync (no interior mutability) and `decode`/`to_device`
/// take `&self` - the same cached CPU decoder both uploads a GPU copy per request and
/// serves the CPU-decode fallback. No GPU VRAM held between requests. See UMT5_CPU_CACHE.
static WAN_VAE_CPU_CACHE: std::sync::OnceLock<
    std::sync::Mutex<Option<std::sync::Arc<crate::inference::model::wan::vae::WanVaeDecoder>>>,
> = std::sync::OnceLock::new();

/// Register the umT5 f16 host staging with the host-cache registry so RAM-pressured CPU
/// plans can drop it (it re-parses on the next video render). try_lock: never drop while a
/// render holds it.
///
/// The size is the checkpoint's, read from disk. It used to be typed here, and by the time
/// anyone looked the constant and the file had already drifted apart.
fn register_umt5_cache_reclaim() {
    let staged = checkpoint_bytes(&umt5_pth());
    crate::inference::place::vram_manager::register_host_cache(
        "wan-umt5-f16-staging",
        staged,
        Box::new(move || {
            let Some(m) = UMT5_CPU_CACHE.get() else {
                return 0;
            };
            let Ok(mut g) = m.try_lock() else { return 0 };
            if g.take().is_some() {
                staged
            } else {
                0
            }
        }),
    );
}

/// A checkpoint's size, from the file. Every placement in this pipeline is charged this and
/// never a figure typed next to it: a typed one is right until the file changes, and then it
/// is wrong in whichever direction hurts, silently. 0 means the file could not be stat'd.
fn checkpoint_bytes(path: &std::path::Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// What one encode holds on the encoder's device, beyond the weights: the encoder's own
/// geometry applied to this render's LONGEST prompt, because T5 attention keeps a full
/// `seq x seq` map per head and that is the part a weights-only demand misses.
fn umt5_encode_reserve(longest_prompt_tokens: usize) -> u64 {
    use crate::tensor::DType;
    crate::inference::model::umt5::encoder::umt5_encode_peak_bytes(
        longest_prompt_tokens,
        DType::BF16,
    )
}

/// What the text encoder needs from a card in total: its weights, plus what one encode
/// holds. The weights are the checkpoint's own size - it is stored at half precision and
/// that is the precision it runs at on a card, so the file IS the resident footprint.
fn umt5_demand(weights: u64, longest_prompt_tokens: usize) -> u64 {
    weights.saturating_add(umt5_encode_reserve(longest_prompt_tokens))
}

/// Where the text encoder runs: the fastest card that holds its weights AND its encode, else
/// the host.
///
/// This used to send one checkpoint variant to the host by NAME, on the grounds that a
/// transient encoder must not hold device memory while the denoiser splits itself across the
/// cards. That overlap no longer exists: the encoder is dropped when `encode_all` returns and
/// the pools are released before the denoiser is loaded, so by the time the denoiser plans,
/// nothing of the encoder is on any card - see `render_many`.
///
/// What the variant's name never said is the thing that actually decides this: whether a card
/// HERE can hold this checkpoint and this prompt's attention. So that is what is asked. The
/// host remains an ordinary answer rather than a failure - the encoder runs once per render,
/// and with the staging cache warm it costs seconds - which is why no card is a case to
/// handle and not a case to refuse.
fn umt5_device(weights: u64, longest_prompt_tokens: usize) -> Device {
    if weights == 0 {
        // The checkpoint could not be stat'd. Charging a card zero would place the encoder
        // by an amount that is not its size; the loader below reports the real problem.
        return Device::Cpu;
    }
    // The gate takes the resident weights and the run-time reserve separately, so split the
    // demand where it was built: the reserve is the part of it that is not the checkpoint.
    let act = umt5_demand(weights, longest_prompt_tokens).saturating_sub(weights);
    // The encoder is TRANSIENT - encoded once, dropped before the denoiser loads - so it
    // needs no dedicated card, only the one that can take the upload fastest, because moving
    // the weights dominates the encode. So it asks for the fastest that fits first, and only
    // yields the card when that one is full - the host being the last resort. No index, no
    // free-VRAM ranking, no name of a checkpoint.
    let reserve = crate::inference::place::runtime_demand::load_runtime_floor(weights) + act;
    match crate::inference::place::plan::place_whole(weights, reserve) {
        Device::Cpu => crate::inference::place::plan::place_aside(weights, reserve),
        fastest => fastest,
    }
}

/// Load the umT5 encoder ONCE, place it where it fits (host when nothing does, and on a CUDA
/// OOM), and encode EVERY prompt. The encoder is dropped on return - its contexts are small
/// host vectors, so it need not stay resident through the denoise. The unconditional ("")
/// branch is prompt-independent, so it is encoded once and shared across clips.
fn encode_all(
    prompts: &[String],
    cfg: f32,
    negative: &str,
    progress: Option<&ph::ProgressFn<'_>>,
) -> Result<Vec<PromptCtx>> {
    // The encoder is the largest thing a render loads and this is where it spends its first
    // minutes. Reporting it is the difference between a slow load and a wedged one.
    ph::note(progress, ph::phase::LOAD_ENCODER, 0, 0);
    use crate::tensor::DType;
    let path = umt5_pth();
    let path = path.to_str().unwrap();
    let t0 = std::time::Instant::now();
    // The `.pth` is bf16; the CPU reference upcasts to F32, which is twice the file and
    // fits no consumer card, so the GPU path keeps BF16 - the same 2-byte footprint as
    // f16 but with f32's exponent range, which this encoder needs: T5-class activations
    // OVERFLOW f16, an all-F16 GPU forward returns 100% NaN, and that decoded to the
    // all-white-video bug. Validated GPU BF16 against CPU F32 at rel_rms 1.35%. Weights
    // are staged on the host and then moved onto the card; a CUDA OOM falls back to the
    // F32 CPU encode.
    //
    // Size comes from the checkpoint on disk, never a figure typed here: a constant
    // stops tracking the file the moment either changes, and it is the loader that pays.
    let weights = checkpoint_bytes(std::path::Path::new(path));
    // Tokenize BEFORE placing. What the encode holds on a card is set by the longest prompt
    // of the render, so the placement cannot be decided without knowing it - and the tokens
    // are then handed to the encoder rather than recomputed per prompt.
    let ids = prompts
        .iter()
        .map(|p| umt5_tokenize(p))
        .collect::<Result<Vec<Vec<u32>>>>()?;
    let uncond_ids = if cfg > 1.0 {
        Some(umt5_tokenize(negative)?)
    } else {
        None
    };
    let longest = ids
        .iter()
        .chain(uncond_ids.as_ref())
        .map(|v| v.len())
        .max()
        .unwrap_or(0);
    let device = umt5_device(weights, longest);
    let enc = if matches!(device, Device::Cpu) {
        // Gate chose CPU: encode on F32 (the CPU forward's requirement) CONVERTED from the
        // shared half-precision host staging cache - populate it on first use (one parse) and
        // pay an in-RAM upcast per render, instead of re-parsing the whole checkpoint as
        // uncached F32 every time no card has room.
        let mut guard = UMT5_CPU_CACHE
            .get_or_init(|| std::sync::Mutex::new(None))
            .lock()
            .unwrap();
        if guard.is_none() {
            *guard = Some(Umt5Encoder::from_pth_dtype(path, Device::Cpu, DType::BF16)?);
            register_umt5_cache_reclaim();
        }
        let cpu = guard
            .as_ref()
            .unwrap()
            .to_dtype_on(&Device::Cpu, DType::F32)?;
        drop(guard);
        cpu
    } else {
        // Cache the CPU-staged bf16 weights (a direct copy of the checkpoint dtype) in
        // host RAM; each
        // request uploads a fresh GPU copy via to_device, leaving the cache resident. The
        // lock is held across the load (miss) or upload (hit) - see UMT5_CPU_CACHE.
        let mut guard = UMT5_CPU_CACHE
            .get_or_init(|| std::sync::Mutex::new(None))
            .lock()
            .unwrap();
        if guard.is_none() {
            *guard = Some(Umt5Encoder::from_pth_dtype(path, Device::Cpu, DType::BF16)?);
            register_umt5_cache_reclaim();
        }
        match guard.as_ref().unwrap().to_device(&device) {
            Ok(e) => e, // GPU copy; the cached host staging stays resident
            Err(e) if is_oom(&e) => {
                // The gate over-estimated the card's room: convert the ALREADY-CACHED host
                // staging to the F32 the CPU forward requires (one in-RAM conversion, seconds)
                // instead of re-parsing the whole checkpoint as uncached F32 per render.
                eprintln!("[ace-wan-umt5] move to {device:?} OOM - encoding on CPU (F32 from the cached f16 staging)");
                let cpu = guard
                    .as_ref()
                    .unwrap()
                    .to_dtype_on(&Device::Cpu, DType::F32)?;
                drop(guard);
                cpu
            }
            Err(e) => return Err(e),
        }
    };
    let load_s = t0.elapsed().as_secs_f32();
    let placed = enc.device();
    let t1 = std::time::Instant::now(); // encode-forward only (load excluded)
                                        // The branch guidance pushes AWAY from. It was pinned to the empty string, so a
                                        // caller naming what to avoid - a watermark, a warped hand - was conditioning
                                        // against nothing and got no word about it. Empty stays the default: it is the
                                        // plain unconditional branch this model was trained against.
    ph::note(progress, ph::phase::ENCODE, 0, 0);
    let uncond = match &uncond_ids {
        Some(t) => Some(encode_text(&enc, t)?),
        None => None,
    };
    let mut out = Vec::with_capacity(prompts.len());
    for (i, tokens) in ids.iter().enumerate() {
        let (cond, s_cond) = encode_text(&enc, tokens)?;
        eprintln!(
            "[wan] umT5 encode clip {}/{}: prompt S={s_cond}{}",
            i + 1,
            prompts.len(),
            uncond
                .as_ref()
                .map(|(_, s)| format!(", uncond S={s}"))
                .unwrap_or_default()
        );
        out.push((cond, s_cond, uncond.clone()));
    }
    eprintln!(
        "[wan] umT5 on {placed:?}: load {load_s:.1}s + encode {} prompt(s) {:.2}s",
        prompts.len(),
        t1.elapsed().as_secs_f32()
    );
    Ok(out)
}

/// Denormalize a decoded RGB tensor `[3, Tout, H, W]` (clamped [-1,1]) into per-frame
/// row-major byte buffers.
fn pack_frames(rgb: &Tensor) -> Result<WanFrames> {
    let d = rgb.dims().to_vec(); // [3, Tout, H, W]
    let (tout, oh, ow) = (d[1], d[2], d[3]);
    let buf = rgb.flatten_all()?.to_vec_f32();
    let plane = oh * ow;
    let chan = tout * plane; // elements per color channel
    let denorm = |v: f32| (((v.clamp(-1.0, 1.0) + 1.0) * 0.5 * 255.0).round()) as u8;
    let mut frames = Vec::with_capacity(tout);
    for f in 0..tout {
        let mut px = vec![0u8; plane * 3];
        for y in 0..oh {
            for xx in 0..ow {
                let p = y * ow + xx;
                for c in 0..3 {
                    px[p * 3 + c] = denorm(buf[c * chan + f * plane + p]);
                }
            }
        }
        frames.push(px);
    }
    Ok(WanFrames {
        frames,
        width: ow,
        height: oh,
    })
}

/// Backward-compatible single-prompt render (delegates to [`render_many`]).
/// Returns the decoded RGB frames (denormalized to bytes). `out_frames` is the requested
/// output length; the true count is `4.(F-1)+1` where `F=(out_frames-1)/4+1`.
/// `height`/`width` must be multiples of 16 (8x VAE x 2x patch).
pub fn render(
    prompt: &str,
    out_frames: usize,
    height: usize,
    width: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    variant: WanVariant,
) -> Result<WanFrames> {
    render_with_progress(
        prompt, out_frames, height, width, steps, cfg, seed, variant, None, None,
    )
}

/// Like [`render`], plus a phase reporter `progress(phase, done, total)` for streaming UIs.
pub fn render_with_progress(
    prompt: &str,
    out_frames: usize,
    height: usize,
    width: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    variant: WanVariant,
    progress: Option<&crate::inference::serve::progress::ProgressFn<'_>>,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> Result<WanFrames> {
    render_sampled(
        prompt,
        out_frames,
        height,
        width,
        steps,
        cfg,
        seed,
        variant,
        WanSampler::default_for(height, width),
        progress,
        cancel,
    )
}

/// Per-request sampler choice. UniPC is the reference pipeline's default (order-2 multistep,
/// one model call per step); Heun (two calls per step) is kept for A/B comparison.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WanSampler {
    UniPc,
    Heun,
}

impl WanSampler {
    /// Measured default per resolution regime. At/above the model's native training scale the
    /// reference UniPC is strictly better (fixed a full-frame saturation at 512x512x49 that
    /// Heun could not); far below it the velocity field is noisy and UniPC's multistep
    /// extrapolation amplifies that noise into abstract artifacts, while Heun's within-step
    /// averaging smooths it (A/B at 256x256: Heun coherent, UniPC degraded).
    pub fn default_for(height: usize, width: usize) -> Self {
        if height * width >= 448 * 448 {
            Self::UniPc
        } else {
            Self::Heun
        }
    }
}

/// [`render_with_progress`] with an explicit sampler.
pub fn render_sampled(
    prompt: &str,
    out_frames: usize,
    height: usize,
    width: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    variant: WanVariant,
    sampler: WanSampler,
    progress: Option<&crate::inference::serve::progress::ProgressFn<'_>>,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> Result<WanFrames> {
    let mut clips = render_many_sampled(
        &[prompt.to_string()],
        out_frames,
        height,
        width,
        steps,
        cfg,
        seed,
        variant,
        sampler,
        progress,
        cancel,
        "",
        None,
        0.0,
        None,
    )?;
    Ok(clips.pop().expect("render_many yields one clip per prompt"))
}

/// Multi-clip text->video render: load umT5/DiT/VAE ONCE and render every prompt in a single
/// process (no per-clip reload or re-encode). Returns one [`WanFrames`] per prompt, in order.
/// Each clip's initial noise is seeded `seed + clip_index` so identical prompts still differ
/// (clip 0 reproduces the single-prompt [`render`] output bit-for-bit).
pub fn render_many(
    prompts: &[String],
    out_frames: usize,
    height: usize,
    width: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    variant: WanVariant,
) -> Result<Vec<WanFrames>> {
    render_many_with_progress(
        prompts, out_frames, height, width, steps, cfg, seed, variant, None, None,
    )
}

/// Like [`render_many`], plus a phase reporter `progress(phase, done, total)`.
pub fn render_many_with_progress(
    prompts: &[String],
    out_frames: usize,
    height: usize,
    width: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    variant: WanVariant,
    progress: Option<&crate::inference::serve::progress::ProgressFn<'_>>,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> Result<Vec<WanFrames>> {
    render_many_sampled(
        prompts,
        out_frames,
        height,
        width,
        steps,
        cfg,
        seed,
        variant,
        WanSampler::default_for(height, width),
        progress,
        cancel,
        "",
        None,
        0.0,
        None,
    )
}

/// RGB8 at any size -> `[3, h, w]` in [-1,1], channel-major, bilinear.
///
/// The VAE reads the render's own resolution: a reference encoded at another scale puts
/// every feature in the wrong latent place, and the result looks like a different shot
/// rather than a continuation.
fn resize_rgb_to(rgb: &[u8], sw: usize, sh: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; 3 * h * w];
    for y in 0..h {
        let sy = ((y as f32 + 0.5) * sh as f32 / h as f32 - 0.5).clamp(0.0, (sh - 1) as f32);
        let (y0, fy) = (sy.floor() as usize, sy - sy.floor());
        let y1 = (y0 + 1).min(sh - 1);
        for x in 0..w {
            let sx = ((x as f32 + 0.5) * sw as f32 / w as f32 - 0.5).clamp(0.0, (sw - 1) as f32);
            let (x0, fx) = (sx.floor() as usize, sx - sx.floor());
            let x1 = (x0 + 1).min(sw - 1);
            for c in 0..3 {
                let at = |yy: usize, xx: usize| rgb[(yy * sw + xx) * 3 + c] as f32 / 255.0;
                let top = at(y0, x0) * (1.0 - fx) + at(y0, x1) * fx;
                let bot = at(y1, x0) * (1.0 - fx) + at(y1, x1) * fx;
                out[c * h * w + y * w + x] = (top * (1.0 - fy) + bot * fy) * 2.0 - 1.0;
            }
        }
    }
    out
}

/// Denoise a clip as a CHAIN of chunks, each continuing the last.
///
/// This is what windows were always trying to be. A window blends its overlap with its
/// neighbour, and a blend between two independently generated scenes is a morph - the thing
/// that made the long renders unusable. A chunk here does not blend: it is handed the
/// previous chunk's last latent frame as an INPUT, on the twenty conditioning channels the
/// checkpoint was trained to read, so it continues rather than being averaged.
///
/// The CLIP embedding is the same for every chunk and is the render's semantic anchor; only
/// the latent reference advances. That is deliberate - it is what keeps a long clip about
/// the same thing instead of walking a little further away with each chunk.
///
/// Chunk 0 produces the whole window; every later one regenerates the frame it continues
/// and contributes the rest.
fn denoise_i2v_chunks(
    dit: &WanDit,
    clip: &[f32],
    z0: &[f32],
    empty: &[f32],
    f_lat: usize,
    hl: usize,
    wl: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    sampler: WanSampler,
    step_reuse: f32,
    shift: f32,
    cond: &[f32],
    s_cond: usize,
    uncond: Option<&(Vec<f32>, usize)>,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    on_pass: &dyn Fn(),
) -> Result<Vec<f32>> {
    use crate::inference::model::flux::sampling::StepReuse;
    use crate::inference::model::wan::dit::RefFrame;
    use crate::tensor::Error;
    let win = WanDit::NATIVE_LATENT_FRAMES.min(f_lat.max(1));
    let plane = hl * wl;
    let mut clean = vec![0f32; LATENT_CH * f_lat * plane];
    let mut reference: Vec<f32> = z0.to_vec();
    let mut at = 0usize;
    let mut chunk = 0usize;
    while at < f_lat {
        let cond20 = i2v_cond20(&reference, empty, win, plane);
        let refframe = RefFrame {
            clip,
            cond20: &cond20,
        };
        let mut xw = gaussian_latent(LATENT_CH * win * plane, seed.wrapping_add(chunk as u64));
        let (mut reuse_c, mut reuse_u) = (StepReuse::new(step_reuse), StepReuse::new(step_reuse));
        let mut err: Option<Error> = None;
        let t_chunk = std::time::Instant::now();
        let mut passes = 0usize;
        let mut velocity = |xt: &[f32], t: f32| -> Vec<f32> {
            if err.is_some() {
                return vec![0f32; xt.len()];
            }
            if cancel.is_some_and(|c| c.is_cancelled()) {
                err = Some(Error("wan render cancelled (client disconnected)".into()));
                return vec![0f32; xt.len()];
            }
            let res = (|| -> Result<Vec<f32>> {
                let xt_t = Tensor::from_vec_f32(xt.to_vec(), (xt.len(),))?;
                let v_cond = match reuse_c.reuse(&xt_t, false)? {
                    Some(c) => c.to_vec_f32(),
                    None => {
                        let v = dit.forward_cancellable(
                            xt,
                            win,
                            hl,
                            wl,
                            t,
                            cond,
                            s_cond,
                            cancel,
                            Some(&refframe),
                        )?;
                        reuse_c.observe(&xt_t, &Tensor::from_vec_f32(v.clone(), (v.len(),))?)?;
                        v
                    }
                };
                match uncond {
                    Some((u, su)) => {
                        let v_un = match reuse_u.reuse(&xt_t, false)? {
                            Some(c) => c.to_vec_f32(),
                            None => {
                                let v = dit.forward_cancellable(
                                    xt,
                                    win,
                                    hl,
                                    wl,
                                    t,
                                    u,
                                    *su,
                                    cancel,
                                    Some(&refframe),
                                )?;
                                reuse_u.observe(
                                    &xt_t,
                                    &Tensor::from_vec_f32(v.clone(), (v.len(),))?,
                                )?;
                                v
                            }
                        };
                        Ok(v_un
                            .iter()
                            .zip(&v_cond)
                            .map(|(a, b)| a + cfg * (b - a))
                            .collect())
                    }
                    None => Ok(v_cond),
                }
            })();
            on_pass();
            // In the SERVER log too, not only down a progress channel. The windowed path
            // learned this the hard way - a render that prints its loads and then nothing
            // for a quarter of an hour is indistinguishable from a wedged one - and the
            // chunk path was written without it.
            passes += 1;
            let per = t_chunk.elapsed().as_secs_f32() / passes as f32;
            eprintln!("[wan] i2v chunk {chunk} pass {passes} ({per:.1}s/pass)",);
            match res {
                Ok(v) => v,
                Err(e) => {
                    err = Some(e);
                    vec![0f32; xt.len()]
                }
            }
        };
        match sampler {
            WanSampler::UniPc => {
                crate::inference::sample::flow_unipc::unipc_integrate(
                    &mut xw,
                    steps,
                    shift as f64,
                    velocity,
                )?;
            }
            WanSampler::Heun => {
                let sched = wan_sigma_schedule(steps, shift);
                heun_integrate(&mut xw, &sched, |xt, sigma| {
                    velocity(xt, sigma * NUM_TRAIN_TIMESTEPS)
                });
            }
        }
        if let Some(e) = err {
            return Err(e);
        }
        // Chunk 0 keeps everything; a later one drops its frame 0, which is the frame it was
        // given and therefore already written.
        let skip = if chunk == 0 { 0 } else { 1 };
        let keep = (win - skip).min(f_lat - at);
        for c in 0..LATENT_CH {
            for f in 0..keep {
                let src = (c * win + skip + f) * plane;
                let dst = (c * f_lat + at + f) * plane;
                clean[dst..dst + plane].copy_from_slice(&xw[src..src + plane]);
            }
        }
        at += keep;
        // The next chunk continues the LAST frame written, which is this chunk's own output
        // and not a blend of anything.
        let last = at - 1;
        for c in 0..LATENT_CH {
            let src = (c * f_lat + last) * plane;
            reference[c * plane..(c + 1) * plane].copy_from_slice(&clean[src..src + plane]);
        }
        chunk += 1;
    }
    Ok(clean)
}

/// The conditioning an image-to-video chunk needs, built from the frame it continues.
///
/// `[20, frames, h, w]` channel-major: four mask channels then sixteen latent ones. The mask
/// is 1 on latent frame 0 and 0 everywhere else - which is what Wan's own construction
/// reduces to once its four-way fold is unwound - and the latent sits in frame 0 with the
/// rest left at zero. The model reads the mask to know which frames it is being given, and
/// getting the order or the polarity wrong loads perfectly and renders the wrong video.
fn i2v_cond20(ref_latent: &[f32], empty_latent: &[f32], frames: usize, plane: usize) -> Vec<f32> {
    const MASK_CH: usize = 4;
    let stride = frames * plane;
    let mut out = vec![0f32; (MASK_CH + LATENT_CH) * stride];
    for c in 0..MASK_CH {
        out[c * stride..c * stride + plane].fill(1.0);
    }
    for c in 0..LATENT_CH {
        let dst = (MASK_CH + c) * stride;
        out[dst..dst + plane].copy_from_slice(&ref_latent[c * plane..(c + 1) * plane]);
        // The frames that are NOT given are filled with what the VAE makes of a black
        // frame - not with zeros. Zero is not black in latent space: it is a perfectly
        // ordinary value the model reads as content, and it followed it. The first render
        // kept the reference frame and then went black over the next twenty.
        for f in 1..frames {
            let d = dst + f * plane;
            out[d..d + plane].copy_from_slice(&empty_latent[c * plane..(c + 1) * plane]);
        }
    }
    out
}

/// Like [`render_many_with_progress`], with an explicit [`WanSampler`].
pub fn render_many_sampled(
    prompts: &[String],
    out_frames: usize,
    height: usize,
    width: usize,
    steps: usize,
    cfg: f32,
    seed: u64,
    variant: WanVariant,
    sampler: WanSampler,
    progress: Option<&crate::inference::serve::progress::ProgressFn<'_>>,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    negative: &str,
    // A fine-tune resolved by `wan_dit_file_for`, or None for the variant's
    // own checkpoint. Its geometry is checked before a weight is read.
    checkpoint: Option<&std::path::Path>,
    // Velocity-reuse threshold, 0 = off. Worth twice here what it is on a
    // single-branch sampler: this loop runs the DiT TWICE per step.
    step_reuse: f32,
    // The frame an image-to-video checkpoint continues, as interleaved RGB8
    // with its size. A text-to-video checkpoint ignores it; an image-to-video
    // one cannot render without it, and says so rather than inventing one.
    start_image: Option<(&[u8], usize, usize)>,
) -> Result<Vec<WanFrames>> {
    use crate::tensor::Error;
    if !height.is_multiple_of(16) || !width.is_multiple_of(16) {
        return Err(Error("wan: height/width must be multiples of 16".into()));
    }
    if prompts.is_empty() {
        return Err(Error("wan: no prompts to render".into()));
    }
    // A step-distilled checkpoint carries its own schedule. It was trained to be integrated
    // in a handful of steps with no classifier-free guidance, so running it at the base
    // model's step count and guidance does not render it better - it renders it wrong, and
    // it does so at ten times the cost. The checkpoint is the only thing that knows this,
    // and nothing in the request could be expected to.
    let (steps, cfg) =
        match checkpoint.and_then(crate::inference::model::wan::dit::wan_distilled_defaults) {
            Some((ds, dc)) if ds != steps || (dc - cfg).abs() > f32::EPSILON => {
                eprintln!("[wan] distilled checkpoint: steps {steps} -> {ds}, cfg {cfg} -> {dc}");
                (ds, dc)
            }
            _ => (steps, cfg),
        };
    let shift = wan_sample_shift(width, height, matches!(variant, WanVariant::B1_3), false);
    let f_lat = (out_frames.saturating_sub(1)) / 4 + 1;
    let (hl, wl) = (height / 8, width / 8);
    let tokens = f_lat * (hl / 2) * (wl / 2);
    eprintln!(
        "[wan] {} clip(s); latent [16,{f_lat},{hl},{wl}] -> {tokens} DiT tokens; \
               output ~[3,{},{height},{width}]; steps={steps} cfg={cfg}",
        prompts.len(),
        4 * (f_lat - 1) + 1
    );

    // -- (1) umT5 text encode for ALL prompts, then the encoder is dropped -----
    let ctxs = encode_all(prompts, cfg, negative, progress)?;

    // GIVE THE CARD BACK BEFORE ASKING HOW MUCH IS LEFT.
    //
    // The encoder is dropped inside `encode_all`, but dropping it does not return its
    // memory to the driver: the pool keeps the block for its own reuse, and a probe from
    // another engine reads a card that still looks full. The encoder is the biggest thing
    // this pipeline loads and it is placed fastest-card-first - the same card the DiT wants
    // - so without this trim the DiT plans against a phantom and spills the blocks that did
    // not fit onto the host. Two blocks on the CPU is not a small penalty: each runs a full
    // attention over every token of the clip, on every step, on both CFG branches.
    //
    // This trim is also WHY the encoder is free to take a card at all: it runs here, before
    // the DiT is planned, so the two never hold device memory at the same time.
    #[cfg(feature = "cuda")]
    crate::inference::engine::llm_engine::release_cuda_pools();

    // -- (2) load the DiT ONCE, through the unified HeteroPlan: whole onto one card when
    //    one holds it, split across the cards that do, host for whatever is left. --------
    ph::note(progress, ph::phase::LOAD_MODEL, 0, 0);
    let t1 = std::time::Instant::now();
    // The DiT never sees the whole clip at once: it denoises over windows of its own
    // trained length, so the activation peak the placement has to fit is ONE window's, not
    // the clip's. Sizing it from the clip refuses cards that would have held the work
    // comfortably - and refuses them by more, the longer the clip.
    let win_frames = f_lat.min(WanDit::NATIVE_LATENT_FRAMES);
    let peak_tokens = win_frames * (hl / 2) * (wl / 2);
    if peak_tokens < tokens {
        eprintln!(
            "[wan] denoise peak is one {win_frames}-frame window: {peak_tokens} tokens, \
                   not the clip's {tokens}"
        );
    }
    let dit = WanDit::load_variant(variant, peak_tokens, checkpoint, progress)?;
    eprintln!(
        "[wan] DiT ({variant:?}) loaded in {:.1}s",
        t1.elapsed().as_secs_f32()
    );

    // A machine WITH cards that could not put a single block on one is not a slow render,
    // it is the wrong request for this hardware - and saying so now is the whole point.
    // Started anyway it runs for hours, holds tens of gigabytes of host memory, and is the
    // shape of request that leaves a machine unusable. Refusing names the reason and the
    // geometry that works instead, which is what someone can act on.
    //
    // A host with no card at all is NOT refused: there the processor is the only path, and
    // for the sizes it can carry it is a real one. What is being refused is the SILENT
    // fallback on a machine whose cards simply cannot hold this frame.
    if dit.is_host_only() {
        let cards = crate::inference::place::vram_manager::probe_under_pressure(0);
        if !cards.is_empty() {
            // Tokens scale with the frame's AREA, so the size that fits is found by
            // shrinking until the count is one the cards took - reported rather than
            // guessed at by the caller.
            return Err(Error(format!(
                concat!(
                    "this clip needs {} tokens per denoising pass at {}x{}, and none of ",
                    "the {} card(s) here can hold a single block of the model at that size ",
                    "- every block would run on the processor, which takes hours and holds ",
                    "tens of gigabytes of memory while it does. Render at a smaller frame: ",
                    "the token count falls with the AREA, so halving each side quarters it."
                ),
                tokens,
                width,
                height,
                cards.len()
            )));
        }
    }

    // -- (3) load the VAE ONCE; place it on the GPU alongside the DiT -----------
    // `conv2d` already has a CUDA path, so the conv-decomposed decode needs no new
    // kernels; the decode is TEMPORAL-STREAMING (bounded peak memory) so it co-resides
    // with the DiT. A CUDA OOM is caught and falls back to a CPU decode (never crashes).
    // Host-RAM cached CPU VAE (Arc-shared): loaded once, then every request reuses it for
    // both the per-request GPU upload and the CPU-decode fallback. See WAN_VAE_CPU_CACHE.
    // Both the staging figure below and the placement gate further down are the
    // checkpoint's own size. Typed constants were standing in for it in three places.
    let vae_path = wan_file("Wan2.1_VAE.pth");
    let vae_file = std::fs::metadata(&vae_path).map(|m| m.len()).unwrap_or(0);
    let dec_cpu: std::sync::Arc<crate::inference::model::wan::vae::WanVaeDecoder> = {
        let mut g = WAN_VAE_CPU_CACHE
            .get_or_init(|| std::sync::Mutex::new(None))
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if g.is_none() {
            *g = Some(std::sync::Arc::new(load_wan_vae_decoder(
                vae_path.to_str().unwrap(),
            )?));
            crate::inference::place::vram_manager::register_host_cache(
                "wan-vae-cpu-staging",
                vae_file,
                Box::new(move || {
                    let Some(m) = WAN_VAE_CPU_CACHE.get() else {
                        return 0;
                    };
                    let Ok(mut g) = m.try_lock() else { return 0 };
                    if g.take().is_some() {
                        vae_file
                    } else {
                        0
                    }
                }),
            );
        }
        g.as_ref().unwrap().clone()
    };
    // VAE weights + the CLIP-DERIVED decode peak: the temporal-streaming decode bounds the
    // feature maps to one full-res frame chunk. The widest live conv works on 192 channels at
    // the OUTPUT resolution (the up-path concat stage - observed OOMing as
    // `conv2d [1,192,H,W]` when this estimate assumed 96), with ~4 buffers alive plus the
    // bounded im2col tile.
    let vae_act = (height as u64) * (width as u64) * 192 * 4 * 4 + (256 << 20);
    // Weights only. The gate takes the decode's transients as `vae_act` right above, so a
    // padded weight figure was counting the same headroom twice; the checkpoint is stored
    // at half precision and the ops want F32, hence the doubling and nothing more.
    let vae_weights = vae_file.saturating_mul(2);
    // The decoder runs ONCE per clip; the denoiser runs on every step of every pass and is
    // already resident on the fastest card. Placing the decoder by the same pack-first gate
    // put the two on the same device, and the decode then had to find its feature maps in
    // what the denoiser had left - which is where the out-of-memory came from, and the host
    // fallback that followed cost several times the denoise.
    let vae_device = crate::inference::place::plan::place_aside(
        vae_weights,
        crate::inference::place::runtime_demand::load_runtime_floor(vae_weights) + vae_act,
    );
    let mut dec_gpu = if matches!(vae_device, Device::Cpu) {
        None
    } else {
        match dec_cpu.to_device(&vae_device) {
            Ok(d) => Some(d),
            Err(e) if is_oom(&e) => {
                eprintln!("[ace-wan-vae] move to {vae_device:?} OOM - decoding on CPU");
                None
            }
            Err(e) => return Err(e),
        }
    };

    let n_lat = LATENT_CH * f_lat * hl * wl;

    // -- image-to-video: the frame every chunk is measured against ------------
    //
    // Prepared ONCE, before any clip: the CLIP embedding is the semantic anchor and stays
    // the same for the whole render - which is deliberate. Each chunk's LATENT reference
    // moves forward (it is the previous chunk's last frame, so the continuation is exact),
    // but the semantics do not, so a thirty-second clip keeps being about the same thing
    // rather than drifting a little further with every chunk.
    let i2v = if dit.wants_reference_frame() {
        let Some((rgb, iw, ih)) = start_image else {
            return Err(Error(
                "this is an image-to-video checkpoint: it renders a clip that CONTINUES a \
                 frame, so it needs one. Give it a start image, or pick a text-to-video \
                 checkpoint."
                    .into(),
            ));
        };
        ph::note(progress, ph::phase::LOAD_ENCODER, 0, 0);
        let clip_path = crate::inference::model::wan::vae::wan_file_in(
            "models--Comfy-Org--Wan_2.1_ComfyUI_repackaged",
            "clip_vision_h.safetensors",
        );
        // These two run ONCE, on one image, so they must not take a card from the denoiser -
        // but pinning them to the host is not the only way to ensure that, and it is an
        // expensive one. The choice was made against a full-precision checkpoint that left
        // its cards holding little more than their activation reserve, where a further
        // gigabyte of one-shot weights really was the difference between a render and an
        // out-of-memory. A quantised checkpoint leaves room, so ask rather than assume: the
        // same role-aware gate the decoder uses takes the slowest card that holds this WITH
        // its pass, and returns the host when none does - which keeps the original
        // behaviour exactly in the case the original comment describes.
        //
        // The reserve passed is the decode's, which is generous for a single 224-pixel
        // image. Over-stating it here is safe in a way it is not for the denoiser: this
        // gate only chooses where a one-shot model goes, it cannot push a resident block
        // onto another device.
        let clip_bytes = std::fs::metadata(&clip_path).map(|m| m.len()).unwrap_or(0);
        let clip_resident = clip_bytes.saturating_mul(2);
        let one_shot = crate::inference::place::plan::place_aside(
            clip_resident,
            crate::inference::place::runtime_demand::load_runtime_floor(clip_resident) + vae_act,
        );
        let t_clip = std::time::Instant::now();
        let tower = crate::inference::model::clip::vision::ClipVisionH::load(
            clip_path.to_str().unwrap_or_default(),
            &one_shot,
        )?;
        let clip = tower.embed_image(rgb, iw, ih)?.flatten_all()?.to_vec_f32();
        drop(tower);
        eprintln!(
            "[wan] start-frame CLIP on {one_shot:?} in {:.1}s",
            t_clip.elapsed().as_secs_f32()
        );
        // The start frame's latent, from the same VAE the decode uses. Resized to the
        // render size first: a reference at another scale lands in the wrong place.
        // The ENCODER lives in the remapped safetensors, not in the decoder's `.pth`: it was
        // ported for Qwen-Image-Edit's reference latents and keyed to that file. Same VAE,
        // same latent space, different container - handing it the decoder's path fails on
        // the header, which is what it did the first time.
        let enc_path = crate::config::hf_models_dir()
            .join("qwen-image-vae")
            .join("wan_keyed.safetensors");
        let t_enc = std::time::Instant::now();
        // Same gate, same reason as the CLIP tower above: one pass over one image has no
        // claim on the denoiser's card, but it does not have to run on the host either.
        let enc_bytes = std::fs::metadata(&enc_path).map(|m| m.len()).unwrap_or(0);
        let enc_resident = enc_bytes.saturating_mul(2);
        let enc_dev = crate::inference::place::plan::place_aside(
            enc_resident,
            crate::inference::place::runtime_demand::load_runtime_floor(enc_resident) + vae_act,
        );
        let enc = crate::inference::model::wan::vae::load_wan_vae_encoder_safetensors_on(
            enc_path.to_str().unwrap_or_default(),
            &enc_dev,
        )?;
        let px = resize_rgb_to(rgb, iw, ih, width, height);
        let img = Tensor::from_vec_f32(px, (3, 1, height, width))?.to_device(&enc_dev)?;
        let z = enc.encode(&img)?; // [16, 1, hl, wl]
                                   // What the VAE makes of the frames the model is NOT given.
                                   //
                                   // ZERO in pixel space, which on a [-1,1] image is mid-GREY - not black. Wan pads its
                                   // reference video with `torch.zeros`, and zeros there mean grey; padding with -1
                                   // instead gives the model a run of black frames to continue, and it does, going dark
                                   // over the clip. Measured: black degraded from frame 8, grey does not.
        let pad = Tensor::from_vec_f32(vec![0.0f32; 3 * height * width], (3, 1, height, width))?
            .to_device(&enc_dev)?;
        let z_black = enc.encode(&pad)?;
        drop(enc);
        eprintln!(
            "[wan] start-frame VAE encode on {enc_dev:?} in {:.1}s",
            t_enc.elapsed().as_secs_f32()
        );
        Some((
            clip,
            z.flatten_all()?.to_vec_f32(),
            z_black.flatten_all()?.to_vec_f32(),
        ))
    } else {
        None
    };

    // -- (4) per clip: denoise (DiT) -> decode (VAE) -> pack frames ---------------
    let mut clips = Vec::with_capacity(prompts.len());
    use crate::inference::model::flux::sampling::StepReuse;
    for (ci, (cond, s_cond, uncond)) in ctxs.iter().enumerate() {
        let (mut reuse_c, mut reuse_u) = (StepReuse::new(step_reuse), StepReuse::new(step_reuse));
        let tc = std::time::Instant::now();
        // Say the denoise has STARTED before doing any of it. A phase that is only reported
        // once its first step completes leaves the client showing the previous phase for as
        // long as that step takes - minutes on a 14B, and indistinguishable from a hang.
        // Count in WINDOWS rather than steps: one step of a long clip is several DiT passes,
        // and a bar that moves once a step does not move.
        let per_forward = WanDit::window_count(f_lat, dit.width());
        let branches = if cfg > 1.0 { 2 } else { 1 };
        // Count the WHOLE render, not this clip. A caller shown a bar that restarts at every
        // scene cannot tell how far along it is, and one that accumulates on its own side
        // drifts from the total - which is what put the count PAST it. Only here are both
        // the clip's index and the number of clips known, so only here can the two agree.
        let per_clip = steps * branches * per_forward;
        let total_windows = per_clip * ctxs.len().max(1);
        let base = ci * per_clip;
        let done_windows = std::cell::Cell::new(0usize);
        ph::note(progress, ph::phase::DENOISE, base, total_windows);
        let on_window = || {
            done_windows.set(done_windows.get() + 1);
            ph::note(
                progress,
                ph::phase::DENOISE,
                base + done_windows.get(),
                total_windows,
            );
        };
        // An image-to-video checkpoint renders the clip as a CHAIN, each chunk continuing
        // the last, which is the only construction that has no seam to blend and therefore
        // nothing to morph between. Everything else keeps the windowed path.
        let x = if let Some((clip, z0, z_black)) = &i2v {
            // Image-to-video has its own shift, and it is not the text path's.
            let i2v_shift = wan_sample_shift(width, height, false, true);
            eprintln!("[wan] i2v schedule: {steps} steps, cfg {cfg}, shift {i2v_shift}");
            denoise_i2v_chunks(
                &dit,
                clip,
                z0,
                z_black,
                f_lat,
                hl,
                wl,
                steps,
                cfg,
                seed.wrapping_add(ci as u64),
                sampler,
                step_reuse,
                i2v_shift,
                cond,
                *s_cond,
                uncond.as_ref(),
                cancel,
                &on_window,
            )?
        } else {
            eprintln!("[wan] schedule: {steps} steps, cfg {cfg}, shift {shift}");
            let mut x = gaussian_latent(n_lat, seed.wrapping_add(ci as u64));
            let mut err: Option<Error> = None;
            let mut step_i = 0usize;
            // Shared velocity closure for either sampler: CFG-combined model output at (x, t).
            let mut velocity = |xt: &[f32], t: f32| -> Vec<f32> {
                if err.is_some() {
                    return vec![0f32; xt.len()];
                }
                // Cooperative cancellation: the client is gone - skip every remaining forward
                // (the loop drains in microseconds) and surface the error after the integrate.
                if cancel.is_some_and(|c| c.is_cancelled()) {
                    err = Some(Error("wan render cancelled (client disconnected)".into()));
                    return vec![0f32; xt.len()];
                }
                let res = (|| -> Result<Vec<f32>> {
                    // A step whose velocity has stopped moving is re-used instead of recomputed.
                    // ONE cache per CFG branch: the two predictions evolve at different rates,
                    // so a shared cache would re-use one branch on the other's evidence. The
                    // sampler drives `xt` as a host vector, so it is wrapped for the comparison
                    // the cache makes - a copy of the latent, against a whole DiT forward.
                    let last = step_i + 1 >= steps;
                    let xt_t = Tensor::from_vec_f32(xt.to_vec(), (xt.len(),))?;
                    let v_cond = match reuse_c.reuse(&xt_t, last)? {
                        Some(c) => c.to_vec_f32(),
                        None => {
                            let v = dit.forward_windowed(
                                xt,
                                f_lat,
                                hl,
                                wl,
                                t,
                                cond,
                                *s_cond,
                                cancel,
                                Some(&on_window),
                                None,
                            )?;
                            reuse_c
                                .observe(&xt_t, &Tensor::from_vec_f32(v.clone(), (v.len(),))?)?;
                            v
                        }
                    };
                    match uncond {
                        Some((u, su)) => {
                            let v_un = match reuse_u.reuse(&xt_t, last)? {
                                Some(c) => c.to_vec_f32(),
                                None => {
                                    let v = dit.forward_windowed(
                                        xt,
                                        f_lat,
                                        hl,
                                        wl,
                                        t,
                                        u,
                                        *su,
                                        cancel,
                                        Some(&on_window),
                                        None,
                                    )?;
                                    reuse_u.observe(
                                        &xt_t,
                                        &Tensor::from_vec_f32(v.clone(), (v.len(),))?,
                                    )?;
                                    v
                                }
                            };
                            Ok(v_un
                                .iter()
                                .zip(&v_cond)
                                .map(|(a, b)| a + cfg * (b - a))
                                .collect())
                        }
                        None => Ok(v_cond),
                    }
                })();
                step_i += 1;
                // Also say it in the SERVER log, not only down a progress channel. The
                // non-streaming endpoint has no reporter to give, so a long render printed the
                // encoder and the DiT load and then nothing at all - half an hour of silence in
                // which a wedged render and a slow one read exactly the same, on the server side
                // as well as the client's. One line per step, with what a step has cost so far,
                // is enough to tell them apart and to project the end.
                let per_step = tc.elapsed().as_secs_f32() / step_i as f32;
                eprintln!(
                    "[wan] clip {}/{} denoise step {step_i}/{steps} ({per_step:.1}s/step, ~{:.0}s left)",
                    ci + 1, prompts.len(), per_step * (steps - step_i.min(steps)) as f32
                );
                match res {
                    Ok(v) => v,
                    Err(e) => {
                        err = Some(e);
                        vec![0f32; xt.len()]
                    }
                }
            };
            match sampler {
                // Official-default: FlowUniPC (order-2 multistep, one model call per step).
                WanSampler::UniPc => {
                    crate::inference::sample::flow_unipc::unipc_integrate(
                        &mut x,
                        steps,
                        shift as f64,
                        velocity,
                    )?;
                }
                // Order-2 with a second model call per step; kept for A/B comparison.
                WanSampler::Heun => {
                    let sched = wan_sigma_schedule(steps, shift);
                    heun_integrate(&mut x, &sched, |xt, sigma| {
                        velocity(xt, sigma * NUM_TRAIN_TIMESTEPS)
                    });
                }
            }
            if let Some(e) = err {
                return Err(e);
            }
            x
        };

        // Decode this clip's latent on the resident GPU VAE (CPU fallback on OOM; on OOM
        // the GPU decoder is dropped so later clips don't re-try and re-OOM). A dead
        // client cancels BEFORE and DURING the decode - it used to run to completion
        // (minutes of CPU) for nobody.
        if cancel.is_some_and(|c| c.is_cancelled()) {
            return Err(Error("wan render cancelled (client disconnected)".into()));
        }
        ph::note(progress, ph::phase::DECODE, 0, 0);
        // GIVE THE CARD BACK BEFORE THE DECODE, for the same reason the text encoder does
        // it before the denoiser is placed. The denoise just released several gigabytes of
        // activations, but releasing them does not return them to the driver - the pool
        // keeps the blocks for its own reuse, so the decoder allocates against a card that
        // still looks full, hits an out-of-memory, and hands the clip to the host. That
        // fallback works and it is slow: it turned a clip whose denoise took a hundred
        // seconds into one that spent ten times that in the decode.
        #[cfg(feature = "cuda")]
        crate::inference::engine::llm_engine::release_cuda_pools();
        let latent = Tensor::from_vec_f32(x, (LATENT_CH, f_lat, hl, wl))?;
        // Come BACK to the GPU when room returns. One clip OOMing used to disable the
        // GPU decoder for the rest of the render, so a single tight moment - another
        // engine holding the card for a few seconds - cost every remaining clip a host
        // decode, minutes each, long after the memory was free again. The CPU decoder
        // stays resident either way, so returning is a weight copy, which is nothing
        // against one clip decoded on the host. Requires DOUBLE the estimated need, so
        // a card hovering at the threshold cannot make this thrash back and forth.
        if dec_gpu.is_none() && !matches!(vae_device, Device::Cpu) {
            let free_now = crate::inference::place::vram_manager::probe(0)
                .into_iter()
                .find(|(_, _, d)| Device::same_device(d, &vae_device))
                .map(|(_, free, _)| free)
                .unwrap_or(0);
            if free_now >= (vae_act + vae_weights) * 2 {
                // A failed move leaves the decoder on the CPU until the next clip.
                if let Ok(d) = dec_cpu.to_device(&vae_device) {
                    eprintln!(
                        "[ace-wan-vae] {:.1} GB free again on {vae_device:?} - moving the \
                             decoder back to the GPU",
                        free_now as f64 / 1e9
                    );
                    dec_gpu = Some(d);
                }
            }
        }
        // Say how far the decode has got. It walks the clip one latent frame at a time and
        // always knew, but only ever announced that it had STARTED - so on a long clip the
        // only thing anyone could see was the word "decoding", for minutes, which is
        // indistinguishable from a hang and was reported as one.
        let on_dec = |done: usize, total: usize| {
            ph::note(progress, ph::phase::DECODE, done, total);
        };
        let on_dec: &dyn Fn(usize, usize) = &on_dec;
        let rgb = match &dec_gpu {
            None => dec_cpu.decode_reporting(&latent, cancel, Some(on_dec))?,
            Some(dec) => {
                let dev = dec.device();
                let lat = latent.to_device(&dev)?;
                match dec.decode_reporting(&lat, cancel, Some(on_dec)) {
                    Ok(r) => r,
                    Err(e) if is_oom(&e) => {
                        eprintln!(
                            "[ace-wan-vae] decode OOM on {dev:?} - freeing the GPU \
                                   decoder for this clip; the next one re-checks the card"
                        );
                        drop(lat);
                        // Frees the GPU weights so this clip can finish on the host. NOT a
                        // permanent decision: the check at the top of the loop moves the
                        // decoder back as soon as the card has room again.
                        dec_gpu = None;
                        dec_cpu.decode_reporting(&latent, cancel, Some(on_dec))?
                    }
                    Err(e) => return Err(e),
                }
            }
        };
        let frames = pack_frames(&rgb)?;
        eprintln!(
            "[wan] clip {}/{} ({} frames) in {:.1}s",
            ci + 1,
            prompts.len(),
            frames.frames.len(),
            tc.elapsed().as_secs_f32()
        );
        clips.push(frames);
    }
    Ok(clips)
}

/// The text encoder's placement, decided without a card and without a server.
///
/// It used to be decided by the checkpoint's NAME: one variant went to the host
/// unconditionally, and the amount that stood for the encoder was whatever someone typed.
/// Neither says anything about the machine the render is on, and both were wrong the moment
/// it changed - a card that could have held the encoder ran it on the processor, and a
/// figure that had stopped tracking its own file sent it to a card that could not.
///
/// So the decision is now a question about the cards, and these are its three answers: no
/// card, one card, several. A byte count typed back into the placement path fails the
/// repo-wide gate in `runtime_demand.rs` (which lists this file); a card index or a
/// checkpoint name typed back into it fails here.
#[cfg(test)]
mod umt5_placement_tests {
    use super::{checkpoint_bytes, umt5_demand, umt5_device};
    use crate::tensor::Device;

    /// The choice `umt5_device` delegates to, reproduced without a GPU: the fleet planner,
    /// given what each card has left over after its own reserve, fastest card first.
    fn planned_card(usable_fastest_first: &[u64], demand: u64) -> Option<usize> {
        use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan};
        let budget: Vec<(usize, u64)> = usable_fastest_first.iter().copied().enumerate().collect();
        let plan = HeteroPlan::calculate_with_kv_reserve(1, demand, &budget, &[], 1.0, 0, 0);
        match plan.segments.first().map(|s| s.kind) {
            Some(DeviceKind::Cuda(i)) => Some(i),
            _ => None,
        }
    }

    /// The size charged for the encoder is the file's, measured, not stated.
    #[test]
    fn the_encoder_is_charged_what_its_file_weighs() {
        let dir = std::env::temp_dir().join(format!("wan-umt5-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir");
        let f = dir.join("checkpoint.pth");
        std::fs::write(&f, vec![0u8; 4096]).expect("write the fake checkpoint");
        assert_eq!(
            checkpoint_bytes(&f),
            4096,
            "the size must come off the file"
        );
        std::fs::write(&f, vec![0u8; 8192]).expect("grow the fake checkpoint");
        assert_eq!(
            checkpoint_bytes(&f),
            8192,
            "and must follow it when it changes"
        );
        std::fs::remove_file(&f).ok();
        assert_eq!(
            checkpoint_bytes(&f),
            0,
            "a checkpoint that cannot be read is charged nothing, not a guess"
        );
        std::fs::remove_dir(&dir).ok();
        // ... and an uncharged checkpoint is never placed on a card: the loader must be the
        // one to report the missing file, with the path in it.
        assert!(matches!(umt5_device(0, 128), Device::Cpu));
    }

    /// The demand has to move with the request, because the attention does: T5 keeps a
    /// `seq x seq` map per head, so a placement sized for a short prompt is not sized for a
    /// long one. Linear growth here would mean the encode was not being charged at all.
    #[test]
    fn the_demand_grows_with_the_prompt_not_just_the_file() {
        let weights = 8192u64;
        assert!(
            umt5_demand(weights, 0) >= weights,
            "the weights are always charged"
        );
        let short = umt5_demand(weights, 128) - weights;
        let long = umt5_demand(weights, 256) - weights;
        assert!(
            long > 3 * short,
            "doubling the prompt must roughly quadruple the encode's footprint: \
             {short} -> {long}"
        );
        assert!(umt5_demand(weights, 1) > umt5_demand(weights, 0));
    }

    /// No card. The host is an answer, not a failure - the encoder runs once per render.
    #[test]
    fn with_no_card_the_encoder_runs_on_the_host() {
        assert_eq!(planned_card(&[], umt5_demand(11_000_000_000, 256)), None);
    }

    /// One card: it holds the whole demand or it does not, and nothing else decides.
    #[test]
    fn with_one_card_the_answer_is_whether_it_holds_the_whole_demand() {
        let demand = umt5_demand(11_000_000_000, 256);
        assert_eq!(planned_card(&[demand + 1], demand), Some(0));
        assert_eq!(
            planned_card(&[demand - 1], demand),
            None,
            "a card that cannot hold the encode must not be given it"
        );
    }

    /// Several cards: the FASTEST that fits, which is neither the first index nor the
    /// emptiest card. Both of those were shipped before, and both put a model where it
    /// could not run.
    #[test]
    fn with_several_cards_the_fastest_that_fits_takes_it() {
        let demand = umt5_demand(11_000_000_000, 256);
        // The list is fastest-first. The fastest card fits, so it wins even though a slower
        // card has far more room free.
        assert_eq!(planned_card(&[demand + 1, demand * 4], demand), Some(0));
        // The fastest cannot hold it: the next one that can takes it - by fit, not by index.
        assert_eq!(
            planned_card(&[demand / 2, demand + 1, demand * 4], demand),
            Some(1)
        );
        // None of them can: the host, even with three cards present.
        assert_eq!(
            planned_card(&[demand / 2, demand / 2, demand / 3], demand),
            None
        );
    }

    /// The gate for the defect itself: a device index or a checkpoint name back in the
    /// placement. Byte counts are covered repo-wide by `runtime_demand.rs`, which lists this
    /// file; what that gate cannot see is a decision taken on a NAME.
    #[test]
    fn the_placement_names_no_card_and_no_checkpoint() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/inference/model/wan/pipeline.rs");
        let src = std::fs::read_to_string(&path).expect("the pipeline source");
        let from = src
            .find("fn umt5_device")
            .expect("umt5_device moved - point this gate at the placement again");
        let to = from
            + src[from..]
                .find("fn pack_frames")
                .expect("pack_frames moved - point this gate at the end of the placement");
        let offenders: Vec<&str> = src[from..to]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .filter(|l| {
                l.contains("WanVariant")
                    || l.contains("Cuda(0")
                    || l.contains("Cuda(1")
                    || l.contains("ordinal(")
                    || l.contains("new_device")
            })
            .collect();
        assert!(
            offenders.is_empty(),
            "the text encoder is being placed by a card index or by which checkpoint asked, \
             instead of by what the cards can hold:\n{}",
            offenders.join("\n")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_schedule_shape() {
        let s = wan_sigma_schedule(50, 5.0);
        assert_eq!(s.len(), 50);
        // monotone decreasing, starts at the shifted σ=1 (==1.0), strictly positive.
        assert!(
            (s[0] - 1.0).abs() < 1e-6,
            "first σ should map 1->1, got {}",
            s[0]
        );
        for w in s.windows(2) {
            assert!(w[0] > w[1], "σ must decrease: {} !> {}", w[0], w[1]);
        }
        assert!(
            *s.last().unwrap() > 0.0,
            "last σ must stay > 0 (final 0 is implicit)"
        );
    }
}

#[cfg(test)]
mod i2v_tests {
    use super::i2v_cond20;

    /// The mask says exactly one thing: latent frame 0 is given, nothing else is. All four
    /// channels carry it, because the checkpoint's four-way fold puts the first video
    /// frame's four repeats there and zeros in every later group.
    #[test]
    fn the_mask_marks_only_the_first_latent_frame() {
        let (frames, plane) = (5usize, 4usize);
        let refl = vec![7.0f32; super::LATENT_CH * plane];
        let empty = vec![0f32; super::LATENT_CH * plane];
        let c = i2v_cond20(&refl, &empty, frames, plane);
        let stride = frames * plane;
        assert_eq!(c.len(), 20 * stride);
        for ch in 0..4 {
            for f in 0..frames {
                let v = &c[ch * stride + f * plane..][..plane];
                let want = if f == 0 { 1.0 } else { 0.0 };
                assert!(v.iter().all(|x| *x == want), "mask ch{ch} frame{f}: {v:?}");
            }
        }
    }

    /// The reference latent sits in frame 0 and nowhere else - the model is being told what
    /// the clip continues FROM, not handed a whole video.
    #[test]
    fn the_reference_latent_occupies_only_the_first_frame() {
        let (frames, plane) = (4usize, 3usize);
        let refl: Vec<f32> = (0..super::LATENT_CH * plane)
            .map(|i| i as f32 + 1.0)
            .collect();
        let empty = vec![0f32; super::LATENT_CH * plane];
        let c = i2v_cond20(&refl, &empty, frames, plane);
        let stride = frames * plane;
        for ch in 0..super::LATENT_CH {
            let got = &c[(4 + ch) * stride..][..plane];
            assert_eq!(
                got,
                &refl[ch * plane..(ch + 1) * plane],
                "latent ch{ch} frame 0"
            );
            for f in 1..frames {
                let v = &c[(4 + ch) * stride + f * plane..][..plane];
                assert!(
                    v.iter().all(|x| *x == 0.0),
                    "latent ch{ch} frame{f} not empty"
                );
            }
        }
    }
}

#[cfg(test)]
mod shift_tests {
    use super::wan_sample_shift;

    /// Wan's published configurations, which is the only thing that makes a shift right.
    #[test]
    fn each_checkpoint_gets_the_shift_it_was_published_with() {
        // 14B text-to-video, any size.
        assert_eq!(wan_sample_shift(832, 480, false, false), 5.0);
        assert_eq!(wan_sample_shift(1280, 720, false, false), 5.0);
        // 1.3B text-to-video at 480p-class.
        assert_eq!(wan_sample_shift(832, 480, true, false), 8.0);
        assert_eq!(wan_sample_shift(512, 512, true, false), 8.0);
        // image-to-video: 3.0 at 480p, 5.0 at 720p.
        assert_eq!(wan_sample_shift(832, 480, false, true), 3.0);
        assert_eq!(wan_sample_shift(1280, 720, false, true), 5.0);
    }

    /// The class is decided on pixel COUNT: a square render and a wide one of the same area
    /// are the same kind of picture, and a width threshold would split them.
    #[test]
    fn the_size_class_follows_the_area_not_the_width() {
        // 720x720 (518k) is 480p-class; 1024x1024 (1.05M) is not.
        assert_eq!(wan_sample_shift(720, 720, false, true), 3.0);
        assert_eq!(wan_sample_shift(1024, 1024, false, true), 5.0);
    }
}

#[cfg(test)]
mod progress_total_tests {
    use crate::inference::model::wan::dit::WanDit;

    // The two paths walk different units, and a progress total that counts the wrong one

    /// The window count must rise with the clip. A count that saturates makes a long
    /// render look nearly finished from the moment it starts.
    #[test]
    fn the_window_count_grows_with_the_clip() {
        let w = |f| WanDit::window_count(f, 5120);
        assert!(w(121) < w(241) && w(241) < w(481));
    }
}
