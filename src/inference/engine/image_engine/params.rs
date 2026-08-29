//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

/// Image generation parameters
#[derive(Debug, Clone)]
pub struct ImageGenParams {
    pub height: usize,
    pub width: usize,
    pub num_steps: usize,
    pub guidance: f64,
    pub seed: Option<u64>,
    /// Input image for img2img (base64-encoded PNG/JPEG)
    pub input_image: Option<String>,
    /// Strength for img2img: 0.0 = keep original, 1.0 = full txt2img (default 0.75)
    pub strength: f64,
    /// FLUX Kontext INSTRUCTION editing: use `input_image` as a fixed context condition
    /// (concatenated tokens), NOT as an img2img init latent. Requires a Kontext checkpoint.
    pub kontext: bool,
    /// Cooperative cancellation: generation loops poll this once per step and bail when the
    /// request's client disconnected (the handler's CancelGuard fires on future drop) -
    /// orphaned renders no longer burn CPU/GPU to completion.
    pub cancel: crate::inference::serve::cancel::CancelToken,
    /// Adaptive step reuse for the sampler (see `native_flux_sampling::StepReuse`):
    /// the fraction of predicted output change a step may accumulate before the DiT
    /// must run again. An APPROXIMATION, so 0 (the default) disables it and every
    /// step is computed.
    ///
    /// The usable threshold is PER MODEL, not portable: measured at 1024^2, Flux
    /// keeps its composition up to 0.10 (1.6x) while Qwen-Image - which runs two
    /// CFG forwards per step - changes framing there and holds only to ~0.04
    /// (1.5x). Raise it per model against a fixed seed and LOOK at the result.
    pub step_reuse: f32,
    /// What the render is steered AWAY from, for the families that use guidance.
    ///
    /// A classifier-free-guidance model is conditioned on two prompts and pushed away
    /// from the second, so this is not decoration - it is half of what decides the
    /// picture, and it is how a caller gets rid of the failures a positive prompt cannot
    /// name: extra fingers, fused limbs, wrong proportions, a watermark. The pipeline
    /// took one all along; nothing carried the caller's, so every render used one fixed
    /// string and no request could say otherwise.
    ///
    /// `None` keeps that string. The distilled flow-matching models run CFG-free and
    /// ignore it entirely - handing them one would cost a second forward per step and
    /// change nothing.
    pub negative_prompt: Option<String>,
    /// Areas of the canvas that carry their own prompt, as
    /// `(prompt, x, y, w, h, strength)` with the rectangle in 0..1 of the image.
    ///
    /// Asking one prompt for two subjects lets the attention blend them - two bodies
    /// merge into one, which no amount of steps or guidance fixes because nothing in
    /// the conditioning keeps them apart. A region denoises against its OWN prompt, so
    /// the descriptions cannot borrow from each other.
    pub regions: Vec<(String, f32, f32, f32, f32, f32)>,
    /// Base64 mask for INPAINTING: it marks where to change `input_image`.
    /// Transparent (or black without alpha) = regenerate here, opaque = leave alone,
    /// matching the OpenAI edit endpoint. Only meaningful together with an input image.
    pub mask: Option<String>,
    pub control_image: Option<Vec<u8>>,
    /// How hard the control image pulls. 0 reproduces the base model exactly.
    pub control_scale: f32,
    /// Solver name (`euler`, `dpmpp_2m`). The SCHEDULER decides which noise levels the
    /// run visits; the SAMPLER decides how it moves between two of them. None = the
    /// family's default.
    pub sampler: Option<String>,
    /// Sigma-curve name (`normal`, `karras`, `exponential`). None = the family default.
    pub scheduler: Option<String>,
    /// LoRA adapters to apply for THIS render: `(path, strength)`.
    ///
    /// Per request rather than per load, because that is how they are used - the same
    /// checkpoint serves a dozen adapters and reloading it for each would cost more than
    /// the render. Applied over the resident weights and dropped afterwards.
    pub loras: Vec<(String, f32)>,
}

impl Default for ImageGenParams {
    fn default() -> Self {
        Self {
            control_image: None,
            control_scale: 1.0,
            negative_prompt: None,
            height: 512,
            width: 512,
            num_steps: 4,
            guidance: 4.0,
            seed: None,
            input_image: None,
            strength: 0.3,
            kontext: false,
            cancel: crate::inference::serve::cancel::CancelToken::new(),
            step_reuse: 0.0,
            regions: Vec::new(),
            mask: None,
            sampler: None,
            scheduler: None,
            loras: Vec::new(),
        }
    }
}

/// Progress event during image generation
#[derive(Debug, Clone)]
pub enum ImageStreamEvent {
    Progress { completed: usize, total: usize },
    Complete { image_base64: String },
    Error(String),
}

/// Which Flux model variant is loaded.
///
/// `Single` and `Hetero` differ in size (~896 vs several KB), but the
/// enum holds at most one model per server process - boxing either
/// arm trades fixed bytes for a hot-path indirection on every block
/// call during denoise. clippy::large_enum_variant ignored
/// intentionally; revisit if we ever store these in a Vec.
#[allow(clippy::large_enum_variant)]
/// ARCHITECTURE of the Flux family, used to DERIVE what a generation needs rather
/// than to state it. There were two fixed reserves here - a 4 GiB "runtime reserve"
/// and a 3 GiB floor beneath it - and both were wrong in the same direction at once:
/// too large at small sizes, refusing placements that would have run, and too small
/// above the size they were written for, admitting a load whose weights fit and whose
/// denoise then could not allocate. Nothing recovers from the second: the model is
/// resident by then, the card is full, and every request afterwards fails on a card
/// the placement itself chose.
///
/// The VAE reduces space by 8 and the transformer packs 2x2 latent neighbourhoods
/// into one token, so a request's pixels become tokens through their product.
pub const FLUX_VAE_STRIDE: usize = 8;
pub const FLUX_PATCH: usize = 2;
/// The text conditioning is padded to a fixed length and rides in the same sequence
/// as the image tokens, so it is part of what the attention is sized for.
pub const FLUX_TEXT_TOKENS: usize = 256;
/// Width of the sinusoidal timestep embedding feeding the conditioning MLPs.
pub(super) const FLUX_TIME_FREQ_DIM: usize = 256;
