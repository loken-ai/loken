//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

/// WHERE the transformer's blocks were placed. One transformer either way - the variant is the
/// placement, not a second implementation.
///
/// The distinction is not cosmetic: a whole placement is the only one that can measure what a
/// whole-placement VRAM reserve needs, the only one an inpaint mask or a set of regions can be
/// honoured on, and the only one with no host-resident block to repatriate mid-render.
pub enum FluxVariant {
    /// Every block on one device. The denoise loop runs from a state cast once
    /// (`flux::sampling::denoise_native`), bridged at each loop boundary rather than per step.
    Whole(HeteroFlux),
    /// Blocks split across CUDA/OpenCL/CPU because no one device holds them.
    Hetero(HeteroFlux),
}

/// State for a loaded Flux image model
pub struct FluxModelState {
    pub flux: FluxVariant,
    /// The caption encoder, resident only while it is worth the room.
    ///
    /// It runs ONCE per prompt and its output is cached, yet it is the second largest
    /// thing on the machine - most of ten gigabytes sitting idle through every denoise
    /// step. At small sizes that costs nothing anyone notices; at large ones it is the
    /// difference between the blocks fitting on the cards and being pushed to the
    /// host, because it holds the very card the split needs.
    ///
    /// `None` means it was released to make room and will be rebuilt on the next cache
    /// miss - which a repeated prompt never triggers.
    pub t5_model: Option<crate::inference::model::t5::encoder::T5EncoderModel>,
    /// What it takes to rebuild the encoder after releasing it.
    pub t5_source: Option<(
        std::path::PathBuf,
        crate::inference::model::t5::encoder::Config,
    )>,
    /// Device the T5 encoder lives on (a second CUDA device when available so it
    /// doesn't compete with the Flux transformer's VRAM, else CPU).
    pub t5_device: Device,
    pub t5_tokenizer: Tokenizer,
    pub clip_model: crate::inference::model::clip::text::Transformer,
    pub clip_tokenizer: Tokenizer,
    pub ae: crate::inference::model::flux::vae::AutoEncoder,
    /// Device the VAE lives on. At 1024^2 the Flux transformer fills GPU0 during
    /// denoise, so the VAE decode is placed on the (idle-by-then) T5 GPU when a
    /// second CUDA device exists; the latent is moved there before `ae.decode`.
    pub vae_device: Device,
    /// CPU-resident VAE used for large images (>=768^2) that OOM the GPU decode.
    /// `None` when the primary VAE is already on CPU.
    pub ae_cpu: Option<crate::inference::model::flux::vae::AutoEncoder>,
    /// The adapter set currently attached to the DiT, in request order. Requests carry
    /// their whole set, so comparing before rebuilding makes a repeat render free -
    /// reading and uploading an adapter is hundreds of megabytes.
    pub attached_loras: Vec<(String, f32)>,
    /// LRU of text-encoder outputs keyed by the input prompt.
    /// Holds `(t5_emb, clip_emb)` per prompt - capacity is small so
    /// alternating prompts stay warm without unbounded growth.
    pub(super) text_cache: PromptTextCache<(Tensor, Tensor)>,
    /// LRU of VAE-encoded init latents for img2img keyed by
    /// `{hash}-{w}x{h}` of the input image base64 + target size.
    /// Skips the ~50-200 ms base64 decode + VAE encode when the user
    /// iterates with the same source image but a different prompt.
    pub(super) vae_latent_cache: PromptTextCache<Tensor>,
}

/// Z-Image scheduler constants
pub const ZIMAGE_BASE_SEQ_LEN: usize = 256;
pub(super) const ZIMAGE_MAX_SEQ_LEN: usize = 4096;
pub(super) const ZIMAGE_BASE_SHIFT: f64 = 0.5;
pub(super) const ZIMAGE_MAX_SHIFT: f64 = 1.15;

/// Which Z-Image transformer variant is loaded.
/// See FluxVariant for the same large_enum_variant rationale.
#[allow(clippy::large_enum_variant)]
pub enum ZImageVariant {
    /// Single-device, NATIVE tensor substrate -
    /// the default when one CUDA device fits the BF16 transformer; the
    /// facade variants below are the fallback when the native load fails
    /// or VRAM is too tight. Holds the native model directly: the denoise
    /// loop in `generate_zimage` runs on native tensors end-to-end,
    /// bridged once per generate at each loop boundary.
    NativeSingle(crate::inference::model::zimage::dit::ZImageTransformer2DModel),
    /// Single-device (all blocks on one GPU or CPU), facade substrate
    Single(ztf::ZImageTransformer2DModel),
    /// Multi-device (blocks split across devices)
    Hetero(HeteroZImage),
}

/// State for a loaded Z-Image model
pub struct ZImageModelState {
    pub transformer: ZImageVariant,
    pub text_encoder: crate::inference::model::zimage::text_encoder::ZImageTextEncoder,
    pub tokenizer: Tokenizer,
    pub vae: crate::inference::model::zimage::vae::AutoEncoderKL,
    /// CPU-resident VAE for large images (>=768^2) whose GPU decode OOMs - the
    /// VAE mid-block runs a 16384^2 spatial attention at 1024^2. Mirrors the Flux
    /// `ae_cpu` fallback; only loaded when the primary VAE is on GPU.
    pub vae_cpu: Option<crate::inference::model::zimage::vae::AutoEncoderKL>,
    /// VAE device - typically the primary CUDA device, falls back to CPU
    /// when no GPU is available. Decode latents are moved here.
    pub vae_device: Device,
    /// VAE dtype - held at F32 to keep the HD=512 mid-block attention
    /// numerically safe (no F16 overflow path in z_image::vae).
    pub vae_dtype: DType,
    /// Where the VAE weights came from, so a VAE that had to load on the CPU under
    /// pressure can go back to a GPU once one frees up. Placement is decided with the
    /// VRAM that exists at LOAD time, and that is not the VRAM that exists at DECODE
    /// time - without this the first crowded load strands the VAE on the CPU for the
    /// process's life, at minutes per image instead of seconds.
    pub vae_path: std::path::PathBuf,
    pub transformer_cfg: crate::inference::model::zimage::dit::Config,
    pub scheduler_cfg: crate::inference::model::zimage::sampling::SchedulerConfig,
    /// Text encoder may be on a different device than the transformer
    pub te_device: Device,
    pub te_dtype: DType,
    /// LRU of text-encoder outputs keyed by the formatted prompt
    /// (after the qwen3 chat-template wrap). Holds `(cap_feats,
    /// cap_mask)` per prompt - keeps a few flips warm for A/B
    /// iteration, mirroring the Flux pipeline's text_cache.
    /// Caching `cap_mask` alongside `cap_feats` lets a cached
    /// generate skip the per-call `Tensor::ones` mask allocation.
    /// (Token count is no longer cached separately - it's encoded
    /// in `cap_mask.dims()` and not needed by the downstream
    /// transformer forward.)
    pub(super) text_cache: PromptTextCache<(Tensor, Tensor)>,
    /// LRU of VAE-encoded init latents for img2img keyed by
    /// `{hash}-{w}x{h}` of the input image base64 + target size.
    /// Saves the ~50-200 ms VAE encode when a user iterates with
    /// the same input image but a different prompt.
    pub(super) vae_latent_cache: PromptTextCache<Tensor>,
}

/// Loaded image model - either Flux or Z-Image.
/// See FluxVariant for the same large_enum_variant rationale (here both
/// arms are sized in KB, so any boxing would force one extra dereference
/// on every attention block during denoise).
#[allow(clippy::large_enum_variant)]
pub enum LoadedImageModel {
    Flux(FluxModelState),
    ZImage(ZImageModelState),
    QwenImage(crate::inference::engine::qwen_image_engine::QwenImageModelState),
    Flux2(crate::inference::engine::flux2_engine::Flux2ModelState),
    Boogu(crate::inference::engine::boogu_engine::BooguModelState),
    Sdxl(crate::inference::model::sdxl::pipeline::SdxlPipeline),
}

/// State for a loaded image generation model
pub struct LoadedImageModelState {
    /// The geometry this model was PLACED for.
    ///
    /// A placement reserves denoise scratch for the request that triggered it, so the
    /// question "can the resident serve this request" is whether the new geometry needs
    /// more than the placed one did - not how much VRAM happens to be free right now.
    /// Free VRAM after a load sits near the reserve BY DESIGN: the resident's own
    /// footprint is what consumed it. Comparing against it treats the intended state as
    /// a shortage and re-plans on every request, which reloads the whole model between
    /// two variations of the same picture.
    pub placed_for: crate::inference::place::runtime_demand::RequestGeometry,
    /// USER-FACING name of the resident model (what /api/ps and the GUI show).
    pub name: String,
    /// Identity of the resident CHECKPOINT FILE, used to decide whether a request
    /// for another tag can reuse this resident (two tags over one file can; two
    /// different files never can). Kept separate from `name` so the reload logic
    /// never leaks into what the user sees.
    pub ckpt_id: Option<String>,
    pub model: LoadedImageModel,
    pub device: Device,
    pub dtype: DType,
    /// GPU bytes this model actually took, measured across its load.
    ///
    /// Not derived from the checkpoint or the plan: an image model is several
    /// networks (DiT, text encoders, VAE) on possibly different cards, and any
    /// formula over one of them under-reports the rest. `/api/ps` used to publish 0
    /// for the single-device variants and, worse, the segment's FREE memory for the
    /// hetero ones - a number about the card, not the model. 0 at least reads as
    /// unknown; free memory reads as a footprint and is not one.
    pub resident_bytes: u64,
}

/// Loading progress events for image model
#[derive(Debug, Clone)]
pub enum LoadingProgress {
    Stage(String), // "Loading T5 encoder...", etc.
    Error(String),
    Done,
}

/// The line a caller is shown while a model loads, and the tensor counts that keep it moving.
///
/// A load names its stage once ("Loading the text encoder") and then says nothing for tens of
/// seconds while the weights are read - which is indistinguishable from a wedged server, and
/// was the most common thing anyone had to sit through. The weight readers count what they
/// read into whatever reporter is published for the load (see `progress::scoped`), and this
/// turns those counts back into the SAME line, so the name a caller already understands gains
/// a number instead of a second widget appearing beside it.
///
/// Two properties matter as much as the count:
///
/// - the line is re-sent only when its PERCENTAGE changes, which bounds a checkpoint of any
///   size to about a hundred messages instead of one per tensor;
/// - a count is DROPPED rather than queued when the client is behind (`try_send`), because a
///   stale count is worth nothing and blocking a loader thread to deliver one would make the
///   load itself slower - the opposite of the point.
pub(crate) struct LoadStage {
    pub(super) tx: Option<tokio::sync::mpsc::Sender<LoadingProgress>>,
    /// Engine name in the log line, e.g. `Flux`.
    pub(super) tag: &'static str,
    pub(super) line: std::sync::Mutex<String>,
}

impl LoadStage {
    pub(crate) fn new(
        tag: &'static str,
        tx: Option<tokio::sync::mpsc::Sender<LoadingProgress>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            tx,
            tag,
            line: std::sync::Mutex::new(String::new()),
        })
    }

    /// Name the stage: logged, sent, and remembered as the line the counts refine.
    pub(crate) fn say(&self, msg: &str) {
        info!("  [{}] {}", self.tag, msg);
        if let Ok(mut l) = self.line.lock() {
            msg.clone_into(&mut *l);
        }
        if let Some(tx) = &self.tx {
            let _ = tx.blocking_send(LoadingProgress::Stage(msg.to_string()));
        }
    }

    /// The model is up. Sent on the same channel so a client stops waiting.
    pub(crate) fn finished(&self) {
        if let Some(tx) = &self.tx {
            let _ = tx.blocking_send(LoadingProgress::Done);
        }
    }

    /// The reporter to publish for this load. The phase is deliberately ignored: the stage
    /// line already says WHICH component is loading, in the words the loader chose.
    pub(crate) fn reporter(
        self: &Arc<Self>,
    ) -> crate::inference::serve::progress::SharedProgressFn {
        let me = self.clone();
        crate::inference::serve::progress::per_percent(Arc::new(
            move |_phase: &str, done: usize, total: usize| {
                if total == 0 {
                    return; // nothing to count - the stage line already says the name
                }
                let line = me.line.lock().map(|l| l.clone()).unwrap_or_default();
                if let Some(tx) = &me.tx {
                    let _ = tx.try_send(LoadingProgress::Stage(format!("{line} {done}/{total}")));
                }
            },
        ))
    }
}

/// Layer distribution info for a loaded image model
#[derive(Debug, Clone)]
pub struct ImageLayerDist {
    pub device_type: String,
    pub device_id: usize,
    pub layer_range: (u32, u32),
    pub memory_bytes: u64,
}

/// Info about a loaded image model (for hardware tab display)
#[derive(Debug, Clone)]
pub struct ImageModelInfo {
    pub name: String,
    pub model_type: String,
    pub total_layers: u32,
    pub layer_distribution: Vec<ImageLayerDist>,
}

/// Pick the best CUDA device for an auxiliary module (VAE, text
/// encoder) given live per-device free VRAM and a `min_bytes`
/// footprint estimate. Returns `Some(gpu_id)` if at least one
/// device has enough room; `None` means caller should fall back
/// to CPU.
///
/// Selection priority:
/// 1. If `prefer` is Some and that device has >= min_bytes free,
///    return it (saves a PCIe transfer when the module would
///    otherwise read tensors from the preferred device).
/// 2. Otherwise return the device with the MOST free VRAM that
///    has at least `min_bytes` free.
///
/// This addresses the long-standing "VAE OOMs on GPU0 while GPU1
/// has 15 GB free" pattern. Replaces the boolean `vae_fits_on_gpu`
/// which only checked the primary device.
pub(crate) fn pick_aux_device(
    per_device_free: &[(usize, u64)],
    min_bytes: u64,
    prefer: Option<usize>,
) -> Option<usize> {
    // 1. Preferred device if it fits - saves the PCIe transfer
    //    cost of moving the input tensor to a different device.
    if let Some(pref) = prefer {
        if let Some((_, free)) = per_device_free.iter().find(|(i, _)| *i == pref) {
            if *free >= min_bytes {
                return Some(pref);
            }
        }
    }
    // 2. Otherwise: FIRST device in the caller's order (fastest-first from the probe) whose free
    //    VRAM fits the peak. A device the hot model packed full simply fails the fit filter and
    //    the next-fastest card takes the aux - no "most free VRAM" heuristic (banned: it routes
    //    work by memory pressure instead of throughput).
    per_device_free
        .iter()
        .find(|(_, free)| *free >= min_bytes)
        .map(|(idx, _)| *idx)
}
