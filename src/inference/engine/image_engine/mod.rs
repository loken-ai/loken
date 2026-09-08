use crate::inference::model::zimage::dit as ztf;
// Re-exported through `super::` by the placement tests, which is why the non-test build calls
// these unused.
#[allow(unused_imports)]
use crate::inference::place::dry_plan::{describe_plan, solve, DeviceLoad, PlanLoad};
use crate::inference::place::layer_executor::DeviceKind;
use crate::tensor::{DType, Device, IndexOp, Tensor};
use anyhow::{anyhow, Result as AnyResult};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::inference::engine::prompt_cache::{
    vae_latent_cache_key, PromptTextCache, IMG2IMG_LATENT_CACHE_CAP, PROMPT_TEXT_CACHE_CAP,
};
use crate::inference::model::flux::hetero::HeteroFlux;
use crate::inference::model::zimage::hetero::HeteroZImage;
use crate::inference::place::layer_executor::HeteroPlan;
use crate::inference::place::model_request::ModelRequest;

mod params;
pub use params::*;
mod demand;
pub use demand::*;
mod state;
pub use state::*;
/// The whole checkpoint on the one device it was placed on.
///
/// The ordinal is read off the device rather than assumed: a single-device load is not
/// necessarily card 0, and this is what the hardware tab shows.
fn all_on_one_device(state: &LoadedImageModelState, total: u32) -> Vec<ImageLayerDist> {
    let (device_type, device_id) = match state.device.location() {
        crate::tensor::DeviceLocation::Cuda { gpu_id } => ("CUDA".to_string(), gpu_id),
        crate::tensor::DeviceLocation::Cpu => ("CPU".to_string(), 0),
    };
    vec![ImageLayerDist {
        device_type,
        device_id,
        layer_range: (0, total),
        memory_bytes: state.resident_bytes,
    }]
}

/// The layers each segment of a placement holds, and that segment's share of the footprint.
///
/// The footprint is one measured total. Attributing it per segment would need per-device
/// accounting the loaders do not keep, so it is split evenly - which at least sums to the truth.
/// It is not the segment's free memory: that is a fact about the card, not about the model.
fn spread_over_plan(plan: &HeteroPlan, resident_bytes: u64) -> Vec<ImageLayerDist> {
    let n_segments = plan.segments.len() as u64;
    plan.segments
        .iter()
        .map(|seg| {
            let (device_type, device_id) = match seg.kind {
                DeviceKind::Cuda(i) => ("CUDA".to_string(), i),
                DeviceKind::OpenCL(i) => ("Arc".to_string(), i),
                DeviceKind::Cpu => ("CPU".to_string(), 0),
            };
            ImageLayerDist {
                device_type,
                device_id,
                layer_range: (seg.layer_start as u32, seg.layer_end as u32),
                memory_bytes: resident_bytes / n_segments.max(1),
            }
        })
        .collect()
}

/// Image generation engine (separate from text LlmEngine)
pub struct ImageEngine {
    model_state: Arc<Mutex<Option<LoadedImageModelState>>>,
}

impl Default for ImageEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageEngine {
    pub fn new() -> Self {
        Self {
            model_state: Arc::new(Mutex::new(None)),
        }
    }

    /// Check if an image model is loaded
    pub async fn is_loaded(&self) -> bool {
        self.model_state.lock().await.is_some()
    }

    /// Lightweight family identifier for the loaded checkpoint. Returns
    /// `Some("flux")` / `Some("zimage")` / None depending on what's warm.
    /// Used by /v1/models to surface `is_loaded` per multimodal entry
    /// without taking the heavyweight `get_loaded_model_info()` path.
    pub async fn loaded_family(&self) -> Option<&'static str> {
        match self.model_state.lock().await.as_ref()?.model {
            LoadedImageModel::Flux(_) => Some("flux"),
            LoadedImageModel::ZImage(_) => Some("zimage"),
            LoadedImageModel::QwenImage(_) => Some("qwen-image"),
            LoadedImageModel::Flux2(_) => Some("flux2"),
            LoadedImageModel::Boogu(_) => Some("boogu"),
            LoadedImageModel::Sdxl(_) => Some("sdxl"),
        }
    }

    /// Get info about the loaded image model (name, device, layer distribution)
    pub async fn get_loaded_model_info(&self) -> Option<ImageModelInfo> {
        let guard = self.model_state.lock().await;
        let state = guard.as_ref()?;
        let (model_type, total_layers, layer_distribution) = match &state.model {
            LoadedImageModel::Flux(flux) => match &flux.flux {
                FluxVariant::Whole(_) => {
                    let total = 57u32;
                    let dist = all_on_one_device(state, total);
                    ("Flux Schnell".into(), total, dist)
                }
                FluxVariant::Hetero(hetero) => {
                    let total = hetero.plan.total_layers as u32;
                    let dist = spread_over_plan(&hetero.plan, state.resident_bytes);
                    ("Flux Schnell".into(), total, dist)
                }
            },
            LoadedImageModel::ZImage(zimg) => match &zimg.transformer {
                ZImageVariant::Single(_) | ZImageVariant::NativeSingle(_) => {
                    let total = 34u32;
                    let dist = all_on_one_device(state, total);
                    ("Z-Image".into(), total, dist)
                }
                ZImageVariant::Hetero(hetero) => {
                    let total = hetero.plan.total_layers as u32;
                    let dist = spread_over_plan(&hetero.plan, state.resident_bytes);
                    ("Z-Image".into(), total, dist)
                }
            },
            LoadedImageModel::QwenImage(_) => {
                let total = 60u32;
                let dist = all_on_one_device(state, total);
                ("Qwen-Image".into(), total, dist)
            }
            LoadedImageModel::Flux2(f2) => {
                // 5 dual-stream + 20 parallel single blocks.
                let cfg = f2.config();
                let total = (cfg.num_layers + cfg.num_single_layers) as u32;
                let dist = all_on_one_device(state, total);
                ("FLUX.2 Klein".into(), total, dist)
            }
            LoadedImageModel::Boogu(_) => {
                let total = 40u32;
                let dist = all_on_one_device(state, total);
                ("Boogu-Image".into(), total, dist)
            }
            LoadedImageModel::Sdxl(_) => {
                // 9 input levels + 3 middle + 9 output, on one device.
                let total = 21u32;
                let dist = all_on_one_device(state, total);
                ("SDXL".into(), total, dist)
            }
        };
        Some(ImageModelInfo {
            name: state.name.clone(),
            model_type,
            total_layers,
            layer_distribution,
        })
    }
}

/// The negative prompt SDXL is conditioned against when the caller gives none.
/// The family expects one (its guidance is trained with it), unlike the distilled
/// flow-matching models that run CFG-free.
const SDXL_NEGATIVE: &str = "blurry, low quality, distorted, deformed, watermark, text";

/// What this request steers away from: the caller's words when they gave any, the
/// default otherwise.
///
/// An EMPTY string is a choice, not an omission - it means "steer away from nothing" -
/// so it is honoured rather than replaced by the default.
fn negative_of(params: &ImageGenParams) -> &str {
    params.negative_prompt.as_deref().unwrap_or(SDXL_NEGATIVE)
}

/// A base64 picture, upright and at exactly `width x height`.
///
/// Every picture a request carries - the source, the mask, the reference - arrives the
/// same way and has to end up on the same grid, so this says it once. EXIF orientation is
/// applied, so a phone photo comes in the way it was taken rather than a quarter turn
/// over, and the resize is unconditional: a sampler reads one latent grid, and a source
/// of another size would be read with the wrong stride.
///
/// The two prefixes belong to the caller because the caller knows what the picture IS to
/// the request, and that is the part of a failure worth reading. Callers that tolerate
/// surrounding whitespace trim before calling; the strict ones pass the field as it came.
fn decode_base64_fitted(
    b64: &str,
    width: usize,
    height: usize,
    filter: image::imageops::FilterType,
    bad_base64: &str,
    bad_picture: &str,
) -> AnyResult<image::DynamicImage> {
    use base64::Engine as _;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| anyhow!("{bad_base64}: {e}"))?;
    let src = crate::inference::media::image_processor::decode_image_oriented(&raw)
        .map_err(|e| anyhow!("{bad_picture}: {e}"))?;
    Ok(src.resize_exact(width as u32, height as u32, filter))
}

/// A base64 image -> interleaved RGB8 at exactly `width x height`.
fn decode_base64_to_rgb8(b64: &str, width: usize, height: usize) -> AnyResult<Vec<u8>> {
    let img = decode_base64_fitted(
        b64.trim(),
        width,
        height,
        image::imageops::FilterType::Lanczos3,
        "input image is not valid base64",
        "cannot decode the input image",
    )?;
    Ok(img.to_rgb8().into_raw())
}

/// The token ids `text` encodes to, named after the tokenizer that could not read it.
///
/// Every caption encoder here wants the same thing - owned ids, special tokens included -
/// and the only part that differs between them is which one failed.
fn encode_ids(tokenizer: &Tokenizer, text: &str, what: &str) -> AnyResult<Vec<u32>> {
    let encoded = tokenizer
        .encode(text, true)
        .map_err(|e| anyhow!("{what} tokenize: {e}"))?;
    Ok(encoded.get_ids().to_vec())
}

/// Free bytes on `dev` as the driver reports them, or 0 when it does not report.
///
/// Zero is the honest answer for a device nothing can measure, and it is also the safe
/// one: every caller compares it against what a decode needs, and no decode needs nothing.
fn free_on_device(dev: &Device) -> u64 {
    crate::inference::place::vram_manager::probe(0)
        .into_iter()
        .find(|(_, _, d)| crate::tensor::Device::same_device(d, dev))
        .map_or(0, |(_, free, _)| free)
}

/// Interleaved RGB8 -> a base64 PNG, the string every image path returns.
fn rgb_to_png_base64(rgb: &[u8], width: usize, height: usize) -> AnyResult<String> {
    use base64::Engine as _;
    let img = image::RgbImage::from_raw(width as u32, height as u32, rgb.to_vec())
        .ok_or_else(|| anyhow!("sdxl: RGB buffer does not match {width}x{height}"))?;
    let mut png = std::io::Cursor::new(Vec::new());
    img.write_to(&mut png, image::ImageFormat::Png)
        .map_err(|e| anyhow!("sdxl: png encode: {e}"))?;
    Ok(base64::engine::general_purpose::STANDARD.encode(png.into_inner()))
}

/// Give the denoise its card back when the caption encoder is in the way.
///
/// The encoder is one-shot and its output is cached, so past this point it is inert
/// weight - and it is most of ten gigabytes, usually on the second card, which is
/// precisely where a split wants to put blocks. Holding it there is what turned a
/// large render into "14 blocks on the GPUs, 43 on the host": not a shortage of VRAM,
/// a shortage of VRAM THAT WAS BEING USED FOR ANYTHING.
///
/// Released only when this request actually needs the room. A small render leaves it
/// resident and pays nothing; a large one frees it and pays a rebuild on the next
/// prompt nobody has rendered yet, which a repeated prompt never triggers.
fn release_text_encoder_if_needed(state: &mut FluxModelState, width: usize, height: usize) {
    if state.t5_model.is_none() || !state.t5_device.is_cuda() {
        return;
    }
    let need = flux_runtime_demand(width, height);
    let free_there = match state.t5_device.location() {
        crate::tensor::DeviceLocation::Cuda { gpu_id } => {
            crate::inference::place::vram_manager::probe(0)
                .into_iter()
                .find(|(i, _, _)| *i == gpu_id)
                .map(|(_, free, _)| free)
        }
        _ => None,
    }
    .unwrap_or(u64::MAX);
    if free_there >= need {
        return;
    }
    info!(
        "Flux: releasing the caption encoder - {:.1} GB free on its card against {:.1} GB \
         this render needs, and it has nothing left to do",
        free_there as f64 / 1e9,
        need as f64 / 1e9,
    );
    state.t5_model = None;
    crate::inference::engine::llm_engine::trim_cuda_pools();
}

/// Helper: encode prompt through T5 and CLIP (Flux pipeline)
fn encode_flux_text(
    state: &mut FluxModelState,
    device: &Device,
    prompt: &str,
) -> AnyResult<(Tensor, Tensor)> {
    // LRU hit: same prompt seen recently - skip the T5+CLIP pass and
    // return the previously cached embeddings.
    if let Some((t5, clip)) = state.text_cache.get(prompt) {
        info!("Flux text: cached (skipping T5+CLIP)");
        return Ok((t5, clip));
    }

    // T5 (runs on CPU, then move to GPU).
    // Flux Schnell was trained with T5 inputs padded to 256 tokens - running
    // T5 on the un-padded prompt produces a (1, real_len, 4096) embedding
    // that propagates through the Flux transformer to NaN (all-black image).
    // Pad with zeros (T5 pad_token_id) up to 256. Truncate if longer.
    let t5_t = std::time::Instant::now();
    let mut t5_tokens = encode_ids(&state.t5_tokenizer, prompt, "T5")?;
    let real_len = t5_tokens.len();
    const T5_LEN: usize = 256;
    /// CLIP's context length: the size of its positional embedding table, and so a
    /// hard limit rather than a preference.
    const CLIP_CONTEXT: usize = 77;
    if t5_tokens.len() > T5_LEN {
        t5_tokens.truncate(T5_LEN);
    } else {
        t5_tokens.resize(T5_LEN, 0);
    }
    // Rebuild the encoder if it was released to make room for a denoise. Only a
    // prompt nobody has rendered pays this, and the alternative is holding ten
    // gigabytes idle through every step of every render.
    if state.t5_model.is_none() {
        let Some((path, cfg)) = state.t5_source.clone() else {
            return Err(anyhow!(
                "Flux: the text encoder was released and cannot be rebuilt"
            ));
        };
        info!("Flux: rebuilding the text encoder for a prompt that is not cached");
        let ndev = &state.t5_device.clone();
        let ndtype = if state.t5_device.is_cuda() {
            crate::tensor::DType::BF16
        } else {
            crate::tensor::DType::F32
        };
        let vb = unsafe { crate::tensor::VarBuilder::from_files(&[&path], ndtype, ndev) }
            .map_err(|e| anyhow!("T5 rebuild: {e}"))?;
        state.t5_model = Some(
            crate::inference::model::t5::encoder::T5EncoderModel::load(vb, &cfg)
                .map_err(|e| anyhow!("T5 rebuild: {e}"))?,
        );
    }
    let t5_input = Tensor::new(&t5_tokens[..], &state.t5_device)?.unsqueeze(0)?;
    let t5_emb = state
        .t5_model
        .as_mut()
        .ok_or_else(|| anyhow!("Flux: no text encoder"))?
        .forward(&t5_input)?
        .to_device(device)?
        .to_dtype(DType::F32)?;
    info!(
        "Flux T5: {:.2}s ({} real tokens, padded to {})",
        t5_t.elapsed().as_secs_f64(),
        real_len,
        T5_LEN
    );

    // CLIP (runs on GPU)
    let clip_t = std::time::Instant::now();
    let mut clip_tokens = encode_ids(&state.clip_tokenizer, prompt, "CLIP")?;
    // CLIP's positional table is exactly CLIP_CONTEXT entries, so a longer prompt
    // indexes past the end and the render dies with "index 77 out of range 77". T5
    // above was already truncated; this one never was.
    //
    // The last token must stay the END marker: CLIP's pooled embedding is read AT that
    // position, so a blunt truncate would pool from whatever word happened to land
    // last and quietly change the conditioning instead of just shortening it.
    if clip_tokens.len() > CLIP_CONTEXT {
        let end = *clip_tokens.last().unwrap_or(&0);
        clip_tokens.truncate(CLIP_CONTEXT - 1);
        clip_tokens.push(end);
        info!("Flux CLIP: prompt over {CLIP_CONTEXT} tokens, truncated (T5 still sees {T5_LEN})");
    }
    let clip_input = Tensor::new(&clip_tokens[..], device)?.unsqueeze(0)?;
    let clip_emb = state
        .clip_model
        .forward(&clip_input)?
        .to_dtype(DType::F32)?;
    info!("Flux CLIP: {:.2}s", clip_t.elapsed().as_secs_f64());

    // Cache for next call (LRU; evicts oldest when over capacity).
    state
        .text_cache
        .insert(prompt.to_string(), (t5_emb.clone(), clip_emb.clone()));

    Ok((t5_emb, clip_emb))
}

/// Make `want` the adapter set attached to the Flux DiT, reusing the current one when it
/// matches.
///
/// EVERY variant carries adapters. It was worth saying otherwise once - the facade path
/// had no adapter code - but that meant a Kontext checkpoint, which loads there by
/// design, could never take one while the picker went on offering them. A model that
/// cannot honour a request must not be advertised as if it could; the fix was to make it
/// honour the request.
///
/// A file that matches NOTHING is still an error rather than a quiet base-model render:
/// the failure people actually hit is a LoRA that appears to load and changes nothing.
fn set_flux_loras(
    flux_state: &mut FluxModelState,
    dev: &crate::tensor::Device,
    want: &[(String, f32)],
) -> anyhow::Result<()> {
    if flux_state.attached_loras == want {
        return Ok(());
    }
    // The adapter set is attached the same way whatever substrate the DiT sits on; only
    // the model object differs, and the two are different types. The macro keeps ONE
    // copy of the sequence - clear, attach each file, refuse a file that matched nothing
    // - because the split path was previously not written at all, and a second hand-copy
    // of it is a second thing to forget to fix.
    macro_rules! attach_to {
        ($flux:expr, $load_dev:expr) => {{
            $flux.clear_lora();
            flux_state.attached_loras.clear();
            let mut matched = 0usize;
            for (path, strength) in want {
                let file = crate::inference::load::lora::LoraFile::load(path, $load_dev)
                    .map_err(|e| anyhow!("lora '{path}': {e}"))?;
                let n = $flux
                    .apply_lora(&file, *strength)
                    .map_err(|e| anyhow!("lora '{path}': {e}"))?;
                if n == 0 {
                    $flux.clear_lora();
                    return Err(anyhow!(
                        "{}",
                        crate::inference::load::lora::no_match_error(path, &file).0
                    ));
                }
                matched += n;
            }
            matched
        }};
    }

    let matched = match &mut flux_state.flux {
        FluxVariant::Whole(flux) => attach_to!(flux, dev),
        // A SPLIT model is the normal case on a multi-GPU host, not an exotic one: its
        // blocks each live on their own card, so the deltas are built on the host and
        // uploaded per block rather than read onto one device.
        FluxVariant::Hetero(flux) => attach_to!(flux, &crate::tensor::Device::Cpu),
    };
    if !want.is_empty() {
        info!("flux: {} lora(s) on {matched} projections", want.len());
    }
    flux_state.attached_loras = want.to_vec();
    Ok(())
}

/// Encode each region's prompt and pair it with its per-token weights.
///
/// Empty when no region was asked for, which is the path that must stay exactly as it
/// was: one forward per step, no extra encode, no behaviour change for every render
/// that does not use this.
fn build_flux_regions(
    flux_state: &mut FluxModelState,
    device: &Device,
    params: &ImageGenParams,
    height: usize,
    width: usize,
    base_t5: &crate::tensor::Tensor,
    base_clip: &crate::tensor::Tensor,
) -> anyhow::Result<Vec<crate::inference::model::flux::sampling::Region>> {
    use crate::inference::model::flux::sampling as fs;
    if params.regions.is_empty() {
        return Ok(Vec::new());
    }
    let ndev = device.clone();
    let rects: Vec<(f32, f32, f32, f32, f32)> = params
        .regions
        .iter()
        .map(|(_, x, y, w, h, s)| (*x, *y, *w, *h, *s))
        .collect();
    let weights = fs::region_weights(height, width, &rects, &ndev)?;

    // Index 0 is the BASE prompt, covering whatever no rectangle claims.
    let mut out = Vec::with_capacity(weights.len());
    let mut push = |t5: &crate::tensor::Tensor,
                    clip: &crate::tensor::Tensor,
                    w: crate::tensor::Tensor|
     -> anyhow::Result<()> {
        let st = fs::State::new(t5, clip, &fs::get_noise(1, height, width, device)?)?;
        let n = fs::NativeState::to_f32(&st)?;
        out.push(fs::Region {
            txt: n.txt,
            txt_ids: n.txt_ids,
            vec: n.vec,
            weight: w,
        });
        Ok(())
    };
    push(base_t5, base_clip, weights[0].clone())?;
    for (i, (prompt, ..)) in params.regions.iter().enumerate() {
        let (t5, clip) = encode_flux_text(flux_state, device, prompt)?;
        release_text_encoder_if_needed(flux_state, width, height);
        push(&t5, &clip, weights[i + 1].clone())?;
    }
    Ok(out)
}

/// Generate image with Flux pipeline (non-streaming)
/// The CUDA ordinal a facade device names, or None when it is not a GPU.
fn gpu_index_of(dev: &Device) -> Option<usize> {
    let nd = dev.clone();
    #[cfg(feature = "cuda")]
    {
        match &nd {
            crate::tensor::Device::Cuda(cd) => Some(cd.ordinal()),
            _ => None,
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = nd;
        None
    }
}

/// Put the cuBLAS reduced-precision flags where this render needs them.
///
/// They are three process-global atomics, set at model LOAD and never cleared, so whichever
/// family loaded last decided the arithmetic for every family afterwards. A Z-Image render
/// following a FLUX load in the same process produced different pixels for the same prompt and
/// seed - silently, because a slightly different teapot is still a teapot.
///
/// Precision belongs to the model being run, not to the process that ran something else
/// earlier. Setting it per render rather than per load makes the answer depend on the request
/// Whether this family renders with the tensor-core (reduced-precision) GEMM paths.
///
/// Declared per family and applied by the dispatcher, because these are PROCESS-WIDE cuBLAS
/// flags: whichever render ran last decides the arithmetic for whichever runs next. That was
/// measured once - the same seed produced two different images depending on what had loaded
/// before - and fixed for the two families it was observed on, which left the other four
/// inheriting an answer instead of giving one. Matching on the variant means a new family
/// cannot be added without deciding.
fn family_wants_reduced_precision(model: &LoadedImageModel) -> bool {
    match model {
        // Measured and tuned with the tensor-core paths on.
        LoadedImageModel::Flux(_) | LoadedImageModel::Flux2(_) => true,
        // Z-Image drifts visibly with TF32 on its attention path.
        LoadedImageModel::ZImage(_) => false,
        // Not characterised either way: full precision is the answer that cannot silently
        // change an output, so it is the one to hold until someone measures.
        LoadedImageModel::QwenImage(_) | LoadedImageModel::Boogu(_) | LoadedImageModel::Sdxl(_) => {
            false
        }
    }
}

/// alone, whatever is resident beside it.
#[cfg(feature = "cuda")]
fn set_render_precision(device: &Device, reduced: bool) {
    if device.is_cuda() {
        crate::tensor::cuda_ext::set_gemm_reduced_precision(reduced);
    }
}
#[cfg(not(feature = "cuda"))]
fn set_render_precision(_device: &Device, _reduced: bool) {}

/// Denoise on a model whose blocks are spread across devices.
///
/// A reference image only changes what the loop carries beside the latent, not what the
/// loop is given, so the state is named once here instead of once per branch.
fn denoise_split(
    hetero: &mut HeteroFlux,
    state: &crate::inference::model::flux::sampling::State,
    kontext: Option<&(Tensor, Tensor)>,
    timesteps: &[f64],
    guidance: f64,
    cancel: &crate::inference::serve::cancel::CancelToken,
) -> AnyResult<Tensor> {
    use crate::inference::model::flux::sampling as fs;
    let (noise, ids) = (&state.img, &state.img_ids);
    let (txt, ids_txt, pooled) = (&state.txt, &state.txt_ids, &state.vec);
    let out = match kontext {
        Some((ctx, ctx_ids)) => fs::denoise_kontext(
            hetero,
            noise,
            ids,
            ctx,
            ctx_ids,
            txt,
            ids_txt,
            pooled,
            timesteps,
            guidance,
            Some(cancel),
        )?,
        None => fs::denoise(
            hetero,
            noise,
            ids,
            txt,
            ids_txt,
            pooled,
            timesteps,
            guidance,
            Some(cancel),
        )?,
    };
    Ok(out)
}

/// Say what the denoise took, and feed it back only when the figure answers the question
/// the reserve asks.
///
/// A reserve sizes ONE card for a WHOLE model. Under a split each card holds a fraction of
/// the blocks and so gives up a fraction of its memory, and feeding that fraction back is
/// how a 1536 render measured at 3.5 GB while split, was then placed whole on a card with
/// 3.7 GB spare, and ran out on its first steps. Reported either way, recorded only when
/// the placement was whole.
fn report_denoise_vram(
    width: usize,
    height: usize,
    free_before: u64,
    watch: crate::inference::place::vram_manager::VramWatch,
    placement_is_whole: bool,
) {
    let low = watch.finish();
    let took = free_before.saturating_sub(low);
    info!(
        "Flux denoise VRAM: {:.2} GB free before, {:.2} GB at the low-water mark \
         -> the denoise needed {:.2} GB (placement reserved {:.2} GB)",
        free_before as f64 / 1e9,
        low as f64 / 1e9,
        took as f64 / 1e9,
        flux_runtime_demand(width, height) as f64 / 1e9,
    );
    if placement_is_whole {
        crate::inference::place::runtime_demand::record_observed_peak("flux", width, height, took);
    } else {
        info!("Flux: split placement - not recorded, a fraction of the blocks gives up a fraction of the card");
    }
}

/// A decoded picture, ready for the PNG encoder and cheap to ship.
///
/// The clamp to [-1,1] is written as two rectifier folds because that is the pair of ops
/// every substrate here has; the scale to [0,255] and the cast happen on the VAE's own
/// device, so what crosses the bus is a quarter of the bytes a float image would be.
fn decoded_to_cpu_u8(decoded: Tensor) -> AnyResult<Tensor> {
    let f32_image = decoded.to_dtype(DType::F32)?;
    drop(decoded);
    let floored = f32_image.affine(1.0, 1.0)?.relu()?.affine(1.0, -1.0)?; // max(x, -1)
    let clamped = floored.affine(-1.0, 1.0)?.relu()?.affine(-1.0, 1.0)?; // min(m, 1)
    let bytes = clamped.affine(127.5, 127.5)?.to_dtype(DType::U8)?;
    Ok(bytes.to_device(&Device::Cpu)?)
}

fn generate_flux_image(
    flux_state: &mut FluxModelState,
    device: &Device,
    dtype: DType,
    prompt: &str,
    params: &ImageGenParams,
) -> AnyResult<String> {
    let pipeline_t = std::time::Instant::now();
    let height = params.height;
    let width = params.width;

    let (t5_emb, clip_emb) = encode_flux_text(flux_state, device, prompt)?;
    release_text_encoder_if_needed(flux_state, width, height);
    // One encode per region. They go through the same prompt cache as the base, so a
    // repeated region costs nothing after the first step of the first render.
    let regions = build_flux_regions(
        flux_state, device, params, height, width, &t5_emb, &clip_emb,
    )?;

    // FLUX Kontext INSTRUCTION editing: encode the reference image to a CLEAN latent and
    // pack it as fixed context tokens (image index 1) - the target starts from fresh noise.
    let kontext_ctx = if params.kontext {
        match &params.input_image {
            Some(b64) => {
                let clat = flux_encode_clean_latent(flux_state, b64, height, width)?
                    .to_device(device)?
                    .to_dtype(DType::F32)?;
                Some(crate::inference::model::flux::sampling::pack_context(
                    &clat,
                )?)
            }
            None => return Err(anyhow!("flux-kontext requires an input image")),
        }
    } else {
        None
    };

    // Prepare latent
    let (initial_img, timesteps) = if kontext_ctx.is_some() {
        let img = crate::inference::model::flux::sampling::get_noise(1, height, width, device)?
            .to_dtype(DType::F32)?;
        let timesteps =
            crate::inference::model::flux::sampling::get_schedule(params.num_steps, None);
        (img, timesteps)
    } else if let Some(ref input_b64) = params.input_image {
        info!("img2img mode: strength={}", params.strength);
        prepare_flux_img2img_latent(
            flux_state,
            device,
            input_b64,
            height,
            width,
            params.strength,
            params.num_steps,
        )?
    } else {
        let img = crate::inference::model::flux::sampling::get_noise(1, height, width, device)?
            .to_dtype(DType::F32)?;
        let timesteps =
            crate::inference::model::flux::sampling::get_schedule(params.num_steps, None);
        (img, timesteps)
    };

    let inpaint = build_flux_inpaint(flux_state, device, params, height, width)?;
    refuse_what_a_split_model_drops(&flux_state.flux, &inpaint, &regions)?;
    let state =
        crate::inference::model::flux::sampling::State::new(&t5_emb, &clip_emb, &initial_img)?;
    drop(t5_emb);
    drop(clip_emb);
    drop(initial_img);

    // Denoise. A whole placement casts the state in ONCE, walks the schedule, and casts the
    // final latent back ONCE for the VAE - no per-step cast.
    set_flux_loras(flux_state, &device.clone(), &params.loras)?;
    let denoise_t = std::time::Instant::now();
    // MEASURE WHAT THIS RENDER ACTUALLY TAKES, WHATEVER PATH IT RUNS ON.
    //
    // The reserve a placement leaves used to be an estimate that nothing corrected: the
    // feedback existed but hung off a per-step callback, so it fired on ONE variant of
    // ONE entry point and silently did nothing for the rest - including the SPLIT path,
    // which is the one the reserve sent there. Measured once it was wired: 4.29 GB
    // reserved at 1024^2 for a denoise that needs 2.61 GB. A reserve that high splits a
    // model that fits one card, and a split runs the cards in sequence.
    //
    // The watcher samples from outside, so no variant can be left out of it.
    //
    // ONLY A SINGLE-DEVICE RENDER MEASURES WHAT A SINGLE-DEVICE RESERVE NEEDS. Under a
    // split the card holds a fraction of the blocks, so the free VRAM it gives up is a
    // fraction too - and feeding that number back as the reserve is how a 1536 render
    // measured at 3.5 GB while split, was then placed whole on a card with 3.7 GB spare,
    // and ran out in `matmul` on its first steps. Observed directly, which is the only
    // reason this guard exists rather than a plausible argument for the other choice.
    let placement_is_whole = !matches!(flux_state.flux, FluxVariant::Hetero(_));
    // A SPLIT model revisits its own placement between steps. Blocks spilled to the host
    // under pressure otherwise stay there for every remaining step, however much VRAM the
    // subsystem that caused the pressure has since given back. The headroom is the same
    // demand the placement used, so a block only goes back onto a card that can still
    // hold the next step's activations.
    if let FluxVariant::Hetero(h) = &mut flux_state.flux {
        h.set_render_headroom(flux_runtime_demand(width, height));
    }
    let watch_gpu = gpu_index_of(device);
    let free_before = watch_gpu.and_then(crate::inference::place::vram_manager::free_on);
    let watch = watch_gpu.and_then(crate::inference::place::vram_manager::VramWatch::start);
    let denoised = match &mut flux_state.flux {
        FluxVariant::Whole(flux) => {
            let nstate = crate::inference::model::flux::sampling::NativeState::to_f32(&state)?;
            let out = if let Some((ctx, ctx_ids)) = &kontext_ctx {
                crate::inference::model::flux::sampling::denoise_kontext_native(
                    flux,
                    &nstate,
                    &ctx.to_dtype(DType::F32)?,
                    &ctx_ids.to_dtype(DType::F32)?,
                    &timesteps,
                    params.guidance,
                    Some(&params.cancel),
                )?
            } else {
                crate::inference::model::flux::sampling::denoise_native(
                    flux,
                    &nstate,
                    &timesteps,
                    params.guidance,
                    |_| {},
                    Some(&params.cancel),
                    params.step_reuse,
                    inpaint.as_ref(),
                    &regions,
                )?
            };
            out.to_device(device)?
        }
        FluxVariant::Hetero(hetero) => denoise_split(
            hetero,
            &state,
            kontext_ctx.as_ref(),
            &timesteps,
            params.guidance,
            &params.cancel,
        )?,
    }
    .to_dtype(dtype)?;
    if let FluxVariant::Hetero(h) = &mut flux_state.flux {
        h.set_render_headroom(0);
    }
    if let (Some(w), Some(before)) = (watch, free_before) {
        report_denoise_vram(width, height, before, w, placement_is_whole);
    }
    info!(
        "Flux denoise: {:.1}s ({} steps, {}x{})",
        denoise_t.elapsed().as_secs_f64(),
        params.num_steps,
        width,
        height
    );
    drop(state);

    // VAE decode. <=512^2 -> GPU (fast cuDNN). Large images (>=768^2) whose GPU decode
    // would OOM (~6 GB peak vs the resident Flux/T5) fall back to the CPU VAE.
    let vae_t = std::time::Instant::now();
    let big = width.max(height) > 512;
    let unpacked = crate::inference::model::flux::sampling::unpack(&denoised, height, width)?
        .to_dtype(DType::F32)?;
    drop(denoised);
    let (decoded, vae_where) = if big {
        // Prefer the WHOLE-image GPU decode: it is seamless, where tiles show faint
        // seams (the VAE mid-block's attention is global over the whole image).
        // But CHOOSE by free VRAM instead of discovering by failure: when an
        // allocation fails the tensor layer bounces that single op to the host
        // rather than returning an error, so a card with no room does not raise the
        // OOM this cascade waits for - it just ping-pongs the whole decode across
        // PCIe. Peak is the widest full-resolution feature map plus its 3x3 im2col.
        let gpu_lat = unpacked.to_device(&flux_state.vae_device)?;
        let whole_peak = vae_whole_decode_peak(width, height);
        let fits_whole = free_on_device(&flux_state.vae_device) >= whole_peak;
        // MEASURE the decode, and DECIDE NOTHING on it yet.
        //
        // Two formulas in this file estimate what a decode costs and they disagree by
        // more than three times - so a decode is admitted onto a card by one figure and
        // refused by the other, and neither was ever checked against a decode. Replacing
        // one guess with another is what stranded a render on the host; the number below
        // is what the next change should be built on instead.
        let vae_gpu = gpu_index_of(&flux_state.vae_device);
        let vae_watch = if fits_whole {
            vae_gpu.and_then(vram_watch_on)
        } else {
            None
        };
        let vae_free_before = vae_gpu.and_then(crate::inference::place::vram_manager::free_on);
        let whole = if fits_whole {
            flux_state.ae.decode(&gpu_lat)
        } else {
            info!(
                "Flux VAE: {:.1} GB whole-image decode does not fit; tiling",
                whole_peak as f64 / 1e9
            );
            Err(crate::tensor::Error::msg(
                "vae: no room for a whole-image decode",
            ))
        };
        if let (Some(w), Some(before)) = (vae_watch, vae_free_before) {
            report_vae_decode_cost("Flux", width, height, whole_peak, before, w, whole.is_ok());
        }
        // Size the tile from the VRAM that is actually free rather than fixing it at 64:
        // a card too tight for a 64-px tile used to skip straight to the CPU, when a
        // smaller tile would still have decoded on the device.
        let flux_free = free_on_device(&flux_state.vae_device);
        let flux_tile =
            choose_vae_tile(flux_free, whole_peak, width, height).unwrap_or(VAE_TILE_OVERLAP * 4);
        match whole {
            Ok(d) => (d, "GPU-whole"),
            Err(_) => match flux_state
                .ae
                .decode_tiled(&gpu_lat, flux_tile, VAE_TILE_OVERLAP)
            {
                Ok(d) => (d, "GPU-tiled"),
                Err(e) => match &flux_state.ae_cpu {
                    Some(ae_cpu) => {
                        warn!("Flux VAE GPU decode failed ({e}); CPU fallback");
                        (ae_cpu.decode(&unpacked.to_device(&Device::Cpu)?)?, "CPU")
                    }
                    None => return Err(anyhow!("Flux VAE GPU decode failed: {e}")),
                },
            },
        }
    } else {
        (
            flux_state
                .ae
                .decode(&unpacked.to_device(&flux_state.vae_device)?)?,
            "GPU",
        )
    };
    let _ = vae_where;

    let img_cpu = decoded_to_cpu_u8(decoded)?;
    let base64_png = tensor_to_png_base64(&img_cpu.squeeze(0)?)?;
    info!("Flux VAE: {:.1}s", vae_t.elapsed().as_secs_f64());
    info!(
        "Flux total: {:.1}s ({}x{}, {} steps)",
        pipeline_t.elapsed().as_secs_f64(),
        width,
        height,
        params.num_steps
    );
    Ok(base64_png)
}

/// Generate image with Flux pipeline (streaming with progress)
fn generate_flux_image_stream(
    flux_state: &mut FluxModelState,
    device: &Device,
    dtype: DType,
    prompt: &str,
    params: &ImageGenParams,
    tx: &tokio::sync::mpsc::Sender<ImageStreamEvent>,
) -> AnyResult<String> {
    let pipeline_t = std::time::Instant::now();
    let height = params.height;
    let width = params.width;

    let (t5_emb, clip_emb) = encode_flux_text(flux_state, device, prompt)?;
    release_text_encoder_if_needed(flux_state, width, height);
    // One encode per region. They go through the same prompt cache as the base, so a
    // repeated region costs nothing after the first step of the first render.
    let regions = build_flux_regions(
        flux_state, device, params, height, width, &t5_emb, &clip_emb,
    )?;

    let (initial_img, timesteps) = if let Some(ref input_b64) = params.input_image {
        info!("img2img stream: strength={}", params.strength);
        prepare_flux_img2img_latent(
            flux_state,
            device,
            input_b64,
            height,
            width,
            params.strength,
            params.num_steps,
        )?
    } else {
        let img = crate::inference::model::flux::sampling::get_noise(1, height, width, device)?
            .to_dtype(DType::F32)?;
        let timesteps =
            crate::inference::model::flux::sampling::get_schedule(params.num_steps, None);
        (img, timesteps)
    };

    let inpaint = build_flux_inpaint(flux_state, device, params, height, width)?;
    refuse_what_a_split_model_drops(&flux_state.flux, &inpaint, &regions)?;
    let state =
        crate::inference::model::flux::sampling::State::new(&t5_emb, &clip_emb, &initial_img)?;
    drop(t5_emb);
    drop(clip_emb);
    drop(initial_img);

    // Inline denoise loop for progress
    let actual_steps = timesteps.len().saturating_sub(1);
    // Initial "starting" event so the client's progress bar doesn't
    // sit at no-update for the first step's latency. MUST use the
    // post-strength-truncation actual_steps as total - img2img cuts
    // the schedule so `actual_steps = num_steps - start_step`, and
    // if we report num_steps here the bar visibly jumps to a smaller
    // total on the very next event.
    let _ = tx.blocking_send(ImageStreamEvent::Progress {
        completed: 0,
        total: actual_steps,
    });
    // The STREAMING path needs the adapters too: it is a second denoise loop, and an
    // adapter honoured on one endpoint but not the other is the kind of split behaviour
    // nobody finds until a render silently ignores their LoRA.
    set_flux_loras(flux_state, &device.clone(), &params.loras)?;

    // What this render actually takes, watched from outside the loop - the same
    // mechanism the non-streaming path uses, so the two entry points cannot disagree
    // about what the machine has learned. The per-step sampler this replaces read
    // through `vram_manager::probe`, which trims the CUDA mempools before it reads:
    // handing memory back to the driver between every step, for a number that is only
    // being observed.
    let watch_gpu = gpu_index_of(device);
    let free_before_denoise = watch_gpu.and_then(crate::inference::place::vram_manager::free_on);
    let watch = watch_gpu.and_then(crate::inference::place::vram_manager::VramWatch::start);
    // A whole placement runs the whole loop from a state cast once (one cast in and out per
    // generate, progress via the denoise_native callback); a split one keeps the inline
    // per-step loop below, which is also where its between-step repatriation would go.
    let img = if let FluxVariant::Whole(flux) = &flux_state.flux {
        let nstate = crate::inference::model::flux::sampling::NativeState::to_f32(&state)?;
        let mut step_t = std::time::Instant::now();
        let out = crate::inference::model::flux::sampling::denoise_native(
            flux,
            &nstate,
            &timesteps,
            params.guidance,
            |step| {
                info!(
                    "Flux step {}/{}: {:.1}s",
                    step + 1,
                    actual_steps,
                    step_t.elapsed().as_secs_f64()
                );
                // Track the LOW-WATER free VRAM across the denoise. The reserve this
                // placement leaves is a constant nobody has checked against a render:
                // too high and a model that fits one card gets split, which serialises
                // the cards and is the worst outcome available; too low and the denoise
                // OOMs mid-render. One NVML read per step is nothing next to a step, and
                // it turns the reserve into a measured number instead of an argument.
                step_t = std::time::Instant::now();
                let _ = tx.blocking_send(ImageStreamEvent::Progress {
                    completed: step + 1,
                    total: actual_steps,
                });
            },
            Some(&params.cancel),
            params.step_reuse,
            inpaint.as_ref(),
            &regions,
        )?;
        out.to_device(device)?
    } else {
        let b_sz = state.img.dim(0)?;
        let dev = state.img.device();
        let guidance_tensor = Tensor::full(params.guidance as f32, b_sz, &dev)?;
        // Everything the forward reads apart from the latent and the timestep is fixed for
        // the whole walk, so it is named once here instead of being projected out of the
        // state on every step.
        let (ids, txt, txt_ids, pooled) = (&state.img_ids, &state.txt, &state.txt_ids, &state.vec);
        let mut img = state.img.clone();

        for (step, window) in timesteps.windows(2).enumerate() {
            // Same per-step cancellation contract as the native loop: a dropped
            // client must not leave this render burning the GPU to completion.
            if params.cancel.is_cancelled() {
                return Err(anyhow!("flux: generation cancelled"));
            }
            let step_t = std::time::Instant::now();
            let (t_curr, t_prev) = (window[0], window[1]);
            let t_vec = Tensor::full(t_curr as f32, b_sz, &dev)?;
            let pred = {
                use crate::inference::model::flux::sampling::WithForward;
                match &flux_state.flux {
                    // Handled by the whole-loop branch above.
                    FluxVariant::Whole(_) => unreachable!("a whole placement denoises above"),
                    FluxVariant::Hetero(hetero) => hetero.forward(
                        &img,
                        ids,
                        txt,
                        txt_ids,
                        &t_vec,
                        pooled,
                        Some(&guidance_tensor),
                    )?,
                }
            };
            img = (img + pred * (t_prev - t_curr))?;
            info!(
                "Flux step {}/{}: {:.1}s",
                step + 1,
                actual_steps,
                step_t.elapsed().as_secs_f64()
            );
            let _ = tx.blocking_send(ImageStreamEvent::Progress {
                completed: step + 1,
                total: actual_steps,
            });
        }
        img
    };

    if let (Some(w), Some(before)) = (watch, free_before_denoise) {
        report_denoise_vram(
            width,
            height,
            before,
            w,
            !matches!(flux_state.flux, FluxVariant::Hetero(_)),
        );
    }
    // VAE decode. This used to read `if width.max(height) > 512 && a CPU VAE exists ->
    // CPU`, unconditionally: a STATIC SIZE GATE, with no idea whether the GPU had room.
    // So every edit above 512 decoded on the host even with both cards nearly empty,
    // which is minutes instead of seconds - the same defect the txt2img path already
    // had fixed above. Decide by free VRAM, and tile before leaving the device.
    let vae_t = std::time::Instant::now();
    let denoised = img.to_dtype(dtype)?;
    drop(state);
    let (decoded, vae_where) = {
        let whole_peak = vae_whole_decode_peak(width, height);
        let free_here = free_on_device(&flux_state.vae_device);
        let fits_whole = free_here >= whole_peak;
        // Same bound as the Z-Image path: tile only at 2x2 or coarser, which measured
        // clean, and leave anything finer to the CPU.
        let tile = if fits_whole || !flux_state.vae_device.is_cuda() {
            None
        } else {
            choose_vae_tile(free_here, whole_peak, width, height)
        };
        if fits_whole || tile.is_some() || flux_state.ae_cpu.is_none() {
            let unpacked =
                crate::inference::model::flux::sampling::unpack(&denoised, height, width)?
                    .to_dtype(DType::F32)?
                    .to_device(&flux_state.vae_device)?;
            drop(denoised);
            match tile {
                Some(t) => {
                    info!(
                        "Flux VAE (edit): {:.1} GB whole-image decode does not fit; tiling at                          {t} latent px",
                        whole_peak as f64 / 1e9
                    );
                    (
                        flux_state.ae.decode_tiled(&unpacked, t, VAE_TILE_OVERLAP)?,
                        "GPU-tiled",
                    )
                }
                None => (flux_state.ae.decode(&unpacked)?, "GPU-whole"),
            }
        } else {
            // Nothing on the device can hold even the smallest tile, and only THEN is
            // the resident CPU VAE the right answer rather than a slow reflex.
            let unpacked =
                crate::inference::model::flux::sampling::unpack(&denoised, height, width)?
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?;
            drop(denoised);
            let ae = flux_state
                .ae_cpu
                .as_ref()
                .ok_or_else(|| crate::tensor::Error::msg("the CPU VAE is not resident"))?;
            (ae.decode(&unpacked)?, "CPU")
        }
    };
    let img_cpu = decoded_to_cpu_u8(decoded)?;
    info!(
        "Flux VAE: {:.1}s ({vae_where})",
        vae_t.elapsed().as_secs_f64()
    );
    info!(
        "Flux total: {:.1}s ({}x{}, {} steps)",
        pipeline_t.elapsed().as_secs_f64(),
        width,
        height,
        params.num_steps
    );
    tensor_to_png_base64(&img_cpu.squeeze(0)?)
}

/// Convert a (C, H, W) U8 tensor to base64-encoded PNG
fn tensor_to_png_base64(img: &Tensor) -> AnyResult<String> {
    let (c, h, w) = img.dims3()?;
    // Returning Err is the right shape - `assert!` would panic the
    // request handler thread and crash the connection rather than
    // produce a clean 500.
    if c != 3 {
        return Err(anyhow!(
            "tensor_to_png_base64: expected 3-channel CHW tensor, got C={c}"
        ));
    }
    let img_data = img.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
    let image_buf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        image::ImageBuffer::from_raw(w as u32, h as u32, img_data)
            .ok_or_else(|| anyhow!("Failed to create image buffer"))?;
    // Pre-size the PNG buffer to the raw pixel byte count. Real-world
    // PNG compression on a photographic image lands between 0.5x and
    // 2x of raw, so this avoids a few mid-encode realloc copies on
    // the 1024^2 path (3 MB raw) without significantly over-allocating
    // on the small-image case.
    let raw_bytes = h * w * 3;
    let mut png_bytes: Vec<u8> = Vec::with_capacity(raw_bytes);
    use std::io::Cursor;
    image_buf.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)?;
    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.encode(&png_bytes))
}

/// Decode a base64-encoded image to a normalized tensor (1, C, H, W) in [-1, 1] range.
/// Resizes to target dimensions if provided.
fn decode_base64_to_tensor(
    base64_str: &str,
    target_h: usize,
    target_w: usize,
    device: &Device,
) -> AnyResult<Tensor> {
    let img_rgb = decode_base64_fitted(
        base64_str,
        target_w,
        target_h,
        image::imageops::FilterType::Lanczos3,
        "Base64 decode",
        "Image decode",
    )?
    .to_rgb8();

    // Convert to (1, 3, H, W) float tensor normalized to [-1, 1]
    let raw = img_rgb.into_raw();
    let tensor = Tensor::from_vec(
        raw.iter()
            .map(|&v| (v as f32 / 127.5) - 1.0)
            .collect::<Vec<f32>>(),
        (target_h, target_w, 3),
        device,
    )?
    .permute((2, 0, 1))? // (H, W, C) -> (C, H, W)
    .unsqueeze(0)?; // (C, H, W) -> (1, C, H, W)

    Ok(tensor)
}

/// Encode an input image to Flux latent space via VAE, then add noise for img2img.
///
/// Pulls the pre-noise latent from the FluxModelState VAE LRU cache when
/// the same input + dimensions were seen recently - saves the base64
/// decode + VAE encode on repeated iterations against one source image.
fn prepare_flux_img2img_latent(
    state: &mut FluxModelState,
    device: &Device,
    input_base64: &str,
    height: usize,
    width: usize,
    strength: f64,
    num_steps: usize,
) -> AnyResult<(Tensor, Vec<f64>)> {
    // Cache key splits on dimensions: a cached 512^2 latent has the
    // wrong shape to mix with 1024^2 noise and would crash here.
    let cache_key = vae_latent_cache_key(input_base64, height, width);
    let latent = if let Some(cached) = state.vae_latent_cache.get(&cache_key) {
        info!("img2img: VAE latent cached (skipping decode+encode)");
        cached
    } else {
        let img_tensor = decode_base64_to_tensor(input_base64, height, width, &Device::Cpu)?
            .to_dtype(DType::F32)?;
        info!("img2img: input tensor shape {:?}", img_tensor.shape());
        let latent = state.ae.encode(&img_tensor)?;
        drop(img_tensor);
        info!("img2img: latent shape {:?}", latent.shape());
        state.vae_latent_cache.insert(cache_key, latent.clone());
        latent
    };

    // REVERTED alongside the Z-Image one: the same reshaping went into both families in
    // the same deploy, so both are restored until the Edit fault is reproduced. The
    // known cost of restoring it is documented on get_schedule_from - the control is
    // quantised, and below ~0.4 strength on a 4-step schedule this errors instead of
    // editing gently.
    let full_schedule = crate::inference::model::flux::sampling::get_schedule(num_steps, None);
    let start_step = ((1.0 - strength) * num_steps as f64).round() as usize;
    let timesteps: Vec<f64> = full_schedule[start_step..].to_vec();
    if timesteps.len() < 2 {
        return Err(anyhow!(
            "Strength too low ({strength}): only {} timestep(s), need at least 2",
            timesteps.len()
        ));
    }
    let t_start = timesteps[0] as f32;
    let noise = crate::inference::model::flux::sampling::get_noise(1, height, width, &Device::Cpu)?
        .to_dtype(DType::F32)?;
    let noisy_latent = ((latent * (1.0 - t_start) as f64)? + (noise * t_start as f64)?)?;
    let noisy_latent = noisy_latent.to_device(device)?.to_dtype(DType::F32)?;

    info!(
        "img2img: strength={strength}, start_step={start_step}, steps={}",
        timesteps.len() - 1
    );
    Ok((noisy_latent, timesteps))
}

/// Refuse what a split model cannot honour, rather than dropping it.
///
/// Neither denoise loop for a model spread across cards takes an inpaint state or a set of
/// regions - the whole-model loop does, and the split one does not. Both were built before the
/// branch either way, so on a split model the mask was decoded, the region embeddings were
/// encoded, and then both were let fall: the request came back as an ordinary render carrying
/// none of what was asked for, and nothing said so.
fn refuse_what_a_split_model_drops(
    flux: &FluxVariant,
    inpaint: &Option<crate::inference::model::flux::sampling::Inpaint>,
    regions: &[crate::inference::model::flux::sampling::Region],
) -> AnyResult<()> {
    if !matches!(flux, FluxVariant::Hetero(_)) {
        return Ok(());
    }
    if inpaint.is_some() {
        return Err(anyhow!(
            "flux: a mask needs the model whole on one card, and this one is split across \
             several - retry without `mask`, or on a card that holds the model"
        ));
    }
    if !regions.is_empty() {
        return Err(anyhow!(
            "flux: regional prompts need the model whole on one card, and this one is split \
             across several - retry without `regions`, or on a card that holds the model"
        ));
    }
    Ok(())
}

/// Assemble the sampler's inpaint state from the request, or `None` when there is no
/// mask to honour.
///
/// Requires a source image: a mask says WHERE to change something, which is only
/// meaningful relative to an existing picture.
fn build_flux_inpaint(
    state: &mut FluxModelState,
    device: &Device,
    params: &ImageGenParams,
    height: usize,
    width: usize,
) -> AnyResult<Option<crate::inference::model::flux::sampling::Inpaint>> {
    let Some(mask_b64) = params.mask.as_deref() else {
        return Ok(None);
    };
    let Some(src_b64) = params.input_image.as_deref() else {
        return Err(anyhow!(
            "`mask` needs an input image: it marks where to change THAT picture"
        ));
    };
    let keep = inpaint_keep_weights(mask_b64, height, width)?;
    // A mask that keeps everything is a request for no change at all; say so rather
    // than spending a full render to return the input.
    if keep.iter().all(|k| *k > 0.999) {
        return Err(anyhow!(
            "`mask` marks nothing to change - the whole image is masked as keep"
        ));
    }
    let clean = flux_encode_clean_latent(state, src_b64, height, width)?;
    let packed = crate::inference::model::flux::sampling::pack(&clean)?;
    let noise = crate::inference::model::flux::sampling::get_noise(1, height, width, device)?
        .to_dtype(DType::F32)?;
    let noise = crate::inference::model::flux::sampling::pack(&noise)?;
    let seq = packed.dim(1)?;
    if keep.len() != seq {
        return Err(anyhow!(
            "mask resolved to {} tokens but the latent has {seq}",
            keep.len()
        ));
    }
    let keep_t = Tensor::from_vec(keep, (1, seq, 1), device)?.to_dtype(DType::F32)?;
    Ok(Some(crate::inference::model::flux::sampling::Inpaint {
        clean: packed.to_device(device)?.to_dtype(DType::F32)?,
        noise: noise.to_device(device)?.to_dtype(DType::F32)?,
        keep: keep_t.to_dtype(DType::F32)?,
    }))
}

/// Turn a mask image into the per-token keep weights the sampler needs.
///
/// Convention, matching the OpenAI edit endpoint: the mask marks what to CHANGE.
/// Transparent (or black, when there is no alpha) means "regenerate here"; opaque
/// means "leave alone". Alpha wins when present, because that is what an image editor
/// produces when you erase a region.
///
/// The mask is reduced to the latent TOKEN grid - one weight per 16x16 block of the
/// image - by averaging, so a soft edge in the mask stays soft in the blend instead of
/// being quantised to a hard step.
fn inpaint_keep_weights(mask_b64: &str, height: usize, width: usize) -> AnyResult<Vec<f32>> {
    let img = decode_base64_fitted(
        mask_b64.trim(),
        width,
        height,
        image::imageops::FilterType::Triangle,
        "mask is not valid base64",
        "cannot decode the mask",
    )?
    .to_rgba8();
    let (tw, th) = (width / 16, height / 16);
    if tw == 0 || th == 0 {
        return Err(anyhow!("image too small to inpaint"));
    }
    let has_alpha = img.pixels().any(|p| p.0[3] != 255);
    let mut keep = vec![0f32; tw * th];
    for ty in 0..th {
        for tx in 0..tw {
            let mut sum = 0f32;
            for y in ty * 16..(ty + 1) * 16 {
                for x in tx * 16..(tx + 1) * 16 {
                    let p = img.get_pixel(x as u32, y as u32).0;
                    // 1 where the source is KEPT.
                    sum += if has_alpha {
                        f32::from(p[3]) / 255.0
                    } else {
                        f32::from(p[0]) / 255.0
                    };
                }
            }
            keep[ty * tw + tx] = sum / 256.0;
        }
    }
    Ok(keep)
}

#[cfg(test)]
mod inpaint_tests {
    use super::*;

    /// Build a PNG mask: an opaque frame with a transparent hole, which is what an
    /// image editor produces when you erase a region.
    fn mask_png(w: u32, h: u32, hole: (u32, u32, u32, u32)) -> String {
        use base64::Engine as _;
        let (hx, hy, hw, hh) = hole;
        let mut img = image::RgbaImage::from_pixel(w, h, image::Rgba([255, 255, 255, 255]));
        for y in hy..(hy + hh).min(h) {
            for x in hx..(hx + hw).min(w) {
                img.put_pixel(x, y, image::Rgba([0, 0, 0, 0]));
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(img)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode");
        base64::engine::general_purpose::STANDARD.encode(png.into_inner())
    }

    /// The weights must be 1 where the source is kept and 0 inside the hole, at the
    /// LATENT token grid. Getting the polarity backwards regenerates the whole picture
    /// except the part the user selected - a failure that looks like the feature simply
    /// not working, with no error.
    #[test]
    fn a_transparent_hole_marks_the_region_to_regenerate() {
        let (w, h) = (256usize, 256usize);
        // Hole over the top-left quadrant.
        let b64 = mask_png(w as u32, h as u32, (0, 0, 128, 128));
        let keep = inpaint_keep_weights(&b64, h, w).expect("weights");
        let (tw, th) = (w / 16, h / 16);
        assert_eq!(keep.len(), tw * th, "one weight per latent token");
        // Inside the hole: regenerate.
        assert!(keep[0] < 0.01, "the hole must not be kept ({})", keep[0]);
        assert!(keep[(th / 2 - 1) * tw + (tw / 2 - 1)] < 0.01);
        // Outside: keep.
        assert!(
            keep[tw - 1] > 0.99,
            "outside the hole must be kept ({})",
            keep[tw - 1]
        );
        assert!(keep[(th - 1) * tw + tw - 1] > 0.99);
    }

    /// Without an alpha channel the luminance decides, so a plain black-on-white mask
    /// drawn in any editor works too.
    #[test]
    fn an_opaque_black_and_white_mask_uses_luminance() {
        use base64::Engine as _;
        let (w, h) = (128u32, 128u32);
        let mut img = image::RgbImage::from_pixel(w, h, image::Rgb([255, 255, 255]));
        for y in 0..64 {
            for x in 0..64 {
                img.put_pixel(x, y, image::Rgb([0, 0, 0]));
            }
        }
        let mut png = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut png, image::ImageFormat::Png)
            .expect("encode");
        let b64 = base64::engine::general_purpose::STANDARD.encode(png.into_inner());
        let keep = inpaint_keep_weights(&b64, h as usize, w as usize).expect("weights");
        let tw = w as usize / 16;
        assert!(keep[0] < 0.01, "black means regenerate");
        assert!(keep[tw - 1] > 0.99, "white means keep");
    }

    /// A soft edge must survive into the blend rather than being rounded to a hard
    /// step - that softness is what hides the seam.
    #[test]
    fn a_partially_covered_token_gets_a_partial_weight() {
        // A hole 8 px wide covers exactly half of the first 16 px token column.
        let b64 = mask_png(64, 64, (0, 0, 8, 64));
        let keep = inpaint_keep_weights(&b64, 64, 64).expect("weights");
        assert!(
            (keep[0] - 0.5).abs() < 0.05,
            "a half-covered token should weigh about 0.5, got {}",
            keep[0]
        );
    }

    #[test]
    fn a_malformed_mask_is_reported() {
        assert!(inpaint_keep_weights("not base64!!", 256, 256).is_err());
        use base64::Engine as _;
        let junk = base64::engine::general_purpose::STANDARD.encode(b"not an image");
        assert!(inpaint_keep_weights(&junk, 256, 256).is_err());
    }
}

/// VAE-encode the reference image to its CLEAN latent (no noising) for FLUX Kontext
/// instruction editing - reuses the same encoder + latent LRU as img2img.
fn flux_encode_clean_latent(
    state: &mut FluxModelState,
    input_base64: &str,
    height: usize,
    width: usize,
) -> AnyResult<Tensor> {
    let cache_key = vae_latent_cache_key(input_base64, height, width);
    if let Some(cached) = state.vae_latent_cache.get(&cache_key) {
        return Ok(cached);
    }
    let img_tensor =
        decode_base64_to_tensor(input_base64, height, width, &Device::Cpu)?.to_dtype(DType::F32)?;
    let latent = state.ae.encode(&img_tensor)?;
    state.vae_latent_cache.insert(cache_key, latent.clone());
    Ok(latent)
}

/// Format prompt for Qwen3 chat template (Z-Image text encoder)
fn format_prompt_for_qwen3(prompt: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        prompt
    )
}

/// Generate image with Z-Image pipeline (streaming or non-streaming)
/// Largest tile whose decode fits `free`, or `None` when tiling cannot help.
///
/// Biggest-first: a larger tile means fewer seams and a wider mid-block attention, so
/// the smallest tile is a last resort rather than a default. `None` means either the
/// image is already no bigger than a tile (splitting it would buy nothing) or even the
/// smallest tile does not fit - and the caller must then fall back, not tile.
fn choose_vae_tile(free: u64, peak: u64, width: usize, height: usize) -> Option<usize> {
    let (hl, wl) = (height / 8, width / 8);
    // Leave a fifth of the free space for fragmentation and the stitch buffer.
    let budget = free - free / 5;
    let pixels = ((width * height) as u64).max(1);
    // NEVER cut finer than 2x2. Measured on a real encoded photograph, the bias each
    // tile picks up - every GroupNorm in the up-path takes its statistics over the
    // extent it is handed, so smaller tiles sit further from the whole image's - grows
    // sharply as the split gets finer:
    //
    //     2x2  spread 0.0067   (~0.85 of 255 levels between neighbours: invisible)
    //     3x3         0.0185
    //     4x4         0.0338   (~4.3 levels: visible blocking on skin and sky)
    //
    // So a coarse split is a legitimate way to keep a decode on the device, and a fine
    // one is a quality bug. Below this floor the answer is the CPU, not more tiles.
    let coarsest = hl.max(wl).div_ceil(2);
    [128usize, 96, 64, 48, 32].into_iter().find(|&t| {
        if t >= hl.max(wl) || t < coarsest {
            return false; // not a split at all, or finer than 2x2
        }
        let side = ((t + 2 * VAE_TILE_OVERLAP) * 8) as u64;
        peak.saturating_mul(side * side) / pixels <= budget
    })
}

/// Which GPU (if any) can host the VAE for a `width x height` decode.
///
/// Two tiers, and the second is the point: the whole-image peak is a PREFERENCE,
/// while the tiled floor - the weights plus the smallest tile's working set - is
/// the actual requirement. Charging the full peak is what stranded the VAE on the
/// CPU with an idle card 130 MB short of it.
///
/// `None` means no card can host it even tiled, and only then is the CPU right.
pub(crate) fn pick_vae_gpu(
    live_free: &[(usize, u64)],
    weight_bytes: u64,
    width: usize,
    height: usize,
) -> Option<usize> {
    let peak = (width * height) as u64 * 4 * (VAE_WIDEST_CH * 9 + VAE_WIDEST_CH * 2);
    const SMALLEST_TILE_SIDE: u64 = (32 + 2 * VAE_TILE_OVERLAP as u64) * 8;
    let floor = weight_bytes + SMALLEST_TILE_SIDE * SMALLEST_TILE_SIDE * VAE_WIDEST_CH * 8 * 4;
    pick_aux_device(live_free, peak, None).or_else(|| pick_aux_device(live_free, floor, None))
}

/// Move a CPU-stranded Z-Image VAE back onto a GPU when one has room again.
///
/// Only ever CPU -> GPU: a VAE already on a GPU stays there, and a failure to reload
/// leaves the working CPU VAE exactly as it was, so the worst case is the status quo.
fn repatriate_zimage_vae(zimg: &mut ZImageModelState, width: usize, height: usize) {
    if zimg.vae_device.is_cuda() {
        return;
    }
    let weights = std::fs::metadata(&zimg.vae_path)
        .map(|m| m.len())
        .unwrap_or(0)
        * 2;
    let live_free: Vec<(usize, u64)> = crate::inference::place::vram_manager::probe(0)
        .into_iter()
        .map(|(i, f, _)| (i, f))
        .collect();
    let Some(gpu) = pick_vae_gpu(&live_free, weights, width, height) else {
        return;
    };
    let Ok(dev) = crate::tensor::Device::new_cuda(gpu) else {
        return;
    };
    let ndev = dev.clone();
    let Ok(path) = zimg.vae_path.clone().into_os_string().into_string() else {
        return;
    };
    let built = unsafe {
        crate::tensor::VarBuilder::from_files(&[path.as_str()], crate::tensor::DType::F32, &ndev)
    }
    .and_then(|vb| {
        crate::inference::model::zimage::vae::AutoEncoderKL::new(
            &crate::inference::model::zimage::vae::VaeConfig::z_image(),
            vb,
        )
    });
    match built {
        Ok(gpu_vae) => {
            info!("Z-Image VAE: VRAM freed up - moving the VAE from the CPU back to CUDA:{gpu}");
            // The CPU copy becomes the fallback instead of being dropped: it is already
            // paid for, and it is exactly what the decode cascade wants underneath.
            let cpu_vae = std::mem::replace(&mut zimg.vae, gpu_vae);
            if zimg.vae_cpu.is_none() {
                zimg.vae_cpu = Some(cpu_vae);
            }
            zimg.vae_device = dev;
            zimg.vae_dtype = DType::F32;
        }
        Err(e) => {
            // Nothing moved; the CPU VAE is untouched and the decode proceeds as before.
            warn!("Z-Image VAE: repatriation to CUDA:{gpu} failed ({e}); staying on the CPU");
        }
    }
}

/// One scheduler step, on the latent layout the transformer works in.
///
/// The prediction is negated before it is applied, and the frame axis the transformer
/// carries is dropped for the scheduler and put back afterwards. Both denoise loops take
/// exactly this step, so it is written here rather than at each of them: a difference
/// between the two would be a difference in the picture, with nothing to show it.
fn zimage_euler_step(
    scheduler: &mut crate::inference::model::zimage::sampling::FlowMatchEulerDiscreteScheduler,
    noise_pred: &Tensor,
    latents: &Tensor,
) -> AnyResult<Tensor> {
    let step_pred = noise_pred.neg()?.squeeze(2)?;
    let frameless = latents.squeeze(2)?;
    Ok(scheduler.step(&step_pred, &frameless)?.unsqueeze(2)?)
}

fn generate_zimage(
    device: &Device,
    dtype: DType,
    zimg: &mut ZImageModelState,
    prompt: &str,
    params: &ImageGenParams,
    tx: Option<&tokio::sync::mpsc::Sender<ImageStreamEvent>>,
) -> AnyResult<String> {
    // Z-Image never asked for the reduced-precision paths: the render this engine is held to
    // is the one a freshly started process produces. Saying so here is what stops another
    // family's load from choosing otherwise.

    let height = params.height;
    let width = params.width;
    let num_steps = params.num_steps;

    // Validate dimensions (must be divisible by 16)
    let vae_align = 16;
    if !height.is_multiple_of(vae_align) || !width.is_multiple_of(vae_align) {
        return Err(anyhow!(
            "Image dimensions must be divisible by {vae_align}. Got {width}x{height}"
        ));
    }

    // 1. Tokenize and encode text (text encoder may be on different device).
    // LRU cache keyed by the formatted prompt: a hit saves the 50-200 ms
    // encoder pass plus the device transfer. cap_mask is cached alongside
    // cap_feats so a cached generate skips the per-call ones allocation.
    let formatted = format_prompt_for_qwen3(prompt);
    let (cap_feats, cap_mask) = if let Some((feats, mask)) = zimg.text_cache.get(formatted.as_str())
    {
        info!("Z-Image: text cached (skipping encoder)");
        (feats, mask)
    } else {
        info!("Z-Image: encoding text...");
        let tokens = encode_ids(&zimg.tokenizer, formatted.as_str(), "Z-Image")?;
        let count = tokens.len();
        let input_ids = Tensor::from_vec(tokens, (1, count), &zimg.te_device)?;
        let feats = zimg
            .text_encoder
            .forward(&input_ids)?
            .to_device(device)?
            .to_dtype(dtype)?;
        // All tokens attended: a (1, count) U8 ones mask. Built once per
        // unique prompt and re-served from cache on subsequent calls.
        let mask = Tensor::ones((1, count), DType::U8, device)?;
        // Cache for next call. Evicted entries drop via Arc Drop.
        zimg.text_cache
            .insert(formatted.clone(), (feats.clone(), mask.clone()));
        info!("Z-Image: text encoded, {count} tokens");
        (feats, mask)
    };

    // 2. Compute latent dimensions and scheduler shift
    let patch_size = zimg.transformer_cfg.all_patch_size[0];
    let latent_h = 2 * (height / vae_align);
    let latent_w = 2 * (width / vae_align);
    let image_seq_len = (latent_h / patch_size) * (latent_w / patch_size);
    let mu = crate::inference::model::zimage::sampling::calculate_shift(
        image_seq_len,
        ZIMAGE_BASE_SEQ_LEN,
        ZIMAGE_MAX_SEQ_LEN,
        ZIMAGE_BASE_SHIFT,
        ZIMAGE_MAX_SHIFT,
    );
    info!(
        "Z-Image: latent {}x{}, seq_len={}, mu={:.4}",
        latent_w, latent_h, image_seq_len, mu
    );

    // 3. Initialize scheduler and noise
    let mut scheduler =
        crate::inference::model::zimage::sampling::FlowMatchEulerDiscreteScheduler::new(
            zimg.scheduler_cfg.clone(),
        );
    scheduler.set_timesteps(num_steps, Some(mu));

    let mut latents =
        crate::inference::model::zimage::sampling::get_noise(1, 16, latent_h, latent_w, device)?
            .to_dtype(dtype)?;
    // Add frame dimension: (B, C, H, W) -> (B, C, 1, H, W)
    latents = latents.unsqueeze(2)?;

    // 3b. img2img: if the caller provided an input image, encode it to
    // latent space via the VAE, mix with the noise from step 3 at the
    // current sigma, and advance the scheduler to the matching step
    // index. Strength controls how much of the original image survives:
    //   1.0 = pure txt2img (start from full noise - no input bias)
    //   0.0 = preserve original (clamped - at least 1 step still runs)
    //
    // For Z-Image's flow-matching scheduler, the noised latent is
    //   x_t = (1 - sigma_t) * latent + sigma_t * noise
    // We pick sigma_t at step_index = round((1 - strength) * num_steps).
    let start_step = if let Some(ref input_b64) = params.input_image {
        info!("Z-Image img2img mode: strength={}", params.strength);
        let strength = params.strength.clamp(0.0, 1.0);
        // LRU cache: same input image at same dimensions -> skip the
        // base64 decode + VAE encode (~50-200 ms total at 1024^2). Key
        // is a hash of the b64 string so the user's image bytes never
        // sit verbatim in process memory beyond decode lifetime.
        let cache_key = vae_latent_cache_key(input_b64, height, width);
        let init_latent = if let Some(cached) = zimg.vae_latent_cache.get(&cache_key) {
            info!("Z-Image img2img: VAE latent cached (skipping decode+encode)");
            cached
        } else {
            // Decode the user image -> (1, 3, H, W) in [-1, 1] on the VAE's
            // device + dtype (the VAE stays on its preferred placement;
            // see vae_fits_on_gpu probe at load time).
            let img_tensor = decode_base64_to_tensor(input_b64, height, width, &zimg.vae_device)?
                .to_dtype(zimg.vae_dtype)?;
            // VAE encode -> (1, 16, latent_h, latent_w). Move to the
            // transformer device + dtype before mixing with noise.
            let latent = zimg
                .vae
                .encode(&img_tensor)?
                .to_device(device)?
                .to_dtype(dtype)?
                .unsqueeze(2)?; // add frame dim to match latents shape
            drop(img_tensor);
            zimg.vae_latent_cache.insert(cache_key, latent.clone());
            latent
        };
        // REVERTED to the earlier ladder start, on a user report that Image Edit
        // "generates anything" with this family.
        //
        // The replacement rebuilt the ladder from the requested sigma, which fixed a
        // real defect - the control had nine reachable settings and 0.95 shared a node
        // with 1.00, where the source is discarded outright. But it is not the shape
        // this sampler was validated against, the fault could not be reproduced by
        // reading it (under this checkpoint's config, shift=1.0, the rebuild reduces to
        // the same linear ramp), and a broken editor now outranks a coarse dial. The
        // rebuild and its tests are kept in the scheduler; reinstating them needs a
        // reproduction, not another argument.
        // The img2img schedule: build for steps/denoise and keep the TAIL rather than
        // rescaling, so the run is the caller's step count starting
        // at the requested noise level, on the model's own schedule shape.
        scheduler.set_timesteps_denoised(num_steps, Some(mu), strength);
        let start = 0usize;
        let sigma_start = scheduler.current_sigma();
        info!("Z-Image img2img: start_step={start}/{num_steps} sigma={sigma_start:.4}",);
        // noisy = (1 - sigma) * init + sigma * noise. Flow-matching mixes
        // the data and noise linearly along the sigma axis.
        let one_minus_sigma = 1.0 - sigma_start;
        latents = ((init_latent * one_minus_sigma)? + (latents * sigma_start)?)?;
        start
    } else {
        0
    };

    // 4. Denoise loop. For img2img, start_step > 0 skips the early
    // (high-noise) iterations because the input image already lives
    // at the corresponding sigma - saving (1 - strength) * num_steps
    // worth of transformer forwards. Progress events report against
    // the REAL step count remaining so the client's progress bar
    // doesn't jump to a non-zero position on first paint.
    let remaining_steps = num_steps - start_step;
    info!("Z-Image: denoising ({remaining_steps} steps, starting at {start_step}/{num_steps})...");
    // WHAT THE CARDS ACTUALLY GIVE UP for this denoise, beside what the walk counted.
    //
    // The walk counts the live set the tensor layer sees; a card reports what its
    // driver holds, and the two are not the same number - the driver hands out 2 MiB
    // pages, an op can allocate below the tensor layer, and the context and its
    // modules are there before any of it. There was no evidence for the SIZE of that
    // difference on this family because nothing measured it: the watch this uses is
    // the same one the VAE decode and the Flux renders already report through, and it
    // was simply never put around this loop.
    //
    // REPORTED, NOT FED BACK. No reserve, no placement and no admission reads this;
    // it is a line in the log, so that the factor those decisions do use can be
    // corrected against measurements rather than against another estimate.
    let denoise_watch: Vec<(usize, u64, crate::inference::place::vram_manager::VramWatch)> = {
        #[cfg(feature = "cuda")]
        {
            crate::inference::place::device_probe::probe_cuda_gpus(1.0)
                .into_iter()
                .filter_map(|g| {
                    Some((
                        g.index,
                        g.stable_free,
                        crate::inference::place::vram_manager::VramWatch::start(g.index)?,
                    ))
                })
                .collect()
        }
        #[cfg(not(feature = "cuda"))]
        Vec::new()
    };
    // Initial "starting" event: 0/remaining_steps so the client's
    // progress bar paints something during the first step's latency
    // (~50-200 ms on warm cache, longer on cold). Matches the Flux
    // streaming path post-fix in this commit.
    if let Some(tx) = tx {
        let _ = tx.blocking_send(ImageStreamEvent::Progress {
            completed: 0,
            total: remaining_steps,
        });
    }
    // NativeSingle runs the whole loop on the native substrate: latents +
    // text-encoder products are bridged in ONCE here, every per-step tensor
    // (timestep, DiT forward, negate, scheduler Euler update) stays native,
    // and the final latent is bridged back ONCE for the facade VAE - no
    // per-step bridge round-trip (the retired ZImageBridge). The facade
    // variants keep the loop below. The scheduler is the same object in both:
    // this loop carries F32 latents (the native activation dtype) where the
    // other carries BF16 - a strict precision upgrade on the same arithmetic.
    if let ZImageVariant::NativeSingle(model) = &zimg.transformer {
        let mut nlatents = latents.to_dtype(DType::F32)?;
        let ncap_feats = cap_feats.to_dtype(DType::F32)?;
        let ncap_mask = cap_mask.to_dtype(DType::F32)?;
        let ndev = nlatents.device();
        for step in start_step..num_steps {
            // Same per-step cancellation contract as the Flux loops: this runs in
            // spawn_blocking, so dropping the request future - a disconnected
            // client, or the server's own request timeout - cannot stop it. Without
            // this check an abandoned render kept a device busy long after the
            // caller received its 408.
            if params.cancel.is_cancelled() {
                return Err(anyhow!("z-image: generation cancelled"));
            }
            let t = scheduler.current_timestep_normalized();
            let nt = crate::tensor::Tensor::from_vec_f32(vec![t as f32], 1)?.to_device(&ndev)?;

            let noise_pred = model.forward(&nlatents, &nt, &ncap_feats, &ncap_mask)?;
            nlatents = zimage_euler_step(&mut scheduler, &noise_pred, &nlatents)?;

            let done_in_loop = step - start_step + 1;
            debug!("Z-Image step {done_in_loop}/{remaining_steps}: t={t:.4} (native)");
            if let Some(tx) = tx {
                let _ = tx.blocking_send(ImageStreamEvent::Progress {
                    completed: done_in_loop,
                    total: remaining_steps,
                });
            }
        }
        latents = nlatents.to_device(device)?.to_dtype(dtype)?;
    } else {
        for step in start_step..num_steps {
            // See the native loop above: cancellation must be polled per step.
            if params.cancel.is_cancelled() {
                return Err(anyhow!("z-image: generation cancelled"));
            }
            let t = scheduler.current_timestep_normalized();
            let t_tensor = Tensor::from_vec(vec![t as f32], (1,), device)?.to_dtype(dtype)?;

            let noise_pred = match &zimg.transformer {
                // Handled by the native whole-loop branch above.
                ZImageVariant::NativeSingle(_) => unreachable!("native z-image denoises above"),
                ZImageVariant::Single(model) => {
                    model.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?
                }
                ZImageVariant::Hetero(hetero) => {
                    hetero.forward(&latents, &t_tensor, &cap_feats, &cap_mask)?
                }
            };

            // CFG would require a negative-prompt forward + interpolation;
            // we don't run unconditional inference here, so the conditional
            // noise_pred is used directly regardless of params.guidance.
            latents = zimage_euler_step(&mut scheduler, &noise_pred, &latents)?;

            let done_in_loop = step - start_step + 1;
            debug!("Z-Image step {done_in_loop}/{remaining_steps}: t={t:.4}");
            if let Some(tx) = tx {
                let _ = tx.blocking_send(ImageStreamEvent::Progress {
                    completed: done_in_loop,
                    total: remaining_steps,
                });
            }
        }
    }

    for (idx, free_before, watch) in denoise_watch {
        let held = free_before.saturating_sub(watch.finish());
        // A card the denoise never touched reads as noise around zero; saying so
        // would bury the cards it did touch.
        if held >= 64 << 20 {
            info!(
                "Z-Image denoise {width}x{height}: GPU{idx} gave up {:.2} GB while it ran",
                held as f64 / 1e9,
            );
        }
    }

    // The VAE's device was chosen with the VRAM that existed when the model LOADED.
    // By the time we decode, a chat model may have been evicted, another render may have
    // finished, or the transformer may have shrunk - so re-ask before paying for the
    // placement of a minute ago. This is the cheap direction of the move: the weights are
    // small, they are in the page cache, and the decode they unlock is ~50x faster.
    repatriate_zimage_vae(zimg, width, height);

    // 5. VAE decode - dynamic OOM-recovery: try GPU (fast; the VAE attention is
    // query-chunked so 1024^2 fits), and on a CUDA OOM fall back to the CPU VAE.
    // Mirrors the LLM hybrid-OOM-recovery pattern instead of a static size gate.
    let latents = latents.squeeze(2)?; // Remove frame dim
    let image = {
        info!("Z-Image: VAE decode on {:?}...", zimg.vae_device.location());
        let latents_gpu = latents
            .to_device(&zimg.vae_device)?
            .to_dtype(zimg.vae_dtype)?;
        // Decide by free VRAM, not by catching an OOM: a failed allocation bounces
        // that op to the host instead of erroring, so a full card would silently
        // stream the whole decode over PCIe rather than take this cascade.
        let peak = (width * height) as u64 * 4 * (VAE_WIDEST_CH * 9 + VAE_WIDEST_CH * 2);
        let free_here = free_on_device(&zimg.vae_device);
        let fits = free_here >= peak;
        // If the whole-image decode does not fit, the answer is a smaller PIECE, not a
        // different device. The peak is a full-resolution feature map, so it shrinks with
        // the tile's area; decoding is spatially local, so the tiles are independent and
        // the seams only need a few pixels of convolution context. Splitting the decoder
        // across devices instead would ship that same feature map over PCIe at every
        // layer boundary, which is strictly worse than shipping nothing.
        // Tiling is back, bounded by the measurement that was missing when it was
        // switched off: on a real latent a 2x2 split drifts tile-against-tile by 0.0067
        // of a [-1,1] range, under a tenth of what the eye picks up, while a 4x4 split
        // drifts 0.0338 and blocks visibly. choose_vae_tile refuses anything finer than
        // 2x2, so the fast path is the one that measured clean and the CPU still catches
        // everything below it. Without this the decode was landing on the host and
        // costing ~116 s a render.
        let tile = if fits || !zimg.vae_device.is_cuda() {
            None
        } else {
            choose_vae_tile(free_here, peak, width, height)
        };
        // Measured, not fed back - see `report_vae_decode_cost`. Only the WHOLE decode
        // is watched: a tiled run gives up a fraction of the memory and would describe a
        // different question than the one the estimate answers.
        let z_gpu = gpu_index_of(&zimg.vae_device);
        let z_watch = if tile.is_none() && fits {
            z_gpu.and_then(vram_watch_on)
        } else {
            None
        };
        let z_free_before = z_gpu.and_then(crate::inference::place::vram_manager::free_on);
        let attempt = if let Some(t) = tile {
            info!(
                "Z-Image VAE: {:.1} GB decode does not fit {:?}; tiling at {t} latent px                  to stay on the device",
                peak as f64 / 1e9,
                zimg.vae_device.location()
            );
            zimg.vae.decode_tiled(&latents_gpu, t, VAE_TILE_OVERLAP)
        } else if fits || zimg.vae_cpu.is_none() {
            if !fits {
                // No CPU VAE was loaded, so refusing here ends the request - which is
                // what happened: a chat model on one card, the transformer on the
                // other, and the decode returned "no room on the decode device" rather
                // than an image. Attempting it is strictly better, because a failed
                // allocation BOUNCES that op to the host instead of erroring: the
                // decode streams over PCIe and is slow, and slow beats refused.
                info!(
                    "Z-Image VAE: {:.1} GB decode does not fit {:?} and no CPU VAE is \
                     resident; attempting it anyway - allocations that fail will bounce \
                     to the host",
                    peak as f64 / 1e9,
                    zimg.vae_device.location()
                );
            }
            zimg.vae.decode(&latents_gpu)
        } else {
            info!(
                "Z-Image VAE: {:.1} GB decode does not fit {:?}; decoding on the CPU",
                peak as f64 / 1e9,
                zimg.vae_device.location()
            );
            Err(crate::tensor::Error::msg(
                "vae: no room on the decode device",
            ))
        };
        if let (Some(w), Some(before)) = (z_watch, z_free_before) {
            report_vae_decode_cost("Z-Image", width, height, peak, before, w, attempt.is_ok());
        }
        match attempt {
            Ok(img) => img,
            Err(e) if zimg.vae_cpu.is_some() => {
                warn!("Z-Image GPU VAE decode failed ({e}) - retrying on CPU VAE");
                drop(latents_gpu);
                let latents_cpu = latents.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                zimg.vae_cpu.as_ref().unwrap().decode(&latents_cpu)?
            }
            Err(e) => return Err(e.into()),
        }
    };
    drop(latents);
    // Postprocess on the VAE's own device (clamp + cast to U8) first,
    // then move to CPU - U8 is 1/4 the bytes of F32 across PCIe.
    let image = crate::inference::model::zimage::sampling::postprocess_image(&image)?;
    let image = image.to_device(&Device::Cpu)?;
    let image = image.i(0)?; // Remove batch dim

    // Convert (C, H, W) U8 tensor to base64 PNG
    let base64_png = tensor_to_png_base64(&image)?;
    info!(
        "Z-Image generated: {}x{}, {} steps",
        width, height, num_steps
    );
    Ok(base64_png)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod t5_size_tests {
    use super::t5_encoder_bytes;
    use crate::inference::model::t5::encoder::Config;

    /// google/t5-v1_1-xxl - the encoder Flux uses. ~4.76 B params, so 9.5 GB in bf16,
    /// while the checkpoint on disk is 42 GB (f32, encoder AND decoder). Placing off
    /// the file size would reject every card; placing off the config is the truth.
    ///
    /// The config is stated as the checkpoint publishes it rather than transcribed into a
    /// struct, so the same reader the loader uses is what turns it into one - a shape the
    /// loader would reject cannot pass here.
    #[test]
    fn t5_xxl_encoder_is_about_nine_and_a_half_gb_in_bf16() {
        let cfg: Config = serde_json::from_value(serde_json::json!({
            "vocab_size": 32128, "d_model": 4096, "d_kv": 64, "d_ff": 10240,
            "num_layers": 24, "num_decoder_layers": 24, "num_heads": 64,
            "relative_attention_num_buckets": 32, "relative_attention_max_distance": 128,
            "dropout_rate": 0.1, "layer_norm_epsilon": 1e-6, "initializer_factor": 1.0,
            "feed_forward_proj": "gated-gelu", "tie_word_embeddings": false,
            "is_decoder": false, "is_encoder_decoder": true, "use_cache": true,
            "pad_token_id": 0, "eos_token_id": 1, "decoder_start_token_id": 0,
        }))
        .expect("the published t5-v1_1-xxl config");
        let bytes = t5_encoder_bytes(&cfg, 2);
        let gb = bytes as f64 / 1e9;
        assert!(
            (9.0..10.0).contains(&gb),
            "expected ~9.5 GB, got {gb:.2} GB"
        );
    }
}

#[cfg(test)]
mod zimage_primary_overhead_tests {
    use super::{zimage_embedder_bytes, zimage_primary_overhead};

    /// The refiner layers cost REAL memory on the primary card and the budget has to say
    /// so. Counting only the projections is what let a 16.0 GB card be given a 7.1 GB
    /// block budget, filled, and come out with 1.4 GB free - then run out on the first
    /// broadcast of the denoise.
    #[test]
    fn the_refiners_are_counted_not_just_the_projections() {
        // A 15 GB checkpoint, the size the failing run reported.
        const CKPT: u64 = 15_099_494_400;
        let with_refiners = zimage_primary_overhead(CKPT);
        assert!(
            with_refiners > zimage_embedder_bytes(),
            "the refiners must add to the projections"
        );
        // Four refiner layers at a thirtieth of the checkpoint each: about 2 GB, which
        // is the shortfall the failing placement showed.
        let refiners = with_refiners - zimage_embedder_bytes();
        assert!(
            (1.5e9..2.5e9).contains(&(refiners as f64)),
            "refiners accounted at {:.2} GB, expected about 2",
            refiners as f64 / 1e9
        );
    }

    /// It scales with the checkpoint rather than being a number typed in once: a
    /// different Z-Image build has different layers, and a fixed figure would be wrong
    /// for it in whichever direction hurts.
    #[test]
    fn it_follows_the_checkpoint_size() {
        let small = zimage_primary_overhead(6_000_000_000);
        let big = zimage_primary_overhead(15_000_000_000);
        assert!(big > small, "a bigger checkpoint means bigger refiners");
    }

    /// A checkpoint whose size could not be read must not make the overhead vanish -
    /// the projections are still there.
    #[test]
    fn an_unknown_checkpoint_still_charges_the_projections() {
        assert_eq!(zimage_primary_overhead(0), zimage_embedder_bytes());
    }

    /// EVERY byte is charged EXACTLY ONCE. The primary card pays the refiners and the
    /// projections; the planner divides only what the main stack weighs. Handing it the
    /// whole file as well would count those bytes twice - and on cards that never hold
    /// them, which pushes layers off the GPUs for no reason.
    #[test]
    fn the_main_stack_and_the_primary_overhead_are_the_whole_checkpoint() {
        const CKPT: u64 = 15_099_494_400;
        let overhead = zimage_primary_overhead(CKPT);
        let main = CKPT.saturating_sub(overhead);
        assert_eq!(main + overhead, CKPT);
        assert!(
            main > 0,
            "the main stack must not be swallowed by the overhead"
        );
    }
}

/// The arithmetic that decides where a Z-Image render runs, checked WITHOUT a GPU.
///
/// Every failure this module pins was reported as "the picture never arrives": no error
/// reached the client, both cards sat at 0%, one core ran at 100%, and twenty minutes
/// later there was still nothing. None of it needed a render to catch - it is division -
/// but until the placement was a function nothing could call it.
#[cfg(test)]
mod zimage_placement_tests;

mod load_flux;
mod load_others;
mod render;
mod resident;
