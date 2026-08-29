//! Qwen-Image text-to-image: Qwen2.5-VL text encoder -> flow-match Euler DiT -> Wan VAE decode.
//!
//! Placement is decided ENTIRELY by the generic adaptive `HeteroPlan`
//! (fastest-GPU-first, spill to the next GPU, then CPU) - never a
//! hardcoded device or a per-model role split. The hot DiT is planned undivided on the fastest GPU
//! that fits it; the one-shot Qwen2.5-VL encoder is planned on whatever capacity remains after the
//! DiT is resident; the Wan VAE decodes on the DiT's I/O device (with a resident CPU fallback). Both
//! loads run through an OOM-fallback cascade so a load can never hard-OOM. Scales to 1 / 2 / N GPUs
//! + CPU with zero hardware assumptions.

use std::collections::HashMap;

use anyhow::{anyhow, Result as AnyResult};
use tracing::info;

use crate::inference::model::qwen_image::dit::{Config as DitConfig, Model as DitModel};
use crate::inference::model::qwen_image::textenc::Qwen2TextEncoder;
use crate::inference::model::wan::vae::{load_wan_vae_decoder_safetensors, WanVaeDecoder};
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::{Device, Tensor as NT};

/// Fixed system prefix for the base text-to-image template. Its token length is the `drop_idx`
/// passed to the encoder so only the user prompt conditions the DiT.
const SYS_PREFIX: &str = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
const NEG_PROMPT: &str = "blurry, low quality, distorted, ugly, deformed";

/// Relative locations under `huggingface_models_dir`. Only the DiT varies between
/// checkpoints of this family - a drop-in replaces those weights and reuses the
/// encoder, the tokenizer and the VAE unchanged.
const REL_ENC_GGUF: &str = "qwen2.5-vl/qwen2.5-vl-7b-q4km.gguf";
const REL_TOKENIZER: &str = "qwen2.5-vl/tokenizer.json";
const REL_DIT_GGUF: &str = "qwen-image-edit/dit-q4km.gguf";
const REL_VAE: &str = "qwen-image-vae/wan_keyed.safetensors";
const REL_MMPROJ: &str = "qwen2.5-vl/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf";

/// Block dtype the safetensors bridge re-quantizes this DiT's 2-D projections to.
/// 20B parameters at Q8_0 (1 B/weight) is ~20.6 GB resident, which fits no mainstream
/// card whole; Q4_K (~0.56 B/weight) lands ~11.6 GB - and it is the level the family's
/// own GGUF ships at, so a drop-in is quantized like the base it replaces.
const DIT_BLOCK_DTYPE: crate::tensor::quantized::GgmlDType =
    crate::tensor::quantized::GgmlDType::Q4K;

/// True when a checkpoint of this family is STEP-DISTILLED: the teacher's
/// classifier-free guidance is folded into the weights, so the sampler runs a single
/// branch over a handful of steps with the fixed shift the distill's own scheduler
/// declares - rather than the family's resolution-dependent one.
///
/// Detected by name because a bare weights file carries no scheduler: the distillation
/// is a property of how the checkpoint was TRAINED, not of any tensor in it.
pub fn is_step_distilled(model_name: &str) -> bool {
    model_name.to_lowercase().contains("flash")
}

/// The fixed sigma shift a step-distilled checkpoint of this family samples with
/// (`scheduler_config.json`: `use_dynamic_shifting: false`, `shift: 3.0`).
const DISTILLED_SHIFT: f32 = 3.0;

/// How the resident checkpoint wants its sigmas built.
#[derive(Clone, Copy, Debug)]
enum SigmaShift {
    /// The family default: the scheduler derives the shift per request from the token
    /// count, so a 512^2 render and a 1024^2 one do not share a schedule.
    Dynamic,
    /// The checkpoint declares one shift for every resolution.
    Fixed(f32),
}

/// Resident VRAM a DiT checkpoint occupies once loaded, whatever container it ships in.
///
/// A GGUF is already block-quantized, so the file IS the resident model. A safetensors
/// checkpoint is re-quantized at load: the conversion is cached as a sidecar GGUF, and
/// when that exists it is the exact answer. Otherwise the estimate comes from the
/// PARAMETER COUNT rather than a fraction of the file - the same 20B DiT is a 20.7 GB
/// fp8 file or a 40.9 GB bf16 one, and a ratio calibrated on the first over-states the
/// second by 2x, which is the difference between one card holding it and a split.
fn dit_resident_bytes(path: &std::path::Path) -> u64 {
    let file = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    if !path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
    {
        return file;
    }
    if let Some(sidecar) = crate::inference::load::fp8_scaled::sidecar_for(path, DIT_BLOCK_DTYPE) {
        if let Ok(m) = std::fs::metadata(&sidecar) {
            if m.len() > 0 {
                return m.len();
            }
        }
    }
    let elems = crate::inference::load::fp8_scaled::total_elems(path);
    if elems == 0 {
        return file;
    }
    // Bytes per weight of the block dtype, derived from the format itself.
    elems * DIT_BLOCK_DTYPE.type_size() as u64 / DIT_BLOCK_DTYPE.block_size() as u64
}

/// The DiT weights a request resolves to: the caller's local checkpoint when the model
/// name maps to one, else the family's own GGUF. Resolution happens ONE level up (the
/// routing layer already resolves a checkpoint per request); this engine loads what it
/// is handed, so a new drop-in needs no arm here.
fn dit_path(hf_models_dir: &str, local_ckpt: Option<&std::path::Path>) -> std::path::PathBuf {
    local_ckpt
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| std::path::Path::new(hf_models_dir).join(REL_DIT_GGUF))
}

pub struct QwenImageModelState {
    dit: DitModel,
    /// VAE decoder on `vae_device` (the GPU with the most free VRAM - the DiT fills the primary,
    /// so this lands on the secondary GPU, or CPU if no GPU has room).
    vae: WanVaeDecoder,
    vae_device: Device,
    /// CPU copy of the VAE, kept as the OOM / large-image fallback when `vae` is on a GPU (the Wan
    /// decoder is only ~243 MB). `None` when `vae` is already on CPU.
    vae_cpu: Option<WanVaeDecoder>,
    tokenizer: tokenizers::Tokenizer,
    cfg: DitConfig,
    dit_device: Device,
    /// Resident text encoder (on the secondary GPU).
    encoder: Qwen2TextEncoder,
    /// Token length of `SYS_PREFIX`, cached at load.
    drop_idx: usize,
    /// How the RESIDENT checkpoint builds its sigmas (see `SigmaShift`).
    shift: SigmaShift,
    /// Models root, kept for the lazily-loaded edit auxiliaries (VAE encoder +
    /// vision tower - only instruction edits pay for them).
    hf_dir: String,
}

/// Size of the family's HOT component checkpoint (the DiT) - the figure the pressure protocol
/// needs BEFORE the engine load runs. 0 when the file is absent.
pub fn hot_component_bytes(hf_models_dir: &str, local_ckpt: Option<&std::path::Path>) -> u64 {
    dit_resident_bytes(&dit_path(hf_models_dir, local_ckpt))
}
/// VRAM this family needs FREE on the resident model's card to run one generation
/// (the runtime scratch, not the weights). Used by the pressure protocol before a
/// generation against an ALREADY-RESIDENT model, where no load-time check runs.
pub fn runtime_headroom_bytes(
    hf_models_dir: &str,
    local_ckpt: Option<&std::path::Path>,
    width: usize,
    height: usize,
) -> u64 {
    runtime_reserve(&dit_path(hf_models_dir, local_ckpt), width, height)
}

/// Deterministic gaussian noise (Box-Muller on a seeded xorshift - native has no randn).
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    (0..n)
        .map(|_| {
            let (a, b) = (u().max(1e-9), u());
            (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
        })
        .collect()
}

/// The sigma schedule a checkpoint samples with: diffusers' FlowMatchEuler base
/// `linspace(1, 1/steps, steps) + [0]`, time-shifted. The SHIFT is where the family and
/// a distilled drop-in part company - the family's scheduler derives it from the token
/// count of the request, a distill declares one value for every resolution - so it is
/// the only thing this takes from the resident checkpoint.
fn sigmas(shift: SigmaShift, steps: usize, img_tokens: usize) -> Vec<f32> {
    let s = match shift {
        SigmaShift::Fixed(s) => s,
        SigmaShift::Dynamic => {
            let mu = (0.5f32 + (1.15 - 0.5) / (4096.0 - 256.0) * (img_tokens as f32 - 256.0))
                .clamp(0.5, 1.15);
            mu.exp()
        }
    };
    let mut sig: Vec<f32> = (0..steps)
        .map(|j| 1.0 - (j as f32) * (1.0 - 1.0 / steps as f32) / (steps as f32 - 1.0))
        .map(|t| s * t / (s * t + 1.0 - t))
        .collect();
    sig.push(0.0);
    sig
}

/// Whether a guidance scale actually asks for a second, unconditional DiT branch.
///
/// The guided step is `v_neg + scale * (v_pos - v_neg)`, which at scale 1 is exactly
/// `v_pos`: the unconditional branch cancels out algebraically. A step-distilled
/// checkpoint samples there by construction (its teacher's guidance is in the weights),
/// so computing that branch would double the cost of every step for a term that is then
/// discarded - and encode a negative prompt the model was never meant to see.
fn needs_guidance_branch(scale: f32) -> bool {
    (scale - 1.0).abs() > f32::EPSILON
}

fn dev_label(d: &Device) -> String {
    match d.location() {
        crate::tensor::DeviceLocation::Cuda { gpu_id } => format!("CUDA:{gpu_id}"),
        _ => "CPU".to_string(),
    }
}

impl QwenImageModelState {
    /// True when the DiT blocks span several devices (see `Model::is_split`).
    pub fn dit_is_split(&self) -> bool {
        self.dit.is_split()
    }
}

/// VAE decode tiling, in LATENT pixels (the Wan decoder upscales 8x): 64 -> 512 px
/// tiles with a 64 px feather. Big enough that the per-tile overhead stays small,
/// small enough that the decode's transients fit next to a resident DiT.
const VAE_TILE_LATENT: usize = 64;
const VAE_TILE_OVERLAP: usize = 8;
/// Peak VRAM of ONE tile's decode. The structural term is
/// `tile_pixels x widest_channel_count x (im2col 3x3 + the live buffers) x f32`;
/// the measured peak of a whole-image decode ran ~1.5x over the same formula
/// (allocator rounding, cuDNN workspaces), so the estimate carries a 2x margin -
/// under-estimating here is what makes a "fits" verdict OOM mid-decode.
const VAE_TILE_TRANSIENT_BYTES: u64 =
    (VAE_TILE_LATENT as u64 * 8) * (VAE_TILE_LATENT as u64 * 8) * 128 * 8 * 4 * 2;

/// Native generation side of the family (Qwen-Image is a 1K model); the runtime reserve is sized
/// for a generation at this resolution.
/// The VAE's spatial reduction and the DiT's patch size on top of it. Together they
/// turn requested PIXELS into the token count the demand below is a function of.
const VAE_STRIDE: usize = 8;
const PATCH: usize = 2;
/// Caption token budget the reserve accounts for.
const MAX_TEXT_TOKENS: usize = 1024;

/// Per-device VRAM reserved for ONE generation's runtime scratch, DERIVED from the architecture
/// and the checkpoint instead of a fixed constant (same structural terms as the Boogu engine):
/// concurrently-live `[seq, dim]` F32 activation buffers of the dual-stream forward, the tiled
/// attention scores, the largest GGUF weight's F32 dequant + BF16 GEMM scratch (from the header,
/// no tensor data read), and the VAE decode's conv buffers on the shared device.
fn runtime_reserve(dit_path: &std::path::Path, width: usize, height: usize) -> u64 {
    let cfg = crate::inference::model::qwen_image::dit::Config::default();
    let dim = cfg.dim() as u64;
    let f32b = 4u64;
    // The resident checkpoint of this engine serves EDITS as well as txt2img, and
    // an edit runs the DiT over [noise tokens ; clean image latent tokens] - TWICE
    // the image tokens of a plain generation. Sizing for one frame is what left the
    // card with ~3 GB and OOM'd every 1024^2 edit mid-denoise.
    const EDIT_FRAMES: u64 = 2;
    // From the REQUESTED geometry, not the resolution the family was trained at.
    // Sizing every request as if it were 1024^2 is wrong in both directions: it
    // over-reserves at 512^2, refusing placements that would have run, and
    // under-reserves above it, admitting a load that then cannot denoise.
    let img_tokens =
        crate::inference::place::runtime_demand::latent_tokens(height, width, VAE_STRIDE, PATCH)
            as u64
            * EDIT_FRAMES;
    let seq = img_tokens + MAX_TEXT_TOKENS as u64;
    let buf = |tokens: u64, width: u64| tokens * width * f32b;

    // Per-block peak of the dual-stream MMDiT, term by term:
    //  - attention: q, k, v and the attention output live together for both streams;
    //  - the modulated stream input and the block residual are live across the block;
    //  - the MLP is the widest tensor in the graph: gate, up and the activation
    //    product at 4x dim (the term that was missing - 3 x 453 MB at 1024^2 edit).
    let attn = 4 * buf(seq, dim);
    let resid = 2 * buf(seq, dim);
    // MLP is computed in token chunks (see `mlp_token_chunked` in the DiT): the 4x-wide
    // hidden buffer is bounded by the chunk, not by the sequence.
    const MLP_CHUNK: u64 = 2048;
    let mlp = 3 * buf(MLP_CHUNK.min(seq), 4 * dim);
    // Tiled attention scores: [heads, QUERY_TILE, seq] F32, doubled by the softmax.
    // Must match QUERY_TILE in the DiT's block attention - under-accounting it is
    // what made a "fits" verdict starve at generation time.
    let attn_tile = 512u64;
    let scores = 2 * attn_tile * seq * cfg.num_attention_heads as u64 * f32b;
    // Sampler state carried ACROSS steps: the latent, the CFG pair and the step delta.
    let sampler = 4 * buf(img_tokens, cfg.in_channels as u64 * 4);
    // Largest 2-D weight's dequant + GEMM scratch, read from the checkpoint's HEADER
    // (no tensor data). Both containers are handled: reading only the GGUF header left
    // every safetensors checkpoint of this family with a zero here, i.e. a reserve
    // missing its single largest term.
    let largest_2d = crate::tensor::quantized::gguf_file::open_header(dit_path)
        .ok()
        .map(|c| {
            c.tensor_infos
                .values()
                .filter(|t| t.shape.dims().len() == 2)
                .map(|t| t.elem_count() as u64)
                .max()
                .unwrap_or(0)
        })
        .unwrap_or_else(|| crate::inference::load::fp8_scaled::largest_2d_elems(dit_path));
    let dequant = largest_2d * 6;
    // VAE decode conv buffers on the shared device.
    let vae_tile_px = (64 * 8) as u64;
    let vae = vae_tile_px * vae_tile_px * 128 * 6 * f32b;
    // The denoise and the VAE decode are SEQUENTIAL phases: the DiT activations are
    // freed before the decoder allocates, so the reserve is the max of the two peaks,
    // not their sum. Summing them cost ~0.8 GB of headroom and was enough to refuse a
    // single-card placement that fits - forcing a slower cross-GPU DiT split.
    let dit_peak = attn + resid + mlp + scores + sampler + dequant;
    let peak = dit_peak.max(vae);
    // Allocator reality: the pool serves these buffers from suballocated blocks and
    // keeps freed blocks cached, so its high-water mark tracks the SUM of what the
    // denoise churns, not the instantaneous live set. Measured on this engine: a
    // 512^2 edit (3072 tokens) peaks ~2.7 GB above the resident weights, i.e. about
    // 0.9 MB per token - roughly 3x the analytic live-set peak above. Scale the
    // analytic terms by that measured factor so placement plans against what the
    // allocator actually does; under-reserving is what OOM'd mid-denoise, and
    // over-reserving only shifts a model to the next card or a split.
    let pool_factor = 3;
    let reserve = peak * pool_factor;
    tracing::info!(
        "Qwen-Image reserve terms: analytic peak {:.2} GB (dit {:.2}, vae {:.2}) x{pool_factor} \
         = {:.2} GB",
        peak as f64 / 1e9,
        dit_peak as f64 / 1e9,
        vae as f64 / 1e9,
        reserve as f64 / 1e9,
    );
    reserve
}

/// Plan a model UNDIVIDED on the fastest GPU that fits it whole (no cross-GPU sync - matters for the
/// hot per-step DiT); if no single GPU fits, let HeteroPlan spill it across GPUs + CPU. `cuda` is
/// fastest-first (by throughput). Placement is thus adaptive and role-aware, never hardcoded.
pub(crate) fn plan_undivided_or_spill(
    n_layers: usize,
    size: u64,
    cuda: &[(usize, u64)],
) -> HeteroPlan {
    for &(idx, free) in cuda {
        if free >= size {
            return HeteroPlan::forced_gpu(n_layers, n_layers, idx);
        }
    }
    HeteroPlan::calculate(n_layers, size, cuda, &[], 1.0)
}

/// Load a model with an OOM-fallback CASCADE so a load can NEVER hard-OOM: ideal placement
/// (undivided on the fastest GPU that fits, or a natural spill) -> on OOM, trim the CUDA pool and
/// retry an even GPU split -> on OOM, CPU. Mirrors the FLUX single-device -> split -> CPU cascade.
pub(crate) fn load_with_fallback<T>(
    n_layers: usize,
    size: u64,
    cuda: &[(usize, u64)],
    what: &str,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
    load: impl Fn(&HeteroPlan) -> AnyResult<T>,
) -> AnyResult<T> {
    // A cancelled load is NOT a placement failure: short-circuit the cascade instead of
    // retrying cheaper plans that would each bail again.
    let cancelled =
        || cancel.is_some_and(crate::inference::serve::cancel::CancelToken::is_cancelled);
    match load(&plan_undivided_or_spill(n_layers, size, cuda)) {
        Ok(m) => return Ok(m),
        Err(e) => {
            if cancelled() {
                return Err(e);
            }
            tracing::warn!("{what}: ideal placement failed ({e:#}); retrying split across GPUs");
        }
    }
    #[cfg(feature = "cuda")]
    crate::inference::engine::llm_engine::trim_cuda_pools();
    if !cuda.is_empty() {
        match load(&HeteroPlan::split_across_cuda(n_layers, cuda)) {
            Ok(m) => return Ok(m),
            Err(e) => {
                if cancelled() {
                    return Err(e);
                }
                tracing::warn!("{what}: GPU split failed ({e:#}); falling back to CPU");
            }
        }
        #[cfg(feature = "cuda")]
        crate::inference::engine::llm_engine::trim_cuda_pools();
    }
    // Empty CUDA list -> the plan puts every layer on CPU (never OOMs).
    load(&HeteroPlan::calculate(n_layers, size, &[], &[], 1.0))
}

/// Load Qwen-Image for generation. Placement is decided by the adaptive `HeteroPlan`
/// (fastest-GPU-first, spill to the next GPU then CPU) with an OOM-fallback cascade - never a
/// hardcoded device. `_primary` is ignored (kept for the loader signature); the plan chooses.
/// Errors only if paths are missing.
pub fn load(
    hf_models_dir: &str,
    model_name: &str,
    local_ckpt: Option<&std::path::Path>,
    _primary: &Device,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> AnyResult<QwenImageModelState> {
    let base = std::path::Path::new(hf_models_dir);
    let enc_gguf = base.join(REL_ENC_GGUF);
    let tok_path = base.join(REL_TOKENIZER);
    // The DiT is whatever checkpoint this request resolved to - a drop-in of the family
    // (same architecture, same walk; the loader picks its builder by extension) or the
    // family's own GGUF. Encoder, tokenizer and VAE are shared by all of them.
    let dit_path = dit_path(hf_models_dir, local_ckpt);
    let vae_path = base.join(REL_VAE);
    for (p, what) in [
        (&enc_gguf, "text encoder GGUF"),
        (&tok_path, "tokenizer"),
        (&dit_path, "DiT GGUF"),
        (&vae_path, "VAE"),
    ] {
        if !p.exists() {
            return Err(anyhow!("Qwen-Image {what} not found at {}", p.display()));
        }
    }

    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow!("Qwen-Image tokenizer: {e}"))?;
    let drop_idx = tokenizer
        .encode(SYS_PREFIX, true)
        .map_err(|e| anyhow!("Qwen-Image tokenize prefix: {e}"))?
        .get_ids()
        .len();

    // Probe once for the device handles (shared by BOTH models so their cross-device transfers see
    // the same physical devices). Plan the hot DiT first (undivided on the fastest GPU that fits),
    // then re-probe FREE (the DiT is now resident) and plan the one-shot encoder on what remains.
    let mut cuda_devices: HashMap<usize, Device> = HashMap::new();
    let gpu_reserve = runtime_reserve(&dit_path, geom.width, geom.height);
    tracing::info!(
        "Qwen-Image runtime reserve for {}x{} (arch-derived): {:.2} GB/GPU",
        geom.width,
        geom.height,
        gpu_reserve as f64 / 1e9
    );
    let dit_cuda: Vec<(usize, u64)> = crate::inference::place::vram_manager::probe(gpu_reserve)
        .into_iter()
        .map(|(i, f, d)| {
            cuda_devices.insert(i, d);
            (i, f)
        })
        .collect();

    // Hot DiT (60 transformer blocks), run per sampling step: undivided on the fastest GPU that
    // fits; OOM-fallback cascade -> split -> CPU (never OOM).
    // Plan on the RESIDENT footprint, not the file size: a safetensors checkpoint
    // re-quantizes its 2-D projections at load, so planning with the raw file refuses
    // single-card fits that the resident model makes comfortably - splitting the DiT and
    // starving the generation of headroom.
    let dit_sz = dit_resident_bytes(&dit_path);
    let dit_path_str = dit_path.to_string_lossy().to_string();
    tracing::info!(
        "Qwen-Image DiT planning: needs {:.2} GB resident; per-GPU free AFTER the {:.2} GB \
         reserve = {:?}",
        dit_sz as f64 / 1e9,
        gpu_reserve as f64 / 1e9,
        dit_cuda
            .iter()
            .map(|(i, f)| format!("GPU{i}:{:.2}GB", *f as f64 / 1e9))
            .collect::<Vec<_>>()
    );
    let dit = load_with_fallback(60, dit_sz, &dit_cuda, "Qwen-Image DiT", cancel, |plan| {
        DitModel::load_hetero(&dit_path_str, &cuda_devices, plan, cancel)
            .map_err(|e| anyhow!("{e}"))
    })?;
    let dit_device = dit.input_device().clone();

    // Re-probe FREE VRAM (the DiT is now resident); reuse the SAME device handles so the encoder's
    // output and the DiT input reference the same physical devices.
    let enc_cuda: Vec<(usize, u64)> = crate::inference::place::vram_manager::probe(gpu_reserve)
        .into_iter()
        .filter(|(i, _, _)| cuda_devices.contains_key(i))
        .map(|(i, f, _)| (i, f))
        .collect();
    // One-shot encoder (28 Qwen2 layers): best-fit remaining capacity; same OOM-fallback cascade.
    let enc_sz = std::fs::metadata(&enc_gguf).map(|m| m.len()).unwrap_or(0);
    let enc_path_str = enc_gguf.to_string_lossy().to_string();
    let encoder = load_with_fallback(
        28,
        enc_sz,
        &enc_cuda,
        "Qwen-Image encoder",
        cancel,
        |plan| {
            Qwen2TextEncoder::from_gguf_hetero(&enc_path_str, &cuda_devices, plan)
                .map_err(|e| anyhow!("{e}"))
        },
    )?;

    info!(
        "Qwen-Image placement: DiT I/O on {}, encoder input on {}",
        dev_label(&dit_device),
        dev_label(encoder.input_device()),
    );

    // VAE loads on CPU (the safetensors loader's home). Its decode copy sits on the DiT's I/O device
    // (where the DiT output lands - no CPU round-trip). Keep the CPU copy as the never-OOM fallback
    // (generate retries on CPU if a very large image's GPU decode OOMs).
    let cpu_vae = load_wan_vae_decoder_safetensors(vae_path.to_string_lossy().as_ref())
        .map_err(|e| anyhow!("Qwen-Image VAE load: {e}"))?;
    let vae_device = dit_device.clone();
    info!(
        "Qwen-Image VAE decodes on {} (CPU fallback on OOM)",
        dev_label(&vae_device)
    );
    let (vae, vae_cpu) = if vae_device.is_cuda() {
        let gpu = cpu_vae
            .to_device(&vae_device)
            .map_err(|e| anyhow!("Qwen-Image VAE -> {}: {e}", dev_label(&vae_device)))?;
        (gpu, Some(cpu_vae))
    } else {
        (cpu_vae, None)
    };

    // The sampler profile belongs to the RESIDENT checkpoint, so it is decided once here
    // rather than re-derived from a name on every generation.
    let shift = if is_step_distilled(model_name) {
        info!("Qwen-Image: step-distilled checkpoint - fixed shift {DISTILLED_SHIFT}, single-branch sampler");
        SigmaShift::Fixed(DISTILLED_SHIFT)
    } else {
        SigmaShift::Dynamic
    };

    Ok(QwenImageModelState {
        dit,
        vae,
        vae_device,
        vae_cpu,
        tokenizer,
        cfg: DitConfig::default(),
        dit_device,
        encoder,
        drop_idx,
        shift,
        hf_dir: hf_models_dir.to_string(),
    })
}

/// Instruction edit (Qwen-Image-Edit) - the checkpoint's ACTUAL purpose. Port
/// of the validated `qwen_image_edit_render` bin: the input image conditions
/// BOTH streams - its Qwen2.5-VL vision embeds are spliced into the edit
/// prompt, and its clean VAE latent tokens are CONCATENATED after the noise
/// tokens (one (2, gh, gw) RoPE grid; only the noise half is denoised). This
/// is NOT img2img: from-noise + concat is how the edit model was trained.
#[allow(clippy::too_many_arguments)]
pub fn generate_edit(
    state: &QwenImageModelState,
    prompt: &str,
    input_image_b64: &str,
    width: usize,
    height: usize,
    num_steps: usize,
    guidance: f32,
    seed: u64,
    cancel: &crate::inference::serve::cancel::CancelToken,
    // What the guidance steers AWAY from. Blank means an unconditioned branch, which
    // is what an edit is tuned against, so "" stays the default here.
    negative: Option<&str>,
) -> AnyResult<String> {
    use base64::Engine as _;
    let dev = &state.dit_device;
    let steps = num_steps.max(2);
    let cfg_scale = if guidance > 0.0 { guidance } else { 4.0 };
    let (lh, lw) = (height / 8, width / 8);
    let (gh, gw) = (lh / 2, lw / 2);
    let si = gh * gw;
    if si == 0 {
        return Err(anyhow!(
            "Qwen-Image-Edit: image too small ({width}x{height})"
        ));
    }
    let c = state.cfg.out_channels;

    let raw = base64::engine::general_purpose::STANDARD
        .decode(input_image_b64)
        .map_err(|e| anyhow!("Qwen-Image-Edit: bad input image base64: {e}"))?;
    let src = crate::inference::media::image_processor::decode_image_oriented(&raw)
        .map_err(|e| anyhow!("Qwen-Image-Edit: undecodable input image: {e}"))?;

    // 1) VAE-encode the (resized) input -> self-normalized latent -> image tokens.
    let rs = src
        .resize_exact(
            width as u32,
            height as u32,
            image::imageops::FilterType::Lanczos3,
        )
        .to_rgb8();
    let mut iv = vec![0f32; 3 * width * height];
    for y in 0..height {
        for x in 0..width {
            let pxl = rs.get_pixel(x as u32, y as u32);
            for ch in 0..3 {
                iv[ch * width * height + y * width + x] = pxl[ch] as f32 / 127.5 - 1.0;
            }
        }
    }
    cancel.bail()?;
    let vae_path = std::path::Path::new(&state.hf_dir)
        .join(REL_VAE)
        .to_str()
        .ok_or_else(|| anyhow!("bad VAE path"))?
        .to_string();
    // Encoder demand DERIVED from the request: weights plus the widest conv
    // activation the encoder holds live at this resolution (192 channels, F32,
    // several tensors in flight) plus slack. A flat guess admitted a 1024^2
    // encode onto a card with 2 GB free and the upload OOM'd.
    // Weights from the file that is about to be loaded, activations from the request.
    // Flat byte counts stood in for both and could only be right at one resolution
    // with one checkpoint.
    let enc_need: u64 = crate::inference::place::runtime_demand::vae_encode_bytes(height, width)
        + std::fs::metadata(&vae_path).map(|m| m.len()).unwrap_or(0);
    let ilat = crate::inference::place::vram_manager::run_staged(
        "qwen-image-edit VAE encode",
        enc_need,
        |edev| {
            let venc =
                crate::inference::model::wan::vae::load_wan_vae_encoder_safetensors(&vae_path)?
                    .to_device(edev)?;
            let x =
                NT::from_vec_f32(iv.clone(), (3usize, 1usize, height, width))?.to_device(edev)?;
            venc.encode(&x)
        },
    )
    .map_err(|e| anyhow!("Qwen-Image-Edit VAE encode: {e}"))?;

    let mut ilv = ilat.to_device(&Device::Cpu)?.to_vec_f32();
    let im_m = ilv.iter().sum::<f32>() / ilv.len() as f32;
    let im_s = (ilv.iter().map(|&x| (x - im_m).powi(2)).sum::<f32>() / ilv.len() as f32)
        .sqrt()
        .max(1e-6);
    for x in ilv.iter_mut() {
        *x = (*x - im_m) / im_s;
    }
    // patchify [c, lh, lw] -> [si, c*4]
    let cpl = lh * lw;
    let mut img_tokens = vec![0f32; si * c * 4];
    for i in 0..gh {
        for j in 0..gw {
            let tb = (i * gw + j) * (c * 4);
            for ch in 0..c {
                for pi in 0..2 {
                    for pj in 0..2 {
                        img_tokens[tb + ch * 4 + pi * 2 + pj] =
                            ilv[ch * cpl + (i * 2 + pi) * lw + (j * 2 + pj)];
                    }
                }
            }
        }
    }

    // 2) Image-aware text conditioning: vision tower embeds spliced at
    //    <|image_pad|> by encode_edit. The tower loads per edit and is dropped
    //    after (1.4 GB; edits are occasional).
    cancel.bail()?;
    // Tower weights (~1.4 GB F16) + patch activations at the source resolution.
    // The whole load+forward runs through the staged placement cascade, so a
    // card that turns out too tight mid-forward degrades to the next device
    // instead of failing the request.
    let mmproj = std::path::Path::new(&state.hf_dir)
        .join(REL_MMPROJ)
        .to_str()
        .ok_or_else(|| anyhow!("bad mmproj path"))?
        .to_string();
    let src_rgb = src.to_rgb8();
    // Same contract: the tower's own weights, plus what the source image costs it.
    let vis_demand: u64 = std::fs::metadata(&mmproj).map(|m| m.len()).unwrap_or(0)
        + crate::inference::place::runtime_demand::vision_tower_bytes(
            src_rgb.height() as usize,
            src_rgb.width() as usize,
        );
    let (vision, vgrid) = crate::inference::place::vram_manager::run_staged(
        "qwen-image-edit vision tower",
        vis_demand,
        |vdev| {
            let vtower =
                crate::inference::model::qwen25::vision::Qwen25Vision::load_mmproj(&mmproj, vdev)
                    .map_err(|e| crate::tensor::Error::msg(format!("vision tower: {e:?}")))?;
            let (pv, vgrid) =
                crate::inference::model::qwen25::vision::preprocess(&src_rgb, vdev)
                    .map_err(|e| crate::tensor::Error::msg(format!("vision preprocess: {e:?}")))?;
            let vision = vtower
                .forward(&pv, vgrid)
                .map_err(|e| crate::tensor::Error::msg(format!("vision forward: {e:?}")))?;
            Ok((vision, vgrid))
        },
    )
    .map_err(|e| anyhow!("Qwen-Image-Edit vision: {e}"))?;
    let guided = needs_guidance_branch(cfg_scale);
    let (txt, neg) = {
        // The tower may have run on a different device than the text encoder
        // (staged placement); `encode_edit` concatenates the vision embeds with
        // its own token embeds, which requires ONE device.
        let vision = vision.to_device(state.encoder.input_device())?;
        let (vlh, vlw) = (vgrid.1 / 2, vgrid.2 / 2);
        let txt = state
            .encoder
            .encode_edit(&state.tokenizer, &vision, prompt, vlh, vlw)
            .map_err(|e| anyhow!("Qwen-Image-Edit encode+: {e}"))?
            .to_device(dev)?;
        let neg = if guided {
            Some(
                state
                    .encoder
                    .encode_edit(
                        &state.tokenizer,
                        &vision,
                        negative
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .unwrap_or(""),
                        vlh,
                        vlw,
                    )
                    .map_err(|e| anyhow!("Qwen-Image-Edit encode-: {e}"))?
                    .to_device(dev)?,
            )
        } else {
            None
        };
        (txt, neg)
    };

    // 3) Denoise the noise half of [noise ; clean image] (frames = 2).
    let sig = sigmas(state.shift, steps, si);
    let mut x: Vec<f32> = noise(si * state.cfg.in_channels, seed);
    // An EDIT starts from a real image, where an approximated step is far more
    // visible than in a from-noise render: every step is computed.
    for i in 0..steps {
        cancel.bail()?;
        let mut cat = x.clone();
        cat.extend_from_slice(&img_tokens);
        let xin = NT::from_vec_f32(cat, (2 * si, state.cfg.in_channels))?.to_device(dev)?;
        let t = sig[i] * 1000.0;
        let vp = state
            .dit
            .forward(&xin, &txt, t, 2, gh, gw)?
            .to_device(&Device::Cpu)?
            .to_vec_f32();
        let dt = sig[i + 1] - sig[i];
        match &neg {
            Some(neg) => {
                let vn = state
                    .dit
                    .forward(&xin, neg, t, 2, gh, gw)?
                    .to_device(&Device::Cpu)?
                    .to_vec_f32();
                for k in 0..x.len() {
                    x[k] += dt * (vn[k] + cfg_scale * (vp[k] - vn[k]));
                }
            }
            None => {
                for k in 0..x.len() {
                    x[k] += dt * vp[k];
                }
            }
        }
    }

    tokens_to_png(state, x, lh, lw, true)
}

/// Generate one image. `width`/`height` are pixels (multiples of 16 recommended), `num_steps` the
/// flow-match Euler steps, `guidance` the CFG scale, `seed` the noise seed.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    state: &QwenImageModelState,
    prompt: &str,
    width: usize,
    height: usize,
    num_steps: usize,
    guidance: f32,
    seed: u64,
    cancel: &crate::inference::serve::cancel::CancelToken,
    step_reuse: f32,
    // What the guidance steers AWAY from. `None`/blank keeps the family default.
    negative: Option<&str>,
) -> AnyResult<String> {
    let dev = &state.dit_device;
    let steps = num_steps.max(2); // sigma schedule divides by (steps-1)
    let cfg_scale = if guidance > 0.0 { guidance } else { 4.0 };
    let (lh, lw) = (height / 8, width / 8); // VAE downsample 8
    let (gh, gw) = (lh / 2, lw / 2); // DiT patch 2
    let si = gh * gw;
    if si == 0 {
        return Err(anyhow!("Qwen-Image: image too small ({width}x{height})"));
    }

    // 1) Encode positive + negative prompts on the encoder GPU, then move to the DiT GPU.
    // The caller's negative prompt is what the guidance pushes away from. It used to be
    // ignored here: this family conditions on two prompts and the second was pinned to
    // NEG_PROMPT, so a request naming what to avoid got the built-in text instead and no
    // word about it. Blank falls back to the family's own default rather than to "",
    // which would drop guidance to an unconditioned branch this model is not tuned for.
    let guided = needs_guidance_branch(cfg_scale);
    let pos_tpl = format!("{SYS_PREFIX}{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let txt = state
        .encoder
        .encode_text(&state.tokenizer, &pos_tpl, state.drop_idx)
        .map_err(|e| anyhow!("Qwen-Image encode+: {e}"))?
        .to_device(dev)?;
    let neg = if guided {
        let neg_text = negative
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(NEG_PROMPT);
        let neg_tpl = format!("{SYS_PREFIX}{neg_text}<|im_end|>\n<|im_start|>assistant\n");
        Some(
            state
                .encoder
                .encode_text(&state.tokenizer, &neg_tpl, state.drop_idx)
                .map_err(|e| anyhow!("Qwen-Image encode-: {e}"))?
                .to_device(dev)?,
        )
    } else {
        None
    };

    // 2) Flow-match Euler in patchified-token space, with the guidance branch only when
    //    the scale asks for one.
    let mut x = NT::from_vec_f32(
        noise(si * state.cfg.in_channels, seed),
        (si, state.cfg.in_channels),
    )?
    .to_device(dev)?;
    let sig = sigmas(state.shift, steps, si);
    // One reuse cache per CFG branch: the conditional and unconditional predictions
    // evolve at different rates, so sharing one would reuse a branch on the other's
    // evidence. This engine runs TWO forwards per step, so a skipped step is worth
    // twice what it is on a single-branch sampler.
    use crate::inference::model::flux::sampling::StepReuse;
    let (mut reuse_p, mut reuse_n) = (StepReuse::new(step_reuse), StepReuse::new(step_reuse));
    for i in 0..steps {
        cancel.bail()?;
        let last = i + 1 == steps;
        let t = sig[i] * 1000.0;
        let vp = match reuse_p.reuse(&x, last)? {
            Some(cached) => cached,
            None => {
                let v = state.dit.forward(&x, &txt, t, 1, gh, gw)?;
                reuse_p.observe(&x, &v)?;
                v
            }
        };
        // v = v_neg + cfg*(v_pos - v_neg); at cfg 1 that IS v_pos, so the branch is not run.
        let v = match &neg {
            Some(neg) => {
                let vn = match reuse_n.reuse(&x, last)? {
                    Some(cached) => cached,
                    None => {
                        let v = state.dit.forward(&x, neg, t, 1, gh, gw)?;
                        reuse_n.observe(&x, &v)?;
                        v
                    }
                };
                vn.add(&vp.add(&vn.affine(-1.0, 0.0)?)?.affine(cfg_scale, 0.0)?)?
            }
            None => vp,
        };
        let dt = sig[i + 1] - sig[i];
        x = x.add(&v.affine(dt, 0.0)?)?; // Euler: x += dt*v
    }
    if reuse_p.skipped + reuse_n.skipped > 0 {
        info!(
            "qwen-image: reused {} of {} DiT forwards (threshold {step_reuse})",
            reuse_p.skipped + reuse_n.skipped,
            if guided { 2 * steps } else { steps }
        );
    }

    let toks = x.to_device(&Device::Cpu)?.to_vec_f32();
    tokens_to_png(state, toks, lh, lw, false)
}

/// Unpatchify `[si, 64]` DiT tokens into the latent, decode through the VAE
/// (device -> other-GPU -> CPU fallback chain) and return a base64 PNG.
/// `self_norm` re-normalizes the latent by its own mean/std first - the
/// edit-concat path's convention (mirrors the validated edit bin).
fn tokens_to_png(
    state: &QwenImageModelState,
    toks: Vec<f32>,
    lh: usize,
    lw: usize,
    self_norm: bool,
) -> AnyResult<String> {
    let gh = lh / 2;
    let gw = lw / 2;
    // Unpatchify [si, 64] -> latent [c, 1, lh, lw]. diffusers pack: token[c*4 + pi*2 + pj].
    let c = state.cfg.out_channels;
    let mut lv = vec![0f32; c * lh * lw];
    for i in 0..gh {
        for j in 0..gw {
            let tb = (i * gw + j) * (c * 4);
            for ch in 0..c {
                for pi in 0..2 {
                    for pj in 0..2 {
                        lv[ch * lh * lw + (i * 2 + pi) * lw + (j * 2 + pj)] =
                            toks[tb + ch * 4 + pi * 2 + pj];
                    }
                }
            }
        }
    }
    if self_norm {
        let lm = lv.iter().sum::<f32>() / lv.len() as f32;
        let ls = (lv.iter().map(|&x| (x - lm).powi(2)).sum::<f32>() / lv.len() as f32)
            .sqrt()
            .max(1e-6);
        for v in lv.iter_mut() {
            *v = (*v - lm) / ls;
        }
    }
    // Wan VAE is a VIDEO VAE: decode expects [C, T, H, W]; an image is T=1. Decode on the VAE's
    // device; if the GPU decode OOMs (a large image's transient buffers don't fit next to the
    // resident DiT), fall back to the CPU copy. Then bring the pixels to CPU for PNG encoding.
    let lat = NT::from_vec_f32(lv, (c, 1usize, lh, lw))?;
    // DECIDE the decode device before running it. The tensor layer bounces an
    // individual op to the host when a device allocation fails, so a card that is
    // too tight does NOT return an error - it ping-pongs every convolution across
    // PCIe and finishes minutes later, and the cascade below (which only fires on
    // Err) never gets a chance. Measured on a 1024^2 decode next to a resident
    // split DiT: 297 s of bouncing where a deliberate CPU decode is a fraction of
    // that. So: fastest card that fits the transients, else the CPU, chosen here.
    // Decoding in overlapping tiles bounds the peak by ONE tile instead of the whole
    // image, so the demand no longer grows with the requested resolution.
    let transient = VAE_TILE_TRANSIENT_BYTES;
    let planned =
        crate::inference::place::vram_manager::pick_device_for("qwen-image vae decode", transient)
            .filter(|(_, free, _)| *free >= transient)
            .map(|(_, _, d)| d)
            .unwrap_or(Device::Cpu);
    // The resident decoder is pinned to `vae_device`; the CPU copy is the movable
    // one, so any other target is served by moving that copy.
    let decode_on = |dev: &Device, whole: bool| -> crate::tensor::Result<NT> {
        let tiled = |v: &WanVaeDecoder, d: &Device| -> crate::tensor::Result<NT> {
            let l = lat.to_device(d)?;
            if whole {
                v.decode_whole(&l)
            } else {
                v.decode_spatial_tiled(&l, VAE_TILE_LATENT, VAE_TILE_OVERLAP)
            }
        };
        if Device::same_device(dev, &state.vae_device) {
            tiled(&state.vae, dev)
        } else if let Some(cpu_vae) = &state.vae_cpu {
            if dev.is_cpu() {
                tiled(cpu_vae, dev)
            } else {
                tiled(&cpu_vae.to_device(dev)?, dev)
            }
        } else {
            tiled(&state.vae, &state.vae_device)
        }
    };
    if !Device::same_device(&planned, &state.vae_device) {
        info!(
            "Qwen-Image VAE decode: {} has no room for the {:.1} GB transients; decoding on {}",
            dev_label(&state.vae_device),
            transient as f64 / 1e9,
            dev_label(&planned)
        );
    }
    // PREFER THE WHOLE-IMAGE DECODE when a card can hold it. This family tiled
    // unconditionally, on every path including the CPU one, and tiling is not free:
    // the decoder's mid-block attends over the WHOLE map, so a per-tile window lets
    // each tile settle on its own colour and brightness. Measured on the Z-Image VAE
    // with the same construction, a real latent tiled at 16 drifted tile-against-tile
    // by 0.034 of a [-1,1] range - about 4 levels of 255 BETWEEN neighbouring blocks,
    // which is what "burnt colours" looks like on flat areas. It also re-runs the
    // convolutions over every overlap, which is the other half of "very slow".
    //
    // So tiling becomes what it was meant to be - the way to survive a card that
    // cannot hold the whole decode - rather than the default.
    // Structural term only, WITHOUT the 2x margin the per-tile constant carries. That
    // margin exists to stop a "fits" verdict OOMing mid-decode, and here it is not
    // needed: the whole decode is only ATTEMPTED, and a failure falls through to the
    // tiled path below rather than to the user. Carrying the margin made the estimate
    // 8.6 GB against 5.6 GB free, so the whole decode was never once chosen and every
    // render kept the tile seams.
    let whole_transient = (lh as u64 * 8) * (lw as u64 * 8) * 128 * 8 * 4;
    let whole_dev = crate::inference::place::vram_manager::pick_device_for(
        "qwen-image vae whole decode",
        whole_transient,
    )
    .filter(|(_, free, _)| *free >= whole_transient)
    .map(|(_, _, d)| d);
    let whole_first: Option<NT> = whole_dev.and_then(|d| {
        info!(
            "Qwen-Image VAE: decoding whole on {} ({:.1} GB transients) - no tile seams",
            dev_label(&d),
            whole_transient as f64 / 1e9
        );
        match decode_on(&d, true) {
            Ok(img) => img.to_device(&Device::Cpu).ok(),
            Err(e) => {
                info!("Qwen-Image VAE: whole decode did not complete ({e}); tiling instead");
                None
            }
        }
    });
    let img = match whole_first {
        Some(img) => img,
        None => match decode_on(&planned, false) {
            Ok(img) => img.to_device(&Device::Cpu)?,
            Err(e) => match &state.vae_cpu {
                Some(cpu_vae) => {
                    // The planned device still failed (an estimate can be wrong, or another
                    // engine took the memory in between). Try the OTHER GPUs, fastest first,
                    // before the slow CPU decode - the decoder weights are small, so moving
                    // them to a card with room beats a host decode by an order of magnitude.
                    let mut gpu_result = None;
                    for (_, _, alt_dev) in crate::inference::place::vram_manager::probe(transient) {
                        if Device::same_device(&alt_dev, &state.vae_device)
                            || Device::same_device(&alt_dev, &planned)
                        {
                            continue;
                        }
                        info!(
                            "Qwen-Image VAE decode on {} failed ({e}); retrying on {}",
                            dev_label(&state.vae_device),
                            dev_label(&alt_dev)
                        );
                        match cpu_vae.to_device(&alt_dev).and_then(|v| {
                            v.decode_spatial_tiled(
                                &lat.to_device(&alt_dev)?,
                                VAE_TILE_LATENT,
                                VAE_TILE_OVERLAP,
                            )
                        }) {
                            Ok(img) => {
                                gpu_result = Some(img.to_device(&Device::Cpu)?);
                                break;
                            }
                            Err(e2) => info!(
                                "Qwen-Image VAE decode retry on {} failed ({e2})",
                                dev_label(&alt_dev)
                            ),
                        }
                    }
                    match gpu_result {
                        Some(img) => img,
                        None => {
                            info!("Qwen-Image VAE decode falls back to CPU");
                            cpu_vae
                                .decode_spatial_tiled(
                                    &lat.to_device(&Device::Cpu)?,
                                    VAE_TILE_LATENT,
                                    VAE_TILE_OVERLAP,
                                )
                                .map_err(|e2| anyhow!("Qwen-Image VAE CPU decode: {e2}"))?
                        }
                    }
                }
                None => return Err(anyhow!("Qwen-Image VAE decode: {e}")),
            },
        },
    };

    // 4) Last 3 dims are [3, H, W], range ~[-1, 1] -> RGB PNG -> base64.
    let d = img.shape().dims().to_vec();
    let v = img.to_vec_f32();
    let (ih, iw) = (d[d.len() - 2], d[d.len() - 1]);
    let plane = ih * iw;
    let mut buf = image::RgbImage::new(iw as u32, ih as u32);
    for y in 0..ih {
        for xx in 0..iw {
            let px = |ch: usize| {
                (((v[ch * plane + y * iw + xx] * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0) as u8
            };
            buf.put_pixel(xx as u32, y as u32, image::Rgb([px(0), px(1), px(2)]));
        }
    }
    let mut png: Vec<u8> = Vec::new();
    image::DynamicImage::ImageRgb8(buf)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| anyhow!("Qwen-Image PNG encode: {e}"))?;
    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.encode(&png))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The distilled schedule against the values diffusers produces for this
    /// checkpoint's own scheduler config (`shift: 3.0`, `use_dynamic_shifting: false`)
    /// at its trained step count: base `linspace(1, 1/steps, steps)` = [1, .75, .5, .25]
    /// pushed through `shift*s / (1 + (shift-1)*s)`, then the terminal 0.
    ///
    /// A schedule is the one thing a port can get wrong while still producing a
    /// plausible image, so it is pinned to the reference rather than to our own output.
    #[test]
    fn distilled_sigmas_match_the_reference_schedule() {
        let sig = sigmas(SigmaShift::Fixed(DISTILLED_SHIFT), 4, 4096);
        let want = [1.0f32, 0.9, 0.75, 0.5, 0.0];
        assert_eq!(sig.len(), want.len());
        for (got, want) in sig.iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "sigmas {sig:?} != {want:?}");
        }
    }

    /// The fixed schedule ignores the request's geometry; the family's own does not.
    /// This is the whole difference between the two profiles, so it is worth a gate:
    /// sampling a distill on the dynamic shift is a silently wrong render.
    #[test]
    fn fixed_shift_is_resolution_independent_and_dynamic_is_not() {
        let small = sigmas(SigmaShift::Fixed(DISTILLED_SHIFT), 8, 1024);
        let large = sigmas(SigmaShift::Fixed(DISTILLED_SHIFT), 8, 4096);
        assert_eq!(small, large);
        assert_ne!(
            sigmas(SigmaShift::Dynamic, 8, 1024),
            sigmas(SigmaShift::Dynamic, 8, 4096)
        );
    }

    /// Guidance 1 is the identity, so the second branch must not be run: that is where
    /// the distill's speed comes from, and running it would also feed the model a
    /// negative prompt it was never trained against.
    #[test]
    fn guidance_branch_runs_only_when_the_scale_asks_for_it() {
        assert!(!needs_guidance_branch(1.0));
        assert!(needs_guidance_branch(4.0));
        assert!(needs_guidance_branch(3.5));
    }

    #[test]
    fn step_distilled_detects_the_flash_checkpoint_only() {
        assert!(is_step_distilled("qwen-image-flash"));
        assert!(is_step_distilled("Qwen-Image-Flash"));
        assert!(!is_step_distilled("qwen-image"));
        assert!(!is_step_distilled("rayqwest"));
    }
}
