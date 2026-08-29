//! Boogu-Image text-to-image engine. Pipeline: Qwen3-VL-8B encode -> DMD few-step student DiT
//! -> FLUX VAE decode -> PNG.
//!
//! All heavy components are RESIDENT: the fp8 checkpoints are folded + re-quantized to Q8_0 once
//! at load (fp8_scaled QVarBuilder) and placed by HeteroPlan (DiT undivided on the fastest GPU
//! that fits, encoder on the best-fit remaining device, OOM-fallback cascade; never hardcoded).
//! The sampler is the DMD student predict/renoise loop and the conditioning byte-matches the
//! reference pipeline's chat template; both are validated against the official implementation
//! (see native_boogu_dit's ref-diff tests).

use anyhow::{anyhow, Result as AnyResult};

use crate::inference::model::boogu::dit::BooguTransformer2DModel;
use crate::inference::model::flux::vae::{AutoEncoder, Config as VaeConfig};
use crate::inference::model::qwen3vl::textenc::Qwen3VlTextEncoder;
use crate::inference::place::layer_executor::HeteroPlan;
use crate::tensor::DType;
use crate::tensor::{Device, Tensor as NT};
use std::collections::HashMap;

/// Relative paths under `huggingface_models_dir`.
const REL_DIT: &str = "boogu/diffusion_models/boogu_image_turbo_fp8_scaled.safetensors";
const REL_ENC: &str = "boogu/text_encoders/qwen3vl_8b_fp8_scaled.safetensors";
const REL_VAE: &str = "boogu/vae/flux1_vae_bf16.safetensors";
/// Qwen3-VL shares the Qwen2 BPE tokenizer (vocab 151936); reuse the local Qwen2.5-VL one.
const REL_TOK: &str = "qwen2.5-vl/tokenizer.json";

const NEG_PROMPT: &str = "blurry, low quality, distorted, ugly, deformed";

/// The DiT conditions on the Qwen3-VL hidden states over the FULL chat-templated sequence,
/// exactly as the reference pipeline builds it: `apply_chat_template([system, user])` with the
/// official T2I system prompt (`SYSTEM_PROMPT_4_T2I_UNIFIED`), NO assistant generation prompt,
/// and NO tokens dropped (the reference passes the whole sequence, system prompt included; only
/// vision-token features are ever removed, which a text-to-image prompt has none of).
const SYS_PREFIX: &str = "<|im_start|>system\nYou are a helpful assistant that generates high-quality images based on user instructions. The instructions are as follows.<|im_end|>\n<|im_start|>user\n";
const SYS_SUFFIX: &str = "<|im_end|>\n";

/// Resident state: the DiT (Q8_0, on the fastest GPU), the Qwen3-VL encoder (Q8_0, on a second
/// GPU when one has room, else CPU), the FLUX VAE, and the tokenizer. Everything is quantized to
/// Q8_0 ONCE at load and kept RESIDENT across generations - `generate` reuses these, it never
/// reloads or requantizes. The DiT (~10 GB) and encoder (~9 GB) do not co-fit one 16 GB GPU, so
/// they are placed on separate GPUs (same policy as qwen_image_engine).
pub struct BooguModelState {
    /// DiT, its blocks distributed across devices by the plan. Run 2x/step under CFG - the hot path.
    dit: BooguTransformer2DModel,
    /// The DiT's I/O device (its input latent + output live here; == `dit.input_device()`).
    dit_device: Device,
    /// Qwen3-VL text encoder, its layers distributed by the plan (placement is never hardcoded).
    encoder: Qwen3VlTextEncoder,
    /// FLUX VAE decoder (facade substrate; handles the latent scale/shift internally).
    vae: AutoEncoder,
    /// Facade device the VAE decodes on (the DiT's GPU).
    vae_facade_dev: crate::tensor::Device,
    /// CPU copy of the FLUX VAE (~160 MB), the never-OOM fallback: the DiT stays resident on the
    /// GPU, so a large (1k-2k) VAE upsampling can exceed the GPU's remaining VRAM even tiled.
    vae_cpu: AutoEncoder,
    tokenizer: tokenizers::Tokenizer,
    /// Leading tokens dropped from the encoder output. 0 for text-to-image: the reference
    /// pipeline conditions on the FULL templated sequence (kept for the edit path, which removes
    /// vision-token features).
    drop_idx: usize,
}

impl BooguModelState {
    /// Whether the DiT is running split across devices - see
    /// [`BooguTransformer2DModel::is_split`].
    pub fn dit_is_split(&self) -> bool {
        self.dit.is_split()
    }
}

/// Native generation side of the family (Boogu-Image is a 1K-resolution model per its card);
/// the runtime reserve is sized for a generation at this resolution.
const NATIVE_SIDE: usize = 1024;
/// Caption token budget the reserve accounts for (the reference pipeline's
/// `max_sequence_length` default).
const MAX_TEXT_TOKENS: usize = 1280;

/// Per-device VRAM reserved for ONE generation's runtime scratch, DERIVED from the architecture
/// and the checkpoint instead of a fixed constant (the plan packs weights into
/// `free - reserve`). Terms, each a structural fact of the forward:
/// - DiT activations: ~12 concurrently-live `[seq, hidden]` F32 buffers (hidden + residual,
///   q/k/v with GQA-repeated k/v, their transposed contiguous copies, attention out, SwiGLU
///   gate/up) at `seq = caption + (side/16)^2` patch tokens.
/// - Tiled SDPA: scores + softmax `[tile, seq, heads]` F32 (tile = 512 in the attention path).
/// - Dequant scratch: the largest 2-D weight materialized F32 (4 B) + its BF16 GEMM copy (2 B),
///   read from the checkpoint header.
/// - Tiled FLUX-VAE decode sharing the device: ~6 live conv buffers at the decode tile
///   (64 latent px * 8 upsample = 512 px) over the decoder's widest (128) channels.
fn runtime_reserve(
    cfg: &crate::inference::model::boogu::dit::Config,
    dit_path: &std::path::Path,
) -> u64 {
    let seq = (MAX_TEXT_TOKENS + (NATIVE_SIDE / 16) * (NATIVE_SIDE / 16)) as u64;
    let f32b = 4u64;
    let act = 12 * seq * cfg.hidden_size as u64 * f32b;
    let attn_tile = 512u64;
    let scores = 2 * attn_tile * seq * cfg.num_heads as u64 * f32b;
    let dequant = crate::inference::load::fp8_scaled::largest_2d_elems(dit_path) * 6;
    let vae_tile_px = (64 * 8) as u64;
    let vae = vae_tile_px * vae_tile_px * 128 * 6 * f32b;
    act + scores + dequant + vae
}

/// Plan a model UNDIVIDED on the fastest GPU that fits it whole (no cross-GPU sync - matters for the
/// hot per-step DiT); if no single GPU fits, let HeteroPlan spill it across GPUs + CPU. `cuda` is
/// fastest-first (by throughput). Placement is thus adaptive and role-aware, never hardcoded.
fn plan_undivided_or_spill(n_blocks: usize, size: u64, cuda: &[(usize, u64)]) -> HeteroPlan {
    for &(idx, free) in cuda {
        if free >= size {
            return HeteroPlan::forced_gpu(n_blocks, n_blocks, idx);
        }
    }
    HeteroPlan::calculate(n_blocks, size, cuda, &[], 1.0)
}

/// Load a model with an OOM-fallback CASCADE so a load can NEVER hard-OOM: ideal placement
/// (undivided on the fastest GPU that fits, or a natural spill) -> on OOM, trim the CUDA pool and
/// retry an even GPU split -> on OOM, CPU. Mirrors the FLUX single-device -> split -> CPU cascade.
fn load_with_fallback<T>(
    n_blocks: usize,
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
    match load(&plan_undivided_or_spill(n_blocks, size, cuda)) {
        Ok(m) => return Ok(m),
        Err(e) => {
            if cancelled() {
                return Err(e);
            }
            tracing::warn!(
                "Boogu {what}: ideal placement failed ({e:#}); retrying split across GPUs"
            )
        }
    }
    #[cfg(feature = "cuda")]
    crate::inference::engine::llm_engine::trim_cuda_pools();
    if !cuda.is_empty() {
        match load(&HeteroPlan::split_across_cuda(n_blocks, cuda)) {
            Ok(m) => return Ok(m),
            Err(e) => {
                if cancelled() {
                    return Err(e);
                }
                tracing::warn!("Boogu {what}: GPU split failed ({e:#}); falling back to CPU")
            }
        }
        #[cfg(feature = "cuda")]
        crate::inference::engine::llm_engine::trim_cuda_pools();
    }
    // Empty CUDA list -> the plan puts every block on CPU (never OOMs).
    load(&HeteroPlan::calculate(n_blocks, size, &[], &[], 1.0))
}

/// Size of the family's HOT component checkpoint (the DiT) - the figure the pressure protocol
/// needs BEFORE the engine load runs. 0 when the file is absent.
pub fn hot_component_bytes(hf_models_dir: &str) -> u64 {
    std::fs::metadata(std::path::Path::new(hf_models_dir).join(REL_DIT))
        .map(|m| m.len())
        .unwrap_or(0)
}
/// Deterministic gaussian noise source (Box-Muller over splitmix64). ONE stateful stream per
/// generation, drawn from sequentially - mirroring the reference pipeline's single torch
/// `generator` that produces the init latent AND every renoise draw. Separate per-draw streams
/// with related seeds are NOT independent (xorshift/LCG state relations survive the whole
/// stream), and the DMD renoise assumes fresh white noise each step: correlated re-injections
/// leave structured residue the denoiser was never trained to remove.
struct NoiseRng {
    s: u64,
}

impl NoiseRng {
    fn new(seed: u64) -> Self {
        // splitmix64 scrambles the (possibly tiny) user seed into a well-mixed state.
        NoiseRng {
            s: seed.wrapping_add(0x9E37_79B9_7F4A_7C15),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // splitmix64: sequential, full-period, avalanching - draws are independent.
        self.s = self.s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn fill(&mut self, n: usize) -> Vec<f32> {
        let mut u = || (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32;
        (0..n)
            .map(|_| {
                let (a, b) = (u().max(1e-9), u());
                (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
            })
            .collect()
    }
}

/// Load Boogu RESIDENT: the DiT (Q8_0) on `primary` (the fastest GPU), the Qwen3-VL encoder (Q8_0)
/// on a second GPU (or CPU on a single-GPU host), and the FLUX VAE - all quantized once and kept
/// across generations. `generate` reuses them; nothing is reloaded or requantized per request.
pub fn load(
    hf_models_dir: &str,
    primary: &Device,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> AnyResult<BooguModelState> {
    let base = std::path::Path::new(hf_models_dir);
    let dit_path = base.join(REL_DIT);
    let enc_path = base.join(REL_ENC);
    let vae_path = base.join(REL_VAE);
    let tok_path = base.join(REL_TOK);
    for (p, what) in [
        (&dit_path, "DiT"),
        (&enc_path, "encoder"),
        (&vae_path, "VAE"),
        (&tok_path, "tokenizer"),
    ] {
        if !p.exists() {
            return Err(anyhow!("Boogu {what} not found at {}", p.display()));
        }
    }

    let tokenizer =
        tokenizers::Tokenizer::from_file(&tok_path).map_err(|e| anyhow!("Boogu tokenizer: {e}"))?;
    // The reference pipeline conditions on the FULL templated sequence (system prompt included) -
    // nothing is dropped for text-to-image.
    let drop_idx = 0usize;

    // Placement is decided by the adaptive HeteroPlan (fastest-GPU-first, spill to the next GPU
    // then CPU) - NEVER a hardcoded device. Probe once for the device handles (shared by BOTH
    // models so their cross-device transfers see the same physical devices); plan the DiT, then
    // re-probe the FREE amounts (the DiT is now resident) and plan the encoder on what remains.
    let _ = primary; // placement comes from the plan, not the caller's hint
    let gpu_reserve = runtime_reserve(
        &crate::inference::model::boogu::dit::Config::default(),
        &dit_path,
    );
    tracing::info!(
        "Boogu runtime reserve (arch-derived): {:.2} GB/GPU",
        gpu_reserve as f64 / 1e9
    );
    let mut cuda_devices: HashMap<usize, Device> = HashMap::new();
    let dit_cuda: Vec<(usize, u64)> = crate::inference::place::vram_manager::probe(gpu_reserve)
        .into_iter()
        .map(|(i, f, d)| {
            cuda_devices.insert(i, d);
            (i, f)
        })
        .collect();
    let dit_sz = std::fs::metadata(&dit_path).map(|m| m.len()).unwrap_or(0);
    // The plan distributes the 40-block main stack (8 double-stream + 32 single-stream). Undivided
    // on the fastest GPU that fits (hot path); OOM-fallback cascade -> split -> CPU (never OOM).
    let dit_path_str = dit_path.to_string_lossy().to_string();
    let dit = load_with_fallback(40, dit_sz, &dit_cuda, "DiT", cancel, |plan| {
        BooguTransformer2DModel::load_cancellable(&dit_path_str, &cuda_devices, plan, cancel)
            .map_err(|e| anyhow!("{e}"))
    })?;

    // Re-probe FREE VRAM (the DiT is now resident); reuse the SAME device handles for the encoder so
    // its output and the DiT input reference the same physical devices.
    let enc_cuda: Vec<(usize, u64)> = crate::inference::place::vram_manager::probe(gpu_reserve)
        .into_iter()
        .filter(|(i, _, _)| cuda_devices.contains_key(i))
        .map(|(i, f, _)| (i, f))
        .collect();
    let enc_sz = std::fs::metadata(&enc_path).map(|m| m.len()).unwrap_or(0);
    // One-shot encoder: best-fit remaining device (undivided on the fastest GPU that still fits,
    // e.g. GPU1 whole rather than splitting GPU0's leftover); same OOM-fallback cascade.
    let enc_path_str = enc_path.to_string_lossy().to_string();
    let encoder = load_with_fallback(36, enc_sz, &enc_cuda, "encoder", cancel, |plan| {
        Qwen3VlTextEncoder::load_cancellable(&enc_path_str, &cuda_devices, plan, cancel)
            .map_err(|e| anyhow!("{e}"))
    })?;

    // The DiT's I/O device (where its output lands) is where the VAE decodes.
    let dit_input_device = dit.input_device().clone();
    // FLUX VAE (small, ~160 MB) on the DiT's I/O device (the facade IS native).
    let vae_facade_dev = dit_input_device.clone();
    let vae_native_dev = dit_input_device.clone();
    let vae_vb = unsafe {
        crate::tensor::VarBuilder::from_files(
            &[vae_path.to_string_lossy().to_string()],
            crate::tensor::DType::F32,
            &vae_native_dev,
        )
        .map_err(|e| anyhow!("Boogu VAE VarBuilder: {e}"))?
    };
    let vae = AutoEncoder::new(&VaeConfig::schnell(), vae_vb)
        .map_err(|e| anyhow!("Boogu VAE load: {e}"))?;

    // CPU copy for the never-OOM decode fallback (small: ~160 MB).
    let vae_cpu_vb = unsafe {
        crate::tensor::VarBuilder::from_files(
            &[vae_path.to_string_lossy().to_string()],
            crate::tensor::DType::F32,
            &crate::tensor::Device::Cpu,
        )
        .map_err(|e| anyhow!("Boogu VAE CPU VarBuilder: {e}"))?
    };
    let vae_cpu = AutoEncoder::new(&VaeConfig::schnell(), vae_cpu_vb)
        .map_err(|e| anyhow!("Boogu VAE CPU load: {e}"))?;

    Ok(BooguModelState {
        dit,
        dit_device: dit_input_device,
        encoder,
        vae,
        vae_facade_dev,
        vae_cpu,
        tokenizer,
        drop_idx,
    })
}

/// Generate one image. Encodes positive + negative prompts (loading then dropping the encoder),
/// runs the flow-match Euler sampler with CFG on the resident DiT, then FLUX-VAE-decodes to a PNG
/// base64 string.
pub fn generate(
    state: &BooguModelState,
    prompt: &str,
    width: usize,
    height: usize,
    num_steps: usize,
    guidance: f32,
    seed: u64,
    cancel: &crate::inference::serve::cancel::CancelToken,
) -> AnyResult<String> {
    let dev = &state.dit_device;
    let steps = num_steps.max(1);
    // Boogu-Turbo reference default is CFG 1.0 (distilled without classifier-free guidance).
    let cfg_scale = if guidance > 0.0 { guidance } else { 1.0 };
    // FLUX VAE downsample 8; latent is 16-channel.
    let (lh, lw) = (height / 8, width / 8);
    if lh == 0 || lw == 0 {
        return Err(anyhow!("Boogu: image too small ({width}x{height})"));
    }

    // 1) Encode positive + negative prompts on the RESIDENT encoder (its own GPU), then move the
    //    conditioning to the DiT's device. No load/requantize here - the encoder stays resident.
    let dit = &state.dit;
    let (txt, txt_len, neg, neg_len) = {
        // Wrap in the Qwen chat template and drop the system prefix (only the user prompt conditions).
        let pos_tpl = format!("{SYS_PREFIX}{prompt}{SYS_SUFFIX}");
        let neg_tpl = format!("{SYS_PREFIX}{NEG_PROMPT}{SYS_SUFFIX}");
        let p = state
            .encoder
            .encode_text(&state.tokenizer, &pos_tpl, state.drop_idx)
            .map_err(|e| anyhow!("Boogu encode+: {e}"))?;
        let n = state
            .encoder
            .encode_text(&state.tokenizer, &neg_tpl, state.drop_idx)
            .map_err(|e| anyhow!("Boogu encode-: {e}"))?;
        let pl = p.shape().dims()[0];
        let nl = n.shape().dims()[0];
        (p.to_device(dev)?, pl, n.to_device(dev)?, nl)
    };

    // 2) Init the noise latent [in_ch, lh, lw] on the DiT's device (the DiT is already resident).
    let in_ch = dit.in_channels();
    let latn = in_ch * lh * lw;
    let mut rng = NoiseRng::new(seed);
    let mut x = NT::from_vec_f32(rng.fill(latn), (in_ch, lh, lw))?.to_device(dev)?;

    // 3) DMD student few-step sampler (the Boogu-Turbo distillation). NOT flow-match Euler: the
    //    turbo model is a Distribution-Matching-Distillation student and inference is a
    //    predict-then-renoise loop over an ASCENDING sigma schedule where sigma is the DATA weight
    //    (sigma=1 => clean, sigma=0 => noise), the opposite of the flow-match convention.
    //      sigmas = linspace(conditioning_sigma, 1.0, steps+1)[:-1]  (conditioning_sigma = 0.001)
    //      predict:  x0 = latents + (1 - sigma) * model_pred     (model_pred = raw DiT output)
    //      renoise:  latents = (1 - sigma_next) * noise + sigma_next * x0   (fresh noise per step)
    //    The distilled student runs WITHOUT classifier-free guidance (guidance is forced to 1.0),
    //    so there is a single conditional forward per step and the negative prompt is unused.
    let _ = (&neg, neg_len, cfg_scale);
    const CONDITIONING_SIGMA: f32 = 0.001;
    let sig: Vec<f32> = (0..steps)
        .map(|k| CONDITIONING_SIGMA + (k as f32) * (1.0 - CONDITIONING_SIGMA) / steps as f32)
        .collect();
    for i in 0..steps {
        cancel.bail()?;
        let sigma = sig[i];
        // predict: x0 = x + (1 - sigma) * model_pred
        let pred = dit.forward(&x, sigma, &txt, txt_len)?;
        let x0 = x.add(&pred.affine(1.0 - sigma, 0.0)?)?;
        if i + 1 < steps {
            // renoise to the next (higher = cleaner) sigma with fresh gaussian noise drawn from
            // the SAME sequential stream as the init latent (independent across steps).
            let sn = sig[i + 1];
            let nz = NT::from_vec_f32(rng.fill(latn), (in_ch, lh, lw))?.to_device(dev)?;
            x = nz.affine(1.0 - sn, 0.0)?.add(&x0.affine(sn, 0.0)?)?;
        } else {
            x = x0;
        }
    }

    // 4) FLUX VAE decode: [in_ch, lh, lw] -> facade [1, in_ch, lh, lw] -> [1, 3, H, W]. The DiT
    //    stays resident on this GPU, so the VAE's 8x upsampling of a 1k-2k image can exceed the
    //    remaining VRAM. Tiled decode bounds peak VRAM (a small latent is a single tile = plain
    //    decode); on any failure (OOM), fall back to the resident CPU VAE, which never OOMs.
    let lat = &x
        .unsqueeze(0)?
        .to_device(&state.vae_facade_dev)
        .map_err(|e| anyhow!("Boogu latent->facade: {e}"))?;
    let img = match state.vae.decode_tiled(&lat, 64, 8) {
        Ok(img) => img,
        Err(e) => {
            tracing::warn!("Boogu VAE GPU decode failed ({e}); falling back to CPU VAE");
            let lat_cpu = &x
                .unsqueeze(0)?
                .to_device(&crate::tensor::Device::Cpu)
                .map_err(|e| anyhow!("Boogu latent->CPU facade: {e}"))?;
            state
                .vae_cpu
                .decode(&lat_cpu)
                .map_err(|e| anyhow!("Boogu VAE CPU decode: {e}"))?
        }
    };
    let img = &img
        .to_dtype(DType::F32)
        .map_err(|e| anyhow!("Boogu image to f32: {e}"))?;

    // 5) Last 3 dims [3, H, W], range ~[-1, 1] -> RGB PNG -> base64.
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
        .map_err(|e| anyhow!("Boogu PNG encode: {e}"))?;
    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.encode(&png))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end: load Boogu on the fastest GPU and generate one small image. Validates the full
    /// pipeline (encoder Q8_0 -> drop -> DiT Q8_0 -> FLUX VAE -> PNG) fits a single GPU sequentially
    /// and produces a decodable PNG. Ignored (heavy, needs local weights + a GPU):
    ///   cargo test --release -p loken boogu_generate_one_image -- --ignored --nocapture
    #[test]
    #[ignore]
    fn boogu_generate_one_image() {
        let hf = crate::inference::cache::hf::models_dir();
        let dev = crate::inference::place::device_probe::probe_cuda_devices(0)
            .into_iter()
            .next()
            .map(|(_, _, d)| d)
            .unwrap_or(Device::Cpu);
        println!("boogu E2E on {dev:?}");
        let state = load(&hf, &dev, None).expect("load boogu");
        // Known-good config (turbo few-step): 4 steps, CFG 4.0.
        let cancel = crate::inference::serve::cancel::CancelToken::new();
        let b64 = generate(
            &state,
            "a red apple on a wooden table",
            512,
            512,
            4,
            4.0,
            42,
            &cancel,
        )
        .expect("generate");
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&b64)
            .expect("b64");
        let out = std::env::temp_dir().join("boogu_e2e.png");
        std::fs::write(&out, &bytes).expect("write png");
        println!("boogu E2E wrote {} bytes -> {}", bytes.len(), out.display());
        assert!(bytes.len() > 1000, "PNG suspiciously small");
    }
}
