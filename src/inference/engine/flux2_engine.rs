//! FLUX.2 Klein text-to-image: Qwen3-4B conditioning -> flow-match Euler DiT -> KL VAE decode.
//!
//! Every component is validated against the diffusers f32 oracle before it got here (the DiT and
//! the VAE bit-for-bit, the conditioning to `corr 0.999994` on the real tokens); this file is the
//! assembly and the placement.
//!
//! Placement is decided ENTIRELY by the generic adaptive `HeteroPlan` - fastest-GPU-first, spill
//! to the next GPU, then CPU - through the same cascade the Qwen-Image engine uses, never a
//! hardcoded device or a per-model role split. What is different here is the SCALE: the DiT is
//! ~4.1 GB resident at Q8_0 and the encoder ~2.3 GB, so the whole family fits one card with room
//! left over. That is the point of the model, and it means the usual pressure of a 20B DiT does
//! not apply.

use std::collections::HashMap;

use anyhow::{anyhow, Result as AnyResult};
use tracing::info;

use crate::inference::engine::qwen_image_engine::load_with_fallback;
use crate::inference::model::flux2::dit::{Config as DitConfig, Model as DitModel};
use crate::inference::model::flux2::sampling;
use crate::inference::model::flux2::textenc as cond;
use crate::inference::model::flux2::vae::{self, Flux2Vae};
use crate::inference::model::qwen3vl::textenc::{Config as EncConfig, Qwen3VlTextEncoder};
use crate::tensor::{DType, Device, Tensor as NT};

/// Relative locations under `huggingface_models_dir` - never absolute paths.
const REL_DIR: &str = "flux2-klein-4b";
const REL_DIT: &str = "flux2-klein-4b/transformer/diffusion_pytorch_model.safetensors";
const REL_VAE: &str = "flux2-klein-4b/vae/diffusion_pytorch_model.safetensors";
const REL_TOKENIZER: &str = "flux2-klein-4b/tokenizer/tokenizer.json";
const REL_ENCODER_DIR: &str = "flux2-klein-4b/text_encoder";

pub struct Flux2ModelState {
    dit: DitModel,
    encoder: Qwen3VlTextEncoder,
    tokenizer: tokenizers::Tokenizer,
    vae: Flux2Vae,
    dit_device: Device,
    cfg: DitConfig,
}

impl Flux2ModelState {
    /// The resident DiT's config, for callers reporting what is loaded.
    pub fn config(&self) -> &DitConfig {
        &self.cfg
    }
}

/// The encoder's shards, sorted so the builder sees a stable order.
fn encoder_shards(hf: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(hf.join(REL_ENCODER_DIR))
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|x| x.eq_ignore_ascii_case("safetensors"))
        })
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// Resident bytes of the hot component (the DiT), for the pressure protocol. The safetensors is
/// re-quantized at load and cached as a sidecar; when that exists it is the exact answer.
pub fn hot_component_bytes(hf_models_dir: &str) -> u64 {
    let p = std::path::Path::new(hf_models_dir).join(REL_DIT);
    let dt = crate::tensor::quantized::GgmlDType::Q8_0;
    if let Some(side) = crate::inference::load::fp8_scaled::sidecar_for(&p, dt) {
        if let Ok(m) = std::fs::metadata(&side) {
            if m.len() > 0 {
                return m.len();
            }
        }
    }
    let elems = crate::inference::load::fp8_scaled::total_elems(&p);
    if elems == 0 {
        return std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
    }
    elems * dt.type_size() as u64 / dt.block_size() as u64
}

/// VRAM one generation needs FREE beyond the weights, derived from the architecture and the
/// REQUESTED geometry rather than a fixed byte count.
pub fn runtime_headroom_bytes(width: usize, height: usize) -> u64 {
    let cfg = DitConfig::klein_4b();
    let dim = cfg.dim() as u64;
    let f32b = 4u64;
    let img = sampling::image_seq_len(height, width) as u64;
    let seq = img + cond::TEXT_LEN as u64;
    let buf = |tokens: u64, width: u64| tokens * width * f32b;

    // Per-block peak: q/k/v and the attention output live together, the stream input and the
    // block residual are live across it, and the single blocks' fused projection is the widest
    // tensor in the graph (3*dim qkv + 2*mlp_hidden gated MLP).
    let attn = 4 * buf(seq, dim);
    let resid = 2 * buf(seq, dim);
    let fused = buf(seq, 3 * dim + 2 * cfg.mlp_hidden() as u64);
    // Tiled attention scores, doubled by the softmax.
    let scores = 2 * 1024 * seq * cfg.num_attention_heads as u64 * f32b;
    // The sampler carries the latent and the step delta across steps.
    let sampler = 2 * buf(img, cfg.in_channels as u64);
    let dit_peak = attn + resid + fused + scores + sampler;

    // The VAE decode is a SEQUENTIAL phase - the denoise scratch is freed first - so the reserve
    // is the max of the two, never their sum.
    let vae = crate::inference::place::runtime_demand::vae_decode_bytes(height, width, 128);
    let peak = dit_peak.max(vae);
    // Allocator reality: the pool's high-water mark tracks what the denoise CHURNS, not the
    // instantaneous live set. Same measured factor the Qwen-Image engine derives.
    let reserve = peak * 3;
    info!(
        "FLUX.2 reserve for {width}x{height}: analytic peak {:.2} GB (dit {:.2}, vae {:.2}) x3 = {:.2} GB",
        peak as f64 / 1e9,
        dit_peak as f64 / 1e9,
        vae as f64 / 1e9,
        reserve as f64 / 1e9,
    );
    reserve
}

fn dev_label(d: &Device) -> String {
    match d.location() {
        crate::tensor::DeviceLocation::Cuda { gpu_id } => format!("CUDA:{gpu_id}"),
        _ => "CPU".to_string(),
    }
}

/// Deterministic gaussian noise (Box-Muller on a seeded xorshift - native has no randn). One
/// sequential stream for the whole latent, as the reference draws it from a single generator.
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

/// Load FLUX.2 Klein. Placement comes from the adaptive plan with an OOM-fallback cascade, so a
/// load can never hard-OOM; `_primary` is ignored (kept for the loader signature).
pub fn load(
    hf_models_dir: &str,
    _primary: &Device,
    geom: crate::inference::place::runtime_demand::RequestGeometry,
    cancel: Option<&crate::inference::serve::cancel::CancelToken>,
) -> AnyResult<Flux2ModelState> {
    let base = std::path::Path::new(hf_models_dir);
    let dit_path = base.join(REL_DIT);
    let vae_path = base.join(REL_VAE);
    let tok_path = base.join(REL_TOKENIZER);
    for (p, what) in [
        (&dit_path, "DiT"),
        (&vae_path, "VAE"),
        (&tok_path, "tokenizer"),
    ] {
        if !p.exists() {
            return Err(anyhow!("FLUX.2 {what} not found at {}", p.display()));
        }
    }
    let shards = encoder_shards(base);
    if shards.is_empty() {
        return Err(anyhow!(
            "FLUX.2 text encoder not found under {}/{REL_ENCODER_DIR}",
            hf_models_dir
        ));
    }

    let tokenizer = tokenizers::Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow!("FLUX.2 tokenizer: {e}"))?;

    let reserve = runtime_headroom_bytes(geom.width, geom.height);
    let mut cuda_devices: HashMap<usize, Device> = HashMap::new();
    let dit_cuda: Vec<(usize, u64)> = crate::inference::place::vram_manager::probe(reserve)
        .into_iter()
        .map(|(i, f, d)| {
            cuda_devices.insert(i, d);
            (i, f)
        })
        .collect();

    // Hot DiT (5 double + 20 single blocks), run per sampling step.
    let dit_cfg = DitConfig::klein_4b();
    let n_blocks = dit_cfg.num_layers + dit_cfg.num_single_layers;
    let dit_sz = hot_component_bytes(hf_models_dir);
    let dit_str = dit_path.to_string_lossy().to_string();
    info!(
        "FLUX.2 DiT planning: needs {:.2} GB resident; per-GPU free after the {:.2} GB reserve = {:?}",
        dit_sz as f64 / 1e9,
        reserve as f64 / 1e9,
        dit_cuda.iter().map(|(i, f)| format!("GPU{i}:{:.2}GB", *f as f64 / 1e9)).collect::<Vec<_>>()
    );
    let dit = load_with_fallback(n_blocks, dit_sz, &dit_cuda, "FLUX.2 DiT", cancel, |plan| {
        DitModel::load_hetero(&dit_str, &cuda_devices, plan, cancel).map_err(|e| anyhow!("{e}"))
    })?;
    let dit_device = dit.input_device().clone();

    // One-shot encoder: re-probe FREE (the DiT is resident now) and take what remains.
    let enc_cuda: Vec<(usize, u64)> = crate::inference::place::vram_manager::probe(reserve)
        .into_iter()
        .filter(|(i, _, _)| cuda_devices.contains_key(i))
        .map(|(i, f, _)| (i, f))
        .collect();
    let enc_cfg = EncConfig::qwen3_4b();
    // Plan on the RESIDENT footprint, not the file size. These shards are bf16 (2 B/weight) and
    // the loader re-quantizes the projections to Q8_0 (~1 B/weight), so the raw 8.05 GB is twice
    // what ends up on the card - and planning with it pushed 26 of 36 layers to the CPU that had
    // room on a GPU. Same trap the DiT's sizing already avoids; the parameter count is the
    // container-independent answer.
    let enc_dt = crate::tensor::quantized::GgmlDType::Q8_0;
    let enc_elems: u64 = shards
        .iter()
        .map(|p| crate::inference::load::fp8_scaled::total_elems(std::path::Path::new(p)))
        .sum();
    let enc_sz: u64 = if enc_elems > 0 {
        enc_elems * enc_dt.type_size() as u64 / enc_dt.block_size() as u64
    } else {
        shards
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum()
    };
    let refs: Vec<&str> = shards.iter().map(String::as_str).collect();
    let encoder = load_with_fallback(
        enc_cfg.n_layers,
        enc_sz,
        &enc_cuda,
        "FLUX.2 encoder",
        cancel,
        |plan| {
            Qwen3VlTextEncoder::load_cfg(&refs, enc_cfg, &cuda_devices, plan, cancel)
                .map_err(|e| anyhow!("{e}"))
        },
    )?;

    // The VAE is 168 MB - it decodes where the DiT's output already lives, no cascade needed.
    let vae = vae::load(vae_path.to_string_lossy().as_ref(), &dit_device, DType::F32)
        .map_err(|e| anyhow!("FLUX.2 VAE load: {e}"))?;

    info!(
        "FLUX.2 placement: DiT I/O on {}, encoder input on {}, VAE on {}",
        dev_label(&dit_device),
        dev_label(encoder.input_device()),
        dev_label(&dit_device),
    );

    Ok(Flux2ModelState {
        dit,
        encoder,
        tokenizer,
        vae,
        dit_device,
        cfg: dit_cfg,
    })
}

/// Generate one image, returning a base64 PNG.
///
/// `guidance` is accepted for signature parity and ignored: Klein is step-distilled
/// (`is_distilled: true`), its pipeline never builds an unconditional branch, and the DiT takes
/// no guidance embedding (`guidance_embeds: false`). Running a CFG pair here would double the
/// cost to steer against a branch the weights already fold in.
#[allow(clippy::too_many_arguments)]
pub fn generate(
    state: &Flux2ModelState,
    prompt: &str,
    width: usize,
    height: usize,
    num_steps: usize,
    _guidance: f32,
    seed: u64,
    cancel: &crate::inference::serve::cancel::CancelToken,
) -> AnyResult<String> {
    let dev = &state.dit_device;
    let steps = num_steps.max(1);
    let (gh, gw) = sampling::grid_for(height, width);
    let si = gh * gw;
    if si == 0 {
        return Err(anyhow!("FLUX.2: image too small ({width}x{height})"));
    }

    // 1) Conditioning: [512, 7680] on the DiT's device.
    let txt = cond::encode(&state.encoder, &state.tokenizer, prompt)
        .map_err(|e| anyhow!("FLUX.2 encode: {e}"))?
        .to_device(dev)?;
    cancel.bail()?;

    // 2) Flow-match Euler, ONE branch per step. The noise is drawn directly in the patchified
    //    space the DiT consumes (in_channels = the 32-channel latent packed 2x2).
    let sig = sampling::sigmas(si, steps);
    let mut x = NT::from_vec_f32(
        noise(si * state.cfg.in_channels, seed),
        (si, state.cfg.in_channels),
    )?
    .to_device(dev)?;
    for i in 0..steps {
        cancel.bail()?;
        // The pipeline hands the model sigma (t/1000) and the model re-multiplies by 1000.
        let v = state.dit.forward(&x, &txt, sig[i] * 1000.0, gh, gw)?;
        let dt = sig[i + 1] - sig[i];
        x = x.add(&v.affine(dt, 0.0)?)?;
    }

    // 3) Decode: the VAE owns the batch-norm un-normalisation and the 2x2 unpatchify.
    cancel.bail()?;
    let img = state
        .vae
        .decode_tokens(&x, gh, gw)
        .map_err(|e| anyhow!("FLUX.2 VAE decode: {e}"))?
        .to_device(&Device::Cpu)?;
    to_png(&img)
}

/// `[1, 3, H, W]` in `[-1, 1]` -> base64 PNG.
fn to_png(img: &NT) -> AnyResult<String> {
    let d = img.shape().dims().to_vec();
    let v = img.to_vec_f32();
    let (ih, iw) = (d[d.len() - 2], d[d.len() - 1]);
    let plane = ih * iw;
    let mut buf = image::RgbImage::new(iw as u32, ih as u32);
    for y in 0..ih {
        for x in 0..iw {
            let px = |c: usize| {
                (((v[c * plane + y * iw + x] * 0.5 + 0.5).clamp(0.0, 1.0)) * 255.0) as u8
            };
            buf.put_pixel(x as u32, y as u32, image::Rgb([px(0), px(1), px(2)]));
        }
    }
    let mut png: Vec<u8> = Vec::new();
    image::DynamicImage::ImageRgb8(buf)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| anyhow!("FLUX.2 PNG encode: {e}"))?;
    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.encode(&png))
}

/// True when this model name is served by this engine. The weights live in a plain `flux2-*`
/// directory, so the name is the directory tag the catalogue advertises.
pub fn is_flux2_model(model_name: &str) -> bool {
    let l = model_name.to_lowercase();
    l.contains("flux2") || l.contains("flux.2") || l.contains("klein")
}

/// Where this family's weights live, for the presence check the catalogue does.
pub fn weights_dir(hf_models_dir: &str) -> std::path::PathBuf {
    std::path::Path::new(hf_models_dir).join(REL_DIR)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn family_detection_covers_the_names_the_catalogue_can_produce() {
        for n in ["flux2-klein-4b", "FLUX.2-klein-9B", "klein"] {
            assert!(is_flux2_model(n), "{n} not detected");
        }
        // Must NOT swallow the FLUX.1 family, which a substring match on "flux" would.
        for n in ["flux", "flux-schnell", "rayflux", "flux-kontext"] {
            assert!(!is_flux2_model(n), "{n} wrongly claimed by FLUX.2");
        }
    }

    /// The reserve must follow the REQUEST. A flat figure is wrong in both directions: it
    /// over-reserves at 512 and under-reserves above the size it was written for.
    #[test]
    fn reserve_scales_with_the_request() {
        let small = runtime_headroom_bytes(512, 512);
        let large = runtime_headroom_bytes(1024, 1024);
        assert!(
            large > small,
            "reserve did not grow with geometry: {small} -> {large}"
        );
        assert!(small > 0);
    }
}
