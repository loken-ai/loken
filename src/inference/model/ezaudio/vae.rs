//! EzAudio (MIT, text->SFX) Oobleck VAE decoder loader.
//!
//! EzAudio ships its weights as torch `.pt` (read full-Rust via [`crate::tensor::pth`]),
//! and its autoencoder is the SAME Oobleck conv architecture as ACE-Step's VAE - only the config
//! differs (24 kHz mono, 4 decoder blocks, strides [10,6,4,2], channels 1024->512->256->128, latent
//! 128, downsample 480). So this module just maps EzAudio's `.pt` decoder tensors onto the existing
//! [`OobleckDecoder`] and reuses its forward (`decode_chunked`) unchanged.

use crate::inference::model::acestep::vae::{
    fuse_weight_norm, BlockW, ConvW, OobleckDecoder, ResUnitW, SnakeW,
};
#[cfg(feature = "cuda")]
use crate::tensor::DType;
use crate::tensor::{Device as ND, Result, Tensor as NT};
use std::collections::HashMap;

/// Resolve one of EzAudio's `.pt` checkpoints from the HF cache.
pub fn ezaudio_pt(rel: &str) -> std::path::PathBuf {
    crate::inference::cache::hf::file("models--OpenSound--EzAudio", rel)
}

fn get<'a>(m: &'a HashMap<String, NT>, name: &str) -> Result<&'a NT> {
    m.get(name)
        .ok_or_else(|| crate::tensor::Error(format!("ezaudio-vae: missing tensor `{name}`")))
}

/// Build a `ConvW` from a weight-norm pair (`weight_v`/`weight_g`) + bias, mirroring
/// `vae_load_conv`'s kernel layout (conv `[c_out,c_in,k]`; convT raw `[c_in,c_out,k]`),
/// incl. the CUDA F16 conv / rearranged-F16 convT fast-path.
fn conv_from_pt(
    m: &HashMap<String, NT>,
    prefix: &str,
    is_convt: bool,
    eps: f32,
    dev: &ND,
) -> Result<ConvW> {
    let vt = get(m, &format!("{prefix}.weight_v"))?;
    let gt = get(m, &format!("{prefix}.weight_g"))?;
    let vd = vt.dims().to_vec();
    let v = vt.to_vec_f32();
    let g = gt.to_vec_f32();
    let d0 = g.len();
    let w = fuse_weight_norm(&v, &g, d0, eps);
    let (c_out, c_in, k) = if is_convt {
        (vd[1], vd[0], vd[2])
    } else {
        (vd[0], vd[1], vd[2])
    };
    let b = match m.get(&format!("{prefix}.bias")) {
        Some(t) => t.to_vec_f32(),
        None => vec![0.0; c_out],
    };
    let wt = if is_convt {
        #[cfg(feature = "cuda")]
        if matches!(dev, ND::Cuda(_)) {
            // [c_in,c_out,k] -> [k*c_out, c_in] so the convT contraction is a plain GEMM, F16.
            let mut wr = vec![0f32; k * c_out * c_in];
            for ci in 0..c_in {
                for oc in 0..c_out {
                    for kk in 0..k {
                        wr[(kk * c_out + oc) * c_in + ci] = w[(ci * c_out + oc) * k + kk];
                    }
                }
            }
            NT::from_vec_f32(wr, (k * c_out, c_in, 1))?
                .to_device(dev)?
                .to_dtype(DType::F16)?
        } else {
            NT::from_vec_f32(w, (c_in, c_out, k))?.to_device(dev)?
        }
        #[cfg(not(feature = "cuda"))]
        NT::from_vec_f32(w, (c_in, c_out, k))?.to_device(dev)?
    } else {
        let t = NT::from_vec_f32(w, (c_out, c_in, k))?.to_device(dev)?;
        #[cfg(feature = "cuda")]
        let t = if matches!(dev, ND::Cuda(_)) {
            t.to_dtype(DType::F16)?
        } else {
            t
        };
        t
    };
    let bt = NT::from_vec_f32(b, (1, c_out, 1))?.to_device(dev)?;
    Ok(ConvW {
        wt,
        bt,
        c_in,
        c_out,
        k,
    })
}

/// Build a `SnakeW` from `alpha`/`beta` (pre-exp'd: `alpha=exp(α)`, `inv_beta=1/exp(β)`),
/// matching `vae_load_snake`.
fn snake_from_pt(m: &HashMap<String, NT>, prefix: &str, dev: &ND) -> Result<SnakeW> {
    let a = get(m, &format!("{prefix}.alpha"))?.to_vec_f32();
    let b = get(m, &format!("{prefix}.beta"))?.to_vec_f32();
    let c = a.len();
    let alpha: Vec<f32> = a.iter().map(|x| x.exp()).collect();
    let inv_beta: Vec<f32> = b.iter().map(|x| 1.0 / x.exp()).collect();
    Ok(SnakeW {
        alpha: NT::from_vec_f32(alpha, (1, c, 1))?.to_device(dev)?,
        inv_beta: NT::from_vec_f32(inv_beta, (1, c, 1))?.to_device(dev)?,
        c,
    })
}

/// What the decoder puts on the card, from the checkpoint already in memory.
///
/// The weight-norm pair is fused into one f32 kernel weight, so the resident set is the
/// fused half of what the file carries; the file also holds the encoder, which this loader
/// does not read. Neither is visible in the file size, which is why this counts tensors.
fn decoder_resident_bytes(m: &HashMap<String, NT>, prefix: &str) -> u64 {
    /// Bytes per element of a fused kernel weight.
    const RESIDENT_BYTES: u64 = 4;
    let matched: u64 = m
        .iter()
        .filter(|(k, _)| k.starts_with(prefix) && !k.ends_with(".weight_g"))
        .map(|(_, t)| t.elem_count() as u64 * RESIDENT_BYTES)
        .sum();
    if matched > 0 {
        return matched;
    }
    // Never plan against zero: a renamed prefix must charge the whole checkpoint.
    m.values()
        .map(|t| t.elem_count() as u64 * RESIDENT_BYTES)
        .sum()
}

/// The decoder's `(channels out, cumulative upsample)` per block, from its transposed
/// convolutions - the two things that set what one decode window costs.
fn decoder_levels(m: &HashMap<String, NT>, prefix: &str, strides: &[usize]) -> Vec<(usize, usize)> {
    let mut levels = Vec::with_capacity(strides.len());
    let mut up = 1usize;
    for (bi, &stride) in strides.iter().enumerate() {
        up *= stride.max(1);
        // ConvT weight is `[c_in, c_out, k]`; the level's cost follows what it EMITS.
        let ch = m
            .get(&format!("{prefix}.{}.layers.1.weight_v", bi + 1))
            .and_then(|t| t.dims().get(1).copied())
            .unwrap_or(0);
        levels.push((ch, up));
    }
    levels
}

/// Load the EzAudio Oobleck VAE decoder from its `.pt`. Key scheme (stable-audio-tools,
/// prefix `autoencoder.decoder.layers.`): 0 = conv_in; 1..=4 = blocks (`.layers.0` Snake,
/// `.layers.1` convT, `.layers.{2,3,4}` ResUnits, each `.layers.{0,1,2,3}` = Snake,conv,Snake,conv);
/// 5 = final Snake; 6 = conv_out. strides [10,6,4,2], dilations [1,3,9].
pub fn load_ezaudio_decoder(pt_path: &str, eps: f32) -> Result<OobleckDecoder> {
    let m = crate::tensor::pth::read_pt(pt_path)?;
    let p = "autoencoder.decoder.layers";
    let strides = [10usize, 6, 4, 2];
    // Placed on what THIS decoder holds and what ONE of its windows costs, both read from
    // the checkpoint in hand. It was gated on the file's size - with a typed 200 MB when
    // that could not be read - against the 48 kHz decoder's reserve, which upsamples by
    // four times as much as this one does and so was never this decoder's figure.
    let model_size = decoder_resident_bytes(&m, p);
    let levels = decoder_levels(&m, p, &strides);
    let dev = crate::inference::model::acestep::vae::vae_best_device_with_reserve(
        model_size,
        crate::inference::place::audio_demand::oobleck_reserve(
            crate::inference::place::audio_demand::ACE_VAE_REFERENCE_WINDOW,
            &levels,
            crate::inference::place::audio_demand::OOBLECK_KERNEL,
        ),
    );

    let conv1 = conv_from_pt(&m, &format!("{p}.0"), false, eps, &dev)?;
    let dils = [1usize, 3, 9];
    let mut blocks = Vec::with_capacity(4);
    for (bi, &stride) in strides.iter().enumerate() {
        let bp = format!("{p}.{}", bi + 1);
        let snake1 = snake_from_pt(&m, &format!("{bp}.layers.0"), &dev)?;
        let conv_t = conv_from_pt(&m, &format!("{bp}.layers.1"), true, eps, &dev)?;
        let mut res = Vec::with_capacity(3);
        for (r, &dilation) in dils.iter().enumerate() {
            let rp = format!("{bp}.layers.{}.layers", r + 2);
            res.push(ResUnitW {
                snake1: snake_from_pt(&m, &format!("{rp}.0"), &dev)?,
                conv1: conv_from_pt(&m, &format!("{rp}.1"), false, eps, &dev)?,
                snake2: snake_from_pt(&m, &format!("{rp}.2"), &dev)?,
                conv2: conv_from_pt(&m, &format!("{rp}.3"), false, eps, &dev)?,
                dilation,
            });
        }
        blocks.push(BlockW {
            snake1,
            conv_t,
            stride,
            padding: stride / 2,
            res,
        });
    }
    let snake_final = snake_from_pt(&m, &format!("{p}.5"), &dev)?;
    let conv2 = conv_from_pt(&m, &format!("{p}.6"), false, eps, &dev)?;
    Ok(OobleckDecoder {
        conv1,
        blocks,
        snake_final,
        conv2,
        device: dev,
    })
}

#[cfg(test)]
mod tests {
    //! The ignored cases here need real weights, a device, or a reference dump on
    //! this machine; nothing about them is automatic. Run one by name with
    //!   cargo test --release -p loken --lib NAME -- --ignored --nocapture
    use super::*;

    #[test]
    #[ignore = "needs EzAudio 1m.pt (config hf_models_dir)"]
    fn dump_decoder_keys() {
        let p = ezaudio_pt("ckpts/vae/1m.pt");
        let m = crate::tensor::pth::read_pt(p.to_str().unwrap()).unwrap();
        let mut keys: Vec<_> = m
            .keys()
            .filter(|k| k.contains("decoder"))
            .cloned()
            .collect();
        keys.sort();
        println!("{} decoder tensors:", keys.len());
        for k in &keys {
            println!("  {k}  {:?}", m[k].dims());
        }
    }

    #[test]
    #[ignore = "needs EzAudio 1m.pt; sanity decode (length + finite)"]
    fn sanity_decode() {
        use crate::inference::model::acestep::vae::encode_wav_s16le;
        let p = ezaudio_pt("ckpts/vae/1m.pt");
        let dec = load_ezaudio_decoder(p.to_str().unwrap(), 1e-12).unwrap();
        let (t, c) = (32usize, 128usize);
        let latent = vec![0.05f32; c * t]; // channel-major [c,t]
                                           // NOTE: `decode()` (untiled) derives t_audio from the actual output. `decode_chunked`
                                           // hardcodes UP=1920 (ACE-Step's 48kHz upsample) and can't yet tile EzAudio's 480x  - 
                                           // parameterizing the chunk upsample ratio is a Stage-4 follow-up for long-form decode.
        let (audio, c_out, t_audio) = dec.decode(&latent, c, t).unwrap();
        assert_eq!(c_out, 1, "EzAudio VAE is mono");
        assert_eq!(t_audio, t * 480, "downsample 480 -> {} samples", t * 480);
        assert!(audio.iter().all(|x| x.is_finite()), "non-finite output");
        let wav = encode_wav_s16le(&audio, c_out, t_audio, 24000);
        std::fs::write("results/acestep/ezaudio_vae_sanity.wav", wav).ok();
        println!("✅ EzAudio VAE decode: {t_audio} samples mono @24kHz, all finite");
    }
}
