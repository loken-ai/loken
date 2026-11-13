//! A whole Z-Image render with every block of the transformer on OpenCL, and its twin on the host.
//!
//! The stages run one after the other, as `generate_zimage` runs them.
//! Both draw the noise of the same seed, so they render the same picture.
//! The caption encoder is dropped before the transformer is loaded.
//! The test prints the time of each stage and the host memory after each one.
//!
//! The checkpoint is the snapshot directory linked as `zimage-turbo` in the working directory.
//! The picture is written there as `zimage-opencl.ppm` or `zimage-host.ppm`.
//!
//! The GPU is the first one that shares the host's memory, else the first one listed.
//!
//! Ignored by default: it needs that checkpoint and an OpenCL device.

use super::state::{ZIMAGE_BASE_SEQ_LEN, ZIMAGE_BASE_SHIFT, ZIMAGE_MAX_SEQ_LEN, ZIMAGE_MAX_SHIFT};
use super::{decoded_to_cpu_u8, encode_ids, format_prompt_for_qwen3, zimage_euler_step};
use crate::inference::kernel::opencl::{
    enumerate_opencl_devices, shares_host_memory, OpenCLPipelines,
};
use crate::inference::model::zimage::dit::Config;
use crate::inference::model::zimage::hetero::HeteroZImage;
use crate::inference::model::zimage::sampling::{
    calculate_shift, get_noise, FlowMatchEulerDiscreteScheduler, SchedulerConfig,
};
use crate::inference::model::zimage::text_encoder::{TextEncoderConfig, ZImageTextEncoder};
use crate::inference::model::zimage::vae::{AutoEncoderKL, VaeConfig};
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};
use crate::tensor::{DType, Device, Tensor, VarBuilder};
use std::sync::Arc;
use std::time::Instant;

/// The side of the square picture, in pixels.
const SIDE: usize = 512;

/// Denoising steps.
const STEPS: usize = 4;

/// The seed of the noise.
const SEED: u64 = 42;

const PROMPT: &str = "A lighthouse on a rocky coast at sunset, waves breaking below";

/// The anonymous and file-backed resident memory of this process and its peak, in GB.
fn memory() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let field = |name: &str| {
        status
            .lines()
            .find(|l| l.starts_with(name))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|kb| kb.parse::<f64>().ok())
            .map_or(0.0, |kb| kb * 1024.0 / 1e9)
    };
    format!(
        "anonymous {:.2} GB, file {:.2} GB, peak {:.2} GB",
        field("RssAnon:"),
        field("RssFile:"),
        field("VmHWM:")
    )
}

/// Drops the page cache of `paths` once their tensors are held elsewhere.
///
/// A run bounded in memory is charged that cache.
/// The GPU of a shared-memory host takes its buffers from the same bound.
fn drop_cached(paths: &[String]) {
    use std::os::fd::AsRawFd;
    for p in paths {
        if let Ok(f) = std::fs::File::open(p) {
            // SAFETY: the descriptor is open for the length of the call.
            unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }
}

fn files(dir: &std::path::Path, sub: &str) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir.join(sub))
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

#[test]
#[ignore = "needs the Z-Image-Turbo snapshot linked as zimage-turbo and an OpenCL device"]
fn a_whole_image_runs_through_opencl() {
    whole_image(DeviceKind::OpenCL(0), "zimage-opencl.ppm");
}

#[test]
#[ignore = "needs the Z-Image-Turbo snapshot linked as zimage-turbo"]
fn a_whole_image_runs_on_the_host() {
    whole_image(DeviceKind::Cpu, "zimage-host.ppm");
}

/// The render with every main block on `kind`, its picture written to `out`.
fn whole_image(kind: DeviceKind, out: &str) {
    let dir = std::path::Path::new("zimage-turbo");
    println!("start: {}", memory());

    // The caption, encoded on the host, then the encoder is dropped.
    let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer/tokenizer.json")).unwrap();
    let ids = encode_ids(&tokenizer, &format_prompt_for_qwen3(PROMPT), "Z-Image").unwrap();
    let count = ids.len();
    let start = Instant::now();
    let te_files = files(dir, "text_encoder");
    let te_refs: Vec<&str> = te_files.iter().map(String::as_str).collect();
    let vb = unsafe { VarBuilder::from_files(&te_refs, DType::F32, &Device::Cpu) }.unwrap();
    let encoder = ZImageTextEncoder::new(&TextEncoderConfig::z_image(), vb).unwrap();
    let loaded = start.elapsed().as_secs_f64();
    println!("text encoder loaded in {loaded:.1} s: {}", memory());
    let start = Instant::now();
    let input = Tensor::from_vec(ids, (1, count), &Device::Cpu).unwrap();
    let cap_feats = encoder.forward(&input).unwrap();
    println!(
        "text encoder: {count} tokens in {:.2} s: {}",
        start.elapsed().as_secs_f64(),
        memory()
    );
    drop(encoder);
    drop_cached(&te_files);
    let cap_mask = Tensor::ones((1, count), DType::U8, &Device::Cpu).unwrap();
    println!("text encoder dropped: {}", memory());

    // The transformer, every main block and the noise refiner on `kind`.
    let cfg = Config::z_image_turbo();
    let plan = HeteroPlan {
        segments: vec![HeteroSegment {
            kind,
            layer_start: 0,
            layer_end: cfg.n_layers,
            free_memory_bytes: u64::MAX,
        }],
        tp_cards: None,
        total_layers: cfg.n_layers,
        host_experts_from: None,
    };
    let start = Instant::now();
    let pipelines = matches!(kind, DeviceKind::OpenCL(_)).then(|| {
        let devices = enumerate_opencl_devices();
        assert!(!devices.is_empty(), "no OpenCL device");
        let device = devices
            .iter()
            .copied()
            .find(|&d| shares_host_memory(d))
            .unwrap_or(devices[0]);
        println!(
            "device: {}",
            opencl3::device::Device::new(device)
                .name()
                .unwrap_or_default()
        );
        Arc::new(OpenCLPipelines::new(device).unwrap())
    });
    let tf_files = files(dir, "transformer");
    let tf_refs: Vec<&str> = tf_files.iter().map(String::as_str).collect();
    let model = HeteroZImage::from_safetensors(
        &tf_refs,
        None,
        &cfg,
        &plan,
        &Device::Cpu,
        DType::F32,
        pipelines,
    )
    .unwrap();
    drop_cached(&tf_files);
    println!(
        "transformer loaded in {:.1} s: {}",
        start.elapsed().as_secs_f64(),
        memory()
    );

    // The denoising loop of `generate_zimage`.
    let latent = 2 * (SIDE / 16);
    let patch = cfg.all_patch_size[0];
    let mu = calculate_shift(
        (latent / patch) * (latent / patch),
        ZIMAGE_BASE_SEQ_LEN,
        ZIMAGE_MAX_SEQ_LEN,
        ZIMAGE_BASE_SHIFT,
        ZIMAGE_MAX_SHIFT,
    );
    let mut scheduler = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
    scheduler.set_timesteps(STEPS, Some(mu));
    Device::Cpu.set_seed(SEED).unwrap();
    let mut latents = get_noise(1, 16, latent, latent, &Device::Cpu)
        .unwrap()
        .unsqueeze(2)
        .unwrap();
    for step in 0..STEPS {
        let start = Instant::now();
        let t = scheduler.current_timestep_normalized();
        let t = Tensor::from_vec(vec![t as f32], (1,), &Device::Cpu).unwrap();
        let pred = model.forward(&latents, &t, &cap_feats, &cap_mask).unwrap();
        latents = zimage_euler_step(&mut scheduler, &pred, &latents).unwrap();
        println!(
            "step {}: {:.2} s: {}",
            step + 1,
            start.elapsed().as_secs_f64(),
            memory()
        );
    }
    drop(model);

    // The VAE on the host.
    let start = Instant::now();
    let vae_files = files(dir, "vae");
    let vae_refs: Vec<&str> = vae_files.iter().map(String::as_str).collect();
    let vb = unsafe { VarBuilder::from_files(&vae_refs, DType::F32, &Device::Cpu) }.unwrap();
    let vae = AutoEncoderKL::new(&VaeConfig::z_image(), vb).unwrap();
    let decoded = vae.decode(&latents.squeeze(2).unwrap()).unwrap();
    let image = decoded_to_cpu_u8(decoded).unwrap();
    println!(
        "VAE decode: {:.2} s: {}",
        start.elapsed().as_secs_f64(),
        memory()
    );

    // The picture, as a binary PPM.
    let rgb: Vec<u8> = image
        .squeeze(0)
        .unwrap()
        .permute((1, 2, 0))
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    assert_eq!(rgb.len(), SIDE * SIDE * 3);
    let mean = rgb.iter().map(|&v| v as f64).sum::<f64>() / rgb.len() as f64;
    let spread =
        (rgb.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / rgb.len() as f64).sqrt();
    println!("picture: mean {mean:.1}, deviation {spread:.1} of 255");
    let mut ppm = format!("P6\n{SIDE} {SIDE}\n255\n").into_bytes();
    ppm.extend_from_slice(&rgb);
    std::fs::write(out, ppm).unwrap();
    assert!(spread > 1.0, "a flat picture");
}
