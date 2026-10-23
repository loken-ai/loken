//! A denoising step of Z-Image through OpenCL against the host: the same weights, the same
//! inputs, the output of both and the time of each.
//!
//! The step is taken over a checkpoint trimmed to what a step needs to be measured - the stem,
//! one block of each refiner and the first `LAYERS` blocks of the stack, with the real
//! dimensions - which `zimage-trim.safetensors` in the working directory holds. The cost of a
//! block is the difference between a stack of `LAYERS` and a stack of one.
//!
//! Ignored by default: it needs that file and an OpenCL device.

use super::dit::Config;
use super::hetero::HeteroZImage;
use crate::inference::kernel::opencl::{enumerate_opencl_devices, OpenCLPipelines};
use crate::inference::place::layer_executor::{DeviceKind, HeteroPlan, HeteroSegment};
use crate::tensor::{DType, Device, Tensor};
use std::sync::Arc;

/// Blocks of the main stack in the trimmed checkpoint.
const LAYERS: usize = 4;

/// The latent height and width of the step: a 512 by 512 image.
const LATENT: usize = 64;

/// Tokens of caption.
const CAPTION: usize = 32;

fn plan(kind: DeviceKind, layers: usize) -> HeteroPlan {
    HeteroPlan {
        segments: vec![HeteroSegment {
            kind,
            layer_start: 0,
            layer_end: layers,
            free_memory_bytes: u64::MAX,
        }],
        tp_cards: None,
        total_layers: layers,
        host_experts_from: None,
    }
}

/// Inputs scaled `(x, caption)` times these; the step is taken at the scale it is trained at.
fn inputs() -> (Tensor, Tensor, Tensor, Tensor) {
    inputs_scaled(1.0, 1.0)
}

fn inputs_scaled(sx: f32, sc: f32) -> (Tensor, Tensor, Tensor, Tensor) {
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let x = Tensor::from_vec_f32(
        (0..16 * LATENT * LATENT)
            .map(|_| next() * 2.0 * sx)
            .collect(),
        vec![1, 16, 1, LATENT, LATENT],
    )
    .unwrap();
    let t = Tensor::from_vec_f32(vec![0.5], vec![1]).unwrap();
    let cap = Tensor::from_vec_f32(
        (0..CAPTION * 2560).map(|_| next() * sc).collect(),
        vec![1, CAPTION, 2560],
    )
    .unwrap();
    let mask = Tensor::from_vec_f32(vec![1.0; CAPTION], vec![1, CAPTION]).unwrap();
    (x, t, cap, mask)
}

fn build(layers: usize, kind: DeviceKind, pipelines: Option<Arc<OpenCLPipelines>>) -> HeteroZImage {
    let mut cfg = Config::z_image_turbo();
    cfg.n_layers = layers;
    cfg.n_refiner_layers = 1;
    HeteroZImage::from_safetensors(
        &["zimage-trim.safetensors"],
        None,
        &cfg,
        &plan(kind, layers),
        &Device::Cpu,
        DType::F32,
        pipelines,
    )
    .unwrap()
}

/// Seconds of one step, the best of `runs` after one to warm up.
fn step_time(m: &HeteroZImage, runs: usize) -> (Tensor, f64) {
    let (x, t, cap, mask) = inputs();
    let mut out = m.forward(&x, &t, &cap, &mask).unwrap();
    let mut best = f64::MAX;
    for _ in 0..runs {
        let start = std::time::Instant::now();
        out = m.forward(&x, &t, &cap, &mask).unwrap();
        best = best.min(start.elapsed().as_secs_f64());
    }
    (out, best)
}

fn host_vec(t: &Tensor) -> Vec<f32> {
    t.to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

#[test]
#[ignore = "needs zimage-trim.safetensors in the working directory and an OpenCL device"]
fn a_denoising_step_on_opencl_matches_the_host() {
    let devices = enumerate_opencl_devices();
    assert!(!devices.is_empty(), "no OpenCL device");
    let pipelines = Arc::new(OpenCLPipelines::new(devices[0]).unwrap());
    let mut rows = Vec::new();
    for layers in [1, LAYERS] {
        let host = build(layers, DeviceKind::Cpu, None);
        let (want, host_s) = step_time(&host, 2);
        drop(host);
        let ocl = build(layers, DeviceKind::OpenCL(0), Some(pipelines.clone()));
        let (got, ocl_s) = step_time(&ocl, 3);
        drop(ocl);
        let (w, g) = (host_vec(&want), host_vec(&got));
        let gap = w
            .iter()
            .zip(&g)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        let scale = w.iter().fold(0f32, |m, a| m.max(a.abs()));
        let mean_gap = w
            .iter()
            .zip(&g)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / w.len() as f64;
        let mean = w.iter().map(|a| a.abs() as f64).sum::<f64>() / w.len() as f64;
        println!(
            "{layers} blocks: host {host_s:.2} s, OpenCL {ocl_s:.2} s, largest difference {gap:.3e} of {scale:.3e}, mean {mean_gap:.3e} of {mean:.3e}"
        );
        assert!(mean_gap <= mean * 1e-2, "{mean_gap} of {mean}");
        rows.push((host_s, ocl_s));
    }
    let blocks = (LAYERS - 1) as f64;
    println!(
        "a block: host {:.3} s, OpenCL {:.3} s",
        (rows[1].0 - rows[0].0) / blocks,
        (rows[1].1 - rows[0].1) / blocks
    );
}

#[test]
#[ignore = "needs zimage-trim.safetensors in the working directory and an OpenCL device"]
fn where_the_opencl_block_departs() {
    let devices = enumerate_opencl_devices();
    let pipelines = Arc::new(OpenCLPipelines::new(devices[0]).unwrap());
    let host = build(1, DeviceKind::Cpu, None);
    let ocl = build(1, DeviceKind::OpenCL(0), Some(pipelines));
    for (name, sx, sc) in [
        ("x only", 1.0, 0.0),
        ("caption only", 0.0, 1.0),
        ("both small", 0.1, 0.1),
        ("x small", 0.1, 1.0),
    ] {
        let (x, t, cap, mask) = inputs_scaled(sx, sc);
        let w = host_vec(&host.forward(&x, &t, &cap, &mask).unwrap());
        let g = host_vec(&ocl.forward(&x, &t, &cap, &mask).unwrap());
        let mean_gap = w
            .iter()
            .zip(&g)
            .map(|(a, b)| (a - b).abs() as f64)
            .sum::<f64>()
            / w.len() as f64;
        let mean = w.iter().map(|a| a.abs() as f64).sum::<f64>() / w.len() as f64;
        println!("{name}: mean difference {mean_gap:.3e} of {mean:.3e}");
    }
}
