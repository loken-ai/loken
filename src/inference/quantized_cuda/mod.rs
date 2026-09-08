//! Quantized-CUDA kernel launchers hosted in loken.
//!
//! These launchers operate on raw `CudaSlice`/`CudaView` handles obtained from
//! substrate tensors through the public `storage_and_layout()` + `as_cuda_slice()`
//! API - no private access is needed. The CUDA kernels live in
//! `cuda/quantized.cu` (a near-verbatim copy of the upstream source) and are
//! compiled once at runtime via NVRTC (same pattern as [`crate::inference::kernel::fused`])
//! into the `"loken_quantized"` module, with `cuda/nvrtc_compat.h`
//! prepended for the stdint/INFINITY definitions NVRTC lacks.
//!
//! This module is the first step of moving the bespoke quantized-attention /
//! KV-cache / MoE kernel launchers OUT of the vendored fork (where they only
//! lived to reach private storage) and INTO loken, shrinking the fork to
//! the minimal MXFP4 dtype + accessors.

use crate::tensor::quantized::GgmlDType;
use anyhow::{anyhow, Result};
// The CUDA types come through the substrate's own boundary rather than from the
// driver crate directly, so a launcher here does not depend on which backing the
// substrate was built with.
use crate::tensor::cuda_ext::{CudaStorage, RawCudaDevice as CudaDevice};
use cudarc::driver::{CudaSlice, CudaView, DevicePtr, LaunchConfig, PushKernelArg};
use half::f16;

// The IMMA (tensor-core) GEMV kernel, compiled by NVCC into libloken_imma.a
// (see build.rs). NVRTC can't JIT `mma.sync`, so this stays an FFI launcher.
extern "C" {
    fn q4k_mmvq_imma(
        vx: *const core::ffi::c_void,
        vy: *const core::ffi::c_void,
        dst: *mut core::ffi::c_void,
        ncols_x: i32,
        nrows_x: i32,
        stream: i64,
    );
}

/// Local equivalent of the reference internal `builder_arg!` macro: push each scalar
/// kernel argument, binding it to a temporary so the pointer stays valid for
/// the launch.
macro_rules! barg {
    ($b:ident, $($arg:expr),* $(,)?) => {
        $( let __a = $arg; $b.arg(&__a); )*
    };
}

const WARP_SIZE: usize = 32;

use std::sync::OnceLock;

/// The quantised attention, KV-cache and dequantisation kernels, compiled once by NVRTC into
/// the "loken_quantized" module. Every launcher below loads its kernel from there.
/// The quantised unit, one file per family. NVRTC compiles one translation unit, so these are
/// concatenated in this order - `helpers` first because the rest assume it.
const QUANTIZED_HELPERS_CU: &str = include_str!("../cuda/quantized/helpers.cu");
const QUANTIZED_KV_CU: &str = include_str!("../cuda/quantized/kv_quantize.cu");
const QUANTIZED_ATTENTION_CU: &str = include_str!("../cuda/quantized/attention.cu");
const QUANTIZED_FLASH_CU: &str = include_str!("../cuda/quantized/flash_splitk.cu");
const QUANTIZED_SAMPLING_CU: &str = include_str!("../cuda/quantized/sampling.cu");
/// NVRTC compatibility shim (stdint types + INFINITY/NAN), prepended to the
/// kernel source so the `.cu` stays free of compiler-workaround cruft.
const NVRTC_COMPAT_H: &str = include_str!("../cuda/nvrtc_compat.h");
static QUANTIZED_PTX: OnceLock<
    std::sync::Mutex<std::collections::HashMap<&'static str, &'static str>>,
> = OnceLock::new();

/// The block layouts and dot products the mat-vec kernels walk. Shared with the MoE and MMQ
/// kernels, which `#include` it - this path is compiled by NVRTC from a string, so it is
/// concatenated instead. It carried a SECOND copy of all of this until then: two versions of
/// the same arithmetic that had to agree, and only one of them got rewritten.
const GGUF_BLOCKS_CUH: &str = include_str!("../../../cuda/moe/gguf.cuh");
/// The mat-vec kernels themselves: the warp-per-row reduction and the entry points.
const MMVQ_CU: &str = include_str!("../cuda/mmvq_gguf.cu");
static MMVQ_PTX: OnceLock<std::sync::Mutex<std::collections::HashMap<&'static str, &'static str>>> =
    OnceLock::new();

// NVRTC-compile the relocated mmvq_gguf.cu (cached). Returns the PTX.
// pub(crate): the native tensor substrate loads the same PTX (tensor).

/// The `--gpu-architecture` NVRTC compiles for: the LOWEST compute capability among the
/// cards actually present, so one PTX serves a mixed machine and JITs upward.
///
/// Leaving it unset hands the driver generic PTX to JIT, and this repo has already paid for
/// that once: build.rs documents IMMA kernels JIT-ed from generic PTX producing silent
/// garbage on one architecture while testing clean on another. The static-lib kernels got an
/// explicit arch list that day; the NVRTC sites did not.
/// The architecture string for ONE device, so each card loads PTX compiled for itself.
/// A single lowest-common PTX would hand the faster card of a mixed machine a JIT from
/// below - the exact pattern build.rs documents as producing silent garbage.
pub(crate) fn nvrtc_arch_of_ordinal(ordinal: usize) -> Option<&'static str> {
    static BY_ORD: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<usize, &'static str>>,
    > = std::sync::OnceLock::new();
    let map = BY_ORD.get_or_init(Default::default);
    let mut g = map.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(a) = g.get(&ordinal) {
        return Some(a);
    }
    use cudarc::driver::result as cu;
    use cudarc::driver::sys::CUdevice_attribute as A;
    let d = cu::device::get(ordinal as i32).ok()?;
    let maj =
        unsafe { cu::device::get_attribute(d, A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR) }
            .ok()?;
    let min =
        unsafe { cu::device::get_attribute(d, A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR) }
            .ok()?;
    let a: &'static str = Box::leak(format!("compute_{maj}{min}").into_boxed_str());
    g.insert(ordinal, a);
    Some(a)
}

pub(crate) fn nvrtc_arch_of(dev: &CudaDevice) -> Option<&'static str> {
    nvrtc_arch_of_ordinal(dev.ordinal())
}

pub(crate) fn nvrtc_arch() -> Option<&'static str> {
    static ARCH: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *ARCH.get_or_init(|| {
        use cudarc::driver::result as cu;
        use cudarc::driver::sys::CUdevice_attribute as A;
        cu::init().ok()?;
        let n = cu::device::get_count().ok()?;
        let mut min: Option<(i32, i32)> = None;
        for i in 0..n {
            let d = cu::device::get(i).ok()?;
            let maj = unsafe {
                cu::device::get_attribute(d, A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            }
            .ok()?;
            let min_ = unsafe {
                cu::device::get_attribute(d, A::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            }
            .ok()?;
            min = Some(match min {
                Some(m) if m <= (maj, min_) => m,
                _ => (maj, min_),
            });
        }
        let (maj, min_) = min?;
        let s: &'static str = Box::leak(format!("compute_{maj}{min_}").into_boxed_str());
        tracing::info!("NVRTC kernels target {s} (lowest capability present)");
        Some(s)
    })
}

pub(crate) fn get_mmvq_ptx(dev: &CudaDevice) -> Result<&'static str> {
    get_mmvq_ptx_for_ordinal(dev.ordinal())
}

pub(crate) fn get_mmvq_ptx_for_ordinal(ordinal: usize) -> Result<&'static str> {
    // One PTX per architecture present: each card loads code compiled for itself, and a
    // mixed machine never JITs one card's PTX onto another.
    let arch = nvrtc_arch_of_ordinal(ordinal);
    let map = MMVQ_PTX.get_or_init(Default::default);
    let mut g = map.lock().unwrap_or_else(|e| e.into_inner());
    let key = arch.unwrap_or("default");
    if let Some(p) = g.get(key) {
        if p.is_empty() {
            return Err(anyhow!("NVRTC compile previously failed for {key}"));
        }
        return Ok(p);
    }
    let compiled: String = {
        let opts = cudarc::nvrtc::safe::CompileOptions {
            include_paths: cuda_include_paths(),
            arch,
            ..Default::default()
        };
        let src = format!("{NVRTC_COMPAT_H}\n{GGUF_BLOCKS_CUH}\n{MMVQ_CU}");
        match cudarc::nvrtc::safe::compile_ptx_with_opts(src, opts) {
            Ok(ptx) => {
                let s = ptx.to_src();
                tracing::info!(
                    "quantized_cuda: NVRTC-compiled mmvq_gguf.cu OK ({} KB PTX)",
                    s.len() / 1024
                );
                s
            }
            Err(e) => {
                tracing::error!("quantized_cuda: mmvq_gguf.cu NVRTC compile FAILED: {e}");
                String::new()
            }
        }
    };
    let leaked: &'static str = Box::leak(compiled.into_boxed_str());
    g.insert(key, leaked);
    let ptx = leaked;
    if ptx.is_empty() {
        return Err(anyhow!("mmvq_gguf.cu failed to NVRTC-compile"));
    }
    Ok(ptx)
}

mod ptx;
pub use ptx::*;
mod awq;
pub use awq::*;
mod attn;
pub use attn::*;
mod quantize;
pub use quantize::*;

#[cfg(all(test, feature = "cuda"))]
mod ncu_harness;

/// What the runtime NVRTC compile of the mat-vec kernels costs, and how big it is.
///
/// The file instantiates one kernel per (format, destination dtype, batch size) - ten times
/// three times eight for the plain set alone. Every one of them is compiled at the first
/// mat-vec of a process, before the first token comes out. Measured rather than assumed.
#[cfg(test)]
mod nvrtc_cost {
    #[test]
    #[ignore = "measurement: needs a CUDA device"]
    fn what_compiling_the_mat_vec_kernels_costs() {
        let Ok(dev) = crate::tensor::cuda::CudaDevice::new(0) else {
            eprintln!("no CUDA device");
            return;
        };
        let start = std::time::Instant::now();
        let ptx = super::get_mmvq_ptx_for_ordinal(dev.ordinal()).expect("nvrtc");
        let first = start.elapsed();
        let start = std::time::Instant::now();
        let _ = super::get_mmvq_ptx_for_ordinal(dev.ordinal()).expect("nvrtc");
        let cached = start.elapsed();
        eprintln!(
            "NVRTC mat-vec kernels: {:.2} s cold, {:.1} us cached, {} KB of PTX",
            first.as_secs_f64(),
            cached.as_secs_f64() * 1e6,
            ptx.len() / 1024
        );
    }
}
