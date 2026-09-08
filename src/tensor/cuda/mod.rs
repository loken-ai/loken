//! Native CUDA storage over the vendored `cudarc`.
//!
//! THE boundary of the whole substrate removal: loken's existing `.cu` kernels
//! launch through `cudarc` `CudaStream`/`CudaSlice`/`LaunchConfig` - by holding
//! device memory in the SAME vendored `cudarc` types, native tensors can feed
//! those launch sites unchanged. This module owns the context/stream handle and
//! typed device buffers; the proof test compiles a kernel via NVRTC and runs it
//! on native-owned memory with no other substrate involved.

use super::{CpuStorage, DType, Error, Result};
use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig,
    PushKernelArg,
};
use std::sync::{Arc, OnceLock};

/// Format a device-allocation `DriverError` into a substrate `Error`, prefixing a
/// stable `[oom]` marker when the driver reports out-of-memory so callers can
/// branch on memory pressure via `Error::is_oom()` (graceful degradation) without
/// fragile reliance on the driver's Display text. `ctx` names the failing op.
pub(crate) fn alloc_err(ctx: &str, e: cudarc::driver::DriverError) -> Error {
    if e.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY {
        Error(format!("[oom] cuda {ctx}: {e}"))
    } else {
        Error(format!("cuda {ctx}: {e}"))
    }
}

mod device;
pub use device::*;
mod storage;
pub(crate) use storage::*;
mod elementwise;
pub use elementwise::*;
mod index;
pub use index::*;
mod dtype;
pub use dtype::*;
mod conv;
pub use conv::*;
mod gemm;
pub use gemm::*;
mod quant;
pub use quant::*;
mod cast;
pub use cast::*;

extern "C" {
    fn loken_moe_gemm_gguf_gate_up_silu_mul(
        inputs: *const f32,
        gate_weights: *const core::ffi::c_void,
        up_weights: *const core::ffi::c_void,
        sorted_token_ids: *const i32,
        expert_ids: *const i32,
        outputs: *mut f32,
        num_experts: i32,
        topk: i32,
        size_m: i32,
        size_n: i32,
        size_k: i32,
        quant_type: i32,
        stream: i64,
    );
}
