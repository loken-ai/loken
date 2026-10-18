//! The GPU runtime the tensor layer and the kernels run on, reached through this one module.
//!
//! With the `cuda` feature it is the published `cudarc` crate, re-exported as it is: the driver,
//! the libraries (cuBLAS, cuBLASLt, NVRTC, NCCL) and their raw bindings. Every GPU path names
//! `crate::tensor::gpu_runtime::...`, so a second runtime with the same surface can stand here
//! without the callers changing.

/// Whether the runtime drives NVIDIA cards: kernels written in NVIDIA's own instructions run
/// only then, whatever a card reports about itself.
pub const IS_NVIDIA: bool = cfg!(feature = "cuda");

#[cfg(feature = "cuda")]
pub use cudarc::{cublas, cublaslt, driver, nccl, nvrtc};

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
mod hip;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub use hip::{cublas, cublaslt, driver, nccl, nvrtc};

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
compile_error!("the gpu feature needs a runtime feature: cuda or rocm");
