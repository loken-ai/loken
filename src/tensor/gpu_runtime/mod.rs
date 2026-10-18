//! The GPU runtime the tensor layer and the kernels run on, reached through this one module.
//!
//! With the `cuda` feature it is the published `cudarc` crate, re-exported as it is: the driver,
//! the libraries (cuBLAS, cuBLASLt, NVRTC, NCCL) and their raw bindings. Every GPU path names
//! `crate::tensor::gpu_runtime::...`, so a second runtime with the same surface can stand here
//! without the callers changing.

#[cfg(feature = "cuda")]
pub use cudarc::{cublas, cublaslt, driver, nccl, nvrtc};

#[cfg(not(feature = "cuda"))]
compile_error!("the gpu feature needs a runtime feature: cuda");
