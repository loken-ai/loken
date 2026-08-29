//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED.load(std::sync::atomic::Ordering::Relaxed)
}
/// API-compat: allow fp16 accumulation in f16 GEMMs.
pub fn set_gemm_reduced_precision_f16(b: bool) {
    MM_F16_REDUCED.store(b, std::sync::atomic::Ordering::Relaxed)
}
pub fn gemm_reduced_precision_f16() -> bool {
    MM_F16_REDUCED.load(std::sync::atomic::Ordering::Relaxed)
}
/// API-compat: allow bf16 (FAST_16BF) reductions in bf16 GEMMs.
pub fn set_gemm_reduced_precision_bf16(b: bool) {
    MM_BF16_REDUCED.store(b, std::sync::atomic::Ordering::Relaxed)
}
pub fn gemm_reduced_precision_bf16() -> bool {
    MM_BF16_REDUCED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Batched row-major matmul via cuBLAS: `[b, m, k] x [b, k, n] -> [b, m, n]`.
/// Row-major through column-major cuBLAS by computing Cᵀ = Bᵀ.Aᵀ (operand
/// swap), the standard trick - the result lands row-major with no transposes.
pub fn matmul_f32(
    dev: &CudaDevice,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<CudaSlice<f32>> {
    matmul_f32_inner(dev, a, b, batch, m, k, n, gemm_reduced_precision_f32())
}

/// f32 matmul with TF32 tensor-core math (~1e-3 rel error) - for conv/image
/// workloads where pixel-level precision tolerates it. Exact paths keep
/// [`matmul_f32`].
pub fn matmul_f32_tf32(
    dev: &CudaDevice,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<CudaSlice<f32>> {
    matmul_f32_inner(dev, a, b, batch, m, k, n, true)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn matmul_f32_inner(
    dev: &CudaDevice,
    a: &CudaSlice<f32>,
    b: &CudaSlice<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    tf32: bool,
) -> Result<CudaSlice<f32>> {
    use cudarc::cublas::sys;
    use cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let stream = dev.stream();
    // beta = 0 - the gemm overwrites every element, no zero-fill needed.
    let mut out = with_oom_retry(dev, "matmul", || unsafe {
        stream.alloc::<f32>(batch * m * n)
    })?;
    // Same compute-type selection as the reference gemm_strided_batched_f32.
    let compute_type = if tf32 {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
    } else {
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
    };
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let blas = dev.blas()?;
    {
        let (a_ptr, _ga) = a.device_ptr(stream);
        let (b_ptr, _gb) = b.device_ptr(stream);
        let (c_ptr, _gc) = out.device_ptr_mut(stream);
        unsafe {
            cudarc::cublas::result::gemm_strided_batched_ex(
                *blas.handle(),
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const _,
                b_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                (k * n) as i64,
                a_ptr as *const _,
                sys::cudaDataType_t::CUDA_R_32F,
                k as i32,
                (m * k) as i64,
                &beta as *const f32 as *const _,
                c_ptr as *mut _,
                sys::cudaDataType_t::CUDA_R_32F,
                n as i32,
                (m * n) as i64,
                batch as i32,
                compute_type,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .map_err(|e| cublas_err("gemm", e))?;
    }
    Ok(out)
}

/// Batched row-major `A[b, m, k] @ B[b, n, k]^T -> [b, m, n]` with B given
/// UNtransposed: the tensor-op facade computes `x.matmul(&w.t()?)`
/// without materializing the transpose - cuBLAS gets the original buffer
/// with `transa=T`. Mirroring that exact call (same operand pointers, same
/// trans flags, same compute type) keeps the flipped substrate BIT-EXACT
/// with the facade on the attention `q @ k^T` matmuls, where the previous
/// materialize-then-OP_N route picked a different cuBLAS kernel (different
/// reduction order, ~1e-6 rel - amplified across 36 layers into greedy-token
/// flips).
macro_rules! matmul_nt_impl {
    ($name:ident, $t:ty, $cuda_t:ident, $reduced:ident, $rt:expr) => {
        pub fn $name(
            dev: &CudaDevice,
            a: &CudaSlice<$t>,
            b: &CudaSlice<$t>,
            batch: usize,
            m: usize,
            k: usize,
            n: usize,
        ) -> Result<CudaSlice<$t>> {
            use cudarc::cublas::sys;
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let stream = dev.stream();
            let mut out = with_oom_retry(dev, "matmul_nt", || unsafe {
                stream.alloc::<$t>(batch * m * n)
            })?;
            let (compute_type, alpha_ptr, beta_ptr, _hold): (
                _,
                *const std::ffi::c_void,
                *const std::ffi::c_void,
                Box<dyn std::any::Any>,
            ) = $rt($reduced());
            let blas = dev.blas()?;
            {
                let (a_ptr, _ga) = a.device_ptr(stream);
                let (b_ptr, _gb) = b.device_ptr(stream);
                let (c_ptr, _gc) = out.device_ptr_mut(stream);
                unsafe {
                    cudarc::cublas::result::gemm_strided_batched_ex(
                        *blas.handle(),
                        sys::cublasOperation_t::CUBLAS_OP_T,
                        sys::cublasOperation_t::CUBLAS_OP_N,
                        n as i32,
                        m as i32,
                        k as i32,
                        alpha_ptr,
                        b_ptr as *const _,
                        sys::cudaDataType_t::$cuda_t,
                        k as i32,
                        (n * k) as i64,
                        a_ptr as *const _,
                        sys::cudaDataType_t::$cuda_t,
                        k as i32,
                        (m * k) as i64,
                        beta_ptr,
                        c_ptr as *mut _,
                        sys::cudaDataType_t::$cuda_t,
                        n as i32,
                        (m * n) as i64,
                        batch as i32,
                        compute_type,
                        sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    )
                }
                .map_err(|e| cublas_err("gemm_nt", e))?;
            }
            Ok(out)
        }
    };
}

matmul_nt_impl!(
    matmul_nt_f32,
    f32,
    CUDA_R_32F,
    gemm_reduced_precision_f32,
    |red: bool| {
        use cudarc::cublas::sys;
        let alpha = Box::new((1.0f32, 0.0f32));
        let ct = if red {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        } else {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        };
        let (ap, bp) = (
            &alpha.0 as *const f32 as *const std::ffi::c_void,
            &alpha.1 as *const f32 as *const std::ffi::c_void,
        );
        (ct, ap, bp, alpha as Box<dyn std::any::Any>)
    }
);
matmul_nt_impl!(
    matmul_nt_bf16,
    half::bf16,
    CUDA_R_16BF,
    gemm_reduced_precision_bf16,
    |red: bool| {
        use cudarc::cublas::sys;
        let alpha = Box::new((1.0f32, 0.0f32));
        let ct = if red {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
        } else {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        };
        let (ap, bp) = (
            &alpha.0 as *const f32 as *const std::ffi::c_void,
            &alpha.1 as *const f32 as *const std::ffi::c_void,
        );
        (ct, ap, bp, alpha as Box<dyn std::any::Any>)
    }
);
matmul_nt_impl!(
    matmul_nt_f16,
    half::f16,
    CUDA_R_16F,
    gemm_reduced_precision_f16,
    |red: bool| {
        use cudarc::cublas::sys;
        if red {
            let alpha = Box::new((half::f16::ONE, half::f16::ZERO));
            let (ap, bp) = (
                &alpha.0 as *const half::f16 as *const std::ffi::c_void,
                &alpha.1 as *const half::f16 as *const std::ffi::c_void,
            );
            (
                sys::cublasComputeType_t::CUBLAS_COMPUTE_16F,
                ap,
                bp,
                alpha as Box<dyn std::any::Any>,
            )
        } else {
            let alpha = Box::new((1.0f32, 0.0f32));
            let (ap, bp) = (
                &alpha.0 as *const f32 as *const std::ffi::c_void,
                &alpha.1 as *const f32 as *const std::ffi::c_void,
            );
            (
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                ap,
                bp,
                alpha as Box<dyn std::any::Any>,
            )
        }
    }
);

pub(super) fn row_block(cols: usize) -> u32 {
    256u32.min(cols as u32).max(1).next_power_of_two()
}

/// `fused_rmsnorm_f32` on a [rows, cols] device buffer.
pub fn rms_norm_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<CudaSlice<f32>> {
    let func = dev.fused_fn("fused_rmsnorm_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "rms_norm", || unsafe {
        stream.alloc::<f32>(rows * cols)
    })?;
    let block = row_block(cols);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    let cols_i32 = cols as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(w);
    b.arg(&out);
    b.arg(&eps);
    b.arg(&cols_i32);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("rms_norm launch: {e}")))?;
    Ok(out)
}

/// `fused_softmax_lastdim_bf16`: BF16 in, BF16 out, F32 arithmetic.
///
/// Bit-identical to `to_dtype(F32) -> softmax_lastdim_f32 -> to_dtype(BF16)`, which is
/// what the attention path did: widening bf16 is exact, the reductions and the exp are
/// the same f32 operations in the same order, and the single rounding at the end is the
/// one the final cast used to do. What it saves is the traffic - the score tile is the
/// largest buffer in a DiT forward and this drops two full passes over it and halves
/// the width of the rest.
pub fn softmax_lastdim_bf16(
    dev: &CudaDevice,
    x: &CudaSlice<half::bf16>,
    rows: usize,
    cols: usize,
) -> Result<CudaSlice<half::bf16>> {
    let func = dev.fused_fn("fused_softmax_lastdim_bf16")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "softmax_bf16", || unsafe {
        stream.alloc::<half::bf16>(rows * cols)
    })?;
    let block = row_block(cols);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    let cols_i32 = cols as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&cols_i32);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("softmax bf16 launch: {e}")))?;
    Ok(out)
}

/// `fused_softmax_lastdim_f32` on a [rows, cols] device buffer.
pub fn softmax_lastdim_f32(
    dev: &CudaDevice,
    x: &CudaSlice<f32>,
    rows: usize,
    cols: usize,
) -> Result<CudaSlice<f32>> {
    let func = dev.fused_fn("fused_softmax_lastdim_f32")?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "softmax", || unsafe {
        stream.alloc::<f32>(rows * cols)
    })?;
    let block = row_block(cols);
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: block * 4,
    };
    let cols_i32 = cols as i32;
    let mut b = stream.launch_builder(&func);
    b.arg(x);
    b.arg(&out);
    b.arg(&cols_i32);
    unsafe { b.launch(cfg) }.map_err(|e| Error(format!("softmax launch: {e}")))?;
    Ok(out)
}

// --- Quantized GEMV: the mmvq kernels on native storage ----
