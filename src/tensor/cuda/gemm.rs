//! Split out of the parent module; see its header for what this file is part of.
//!
//! `use super::*` keeps the names its items referred to before the split in reach.

use super::*;

pub fn gemm_reduced_precision_f32() -> bool {
    MM_F32_REDUCED.load(std::sync::atomic::Ordering::Relaxed)
        || TF32_SCOPE.with(std::cell::Cell::get)
}

thread_local! {
    static TF32_SCOPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// While alive, the F32 matmuls of this thread run with TF32 tensor-core math: inputs rounded to
/// ten mantissa bits, sums in F32. For products whose operands are already that precise, such as
/// the scores of an attention over an F16 cache, where the global switch would also touch the
/// projections around them.
pub struct Tf32Scope {
    previous: bool,
}

impl Tf32Scope {
    pub fn enter() -> Self {
        Self {
            previous: TF32_SCOPE.with(|s| s.replace(true)),
        }
    }
}

impl Drop for Tf32Scope {
    fn drop(&mut self) {
        TF32_SCOPE.with(|s| s.set(self.previous));
    }
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
    use crate::tensor::gpu_runtime::cublas::sys;
    use crate::tensor::gpu_runtime::cublas::sys::cublasOperation_t::CUBLAS_OP_N;
    use crate::tensor::gpu_runtime::driver::{DevicePtr, DevicePtrMut};
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
            crate::tensor::gpu_runtime::cublas::result::gemm_strided_batched_ex(
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
            use crate::tensor::gpu_runtime::cublas::sys;
            use crate::tensor::gpu_runtime::driver::{DevicePtr, DevicePtrMut};
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
                    crate::tensor::gpu_runtime::cublas::result::gemm_strided_batched_ex(
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

/// `A[b, m, k] @ B[b, n, k]^T` over half-width operands of cuBLAS type `ty`, accumulated and
/// written in F32 into `out`, with cuBLAS held to `compute`.
#[allow(clippy::too_many_arguments)]
fn gemm_nt_half_f32(
    dev: &CudaDevice,
    a: u64,
    b: u64,
    out: u64,
    ty: crate::tensor::gpu_runtime::cublas::sys::cudaDataType_t,
    compute: crate::tensor::gpu_runtime::cublas::sys::cublasComputeType_t,
    [batch, m, k, n]: [usize; 4],
) -> Result<()> {
    use crate::tensor::gpu_runtime::cublas::sys;
    let (alpha, beta) = (1.0f32, 0.0f32);
    let blas = dev.blas()?;
    unsafe {
        crate::tensor::gpu_runtime::cublas::result::gemm_strided_batched_ex(
            *blas.handle(),
            sys::cublasOperation_t::CUBLAS_OP_T,
            sys::cublasOperation_t::CUBLAS_OP_N,
            n as i32,
            m as i32,
            k as i32,
            &alpha as *const f32 as *const std::ffi::c_void,
            b as *const _,
            ty,
            k as i32,
            (n * k) as i64,
            a as *const _,
            ty,
            k as i32,
            (m * k) as i64,
            &beta as *const f32 as *const std::ffi::c_void,
            out as *mut _,
            sys::cudaDataType_t::CUDA_R_32F,
            n as i32,
            (m * n) as i64,
            batch as i32,
            compute,
            sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        )
    }
    .map_err(|e| cublas_err("gemm_nt_half_f32", e))
}

/// Timed runs of each compute type in [`half_tensor_ops_pay`], after one that is not timed;
/// the fastest of them is the one compared.
const HALF_PROBE_RUNS: usize = 3;

/// Whether F16 products of more than one row run faster on this card with cuBLAS free to pick
/// its kernels than held to F32 arithmetic (`CUBLAS_COMPUTE_32F_PEDANTIC`). Measured once per
/// card on the first such product `run` issues, and kept; `run` launches it under the compute
/// type it is given. Inside a graph capture nothing can be timed and the free choice stands.
fn half_tensor_ops_pay(
    dev: &CudaDevice,
    run: impl Fn(crate::tensor::gpu_runtime::cublas::sys::cublasComputeType_t) -> Result<()>,
) -> Result<bool> {
    use crate::tensor::gpu_runtime::cublas::sys::cublasComputeType_t as C;
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static MEASURED: OnceLock<Mutex<HashMap<usize, bool>>> = OnceLock::new();
    let measured = MEASURED.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(&pays) = measured
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&dev.ordinal())
    {
        return Ok(pays);
    }
    if crate::tensor::cuda_ext::capture_active_on(dev.stream()) {
        return Ok(true);
    }
    let fastest = |compute: C| -> Result<std::time::Duration> {
        run(compute)?;
        dev.synchronize()?;
        let mut best = std::time::Duration::MAX;
        for _ in 0..HALF_PROBE_RUNS {
            let t = std::time::Instant::now();
            run(compute)?;
            dev.synchronize()?;
            best = best.min(t.elapsed());
        }
        Ok(best)
    };
    let pays = fastest(C::CUBLAS_COMPUTE_32F)? <= fastest(C::CUBLAS_COMPUTE_32F_PEDANTIC)?;
    measured
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(dev.ordinal(), pays);
    Ok(pays)
}

/// [`matmul_nt_bf16`] summed and written in F32: BF16 operands, F32 accumulation and output.
pub fn matmul_nt_bf16_f32(
    dev: &CudaDevice,
    a: &CudaSlice<half::bf16>,
    b: &CudaSlice<half::bf16>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<CudaSlice<f32>> {
    use crate::tensor::gpu_runtime::cublas::sys;
    use crate::tensor::gpu_runtime::driver::{DevicePtr, DevicePtrMut};
    let stream = dev.stream();
    let mut out = with_oom_retry(dev, "matmul_nt_bf16_f32", || unsafe {
        stream.alloc::<f32>(batch * m * n)
    })?;
    {
        let (a_ptr, _ga) = a.device_ptr(stream);
        let (b_ptr, _gb) = b.device_ptr(stream);
        let (c_ptr, _gc) = out.device_ptr_mut(stream);
        gemm_nt_half_f32(
            dev,
            a_ptr as u64,
            b_ptr as u64,
            c_ptr as u64,
            sys::cudaDataType_t::CUDA_R_16BF,
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            [batch, m, k, n],
        )?;
    }
    Ok(out)
}

/// [`matmul_nt_f16`] summed and written in F32: F16 operands, F32 accumulation and output.
///
/// A product of more than one row runs under the compute type [`half_tensor_ops_pay`] measured
/// faster on this card; a single row keeps cuBLAS's own choice.
pub fn matmul_nt_f16_f32(
    dev: &CudaDevice,
    a: &CudaSlice<half::f16>,
    b: &CudaSlice<half::f16>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<CudaSlice<f32>> {
    use crate::tensor::gpu_runtime::cublas::sys;
    use crate::tensor::gpu_runtime::driver::{DevicePtr, DevicePtrMut};
    let stream = dev.stream();
    let mut out = with_oom_retry(dev, "matmul_nt_f16_f32", || unsafe {
        stream.alloc::<f32>(batch * m * n)
    })?;
    {
        let (a_ptr, _ga) = a.device_ptr(stream);
        let (b_ptr, _gb) = b.device_ptr(stream);
        let (c_ptr, _gc) = out.device_ptr_mut(stream);
        let run = |compute| {
            gemm_nt_half_f32(
                dev,
                a_ptr as u64,
                b_ptr as u64,
                c_ptr as u64,
                sys::cudaDataType_t::CUDA_R_16F,
                compute,
                [batch, m, k, n],
            )
        };
        let compute = if batch * m == 1 || half_tensor_ops_pay(dev, &run)? {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        } else {
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC
        };
        run(compute)?;
    }
    Ok(out)
}

matmul_nt_impl!(
    matmul_nt_f32,
    f32,
    CUDA_R_32F,
    gemm_reduced_precision_f32,
    |red: bool| {
        use crate::tensor::gpu_runtime::cublas::sys;
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
        use crate::tensor::gpu_runtime::cublas::sys;
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
        use crate::tensor::gpu_runtime::cublas::sys;
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

/// Which RMSNorm kernel a launch of this shape takes, and how it is configured.
///
/// Declared once. The decode path launches through `rms_norm_f32` below and the tensor path
/// through the fused buffers; only the second knew the wide kernel existed, so every decode
/// ran the narrow one and read its row from global memory twice.
///
/// Few rows means one block per row leaves the SM mostly idle. The wide variant stages the
/// row into shared memory with the full block and keeps the narrow kernel's exact
/// accumulation order, so the two produce bit-identical output.
pub fn wide_row_block(rows: usize, cols: usize) -> bool {
    rows <= 64 && cols >= 256 && cols * 4 + 1024 <= 48 * 1024
}

/// The launch geometry a row-per-block norm takes, wide or narrow.
pub fn row_norm_cfg(rows: usize, cols: usize, wide: bool) -> LaunchConfig {
    let (block, smem) = if wide {
        (
            (cols as u32).next_power_of_two().clamp(256, 1024),
            (cols as u32) * 4,
        )
    } else {
        let b = row_block(cols);
        (b, b * 4)
    };
    LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: smem,
    }
}

pub fn rms_norm_launch(rows: usize, cols: usize) -> (&'static str, LaunchConfig) {
    let wide = wide_row_block(rows, cols);
    let name = if wide {
        "fused_rmsnorm_wide_f32"
    } else {
        "fused_rmsnorm_f32"
    };
    (name, row_norm_cfg(rows, cols, wide))
}

/// The same choice for the add+norm kernel, which reads its summed row back from global
/// memory in the narrow form.
pub fn add_rms_norm_launch(rows: usize, cols: usize) -> (&'static str, LaunchConfig) {
    let wide = wide_row_block(rows, cols);
    let name = if wide {
        "fused_add_rmsnorm_dual_wide_f32"
    } else {
        "fused_add_rmsnorm_dual_f32"
    };
    (name, row_norm_cfg(rows, cols, wide))
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
    let (kname, cfg) = rms_norm_launch(rows, cols);
    let func = dev.fused_fn(kname)?;
    let stream = dev.stream();
    let out = with_oom_retry(dev, "rms_norm", || unsafe {
        stream.alloc::<f32>(rows * cols)
    })?;
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
