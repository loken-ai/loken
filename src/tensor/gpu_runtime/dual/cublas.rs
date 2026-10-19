//! Dense linear algebra over both runtimes: cuBLAS for an NVIDIA card's handle, hipBLAS for an
//! AMD card's. A raw call goes to the library that created the handle it names.

use super::Handles;

/// The handles hipBLAS created.
static HIP_HANDLES: Handles = Handles::new();

pub mod sys {
    #![allow(non_camel_case_types, non_snake_case, clippy::missing_safety_doc)]

    use std::ffi::{c_int, c_longlong, c_void};

    use super::HIP_HANDLES;
    use crate::tensor::gpu_runtime::dual::driver::sys::same;
    use crate::tensor::gpu_runtime::hip::cublas::sys as hs;
    use cudarc::cublas::sys as cs;

    pub use hs::{
        cublasComputeType_t, cublasContext, cublasGemmAlgo_t, cublasHandle_t, cublasOperation_t,
        cublasStatus_t, cudaDataType, cudaDataType_t,
    };

    /// Whether a runtime's matrix library opens.
    pub unsafe fn is_culib_present() -> bool {
        unsafe {
            (crate::tensor::gpu_runtime::dual::cuda_usable() && cs::is_culib_present())
                || hs::is_culib_present()
        }
    }

    /// A cuBLAS status under its own number.
    pub(crate) fn status(s: cs::cublasStatus_t) -> cublasStatus_t {
        use cublasStatus_t::*;
        match s as u32 {
            0 => CUBLAS_STATUS_SUCCESS,
            1 => CUBLAS_STATUS_NOT_INITIALIZED,
            3 => CUBLAS_STATUS_ALLOC_FAILED,
            7 => CUBLAS_STATUS_INVALID_VALUE,
            8 => CUBLAS_STATUS_ARCH_MISMATCH,
            11 => CUBLAS_STATUS_MAPPING_ERROR,
            13 => CUBLAS_STATUS_EXECUTION_FAILED,
            15 => CUBLAS_STATUS_NOT_SUPPORTED,
            16 => CUBLAS_STATUS_LICENSE_ERROR,
            _ => CUBLAS_STATUS_INTERNAL_ERROR,
        }
    }

    fn is_hip(handle: cublasHandle_t) -> bool {
        HIP_HANDLES.contains(handle)
    }

    fn ch(handle: cublasHandle_t) -> cs::cublasHandle_t {
        handle as cs::cublasHandle_t
    }

    pub unsafe fn cublasSetWorkspace_v2(
        handle: cublasHandle_t,
        workspace: *mut c_void,
        bytes: usize,
    ) -> cublasStatus_t {
        if is_hip(handle) {
            unsafe { hs::cublasSetWorkspace_v2(handle, workspace, bytes) }
        } else {
            status(unsafe { cs::cublasSetWorkspace_v2(ch(handle), workspace, bytes) })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const f32,
        lda: c_int,
        b: *const f32,
        ldb: c_int,
        beta: *const f32,
        c: *mut f32,
        ldc: c_int,
    ) -> cublasStatus_t {
        if is_hip(handle) {
            unsafe {
                hs::cublasSgemm_v2(
                    handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
                )
            }
        } else {
            // SAFETY: the operation enum carries cuBLAS's own numbers.
            status(unsafe {
                cs::cublasSgemm_v2(
                    ch(handle),
                    same(transa),
                    same(transb),
                    m,
                    n,
                    k,
                    alpha,
                    a,
                    lda,
                    b,
                    ldb,
                    beta,
                    c,
                    ldc,
                )
            })
        }
    }

    pub unsafe fn cublasSscal_v2(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f32,
        x: *mut f32,
        incx: c_int,
    ) -> cublasStatus_t {
        if is_hip(handle) {
            unsafe { hs::cublasSscal_v2(handle, n, alpha, x, incx) }
        } else {
            status(unsafe { cs::cublasSscal_v2(ch(handle), n, alpha, x, incx) })
        }
    }

    pub unsafe fn cublasScopy_v2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t {
        if is_hip(handle) {
            unsafe { hs::cublasScopy_v2(handle, n, x, incx, y, incy) }
        } else {
            status(unsafe { cs::cublasScopy_v2(ch(handle), n, x, incx, y, incy) })
        }
    }

    pub unsafe fn cublasSaxpy_v2(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f32,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t {
        if is_hip(handle) {
            unsafe { hs::cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy) }
        } else {
            status(unsafe { cs::cublasSaxpy_v2(ch(handle), n, alpha, x, incx, y, incy) })
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub unsafe fn cublasGemmStridedBatchedEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: cudaDataType,
        lda: c_int,
        stride_a: c_longlong,
        b: *const c_void,
        b_type: cudaDataType,
        ldb: c_int,
        stride_b: c_longlong,
        beta: *const c_void,
        c: *mut c_void,
        c_type: cudaDataType,
        ldc: c_int,
        stride_c: c_longlong,
        batch_count: c_int,
        compute_type: cublasComputeType_t,
        algo: cublasGemmAlgo_t,
    ) -> cublasStatus_t {
        if is_hip(handle) {
            unsafe {
                hs::cublasGemmStridedBatchedEx(
                    handle,
                    transa,
                    transb,
                    m,
                    n,
                    k,
                    alpha,
                    a,
                    a_type,
                    lda,
                    stride_a,
                    b,
                    b_type,
                    ldb,
                    stride_b,
                    beta,
                    c,
                    c_type,
                    ldc,
                    stride_c,
                    batch_count,
                    compute_type,
                    algo,
                )
            }
        } else {
            // SAFETY: every enum here carries cuBLAS's own numbers.
            status(unsafe {
                cs::cublasGemmStridedBatchedEx(
                    ch(handle),
                    same(transa),
                    same(transb),
                    m,
                    n,
                    k,
                    alpha,
                    a,
                    same(a_type),
                    lda,
                    stride_a,
                    b,
                    same(b_type),
                    ldb,
                    stride_b,
                    beta,
                    c,
                    same(c_type),
                    ldc,
                    stride_c,
                    batch_count,
                    same(compute_type),
                    same(algo),
                )
            })
        }
    }
}

pub mod result {
    use std::ffi::{c_int, c_longlong, c_void};

    use super::sys;

    pub use crate::tensor::gpu_runtime::hip::cublas::result::CublasError;

    /// A batch of strided matrix products with element types and arithmetic of the caller's
    /// choice, by the library that created `handle`.
    ///
    /// # Safety
    /// Every pointer must address device memory of the extent the shapes describe.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_strided_batched_ex(
        handle: sys::cublasHandle_t,
        transa: sys::cublasOperation_t,
        transb: sys::cublasOperation_t,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const c_void,
        a: *const c_void,
        a_type: sys::cudaDataType,
        lda: c_int,
        stride_a: c_longlong,
        b: *const c_void,
        b_type: sys::cudaDataType,
        ldb: c_int,
        stride_b: c_longlong,
        beta: *const c_void,
        c: *mut c_void,
        c_type: sys::cudaDataType,
        ldc: c_int,
        stride_c: c_longlong,
        batch_count: c_int,
        compute_type: sys::cublasComputeType_t,
        algo: sys::cublasGemmAlgo_t,
    ) -> Result<(), CublasError> {
        unsafe {
            sys::cublasGemmStridedBatchedEx(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a,
                a_type,
                lda,
                stride_a,
                b,
                b_type,
                ldb,
                stride_b,
                beta,
                c,
                c_type,
                ldc,
                stride_c,
                batch_count,
                compute_type,
                algo,
            )
        }
        .result()
    }
}

use std::sync::Arc;

use super::driver::CudaStream;
use crate::tensor::gpu_runtime::hip::cublas as hb;
use result::CublasError;

/// The handle of the library of the stream's runtime.
#[derive(Debug)]
enum Blas {
    Cuda(cudarc::cublas::CudaBlas),
    Hip(hb::CudaBlas),
}

/// A library handle bound to one stream.
#[derive(Debug)]
pub struct CudaBlas {
    inner: Blas,
    handle: sys::cublasHandle_t,
}

unsafe impl Send for CudaBlas {}
unsafe impl Sync for CudaBlas {}

impl CudaBlas {
    /// A handle whose work runs on `stream`, made by the library of the stream's runtime.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CublasError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let inner = match stream.cuda_stream() {
            Some(s) => Blas::Cuda(
                cudarc::cublas::CudaBlas::new(s.clone())
                    .map_err(|e| CublasError(sys::status(e.0)))?,
            ),
            None => Blas::Hip(hb::CudaBlas::new(
                stream
                    .hip_stream()
                    .expect("a stream of one runtime")
                    .clone(),
            )?),
        };
        let handle = match &inner {
            Blas::Cuda(b) => *b.handle() as sys::cublasHandle_t,
            Blas::Hip(b) => {
                let h = *b.handle();
                HIP_HANDLES.insert(h);
                h
            }
        };
        Ok(Self { inner, handle })
    }

    /// The raw handle.
    pub fn handle(&self) -> &sys::cublasHandle_t {
        &self.handle
    }

    /// Moves the handle's work to `stream`, a stream of the same card.
    ///
    /// # Safety
    /// Work already queued must not depend on the change.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CublasError> {
        match (&mut self.inner, stream.cuda_stream(), stream.hip_stream()) {
            (Blas::Cuda(b), Some(s), _) => {
                unsafe { b.set_stream(s.clone()) }.map_err(|e| CublasError(sys::status(e.0)))
            }
            (Blas::Hip(b), _, Some(s)) => unsafe { b.set_stream(s.clone()) },
            _ => Err(CublasError(
                sys::cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE,
            )),
        }
    }
}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        if matches!(self.inner, Blas::Hip(_)) {
            HIP_HANDLES.remove(self.handle);
        }
    }
}
