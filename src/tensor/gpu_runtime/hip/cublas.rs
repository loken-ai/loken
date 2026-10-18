//! Dense linear algebra under the cuBLAS names, done by hipBLAS. Enums keep the cuBLAS variant
//! names and are translated to hipBLAS values at the call.

pub mod sys {
    #![allow(non_camel_case_types, non_snake_case, clippy::missing_safety_doc)]

    use std::ffi::{c_int, c_longlong, c_void};

    crate::tensor::gpu_runtime::hip::dynamic_api! {
        /// The entry points of the hipBLAS library.
        pub(crate) struct Hipblas from "hipblas" {
            fn hipblasCreate(h: *mut cublasHandle_t) -> c_int;
            fn hipblasDestroy(h: cublasHandle_t) -> c_int;
            fn hipblasSetStream(h: cublasHandle_t, s: *mut c_void) -> c_int;
            fn hipblasSetWorkspace(h: cublasHandle_t, ws: *mut c_void, bytes: usize) -> c_int;
            fn hipblasSgemm(
                h: cublasHandle_t,
                ta: c_int,
                tb: c_int,
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
            ) -> c_int;
            fn hipblasSscal(h: cublasHandle_t, n: c_int, alpha: *const f32, x: *mut f32, incx: c_int) -> c_int;
            fn hipblasScopy(
                h: cublasHandle_t,
                n: c_int,
                x: *const f32,
                incx: c_int,
                y: *mut f32,
                incy: c_int,
            ) -> c_int;
            fn hipblasSaxpy(
                h: cublasHandle_t,
                n: c_int,
                alpha: *const f32,
                x: *const f32,
                incx: c_int,
                y: *mut f32,
                incy: c_int,
            ) -> c_int;
            fn hipblasGemmStridedBatchedEx(
                h: cublasHandle_t,
                ta: c_int,
                tb: c_int,
                m: c_int,
                n: c_int,
                k: c_int,
                alpha: *const c_void,
                a: *const c_void,
                a_type: c_int,
                lda: c_int,
                stride_a: c_longlong,
                b: *const c_void,
                b_type: c_int,
                ldb: c_int,
                stride_b: c_longlong,
                beta: *const c_void,
                c: *mut c_void,
                c_type: c_int,
                ldc: c_int,
                stride_c: c_longlong,
                batch: c_int,
                compute: c_int,
                algo: c_int,
            ) -> c_int;
        }
    }

    /// Calls a hipBLAS entry point and returns its status as a `cublasStatus_t`.
    macro_rules! hipblas {
        ($f:ident ( $( $a:expr ),* $(,)? )) => {
            match Hipblas::get() {
                None => cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED,
                Some(api) => match api.$f {
                    None => cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED,
                    Some(f) => cublasStatus_t::from_hip(f($( $a ),*)),
                },
            }
        };
    }

    /// Whether the hipBLAS library opens.
    pub unsafe fn is_culib_present() -> bool {
        Hipblas::present()
    }

    #[repr(C)]
    #[derive(Debug, Copy, Clone)]
    pub struct cublasContext {
        _unused: [u8; 0],
    }
    /// A library handle.
    pub type cublasHandle_t = *mut cublasContext;

    /// A library status, under cuBLAS's names and numbers.
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
    pub enum cublasStatus_t {
        CUBLAS_STATUS_SUCCESS = 0,
        CUBLAS_STATUS_NOT_INITIALIZED = 1,
        CUBLAS_STATUS_ALLOC_FAILED = 3,
        CUBLAS_STATUS_INVALID_VALUE = 7,
        CUBLAS_STATUS_ARCH_MISMATCH = 8,
        CUBLAS_STATUS_MAPPING_ERROR = 11,
        CUBLAS_STATUS_EXECUTION_FAILED = 13,
        CUBLAS_STATUS_INTERNAL_ERROR = 14,
        CUBLAS_STATUS_NOT_SUPPORTED = 15,
        CUBLAS_STATUS_LICENSE_ERROR = 16,
    }

    impl cublasStatus_t {
        fn from_hip(s: c_int) -> Self {
            use cublasStatus_t::*;
            match s {
                0 => CUBLAS_STATUS_SUCCESS,
                1 => CUBLAS_STATUS_NOT_INITIALIZED,
                2 => CUBLAS_STATUS_ALLOC_FAILED,
                3 | 9 | 10 => CUBLAS_STATUS_INVALID_VALUE,
                4 => CUBLAS_STATUS_MAPPING_ERROR,
                5 => CUBLAS_STATUS_EXECUTION_FAILED,
                7 => CUBLAS_STATUS_NOT_SUPPORTED,
                8 => CUBLAS_STATUS_ARCH_MISMATCH,
                _ => CUBLAS_STATUS_INTERNAL_ERROR,
            }
        }

        /// `Ok` for success, the status as an error otherwise.
        pub fn result(self) -> Result<(), super::result::CublasError> {
            match self {
                Self::CUBLAS_STATUS_SUCCESS => Ok(()),
                e => Err(super::result::CublasError(e)),
            }
        }
    }

    /// Whether an operand is transposed.
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
    pub enum cublasOperation_t {
        CUBLAS_OP_N = 0,
        CUBLAS_OP_T = 1,
        CUBLAS_OP_C = 2,
    }

    impl cublasOperation_t {
        fn to_hip(self) -> c_int {
            match self {
                Self::CUBLAS_OP_N => 111,
                Self::CUBLAS_OP_T => 112,
                Self::CUBLAS_OP_C => 113,
            }
        }
    }

    /// The arithmetic a matrix product runs in.
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
    pub enum cublasComputeType_t {
        CUBLAS_COMPUTE_16F = 64,
        CUBLAS_COMPUTE_32F = 68,
        CUBLAS_COMPUTE_32F_FAST_16F = 74,
        CUBLAS_COMPUTE_32F_FAST_16BF = 75,
        CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
        CUBLAS_COMPUTE_64F = 70,
        CUBLAS_COMPUTE_32I = 72,
    }

    impl cublasComputeType_t {
        fn to_hip(self) -> c_int {
            match self {
                Self::CUBLAS_COMPUTE_16F => 0,
                // A reduced-precision f32 product is a permission, not a demand: the full f32
                // product every AMD card runs answers it.
                Self::CUBLAS_COMPUTE_32F
                | Self::CUBLAS_COMPUTE_32F_FAST_16F
                | Self::CUBLAS_COMPUTE_32F_FAST_16BF
                | Self::CUBLAS_COMPUTE_32F_FAST_TF32 => 2,
                Self::CUBLAS_COMPUTE_64F => 7,
                Self::CUBLAS_COMPUTE_32I => 9,
            }
        }
    }

    /// The algorithm of a matrix product; every choice runs the library's default.
    #[repr(i32)]
    #[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
    pub enum cublasGemmAlgo_t {
        CUBLAS_GEMM_DEFAULT = -1,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP = 99,
    }

    impl cublasGemmAlgo_t {
        fn to_hip(self) -> c_int {
            160
        }
    }

    /// An element type.
    #[repr(u32)]
    #[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
    pub enum cudaDataType_t {
        CUDA_R_32F = 0,
        CUDA_R_64F = 1,
        CUDA_R_16F = 2,
        CUDA_R_8I = 3,
        CUDA_R_8U = 8,
        CUDA_R_32I = 10,
        CUDA_R_16BF = 14,
        CUDA_R_8F_E4M3 = 28,
        CUDA_R_8F_E5M2 = 29,
    }
    pub use self::cudaDataType_t as cudaDataType;

    impl cudaDataType_t {
        /// HIP numbers element types as CUDA does.
        fn to_hip(self) -> c_int {
            self as u32 as c_int
        }
    }

    pub unsafe fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t {
        unsafe { hipblas!(hipblasCreate(handle)) }
    }

    pub unsafe fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t {
        unsafe { hipblas!(hipblasDestroy(handle)) }
    }

    pub unsafe fn cublasSetStream_v2(
        handle: cublasHandle_t,
        stream: *mut c_void,
    ) -> cublasStatus_t {
        unsafe { hipblas!(hipblasSetStream(handle, stream)) }
    }

    pub unsafe fn cublasSetWorkspace_v2(
        handle: cublasHandle_t,
        workspace: *mut c_void,
        bytes: usize,
    ) -> cublasStatus_t {
        unsafe { hipblas!(hipblasSetWorkspace(handle, workspace, bytes)) }
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
        unsafe {
            hipblas!(hipblasSgemm(
                handle,
                transa.to_hip(),
                transb.to_hip(),
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
                ldc
            ))
        }
    }

    pub unsafe fn cublasSscal_v2(
        handle: cublasHandle_t,
        n: c_int,
        alpha: *const f32,
        x: *mut f32,
        incx: c_int,
    ) -> cublasStatus_t {
        unsafe { hipblas!(hipblasSscal(handle, n, alpha, x, incx)) }
    }

    pub unsafe fn cublasScopy_v2(
        handle: cublasHandle_t,
        n: c_int,
        x: *const f32,
        incx: c_int,
        y: *mut f32,
        incy: c_int,
    ) -> cublasStatus_t {
        unsafe { hipblas!(hipblasScopy(handle, n, x, incx, y, incy)) }
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
        unsafe { hipblas!(hipblasSaxpy(handle, n, alpha, x, incx, y, incy)) }
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
        unsafe {
            hipblas!(hipblasGemmStridedBatchedEx(
                handle,
                transa.to_hip(),
                transb.to_hip(),
                m,
                n,
                k,
                alpha,
                a,
                a_type.to_hip(),
                lda,
                stride_a,
                b,
                b_type.to_hip(),
                ldb,
                stride_b,
                beta,
                c,
                c_type.to_hip(),
                ldc,
                stride_c,
                batch_count,
                compute_type.to_hip(),
                algo.to_hip()
            ))
        }
    }
}

pub mod result {
    use std::ffi::{c_int, c_longlong, c_void};

    use super::sys;

    /// A failed library call: its status.
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct CublasError(pub sys::cublasStatus_t);

    impl std::fmt::Display for CublasError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{self:?}")
        }
    }

    impl std::error::Error for CublasError {}

    /// A new library handle.
    pub fn create_handle() -> Result<sys::cublasHandle_t, CublasError> {
        let mut handle = std::ptr::null_mut();
        unsafe { sys::cublasCreate_v2(&mut handle) }.result()?;
        Ok(handle)
    }

    /// Destroys a library handle.
    ///
    /// # Safety
    /// `handle` must come from [create_handle] and not be destroyed twice.
    pub unsafe fn destroy_handle(handle: sys::cublasHandle_t) -> Result<(), CublasError> {
        unsafe { sys::cublasDestroy_v2(handle) }.result()
    }

    /// Queues the handle's work on `stream`.
    ///
    /// # Safety
    /// `handle` and `stream` must be valid.
    pub unsafe fn set_stream(
        handle: sys::cublasHandle_t,
        stream: *mut c_void,
    ) -> Result<(), CublasError> {
        unsafe { sys::cublasSetStream_v2(handle, stream) }.result()
    }

    /// A batch of strided matrix products with element types and arithmetic of the caller's
    /// choice.
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

use crate::tensor::gpu_runtime::hip::driver::CudaStream;
use result::CublasError;

/// A library handle bound to one stream.
#[derive(Debug)]
pub struct CudaBlas {
    handle: sys::cublasHandle_t,
    stream: Arc<CudaStream>,
}

unsafe impl Send for CudaBlas {}
unsafe impl Sync for CudaBlas {}

impl CudaBlas {
    /// A handle whose work runs on `stream`.
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CublasError> {
        let ctx = stream.context();
        ctx.record_err(ctx.bind_to_thread());
        let handle = result::create_handle()?;
        let blas = Self { handle, stream };
        unsafe { result::set_stream(handle, blas.stream.cu_stream() as *mut _) }?;
        Ok(blas)
    }

    /// The raw handle.
    pub fn handle(&self) -> &sys::cublasHandle_t {
        &self.handle
    }

    /// Moves the handle's work to `stream`.
    ///
    /// # Safety
    /// Work already queued must not depend on the change.
    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CublasError> {
        self.stream = stream;
        unsafe { result::set_stream(self.handle, self.stream.cu_stream() as *mut _) }
    }
}

impl Drop for CudaBlas {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            let _ = unsafe { result::destroy_handle(handle) };
        }
    }
}
