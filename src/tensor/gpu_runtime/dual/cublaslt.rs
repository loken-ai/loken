//! Matrix products with epilogues under the cuBLASLt names: only the presence of a runtime's
//! library is asked for.

pub mod sys {
    /// Whether a runtime's matrix-epilogue library opens: cuBLASLt beside a usable CUDA, or
    /// hipBLASLt.
    ///
    /// # Safety
    /// None: it only opens the libraries.
    pub unsafe fn is_culib_present() -> bool {
        unsafe {
            (super::super::cuda_usable() && cudarc::cublaslt::sys::is_culib_present())
                || crate::tensor::gpu_runtime::hip::cublaslt::sys::is_culib_present()
        }
    }
}
