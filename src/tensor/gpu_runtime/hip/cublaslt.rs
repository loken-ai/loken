//! Matrix products with epilogues under the cuBLASLt names, done by hipBLASLt. Only its presence
//! is asked for.

pub mod sys {
    crate::tensor::gpu_runtime::hip::dynamic_api! {
        /// The hipBLASLt library; no entry point is called through it.
        pub(crate) struct Hipblaslt from "hipblaslt" {}
    }

    /// Whether the hipBLASLt library opens.
    ///
    /// # Safety
    /// None: it only opens the library.
    pub unsafe fn is_culib_present() -> bool {
        Hipblaslt::present()
    }
}
