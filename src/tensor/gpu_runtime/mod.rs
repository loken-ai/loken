//! The GPU runtime the tensor layer and the kernels run on, reached through this one module.
//!
//! With the `cuda` feature alone it is the published `cudarc` crate, re-exported as it is: the
//! driver, the libraries (cuBLAS, cuBLASLt, NVRTC, NCCL) and their raw bindings. With `rocm`
//! alone it is the HIP implementation of the same surface. With both, every item dispatches on
//! the runtime that owns it: the cards are numbered NVIDIA first, then AMD, and each context
//! knows which runtime opened it. Every GPU path names `crate::tensor::gpu_runtime::...`, so the
//! callers do not change between the three builds.

#[cfg(all(feature = "cuda", not(feature = "rocm")))]
pub use cudarc::{cublas, cublaslt, driver, nccl, nvrtc};

// Beside CUDA, only the part of the HIP surface the dispatch calls is used.
#[cfg(feature = "rocm")]
#[cfg_attr(feature = "cuda", allow(dead_code, unused_imports))]
mod hip;
#[cfg(all(feature = "rocm", not(feature = "cuda")))]
pub use hip::{cublas, cublaslt, driver, nccl, nvrtc};

#[cfg(all(feature = "cuda", feature = "rocm"))]
mod dual;
#[cfg(all(feature = "cuda", feature = "rocm"))]
pub use dual::{cublas, cublaslt, driver, nccl, nvrtc};

#[cfg(not(any(feature = "cuda", feature = "rocm")))]
compile_error!("the gpu feature needs a runtime feature: cuda or rocm");

/// Whether the card of `ordinal` is driven by NVIDIA's runtime: kernels written in NVIDIA's own
/// instructions run only there, whatever a card reports about itself.
pub fn is_nvidia_ordinal(ordinal: usize) -> bool {
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    return dual::topology().is_cuda(ordinal);
    #[cfg(not(all(feature = "cuda", feature = "rocm")))]
    {
        let _ = ordinal;
        cfg!(feature = "cuda")
    }
}

/// Whether every card the process lists is an NVIDIA card. True in a CUDA build, false in a
/// ROCm build, and in a build with both, true when the HIP runtime lists no card.
pub fn only_nvidia_cards() -> bool {
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    return dual::topology().hip == 0;
    #[cfg(not(all(feature = "cuda", feature = "rocm")))]
    cfg!(feature = "cuda")
}

/// Whether the HIP runtime is the one bound on the calling thread: the last context made
/// current there was an AMD card's. Always false without ROCm, always true without CUDA.
#[inline]
pub fn bound_is_hip() -> bool {
    #[cfg(all(feature = "cuda", feature = "rocm"))]
    return dual::bound_is_hip();
    #[cfg(not(all(feature = "cuda", feature = "rocm")))]
    cfg!(feature = "rocm")
}

/// Declares the launchers of the statically linked kernel libraries, one `extern "C"` block at
/// a time. With one runtime it is that block. With both, each entry becomes a function of the
/// same name and signature that calls the NVIDIA library's symbol, or the AMD library's
/// `loken_hip_`-prefixed one when the HIP runtime is bound on the calling thread.
macro_rules! aot_kernels {
    (
        $(#[$outer:meta])*
        extern "C" {
            $(
                $(#[$attr:meta])*
                $vis:vis fn $name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ) $( -> $ret:ty )? ;
            )*
        }
    ) => {
        $crate::tensor::gpu_runtime::aot_kernels!(@each [$(#[$outer])*] $(
            { $(#[$attr])* $vis fn $name ( $( $arg : $ty ),* ) $( -> $ret )? }
        )*);
    };
    (@each $outer:tt $( { $($item:tt)* } )*) => {
        $( $crate::tensor::gpu_runtime::aot_kernels!(@one $outer $($item)*); )*
    };
    (
        @one [$(#[$outer:meta])*]
        $(#[$attr:meta])*
        $vis:vis fn $name:ident ( $( $arg:ident : $ty:ty ),* ) $( -> $ret:ty )?
    ) => {
        #[cfg(not(all(feature = "cuda", feature = "rocm")))]
        $(#[$outer])*
        extern "C" {
            $(#[$attr])*
            $vis fn $name( $( $arg: $ty ),* ) $( -> $ret )?;
        }
        #[cfg(all(feature = "cuda", feature = "rocm"))]
        $(#[$outer])*
        $(#[$attr])*
        #[allow(non_snake_case, clippy::too_many_arguments, clippy::missing_safety_doc)]
        #[inline]
        $vis unsafe extern "C" fn $name( $( $arg: $ty ),* ) $( -> $ret )? {
            extern "C" {
                #[link_name = stringify!($name)]
                fn cuda( $( $arg: $ty ),* ) $( -> $ret )?;
                #[link_name = concat!("loken_hip_", stringify!($name))]
                fn hip( $( $arg: $ty ),* ) $( -> $ret )?;
            }
            if $crate::tensor::gpu_runtime::bound_is_hip() {
                unsafe { hip( $( $arg ),* ) }
            } else {
                unsafe { cuda( $( $arg ),* ) }
            }
        }
    };
}
pub(crate) use aot_kernels;
