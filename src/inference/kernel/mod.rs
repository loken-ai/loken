//! Hand-written compute: fused CUDA/CPU kernels, the tensor-core decode, and
//! the OpenCL layer.
//!
//! Callers name a module through this directory - `crate::inference::kernel::<module>` - so the path says which
//! part of the system a file belongs to, which is the whole reason the directory exists.

/// Declare the CUDA launchers this crate calls, one row per kernel.
///
/// Every launcher takes the stream it runs on as its last argument, so the macro appends it: a
/// row states only what makes its kernel different, and no kernel can be declared without one.
/// The names and types have to match the definitions under `cuda/` - nothing but the linker
/// checks that, which is the reason to keep a declaration short enough to read against its
/// definition.
macro_rules! cuda_launchers {
    ($(
        $(#[$attr:meta])*
        $name:ident($($arg:ident: $ty:ty),* $(,)?);
    )+) => {
        extern "C" {
            $(
                $(#[$attr])*
                fn $name($($arg: $ty,)* stream: i64);
            )+
        }
    };
}
pub(crate) use cuda_launchers;

pub mod cpu_decode_exec;
#[cfg(feature = "cuda")]
pub mod flash_decode_tc;
// One module, two implementations chosen by the build. The CPU one answers `None` for the
// optional fast paths so the caller falls back, and carries portable code for the rest - so a
// caller writes `kernel::fused::...` without knowing which build it is in.
#[cfg(feature = "cuda")]
pub mod fused;
#[cfg(not(feature = "cuda"))]
#[path = "fused_cpu.rs"]
pub mod fused;
#[cfg(feature = "opencl")]
pub mod opencl;
pub mod opencl_probe;
