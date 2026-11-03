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
///
/// A row that ends `-> i32` returns a status: 0 when the kernel was launched, nonzero when the
/// launcher skipped it.
macro_rules! cuda_launchers {
    ($(
        $(#[$attr:meta])*
        $vis:vis $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?;
    )+) => {
        crate::tensor::gpu_runtime::aot_kernels! {
            extern "C" {
                $(
                    $(#[$attr])*
                    $vis fn $name($($arg: $ty,)* stream: i64) $(-> $ret)?;
                )+
            }
        }
    };
}
pub(crate) use cuda_launchers;

/// Turns the status of a launcher that reports one into an error naming it.
///
/// 0 means the kernel was launched. Anything else means the launcher skipped it because a
/// scratch allocation failed, and the caller's output buffer was not written.
#[cfg(feature = "gpu")]
pub(crate) fn launch_status(status: i32, launcher: &str) -> crate::tensor::Result<()> {
    if status != 0 {
        crate::tensor::bail!(
            "{launcher}: launch skipped because a scratch allocation failed (status {status})"
        );
    }
    Ok(())
}

pub mod cpu_decode_exec;
#[cfg(feature = "gpu")]
// One module, two implementations chosen by the build. The CPU one answers `None` for the
// optional fast paths so the caller falls back, and carries portable code for the rest - so a
// caller writes `kernel::fused::...` without knowing which build it is in.
#[cfg(feature = "gpu")]
pub mod fused;
#[cfg(not(feature = "gpu"))]
#[path = "fused_cpu.rs"]
pub mod fused;
pub mod image_attention;
#[cfg(feature = "opencl")]
pub mod opencl;
pub mod opencl_probe;
#[cfg(feature = "opencl")]
pub mod opencl_stack;
#[cfg(feature = "vulkan")]
pub mod vulkan;
