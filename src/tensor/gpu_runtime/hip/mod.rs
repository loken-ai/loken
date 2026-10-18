//! The HIP runtime under the surface the GPU paths use: the items of `cudarc` they call, with
//! the same names and signatures, implemented over the HIP runtime, hipBLAS, hipRTC and RCCL.
//!
//! Every library is opened at run time. A machine without ROCm builds and starts: the probes
//! report the libraries missing and the device count call fails, so no card is listed. A symbol
//! a library lacks makes its call return a not-supported error instead of aborting.

pub mod cublas;
pub mod cublaslt;
pub mod driver;
pub mod nccl;
pub mod nvrtc;

/// Opens the shared library `lib<stem>.so`: by name through the loader's search path, then by
/// its versioned names.
pub(crate) fn open_library(stem: &str) -> Option<libloading::Library> {
    let file = format!("lib{stem}.so");
    std::iter::once(file.clone())
        .chain((0..=MAX_SONAME_MAJOR).rev().map(|v| format!("{file}.{v}")))
        .find_map(|c| unsafe { libloading::Library::new(c.as_str()) }.ok())
}

/// The highest major version tried when a library has no unversioned name on the search path.
const MAX_SONAME_MAJOR: u32 = 16;

/// Declares the entry points of one shared library, resolved once on first use. Each entry is
/// `None` when the library does not export it; `get` is `None` when the library does not open.
macro_rules! dynamic_api {
    (
        $(#[$meta:meta])*
        $vis:vis struct $api:ident from $stem:literal {
            $( fn $name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ) $( -> $ret:ty )? ; )*
        }
    ) => {
        $(#[$meta])*
        #[allow(non_snake_case, dead_code)]
        $vis struct $api {
            _library: libloading::Library,
            $( pub $name: Option<unsafe extern "C" fn( $( $ty ),* ) $( -> $ret )?>, )*
        }

        #[allow(dead_code)]
        impl $api {
            fn load() -> Option<Self> {
                let library = crate::tensor::gpu_runtime::hip::open_library($stem)?;
                // SAFETY: each symbol is read with the C signature its header declares.
                #[allow(unused_unsafe)]
                unsafe {
                    Some(Self {
                        $( $name: library
                            .get::<unsafe extern "C" fn( $( $ty ),* ) $( -> $ret )?>(
                                stringify!($name),
                            )
                            .ok()
                            .map(|s| *s), )*
                        _library: library,
                    })
                }
            }

            /// The library's entry points, or `None` when it does not open.
            pub fn get() -> Option<&'static Self> {
                static API: std::sync::OnceLock<Option<$api>> = std::sync::OnceLock::new();
                API.get_or_init(Self::load).as_ref()
            }

            /// Whether the library opens.
            pub fn present() -> bool {
                Self::get().is_some()
            }
        }
    };
}
pub(crate) use dynamic_api;
