//! Which CUDA libraries this process can open.
//!
//! The binary links none of them: cudarc opens each one with `dlopen` on first use, and a call
//! into a library that is missing aborts the process. So the question is asked here, before any
//! such call, and a machine that lacks an answer is a machine without CUDA, which the placement
//! ladder already serves on the host.

use std::sync::OnceLock;

/// A library the CUDA backend cannot run without, and the probe that says whether it opens.
struct Library {
    name: &'static str,
    present: unsafe fn() -> bool,
}

/// Every library a CUDA load touches: the driver, the kernel compiler, and the two matrix
/// libraries. NCCL is separate because only tensor parallelism needs it.
const REQUIRED: [Library; 4] = [
    Library {
        name: "libcuda",
        present: cudarc::driver::sys::is_culib_present,
    },
    Library {
        name: "libnvrtc",
        present: cudarc::nvrtc::sys::is_culib_present,
    },
    Library {
        name: "libcublas",
        present: cudarc::cublas::sys::is_culib_present,
    },
    Library {
        name: "libcublasLt",
        present: cudarc::cublaslt::sys::is_culib_present,
    },
];

/// The required libraries that did not open, by name. Probed once; a library does not appear
/// while the process runs.
pub fn missing() -> &'static [&'static str] {
    static MISSING: OnceLock<Vec<&'static str>> = OnceLock::new();
    MISSING.get_or_init(|| {
        let missing: Vec<&'static str> = REQUIRED
            .iter()
            .filter(|lib| !unsafe { (lib.present)() })
            .map(|lib| lib.name)
            .collect();
        if !missing.is_empty() {
            tracing::warn!(
                "CUDA unavailable, serving on the host: cannot open {}",
                missing.join(", ")
            );
        }
        missing
    })
}

/// Whether every library the CUDA backend needs opens.
pub fn usable() -> bool {
    missing().is_empty()
}

/// Whether NCCL opens, which tensor parallelism across cards requires.
pub fn nccl_present() -> bool {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    *PRESENT.get_or_init(|| unsafe { cudarc::nccl::sys::is_culib_present() })
}

/// The error a CUDA entry point returns when the libraries are missing, so that the caller
/// takes its host path instead of calling into a library that is not there.
pub(crate) fn require() -> super::Result<()> {
    match missing() {
        [] => Ok(()),
        names => Err(super::Error(format!(
            "CUDA libraries not found: {}",
            names.join(", ")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The verdict and the list agree, and asking twice gives the same answer.
    #[test]
    fn usable_means_nothing_is_missing() {
        assert_eq!(usable(), missing().is_empty());
        assert_eq!(missing(), missing());
        assert_eq!(usable(), require().is_ok());
    }

    /// Opening a device on a machine without the libraries is an error, never an abort.
    #[test]
    fn opening_a_device_never_aborts() {
        let opened = super::super::CudaDevice::get(0);
        if !usable() {
            assert!(opened.is_err());
        }
    }
}
