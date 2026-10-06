//! Which CUDA libraries this process can open.
//!
//! The binary links none of them: cudarc opens each one with `dlopen` on first use, and a call
//! into a library that is missing aborts the process. So the question is asked here, before any
//! such call, and a machine that lacks an answer is a machine without CUDA, which the placement
//! ladder already serves on the host.
//!
//! A release archive may carry the libraries in `lib/` beside the executable. They are opened by
//! path before the first probe, so cudarc's lookup by name finds them already loaded and the
//! system's copies are only a fallback.

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

/// The shared objects in `dir` that are real files, by path. Symbolic links name a file that is
/// listed under its own name.
fn bundled_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut found: Vec<std::path::PathBuf> = entries
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("lib") && n.contains(".so"))
        })
        .collect();
    found.sort();
    found
}

/// Open the libraries the archive carries in `lib/` beside the executable. The handles stay
/// open for the life of the process; a library that does not open is left to the probes below.
fn open_bundled() {
    let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|d| d.join("lib")))
    else {
        return;
    };
    for path in bundled_in(&dir) {
        let Ok(name) = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()) else {
            continue;
        };
        // SAFETY: opening a shared object runs its initialisers, which is the point; the
        // archive's libraries are the vendor's own.
        unsafe {
            libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
        }
    }
}

/// The required libraries that did not open, by name. Probed once; a library does not appear
/// while the process runs.
pub fn missing() -> &'static [&'static str] {
    static MISSING: OnceLock<Vec<&'static str>> = OnceLock::new();
    MISSING.get_or_init(|| {
        open_bundled();
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

    /// Only real library files in the directory are opened, in a fixed order.
    #[test]
    fn the_bundle_lists_library_files_only() {
        let dir = std::env::temp_dir().join(format!("loken-bundle-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in ["libb.so.1", "liba.so.2", "notes.txt", "libc.a"] {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
        std::os::unix::fs::symlink(dir.join("liba.so.2"), dir.join("liba.so")).unwrap();
        let names: Vec<String> = bundled_in(&dir)
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(names, ["liba.so.2", "libb.so.1"]);
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
