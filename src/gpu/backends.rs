//! The accelerator backends the server may use.
//!
//! Every backend is allowed until `set_allowed` names the ones to keep.
//! The host always computes: `Backend::Cpu` only names it.

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cuda,
    Rocm,
    Vulkan,
    OpenCl,
    Cpu,
}

impl Backend {
    const ALL: [Backend; 5] = [
        Backend::Cuda,
        Backend::Rocm,
        Backend::Vulkan,
        Backend::OpenCl,
        Backend::Cpu,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "cuda",
            Backend::Rocm => "rocm",
            Backend::Vulkan => "vulkan",
            Backend::OpenCl => "opencl",
            Backend::Cpu => "cpu",
        }
    }

    fn bit(self) -> u8 {
        1 << self as u8
    }
}

static ALLOWED: AtomicU8 = AtomicU8::new(0xff);

/// The backends a comma separated list names.
pub fn parse(list: &str) -> Result<Vec<Backend>, String> {
    let mut out = Vec::new();
    for word in list.split(',').map(|w| w.trim().to_ascii_lowercase()) {
        let backend = Backend::ALL
            .into_iter()
            .find(|b| b.name() == word)
            .ok_or_else(|| {
                let names: Vec<_> = Backend::ALL.iter().map(|b| b.name()).collect();
                format!("unknown backend `{word}`, expected {}", names.join(", "))
            })?;
        out.push(backend);
    }
    Ok(out)
}

/// Keeps only `backends`. Call before any device is probed.
pub fn set_allowed(backends: &[Backend]) {
    let bits = backends.iter().fold(0, |bits, b| bits | b.bit());
    ALLOWED.store(bits, Ordering::SeqCst);
}

pub fn allows(backend: Backend) -> bool {
    ALLOWED.load(Ordering::SeqCst) & backend.bit() != 0
}

/// Whether a card of the GPU runtime this build was made with may be used.
pub fn cards_allowed() -> bool {
    (cfg!(feature = "cuda") && allows(Backend::Cuda))
        || (cfg!(feature = "rocm") && allows(Backend::Rocm))
}

/// The PCI vendor of the cards CUDA drives.
const VENDOR_NVIDIA: u32 = 0x10de;
/// The PCI vendor of the cards ROCm drives.
const VENDOR_AMD: u32 = 0x1002;

/// Whether a native runtime drives the cards of PCI vendor `vendor_id`.
///
/// CUDA drives NVIDIA's cards and ROCm AMD's.
/// A runtime drives them only when `built` says the build has it and `allowed` keeps it.
fn native_drives(
    vendor_id: u32,
    built: impl Fn(Backend) -> bool,
    allowed: impl Fn(Backend) -> bool,
) -> bool {
    let runtime = match vendor_id {
        VENDOR_NVIDIA => Backend::Cuda,
        VENDOR_AMD => Backend::Rocm,
        _ => return false,
    };
    built(runtime) && allowed(runtime)
}

/// Whether a native runtime of this build, kept by `set_allowed`, drives the cards of PCI
/// vendor `vendor_id`.
pub fn native_runtime_drives(vendor_id: u32) -> bool {
    let built = |b: Backend| match b {
        Backend::Cuda => cfg!(feature = "cuda"),
        Backend::Rocm => cfg!(feature = "rocm"),
        _ => false,
    };
    native_drives(vendor_id, built, allows)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A list names its backends in any case and with spaces.
    /// A name that is not a backend is refused.
    #[test]
    fn a_list_names_backends() {
        assert_eq!(
            parse("CUDA, vulkan").unwrap(),
            vec![Backend::Cuda, Backend::Vulkan]
        );
        assert!(parse("cuda,tpu").unwrap_err().contains("tpu"));
    }

    /// A vendor's cards belong to its native runtime only when the build has it and the
    /// backends keep it.
    #[test]
    fn a_native_runtime_drives_its_vendor_only_when_allowed() {
        let all = |_: Backend| true;
        let only = |kept: Backend| move |b: Backend| b == kept;
        assert!(native_drives(VENDOR_NVIDIA, all, all));
        assert!(native_drives(VENDOR_AMD, all, all));
        assert!(!native_drives(VENDOR_NVIDIA, all, only(Backend::Vulkan)));
        assert!(!native_drives(VENDOR_AMD, all, only(Backend::Vulkan)));
        assert!(native_drives(VENDOR_AMD, all, only(Backend::Rocm)));
        assert!(!native_drives(VENDOR_NVIDIA, all, only(Backend::Rocm)));
        assert!(!native_drives(VENDOR_NVIDIA, only(Backend::Rocm), all));
        assert!(!native_drives(0x8086, all, all));
    }
}
