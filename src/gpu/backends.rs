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
}
