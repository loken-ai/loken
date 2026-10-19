//! Both runtimes in one process: the items of the surface the GPU paths use, each one calling
//! `cudarc` for an NVIDIA card and the HIP implementation for an AMD card.
//!
//! The cards are numbered NVIDIA first, then AMD: ordinal `i` is CUDA's device `i` below the
//! CUDA count, HIP's device `i - cuda` above it. A runtime whose libraries do not open lists no
//! card, so a machine with one vendor's stack lists that vendor's cards only.
//!
//! The types are the HIP implementation's, which carry CUDA's names and numbers; an NVIDIA call
//! converts them to `cudarc`'s at the call. A raw call is routed by what it names: a device by
//! its ordinal, a stream by the runtime that created it, anything else by the runtime bound on
//! the calling thread, which every context operation sets.

use std::cell::Cell;
use std::collections::HashSet;
use std::sync::{OnceLock, RwLock};

pub mod cublas;
pub mod cublaslt;
pub mod driver;
pub mod nccl;
pub mod nvrtc;

/// The runtime that owns a card or a handle.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Runtime {
    Cuda,
    Hip,
}

/// How many cards each runtime lists.
#[derive(Debug)]
pub(crate) struct Topology {
    pub cuda: usize,
    pub hip: usize,
}

impl Topology {
    /// Whether `ordinal` is an NVIDIA card.
    pub fn is_cuda(&self, ordinal: usize) -> bool {
        ordinal < self.cuda
    }

    /// The runtime of `ordinal` and its number there; `None` past the last card.
    pub fn locate(&self, ordinal: usize) -> Option<(Runtime, usize)> {
        if ordinal < self.cuda {
            Some((Runtime::Cuda, ordinal))
        } else if ordinal < self.cuda + self.hip {
            Some((Runtime::Hip, ordinal - self.cuda))
        } else {
            None
        }
    }

    /// The ordinal of a runtime's device `local`.
    pub fn global(&self, runtime: Runtime, local: usize) -> usize {
        match runtime {
            Runtime::Cuda => local,
            Runtime::Hip => self.cuda + local,
        }
    }

    /// Every card, both runtimes.
    pub fn count(&self) -> usize {
        self.cuda + self.hip
    }
}

/// Whether every library the CUDA paths need opens: the driver, NVRTC, cuBLAS and cuBLASLt.
pub(crate) fn cuda_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| unsafe {
        cudarc::driver::sys::is_culib_present()
            && cudarc::nvrtc::sys::is_culib_present()
            && cudarc::cublas::sys::is_culib_present()
            && cudarc::cublaslt::sys::is_culib_present()
    })
}

/// Whether every library the HIP paths need opens: the runtime, hipRTC, hipBLAS and hipBLASLt.
pub(crate) fn hip_usable() -> bool {
    static USABLE: OnceLock<bool> = OnceLock::new();
    *USABLE.get_or_init(|| unsafe {
        super::hip::driver::sys::is_culib_present()
            && super::hip::nvrtc::sys::is_culib_present()
            && super::hip::cublas::sys::is_culib_present()
            && super::hip::cublaslt::sys::is_culib_present()
    })
}

/// The cards of both runtimes, counted once.
pub(crate) fn topology() -> &'static Topology {
    static TOPOLOGY: OnceLock<Topology> = OnceLock::new();
    TOPOLOGY.get_or_init(|| {
        // The libraries an archive carries beside the executable are opened before the probes.
        let _ = crate::tensor::cuda::runtime::usable();
        let cuda = if cuda_usable() && cudarc::driver::result::init().is_ok() {
            cudarc::driver::result::device::get_count()
                .unwrap_or(0)
                .max(0) as usize
        } else {
            0
        };
        let hip = if hip_usable() && super::hip::driver::result::init().is_ok() {
            super::hip::driver::result::device::get_count()
                .unwrap_or(0)
                .max(0) as usize
        } else {
            0
        };
        tracing::info!(
            "GPU runtimes: {cuda} NVIDIA card(s) through CUDA, {hip} AMD card(s) through HIP"
        );
        Topology { cuda, hip }
    })
}

thread_local! {
    static BOUND: Cell<Option<Runtime>> = const { Cell::new(None) };
}

/// The runtime a thread uses before any context is bound on it: the one that lists cards, CUDA
/// when both do, and a runtime whose libraries open when neither does.
fn default_runtime() -> Runtime {
    let t = topology();
    if t.cuda > 0 || (t.hip == 0 && cuda_usable()) {
        Runtime::Cuda
    } else {
        Runtime::Hip
    }
}

/// The runtime bound on the calling thread.
#[inline]
pub(crate) fn bound() -> Runtime {
    BOUND.with(|b| b.get()).unwrap_or_else(|| {
        let rt = default_runtime();
        BOUND.with(|b| b.set(Some(rt)));
        rt
    })
}

/// Records `runtime` as the one bound on the calling thread.
#[inline]
pub(crate) fn bind(runtime: Runtime) {
    BOUND.with(|b| b.set(Some(runtime)));
}

/// Whether the HIP runtime is bound on the calling thread.
#[inline]
pub(crate) fn bound_is_hip() -> bool {
    bound() == Runtime::Hip
}

/// A set of raw handles one runtime created, for the calls that name a handle and nothing else.
pub(crate) struct Handles(OnceLock<RwLock<HashSet<usize>>>);

impl Handles {
    pub const fn new() -> Self {
        Self(OnceLock::new())
    }

    fn set(&self) -> &RwLock<HashSet<usize>> {
        self.0.get_or_init(|| RwLock::new(HashSet::new()))
    }

    pub fn insert<T>(&self, h: *mut T) {
        self.set()
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(h as usize);
    }

    pub fn remove<T>(&self, h: *mut T) -> bool {
        self.set()
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&(h as usize))
    }

    pub fn contains<T>(&self, h: *const T) -> bool {
        self.set()
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(&(h as usize))
    }
}

/// The streams the HIP runtime created.
pub(crate) static HIP_STREAMS: Handles = Handles::new();

/// The runtime a stream belongs to: the default stream is the bound runtime's.
#[inline]
pub(crate) fn stream_runtime<T>(stream: *mut T) -> Runtime {
    let t = topology();
    if t.hip == 0 {
        return if t.cuda == 0 { bound() } else { Runtime::Cuda };
    }
    if t.cuda == 0 {
        return Runtime::Hip;
    }
    if stream.is_null() {
        bound()
    } else if HIP_STREAMS.contains(stream) {
        Runtime::Hip
    } else {
        Runtime::Cuda
    }
}

/// CUDA's description of a status, for an error raised on a thread bound to CUDA.
pub(crate) fn cuda_error_string(e: driver::sys::CUresult) -> Option<String> {
    if bound() != Runtime::Cuda || !cuda_usable() {
        return None;
    }
    let e = cudarc::driver::DriverError(driver::sys::to_cuda_result(e)?);
    e.error_string()
        .ok()
        .map(|s| s.to_string_lossy().into_owned())
}
