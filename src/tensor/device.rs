//! Device + unified storage for the native tensor (CPU and, with the `cuda`
//! feature, GPU via the vendored cudarc).

#[cfg(feature = "cuda")]
use super::cuda::{CudaDevice, CudaStorage};
use super::dry::{DryDevice, DryStorage};
use super::CpuStorage;
use super::DType;
use super::Result;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum Device {
    Cpu,
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaDevice>),
    /// Owns no memory: a forward placed here runs its real code and allocates
    /// nothing, so what it would have held can be read off afterwards. See
    /// [`super::dry`] for why this is a device and not a flag on the CUDA one.
    Dry(Arc<DryDevice>),
}

/// Where a tensor lives (mirrors the surface the inference layer reads off `Device`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceLocation {
    Cpu,
    Cuda { gpu_id: usize },
}

impl Device {
    /// Whether a card is about to be TOUCHED - an upload, a launch, a stream, a
    /// handle. False on the counting device, which is what keeps every one of those
    /// refusing there. For "which arrangement does this run", see
    /// [`Self::runs_as_card`].
    pub fn is_cuda(&self) -> bool {
        match self {
            Self::Cpu => false,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => true,
            Self::Dry(_) => false,
        }
    }

    /// Whether the code should take the ARRANGEMENT a card gets.
    ///
    /// A branch on a device asks one of two questions, and until this existed they
    /// shared one spelling:
    ///
    ///  - which computation to run - fused or unfused, one kernel or six tensor ops.
    ///    That is a question about SHAPE, and its answer decides what is allocated;
    ///  - whether a card is about to be touched. That is a question about HARDWARE,
    ///    and it is [`Self::is_cuda`].
    ///
    /// A device that counts instead of allocating must answer yes to the first and no
    /// to the second: a reserve is worth something only if it measures the arrangement
    /// that will actually run, and it is safe only if nothing reaches a card.
    ///
    /// This differs from `is_cuda` on the dry device and NOWHERE else, so a site moved
    /// from one to the other resolves identically on a card and on the host.
    pub fn runs_as_card(&self) -> bool {
        match self {
            Self::Cpu => false,
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => true,
            Self::Dry(_) => true,
        }
    }

    /// A device that counts instead of allocating.
    pub fn is_dry(&self) -> bool {
        matches!(self, Self::Dry(_))
    }

    /// The ledger behind a dry device, for the caller that opened the dry run.
    pub fn dry_ledger(&self) -> Option<&Arc<DryDevice>> {
        match self {
            Self::Dry(d) => Some(d),
            _ => None,
        }
    }

    /// A fresh device that owns no memory.
    pub fn dry() -> Self {
        Self::Dry(DryDevice::new())
    }

    pub fn is_cpu(&self) -> bool {
        matches!(self, Self::Cpu)
    }

    /// Open CUDA device `ordinal` (a real stream, not the NULL stream).
    #[cfg(feature = "cuda")]
    pub fn new_cuda(ordinal: usize) -> Result<Self> {
        // Through the per-ordinal registry, NEVER a fresh instance: every CudaDevice
        // carries its own stream, and per-slice event tracking - the thing that would
        // order one stream against another - is off. Two tensors created through two
        // instances of the same card then execute on two unordered streams, and a
        // pipeline assembled from several `new_cuda` calls races with itself: measured
        // as a fixed seed answering different audio on every render until the whole
        // stack shared one stream.
        Ok(Self::Cuda(super::cuda::CudaDevice::get(ordinal)?))
    }
    #[cfg(not(feature = "cuda"))]
    pub fn new_cuda(_ordinal: usize) -> Result<Self> {
        Err(super::Error("cuda feature not enabled".into()))
    }

    /// CUDA device `ordinal` if it opens, else CPU.
    pub fn cuda_if_available(ordinal: usize) -> Result<Self> {
        Ok(Self::new_cuda(ordinal).unwrap_or(Self::Cpu))
    }

    pub fn location(&self) -> DeviceLocation {
        match self {
            Self::Cpu => DeviceLocation::Cpu,
            #[cfg(feature = "cuda")]
            Self::Cuda(d) => DeviceLocation::Cuda {
                gpu_id: d.ordinal(),
            },
            // KNOWN LIMIT, and the one thing that can make a dry run measure a
            // forward other than the one that will run. Code that branches on the
            // device - `if device.is_cuda()` - takes the host branch here, so a
            // model whose two branches allocate differently would be counted on the
            // wrong one. It is reported as CPU rather than as a card because the
            // alternative is worse: claiming to be CUDA sends the code to
            // `as_cuda_device`, which a dry device cannot answer, and the forward
            // fails instead of diverging quietly. A dry run is only trustworthy for
            // a model whose branches agree, which is a property to check per engine.
            Self::Dry(_) => DeviceLocation::Cpu,
        }
    }

    /// Install the seed `Tensor::randn` and `randn_like` draw from.
    ///
    /// PER THREAD, not per process and not per device - see [`rng`](super::rng) for why: two
    /// concurrent renders on one stream would steal each other's values and neither would
    /// replay. A generation runs its noise draw on the thread that installs the seed.
    ///
    /// This was a no-op once, which made every `seed` parameter decorative on the paths that
    /// draw their noise through the tensor layer.
    pub fn set_seed(&self, seed: u64) -> Result<()> {
        crate::tensor::rng::set_global_seed(seed);
        Ok(())
    }

    pub fn synchronize(&self) -> Result<()> {
        match self {
            Self::Cpu => Ok(()),
            Self::Dry(_) => Ok(()),
            #[cfg(feature = "cuda")]
            Self::Cuda(d) => d.synchronize(),
        }
    }

    /// The handle a kernel launch needs: an `Arc` inside, so cloning it is cheap, and the
    /// alloc / memcpy / custom-launch sites read it rather than the device enum.
    pub fn as_cuda_device(&self) -> Result<super::kernel_ffi::CudaDevice> {
        match self {
            Self::Cpu => Err(super::Error("expected a cuda device, got cpu".to_string())),
            Self::Dry(_) => Err(super::Error(
                "expected a cuda device, got a dry one".to_string(),
            )),
            #[cfg(feature = "cuda")]
            Self::Cuda(d) => Ok(super::kernel_ffi::CudaDevice(d.clone())),
        }
    }

    pub fn same_device(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu, Self::Cpu) => true,
            #[cfg(feature = "cuda")]
            (Self::Cuda(a), Self::Cuda(b)) => a.ordinal() == b.ordinal(),
            (Self::Dry(a), Self::Dry(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

#[derive(Debug)]
pub enum Storage {
    Cpu(CpuStorage),
    #[cfg(feature = "cuda")]
    Cuda {
        data: CudaStorage,
        dev: Arc<CudaDevice>,
    },
    /// A shape and a dtype, charged to a ledger and given back on drop.
    Dry(DryStorage),
}

impl Storage {
    pub fn dtype(&self) -> DType {
        match self {
            Self::Cpu(s) => s.dtype(),
            #[cfg(feature = "cuda")]
            Self::Cuda { data, .. } => data.dtype(),
            Self::Dry(s) => s.dtype(),
        }
    }

    /// Elements the allocation holds, as opposed to what a view over it covers.
    ///
    /// Needed to tell a view that spans the whole buffer from one that narrows it: the
    /// first is already packed and materialising it copies a buffer onto itself.
    pub fn elem_count(&self) -> usize {
        match self {
            Self::Cpu(s) => s.len(),
            #[cfg(feature = "cuda")]
            Self::Cuda { data, .. } => data.len(),
            Self::Dry(s) => s.len(),
        }
    }

    pub fn device(&self) -> Device {
        match self {
            Self::Cpu(_) => Device::Cpu,
            #[cfg(feature = "cuda")]
            Self::Cuda { dev, .. } => Device::Cuda(dev.clone()),
            Self::Dry(s) => Device::Dry(s.device().clone()),
        }
    }
}
