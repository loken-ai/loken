//! Heterogeneous device tag shared by hetero pipelines.
//!
//! Language models place their layers through `GenericHeteroTransformer` and its own
//! `HeteroPlan`; what lives here is the tag those placements are written in, which the flux
//! image pipeline (`hetero_flux`) also uses to label a block.

/// Which device a layer/block runs on.
///
/// Compared by VALUE, not by discriminant: `Cuda(0)` and `Cuda(1)` are different
/// cards, and code that groups blocks into segments has to see that difference or it
/// runs one card's blocks against another card's tensors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeteroDevice {
    Cpu,
    Cuda(usize),   // CUDA device index
    OpenCL(usize), // OpenCL device index
}

impl std::fmt::Display for HeteroDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda(idx) => write!(f, "CUDA({})", idx),
            Self::OpenCL(idx) => write!(f, "opencl({})", idx),
        }
    }
}
