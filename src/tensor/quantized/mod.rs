//! The quantized substrate, ONE home (a fold of the ex-`quant.rs` engine):
//! GGML block dtypes, the from-scratch GGUF reader (`gguf_file`), the host
//! block container [`QHostTensor`], the production kernel matmul
//! [`QKernelMatMul`] (mmvq/MMQ/CPU dot dispatch), and the facade layer the
//! LLM loaders consume - [`QTensor`] (host blocks + device blob + the
//! OnceLock-cached kernel matmul) and the [`QMatMul`] enum (quantized kernels
//! + dense fallbacks). Re-exported as `crate::tensor::quantized`.
//!
//! `QTensor`/`QMatMul` and `QHostTensor`/`QKernelMatMul`
//! stay DISTINCT types on purpose: the facade caches a `QKernelMatMul` built
//! over its own shared device blob (OnceLock, the hot weight path), and a
//! `QKernelMatMul` holds an `Arc<QHostTensor>` - folding the cache into the
//! host tensor itself would make every cached weight an Arc self-cycle (leak
//! on model unload).
#![allow(clippy::result_large_err)]

use crate::tensor;
use crate::tensor::kernel_ffi::CudaDevice;
use crate::tensor::Shape;
// Fully de-wrapped: the quantized public API takes/returns the concrete
// native `Device`/`Error`/`Result`/`Tensor` directly - ZERO compat. The
// device-resident `QCudaStorage` still keys off the compat `CudaDevice`
// newtype (shared with `fast_mmvq`/`wrap_cuda_slice`), constructed inline
// from the native `Arc<CudaDevice>` at the two upload sites.
use std::collections::HashMap;
use std::sync::Arc;
use tensor::{Device, Error, Result, Tensor};

/// Bytes past the end of the block data every device blob carries.
///
/// The tile and mat-vec kernels read a row padded to 512 elements, so a weight whose
/// input dim is not a multiple of that is read past its last row. Stated once: the
/// upload, the allocation and the count of either cannot disagree about how much room
/// a weight takes.
pub(crate) const BLOB_TAIL_PAD_BYTES: usize = 4096;

/// The ledger entry a placement on a counting device takes for a quantised weight.
///
/// A dry device stands in for a card, and what a card gives up for these blocks is the
/// padded blob above - so the entry is taken at the moment the upload would happen and
/// released by the drop that would have freed it, and a weight costs the ledger for
/// exactly as long as it would cost the card. `None` for a real device, whose own
/// allocation is the record. Nothing is read and nothing is copied: the byte count is
/// the entire truth about what a weight takes.
pub(crate) fn dry_blob_room(
    qt: &QHostTensor,
    device: &Device,
) -> Option<crate::tensor::dry::DryStorage> {
    let led = device.dry_ledger()?;
    Some(crate::tensor::dry::DryStorage::new(
        led.clone(),
        crate::tensor::DType::U8,
        qt.data().len() + BLOB_TAIL_PAD_BYTES,
    ))
}

mod cuda_storage;
mod dtype;
mod host;
mod kernel_matmul;
mod matmul;
mod mmq;
mod qtensor;
mod storage;
mod varbuilder;

/// The GGUF container: a from-scratch reader for v2 and v3 headers.
pub mod gguf_file;

/// The fused gate-and-up Q4_K decode entries, over the NVRTC-compiled mat-vec kernels.
#[cfg(feature = "cuda")]
pub mod fast_mmvq;

/// A quantised row is padded up to this many values.
///
/// The mat-vec and mat-mul kernels read whole blocks and over-read past a row's logical end,
/// so the buffer has to be there. It was written out five times in four files, three of them
/// as constants local to a function - five chances for a padding rule and a kernel's over-read
/// to disagree, which is a read past the end of a buffer.
pub const MATRIX_ROW_PADDING: usize = 512;

/// Values one quantise-to-q8_1 block covers, which is the launch's block width.
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;

/// `x` raised to the next multiple of `q`.
pub fn pad_to(x: usize, q: usize) -> usize {
    x.div_ceil(q) * q
}

pub use cuda_storage::QCudaStorage;
pub use dtype::GgmlDType;
pub use host::QHostTensor;
pub use kernel_matmul::{kernel_known_answer_test, QKernelMatMul};
pub use matmul::QMatMul;
#[cfg(feature = "cuda")]
pub use mmq::release_mmq_workspaces;
pub use qtensor::QTensor;
pub use storage::{QStorage, QStorageWithHost};
pub use varbuilder::QVarBuilder;

#[cfg(test)]
mod tests;
