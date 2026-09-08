//! Quantised blocks on the host: the bytes, their alignment, and the tensor over them.
//!
//! The alignment is not incidental. `quant_cpu` casts these bytes to typed block slices, and
//! a q8_K block holds f32 and i16 members - so owned storage is eight-byte aligned and the
//! cast is always valid.

use super::*;

// ------------------------------------------------------------
// The quantized ENGINE (ex-`native/quant.rs`): host block container
// (`QHostTensor`), production kernel matmul (`QKernelMatMul` - mmvq/MMQ/CPU
// dot dispatch), MMQ FFI boundary, and the GGUF `QVarBuilder`.
// ------------------------------------------------------------

/// Owned block bytes with 8-byte alignment (>= every GGML block's alignment:
/// BlockQ8K holds f32/i16, the rest f16/u8), so `quant_cpu::cast_blocks` is
/// always valid on owned storage. Backed by a `Vec<u64>`.
#[derive(Clone)]
pub(super) struct AlignedBytes {
    buf: Vec<u64>,
    len: usize,
}

impl AlignedBytes {
    pub(crate) fn zeroed(len: usize) -> Self {
        Self {
            buf: vec![0u64; len.div_ceil(8)],
            len,
        }
    }

    pub(crate) fn from_slice(b: &[u8]) -> Self {
        let mut a = Self::zeroed(b.len());
        a.as_mut_slice().copy_from_slice(b);
        a
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const u8, self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.buf.as_mut_ptr() as *mut u8, self.len) }
    }
}

/// Block storage: owned (aligned heap copy) or a zero-copy read-only view
/// into memory owned by something else (mmap'd GGUF file, parent expert
/// stack), re-hosted from `tensor/quant_view.rs`.
pub enum QHostStorage {
    Owned(AlignedBytes),
    View {
        _owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
        ptr: *const u8,
        len: usize,
    },
}

// Safe: views are read-only and the Arc'd owner (mmap / parent QHostTensor heap
// data) is neither mutated nor moved while the view is alive.
unsafe impl Send for QHostStorage {}
unsafe impl Sync for QHostStorage {}

impl Clone for QHostStorage {
    fn clone(&self) -> Self {
        match self {
            Self::Owned(b) => Self::Owned(b.clone()),
            Self::View { _owner, ptr, len } => Self::View {
                _owner: _owner.clone(),
                ptr: *ptr,
                len: *len,
            },
        }
    }
}

/// Quantized tensor: raw GGML block data + dtype + shape (host side; the
/// device side reuses these bytes via CudaStorage::U8 upload).
#[derive(Clone)]
pub struct QHostTensor {
    pub(super) storage: QHostStorage,
    pub dtype: GgmlDType,
    pub dims: Vec<usize>,
    /// What identifies these weights for the rest of their life.
    ///
    /// Not the address: a cache keyed by one answers a NEW tensor with the data of a freed
    /// one that happened to land at the same place. This counter only goes up, so an entry
    /// left behind by a dropped tensor can never be mistaken for a live one - it is a leak
    /// until the cache is cleared, which is what it always was, and no longer a wrong answer.
    pub id: u64,
}

/// The next weight identity. Wraps after 2^64 tensors, which no process reaches.
pub(crate) fn next_tensor_id() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl std::fmt::Debug for QHostTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QHostTensor")
            .field("dtype", &self.dtype)
            .field("dims", &self.dims)
            .field("bytes", &self.data().len())
            .field(
                "storage",
                &match self.storage {
                    QHostStorage::Owned(_) => "owned",
                    QHostStorage::View { .. } => "view",
                },
            )
            .finish()
    }
}

impl QHostTensor {
    /// Owned tensor from raw block bytes (copied into aligned storage).
    pub fn from_bytes(data: &[u8], dtype: GgmlDType, dims: Vec<usize>) -> Result<Self> {
        let elems: usize = dims.iter().product();
        let want = elems / dtype.block_size() * dtype.type_size();
        if data.len() != want {
            return Err(Error(format!(
                "QHostTensor::from_bytes: {} bytes != expected {want} for {dtype:?} {dims:?}",
                data.len()
            )));
        }
        Ok(Self {
            storage: QHostStorage::Owned(AlignedBytes::from_slice(data)),
            dtype,
            dims,
            id: next_tensor_id(),
        })
    }

    /// Zero-copy view of `byte_len` block bytes at `base + byte_offset`, kept
    /// alive by `owner` ( re-host). Alignment and length are checked here so
    /// every downstream block cast is safe.
    ///
    /// # Safety
    /// `base + byte_offset` must be in bounds of a single allocation of at least
    /// `byte_len` bytes, and that memory must stay valid and unmutated for as long
    /// as `owner` keeps it alive - the view reads through the pointer for its whole
    /// lifetime, so a dangling or shrinking base is undefined behaviour that no
    /// check here can catch.
    pub unsafe fn view(
        owner: std::sync::Arc<dyn std::any::Any + Send + Sync>,
        base: *const u8,
        byte_offset: usize,
        byte_len: usize,
        dtype: GgmlDType,
        dims: Vec<usize>,
    ) -> Result<Self> {
        let elems: usize = dims.iter().product();
        let want = elems / dtype.block_size() * dtype.type_size();
        if byte_len != want {
            return Err(Error(format!(
                "QHostTensor::view: {byte_len} bytes != expected {want} for {dtype:?} {dims:?}"
            )));
        }
        let ptr = unsafe { base.add(byte_offset) };
        // 8 covers every block type's alignment (see AlignedBytes); f32/Q8K
        // need 4, f16-headed blocks 2.
        if !(ptr as usize).is_multiple_of(8) {
            return Err(Error(format!(
                "QHostTensor::view: pointer {ptr:?} not 8-byte aligned for {dtype:?}"
            )));
        }
        Ok(Self {
            storage: QHostStorage::View {
                _owner: owner,
                ptr,
                len: byte_len,
            },
            dtype,
            dims,
            id: next_tensor_id(),
        })
    }

    /// Raw GGML block bytes.
    pub fn data(&self) -> &[u8] {
        match &self.storage {
            QHostStorage::Owned(b) => b.as_slice(),
            QHostStorage::View { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }

    pub fn elem_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// Dequantize to f32 (reference CPU path; kernels do this on device).
    /// Routed through the lifted k_quants engine (`quant_cpu`) - bit-exact
    /// with the fork for every supported dtype, incl. MxFp4 and the K-quants.
    pub fn dequantize_f32(&self) -> Result<Vec<f32>> {
        let n = self.elem_count();
        let mut out = vec![0f32; n];
        crate::tensor::quant_cpu::to_float_bytes(self.dtype, self.data(), &mut out)?;
        Ok(out)
    }
}
