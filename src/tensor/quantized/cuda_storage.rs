//! Quantised blocks resident on a card.
//!
//! Upload takes the host bytes as they are: a block format is a byte layout, so there is
//! nothing to convert on the way in. What the device copy adds is row padding - the kernels
//! read whole blocks past the logical end of a row - and the dequantisation path back out.

use super::*;

/// Device-resident quantized blob: a `QCudaStorage`.
/// The blob is `Arc`-shared so a weight's `QMatMul` kernels and its
/// `storage()` projection reuse ONE padded upload (no double VRAM).
#[cfg(feature = "cuda")]

pub struct QCudaStorage {
    pub(crate) blob: Arc<cudarc::driver::CudaSlice<u8>>,
    /// Unpadded byte length of the block data.
    pub(crate) len: usize,
    dtype: GgmlDType,
    device: CudaDevice,
}

/// Zeroed device room for `len` bytes of blocks plus the over-read tail.
///
/// Every buffer this type hands to a kernel is allocated this way - a fresh blob, an upload,
/// and a re-quantisation that outgrew its blob all want the same thing, so the tail is added
/// in one place rather than remembered in three.
#[cfg(feature = "cuda")]
fn padded_blob(
    device: &CudaDevice,
    what: &'static str,
    len: usize,
) -> Result<cudarc::driver::CudaSlice<u8>> {
    crate::tensor::cuda::with_oom_retry(&device.0, what, || {
        device
            .0
            .stream()
            .alloc_zeros::<u8>(len + BLOB_TAIL_PAD_BYTES)
    })
    .map_err(|e: crate::tensor::Error| Error::msg(e.to_string()))
}

#[cfg(feature = "cuda")]
impl QCudaStorage {
    /// Zeroed storage for `el_count` elements of `dtype` (KV staging).
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let len = el_count.div_ceil(dtype.block_size()) * dtype.type_size();
        Self::over(
            padded_blob(device, "QCudaStorage::zeros", len)?,
            len,
            dtype,
            device,
        )
    }

    /// Padded upload of host block bytes (weights).
    pub(crate) fn upload(device: &CudaDevice, data: &[u8], dtype: GgmlDType) -> Result<Self> {
        let mut blob = padded_blob(device, "QCudaStorage upload", data.len())?;
        device
            .0
            .stream()
            .memcpy_htod(data, &mut blob.slice_mut(0..data.len()))
            .map_err(|e| Error::msg(format!("QCudaStorage upload: {e}")))?;
        Self::over(blob, data.len(), dtype, device)
    }

    /// This type over an already-allocated blob whose first `len` bytes are the block data.
    fn over(
        blob: cudarc::driver::CudaSlice<u8>,
        len: usize,
        dtype: GgmlDType,
        device: &CudaDevice,
    ) -> Result<Self> {
        Ok(Self {
            blob: Arc::new(blob),
            len,
            dtype,
            device: device.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    /// Raw row-padded quantized blob (kernel launchers pair this with
    /// [`Self::dtype`]).
    pub fn weight_cuda_slice(&self) -> &cudarc::driver::CudaSlice<u8> {
        &self.blob
    }

    /// First `byte_len` (unpadded) bytes as a device view.
    pub fn data_slice_for_copy(&self, byte_len: usize) -> Result<cudarc::driver::CudaView<'_, u8>> {
        if byte_len > self.len {
            return Err(Error::msg(format!(
                "data_slice_for_copy: byte_len {byte_len} exceeds unpadded length {}",
                self.len
            )));
        }
        Ok(self.blob.slice(..byte_len))
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        use cudarc::driver::DevicePtr;
        Ok(self.blob.device_ptr(self.blob.stream()).0 as *const u8)
    }

    /// Device-side quantize of an F32/F16 source tensor's storage into
    /// this blob (Q8_0/Q4_0 fast path - the KV-cache staging op). Same
    /// kernels as the fork (`quantize_q8_0`/`quantize_q4_0` [+ `_f16`] in
    /// the in-crate quantized.cu).
    pub fn quantize_with_layout<L: std::borrow::Borrow<crate::tensor::kernel_ffi::Layout>>(
        &mut self,
        src: &crate::tensor::kernel_ffi::CudaStorage,
        layout: L,
    ) -> Result<()> {
        let layout = layout.borrow();
        use cudarc::driver::PushKernelArg;
        let offset = layout.start_offset();
        let elem_count = layout.shape().elem_count();
        let (is_q4, kname_f32, kname_f16) = match self.dtype {
            GgmlDType::Q4_0 => (true, "quantize_q4_0", "quantize_q4_0_f16"),
            GgmlDType::Q8_0 => (false, "quantize_q8_0", "quantize_q8_0_f16"),
            other => {
                return Err(Error::msg(format!(
                    "quantize_with_layout: unsupported target dtype {other:?} (Q8_0/Q4_0 only)"
                )))
            }
        };
        let _ = is_q4;
        let block_size = self.dtype.block_size();
        let type_size = self.dtype.type_size();
        if elem_count % block_size != 0 {
            return Err(Error::msg(format!(
                "quantize_with_layout: {elem_count} elements not divisible by block {block_size}"
            )));
        }
        // The row padding and the quantiser's block width are the kernels' own, and are
        // declared where the kernels are launched from.
        use crate::inference::quantized_cuda::{
            pad, quantize_launch, CUDA_QUANTIZE_BLOCK_SIZE, MATRIX_ROW_PADDING,
        };
        let kx_padded = pad(elem_count, MATRIX_ROW_PADDING);
        let unpadded_bytes = (elem_count / block_size) * type_size;
        let needed = kx_padded / block_size * type_size;

        // (Re)allocate if the current blob can't hold the padded write.
        if self.blob.len() < needed {
            let device = self.device.clone();
            self.blob = Arc::new(padded_blob(&device, "quantize_with_layout", needed)?);
        }
        self.len = unpadded_bytes;
        let blob = Arc::get_mut(&mut self.blob).ok_or_else(|| {
            Error::msg(
                "quantize_with_layout: blob is shared (only exclusively-owned staging \
                     storages may be re-quantized)",
            )
        })?;

        let cfg = quantize_launch(kx_padded.div_ceil(CUDA_QUANTIZE_BLOCK_SIZE), 1);
        let kx = elem_count as i32;
        let kxp = kx_padded as i32;
        let stream = self.device.0.stream().clone();
        let src_view = Self::blob_src_view(src, offset, elem_count)?;
        match src_view {
            QuantizeSrc::F32(view) => {
                let func = self.device.0.quantized_fn(kname_f32)?;
                let mut b = stream.launch_builder(&func);
                b.arg(&view).arg(blob).arg(&kx).arg(&kxp);
                unsafe { b.launch(cfg) }
                    .map_err(|e| Error::msg(format!("quantize launch: {e}")))?;
            }
            QuantizeSrc::F16(view) => {
                let func = self.device.0.quantized_fn(kname_f16)?;
                let mut b = stream.launch_builder(&func);
                b.arg(&view).arg(blob).arg(&kx).arg(&kxp);
                unsafe { b.launch(cfg) }
                    .map_err(|e| Error::msg(format!("quantize launch: {e}")))?;
            }
        }
        Ok(())
    }

    fn blob_src_view<'a>(
        src: &'a crate::tensor::kernel_ffi::CudaStorage,
        offset: usize,
        elem_count: usize,
    ) -> Result<QuantizeSrc<'a>> {
        if let Ok(s) = src.as_cuda_slice::<f32>() {
            return Ok(QuantizeSrc::F32(s.slice(offset..offset + elem_count)));
        }
        if let Ok(s) = src.as_cuda_slice::<half::f16>() {
            return Ok(QuantizeSrc::F16(s.slice(offset..offset + elem_count)));
        }
        Err(Error::msg(
            "quantize_with_layout: source must be f32 or f16 CUDA storage",
        ))
    }
}

#[cfg(feature = "cuda")]
enum QuantizeSrc<'a> {
    F32(cudarc::driver::CudaView<'a, f32>),
    F16(cudarc::driver::CudaView<'a, half::f16>),
}
